"""CascadeVoice: local VAD -> STT -> streaming brain reply -> sentence TTS -> speaker.

The fallback path, and today's working default (no Gemini key yet). Every
component explicitly supports Sorani (reports/realtime-voice.json):
KurdishTTS STT (dialect sorani) -> ``app.conversation.respond_stream`` (the
brain: tools, memory, persona) -> Gemini 3.8 Flash-Lite TTS (lists "Central
Kurdish") or KurdishTTS -> the one continuous Speaker.

Latency plan (v1: serial, 13-20 s to first sound, reports/audit-latency.json):
the FIRST sentence goes to TTS the moment it is complete while the model keeps
writing; later sentences are packed to save requests, but released early if
the speaker is about to run dry. Research estimate: 2.3-4.5 s TTFA; target
<= 4.5 s, logged per turn in ``timings`` (end_of_speech, stt, llm_first_token,
tts_first_audio, first_audio).

Barge-in: the engine calls ``barge_in`` after 400 ms of voiced audio over
SAM's voice (shorter sounds only duck it; engine.py); playback is flushed and
the reply task is cancelled unless it is inside a tool or waiting for a
confirmation -- then it keeps running muted, so a spoken "بەڵێ" can still
answer the question and the action finishes honestly. A reply cancelled
before it made any sound is merged with what the user says next ("carry"). A
one-word backchannel («ئەها», «باشە») that cut a reply starts no new turn.

Confirmation echo: while SAM reads a confirmation question aloud and for 1 s
after, what the mic hears is not offered to the ConfirmBroker (the review
showed SAM's own question «ئەم نامەیە بنێرم؟ «باشە ...»» classify as a yes
when the room echo reaches the mic).

KurdishTTS budget: under ``voice.tts_low_budget_share`` of the month's
characters left (and no Gemini TTS), only the first sentence of an answer is
spoken; the full text is in the panel (20,000 characters a month is about 90
analyses).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from ..events import Caption, Error, Transcript
from ..textnorm import normalize_ckb
from . import strings
from .live_config import FALLBACK_INSTRUCTION, join_parts
from .audio import MIC_RATE, pcm_seconds
from .speech_text import SentenceSplitter, split_for_tts
from .stt import SttError
from .tts import TtsError

log = logging.getLogger("sam.voice.cascade")

CARRY_TTL_S = 15.0
STARVING_MS = 700.0


def _key(text: str) -> str:
    return normalize_ckb(str(text or ""), strip_punct=True)


# One-word backchannels: "I'm listening", not a request (review 2026-09-24).
BACKCHANNELS = frozenset(_key(w) for w in ("ئەها", "ئەهە", "ئا", "ئاها", "باشە", "ئۆکێ", "هممم", "همم", "ئێ",
                                           "بەڵێ بەڵێ", "uh huh", "ok", "okay", "mhm", "yeah"))


def is_backchannel(text: str) -> bool:
    return _key(text) in BACKCHANNELS


@dataclass
class Utterance:
    pcm: bytes
    eos_at: float
    text: str | None = None        # already transcribed (Live's input transcription)
    published: bool = False        # the user Transcript was already published
    cut_reply: bool = False        # this utterance barged in on SAM's reply


@dataclass
class _Reply:
    text: str
    turn: Any
    task: "asyncio.Task[Any] | None" = None
    muted: bool = False
    audio_started: bool = False
    tts_busy: bool = False
    pieces_sent: int = 0
    parts: list[str] = field(default_factory=list)
    tts_error_reported: bool = False
    ack_texts: list[str] = field(default_factory=list)   # acknowledgement chunks (not the answer)
    answer_audio: bool = False                           # first_answer_audio recorded
    answer_pieces: int = 0                               # answer pieces sent to TTS

    @property
    def done(self) -> bool:
        return self.task is None or self.task.done()


class CascadeVoice:
    def __init__(self, app: Any, speaker: Any, stt: Any, tts: Any, hooks: Any, *,
                 llm_stream: Callable[[str, Any], AsyncIterator[str]] | None = None) -> None:
        self.app = app
        self.speaker = speaker
        self.stt = stt
        self.tts = tts
        self.hooks = hooks
        self._llm_stream = llm_stream
        self._queue: asyncio.Queue[Utterance] = asyncio.Queue()
        self._worker: asyncio.Task[Any] | None = None
        self._line = asyncio.Lock()
        self._current: _Reply | None = None
        self._carry: tuple[str, float] | None = None
        self._processing = False
        self._after: asyncio.Task[Any] | None = None
        self.last_ttfa_ms: float | None = None
        self.last_answer_ms: float | None = None
        self.confirm_quiet_until = 0.0     # perf_counter: SAM's confirmation question may still echo
        self._stt_warned = False

    # -- lifecycle ------------------------------------------------------------------------------
    def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.ensure_future(self._work())

    async def close(self) -> None:
        tasks = [t for t in (self._worker, self._after, self._current.task if self._current else None) if t]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._worker = None
        self._current = None

    @property
    def speaking(self) -> bool:
        return self.speaker.playing

    @property
    def busy(self) -> bool:
        return (self._processing or not self._queue.empty() or self.speaker.playing
                or (self._current is not None and not self._current.done))

    # -- input side --------------------------------------------------------------------------------
    def submit_utterance(self, pcm: bytes, eos_at: float, *, text: str | None = None,
                         published: bool = False, cut_reply: bool = False) -> None:
        self.start()
        self._queue.put_nowait(Utterance(pcm=pcm, eos_at=eos_at, text=text, published=published,
                                         cut_reply=cut_reply))

    def barge_in(self) -> bool:
        """The user started speaking. True when SAM was talking or thinking."""
        reply = self._current
        active = reply is not None and not reply.done
        if not active and not self.speaker.playing:
            return False
        self.speaker.flush()
        if active:
            self._interrupt(reply)  # type: ignore[arg-type]
        self.hooks.set_state("listening")
        return True

    def resume_carry(self) -> None:
        """A barge-in cancelled a silent reply but nothing new was said
        (cough, empty transcript): answer the carried request after all."""
        carry = self._take_carry()
        if carry:
            self._start_reply(carry, self.app.timing.turn("cascade"))

    async def stop_speaking(self) -> None:
        self.speaker.flush()
        if self._current is not None and not self._current.done:
            self._interrupt(self._current)

    def _in_tool_or_confirm(self) -> bool:
        try:
            running = any(r.get("source") == "cascade" for r in self.app.tools.running())
        except Exception:  # noqa: BLE001
            running = False
        return running or bool(getattr(self.app.confirm, "has_pending", False))

    def _interrupt(self, reply: _Reply) -> None:
        reply.muted = True
        if self._in_tool_or_confirm():
            return  # keep running silently: the tool finishes, a yes/no can still arrive
        if reply.task is not None and not reply.task.done():
            if not reply.audio_started:
                self._carry = (reply.text, time.monotonic())
            reply.task.cancel()

    def _take_carry(self) -> str | None:
        carry, self._carry = self._carry, None
        if carry and time.monotonic() - carry[1] <= CARRY_TTL_S:
            return carry[0]
        return None

    # -- utterance worker ---------------------------------------------------------------------------
    async def _work(self) -> None:
        while True:
            item = await self._queue.get()
            self._processing = True
            try:
                await self._handle(item)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - one bad turn must not stop listening
                log.exception("cascade turn failed")
                self.hooks.set_state("listening")
            finally:
                self._processing = False

    async def _handle(self, item: Utterance) -> None:
        turn = self.app.timing.turn("cascade")
        turn.t0 = item.eos_at
        turn.mark("end_of_speech")
        text = item.text
        if text is None:
            if not self.stt.configured():
                turn.finish(outcome="stt_unconfigured")
                if not self._stt_warned:
                    self._stt_warned = True
                    self.app.bus.publish(Error(where="voice", message_ckb=strings.STT_UNCONFIGURED))
                self.hooks.set_state("listening")
                return
            self.hooks.set_state("thinking")
            try:
                with turn.stage("stt"):
                    result = await self.stt.transcribe(item.pcm)
            except SttError as exc:
                turn.finish(outcome=f"stt_{exc.kind}")
                self.app.bus.publish(Error(where="voice.stt", message_ckb=strings.STT_FAILED,
                                           detail=self.app.redact(str(exc))[:200]))
                await self.speak(strings.STT_FAILED_SPOKEN, source="system")
                return
            text = result.text
        if not text.strip():
            turn.finish(outcome="empty")
            self.hooks.set_state("listening")
            self.resume_carry()
            return
        if self._confirm_echo(item):
            turn.finish(outcome="confirm_echo")
            self.hooks.set_state("listening")
            return
        if item.cut_reply and is_backchannel(text):
            turn.finish(outcome="backchannel")
            self.hooks.set_state("listening")
            return
        consumed = False
        if not item.published:
            self.app.bus.publish(Caption(text=text, role="user", final=True))
            consumed = self.app.confirm.offer_transcript(text)
            self.app.bus.publish(Transcript(role="user", text=text, source="cascade"))
        if consumed:  # "بەڵێ"/"نەخێر" answered a pending confirmation; the waiting tool goes on
            turn.finish(outcome="confirm_answer")
            self.hooks.set_state("working")
            return
        carry = self._take_carry()
        self._start_reply(f"{carry} {text}" if carry else text, turn)

    def _confirm_echo(self, item: Utterance) -> bool:
        """Heard while SAM's confirmation question played (or within 1 s):
        probably SAM's own voice, never an answer."""
        if not getattr(self.app.confirm, "has_pending", False):
            return False
        began = item.eos_at - max(0.0, pcm_seconds(item.pcm, MIC_RATE) - 0.6)
        return began < self.confirm_quiet_until

    def _start_reply(self, text: str, turn: Any) -> None:
        previous = self._current
        if previous is not None and not previous.done:
            self._interrupt(previous)
        reply = _Reply(text=text, turn=turn)
        self._current = reply
        reply.task = asyncio.ensure_future(self._run_reply(reply))

    # -- reply -----------------------------------------------------------------------------------------
    def _llm(self, text: str, turn: Any) -> tuple[AsyncIterator[str], bool]:
        """(stream, from_brain). The brain's ``respond_stream`` yields finished
        sentence chunks and stores both turns itself (sam/brain/conversation.py);
        the other streams yield raw model tokens."""
        if self._llm_stream is not None:
            return self._llm_stream(text, turn), False
        conversation = getattr(self.app, "conversation", None)
        if conversation is not None and hasattr(conversation, "respond_stream"):
            return conversation.respond_stream(text, source="cascade", turn=turn), True
        return self._fallback_stream(text, turn), False

    async def _fallback_stream(self, text: str, turn: Any) -> AsyncIterator[str]:
        """No brain package loaded (parallel build): plain streamed chat, no tools."""
        messages = [{"role": "system", "content": FALLBACK_INSTRUCTION}, {"role": "user", "content": text}]
        async for chunk in self.app.llm.stream(messages, ladder="chat", turn=turn):
            if chunk.kind == "text" and chunk.text:
                yield chunk.text

    def _release_if_starving(self, reply: _Reply, splitter: SentenceSplitter,
                             pieces: "asyncio.Queue[str | None]") -> None:
        """Packed sentences wait for more text -- unless the speaker is about to
        go quiet (checked on every delta AND every 100 ms, so a tool round in
        the middle of a reply never leaves a finished sentence unspoken)."""
        if (reply.pieces_sent and pieces.empty() and not reply.tts_busy
                and self.speaker.buffered_ms() < STARVING_MS):
            for piece in splitter.take_ready():
                reply.pieces_sent += 1
                pieces.put_nowait(piece)

    async def _deltas(self, stream: AsyncIterator[str], on_tick: Callable[[], None]) -> AsyncIterator[str]:
        """Iterate ``stream`` but wake up every 100 ms to run ``on_tick``."""
        iterator = stream.__aiter__()
        pending: asyncio.Future[str] | None = None
        try:
            while True:
                if pending is None:
                    pending = asyncio.ensure_future(iterator.__anext__())
                done, _ = await asyncio.wait({pending}, timeout=0.1)
                if not done:
                    on_tick()
                    continue
                future, pending = pending, None
                try:
                    yield future.result()
                except StopAsyncIteration:
                    return
        finally:
            if pending is not None and not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            closer = getattr(iterator, "aclose", None)
            if closer is not None:
                try:
                    await closer()
                except Exception:  # noqa: BLE001
                    pass

    async def _run_reply(self, reply: _Reply) -> None:
        splitter = SentenceSplitter(max_chars=self._max_chars())
        pieces: asyncio.Queue[str | None] = asyncio.Queue()
        voice = asyncio.ensure_future(self._voice_pieces(reply, pieces))
        self.hooks.set_state("thinking")
        outcome = "ok"
        stream, from_brain = self._llm(reply.text, reply.turn)
        try:
            async for delta in self._deltas(stream, lambda: self._release_if_starving(reply, splitter, pieces)):
                if not delta:
                    continue
                if from_brain:
                    if type(delta).__name__ == "AckText":  # sam.brain.responder.AckText
                        reply.ack_texts.append(_key(delta))
                    # A finished sentence: the trailing space makes its end a
                    # boundary now (a final "." could otherwise still be a decimal
                    # point) and keeps sentences apart.
                    delta = delta.strip() + " "
                reply.parts.append(delta)
                if not reply.muted:
                    self.app.bus.publish(Caption(text=join_parts(reply.parts)[-160:], role="assistant", final=False))
                for piece in splitter.feed(delta):
                    reply.pieces_sent += 1
                    pieces.put_nowait(piece)
                self._release_if_starving(reply, splitter, pieces)
            for piece in splitter.flush():
                reply.pieces_sent += 1
                pieces.put_nowait(piece)
            pieces.put_nowait(None)
            await voice
        except asyncio.CancelledError:
            outcome = "cancelled"
            voice.cancel()
            raise
        except Exception as exc:  # noqa: BLE001 - the brain failed: say so honestly
            outcome = f"error:{type(exc).__name__}"
            log.warning("reply failed: %s", self.app.redact(f"{type(exc).__name__}: {exc}"))
            pieces.put_nowait(None)
            await asyncio.gather(voice, return_exceptions=True)
            self.app.bus.publish(Error(where="voice.reply", message_ckb=strings.REPLY_FAILED,
                                       detail=self.app.redact(f"{type(exc).__name__}: {exc}")[:200]))
        finally:
            text = join_parts(reply.parts)
            if text and (outcome != "cancelled" or reply.audio_started):
                if outcome == "cancelled":
                    text += " …"
                self.app.bus.publish(Caption(text=text, role="assistant", final=True))
                if not from_brain:  # the brain stores its own reply (single writer of turns)
                    self.app.bus.publish(Transcript(role="assistant", text=text, source="cascade"))
            reply.turn.finish(outcome=outcome, tts=getattr(self.tts, "last_provider", None),
                              stt=getattr(self.stt, "last_provider", None), muted=reply.muted)
            self._schedule_listening()

    async def _voice_pieces(self, reply: _Reply, pieces: "asyncio.Queue[str | None]") -> None:
        while True:
            piece = await pieces.get()
            if piece is None:
                return
            if reply.muted:
                continue
            is_ack = _key(piece) in reply.ack_texts
            if not is_ack and reply.answer_pieces and self._tts_low():
                continue  # KurdishTTS budget nearly used: the rest is in the panel only
            if not is_ack:
                reply.answer_pieces += 1
            reply.tts_busy = True
            try:
                await self._say_piece(piece, reply)
            finally:
                reply.tts_busy = False

    def _tts_low(self) -> bool:
        check = getattr(self.tts, "low_budget", None)
        try:
            return bool(check(float(self.app.config.get("voice.tts_low_budget_share", 0.2)))) if check else False
        except Exception:  # noqa: BLE001
            return False

    def _max_chars(self) -> int:
        getter = getattr(self.tts, "max_chars", None)
        return int(getter()) if callable(getter) else 480

    async def _say_piece(self, text: str, reply: _Reply | None) -> bool:
        """Synthesize one piece into the speaker, in order with every other
        piece (one voice line). False if nothing was played."""
        async with self._line:
            if reply is not None and reply.muted:
                return False
            if not self.tts.configured():
                self.app.bus.publish(Error(where="voice.tts", message_ckb=strings.TTS_UNCONFIGURED))
                return False
            epoch = self.speaker.epoch
            began = time.perf_counter()
            played = False
            try:
                async for pcm in self.tts.stream(text):
                    if reply is not None and reply.muted:
                        break
                    if not played:
                        played = True
                        self._first_audio(reply, (time.perf_counter() - began) * 1000.0, text)
                    if not self.speaker.write(pcm, epoch=epoch):
                        break  # flushed by a barge-in
            except TtsError as exc:
                if reply is None or not reply.tts_error_reported:
                    if reply is not None:
                        reply.tts_error_reported = True
                    self.app.bus.publish(Error(where="voice.tts", message_ckb=strings.TTS_FAILED,
                                               detail=self.app.redact(str(exc))[:200]))
            return played

    def _first_audio(self, reply: _Reply | None, tts_ms: float, text: str = "") -> None:
        if reply is not None and not reply.audio_started:
            reply.audio_started = True
            reply.turn.add("tts_first_audio", tts_ms)
            ttfa = reply.turn.mark("first_audio")
            self.last_ttfa_ms = round(ttfa, 1)
        if reply is not None and not reply.answer_audio and _key(text) not in reply.ack_texts:
            # In tool turns first_audio is the cached acknowledgement (2 ms from the
            # phrase cache); the answer itself comes after the tool and the wording
            # round. The review saw a smoke pass at 4987 ms TTFA whose real answer
            # was a failure at 9.5 s, so both are recorded.
            reply.answer_audio = True
            self.last_answer_ms = round(reply.turn.mark("first_answer_audio"), 1)
        self.hooks.set_state("speaking")

    async def speak(self, text: str, *, interrupt: bool = False, source: str = "system") -> None:
        """Say a fixed text (alert, confirmation question, worker summary)."""
        if not text or not text.strip():
            return
        if interrupt:
            await self.stop_speaking()
        self.app.bus.publish(Caption(text=text, role="assistant", final=True))
        for piece in split_for_tts(text, self._max_chars()):
            await self._say_piece(piece, None)
        if source == "confirm":
            self.confirm_quiet_until = time.perf_counter() + self.speaker.buffered_ms() / 1000.0 + 1.0
        self._schedule_listening()

    def _schedule_listening(self) -> None:
        if self._after is None or self._after.done():
            self._after = asyncio.ensure_future(self._back_to_listening())

    async def _back_to_listening(self) -> None:
        await self.speaker.wait_idle(timeout=120.0)
        reply = self._current
        if (reply is None or reply.done) and not self._processing and self._queue.empty():
            self.hooks.set_state("listening")


__all__ = ["CascadeVoice", "Utterance"]
