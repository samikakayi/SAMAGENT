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

Only the newest turn is heard (real use 2026-09-25: two overlapping turns
both spoke -- the older turn's acknowledgement was still in the speaker buffer
and its tool's confirmation question was read out after the newer turn had
started). Every reply gets a generation number; starting a newer reply flushes
audio an older reply left in the speaker (``Speaker.flush`` bumps the epoch, so
its in-flight writes are dropped too), an older reply never writes again, and
a confirmation question asked from inside an older turn (``REPLY_GEN``, a
context variable every task of a reply inherits) is not spoken -- its card is
still on the island.

One utterance split by a pause is one turn (real use 2026-09-25: «وەڵاهی جارێ
ترێیت ملیۆم لۆ بکەوە» + «بڕۆ سەر چار چی دەکەی؟» became two turns: the first
was cancelled and a merged one re-ran). When a new utterance starts within
``voice.merge_window_s`` of the last one's end (frames.py decides it is the
same talker), ``hold_for_continuation`` keeps that one's reply from starting:
its transcript waits (``_held``) and is joined with the continuation's into
ONE request -- no cancelled model call. If its reply had already started but
made no sound, it is cancelled and carried over as before. A continuation
that turns out to be a cough or another voice releases the held request.

Stopping (real use 2026-09-25: «کوڕە دەنگی بنەکەرە!» -- "stop talking" -- went
to the model and ended as a system mute 55 s later): ``stop_speaking()`` is
immediate and safe to call from inside a turn (the brain's "stop talking"): it
never cuts the calling turn and never carries anything over. A transcript
that is only a stop phrase («بەسە», «بوەستە», «بێدەنگ بە», ``is_stop_phrase``)
stops playback the moment it is known -- also when it was said over SAM's
voice too briefly to cut it (``short_barge``: then only a stop phrase or a
yes/no to a pending question counts) -- and then still goes to the brain,
which decides whether tools stop too.

Confirmation echo: while SAM reads a confirmation question aloud and for 1 s
after, what the mic hears is not offered to the ConfirmBroker (the review
showed SAM's own question «ئەم نامەیە بنێرم؟ «باشە ...»» classify as a yes
when the room echo reaches the mic).

Last check before a model hears the words (``hooks.admit_transcript``,
engine.py): «دەنگم بناسە» opens the voice enrollment; in always-listening mode
an utterance must start with «سام»; nothing else is dropped here (the
near-field gate, "only my voice" and the listening policy already ran before
STT, frames.py / listening.py).

While a confirmation waits, an utterance that is not a clear yes/no never
starts a new turn: that would silence the reply waiting for the answer and
the question would time out to NO unheard (adversarial review 2026-09-24:
«باشە» started a model turn and «نەمنارد» was never spoken). SAM asks
«بەڵێ یان نەخێر؟» once instead. Only the user's own voice (voiceprint match,
or the owner's turn right after a click) with a longer new request replaces
the question: it is answered NO at once.

KurdishTTS budget: under ``voice.tts_low_budget_share`` of the month's
characters left (and no Gemini TTS), only the first sentence of an answer is
spoken; the full text is in the panel (20,000 characters a month is about 90
analyses).
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from ..brain.confirm import ASK_AGAIN_CKB
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
HOLD_MAX_S = 40.0          # a held request is answered anyway after this (max utterance 30 s + margin)

# The generation of the reply whose task (or a task it started: tools,
# confirmation questions) is running; None outside replies.
REPLY_GEN: contextvars.ContextVar[int | None] = contextvars.ContextVar("sam_voice_reply_gen", default=None)


def _key(text: str) -> str:
    return normalize_ckb(str(text or ""), strip_punct=True)


# One-word backchannels: "I'm listening", not a request (review 2026-09-24).
BACKCHANNELS = frozenset(_key(w) for w in ("ئەها", "ئەهە", "ئا", "ئاها", "باشە", "ئۆکێ", "هممم", "همم", "ئێ",
                                           "بەڵێ بەڵێ", "uh huh", "ok", "okay", "mhm", "yeah"))

# "Stop talking": the whole utterance is stop words, optionally with the name or a filler.
STOP_WORDS = frozenset(_key(w) for w in (
    "بەسە", "بەسێتی", "بەس", "بەسیە", "بوەستە", "بووەستە", "وەستە", "ڕاوەستە", "ڕابوەستە", "بێدەنگ", "بێدەنگبە",
    "کپ", "کپبە", "ستۆپ", "stop", "enough", "quiet", "silence", "shush", "hush"))
STOP_FILLERS = frozenset(_key(w) for w in (
    "سام", "سامی", "کوڕە", "کوڕ", "ئەی", "هەی", "ئیتر", "ئیدی", "تکایە", "دە", "دەی", "جا", "ئێستا", "یەکسەر", "بە",
    "sam", "ok", "okay", "please", "now", "be", "just", "it", "that's", "thats"))
STOP_PHRASES = tuple(_key(p) for p in ("قسە مەکە", "قسە نەکە", "shut up", "be quiet", "stop talking", "that's enough"))


def is_backchannel(text: str) -> bool:
    return _key(text) in BACKCHANNELS


def is_stop_phrase(text: str) -> bool:
    """«بەسە» / «بوەستە» / «بێدەنگ بە» / «سام بەسە» / "stop": the utterance is
    ONLY a request to stop talking («بوەستە لەسەر چارتەکە» is not)."""
    key = f" {_key(text)} "
    for phrase in STOP_PHRASES:
        key = key.replace(f" {phrase} ", " stop ")
    words = key.split()
    if not words or len(words) > 5:
        return False
    core = [w for w in words if w not in STOP_FILLERS]
    return bool(core) and all(w in STOP_WORDS for w in core)


@dataclass
class Utterance:
    pcm: bytes
    eos_at: float
    text: str | None = None        # already transcribed (Live's input transcription)
    published: bool = False        # the user Transcript was already published
    cut_reply: bool = False        # this utterance barged in on SAM's reply
    meta: dict[str, Any] = field(default_factory=dict)   # gate levels, first-after-click, verified
    submitted_at: float = 0.0      # monotonic
    stage: str = "queued"          # queued | stt | held | replying | merged | done
    hold: bool = False             # a continuation started: do not start the reply yet
    reply: "_Reply | None" = None


@dataclass
class _Held:
    item: Utterance
    text: str
    turn: Any
    published: bool


@dataclass
class _Reply:
    text: str
    turn: Any
    gen: int = 0
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
        self._asked_again: set[str] = set()
        self._tts_resting_noted: float = 0.0
        self._gen = 0                      # generation of the newest reply
        self._audio_gen: int | None = None # generation of the reply that wrote the last audio (None: other speech)
        self._stop_gen = 0                 # bumped by every stop / barge-in: queued fixed speech stops too
        self._last_item: Utterance | None = None
        self._held: _Held | None = None
        self.merged = 0                    # continuations joined into one request (status / tests)
        self.stale_dropped = 0             # older turns' speech not played (status / tests)

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
        self._held = None

    @property
    def speaking(self) -> bool:
        return self.speaker.playing

    @property
    def reply_active(self) -> bool:
        """SAM is thinking about or saying an answer right now."""
        return (self.speaker.playing or (self._current is not None and not self._current.done)
                or self._held is not None)

    @property
    def busy(self) -> bool:
        return (self._processing or not self._queue.empty() or self.speaker.playing or self._held is not None
                or (self._current is not None and not self._current.done))

    # -- input side --------------------------------------------------------------------------------
    def submit_utterance(self, pcm: bytes, eos_at: float, *, text: str | None = None,
                         published: bool = False, cut_reply: bool = False,
                         meta: dict[str, Any] | None = None) -> Utterance:
        self.start()
        item = Utterance(pcm=pcm, eos_at=eos_at, text=text, published=published, cut_reply=cut_reply,
                         meta=dict(meta or {}), submitted_at=time.monotonic())
        if not item.meta.get("short_barge"):
            self._last_item = item
        self._queue.put_nowait(item)
        return item

    def barge_in(self) -> bool:
        """The user started speaking. True when SAM was talking or thinking."""
        reply = self._current
        active = reply is not None and not reply.done
        if not active and not self.speaker.playing:
            return False
        self._stop_gen += 1          # fixed speech being read (an alert) stops as well
        self.speaker.flush()
        self._audio_gen = None
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
        self.stop_now()

    def stop_now(self, *, reason: str = "") -> None:
        """Silence at once: drop queued audio (the epoch bump also drops writes
        in flight), stop fixed speech being read, mute the running reply --
        unless the caller IS that reply (the brain answering «بێدەنگ بە» may
        still say its own short answer). Nothing is carried over."""
        self._stop_gen += 1
        self.speaker.flush()
        self._audio_gen = None
        self._carry = None
        caller = REPLY_GEN.get()
        reply = self._current
        if reply is not None and not reply.done and reply.gen != caller:
            self._interrupt(reply, carry=False)
        if reason:
            log.info("speech stopped (%s)", reason)

    def _in_tool_or_confirm(self) -> bool:
        try:
            running = any(r.get("source") == "cascade" for r in self.app.tools.running())
        except Exception:  # noqa: BLE001
            running = False
        return running or bool(getattr(self.app.confirm, "has_pending", False))

    def _interrupt(self, reply: _Reply, *, carry: bool = True) -> None:
        reply.muted = True
        if self._in_tool_or_confirm():
            return  # keep running silently: the tool finishes, a yes/no can still arrive
        if reply.task is not None and not reply.task.done():
            if carry and not reply.audio_started:
                self._carry = (reply.text, time.monotonic())
            reply.task.cancel()

    def _void(self, *, reopen: bool = False) -> None:
        """The utterance produced no request: tell the listening policy (a
        following "listening" state is not the end of an answer)."""
        hook = getattr(self.hooks, "utterance_void", None)
        if hook is not None:
            hook(reopen=reopen)

    def _take_carry(self) -> str | None:
        carry, self._carry = self._carry, None
        if carry and time.monotonic() - carry[1] <= CARRY_TTL_S:
            return carry[0]
        return None

    # -- one utterance split by a pause (frames.py) ---------------------------------------------------
    def hold_for_continuation(self, item: Utterance | None, window_s: float) -> str | None:
        """A new utterance started ``window_s`` or less after ``item`` ended.
        "hold": its reply has not started -- it waits for the continuation;
        "carry": its reply started but is silent -- the caller cancels it now
        with ``barge_in`` and its words are carried over; None: too late (SAM
        answered, a tool runs)."""
        if item is None or item is not self._last_item or item.meta.get("short_barge"):
            return None
        if time.monotonic() - item.submitted_at > window_s:
            return None
        if item.stage in ("queued", "stt"):
            item.hold = True
            try:
                asyncio.get_running_loop().call_later(HOLD_MAX_S, self.release_hold, item)
            except RuntimeError:
                pass
            return "hold"
        if item.stage == "held":
            return "hold"
        reply = item.reply
        if (item.stage == "replying" and reply is not None and reply is self._current and not reply.done
                and not reply.audio_started and not self.speaker.playing and not self._in_tool_or_confirm()):
            return "carry"
        return None

    def release_hold(self, item: Utterance | None) -> None:
        """The continuation was not one (a cough, another voice, no words):
        answer the held request now."""
        if item is None:
            return
        held = self._held
        if held is not None and held.item is item:
            self._held = None
            if not held.published:
                self.app.bus.publish(Transcript(role="user", text=held.text, source="cascade"))
            carry = self._take_carry()
            self._start_reply(f"{carry} {held.text}" if carry else held.text, held.turn, item)
        elif item.stage in ("queued", "stt"):
            item.hold = False

    def release_holds(self) -> None:
        """Listening closed mid-continuation: nothing more is coming."""
        if self._held is not None:
            self.release_hold(self._held.item)
        if self._last_item is not None:
            self.release_hold(self._last_item)

    def _take_held(self, item: Any) -> _Held | None:
        held = self._held
        if item is None or held is None or held.item is not item:
            return None
        self._held = None
        item.stage = "merged"
        return held

    # -- utterance worker ---------------------------------------------------------------------------
    async def _work(self) -> None:
        REPLY_GEN.set(None)    # the worker is nobody's reply (it may have been created inside one)
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
        item.stage = "stt"
        try:
            await self._handle_item(item)
        finally:
            if item.stage in ("queued", "stt"):
                item.stage = "done"
            # A continuation that did not become part of a request: its head is answered alone.
            self.release_hold(item.meta.get("continues"))
            if item.meta.get("short_barge"):
                self.speaker.gain = 1.0      # frames.py ducked SAM while the short word was checked

    async def _handle_item(self, item: Utterance) -> None:
        turn = self.app.timing.turn("cascade")
        turn.t0 = item.eos_at
        turn.mark("end_of_speech")
        short = bool(item.meta.get("short_barge"))
        text = item.text
        if text is None:
            if not self.stt.configured():
                turn.finish(outcome="stt_unconfigured")
                if not self._stt_warned:
                    self._stt_warned = True
                    self.app.bus.publish(Error(where="voice", message_ckb=strings.STT_UNCONFIGURED))
                self._void()
                self.hooks.set_state("listening")
                return
            if not short:                    # a word over SAM's voice: SAM is still speaking
                self.hooks.set_state("thinking")
            try:
                with turn.stage("stt"):
                    result = await self.stt.transcribe(item.pcm)
            except SttError as exc:
                turn.finish(outcome=f"stt_{exc.kind}")
                self.app.bus.publish(Error(where="voice.stt", message_ckb=strings.STT_FAILED,
                                           detail=self.app.redact(str(exc))[:200]))
                if short:
                    return
                self._void(reopen=True)   # «دووبارەی بکەرەوە»: the user may say it again without a click
                await self.speak(strings.STT_FAILED_SPOKEN, source="system")
                return
            text = result.text
        if not text.strip():
            turn.finish(outcome="empty")
            if short:
                return
            self._void(reopen=True)
            self.hooks.set_state("listening")
            self.resume_carry()
            return
        stop = is_stop_phrase(text)
        if stop:
            self.stop_now(reason="stop phrase")          # at once: before any model hears it
            self.hooks.set_state("listening")
        elif short and not self._answers_pending(text):
            turn.finish(outcome="short_barge_ignored")   # «ئەها» over SAM's voice: not a request
            return
        if self._confirm_echo(item):
            turn.finish(outcome="confirm_echo")
            self._void()
            self.hooks.set_state("listening")
            return
        if item.cut_reply and is_backchannel(text):
            turn.finish(outcome="backchannel")
            self._void()
            self.hooks.set_state("listening")
            return
        admit = getattr(self.hooks, "admit_transcript", None)
        if admit is not None and not item.published:
            admitted = admit(text, item.meta)
            if admitted is None:
                turn.finish(outcome="not_admitted")
                self._void()
                self.hooks.set_state("listening")
                return
            text = admitted
        held = self._take_held(item.meta.get("continues"))
        if held is not None and stop:
            # «... بەسە»: the stop cancels the request it would have joined (kept in the history only).
            if not held.published:
                self.app.bus.publish(Transcript(role="user", text=held.text, source="cascade"))
            held.turn.finish(outcome="stopped")
        elif held is not None:
            # One utterance split by a pause: ONE request, ONE model call.
            text = f"{held.text} {text}".strip()
            held.turn.finish(outcome="merged")
            self.merged += 1
        consumed = False
        if not item.published:
            self.app.bus.publish(Caption(text=text, role="user", final=True))
            consumed = self.app.confirm.offer_transcript(text)
        if consumed:  # "بەڵێ"/"نەخێر" answered a pending confirmation; the waiting tool goes on
            self._publish_user(text, item)
            turn.finish(outcome="confirm_answer")
            self.hooks.set_state("working")
            return
        if not item.published and not stop and getattr(self.app.confirm, "has_pending", False):
            if not self._replaces_question(text, item):
                self._publish_user(text, item)
                turn.finish(outcome="ask_again")
                await self._ask_again()
                return
        if item.hold and not stop:
            # A continuation began right after this utterance (frames.py): wait for it.
            self._held = _Held(item=item, text=text, turn=turn, published=item.published)
            item.stage = "held"
            self.hooks.set_state("listening")
            return
        self._publish_user(text, item)
        carry = None if stop else self._take_carry()
        self._start_reply(f"{carry} {text}" if carry else text, turn, item)

    def _publish_user(self, text: str, item: Utterance) -> None:
        if not item.published:
            self.app.bus.publish(Transcript(role="user", text=text, source="cascade"))

    def _answers_pending(self, text: str) -> bool:
        classify = getattr(self.app.confirm, "classify_pending", None)
        try:
            return bool(classify is not None and classify(text) is not None)
        except Exception:  # noqa: BLE001
            return False

    def _replaces_question(self, text: str, item: Utterance) -> bool:
        """A new request instead of the yes/no SAM waits for: only the user's
        own voice (voiceprint match, or the owner's turn right after a click)
        with more than a word or two. The waiting question is then answered NO
        at once (never left to time out)."""
        needs_answer = getattr(self.app.confirm, "needs_clear_answer", lambda _t: True)
        if not (item.meta.get("verified") or item.meta.get("owner")) or needs_answer(text):
            return False
        self.app.confirm.resolve(None, False, via="superseded")
        return True

    async def _ask_again(self) -> None:
        """«بەڵێ یان نەخێر؟» -- once per pending question; later unclear speech
        is ignored until the answer, a click or the 20 s expiry."""
        pending = self.app.confirm.pending() if hasattr(self.app.confirm, "pending") else []
        confirm_id = str(pending[-1].get("confirm_id", "")) if pending else ""
        if confirm_id in self._asked_again:
            self._void()
            self.hooks.set_state("listening")
            return
        self._asked_again.add(confirm_id)
        self._void()
        await self.speak(ASK_AGAIN_CKB, source="confirm")

    def _confirm_echo(self, item: Utterance) -> bool:
        """Heard while SAM's confirmation question played (or within 1 s):
        probably SAM's own voice, never an answer."""
        if not getattr(self.app.confirm, "has_pending", False):
            return False
        began = item.eos_at - max(0.0, pcm_seconds(item.pcm, MIC_RATE) - 0.6)
        return began < self.confirm_quiet_until

    def _start_reply(self, text: str, turn: Any, item: Utterance | None = None) -> None:
        previous = self._current
        if previous is not None and not previous.done:
            # The new request is decided already: the older one is not carried into the NEXT utterance.
            self._interrupt(previous, carry=False)
        self._gen += 1
        reply = _Reply(text=text, turn=turn, gen=self._gen)
        if self._audio_gen is not None and self._audio_gen != reply.gen:
            # An older turn's audio may still be queued (its acknowledgement, a
            # sentence): only the newest turn is heard.
            self.speaker.flush()
            self._audio_gen = None
        self._current = reply
        if item is not None:
            item.reply = reply
            item.stage = "replying"
        reply.task = asyncio.ensure_future(self._run_reply(reply))

    def _stale(self, reply: _Reply | None) -> bool:
        return reply is not None and (reply.muted or reply.gen != self._gen)

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
            elif pending is not None and not pending.cancelled():
                pending.exception()     # finished while we were cancelled: retrieved, never "never retrieved"
            closer = getattr(iterator, "aclose", None)
            if closer is not None:
                try:
                    await closer()
                except Exception:  # noqa: BLE001
                    pass

    async def _run_reply(self, reply: _Reply) -> None:
        REPLY_GEN.set(reply.gen)     # tools / confirmation questions started from here inherit it
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
            superseded = reply.gen != self._gen
            if text and (outcome != "cancelled" or reply.audio_started) and not (superseded and reply.muted):
                if outcome == "cancelled":
                    text += " …"
                self.app.bus.publish(Caption(text=text, role="assistant", final=True))
                if not from_brain:  # the brain stores its own reply (single writer of turns)
                    self.app.bus.publish(Transcript(role="assistant", text=text, source="cascade"))
            reply.turn.finish(outcome=outcome, tts=getattr(self.tts, "last_provider", None),
                              stt=getattr(self.stt, "last_provider", None), muted=reply.muted,
                              superseded=superseded)
            self._schedule_listening()

    async def _voice_pieces(self, reply: _Reply, pieces: "asyncio.Queue[str | None]") -> None:
        while True:
            piece = await pieces.get()
            if piece is None:
                return
            if self._stale(reply):
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

    async def _say_piece(self, text: str, reply: _Reply | None, wanted: Callable[[], bool] | None = None) -> bool:
        """Synthesize one piece into the speaker, in order with every other
        piece (one voice line). False if nothing was played. A reply's piece
        stops the moment a newer turn starts or the reply is muted; fixed
        speech stops when ``wanted()`` turns False (a stop / barge-in)."""
        async with self._line:
            if self._stale(reply) or (wanted is not None and not wanted()):
                if reply is not None and reply.gen != self._gen:
                    self.stale_dropped += 1
                return False
            if not self.tts.configured():
                self._tts_unavailable()
                return False
            epoch = self.speaker.epoch
            began = time.perf_counter()
            played = False
            try:
                async for pcm in self.tts.stream(text):
                    if self._stale(reply) or (wanted is not None and not wanted()):
                        break
                    if not played:
                        played = True
                        self._first_audio(reply, (time.perf_counter() - began) * 1000.0, text)
                    if not self.speaker.write(pcm, epoch=epoch):
                        break  # flushed by a barge-in / a newer turn / stop
                    self._audio_gen = reply.gen if reply is not None else None
            except TtsError as exc:
                if reply is None or not reply.tts_error_reported:
                    if reply is not None:
                        reply.tts_error_reported = True
                    self.app.bus.publish(Error(where="voice.tts", message_ckb=strings.TTS_FAILED,
                                               detail=self.app.redact(str(exc))[:200]))
            return played

    def _tts_unavailable(self) -> None:
        """No voice right now. A key exists but every provider rests after a
        quota / timeout error: say so once a minute (the text is on screen);
        otherwise no key is set at all."""
        resting = getattr(self.tts, "resting_only", None)
        if resting is not None and resting():
            now = time.monotonic()
            if now - self._tts_resting_noted >= 60.0:
                self._tts_resting_noted = now
                self.app.bus.publish(Error(where="voice.tts", message_ckb=strings.TTS_RESTING))
            return
        self.app.bus.publish(Error(where="voice.tts", message_ckb=strings.TTS_UNCONFIGURED))

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
        """Say a fixed text (alert, confirmation question, worker summary). A
        confirmation question asked from inside an older turn (a newer one has
        started since) is not spoken: only the newest turn is heard."""
        if not text or not text.strip():
            return
        origin = REPLY_GEN.get()
        if source == "confirm" and origin is not None and origin != self._gen:
            self.stale_dropped += 1
            log.info("not speaking an older turn's confirmation question (turn %s, newest %s)", origin, self._gen)
            return
        if interrupt:
            await self.stop_speaking()
        stop_gen = self._stop_gen

        def wanted() -> bool:
            if self._stop_gen != stop_gen:
                return False
            return source != "confirm" or origin is None or origin == self._gen

        self.app.bus.publish(Caption(text=text, role="assistant", final=True))
        for piece in split_for_tts(text, self._max_chars()):
            if not wanted():
                break
            await self._say_piece(piece, None, wanted)
        if source == "confirm":
            self.confirm_quiet_until = time.perf_counter() + self.speaker.buffered_ms() / 1000.0 + 1.0
        self._schedule_listening()

    def _schedule_listening(self) -> None:
        if self._after is None or self._after.done():
            self._after = asyncio.ensure_future(self._back_to_listening())

    async def _back_to_listening(self) -> None:
        await self.speaker.wait_idle(timeout=120.0)
        reply = self._current
        if (reply is None or reply.done) and not self._processing and self._queue.empty() and self._held is None:
            self.hooks.set_state("listening")

    def status(self) -> dict[str, Any]:
        return {"generation": self._gen, "held": self._held is not None, "merged": self.merged,
                "stale_dropped": self.stale_dropped}


__all__ = ["CascadeVoice", "Utterance", "REPLY_GEN", "is_stop_phrase", "is_backchannel", "STOP_WORDS"]
