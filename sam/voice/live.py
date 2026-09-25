"""LiveVoice: one Gemini Live session (native audio in, audio out, tools).

Flow per connection (google-genai 2.25.0 ``client.aio.live.connect``):
mic frames -> ``send_realtime_input(audio=Blob(pcm, "audio/pcm;rate=16000"))``;
``receive()`` yields ``LiveServerMessage``s: model audio -> the continuous
Speaker; ``input/output_transcription`` -> Caption + Transcript events;
``interrupted`` -> flush the speaker at once (barge-in); ``tool_call`` ->
``app.tools.dispatch(..., source="live")`` concurrently -> ``send_tool_response``
(NON_BLOCKING tools answered with scheduling WHEN_IDLE); ``go_away`` /
dropped socket -> reconnect with the last ``session_resumption_update`` handle.

Watchdog (design 2.1): if no model audio, text or tool call arrives within
``voice.watchdog_s`` (5 s) after the local VAD saw the user stop speaking AND
Live transcribed words for that utterance, the engine answers it through the
cascade and Live is marked degraded for this conversation window. A forum
report shows 3.1 Flash Live's time to first audio regressing from ~1 s to
9-15 s on 2026-09-05 (reports/realtime-voice.json). A sound Live heard no
words in (a cough, a door, the TV, a word its VAD ignored) is not a stall:
the repair review (2026-09-24) found every such sound dropping Live for the
whole window and spending KurdishTTS STT (2 h a month) on the noise.

Hybrid VAD (live-guide "Hybrid VAD"): when the local endpointer sees the end
of speech, ``audio_stream_end`` is sent so the server finalizes the turn at
once instead of waiting for its own silence timer.

Confirmations: a tool whose risk is ``confirm`` waits in the ConfirmBroker. If
the server cancels that call while the user is answering (a spoken "بەڵێ" is a
new user turn), the call is DETACHED instead of cancelled: it still resolves on
the user's own words, and its result is told to the model as a [SAM] text turn.
A call cancelled before it started (and not waiting for a confirmation) is
cancelled, as the server asked.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import time
from typing import Any, Callable

from ..events import Caption, ToolStarted, Transcript
from .live_calls import LiveCalls, _Call
from ..brain import taint
from .live_config import (SAY_EXACTLY, audio_rate, build_live_config, classify_error, join_parts, parse_duration_s,
                          supports_async_tools, turn_is_idle)
from .selftest import selftest_verdict

log = logging.getLogger("sam.voice.live")




class LiveVoice(LiveCalls):
    def __init__(self, app: Any, speaker: Any, hooks: Any, *, client_factory: Callable[[str], Any] | None = None,
                 reconnect_delays: tuple[float, ...] = (0.5, 1.0, 2.0), ready_timeout_s: float = 10.0,
                 say_wait_s: float = 8.0) -> None:
        self.app = app
        self.speaker = speaker
        self.hooks = hooks
        self._client_factory = client_factory
        self._client: Any = None
        self._client_fp: str | None = None
        self.reconnect_delays = reconnect_delays
        self.ready_timeout_s = ready_timeout_s
        self.say_wait_s = say_wait_s
        self.model = ""
        self.degraded = False
        self.degraded_reason = ""
        self.connects = 0
        self.last_ttfa_ms: float | None = None
        self._handle: str | None = None
        self._session: Any = None
        self._generation = 0
        self._ready = asyncio.Event()
        self._reconnect = asyncio.Event()
        self._closing = False
        self._task: asyncio.Task[Any] | None = None
        self._send_lock = asyncio.Lock()
        self._pending_audio: collections.deque[bytes] = collections.deque(maxlen=100)  # ~3 s of 30 ms frames
        self._types_mod: Any = None
        # turn state
        self._user_parts: list[str] = []
        self._assistant_parts: list[str] = []
        # What Live heard of the current request (published fragments), and
        # whether it was only a yes/no for a pending confirmation: the
        # watchdog hands exactly this to the cascade (no second STT, no
        # duplicate user turn) and never "answers" a consumed yes/no.
        self._utt_heard: list[str] = []
        self._utt_consumed = False
        self._model_active = False
        self._audio_started = False
        self._suppress = False
        self._say_turn = False
        self._say_queue: collections.deque[tuple[str, str, str, float]] = collections.deque()
        self._say_timer: asyncio.TimerHandle | None = None
        self._calls: dict[str, _Call] = {}
        self._watchdog: asyncio.TimerHandle | None = None
        self._eos_at: float | None = None
        self._responded = True
        self._ttfa_pending = False
        self._last_utterance = b""
        self._turn: Any = None
        self._go_away_timer: asyncio.TimerHandle | None = None
        self._go_away_pending = False
        self._connected_at: float | None = None
        self._tokens = [0, 0]
        self._unsubscribe: Callable[[], None] | None = None
        self._after_turn_task: asyncio.Task[Any] | None = None

    # -- public ----------------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self._ready.is_set() and self._session is not None

    @property
    def busy(self) -> bool:
        return self._model_active or bool(self._calls) or self.speaker.playing

    def status(self) -> dict[str, Any]:
        return {"model": self.model, "connected": self.ready, "degraded": self.degraded,
                "degraded_reason": self.degraded_reason, "resumable": bool(self._handle),
                "connects": self.connects, "last_ttfa_ms": self.last_ttfa_ms,
                "async_tools": supports_async_tools(self.model) if self.model else None}

    async def open(self) -> bool:
        """Connect (primary model, then the fallback). True when the session is ready."""
        if self._task is None or self._task.done():
            self._closing = False
            self.degraded = False
            self.degraded_reason = ""
            if self._unsubscribe is None:
                self._unsubscribe = self.app.bus.subscribe(ToolStarted, self._on_tool_started)
            self._task = asyncio.ensure_future(self._supervise())
        waiter = asyncio.ensure_future(self._ready.wait())
        done, _ = await asyncio.wait({waiter, self._task}, timeout=self.ready_timeout_s,
                                     return_when=asyncio.FIRST_COMPLETED)
        if waiter not in done:
            waiter.cancel()
        return self._ready.is_set()

    async def close(self) -> None:
        self._closing = True
        self._disarm_watchdog()
        for timer in (self._say_timer, self._go_away_timer):
            if timer is not None:
                timer.cancel()
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if self._after_turn_task is not None:
            self._after_turn_task.cancel()
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        self._finish_user()
        self._finish_assistant()
        self._ready.clear()
        self._session = None

    async def send_audio(self, pcm: bytes) -> None:
        if not self.ready:
            self._pending_audio.append(pcm)
            return
        blob = self._types().Blob(data=pcm, mime_type="audio/pcm;rate=16000")
        await self._send(lambda s: s.send_realtime_input(audio=blob))

    def note_speech_start(self) -> None:
        """Local VAD: the user started (or continued) talking. After SAM
        answered, this is a new request; after a pause with no answer yet, the
        user is still finishing the same one (its heard text is kept)."""
        self._disarm_watchdog()
        if self._responded:
            self._utt_heard = []
            self._utt_consumed = False

    def note_end_of_speech(self, pcm: bytes, eos_at: float) -> None:
        """Local VAD: the user stopped talking (``eos_at`` = perf_counter of the
        last voiced frame). Arms the watchdog and starts the turn timer."""
        self._last_utterance = pcm
        self._eos_at = eos_at
        self._responded = False
        self._ttfa_pending = True
        if self._turn is not None:
            self._turn.finish(outcome="superseded")
        self._turn = self.app.timing.turn("live")
        self._turn.t0 = eos_at
        self._turn.mark("end_of_speech")
        self._arm_watchdog()
        if self.ready and self.app.config.get("voice.live_hybrid_vad", True):
            asyncio.ensure_future(self._send(lambda s: s.send_realtime_input(audio_stream_end=True)))

    async def say(self, text: str, *, interrupt: bool = False, source: str = "system") -> bool:
        """Make the model say ``text`` (alerts, confirmation questions).

        ``send_client_content(turn_complete=True)`` unconditionally interrupts
        generation on 3.8 Live (live-guide), so it is only sent when the model
        is idle (or ``interrupt``); confirmation questions during a running turn
        go to TTS (contract 3.1), other messages wait up to ``say_wait_s``."""
        if not self.ready:
            return False
        prompt = SAY_EXACTLY.format(text=text)
        if interrupt:
            self.speaker.flush()
            self._finish_assistant(interrupted=True)
            return await self._send_text(prompt, say_turn=True)
        if self._model_active or any(c.blocking for c in self._calls.values()):
            if source == "confirm":
                await self.hooks.speak_fallback(text, source)
                return True
            self._say_queue.append((prompt, text, source, time.monotonic() + self.say_wait_s))
            self._arm_say_timer()
            return True
        return await self._send_text(prompt, say_turn=True)

    async def send_user_text(self, text: str) -> bool:
        """Typed input while a Live session is open: goes into the same session
        as speech would (``send_realtime_input(text=...)``, live-guide "Sending
        text"); the spoken reply comes back like any other turn."""
        if not self.ready or not text.strip():
            return False
        self._model_active = True
        self._responded = True  # no local end-of-speech: the watchdog stays off
        return await self._send(lambda s: s.send_realtime_input(text=text))

    def stop_output(self) -> None:
        """stop_all / stop_speaking: silence now and drop the rest of this turn."""
        self.speaker.flush()
        if self._model_active:
            self._suppress = True
        self._finish_assistant(interrupted=True)

    # -- connection supervision ------------------------------------------------------------------
    def _types(self) -> Any:
        if self._types_mod is None:
            from google.genai import types
            self._types_mod = types
        return self._types_mod

    def _genai(self) -> Any:
        key = self.app.secrets.get("gemini_api_key")
        if not key:
            raise PermissionError("no gemini key")
        import hashlib
        fingerprint = hashlib.sha256(key.encode()).hexdigest()[:12]
        if self._client is None or fingerprint != self._client_fp:
            if self._client_factory is not None:
                self._client = self._client_factory(key)
            else:
                from google import genai
                self._client = genai.Client(api_key=key)
            self._client_fp = fingerprint
        return self._client

    def _models(self) -> list[str]:
        primary = str(self.app.config.get("voice.live_model", "gemini-3.8-live"))
        fallback = str(self.app.config.get("voice.live_fallback_model", "") or "")
        return [primary] + ([fallback] if fallback and fallback != primary else [])

    async def _supervise(self) -> None:
        models = self._models()
        index, failures = 0, 0
        while not self._closing:
            model = models[index]
            try:
                client = self._genai()
                config = build_live_config(self.app, model, handle=self._handle)
                began = time.perf_counter()
                async with client.aio.live.connect(model=model, config=config) as session:
                    self._on_connected(session, model, (time.perf_counter() - began) * 1000.0)
                    failures = 0
                    await self._flush_pending_audio()
                    planned = await self._run_session(session)
                if planned:
                    continue  # GoAway: reconnect at once with the resumption handle
                raise ConnectionError("session ended")
            except asyncio.CancelledError:
                raise
            except PermissionError:
                self._fail("no Gemini key")
                return
            except Exception as exc:  # noqa: BLE001 - classified below
                kind = classify_error(exc)
                detail = self.app.redact(f"{type(exc).__name__}: {exc}")[:300]
                log.warning("live session error (%s, model %s): %s", kind, model, detail)
                if self._closing:
                    return
                if kind == "rejected" and self._handle:
                    self._handle = None  # an expired/invalid resumption handle: start fresh once
                    continue
                if kind in ("model", "rejected") and index + 1 < len(models):
                    self.app.db.log_activity("voice", "live_setup", ok=False, summary=f"{model}: {detail}"[:300])
                    index += 1
                    continue
                if kind in ("auth", "quota", "model", "rejected"):
                    self.app.db.log_activity("voice", "live_setup", ok=False, summary=f"{model}: {detail}"[:300])
                    self._fail(f"{kind}: {detail[:120]}")
                    return
                failures += 1
                if failures > len(self.reconnect_delays):
                    self._fail(f"connection lost: {detail[:120]}")
                    return
                await asyncio.sleep(self.reconnect_delays[failures - 1])
            finally:
                self._on_disconnected()

    def _on_connected(self, session: Any, model: str, connect_ms: float) -> None:
        self._session = session
        self._generation += 1
        self.model = model
        self.connects += 1
        self._connected_at = time.monotonic()
        self._tokens = [0, 0]
        self._reconnect.clear()
        self._go_away_pending = False
        if self._go_away_timer is not None:
            self._go_away_timer.cancel()
            self._go_away_timer = None
        self._ready.set()
        self.app.timing.record("live_connect", connect_ms, kind="live", model=model, resumed=bool(self._handle))
        self.app.publish_status("live", "ok", model)
        self._model_active = False  # a resumed session starts between turns
        self._drain_say_queue()

    def _on_disconnected(self) -> None:
        if self._session is None:
            return
        self._session = None
        self._ready.clear()
        seconds = time.monotonic() - (self._connected_at or time.monotonic())
        try:
            self.app.db.bump_usage("gemini", self.model, kind="live", units=round(seconds, 1),
                                   tokens_in=self._tokens[0], tokens_out=self._tokens[1])
        except Exception:  # noqa: BLE001
            log.debug("live usage count failed", exc_info=True)

    def _fail(self, reason: str) -> None:
        self.degraded = True
        self.degraded_reason = reason
        self.app.publish_status("live", "degraded", reason[:200])
        self.hooks.live_failed(reason)

    async def _run_session(self, session: Any) -> bool:
        """Receive until the socket drops (raises) or a planned reconnect (True)."""
        receiver = asyncio.ensure_future(self._receive_loop(session))
        reconnect = asyncio.ensure_future(self._reconnect.wait())
        try:
            done, _ = await asyncio.wait({receiver, reconnect}, return_when=asyncio.FIRST_COMPLETED)
            if reconnect in done:
                return True
            receiver.result()
            return False
        finally:
            for task in (receiver, reconnect):
                if not task.done():
                    task.cancel()
            await asyncio.gather(receiver, reconnect, return_exceptions=True)

    async def _receive_loop(self, session: Any) -> None:
        while not self._closing:
            async for message in session.receive():
                self._on_message(message)
            await asyncio.sleep(0)  # receive() returns after each completed turn; keep listening

    async def _flush_pending_audio(self) -> None:
        while self._pending_audio and self.ready:
            await self.send_audio(self._pending_audio.popleft())

    async def _send(self, fn: Callable[[Any], Any]) -> bool:
        session = self._session
        if session is None:
            return False
        async with self._send_lock:
            try:
                await fn(session)
                return True
            except Exception as exc:  # noqa: BLE001 - the receive loop notices a dead socket
                log.debug("live send failed: %s", self.app.redact(f"{type(exc).__name__}: {exc}"))
                return False

    async def _send_text(self, prompt: str, *, say_turn: bool) -> bool:
        types = self._types()
        content = types.Content(role="user", parts=[types.Part(text=prompt)])
        self._say_turn = say_turn
        self._model_active = True
        return await self._send(lambda s: s.send_client_content(turns=content, turn_complete=True))

    # -- server messages ------------------------------------------------------------------------------
    def _on_message(self, msg: Any) -> None:
        update = getattr(msg, "session_resumption_update", None)
        if update is not None and getattr(update, "resumable", False) and getattr(update, "new_handle", None):
            self._handle = update.new_handle
        usage = getattr(msg, "usage_metadata", None)
        if usage is not None:
            self._tokens[0] += int(getattr(usage, "prompt_token_count", 0) or 0)
            self._tokens[1] += int(getattr(usage, "response_token_count", 0) or 0)
        cancellation = getattr(msg, "tool_call_cancellation", None)
        if cancellation is not None:
            self._on_cancellation(list(getattr(cancellation, "ids", None) or []))
        tool_call = getattr(msg, "tool_call", None)
        if tool_call is not None:
            self._on_tool_call(tool_call)
        sc = getattr(msg, "server_content", None)
        if sc is not None:
            self._on_server_content(sc)
        go_away = getattr(msg, "go_away", None)
        if go_away is not None:
            self._on_go_away(go_away)

    def _on_server_content(self, sc: Any) -> None:
        if getattr(sc, "interrupted", False):
            self._on_interrupted()
        interim = getattr(sc, "interim_input_transcription", None)
        if interim is not None and getattr(interim, "text", None):
            # Speculative partial (live-transcribe docs: "use it to render live
            # subtitles"); only the finalized input_transcription is stored.
            self.app.bus.publish(Caption(text=join_parts([*self._user_parts, interim.text])[-160:], role="user",
                                         final=False))
            self.hooks.activity()
        heard = getattr(sc, "input_transcription", None)
        if heard is not None and getattr(heard, "text", None):
            self._user_parts.append(heard.text)
            self.app.bus.publish(Caption(text=join_parts(self._user_parts)[-160:], role="user", final=False))
            self.hooks.activity()
        if heard is not None and getattr(heard, "finished", False):
            self._finish_user()
        model_turn = getattr(sc, "model_turn", None)
        for part in (getattr(model_turn, "parts", None) or []):
            if getattr(part, "thought", False):
                continue
            blob = getattr(part, "inline_data", None)
            if blob is not None and getattr(blob, "data", None):
                self._on_audio(blob.data, audio_rate(getattr(blob, "mime_type", None)))
        said = getattr(sc, "output_transcription", None)
        if said is not None and getattr(said, "text", None):
            self._mark_responded()
            self._finish_user()
            self._model_active = True
            if not self._suppress:
                self._assistant_parts.append(said.text)
                self.app.bus.publish(Caption(text=join_parts(self._assistant_parts)[-160:], role="assistant",
                                             final=False))
        if getattr(sc, "waiting_for_input", False):
            self._mark_responded()
        if turn_is_idle(sc):
            self._on_idle()

    def _on_audio(self, data: bytes, rate: int) -> None:
        self._model_active = True
        self._finish_user()
        if self._suppress:
            return
        if not self._audio_started:
            self._audio_started = True
            if self._ttfa_pending and self._eos_at is not None:
                self._ttfa_pending = False
                ms = (time.perf_counter() - self._eos_at) * 1000.0
                self.last_ttfa_ms = round(ms, 1)
                if self._turn is not None:
                    self._turn.mark("first_audio")
            self.hooks.set_state("speaking")
        self._mark_responded()
        self.speaker.write(data, rate=rate)

    def _on_interrupted(self) -> None:
        """Barge-in: the server stopped generating; drop queued audio NOW."""
        self.speaker.flush()
        self._finish_assistant(interrupted=True)
        self._audio_started = False
        self.hooks.set_state("listening")

    def _on_idle(self) -> None:
        self._finish_user()
        self._finish_assistant()
        if self._turn is not None:
            self._turn.finish(model=self.model)
            self._turn = None
        self._model_active = False
        self._audio_started = False
        self._suppress = False
        self._say_turn = False
        if self._go_away_pending and not self._calls:
            self._reconnect.set()  # GoAway arrived mid-turn: move to a fresh connection now
        else:
            self._drain_say_queue()  # (after a reconnect _on_connected drains it)
        if self._after_turn_task is None or self._after_turn_task.done():
            self._after_turn_task = asyncio.ensure_future(self._after_turn())

    async def _after_turn(self) -> None:
        await self.speaker.wait_idle(timeout=120.0)
        if not self._model_active and not self._calls:
            self.hooks.set_state("listening")

    def _finish_user(self) -> None:
        text = join_parts(self._user_parts)
        self._user_parts = []
        if not text:
            return
        taint.begin(text)  # a new user request: a fresh taint scope for its tool calls (brain/taint.py)
        quiet = getattr(self.hooks, "confirm_quiet", None)
        echo = callable(quiet) and bool(quiet())  # SAM's own confirmation question may still be in the mic
        if not echo and self.app.confirm.offer_transcript(text):  # the user's own yes/no answers it
            self._utt_consumed = True
        self._utt_heard.append(text)
        self.app.bus.publish(Caption(text=text, role="user", final=True))
        self.app.bus.publish(Transcript(role="user", text=text, source="live"))

    def _finish_assistant(self, *, interrupted: bool = False) -> None:
        text = join_parts(self._assistant_parts)
        self._assistant_parts = []
        if not text:
            return
        if interrupted:
            text += " …"
        self.app.bus.publish(Caption(text=text, role="assistant", final=True))
        if not self._say_turn:  # say(): the requester (monitor/worker) records its own text
            self.app.bus.publish(Transcript(role="assistant", text=text, source="live"))

    def _on_go_away(self, go_away: Any) -> None:
        """Reconnect with the resumption handle: now when idle, else when the
        turn ends or shortly before the server's deadline."""
        left = parse_duration_s(getattr(go_away, "time_left", None))
        log.info("live GoAway: %.1f s left", left)
        if not self._model_active and not self._calls:
            self._reconnect.set()
            return
        self._go_away_pending = True
        loop = asyncio.get_running_loop()
        if self._go_away_timer is not None:
            self._go_away_timer.cancel()
        self._go_away_timer = loop.call_later(max(0.0, left - 1.5), self._reconnect.set)

    # -- watchdog ------------------------------------------------------------------------------------------
    def _arm_watchdog(self) -> None:
        self._disarm_watchdog()
        delay = float(self.app.config.get("voice.watchdog_s", 5))
        self._watchdog = asyncio.get_running_loop().call_later(delay, self._on_watchdog)

    def _disarm_watchdog(self) -> None:
        if self._watchdog is not None:
            self._watchdog.cancel()
            self._watchdog = None

    def _mark_responded(self) -> None:
        self._responded = True
        self._disarm_watchdog()

    def _on_watchdog(self) -> None:
        self._watchdog = None
        if self._responded or self._closing or self._calls:
            return
        self._finish_user()  # publish what Live heard so far (once, with the confirmation check)
        if self._utt_consumed:
            # The utterance was a spoken yes/no for a pending confirmation
            # (e.g. a worker's question): no reply is owed, Live is not stalled.
            self._responded = True
            if self._turn is not None:
                self._turn.finish(outcome="confirm_answer")
                self._turn = None
            return
        text = join_parts(self._utt_heard)
        if not text:
            # No words were heard: noise or a word Live's VAD ignored. Not a
            # stall -- Live stays, and no STT is spent on it.
            self._responded = True
            if self._turn is not None:
                self._turn.finish(outcome="no_transcript")
                self._turn = None
            return
        waited = float(self.app.config.get("voice.watchdog_s", 5))
        self.degraded = True
        self.degraded_reason = f"no reply within {waited:g} s"
        self.app.timing.record("live_watchdog", waited * 1000.0, kind="live", model=self.model)
        self.app.publish_status("live", "degraded", self.degraded_reason)
        # Live's own transcript replaces STT only after a PASSING self-test (it
        # saves the ~1.4-2.9 s KurdishTTS STT call); otherwise the cascade
        # transcribes the audio itself -- the first real self-test (2026-09-24)
        # measured Live's Sorani CER at 0.54. The Transcript was already
        # published by _finish_user, so the cascade does not publish it again.
        trusted = selftest_verdict(self.app.config.get("voice.selftest")) == "pass"
        if self._turn is not None:
            self._turn.finish(outcome="watchdog")
            self._turn = None
        self.hooks.live_stalled(self._last_utterance, text if trusted else "", self._eos_at or time.perf_counter(),
                                True)


__all__ = ["LiveVoice"]
