"""VoiceEngine (``app.voice``): chooses Live or Cascade, owns mic/speaker,
the hotkey and the conversation window (contract 3.1).

Triggers (design 2.1): the global hotkey (default Ctrl+Alt+Space) or a click
on the island toggles listening; the window stays open while the user talks
and sleeps after ``voice.conversation_timeout_s`` (45 s) of silence unless
``voice.always_listening``. There is no always-on wake word: v1's Whisper
"Hey SAM" check cost 2.6-3.3 s of CPU per sound (reports/audit-latency.json).

Engine choice (setting ``voice.engine`` auto/live/cascade): Live needs a Gemini
key; "auto" uses Live ONLY after a self-test passed (design 2.1: Live's Sorani
is unproven -- the first real self-test on 2026-09-24 measured CER 0.54, one
sentence heard as Latin "Kashmir Chand o bazaar kharidari"), else Cascade. The
self-test starts as soon as a key exists, even while listening (it is its own
short session). A Live stall or connect failure switches the rest of the
conversation window to Cascade (``live_degraded``); the next window tries Live
again.

Barge-in (cascade): while SAM speaks, a voice start only ducks the speaker;
the reply is cut after ``voice.barge_in_ms`` (400 ms) of continuous voiced
audio. A shorter sound mid-reply (a quick "aha", a cough) is dropped without
STT -- the old rule cut the answer on ~150 ms of any voiced sound.

Echo guard: sounddevice has no echo cancellation (reports/realtime-voice.json).
On laptop speakers SAM would hear itself and barge in on its own voice, so
while it speaks (and 300 ms after) quiet mic frames are replaced by silence;
only speech louder than ``voice.barge_in_rms`` passes. "auto" enables it when
the output device does not look like a headset. The threshold is NOT measured
on this PC yet (no headset was connected on 2026-09-24).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

from ..events import (ConfirmRequest, Error, LevelMeter, SettingsChanged, SpeakRequest, ToolFinished,
                      ToolStarted, Transcript, VoiceState)
from . import kurdish_http, strings
from .audio import (MicStream, Speaker, default_device_name, level_from_rms, list_devices, looks_like_headset,
                    pcm_rms, pick_device, refresh_devices)
from .cascade import CascadeVoice
from .engine_support import EngineSupport
from .hotkey import GlobalHotkey
from .live import LiveVoice
from .selftest import selftest_verdict
from .stt import SttRouter
from .tts import TtsRouter
from .vad import Endpointer, FrameClassifier, VadEvent

log = logging.getLogger("sam.voice")

# Settings owned by the voice package beyond foundation's voice.* defaults.
VOICE_DEFAULTS: dict[str, Any] = {
    "voice.kurdishtts_base_url": "https://www.kurdishtts.com/api",
    "voice.kurdishtts_speaker": "sorani_1",          # free plan, v4 group (kurdishtts.com/llms.txt)
    "voice.kurdishtts_model_version": "v4",
    "voice.kurdishtts_monthly_tts_chars": 20000,     # free plan (kurdishtts.com/pricing, 2026-09-24)
    "voice.kurdishtts_monthly_stt_s": 7200,          # free plan: 2 h a month
    "voice.tts_style": "",                           # 3.8 TTS docs: leave empty for voice agents
    "voice.tts_max_chars": 480,                      # KurdishTTS refuses > 500 per request
    "voice.mic_block_ms": 30,                        # 20-40 ms (Live docs) and a webrtcvad frame
    "voice.preferred_devices": ["A50"],              # the user's headset, when plugged in
    "voice.echo_guard": "auto",                      # auto|on|off
    "voice.barge_in_rms": 0.05,                      # unmeasured default; tune on the real mic
    "voice.vad_aggressiveness": 2,
    "voice.vad_energy_floor": 0.004,
    "voice.min_speech_ms": 250,
    "voice.max_utterance_s": 30,
    "voice.live_prefix_padding_ms": 200,
    "voice.speaker_idle_close_s": 30,
    "voice.listen_on_confirm": True,
    # Ctrl+Alt+Space was already taken on this PC (2026-09-24); these were free.
    "voice.hotkey_fallbacks": ["win+alt+space", "ctrl+shift+alt+space"],
    "voice.selftest_auto": True,
    "voice.tts_prewarm": True,                       # cache the brain's short acknowledgements once
    "voice.selftest_max_cer": 0.35,
    "voice.selftest_max_ttfa_ms": 4000,
    "voice.barge_in_ms": 400,                        # voiced time before SAM's reply is cut
    "voice.duck_gain": 0.35,                         # speaker volume while a possible barge-in is checked
    "voice.live_hybrid_vad": True,                   # live-guide "Hybrid VAD": audio_stream_end at local end of speech
    "voice.tts_low_budget_share": 0.2,               # KurdishTTS: under 20% of the month left -> first sentence only
    "voice.tts_verbalize_numbers": "decimals",       # decimals|all|off (KurdishTTS reads whole numbers itself)
}


class VoiceEngine(EngineSupport):
    def __init__(self, app: Any, *, mic_factory: Callable[[], Any] | None = None, speaker: Any = None,
                 stt: Any = None, tts: Any = None, live_factory: Callable[[], Any] | None = None,
                 hotkey_factory: Callable[[str, Callable[[], None]], Any] | None = None,
                 llm_stream: Any = None) -> None:
        self.app = app
        # Also done by register(); repeated here (setdefault: idempotent) so an
        # engine built directly -- tests, acceptance scripts -- sees the same
        # defaults, e.g. the hotkey fallbacks.
        app.config.register_defaults(VOICE_DEFAULTS)
        self.speaker = speaker or Speaker(on_level=self._speaker_level,
                                          idle_close_s=float(app.config.get("voice.speaker_idle_close_s", 30)))
        self.stt = stt or SttRouter.default(app)
        self.tts = tts or TtsRouter.default(app)
        register_phrases = getattr(self.tts, "register_phrases", None)
        if register_phrases is not None:  # cache hits work even before/without the prewarm
            register_phrases(self.fixed_phrases())
        self.cascade = CascadeVoice(app, self.speaker, self.stt, self.tts, self, llm_stream=llm_stream)
        self._live_factory = live_factory or (lambda: LiveVoice(app, self.speaker, self))
        self._mic_factory = mic_factory or self._default_mic
        self._hotkey_factory = hotkey_factory or GlobalHotkey
        self.live: Any = None
        self.mic: Any = None
        self.state = "idle"
        self.engine_name = ""
        self.listening = False
        self.muted = False
        self.live_degraded = False
        self.devices: dict[str, Any] = {}
        self.hotkey_error: str | None = None
        self._hotkey_note = ""
        self._resume_after_mute = False
        self._classifier: FrameClassifier | None = None
        self._endpointer: Endpointer | None = None
        self._echo_guard = False
        self._echo_until = 0.0
        self._mic_task: asyncio.Task[Any] | None = None
        self._watch_task: asyncio.Task[Any] | None = None
        self._selftest_task: asyncio.Task[Any] | None = None
        self._hotkey_task: asyncio.Task[Any] | None = None
        self._hotkey_lock: asyncio.Lock | None = None
        self._stopped = False
        self._prewarm_started = False
        self._hotkey: Any = None
        self._unsubs: list[Callable[[], None]] = []
        self._last_activity = time.monotonic()
        self._last_level_at = 0.0
        self._lock: asyncio.Lock | None = None
        self._published: tuple[str, str] | None = None
        self._barge_pending = False
        self._utt_cut = False          # the current utterance barged in on SAM's reply

    # -- lifecycle ---------------------------------------------------------------------------------------
    async def start(self) -> None:
        self._stopped = False
        bus = self.app.bus
        self._unsubs = [
            bus.subscribe(SpeakRequest, self._on_speak_request),
            bus.subscribe(ConfirmRequest, self._on_confirm_request),
            bus.subscribe((ToolStarted, ToolFinished), self._on_tool_event),
            bus.subscribe(SettingsChanged, self._on_setting),
            bus.subscribe(Transcript, self._on_transcript),
        ]
        # Off the startup path: RegisterHotKey runs on a new thread and waits
        # for it (plus fallbacks when the chord is taken); the launcher builder
        # measured voice start at 0.56-5.4 s on this PC with it inline, and the
        # island only appears after every package's start() (design: < 3 s).
        self._hotkey_task = self.app.spawn(self._register_hotkey(), "voice-hotkey-register")
        self.app.spawn(self._scan_devices(), "voice-devices")
        self._watch_task = self.app.spawn(self._watch(), "voice-window")
        if self.app.config.get("voice.selftest_auto", True):
            self._selftest_task = self.app.spawn(self._auto_selftest(), "voice-selftest-auto")
        self._publish("idle", force=True)
        self._publish_component()

    async def stop(self) -> None:
        for unsubscribe in self._unsubs:
            unsubscribe()
        self._unsubs = []
        try:
            await self._stop_listening(reason="shutdown", publish=False)
        except Exception:  # noqa: BLE001
            log.exception("stop listening failed")
        for task in (self._watch_task, self._selftest_task):
            if task is not None:
                task.cancel()
        await self.cascade.close()
        # A registration still running is NOT cancelled: its thread would go on
        # and own the chord with nobody to release it. Each attempt waits at
        # most 2 s, so this stays inside the 8 s stop budget.
        self._stopped = True
        async with self._get_hotkey_lock():
            if self._hotkey is not None:
                await asyncio.to_thread(self._hotkey.stop)
                self._hotkey = None
        await asyncio.to_thread(self.speaker.close)
        for router in (self.stt, self.tts):
            closer = getattr(router, "aclose", None)
            if closer is not None:
                await closer()
        await kurdish_http.shared(self.app).close()

    # -- contract API ------------------------------------------------------------------------------------------
    def choose_engine(self) -> str:
        mode = str(self.app.config.get("voice.engine", "auto") or "auto")
        if mode == "cascade" or not self.app.secrets.has("gemini_api_key") or self.live_degraded:
            return "cascade"
        if mode == "live":
            return "live"
        # auto: Live only after a passing self-test (design 2.1). Not run yet,
        # inconclusive or failed -> the cascade, whose Sorani STT is measured.
        verdict = selftest_verdict(self.app.config.get("voice.selftest"))
        return "live" if verdict == "pass" else "cascade"

    def live_text_trusted(self) -> bool:
        """Live's own transcript may replace STT only after a passing self-test."""
        return selftest_verdict(self.app.config.get("voice.selftest")) == "pass"

    async def toggle_listening(self) -> bool:
        if self.listening:
            await self.stop_listening()
            return False
        await self.start_listening()
        return self.listening

    async def start_listening(self) -> None:
        async with self._get_lock():
            if self.listening:
                return
            self.muted = False
            engine = self.choose_engine()
            try:
                mic = await asyncio.to_thread(self._mic_factory)
                await mic.start()
            except Exception as exc:  # noqa: BLE001 - a missing mic is reported in Sorani
                self.app.bus.publish(Error(where="voice.mic", message_ckb=strings.MIC_FAILED,
                                           detail=self.app.redact(f"{type(exc).__name__}: {exc}")[:200]))
                self._publish("error", detail="mic", force=True)
                return
            self.mic = mic
            block_ms = int(self.app.config.get("voice.mic_block_ms", 30))
            self._classifier = FrameClassifier(aggressiveness=int(self.app.config.get("voice.vad_aggressiveness", 2)),
                                               energy_floor=float(self.app.config.get("voice.vad_energy_floor", 0.004)))
            self._endpointer = Endpointer(frame_ms=block_ms, silence_ms=int(self.app.config.get("voice.silence_ms", 600)),
                                          min_speech_ms=int(self.app.config.get("voice.min_speech_ms", 250)),
                                          max_utterance_s=float(self.app.config.get("voice.max_utterance_s", 30)))
            self._echo_guard = self._echo_guard_wanted()
            self.listening = True
            self.engine_name = engine
            self._last_activity = time.monotonic()
            self.cascade.start()
            if engine == "cascade":
                self._start_prewarm()
                self.app.spawn(self._warm_connections(), "voice-warm")
            self._mic_task = asyncio.ensure_future(self._mic_loop(mic))
            self.app.spawn(self.speaker.open(), "voice-speaker-open")
            if engine == "live":
                self.live = self._live_factory()
                self.app.spawn(self._open_live(self.live), "voice-live-open")
            self._publish("listening", force=True)

    async def stop_listening(self) -> None:
        await self._stop_listening(reason="user")

    async def set_muted(self, muted: bool) -> None:
        muted = bool(muted)
        if muted == self.muted:
            return
        if muted:
            self._resume_after_mute = self.listening
            self.muted = True
            await self._stop_listening(reason="mute", publish=False)
            self._publish("muted", force=True)
        else:
            self.muted = False
            if self._resume_after_mute:
                await self.start_listening()
            else:
                self._publish("idle", force=True)

    async def speak(self, text_ckb: str, *, interrupt: bool = False, source: str = "system") -> None:
        if not text_ckb or not text_ckb.strip():
            return
        live = self.live
        if live is not None and self.engine_name == "live" and live.ready:
            if await live.say(text_ckb, interrupt=interrupt, source=source):
                return
        await self.cascade.speak(text_ckb, interrupt=interrupt, source=source)

    @property
    def live_session_open(self) -> bool:
        """True while typed text should go into the open Live session (read by
        ``Conversation._live_forwarder`` and the chat page)."""
        return self.listening and self.engine_name == "live" and self.live is not None and self.live.ready

    async def send_text(self, text: str) -> bool:
        """Typed text into the open Live session (addition to contract 3.1,
        used by sam/brain/conversation.py and sam/ui/pages/chat.py). Publishes
        the user's Transcript (source "text"); False when no session is open,
        so the caller answers through the text brain instead."""
        text = (text or "").strip()
        live = self.live
        if not text or live is None or not self.live_session_open:
            return False
        consumed = self.app.confirm.offer_transcript(text)
        self.app.bus.publish(Transcript(role="user", text=text, source="text"))
        self.activity()
        if consumed:
            return True
        return bool(await live.send_user_text(text))

    async def stop_speaking(self) -> None:
        self.speaker.flush()
        if self.live is not None:
            self.live.stop_output()
        await self.cascade.stop_speaking()

    async def run_selftest(self) -> dict[str, Any]:
        from .selftest import run_selftest

        gemini_tts = getattr(self.tts, "providers", {}).get("gemini") if hasattr(self.tts, "providers") else None
        result = await run_selftest(self.app, tts=gemini_tts)
        self._publish_component()
        return result

    def status(self) -> dict[str, Any]:
        return {
            "engine": self.engine_name or self.choose_engine(), "state": self.state, "listening": self.listening,
            "muted": self.muted, "live_degraded": self.live_degraded,
            "live": self.live.status() if self.live is not None else None,
            "devices": self.devices, "echo_guard": self._echo_guard,
            "last_ttfa_ms": {"live": getattr(self.live, "last_ttfa_ms", None), "cascade": self.cascade.last_ttfa_ms},
            "last_answer_ms": {"cascade": getattr(self.cascade, "last_answer_ms", None)},
            "hotkey": {"keys": getattr(self._hotkey, "hotkey", None) or getattr(self._hotkey, "keys", None),
                       "registered": bool(getattr(self._hotkey, "registered", False)), "error": self.hotkey_error,
                       "note": self._hotkey_note},
            "stt": self.stt.status() if hasattr(self.stt, "status") else {},
            "tts": self.tts.status() if hasattr(self.tts, "status") else {},
            "selftest": self.app.config.get("voice.selftest"),
            "speaker": {"open": self.speaker.is_open, "underflows": getattr(self.speaker, "underflows", 0),
                        "error": getattr(self.speaker, "last_error", None)},
        }

    # -- hooks used by LiveVoice / CascadeVoice --------------------------------------------------------------
    def set_state(self, state: str, detail: str = "") -> None:
        # "sleeping" is published only when a conversation window closes: the
        # conversation ends on it (contract), so a spoken alert outside a
        # window must fall back to "idle", not "sleeping".
        if self.muted and state in ("listening", "idle"):
            state = "muted"
        elif not self.listening and state == "listening":
            state = "idle"
        self._publish(state, detail=detail)

    def activity(self) -> None:
        self._last_activity = time.monotonic()

    def confirm_quiet(self) -> bool:
        """True while SAM's own confirmation question may still be in the mic."""
        return time.perf_counter() < getattr(self.cascade, "confirm_quiet_until", 0.0)

    def live_stalled(self, pcm: bytes, user_text: str, eos_at: float, published: bool) -> None:
        self.live_degraded = True
        try:
            self.app.db.log_activity("voice", "live_watchdog", ok=False, summary=strings.LIVE_DEGRADED,
                                     source="live")
        except Exception:  # noqa: BLE001
            pass
        self.app.spawn(self._switch_to_cascade("watchdog", pcm=pcm, text=user_text or None, eos_at=eos_at,
                                               published=published), "voice-switch")

    def live_failed(self, reason: str) -> None:
        self.live_degraded = True
        self.app.spawn(self._switch_to_cascade(reason), "voice-switch")

    async def speak_fallback(self, text: str, source: str) -> None:
        await self.cascade.speak(text, source=source)

    # -- internals -----------------------------------------------------------------------------------------------
    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _publish(self, state: str, *, detail: str = "", force: bool = False) -> None:
        key = (state, self.engine_name)
        if not force and key == self._published:
            return
        self._published = key
        self.state = state
        self.app.bus.publish(VoiceState(state=state, engine=self.engine_name, detail=detail))  # type: ignore[arg-type]

    def _publish_component(self) -> None:
        stt_ok = self.stt.configured() if hasattr(self.stt, "configured") else True
        tts_ok = self.tts.configured() if hasattr(self.tts, "configured") else True
        state = "ok" if stt_ok and tts_ok else "unconfigured"
        self.app.publish_status("voice", state, f"engine={self.choose_engine()} stt={stt_ok} tts={tts_ok}")

    def _default_mic(self) -> MicStream:
        """Runs in a worker thread: re-scan devices (hot-plugged headset) and pick one."""
        idle_audio = not self.speaker.is_open and not getattr(self.speaker, "_opening", False)
        if idle_audio:  # PortAudio may only be re-initialised while no stream exists
            refresh_devices()
        keywords = tuple(self.app.config.get("voice.preferred_devices", ["A50"]) or ())
        mic_dev = pick_device("input", self.app.config.get("voice.input_device"), keywords)
        out_dev = pick_device("output", self.app.config.get("voice.output_device"), keywords)
        if idle_audio:
            self.speaker.device = out_dev.index if out_dev else None
        self.devices = {**self.devices,
                        "input": mic_dev.name if mic_dev else default_device_name("input"),
                        "output": out_dev.name if out_dev else default_device_name("output")}
        return MicStream(device=mic_dev.index if mic_dev else None,
                         block_ms=int(self.app.config.get("voice.mic_block_ms", 30)))

    async def _scan_devices(self) -> None:
        listing = await asyncio.to_thread(list_devices)
        self.devices = {**self.devices, "inputs": [d.public() for d in listing.get("inputs", [])],
                        "outputs": [d.public() for d in listing.get("outputs", [])],
                        "hostapi": listing.get("hostapi", ""), "ok": listing.get("ok", False)}

    def _echo_guard_wanted(self) -> bool:
        mode = str(self.app.config.get("voice.echo_guard", "auto") or "auto")
        if mode in ("on", "off"):
            return mode == "on"
        return not looks_like_headset(self.devices.get("output", ""))

    def _speaker_level(self, level: float) -> None:  # PortAudio thread
        self.app.bus.publish_threadsafe(LevelMeter(source="speaker", level=level))

    async def _open_live(self, live: Any) -> None:
        ok = await live.open()
        if not ok and self.live is live and not self.live_degraded:
            self.live_failed("connect timeout")

    async def _switch_to_cascade(self, reason: str, *, pcm: bytes | None = None, text: str | None = None,
                                 eos_at: float | None = None, published: bool = False) -> None:
        live, self.live = self.live, None
        if self.listening:
            self.engine_name = "cascade"
        if live is not None:
            if pcm is None and not getattr(live, "_responded", True) and getattr(live, "_last_utterance", b""):
                pcm, eos_at = live._last_utterance, live._eos_at  # noqa: SLF001 - unanswered utterance
            await live.close()
        message = strings.LIVE_DEGRADED if reason == "watchdog" else strings.LIVE_FAILED
        self.app.bus.publish(Error(where="voice.live", message_ckb=message, detail=self.app.redact(reason)[:200]))
        if pcm or text:
            self.cascade.submit_utterance(pcm or b"", eos_at or time.perf_counter(), text=text, published=published)
        self._publish("listening" if self.listening else "idle", detail=reason)

    async def _stop_listening(self, *, reason: str, publish: bool = True) -> None:
        async with self._get_lock():
            if not self.listening:
                return
            self.listening = False
            mic, self.mic = self.mic, None
            task, self._mic_task = self._mic_task, None
            if mic is not None:
                await mic.stop()
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            live, self.live = self.live, None
            if live is not None:
                await live.close()
            if reason == "user":
                await self.cascade.stop_speaking()
            self.live_degraded = False
            self.engine_name = ""
            if publish:
                self._publish("sleeping", detail=reason, force=True)

    async def _mic_loop(self, mic: Any) -> None:
        try:
            async for frame in mic.frames():
                await self._on_frame(frame)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("mic loop failed")
            self._publish("error", detail="mic", force=True)

    async def _on_frame(self, frame: bytes) -> None:
        now = time.perf_counter()
        mono = time.monotonic()
        rms = pcm_rms(frame)
        if mono - self._last_level_at >= 0.05:  # <= 20 Hz for the orb
            self._last_level_at = mono
            self.app.bus.publish(LevelMeter(source="mic", level=level_from_rms(rms)))
        if self._echo_guard:
            if self.speaker.playing:
                self._echo_until = mono + 0.3
            if mono < self._echo_until and rms < float(self.app.config.get("voice.barge_in_rms", 0.05)):
                frame, rms = bytes(len(frame)), 0.0  # digital silence of the same length
        assert self._classifier is not None and self._endpointer is not None
        event = self._endpointer.process(frame, self._classifier.is_speech(frame, rms), now)
        live = self.live if self.engine_name == "live" else None
        if event is not None and event.kind == "start":
            self.activity()
            if live is not None:
                live.note_speech_start()
            elif self.speaker.playing:
                # Maybe a barge-in, maybe a backchannel or a cough: duck now, decide on voiced time.
                self._barge_pending = True
                self.speaker.gain = float(self.app.config.get("voice.duck_gain", 0.35))
            else:
                self.cascade.barge_in()  # nothing audible yet: a new request replaces the thinking one
        if self._barge_pending and self._endpointer.current_speech_ms >= float(
                self.app.config.get("voice.barge_in_ms", 400)):
            self._barge_pending = False
            self.speaker.gain = 1.0
            self._utt_cut = self.cascade.barge_in()
        if live is not None:
            await live.send_audio(frame)
        if event is not None and event.kind == "end":
            if self._barge_pending:  # too short to be a barge-in: SAM keeps talking, no STT spent
                self._barge_pending = False
                self.speaker.gain = 1.0
                self.activity()
                return
            self._on_utterance(event, live)

    def _on_utterance(self, event: VadEvent, live: Any) -> None:
        self.activity()
        if event.too_short:
            if live is None:
                self.cascade.resume_carry()
            return
        cut, self._utt_cut = self._utt_cut, False
        if live is not None:
            live.note_end_of_speech(event.pcm, event.eos_at)
        else:
            self.cascade.submit_utterance(event.pcm, event.eos_at, cut_reply=cut)

    def _busy(self) -> bool:
        try:
            tools_running = bool(self.app.tools.running())
        except Exception:  # noqa: BLE001
            tools_running = False
        return (self.speaker.playing or self.cascade.busy or (self.live is not None and self.live.busy)
                or bool(getattr(self.app.confirm, "has_pending", False)) or tools_running
                or (self._endpointer is not None and self._endpointer.in_speech))

    async def _watch(self) -> None:
        """Conversation window + idle speaker release, once a second."""
        while True:
            await asyncio.sleep(1.0)
            try:
                if not self.listening:
                    await self.speaker.close_if_idle()
                    continue
                if self.muted or self.app.config.get("voice.always_listening", False):
                    continue
                if self._busy():
                    self.activity()
                    continue
                timeout = float(self.app.config.get("voice.conversation_timeout_s", 45))
                if time.monotonic() - self._last_activity >= timeout:
                    await self._stop_listening(reason="timeout")
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("voice window watcher failed")

    async def _auto_selftest(self, first_delay_s: float = 5.0, every_s: float = 15.0,
                             retry_after_s: float = 3600.0) -> None:
        """Run the self-test as soon as a Gemini key appears (design 2.1) --
        also while listening: it is its own short Live session, and until it
        passes "Automatic" stays on the cascade -- and again (at most once per
        run, an hour later) after an inconclusive one."""
        await asyncio.sleep(first_delay_s)
        while True:
            result = self.app.config.get("voice.selftest")
            verdict = selftest_verdict(result)
            stale = verdict == "inconclusive" and time.time() - float(result.get("at") or 0) >= retry_after_s
            due = verdict == "none" or stale
            if due and self.app.secrets.has("gemini_api_key") and not self.muted:
                try:
                    await self.run_selftest()
                except Exception:  # noqa: BLE001
                    log.exception("automatic self-test failed")
                return
            await asyncio.sleep(every_s)

    # -- event handlers ------------------------------------------------------------------------------------------
    def _on_speak_request(self, event: SpeakRequest) -> None:
        self.app.spawn(self.speak(event.text_ckb, interrupt=event.interrupt, source=event.source), "voice-speak")

    def _on_confirm_request(self, event: ConfirmRequest) -> None:
        async def ask() -> None:
            if (self.app.config.get("voice.listen_on_confirm", True) and not self.listening and not self.muted):
                await self.start_listening()
            await self.speak(event.question_ckb, source="confirm")
        self.app.spawn(ask(), "voice-confirm")

    def _on_tool_event(self, event: Any) -> None:
        self.activity()
        if getattr(event, "source", "") != "cascade" or not self.listening:
            return
        if isinstance(event, ToolStarted):
            self.set_state("working", event.name)
        elif not self.speaker.playing:
            self.set_state("thinking")

    def _on_transcript(self, event: Transcript) -> None:
        self.activity()

    def _on_setting(self, event: SettingsChanged) -> None:
        if event.key == "voice.hotkey":
            current = getattr(self._hotkey, "hotkey", None) or getattr(self._hotkey, "keys", None)
            if not (getattr(self._hotkey, "registered", False) and current == event.value):
                self.app.spawn(self._register_hotkey(), "voice-hotkey-register")
        elif event.key in ("voice.engine", "voice.selftest") or event.key.startswith("voice.tts") \
                or event.key.startswith("voice.stt"):
            self._publish_component()


__all__ = ["VoiceEngine", "VOICE_DEFAULTS"]
