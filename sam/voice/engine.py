"""VoiceEngine (``app.voice``): chooses Live or Cascade, owns mic/speaker,
the hotkey and the listening windows (contract 3.1).

Triggers: the global hotkey (default Ctrl+Alt+Space) or a click on the island
opens listening. Since the real use on 2026-09-24 (the TV and the family were
transcribed as commands, 149 STT calls, every free model quota used up) the
default is push-to-talk turns with a short follow-up window, and "always
listening" is an opt-in that also needs the spoken name (listening.py). Every
frame passes a near-field gate (gate.py) and, once the user recorded a
voiceprint, every utterance is checked against it before STT or Live
(voiceprint.py, frames.py). There is no always-on wake word: v1's Whisper
"Hey SAM" check cost 2.6-3.3 s of CPU per sound (reports/audit-latency.json).

Engine choice (setting ``voice.engine`` auto/live/cascade): Live needs a Gemini
key; "auto" uses Live ONLY after a self-test passed (design 2.1: Live's Sorani
is unproven -- the first real self-test on 2026-09-24 measured CER 0.54, one
sentence heard as Latin "Kashmir Chand o bazaar kharidari"), else Cascade. The
automatic self-test runs at most once a day (it costs 3 Gemini TTS requests of
a tiny free quota). A Live stall or connect failure switches the rest of the
listening window to Cascade (``live_degraded``); the next window tries Live
again. Always-listening uses the cascade (the name is checked on the
transcript before any model call; Live would answer first).

Barge-in: while SAM speaks, near-field speech only ducks the speaker; the
reply is cut after ``voice.barge_in_ms`` (400 ms) and -- with a voiceprint --
only by the user's own voice (frames.py).

Echo guard: sounddevice has no echo cancellation (reports/realtime-voice.json).
On laptop speakers SAM would hear itself, so while it speaks (and 300 ms after)
quiet mic frames are replaced by silence; only speech louder than
``voice.barge_in_rms`` passes. "auto" enables it when the output device does
not look like a headset.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

from ..events import (ConfirmRequest, Error, LevelMeter, SettingsChanged, SpeakRequest, ToolFinished,
                      ToolStarted, Transcript, VoiceState)
from . import kurdish_http, strings
from .audio import MicStream, Speaker, default_device_name, list_devices, looks_like_headset, pick_device, refresh_devices
from .cascade import CascadeVoice
from .engine_support import EngineSupport
from .enroll import EnrollmentSupport
from .frames import FramePipeline
from .gate import GateSettings, NearFieldGate
from .hotkey import GlobalHotkey
from .listening import ListeningPolicy, asks_enrollment, starts_with_name
from .live import LiveVoice
from .notices import VoiceNotice
from .quota import next_pacific_midnight, reset_time_ckb, rests
from .selftest import selftest_verdict
from .stt import SttRouter
from .tts import TtsRouter
from .vad import Endpointer, FrameClassifier
from .voiceprint import SpeakerCheck

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
    "voice.selftest_min_interval_s": 86400,          # the automatic self-test: at most once a day
    "voice.tts_prewarm": False,                      # lazy: phrases are cached when really spoken
    "voice.selftest_max_cer": 0.35,
    "voice.selftest_max_ttfa_ms": 4000,
    "voice.barge_in_ms": 400,                        # voiced time before SAM's reply is cut
    "voice.duck_gain": 0.35,                         # speaker volume while a possible barge-in is checked
    "voice.live_hybrid_vad": True,                   # live-guide "Hybrid VAD": audio_stream_end at local end of speech
    "voice.live_start_sensitivity": "low",           # Live automatic VAD: start-of-speech sensitivity (noisy homes)
    "voice.tts_low_budget_share": 0.2,               # KurdishTTS: under 20% of the month left -> first sentence only
    "voice.tts_verbalize_numbers": "decimals",       # decimals|all|off (KurdishTTS reads whole numbers itself)
    "voice.tts_first_audio_s": 2.5,                  # Gemini TTS: no audio by then -> KurdishTTS
    "voice.tts_chunk_gap_s": 3.0,                    # Gemini TTS: a longer gap after audio ends the piece
    "voice.stt_gemini_timeout_s": 8.0,
    # Listening windows (listening.py) and the near-field gate (gate.py).
    # End of speech for the local endpointer (cascade and Live's hybrid VAD): the
    # real use of 2026-09-25 split «... لۆ بکەوە» + «بڕۆ سەر چار ...» at a ~0.7 s
    # pause with 600 ms (voice.silence_ms stays Live's server-side VAD setting).
    "voice.end_silence_ms": 900,
    "voice.merge_window_s": 1.2,                     # speech this soon after an utterance continues it (frames.py)
    "voice.start_timeout_s": 8,                      # a click opens listening for one utterance
    "voice.followup_s": 6,                           # after SAM's answer: one more utterance ...
    "voice.followup_turns": 2,                       # ... at most this many per click (listening.py)
    "voice.kurdishtts_tts_first_audio_s": 5.0,       # KurdishTTS TTS: no audio by then -> give up, rest 60 s
    "voice.kurdishtts_stt_timeout_s": 6.0,           # KurdishTTS STT: + the utterance's length
    "voice.gate_margin_db": 14.0,
    "voice.gate_abs_min_db": -50.0,
    "voice.gate_ceiling_db": -30.0,
    "voice.gate_min_voiced_ms": 300,
    "voice.gate_user_level_db": None,                # measured by the enrollment / learned from turns
    "voice.gate_learn": True,
    # «تەنها دەنگی من» (voiceprint.py): on by default once a voiceprint exists; it
    # gates follow-ups / barge-ins / always-listening, never the first utterance after a click.
    "voice.only_my_voice": True,
    "voice.only_my_voice_sensitivity": "normal",     # low 0.15 | normal 0.20 | high 0.28, then owner-adaptive
    "voice.speaker_model_path": "",
}


class VoiceEngine(EngineSupport, FramePipeline, ListeningPolicy, EnrollmentSupport):
    def __init__(self, app: Any, *, mic_factory: Callable[[], Any] | None = None, speaker: Any = None,
                 stt: Any = None, tts: Any = None, live_factory: Callable[[], Any] | None = None,
                 hotkey_factory: Callable[[str, Callable[[], None]], Any] | None = None,
                 llm_stream: Any = None, speaker_check: SpeakerCheck | None = None) -> None:
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
        self.speaker_check = speaker_check or SpeakerCheck(app)
        self.gate = NearFieldGate(GateSettings.from_config(app.config),
                                  frame_ms=int(app.config.get("voice.mic_block_ms", 30)))
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
        self._mic_task: asyncio.Task[Any] | None = None
        self._watch_task: asyncio.Task[Any] | None = None
        self._selftest_task: asyncio.Task[Any] | None = None
        self._hotkey_task: asyncio.Task[Any] | None = None
        self._hotkey_lock: asyncio.Lock | None = None
        self._stopped = False
        self._prewarm_started = False
        self._hotkey: Any = None
        self._unsubs: list[Callable[[], None]] = []
        self._lock: asyncio.Lock | None = None
        self._published: tuple[str, str] | None = None
        self._init_frames()
        self._init_listening()
        self._init_enrollment()

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
        self._notice_active_rests()

    async def stop(self) -> None:
        for unsubscribe in self._unsubs:
            unsubscribe()
        self._unsubs = []
        try:
            await self.enroll_cancel()
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
        if self.always_listening():
            return "cascade"  # the name «سام» is checked on the transcript before any model hears it
        if mode == "live":
            return "live"
        # auto = the cascade. The user's real test (2026-09-24): Live heard
        # «سڵاو سام چۆنی» as Korean and answered in English, then Italian -- Live
        # is an explicit choice only. ``voice.auto_live`` (off) restores the old
        # rule: Live after a passing self-test (design 2.1).
        if not self.app.config.get("voice.auto_live", False):
            return "cascade"
        verdict = selftest_verdict(self.app.config.get("voice.selftest"))
        return "live" if verdict == "pass" else "cascade"

    def live_wanted(self) -> bool:
        """The user's settings can put Live to use (it was chosen explicitly,
        or "auto" may pick it): only then is a Gemini self-test worth its quota."""
        mode = str(self.app.config.get("voice.engine", "auto") or "auto")
        return mode == "live" or (mode == "auto" and bool(self.app.config.get("voice.auto_live", False)))

    def live_text_trusted(self) -> bool:
        """Live's own transcript may replace STT only after a passing self-test."""
        return selftest_verdict(self.app.config.get("voice.selftest")) == "pass"

    async def toggle_listening(self) -> bool:
        """The island click / the hotkey: an explicit activation (the next
        utterance is the owner's). Right after «دەنگەکەت نەناسرایەوە — کلیک
        بکە» a click while listening re-opens the owner's turn instead of
        closing listening (listening.py)."""
        if self.listening:
            if self.rearm_on_click():
                # SAM goes quiet: over his voice the owner's words would be a barge-in (voiceprint-checked).
                self.stop_speaking_now()
                self._open_window(explicit=True)
                self._publish("listening", detail="rearm", force=True)
                try:
                    self.app.db.log_activity("voice", "owner_rearm", ok=True, source="voice", summary="click after hint")
                except Exception:  # noqa: BLE001
                    pass
                return True
            await self.stop_listening()
            return False
        await self.start_listening(explicit=True)
        return self.listening

    async def start_listening(self, *, explicit: bool = True) -> None:
        """``explicit``: the user asked (click / hotkey / the panel's mic
        button): the first utterance is the owner by definition. A window
        opened for a confirmation question or by unmuting passes False."""
        async with self._get_lock():
            if self.listening:
                return
            if self._enrolling:
                self.app.bus.publish(VoiceNotice(kind="enroll", text_ckb=strings.ENROLL_BUSY, detail="enrolling"))
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
            self.gate.configure(GateSettings.from_config(self.app.config))
            min_speech = max(int(self.app.config.get("voice.min_speech_ms", 250)),
                             int(self.app.config.get("voice.gate_min_voiced_ms", 300)))
            self._endpointer = Endpointer(frame_ms=block_ms, silence_ms=self._end_silence_ms(),
                                          min_speech_ms=min_speech,
                                          max_utterance_s=float(self.app.config.get("voice.max_utterance_s", 30)))
            self._echo_guard = self._echo_guard_wanted()
            self._utt = None
            self.listening = True
            self.engine_name = engine
            self._open_window(explicit=explicit)
            self.cascade.start()
            if engine == "cascade":
                self._start_prewarm()
                self.app.spawn(self._warm_connections(), "voice-warm")
            if self.speaker_check.enabled:
                self.app.spawn(self.speaker_check.warm(), "voice-speaker-model")
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
                await self.start_listening(explicit=False)
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
        """Silence SAM at once (contract 3.1): the island's stop, ``stop_all``,
        a spoken «بەسە» / «بوەستە» / «بێدەنگ بە», or the brain's "stop talking"
        intent (real use 2026-09-25: «کوڕە دەنگی بنەکەرە!»). Queued audio,
        fixed speech being read and every older turn's reply stop now; called
        from inside a turn, that turn itself is not cut (its short answer may
        still be said) and nothing is carried over to the next request. The
        microphone is untouched (``set_muted`` closes it)."""
        self.stop_speaking_now()

    def stop_speaking_now(self) -> None:
        """``stop_speaking`` for synchronous callers (same effect, no await)."""
        self.speaker.gain = 1.0
        if self.live is not None:
            self.live.stop_output()
        self.cascade.stop_now(reason="stop_speaking")

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
            "listening_window": self.listening_status(), "gate": self.gate.status(),
            "voiceprint": self.speaker_check.status(), "rests": rests(self.app).status(),
            "turns": self.cascade.status(), "end_silence_ms": self._end_silence_ms(),
        }

    # -- hooks used by LiveVoice / CascadeVoice --------------------------------------------------------------
    def set_state(self, state: str, detail: str = "") -> None:
        # "sleeping" is published only when a conversation ends (listening.py):
        # the conversation ends on it (contract), so a spoken alert outside a
        # window must fall back to "idle", not "sleeping".
        if self.muted and state in ("listening", "idle"):
            state = "muted"
        elif not self.listening and state == "listening":
            state = "idle"
        self.note_state(state)
        self._publish(state, detail=detail)

    def confirm_quiet(self) -> bool:
        """True while SAM's own confirmation question may still be in the mic."""
        return time.perf_counter() < getattr(self.cascade, "confirm_quiet_until", 0.0)

    def admit_transcript(self, text: str, meta: dict[str, Any] | None = None) -> str | None:
        """The cascade's last check before a model hears ``text`` (None = drop).

        - a clear yes/no to a pending confirmation always passes (the broker
          takes it);
        - always-listening: the utterance must start with «سام», except the
          one follow-up right after an answer that sounds like the user
          (listening.py) -- checked BEFORE the enrollment phrase, so the TV
          cannot open the enrollment either;
        - «دەنگم بناسە» opens the voice enrollment dialog (no model call);
        - a first utterance after a click that produced words (or any verified
          one) may teach the gate the user's level (gate.py: guarded)."""
        meta = meta or {}
        levels = meta.get("levels") or {}
        classify = getattr(self.app.confirm, "classify_pending", None)
        if classify is not None and classify(text) is not None:
            return text
        continued = getattr(meta.get("continues"), "stage", "") in ("held", "replying", "merged")
        if self.requires_name() and not continued:   # a continuation joins an utterance that had the name
            if starts_with_name(text):
                self.named_request()
            elif not self.name_exempt(meta):
                self.gate.note_background(levels.get("p50_db") if levels.get("frames") else None)
                self.app.bus.publish(VoiceNotice(kind="ignored", text_ckb=strings.IGNORED_NO_NAME, detail="no_name"))
                try:
                    self.app.db.log_activity("voice", "no_name", ok=True, source="voice",
                                             summary="ignored: no «سام»")
                except Exception:  # noqa: BLE001
                    pass
                return None
        if asks_enrollment(text):
            self.request_enrollment("voice")
            self.app.spawn(self.cascade.speak(strings.ENROLL_SPOKEN, source="system"), "voice-enroll-ack")
            return None
        if (meta.get("first") or meta.get("verified")) and levels.get("frames", 0) >= 5 \
                and self.app.config.get("voice.gate_learn", True):
            learned = self.gate.learn_user_level(levels)
            stored = self.app.config.get("voice.gate_user_level_db", None)
            if learned is not None and (not isinstance(stored, (int, float)) or abs(stored - learned) >= 1.0):
                self.app.config.set("voice.gate_user_level_db", learned)
        embed = meta.get("owner_embed")
        if meta.get("owner") and embed is not None:
            meta["owner_embed"] = None          # once per utterance
            self.app.spawn(self._adopt_owner(embed, float(meta.get("speech_ms") or 0.0),
                                             levels.get("p50_db")), "voice-owner-adapt")
        return text

    async def _adopt_owner(self, embed: Any, speech_ms: float, level: Any) -> None:
        """The owner's turn after a click produced words: it joins the
        voiceprint (the last 10 owner utterances with the enrollment,
        DPAPI-protected), and the follow-up threshold follows how the owner
        scores against it (voiceprint.py)."""
        try:
            vector = await embed
        except Exception as exc:  # noqa: BLE001 - adaptation is optional, but a broken model is reported
            self.speaker_check.unavailable_reason = f"{type(exc).__name__}: {exc}"[:160]
            self._voiceprint_unavailable()
            return
        learned = await self.speaker_check.learn_owner(vector, speech_ms=speech_ms)
        if not learned:
            return
        try:
            self.app.db.log_activity("voice", "voiceprint_adapted", ok=True, source="voice",
                                     summary=f"score={learned['score']} n={learned['n']} level={level} "
                                             f"threshold={learned['threshold']}"[:200])
        except Exception:  # noqa: BLE001
            pass

    def reset_user_level(self) -> dict[str, Any]:
        """Settings «ئاستی دەنگم لەبیر بکە»: forget the learned speech level."""
        self.gate.forget_user_level()
        self.app.config.set("voice.gate_user_level_db", None)
        try:
            self.app.db.log_activity("voice", "user_level_reset", ok=True, source="voice", summary="")
        except Exception:  # noqa: BLE001
            pass
        return {"ok": True}

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
    def _end_silence_ms(self) -> int:
        try:
            return max(300, int(self.app.config.get("voice.end_silence_ms", 900)))
        except (TypeError, ValueError):
            return 900

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

    def _notice_active_rests(self) -> None:
        """A daily Gemini rest from before a restart is shown again on the island."""
        holder = rests(self.app)
        until = holder.until("gemini_tts")
        if until and holder.reason("gemini_tts") == "daily":
            self.app.bus.publish(VoiceNotice(kind="quota", text_ckb=strings.GEMINI_VOICE_DAILY.format(
                time=reset_time_ckb(until)), detail="gemini_tts", until=until))

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
            self.cascade.release_holds()    # a continuation that was being spoken will not end now
            if reason == "user":
                await self.cascade.stop_speaking()
            self._utt = None
            self.speaker.gain = 1.0
            self.live_degraded = False
            self.engine_name = ""
            if publish:
                # The user closed listening: the conversation ends (contract 3.1).
                self.conversation_ended()
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

    def _busy(self) -> bool:
        try:
            tools_running = bool(self.app.tools.running())
        except Exception:  # noqa: BLE001
            tools_running = False
        return (self.speaker.playing or self.cascade.busy or (self.live is not None and self.live.busy)
                or bool(getattr(self.app.confirm, "has_pending", False)) or tools_running or self._verifying > 0
                or (self._endpointer is not None and self._endpointer.in_speech))

    async def _watch(self, interval_s: float = 0.25) -> None:
        """Listening windows, the end of a conversation, idle speaker release."""
        while True:
            await asyncio.sleep(interval_s)
            try:
                if not self.listening:
                    await self.speaker.close_if_idle()
                await self._window_tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("voice window watcher failed")

    async def _auto_selftest(self, first_delay_s: float = 5.0, every_s: float = 15.0) -> None:
        """Run the self-test as soon as a Gemini key appears (design 2.1), and
        again only after an inconclusive one -- never more than once per
        ``voice.selftest_min_interval_s`` (a day): it spends 3 Gemini TTS
        requests, and on 2026-09-24 the self-test plus the phrase prewarm used
        up the free TTS quota before the user's first sentence. Never while
        Gemini TTS rests after a quota error, and only while the settings can
        use Live at all (``live_wanted``: "auto" no longer picks Live)."""
        await asyncio.sleep(first_delay_s)
        while True:
            result = self.app.config.get("voice.selftest")
            verdict = selftest_verdict(result)
            interval = float(self.app.config.get("voice.selftest_min_interval_s", 86400) or 86400)
            last = float(result.get("at") or 0) if isinstance(result, dict) else 0.0
            recent = time.time() - last < interval
            due = verdict == "none" or (verdict == "inconclusive" and not recent)
            if due and self.live_wanted() and self.app.secrets.has("gemini_api_key") and not self.muted \
                    and not rests(self.app).resting("gemini_tts"):
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
        # Runs inside the turn that asked (cascade.REPLY_GEN is inherited by the
        # task below): an older turn's question is not read out (cascade.speak).
        async def ask() -> None:
            if (self.app.config.get("voice.listen_on_confirm", True) and not self.listening and not self.muted):
                await self.start_listening(explicit=False)
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
        if event.role == "assistant" and _no_model_reply(event.text):
            text, until = self._models_notice()
            self.app.bus.publish(VoiceNotice(kind="models", text_ckb=text, detail="exhausted", until=until))

    def _models_notice(self) -> tuple[str, float]:
        """(«سنووری ئەمڕۆ پڕە — دوای کاتژمێر ١٠ی بەیانی», reset time) when Gemini
        answered 429 today; else the generic sentence and 0."""
        try:
            limited = self.app.db.scalar("SELECT COALESCE(SUM(rate_limited),0) FROM usage_counters WHERE day=? "
                                         "AND provider='gemini'", (self.app.db.quota_day("gemini"),))
        except Exception:  # noqa: BLE001
            limited = 0
        if limited or rests(self.app).reason("gemini_tts") == "daily":
            reset = next_pacific_midnight()
            return strings.MODELS_EXHAUSTED_DAILY.format(time=reset_time_ckb(reset)), reset
        return strings.MODELS_EXHAUSTED, 0.0

    def _on_setting(self, event: SettingsChanged) -> None:
        if event.key == "voice.hotkey":
            current = getattr(self._hotkey, "hotkey", None) or getattr(self._hotkey, "keys", None)
            if not (getattr(self._hotkey, "registered", False) and current == event.value):
                self.app.spawn(self._register_hotkey(), "voice-hotkey-register")
        elif event.key.startswith("voice.gate_"):
            self.gate.configure(GateSettings.from_config(self.app.config))
            if event.key == "voice.gate_user_level_db" and event.value is None:
                self.gate.forget_user_level()
        elif event.key in ("voice.only_my_voice",):
            self.speaker_check.forget_cache()
        elif event.key in ("voice.engine", "voice.selftest") or event.key.startswith("voice.tts") \
                or event.key.startswith("voice.stt"):
            self._publish_component()


def _no_model_reply(text: str) -> bool:
    """The brain's "no model answered" sentence (sam/brain/responder.py SORANI_NO_MODEL)."""
    try:
        from ..brain.responder import SORANI_NO_MODEL
        if text.strip() == SORANI_NO_MODEL.strip():
            return True
    except Exception:  # noqa: BLE001 - the brain may be missing
        pass
    return "ناتوانم پەیوەندی بە مۆدێلەکانەوە بکەم" in (text or "")


__all__ = ["VoiceEngine", "VOICE_DEFAULTS"]
