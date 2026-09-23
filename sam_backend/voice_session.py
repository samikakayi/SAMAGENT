"""The hands-free loop: wake, listen, ask SAM the ordinary way, speak, repeat.

This controller deliberately owns no intelligence. It captures an utterance
with the voice service that already exists, hands the text to `agent.chat` --
the same entry point the typed box uses -- and speaks whatever comes back. A
spoken request is therefore an ordinary request: the same policy, the same
approvals, the same router, the same task store. Nothing gets easier because
it arrived through a microphone.

Two things it does own, because nothing else can:

*Turn-taking.* Wake detection is suppressed for as long as SAM is speaking and
for a moment afterwards, so its own voice cannot start a new session. Without
that the assistant answers itself forever.

*The continuation window.* After an answer, a short period where a follow-up
needs no wake phrase, because saying "Hey SAM" before every sentence is not a
conversation.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from .wake import (
    FRAME_MS,
    POST_WAKE_MAX_FRAMES,
    POST_WAKE_MIN_SPEECH_FRAMES,
    POST_WAKE_ONSET_CAP_FRAMES,
    LocalPhraseDetector,
    VoiceState,
    VoiceStatus,
    WakeDetection,
    WakeWordService,
)

# Bounds on one spoken turn. A microphone that records without end is a
# different product, and a worse one.
MAX_COMMAND_SECONDS = 20.0
SPEECH_START_TIMEOUT = 5.0
# How long to wait for the listener to finish capturing what followed the
# phrase: its own bounds, plus room for the examination that preceded them.
POST_WAKE_WAIT_SECONDS = (POST_WAKE_MAX_FRAMES + POST_WAKE_ONSET_CAP_FRAMES) * FRAME_MS / 1000 + 10.0


class VoiceConversationController:
    """Drives one hands-free session and returns to waiting for the phrase."""

    def __init__(self, settings: Any, voice: Any, agent: Any, *,
                 wake: WakeWordService | None = None,
                 submit: Callable[[str], dict[str, Any]] | None = None) -> None:
        self.settings = settings
        self.voice = voice
        self.agent = agent
        # One speech model, not two: the wake detector listens with the same
        # weights dictation uses, which are already resident.
        self.wake = wake or WakeWordService(
            settings,
            detector=LocalPhraseDetector(
                str(getattr(settings, 'local_stt_model', 'small') or 'small'),
                shared=getattr(voice, 'stt', None)))
        # Whoever asks SAM to speak -- this loop or the manual voice endpoint --
        # the wake listener goes deaf for the duration. Doing it here rather
        # than around our own playback means no other caller can make SAM hear
        # itself say the phrase and answer itself.
        voice.on_speaking = self._while_speaking
        # Injectable so tests can drive the loop without an event loop or a
        # model; production passes nothing and gets the real agent.
        self._submit = submit
        self.status = VoiceStatus(wake_phrase=self.wake.phrase)
        self._lock = threading.Lock()
        self._busy = threading.Event()
        self._listeners: list[Callable[[VoiceStatus], None]] = []
        self.transcripts: list[str] = []

    # -- state --------------------------------------------------------------
    def _set(self, state: VoiceState, detail: str = "", **fields: Any) -> None:
        with self._lock:
            self.status.state = state
            self.status.detail = detail
            self.status.wake_phrase = self.wake.phrase
            self.status.sensitivity = self.wake.sensitivity
            self.status.continuation_seconds = self.continuation_seconds
            self.status.updated_at = time.time()
            for key, value in fields.items():
                setattr(self.status, key, value)
            snapshot = self.status
        for listener in list(self._listeners):
            try:
                listener(snapshot)
            except Exception:  # noqa: BLE001 - a panel must not break the loop
                pass

    def subscribe(self, listener: Callable[[VoiceStatus], None]) -> None:
        self._listeners.append(listener)

    # -- configuration ------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return bool(getattr(self.settings, "hands_free_enabled", False))

    @property
    def continuation_seconds(self) -> float:
        return float(getattr(self.settings, "hands_free_continuation_seconds", 10.0) or 0.0)

    @property
    def auto_speak(self) -> bool:
        return bool(getattr(self.settings, "hands_free_auto_speak", True))

    @property
    def language(self) -> str | None:
        return getattr(self.settings, "voice_language", None)

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> dict[str, Any]:
        """Begin listening for the wake phrase. Off means off."""
        if not self.enabled:
            self._set(VoiceState.OFF, "Hands-free voice is switched off.")
            return self.describe()
        if not self.wake.detector.available:
            self._set(VoiceState.ERROR,
                      "No local wake-word engine is available, so SAM cannot listen hands-free. "
                      "The microphone button still works.",
                      last_error="wake engine unavailable")
            return self.describe()
        self.wake.start(self._on_wake)
        if self.wake.running:
            self._set(VoiceState.WAKE_LISTENING, f"Waiting for {self.wake.phrase}…")
        else:
            self._set(VoiceState.ERROR, "The microphone could not be opened.",
                      last_error=self.wake.error or "microphone unavailable")
        return self.describe()

    def stop(self) -> dict[str, Any]:
        self.wake.stop()
        self._set(VoiceState.OFF, "Hands-free voice is switched off.")
        return self.describe()

    # -- one session --------------------------------------------------------
    def _on_wake(self) -> None:
        if self._busy.is_set():
            # Left unclaimed, so the listener carries nothing forward for it.
            return
        detection = self.wake.take_detection()
        threading.Thread(target=self.run_session, kwargs={"detection": detection},
                         name="sam-voice-turn", daemon=True).start()

    def run_session(self, detection: WakeDetection | None = None) -> dict[str, Any]:
        """Wake acknowledged: capture, answer, speak, then offer a follow-up."""
        if self._busy.is_set():
            return self.describe()
        self._busy.set()
        try:
            self._set(VoiceState.WAKE_DETECTED, "Yes?")
            outcome = self._turn(detection=detection)
            deadline = time.monotonic() + self.continuation_seconds
            while outcome.get("spoke") and self.continuation_seconds > 0 and time.monotonic() < deadline:
                # A follow-up needs no wake phrase, but it does need speech:
                # silence simply lets the window close.
                self._set(VoiceState.LISTENING, "Listening for a follow-up…",
                          continuation_active=True)
                outcome = self._turn(follow_up=True)
                if not outcome.get("captured"):
                    break
                deadline = time.monotonic() + self.continuation_seconds
            self._set(VoiceState.WAKE_LISTENING, f"Waiting for {self.wake.phrase}…",
                      continuation_active=False)
            return self.describe()
        finally:
            self._busy.clear()

    def _turn(self, *, follow_up: bool = False, detection: WakeDetection | None = None) -> dict[str, Any]:
        heard = self._after_phrase(detection) if detection is not None else None
        if heard is None:
            if not follow_up and detection is None:
                self._set(VoiceState.LISTENING, "Listening…")
            heard = self._capture()
        if not heard.get("captured"):
            if not follow_up:
                self._set(VoiceState.WAKE_LISTENING, "I did not catch that.",
                          last_error=str(heard.get("reason") or ""))
            return {"captured": False, "spoke": False}

        text = str(heard.get("text") or "").strip()
        self.transcripts.append(text)
        if not text:
            # Audio arrived but came back as nothing. That is usually a reason,
            # not a mystery -- a command language local speech cannot handle,
            # or a provider that needs a credential -- and saying so beats
            # "I did not catch that" forever.
            why = str(heard.get("error") or heard.get("reason") or "").strip()
            self._set(VoiceState.WAKE_LISTENING, why[:160] or "I did not catch that.",
                      last_error=why[:200])
            return {"captured": False, "spoke": False}
        self._set(VoiceState.THINKING, "Thinking…", last_transcript=text)

        answer = self._ask(text)
        reply = str(answer.get("reply") or "").strip()
        if answer.get("error"):
            self._set(VoiceState.ERROR, str(answer["error"])[:160], last_error=str(answer["error"])[:200])
            return {"captured": True, "spoke": False}

        spoke = self._speak(reply)
        return {"captured": True, "spoke": True, "reply": reply, "spoken": spoke}

    def _after_phrase(self, detection: WakeDetection) -> dict[str, Any] | None:
        """The command spoken in the same breath as the phrase, if there was one.

        The wake listener has already captured it on the stream that heard the
        phrase; this reads it in the command language. None means "ask for the
        command the ordinary way": nothing followed the phrase, or what did
        could not be read -- in which case the reason is shown, and the user
        gets a second chance rather than a session that silently vanishes.
        """
        self._set(VoiceState.LISTENING, "Listening…")
        if not detection.wait(POST_WAKE_WAIT_SECONDS) or not detection.speech:
            return None
        audio = detection.audio
        # Transcribed once, then gone: the loop keeps text, never audio.
        detection.release()
        if audio is None:
            return None
        self._set(VoiceState.TRANSCRIBING, "Getting that down…")
        try:
            heard = self.voice.transcribe_captured(audio, language=self.language)
        except Exception as exc:  # noqa: BLE001 - a provider failure is a state
            heard = {"text": "", "error": f"{type(exc).__name__}: {exc}"}
        finally:
            del audio
        if str(heard.get("text") or "").strip():
            return {**heard, "captured": True}
        why = str(heard.get("error") or heard.get("reason") or "").strip()
        self._set(VoiceState.LISTENING,
                  f"{why[:120]} Please say it again." if why else "I did not catch that. Please say it again.",
                  last_error=why[:200])
        return None

    def _capture(self) -> dict[str, Any]:
        """One utterance, bounded, through the voice service that already exists.

        The wake listener stands down for the duration. Both of them want the
        same microphone, and leaving the listener running meant it transcribed
        the command as a wake candidate while `listen_once` got nothing at all.
        """
        self.wake.suppress()
        try:
            return self.voice.listen_once(
                max_seconds=MAX_COMMAND_SECONDS,
                device=getattr(self.settings, "voice_input_device", None),
                language=self.language,
                # The same bar the wake listener uses for this room, and a
                # syllable of voice, so a click after an answer is not sent to
                # be transcribed into a question nobody asked.
                threshold=self.wake.threshold,
                min_speech_frames=POST_WAKE_MIN_SPEECH_FRAMES,
            )
        except Exception as exc:  # noqa: BLE001 - a lost device is a state
            return {"captured": False, "reason": f"{type(exc).__name__}: {exc}"}
        finally:
            # No cooldown: the pause exists to miss SAM's own echo, and this
            # was the user talking.
            self.wake.resume(cooldown=0.0)

    def _ask(self, text: str) -> dict[str, Any]:
        """The ordinary agent path. A spoken request earns no extra authority."""
        if self._submit is not None:
            try:
                return self._submit(text)
            except Exception as exc:  # noqa: BLE001
                return {"error": f"{type(exc).__name__}: {exc}"}
        try:
            import asyncio

            async def run():
                return await self.agent.chat(text, conversation_id=None)

            try:
                asyncio.get_running_loop()
            except RuntimeError:
                result = asyncio.run(run())
            else:
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(lambda: asyncio.run(run())).result()
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}"}
        message = result.get("message") if isinstance(result, dict) else None
        reply = (message or {}).get("content", "") if isinstance(message, dict) else ""
        return {"reply": reply, "error": result.get("error") if isinstance(result, dict) else None,
                "approvals": (result or {}).get("approvals") if isinstance(result, dict) else None}

    def _speak(self, reply: str) -> bool:
        """Say the answer, with the wake listener deaf until well after.

        A failure to speak is reported and nothing else: the answer is already
        text on the screen, and losing a whole turn because an audio device is
        busy would be a worse outcome than a silent one.
        """
        if not reply:
            self._set(VoiceState.WAKE_LISTENING, "There was nothing to say.")
            return False
        if not self.auto_speak:
            self._set(VoiceState.WAKE_LISTENING, "Answered.", last_reply=reply)
            return True
        # No suppression here: `voice.speak` does it for every caller.
        self._set(VoiceState.SPEAKING, "Speaking…", last_reply=reply)
        spoken = True
        try:
            outcome = self.voice.speak(reply, language=self.language)
            if isinstance(outcome, dict) and outcome.get("ok") is False:
                spoken = False
                self._set(VoiceState.SPEAKING,
                          "The answer is on screen; SAM could not speak it.",
                          last_error=str(outcome.get("error") or "voice output unavailable"))
        except Exception as exc:  # noqa: BLE001
            spoken = False
            self._set(VoiceState.SPEAKING, "The answer is on screen; SAM could not speak it.",
                      last_error=f"{type(exc).__name__}: {exc}")
            # A voice service that threw may never have reported the end of
            # speech, so the listener would stay deaf for good.
            self.wake.resume()
        return spoken

    def _while_speaking(self, speaking: bool) -> None:
        if speaking:
            self.wake.suppress()
        else:
            self.wake.resume()

    # -- reporting ----------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        return {
            **self.status.as_dict(),
            "enabled": self.enabled,
            "auto_speak": self.auto_speak,
            "wake": self.wake.describe(),
            # Said plainly, because it is the whole privacy claim.
            "privacy": ("Wake detection runs locally on a bounded in-memory buffer. No audio is "
                        "written to disk, and nothing leaves this machine before the wake phrase."),
        }
