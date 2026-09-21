"""Local realtime voice: capture -> VAD -> streaming STT -> TTS, with barge-in.

Everything here runs on this machine. Speech recognition uses faster-whisper with
the Silero VAD that ships alongside it (ONNX, no torch), and speech output uses
Piper when a voice model is installed, falling back to the Windows SAPI voices.

Language honesty matters here: Whisper's model covers 100 languages and Kurdish
is not among them. `language_support()` reports that plainly instead of letting a
Sorani utterance be silently mistranscribed as Arabic or Persian.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from . import sorani as sorani_speech
from .contracts import CapabilityState

SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000


def resample_mono(samples: Any, from_rate: int, to_rate: int = SAMPLE_RATE) -> Any:
    """Linear resample of a mono float32 buffer.

    The browser typically captures at 48 kHz; KurdishTTS and Whisper both want 16 kHz.
    """
    import numpy

    audio = numpy.asarray(samples, dtype=numpy.float32).reshape(-1)
    if from_rate <= 0 or to_rate <= 0:
        raise ValueError("Sample rate must be positive.")
    if from_rate == to_rate or audio.size == 0:
        return audio
    duration = audio.size / float(from_rate)
    target_len = max(1, int(round(duration * to_rate)))
    source_x = numpy.linspace(0.0, 1.0, num=audio.size, endpoint=False)
    target_x = numpy.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return numpy.interp(target_x, source_x, audio).astype(numpy.float32)

# Kurdish has no Whisper model. These are the codes users most often ask for.
UNSUPPORTED_STT_LANGUAGES = {
    "ckb": "Central Kurdish (Sorani)",
    "ku": "Kurdish",
    "kmr": "Northern Kurdish (Kurmanji)",
}


def _spec(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


# --- Capability probing -------------------------------------------------------


def audio_devices() -> dict[str, Any]:
    """Enumerate real input and output devices for microphone/speaker selection."""
    if not _spec("sounddevice"):
        return {"state": CapabilityState.UNCONFIGURED.value, "reason": "sounddevice is not installed", "inputs": [], "outputs": []}
    try:
        import sounddevice

        devices = sounddevice.query_devices()
        default_input, default_output = sounddevice.default.device
    except Exception as exc:
        return {"state": CapabilityState.UNAVAILABLE.value, "reason": str(exc), "inputs": [], "outputs": []}
    inputs = [
        {"index": index, "name": device["name"], "channels": device["max_input_channels"],
         "sample_rate": int(device["default_samplerate"]), "default": index == default_input}
        for index, device in enumerate(devices) if device["max_input_channels"] > 0
    ]
    outputs = [
        {"index": index, "name": device["name"], "channels": device["max_output_channels"],
         "sample_rate": int(device["default_samplerate"]), "default": index == default_output}
        for index, device in enumerate(devices) if device["max_output_channels"] > 0
    ]
    return {
        "state": CapabilityState.AVAILABLE.value if inputs and outputs else CapabilityState.PARTIALLY_AVAILABLE.value,
        "inputs": inputs,
        "outputs": outputs,
        "default_input": default_input,
        "default_output": default_output,
    }


def language_support(language: str, sorani_status: dict[str, Any] | None = None) -> dict[str, Any]:
    """Whether speech recognition can actually handle this language.

    Whisper has no Kurdish model, but Sorani is served by KurdishTTS instead, so
    the answer for Sorani depends on whether that provider is actually reachable
    right now -- not on Whisper's language list.
    """
    code = (language or "").split("-")[0].strip().lower()
    if code in UNSUPPORTED_STT_LANGUAGES:
        connected = (sorani_status or {}).get("status") == "CONNECTED"
        if connected and sorani_speech.is_sorani(language):
            return {
                "language": language,
                "code": code,
                "stt_supported": True,
                "engine": (sorani_status or {}).get("provider", "kurdishtts"),
                "state": CapabilityState.AVAILABLE.value,
                "reason": None,
                "note": (
                    f"Whisper has no {UNSUPPORTED_STT_LANGUAGES[code]} model, so recognition for this "
                    "language runs through the configured Sorani provider rather than locally."
                ),
            }
        detail = (sorani_status or {}).get("detail")
        return {
            "language": language,
            "code": code,
            "stt_supported": False,
            "state": CapabilityState.UNAVAILABLE.value,
            "reason": (
                f"{UNSUPPORTED_STT_LANGUAGES[code]} is not one of Whisper's 100 supported languages, "
                "so local speech recognition cannot transcribe it. "
                + (
                    f"The Sorani provider is not usable either: {detail} "
                    if detail and sorani_speech.is_sorani(language)
                    else "Configure a KurdishTTS key to recognise Sorani speech. "
                    if sorani_speech.is_sorani(language)
                    else ""
                )
                # The limit must not be overstated: typing this language always works.
                + "Typed input in this language works normally."
            ),
        }
    try:
        from faster_whisper.tokenizer import _LANGUAGE_CODES

        supported = code in _LANGUAGE_CODES
    except Exception:
        return {"language": language, "code": code, "stt_supported": False,
                "state": CapabilityState.UNCONFIGURED.value, "reason": "faster-whisper is not installed"}
    return {
        "language": language,
        "code": code,
        "stt_supported": supported,
        "state": CapabilityState.AVAILABLE.value if supported else CapabilityState.UNAVAILABLE.value,
        "reason": None if supported else f"Whisper has no model for language code {code!r}.",
    }


# --- Voice activity detection -------------------------------------------------


class SileroVad:
    """Speech/silence segmentation using the Silero model bundled with faster-whisper."""

    def __init__(self, *, threshold: float = 0.5, min_silence_ms: int = 800, min_speech_ms: int = 250) -> None:
        self.threshold = threshold
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self._options: Any = None

    def available(self) -> bool:
        try:
            from faster_whisper.vad import VadOptions  # noqa: F401

            return True
        except Exception:
            return False

    def _opts(self) -> Any:
        if self._options is None:
            from faster_whisper.vad import VadOptions

            self._options = VadOptions(
                threshold=self.threshold,
                min_silence_duration_ms=self.min_silence_ms,
                min_speech_duration_ms=self.min_speech_ms,
            )
        return self._options

    def speech_segments(self, audio: Any) -> list[dict[str, int]]:
        """Speech spans in a float32 mono 16 kHz buffer, as sample offsets."""
        from faster_whisper.vad import get_speech_timestamps

        return get_speech_timestamps(audio, self._opts())

    def contains_speech(self, audio: Any) -> bool:
        try:
            return bool(self.speech_segments(audio))
        except Exception:
            return False

    def capability(self) -> dict[str, Any]:
        ok = self.available()
        return {
            "name": "vad",
            "engine": "silero (faster-whisper bundled, onnxruntime)",
            "state": CapabilityState.AVAILABLE.value if ok else CapabilityState.UNCONFIGURED.value,
            "threshold": self.threshold,
            "min_silence_ms": self.min_silence_ms,
        }


def rms_energy(audio: Any) -> float:
    """Cheap loudness gate used to skip silent buffers before touching the model."""
    import numpy

    array = numpy.asarray(audio, dtype=numpy.float32)
    if array.size == 0:
        return 0.0
    return float(numpy.sqrt(numpy.mean(numpy.square(array))))


# --- Speech to text -----------------------------------------------------------


class WhisperSTT:
    """faster-whisper transcription with a lazily loaded, cached model."""

    def __init__(self, model_size: str = "small", *, device: str = "auto", compute_type: str = "int8") -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model: Any = None
        self._lock = threading.Lock()
        self._load_error: str | None = None

    def available(self) -> bool:
        return _spec("faster_whisper")

    def load(self) -> Any:
        with self._lock:
            if self._model is not None:
                return self._model
            if self._load_error:
                raise RuntimeError(self._load_error)
            try:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            except Exception as exc:
                self._load_error = f"Could not load Whisper model {self.model_size!r}: {exc}"
                raise RuntimeError(self._load_error) from exc
            return self._model

    def transcribe(
        self,
        audio: Any,
        *,
        language: str | None = None,
        beam_size: int = 1,
        vad_filter: bool = True,
    ) -> dict[str, Any]:
        """Transcribe a float32 mono 16 kHz buffer."""
        started = time.perf_counter()
        code = (language or "").split("-")[0].strip().lower() or None
        if code and code in UNSUPPORTED_STT_LANGUAGES:
            return {
                "text": "",
                "language": code,
                "supported": False,
                "reason": language_support(code)["reason"],
                "duration_ms": 0.0,
                "segments": [],
            }
        model = self.load()
        segments, info = model.transcribe(
            audio, language=code, beam_size=beam_size, vad_filter=vad_filter, condition_on_previous_text=False
        )
        collected = [
            {"start": segment.start, "end": segment.end, "text": segment.text.strip()}
            for segment in segments
        ]
        return {
            "text": " ".join(item["text"] for item in collected if item["text"]).strip(),
            "language": info.language,
            "language_probability": round(float(info.language_probability), 4),
            "supported": True,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "segments": collected,
        }

    def capability(self) -> dict[str, Any]:
        if not self.available():
            return {"name": "stt", "engine": "faster-whisper", "state": CapabilityState.UNCONFIGURED.value,
                    "reason": "faster-whisper is not installed", "model": self.model_size}
        return {
            "name": "stt",
            "engine": "faster-whisper",
            "state": CapabilityState.AVAILABLE.value,
            "model": self.model_size,
            "loaded": self._model is not None,
            "device": self.device,
            "compute_type": self.compute_type,
        }


# --- Text to speech -----------------------------------------------------------


class TextToSpeech:
    """Piper when a voice model is installed; Windows SAPI otherwise."""

    def __init__(self, *, voice_path: str | None = None, rate: float = 1.0, volume: float = 1.0) -> None:
        self.voice_path = voice_path
        self.rate = rate
        self.volume = volume
        self._piper: Any = None
        self._lock = threading.Lock()

    def piper_voice_available(self) -> bool:
        return bool(self.voice_path and Path(self.voice_path).is_file() and _spec("piper"))

    def _load_piper(self) -> Any:
        with self._lock:
            if self._piper is None:
                from piper import PiperVoice

                self._piper = PiperVoice.load(self.voice_path)
            return self._piper

    def sapi_voices(self) -> list[dict[str, str]]:
        try:
            import win32com.client

            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            return [{"name": voice.GetDescription()} for voice in speaker.GetVoices()]
        except Exception:
            return []

    def synthesize_chunks(self, text: str) -> Iterator[bytes]:
        """Yield PCM chunks so playback can start before synthesis completes."""
        if not self.piper_voice_available():
            raise RuntimeError("No Piper voice model is configured; use speak() for the SAPI fallback.")
        voice = self._load_piper()
        for chunk in voice.synthesize(text):
            yield chunk.audio_int16_bytes if hasattr(chunk, "audio_int16_bytes") else bytes(chunk)

    def speak(self, text: str, *, cancel: Callable[[], bool] | None = None) -> dict[str, Any]:
        """Speak text, stopping early when `cancel` starts returning True."""
        started = time.perf_counter()
        if self.piper_voice_available():
            try:
                import numpy
                import sounddevice

                voice = self._load_piper()
                spoken = 0
                for chunk in voice.synthesize(text):
                    if cancel and cancel():
                        sounddevice.stop()
                        return {"engine": "piper", "interrupted": True, "chunks": spoken,
                                "duration_ms": round((time.perf_counter() - started) * 1000, 2)}
                    raw = chunk.audio_int16_bytes if hasattr(chunk, "audio_int16_bytes") else bytes(chunk)
                    samples = numpy.frombuffer(raw, dtype=numpy.int16)
                    rate = getattr(chunk, "sample_rate", None) or voice.config.sample_rate
                    sounddevice.play(samples, rate)
                    sounddevice.wait()
                    spoken += 1
                return {"engine": "piper", "interrupted": False, "chunks": spoken,
                        "duration_ms": round((time.perf_counter() - started) * 1000, 2)}
            except Exception as exc:
                return {"engine": "piper", "error": str(exc), "interrupted": False,
                        "duration_ms": round((time.perf_counter() - started) * 1000, 2)}
        return self._speak_sapi(text, cancel=cancel, started=started)

    def _speak_sapi(self, text: str, *, cancel: Callable[[], bool] | None, started: float) -> dict[str, Any]:
        try:
            import win32com.client

            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Rate = max(-10, min(10, int((self.rate - 1.0) * 10)))
            speaker.Volume = max(0, min(100, int(self.volume * 100)))
            # 1 = SVSFlagsAsync, so cancellation can interrupt mid-utterance.
            speaker.Speak(text, 1)
            while speaker.Status.RunningState != 1:
                if cancel and cancel():
                    speaker.Speak("", 2)  # 2 = SVSFPurgeBeforeSpeak
                    return {"engine": "sapi", "interrupted": True,
                            "duration_ms": round((time.perf_counter() - started) * 1000, 2)}
                time.sleep(0.05)
            return {"engine": "sapi", "interrupted": False,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2)}
        except Exception as exc:
            return {"engine": "sapi", "error": str(exc), "interrupted": False,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2)}

    def capability(self) -> dict[str, Any]:
        voices = self.sapi_voices()
        if self.piper_voice_available():
            return {"name": "tts", "engine": "piper", "state": CapabilityState.AVAILABLE.value,
                    "voice": self.voice_path, "streaming": True, "sapi_fallback_voices": len(voices)}
        if voices:
            return {
                "name": "tts", "engine": "windows-sapi", "state": CapabilityState.PARTIALLY_AVAILABLE.value,
                "voices": voices, "streaming": False,
                "reason": "No Piper voice model is configured, so SAM speaks through the installed Windows voices.",
            }
        return {"name": "tts", "engine": "none", "state": CapabilityState.UNCONFIGURED.value,
                "reason": "Neither a Piper voice model nor a Windows SAPI voice is available."}


# --- Realtime session ---------------------------------------------------------


@dataclass(slots=True)
class VoiceEvent:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=time.monotonic)


class BargeInController:
    """Tracks whether SAM is speaking and whether the user has interrupted."""

    STOP_WORDS = ("وەستە", "بوەستە", "stop", "cancel", "wait", "hold on")

    def __init__(self) -> None:
        self._speaking = threading.Event()
        self._interrupted = threading.Event()
        self._lock = threading.Lock()
        self.interruptions = 0

    def begin_speaking(self) -> None:
        with self._lock:
            self._interrupted.clear()
            self._speaking.set()

    def end_speaking(self) -> None:
        self._speaking.clear()

    @property
    def speaking(self) -> bool:
        return self._speaking.is_set()

    @property
    def interrupted(self) -> bool:
        return self._interrupted.is_set()

    def interrupt(self, reason: str = "user_speech") -> bool:
        """Signal an interruption. Returns True when SAM was actually speaking."""
        with self._lock:
            was_speaking = self._speaking.is_set()
            self._interrupted.set()
            self._speaking.clear()
            if was_speaking:
                self.interruptions += 1
            return was_speaking

    def reset(self) -> None:
        with self._lock:
            self._speaking.clear()
            self._interrupted.clear()

    @classmethod
    def is_stop_command(cls, text: str) -> bool:
        lowered = (text or "").strip().lower()
        return any(word in lowered for word in cls.STOP_WORDS)

    def should_stop_tts(self) -> bool:
        return self._interrupted.is_set()


class MicrophoneStream:
    """Continuous microphone capture into a bounded queue of float32 frames."""

    def __init__(self, *, device: int | None = None, sample_rate: int = SAMPLE_RATE, frame_samples: int = FRAME_SAMPLES) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self._queue: queue.Queue = queue.Queue(maxsize=200)
        self._stream: Any = None
        self.overflows = 0

    def __enter__(self) -> "MicrophoneStream":
        import sounddevice

        def callback(indata, frames, time_info, status):  # noqa: ANN001
            if status:
                self.overflows += 1
            try:
                self._queue.put_nowait(indata.copy().reshape(-1))
            except queue.Full:
                # Drop the oldest frame so live audio keeps flowing.
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(indata.copy().reshape(-1))
                except queue.Empty:
                    pass

        self._stream = sounddevice.InputStream(
            samplerate=self.sample_rate, blocksize=self.frame_samples, device=self.device,
            channels=1, dtype="float32", callback=callback,
        )
        self._stream.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def frames(self, timeout: float = 0.5) -> Iterator[Any]:
        while True:
            try:
                yield self._queue.get(timeout=timeout)
            except queue.Empty:
                return


class VoiceService:
    """Assembles device, VAD, STT, TTS, and barge-in into one capability surface."""

    MODES = ("PUSH_TO_TALK", "CONVERSATION", "ALWAYS_LISTENING", "WAKE_WORD")

    def __init__(self, settings: Any, secret_store: Any = None) -> None:
        self.settings = settings
        # The store is consulted on every use, so a key entered in the UI takes
        # effect without a restart.
        self.secret_store = secret_store
        self._sorani_stack: tuple[Any, Any] | None = None
        self._sorani_signature: tuple[Any, ...] | None = None
        self.vad = SileroVad(
            threshold=float(getattr(settings, "voice_vad_threshold", 0.5) or 0.5) if float(getattr(settings, "voice_vad_threshold", 0.5) or 0.5) > 0.1 else 0.5,
            min_silence_ms=int(getattr(settings, "voice_silence_ms", 800) or 800),
        )
        self.stt = WhisperSTT(str(getattr(settings, "local_stt_model", "small") or "small"))
        self.tts = TextToSpeech(voice_path=str(getattr(settings, "local_tts_voice", "") or "") or None)
        self.barge_in = BargeInController()

    @property
    def mode(self) -> str:
        value = str(getattr(self.settings, "voice_mode", "PUSH_TO_TALK") or "PUSH_TO_TALK").upper()
        return value if value in self.MODES else "PUSH_TO_TALK"

    @property
    def wake_word(self) -> str:
        return str(getattr(self.settings, "voice_wake_word", "SAM") or "SAM")

    def heard_wake_word(self, text: str) -> bool:
        heard = (text or "").strip().lower()
        if self.wake_word.strip().lower() in heard:
            return True
        # The user says the wake word in Sorani ("سام"), which never matches an
        # ASCII comparison, so the Sorani matcher gets a look too.
        from .sorani_intent import has_wake_word

        return has_wake_word(text or "", self.wake_word)

    # --- Sorani speech --------------------------------------------------------

    def _sorani_credentials(self) -> tuple[str | None, str | None, str | None]:
        """Resolve the Sorani keys. Values are used here and never returned."""
        import os

        def resolve(name: str) -> str | None:
            if self.secret_store is not None:
                from .secrets import resolve_credential

                value, _ = resolve_credential(name, self.secret_store)
                return value
            return os.getenv(name.upper())

        return (
            resolve("kurdishtts_stt_api_key"),
            resolve("kurdishtts_tts_api_key"),
            resolve("google_stt_credentials_path"),
        )

    def sorani_input_configured(self) -> bool:
        """Whether a Sorani recogniser key is present. Does not call the provider."""
        stt_key, _, google = self._sorani_credentials()
        return bool(stt_key or google)

    def sorani_stack(self) -> tuple[Any, Any]:
        """The (STT, TTS) routers, rebuilt when the configured keys change."""
        stt_key, tts_key, google = self._sorani_credentials()
        speaker = str(getattr(self.settings, "sorani_speaker_id", "") or "") or None
        # Compare by fingerprint so a key is never held in a comparable field.
        from .secrets import SecretStore

        signature = (SecretStore.fingerprint(stt_key), SecretStore.fingerprint(tts_key), google, speaker)
        if self._sorani_stack is None or signature != self._sorani_signature:
            self._sorani_stack = sorani_speech.build_routers(
                stt_key=stt_key, tts_key=tts_key,
                google_credentials_path=google, preferred_speaker=speaker,
            )
            self._sorani_signature = signature
        return self._sorani_stack

    def sorani_status(self) -> dict[str, Any]:
        """Live Sorani provider health. Never includes a credential."""
        stt_router, tts_router = self.sorani_stack()
        stt = stt_router.status()
        tts = tts_router.status()
        primary_stt = stt["primary"]
        speech_in = primary_stt["status"] == "CONNECTED" or stt["fallback"]["status"] == "CONNECTED"
        speech_out = tts["primary"]["status"] == "CONNECTED"
        if speech_in and speech_out:
            state = CapabilityState.AVAILABLE.value
        elif speech_in or speech_out:
            state = CapabilityState.PARTIALLY_AVAILABLE.value
        else:
            state = CapabilityState.UNCONFIGURED.value if primary_stt["status"] == "UNCONFIGURED" \
                else CapabilityState.UNAVAILABLE.value
        return {
            "state": state, "language": "ckb", "dialect": sorani_speech.SORANI_DIALECT,
            "stt": stt, "tts": tts,
            "speech_to_text": speech_in, "text_to_speech": speech_out,
            "round_trip": speech_in and speech_out,
        }

    def _sorani_stt_status(self) -> dict[str, Any]:
        """Just enough of the STT health for language reporting."""
        try:
            stt_router, _ = self.sorani_stack()
            primary = stt_router.status()["primary"]
        except Exception as exc:  # a broken provider must not break capabilities
            return {"status": "ERROR", "detail": str(exc)[:160]}
        return {**primary, "provider": primary.get("provider", "kurdishtts")}

    def capabilities(self) -> dict[str, Any]:
        language = str(getattr(self.settings, "voice_language", "en") or "en")
        sorani = self.sorani_status()
        support = language_support(language, sorani["stt"]["primary"] if sorani_speech.is_sorani(language) else None)
        devices = audio_devices()
        stt = self.stt.capability()
        tts = self.tts.capability()
        vad = self.vad.capability()
        speaking_sorani = sorani_speech.is_sorani(language)
        if speaking_sorani:
            # For Sorani the local Whisper/Piper engines are not the pipeline;
            # the Sorani providers are, so readiness is judged on those.
            pipeline_ready = (
                devices.get("state") == CapabilityState.AVAILABLE.value
                and vad["state"] == CapabilityState.AVAILABLE.value
                and sorani["round_trip"]
            )
        else:
            pipeline_ready = (
                devices.get("state") == CapabilityState.AVAILABLE.value
                and stt["state"] == CapabilityState.AVAILABLE.value
                and vad["state"] == CapabilityState.AVAILABLE.value
                and tts["state"] in {CapabilityState.AVAILABLE.value, CapabilityState.PARTIALLY_AVAILABLE.value}
            )
        state = CapabilityState.AVAILABLE.value if pipeline_ready else CapabilityState.PARTIALLY_AVAILABLE.value
        if not support["stt_supported"]:
            # The pipeline runs, but not for the configured language.
            state = CapabilityState.PARTIALLY_AVAILABLE.value
        return {
            "state": state,
            "mode": self.mode,
            "modes": list(self.MODES),
            "wake_word": self.wake_word,
            "language": support,
            "devices": devices,
            "vad": vad,
            "stt": stt,
            "tts": tts,
            "barge_in": {
                "state": CapabilityState.AVAILABLE.value,
                "stop_words": list(BargeInController.STOP_WORDS),
                "interruptions": self.barge_in.interruptions,
            },
            "sorani": sorani,
            "sample_rate": SAMPLE_RATE,
            "frame_ms": FRAME_MS,
        }

    def listen_once(
        self,
        *,
        max_seconds: float = 12.0,
        silence_ms: int | None = None,
        device: int | None = None,
        language: str | None = None,
        on_event: Callable[[VoiceEvent], None] | None = None,
    ) -> dict[str, Any]:
        """Capture one utterance: wait for speech, stop on silence, transcribe."""
        import numpy

        silence = (silence_ms if silence_ms is not None else int(getattr(self.settings, "voice_silence_ms", 800) or 800)) / 1000
        emit = on_event or (lambda event: None)
        collected: list[Any] = []
        speech_started = False
        last_voice = time.monotonic()
        deadline = time.monotonic() + max_seconds

        emit(VoiceEvent("listening"))
        with MicrophoneStream(device=device) as microphone:
            for frame in microphone.frames(timeout=0.4):
                now = time.monotonic()
                if now > deadline:
                    break
                energy = rms_energy(frame)
                voiced = energy > 0.006
                if voiced:
                    if not speech_started:
                        speech_started = True
                        emit(VoiceEvent("speech_start", {"energy": energy}))
                    last_voice = now
                if speech_started:
                    collected.append(frame)
                    if not voiced and now - last_voice >= silence:
                        emit(VoiceEvent("speech_end", {"seconds": len(collected) * FRAME_MS / 1000}))
                        break

        if not collected:
            return {"text": "", "captured": False, "reason": "No speech was detected before the timeout."}
        audio = numpy.concatenate(collected).astype("float32")
        seconds = round(len(audio) / SAMPLE_RATE, 2)
        spoken_language = language or getattr(self.settings, "voice_language", None)
        emit(VoiceEvent("transcribing", {"seconds": seconds}))
        if sorani_speech.is_sorani(spoken_language):
            result = self._transcribe_sorani(audio, seconds)
        else:
            result = self.stt.transcribe(audio, language=spoken_language)
        emit(VoiceEvent("transcript", {"text": result.get("text", "")}))
        return {**result, "captured": True, "seconds": seconds}

    def transcribe_audio_bytes(self, payload: bytes, language: str | None = None) -> dict[str, Any]:
        """Transcribe a WAV blob recorded in the browser.

        The UI microphone cannot use Chrome/Edge Web Speech for Sorani, so the
        page records PCM, wraps it as WAV, and sends it here. Whisper is still
        never consulted for Kurdish.
        """
        if not payload:
            return {"text": "", "captured": False, "reason": sorani_speech.MESSAGE_EMPTY_AUDIO}
        try:
            samples, rate = sorani_speech.wav_to_float32(payload)
        except Exception as exc:
            raise ValueError(sorani_speech.MESSAGE_EMPTY_AUDIO) from exc
        audio = resample_mono(samples, rate, SAMPLE_RATE)
        seconds = round(len(audio) / SAMPLE_RATE, 2)
        spoken_language = language or getattr(self.settings, "voice_language", None)
        if sorani_speech.is_sorani(spoken_language):
            result = self._transcribe_sorani(audio, seconds)
        else:
            result = self.stt.transcribe(audio, language=spoken_language)
        return {**result, "captured": True, "seconds": seconds}

    def _transcribe_sorani(self, audio: Any, seconds: float) -> dict[str, Any]:
        """Recognise Sorani speech, never by pretending it is another language.

        Whisper is not consulted here even as a last resort: asking it for
        Kurdish returns Arabic or Persian text that reads as a real transcript
        while meaning something the user did not say.
        """
        stt_router, _ = self.sorani_stack()
        try:
            result = stt_router.transcribe(audio, SAMPLE_RATE)
        except PermissionError:
            return {"text": "", "language": "ckb", "engine": "sorani",
                    "error": sorani_speech.MESSAGE_NO_PROVIDER, "seconds": seconds}
        except Exception as exc:
            return {"text": "", "language": "ckb", "engine": "sorani",
                    "error": str(exc)[:200], "seconds": seconds}
        text = (result.get("text") or "").strip()
        if not text and seconds < sorani_speech.MIN_RELIABLE_SECONDS:
            # Silence and "too short to hear" are different problems, and the
            # user can act on the second one.
            result = {**result, "reason": sorani_speech.MESSAGE_TOO_SHORT, "too_short": True}
        return {**result, "language": "ckb", "engine": f"sorani:{result.get('route', 'kurdishtts')}"}

    def speak(self, text: str, language: str | None = None) -> dict[str, Any]:
        """Speak, aborting as soon as barge-in fires.

        Sorani is spoken by the Sorani provider or not at all. There is no
        fallback to an English Windows voice: reading Kurdish text through an
        English voice produces sounds that are not the language, and passing it
        off as Sorani would be a lie about what SAM can do.
        """
        wanted = language or getattr(self.settings, "voice_language", None)
        use_sorani = sorani_speech.is_sorani(wanted) or sorani_speech.looks_sorani(text)
        self.barge_in.begin_speaking()
        try:
            if use_sorani:
                return self._speak_sorani(text)
            return self.tts.speak(text, cancel=self.barge_in.should_stop_tts)
        finally:
            self.barge_in.end_speaking()

    def synthesize(self, text: str, language: str | None = None) -> dict[str, Any]:
        """Produce audio bytes without playing them, so the browser can.

        Playing on the server speakers while the UI also plays the clip would
        double the reply. Sorani still has no English-voice fallback.
        """
        wanted = language or getattr(self.settings, "voice_language", None)
        use_sorani = sorani_speech.is_sorani(wanted) or sorani_speech.looks_sorani(text)
        if use_sorani:
            return self._synthesize_sorani(text)
        return {
            "audio": None, "engine": "browser", "language": wanted or "en",
            "error": "Non-Sorani speech uses the browser voice.",
        }

    def _synthesize_sorani(self, text: str) -> dict[str, Any]:
        started = time.perf_counter()
        _, tts_router = self.sorani_stack()
        try:
            audio, metadata = tts_router.synthesize(text)
        except PermissionError:
            return {"audio": None, "engine": "sorani", "language": "ckb",
                    "error": sorani_speech.MESSAGE_NO_PROVIDER,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2)}
        except Exception as exc:
            return {"audio": None, "engine": "sorani", "language": "ckb",
                    "error": str(exc)[:200],
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2)}
        return {
            "audio": audio,
            "engine": f"sorani:{metadata.get('route', 'kurdishtts')}",
            "language": "ckb",
            "speaker_id": metadata.get("speaker_id"),
            "bytes": len(audio),
            "content_type": "audio/wav",
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def _speak_sorani(self, text: str) -> dict[str, Any]:
        result = dict(self._synthesize_sorani(text))
        audio = result.pop("audio", None)
        if not audio:
            return {**result, "spoken": False}
        played = self._play_wav(audio, cancel=self.barge_in.should_stop_tts)
        return {**result, "spoken": played["played"], "interrupted": played["interrupted"]}

    def _play_wav(self, payload: bytes, *, cancel: Callable[[], bool] | None = None) -> dict[str, Any]:
        """Play WAV audio in slices so barge-in can cut it off mid-sentence."""
        try:
            import sounddevice

            samples, rate = sorani_speech.wav_to_float32(payload)
        except Exception as exc:
            return {"played": False, "interrupted": False, "error": str(exc)[:160]}
        # A fifth of a second is short enough to feel immediate when interrupted
        # and long enough not to click between slices.
        slice_samples = max(1, int(rate * 0.2))
        try:
            for offset in range(0, len(samples), slice_samples):
                if cancel and cancel():
                    sounddevice.stop()
                    return {"played": True, "interrupted": True}
                sounddevice.play(samples[offset:offset + slice_samples], rate)
                sounddevice.wait()
        except Exception as exc:
            return {"played": False, "interrupted": False, "error": str(exc)[:160]}
        return {"played": True, "interrupted": False}
