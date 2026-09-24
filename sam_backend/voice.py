"""Local realtime voice: capture -> VAD -> streaming STT -> TTS, with barge-in.

Everything here runs on this machine. Speech recognition uses faster-whisper with
the Silero VAD that ships alongside it (ONNX, no torch), and speech output uses
Piper when a voice model is installed, falling back to the Windows SAPI voices.

Language honesty matters here: Whisper's model covers 100 languages and Kurdish
is not among them. `language_support()` reports that plainly instead of letting a
Sorani utterance be silently mistranscribed as Arabic or Persian.
"""

from __future__ import annotations

import collections
import concurrent.futures
import itertools
import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from . import sorani as sorani_speech
from .contracts import CapabilityState

logger = logging.getLogger(__name__)

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


# --- Speaker output -----------------------------------------------------------

# Measured on this machine's default output (MME, a USB headset; 22.05 kHz
# mono like KurdishTTS; digital silence only): playing a reply as 200 ms slices
# with sounddevice.play()+wait() per slice cost 466 ms of wall time per slice --
# 162 ms to open and start a stream, 309 ms to wait it out -- so 5.0 s of
# speech took 11.8 s: a fifth of a second of voice, then about a quarter of a
# second of nothing, over and over. Each slice also ended in CallbackAbort,
# which discards queued buffers, and PortAudio's clock had 168 ms of the slice
# still queued at that moment. One stream written in 100 ms blocks played the
# same 5.0 s in 5.4 s -- one 160 ms open, one 200 ms drain, no underflows.
PLAYBACK_BLOCK_SECONDS = 0.1
# 'high' measured 183 ms of buffering here against 91 ms for 'low'. The margin
# is for a busy machine, and it costs barge-in nothing: abort() drops whatever
# is queued (11 ms, measured), so only the block being written delays a stop --
# 86 ms from the cancel to noticing it with 100 ms blocks, against 178 ms with
# 200 ms blocks and a slice every 466 ms before.
PLAYBACK_LATENCY = "high"


class SpeakerQueue:
    """Whose turn it is on the speakers: one voice at a time, first come first served.

    sounddevice.play() keeps one module-global stream and stops whatever it
    was playing when it is called again, so two replies spoken at once -- a
    hands-free turn and /api/voice/speak, say -- cut each other off every
    slice and came out interleaved. A later utterance waits for the earlier
    one rather than cutting it off: every caller gets its whole sentence, in
    the order asked, and cutting speech short stays barge-in's job alone.
    A waiting utterance still honours its own cancel, so a barge-in empties
    the queue instead of letting the next reply start where the last stopped.
    """

    def __init__(self) -> None:
        self._changed = threading.Condition()
        self._line: collections.deque[object] = collections.deque()

    def enter(self, cancel: Callable[[], bool] | None = None) -> object | None:
        """Wait for a turn. None when `cancel` fired while waiting."""
        ticket = object()
        with self._changed:
            self._line.append(ticket)
            try:
                while self._line[0] is not ticket:
                    if cancel is not None and cancel():
                        self._line.remove(ticket)
                        self._changed.notify_all()
                        return None
                    self._changed.wait(0.05)
            except BaseException:
                if ticket in self._line:
                    self._line.remove(ticket)
                self._changed.notify_all()
                raise
        return ticket

    def leave(self, ticket: object) -> None:
        with self._changed:
            if ticket in self._line:
                self._line.remove(ticket)
            self._changed.notify_all()

    @property
    def busy(self) -> bool:
        with self._changed:
            return bool(self._line)


# Process-wide, because the output device is: every VoiceService and every
# engine (Sorani, Piper, SAPI) takes its turn here.
SPEAKERS = SpeakerQueue()


def _output_stream(**kwargs: Any) -> Any:
    """Open a PortAudio output stream. Its own function so tests can play in silence."""
    import sounddevice

    return sounddevice.OutputStream(**kwargs)


def _output_defaults(device: int | None) -> tuple[int, int]:
    """The output device's own sample rate and channel count."""
    import sounddevice

    info = sounddevice.query_devices(device, "output")
    return int(info["default_samplerate"]), int(info["max_output_channels"])


def _open_output(rate: int, channels: int, device: int | None) -> tuple[Any, int, int]:
    """A stream at the audio's own rate and channels, or at the device's if it refuses those.

    MME resamples anything it is given; a stricter host API can reject 22.05 kHz
    or a channel count, and then the audio is converted rather than not played.
    """
    try:
        stream = _output_stream(samplerate=rate, channels=channels, dtype="float32",
                                device=device, latency=PLAYBACK_LATENCY)
        return stream, rate, channels
    except Exception as refused:  # noqa: BLE001 - retried once at the device's format
        try:
            device_rate, device_channels = _output_defaults(device)
        except Exception:  # noqa: BLE001
            raise refused from None
        fitted = max(1, min(channels, device_channels))
        if (device_rate, fitted) == (rate, channels):
            raise
        try:
            stream = _output_stream(samplerate=device_rate, channels=fitted, dtype="float32",
                                    device=device, latency=PLAYBACK_LATENCY)
        except Exception:  # noqa: BLE001 - the first refusal is the one that explains it
            raise refused from None
        return stream, device_rate, fitted


def _as_frames(chunk: Any) -> Any:
    import numpy

    array = numpy.asarray(chunk, dtype=numpy.float32)
    return array.reshape(-1, 1) if array.ndim == 1 else array


def _fit_audio(frames: Any, from_rate: int, to_rate: int, to_channels: int) -> Any:
    """Convert (samples, channels) audio to the stream's rate and channel count."""
    import numpy

    if frames.shape[1] != to_channels:
        mono = frames.mean(axis=1, keepdims=True)
        frames = mono if to_channels == 1 else numpy.repeat(mono, to_channels, axis=1)
    if from_rate != to_rate:
        frames = numpy.stack(
            [resample_mono(frames[:, channel], from_rate, to_rate) for channel in range(frames.shape[1])],
            axis=1,
        )
    return numpy.ascontiguousarray(frames, dtype=numpy.float32)


def _next_audio(source: Iterator[Any]) -> Any | None:
    """The next non-empty chunk as (samples, channels) frames, or None at the end."""
    for chunk in source:
        frames = _as_frames(chunk)
        if frames.shape[0]:
            return frames
    return None


def play_audio(
    chunks: Iterable[Any],
    rate: int,
    *,
    cancel: Callable[[], bool] | None = None,
    device: int | None = None,
    block_seconds: float = PLAYBACK_BLOCK_SECONDS,
) -> dict[str, Any]:
    """Play float32 audio through one output stream, stopping the moment `cancel` says so.

    `chunks` are arrays shaped (samples,) or (samples, channels), all at
    `rate`. An iterator is consumed as it plays, so a streaming synthesiser
    speaks its first sentence while it works on the next. The stream is
    opened once, written in `block_seconds` blocks with `cancel` checked
    between them, drained at the end, aborted on barge-in, and closed on
    every path. The result reports what happened: `played` is True only once
    audio was handed to the device, and a failure carries its `error`.
    """
    started = time.perf_counter()
    report: dict[str, Any] = {"played": False, "interrupted": False, "frames": 0, "underflows": 0}
    out_rate = rate

    def done(**extra: Any) -> dict[str, Any]:
        seconds = report["frames"] / out_rate if out_rate else 0.0
        return {**report, **extra, "seconds": round(seconds, 3),
                "duration_ms": round((time.perf_counter() - started) * 1000, 2)}

    source = iter(chunks)
    try:
        first = _next_audio(source)
    except Exception as exc:  # noqa: BLE001 - a synthesiser failing is a result
        return done(error=str(exc)[:160] or type(exc).__name__)
    if first is None:
        return done(error="There was no audio to play.")
    if rate <= 0:
        return done(error=f"Invalid sample rate: {rate}")
    if cancel is not None and cancel():
        return done(interrupted=True)
    ticket = SPEAKERS.enter(cancel)
    if ticket is None:
        return done(interrupted=True)
    stream = None
    try:
        if cancel is not None and cancel():
            # The user said stop while this reply waited behind another one:
            # the turn came, but the device is not even opened for it.
            report["interrupted"] = True
            return done()
        stream, out_rate, out_channels = _open_output(rate, int(first.shape[1]), device)
        stream.start()
        block = max(1, int(out_rate * block_seconds))
        chunk = first
        while chunk is not None:
            frames = _fit_audio(chunk, rate, out_rate, out_channels)
            for offset in range(0, frames.shape[0], block):
                if cancel is not None and cancel():
                    stream.abort()
                    report["interrupted"] = True
                    return done()
                piece = frames[offset:offset + block]
                if stream.write(piece):
                    report["underflows"] += 1
                report["frames"] += piece.shape[0]
                report["played"] = True
            chunk = _next_audio(source)
        # stop() drains what is queued; abort() would cut the last words off.
        stream.stop()
        return done()
    except Exception as exc:  # noqa: BLE001 - a lost device is a result, not a crash
        return done(error=str(exc)[:160] or type(exc).__name__)
    finally:
        if stream is not None:
            try:
                # Closing a stream that is still running discards its buffers.
                stream.close()
            except Exception:  # noqa: BLE001
                pass
        SPEAKERS.leave(ticket)


# --- What gets read aloud -----------------------------------------------------

# KurdishTTS's free plan refuses a request over 500 characters, and the reply
# that went silent on 2026-09-24 was 552: one request for the whole reply, a
# fast rejection, nothing played. The browser already reads replies in
# sentence chunks of at most 180 characters (frontend/app.js
# takeSpeakableChunks); the server path now does the same, so no reply is too
# long and the first sentence is heard while the next is being synthesised.
SPEECH_CHUNK_CHARS = 180
_CODE_BLOCK = re.compile(r"```[\s\S]*?```")
_LIST_MARKER = re.compile(r"(?m)^\s*(?:[-*+•]|\d+[.)])\s+")
_MARKUP = re.compile(r"[*_`#>\[\]()|]")
_SENTENCE = re.compile(r"[^.!?؟…\n]+(?:[.!?؟…]+[\"»”’']*|\n|$)")


def speakable_text(text: str) -> str:
    """The reply as it should sound: Markdown gone, lines joined.

    Models answer in Markdown even when the reply is spoken, and the voice
    read "**", list dashes and backticks aloud. Code is dropped rather than
    read -- nobody wants a function spoken character by character. List items
    become their own sentences, so a bullet list is read as a list.
    """
    plain = _CODE_BLOCK.sub(" ", text or "")
    plain = _LIST_MARKER.sub("\n", plain)
    plain = _MARKUP.sub(" ", plain)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in plain.splitlines()]
    return "\n".join(line for line in lines if line)


def speakable_chunks(text: str, limit: int = SPEECH_CHUNK_CHARS) -> list[str]:
    """Sentences packed into pieces of at most `limit` characters.

    Short sentences share a piece, so a reply costs as few provider requests as
    its length allows. A sentence longer than the limit is cut at the last
    space before it, as the browser does.
    """
    chunks: list[str] = []
    current = ""
    for match in _SENTENCE.finditer(speakable_text(text)):
        sentence = match.group(0).strip()
        while len(sentence) > limit:
            cut = sentence.rfind(" ", 0, limit)
            cut = cut if cut >= limit // 4 else limit
            head, sentence = sentence[:cut].strip(), sentence[cut:].strip()
            if current:
                chunks.append(current)
                current = ""
            chunks.append(head)
        if not sentence:
            continue
        joined = f"{current} {sentence}" if current else sentence
        if len(joined) <= limit:
            current = joined
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


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
        """Speak text, stopping early when `cancel` starts returning True.

        `spoken` says whether any of it reached the speakers and `error` why
        not; VoiceService.speak turns the two into `ok`.
        """
        started = time.perf_counter()
        if self.piper_voice_available():
            return self._speak_piper(text, cancel=cancel, started=started)
        return self._speak_sapi(text, cancel=cancel, started=started)

    def _speak_piper(self, text: str, *, cancel: Callable[[], bool] | None, started: float) -> dict[str, Any]:
        """Piper's sentences through one continuous stream, not a stream per sentence.

        Each sentence used to be its own sounddevice.play()+wait(): a new
        PortAudio stream per chunk (162 ms to open on this machine's output),
        and a play() that stopped whatever anything else was playing.
        """
        def elapsed() -> float:
            return round((time.perf_counter() - started) * 1000, 2)

        try:
            import numpy

            voice = self._load_piper()
            produced = iter(voice.synthesize(text))
            first = next(produced, None)
        except Exception as exc:  # noqa: BLE001 - reported as the result
            return {"engine": "piper", "error": str(exc) or type(exc).__name__, "interrupted": False,
                    "spoken": False, "chunks": 0, "duration_ms": elapsed()}
        if first is None:
            return {"engine": "piper", "error": "Piper produced no audio.", "interrupted": False,
                    "spoken": False, "chunks": 0, "duration_ms": elapsed()}
        rate = int(getattr(first, "sample_rate", None) or voice.config.sample_rate)
        counted = {"chunks": 0}

        def pcm() -> Iterator[Any]:
            for chunk in itertools.chain([first], produced):
                raw = chunk.audio_int16_bytes if hasattr(chunk, "audio_int16_bytes") else bytes(chunk)
                counted["chunks"] += 1
                yield numpy.frombuffer(raw, dtype=numpy.int16).astype(numpy.float32) / 32768.0

        played = play_audio(pcm(), rate, cancel=cancel)
        result = {"engine": "piper", "interrupted": played["interrupted"], "spoken": played["played"],
                  "chunks": counted["chunks"], "seconds": played.get("seconds"),
                  "underflows": played.get("underflows", 0), "duration_ms": elapsed()}
        if played.get("error"):
            result["error"] = played["error"]
        return result

    def _sapi_speaker(self) -> Any:
        """The Windows voice. Its own method so tests can speak without a sound."""
        import win32com.client

        return win32com.client.Dispatch("SAPI.SpVoice")

    def _speak_sapi(self, text: str, *, cancel: Callable[[], bool] | None, started: float) -> dict[str, Any]:
        def elapsed() -> float:
            return round((time.perf_counter() - started) * 1000, 2)

        # SAPI plays through its own audio object, but it is the same pair of
        # speakers, so it waits its turn like everything else.
        ticket = SPEAKERS.enter(cancel)
        if ticket is None:
            return {"engine": "sapi", "interrupted": True, "spoken": False, "duration_ms": elapsed()}
        try:
            speaker = self._sapi_speaker()
            speaker.Rate = max(-10, min(10, int((self.rate - 1.0) * 10)))
            speaker.Volume = max(0, min(100, int(self.volume * 100)))
            # 1 = SVSFlagsAsync, so cancellation can interrupt mid-utterance.
            speaker.Speak(text, 1)
            while speaker.Status.RunningState != 1:
                if cancel and cancel():
                    speaker.Speak("", 2)  # 2 = SVSFPurgeBeforeSpeak
                    return {"engine": "sapi", "interrupted": True, "spoken": True, "duration_ms": elapsed()}
                time.sleep(0.05)
            return {"engine": "sapi", "interrupted": False, "spoken": True, "duration_ms": elapsed()}
        except Exception as exc:  # noqa: BLE001 - reported as the result
            return {"engine": "sapi", "error": str(exc) or type(exc).__name__, "interrupted": False,
                    "spoken": False, "duration_ms": elapsed()}
        finally:
            SPEAKERS.leave(ticket)

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

    def pending(self) -> int:
        """Frames captured but not yet read: zero means the reader has caught up."""
        return self._queue.qsize()

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
        # Called with True when SAM starts speaking and False when it stops.
        # The hands-free wake listener uses it to go deaf while SAM talks.
        self.on_speaking: Callable[[bool], None] | None = None
        # How many speak() calls are under way. Two can overlap -- the second
        # waits for the speakers -- and the listener must stay deaf until the
        # last of them has finished, not wake when the first one does.
        self._speech_calls = 0
        self._speech_span = threading.Lock()

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

    def sorani_output_configured(self) -> bool:
        """Whether a Sorani speech key is present. Does not call the provider.

        The TTS health check lists speakers over the network, far too slow to
        ask before every chat turn; the chat model's capability brief only
        needs to know whether the key is there.
        """
        _, tts_key, _ = self._sorani_credentials()
        return bool(tts_key)

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
        threshold: float | None = None,
        min_speech_frames: int = 1,
    ) -> dict[str, Any]:
        """Capture one utterance: wait for speech, stop on silence, transcribe.

        `threshold` and `min_speech_frames` exist for hands-free. With nobody
        at the keyboard to press a button, a fixed gate and a single loud frame
        were enough to "capture" a fan surge or a click -- four times out of
        four in a silent room, 0.8 to 9.3 seconds of nothing -- which the Sorani
        provider, having no silence filter, then transcribed into words nobody
        said. The defaults leave push-to-talk exactly as it was.
        """
        import numpy

        silence = (silence_ms if silence_ms is not None else int(getattr(self.settings, "voice_silence_ms", 800) or 800)) / 1000
        gate = 0.006 if threshold is None else float(threshold)
        emit = on_event or (lambda event: None)
        collected: list[Any] = []
        speech_started = False
        voiced_frames = 0
        last_voice = time.monotonic()
        deadline = time.monotonic() + max_seconds

        emit(VoiceEvent("listening"))
        with MicrophoneStream(device=device) as microphone:
            for frame in microphone.frames(timeout=0.4):
                now = time.monotonic()
                if now > deadline:
                    break
                energy = rms_energy(frame)
                voiced = energy > gate
                if voiced:
                    if not speech_started:
                        speech_started = True
                        emit(VoiceEvent("speech_start", {"energy": energy}))
                    voiced_frames += 1
                    last_voice = now
                if speech_started:
                    collected.append(frame)
                    if not voiced and now - last_voice >= silence:
                        if voiced_frames < min_speech_frames:
                            # A blip, not an utterance: forget it and keep waiting.
                            collected.clear()
                            speech_started = False
                            voiced_frames = 0
                            continue
                        emit(VoiceEvent("speech_end", {"seconds": len(collected) * FRAME_MS / 1000}))
                        break
        if collected and voiced_frames < min_speech_frames:
            collected.clear()

        if not collected:
            return {"text": "", "captured": False, "reason": "No speech was detected before the timeout."}
        audio = numpy.concatenate(collected).astype("float32")
        return self.transcribe_captured(audio, language=language, on_event=on_event)

    def transcribe_captured(
        self,
        audio: Any,
        *,
        language: str | None = None,
        on_event: Callable[[VoiceEvent], None] | None = None,
    ) -> dict[str, Any]:
        """Transcribe speech that has already been captured, in the command language.

        `listen_once` ends here, and so does a hands-free command spoken in
        the same breath as the wake phrase: that audio was captured by the
        wake listener, and must be read by the recogniser the user configured
        -- the Sorani provider for Sorani -- not by the English wake pass.
        """
        emit = on_event or (lambda event: None)
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

        Speech never overlaps: a second call synthesises alongside the first
        and then waits for the speakers (SpeakerQueue). The result always
        carries `ok` -- False, with an `error`, when nothing was spoken
        because synthesis or playback failed.
        """
        if not (text or "").strip():
            return {"ok": False, "spoken": False, "interrupted": False, "error": "There was nothing to say."}
        wanted = language or getattr(self.settings, "voice_language", None)
        use_sorani = sorani_speech.is_sorani(wanted) or sorani_speech.looks_sorani(text)
        self._speech_begins()
        try:
            try:
                if use_sorani:
                    result = self._speak_sorani(text)
                else:
                    result = self.tts.speak(text, cancel=self.barge_in.should_stop_tts)
            except Exception as exc:  # noqa: BLE001 - reported in the outcome, like every other failure
                result = {"engine": "sorani" if use_sorani else "local",
                          "error": f"{type(exc).__name__}: {exc}"[:200]}
        finally:
            self._speech_ends()
        return self._speech_outcome(result)

    @staticmethod
    def _speech_outcome(result: dict[str, Any]) -> dict[str, Any]:
        """`ok` says whether the words reached the speakers, for every engine.

        Callers used to be left to guess. The Sorani path answered `spoken`
        and dropped playback's error, the local engines answered neither, and
        the hands-free loop -- which looks for `ok` being False -- took a reply
        that failed to synthesise or play as spoken. Being interrupted is not
        a failure: somebody asked for it, and `interrupted` says so.
        """
        outcome = dict(result)
        spoken = bool(outcome.get("spoken"))
        interrupted = bool(outcome.get("interrupted"))
        if not outcome.get("error") and not spoken and not interrupted:
            outcome["error"] = "Nothing was spoken."
        outcome.update(ok=not outcome.get("error"), spoken=spoken, interrupted=interrupted)
        return outcome

    def _speech_begins(self) -> None:
        """The first overlapping speak() deafens the listener; the rest join it.

        The lock is held through the hook on purpose: the wake listener's
        suppression is a flag, not a count, and its resume sleeps out the echo
        before clearing it. A second reply starting inside that sleep would
        suppress first and then be un-suppressed by the first reply's resume,
        leaving the listener awake while SAM talks.
        """
        with self._speech_span:
            self._speech_calls += 1
            if self._speech_calls == 1:
                self.barge_in.begin_speaking()
                self._notify_speaking(True)

    def _speech_ends(self) -> None:
        with self._speech_span:
            self._speech_calls -= 1
            if self._speech_calls == 0:
                self.barge_in.end_speaking()
                self._notify_speaking(False)

    def _notify_speaking(self, speaking: bool) -> None:
        """Tell whoever is listening that SAM's own voice is on the speakers.

        This lives here rather than in the hands-free loop because the loop is
        not the only caller: the manual voice endpoint speaks too, and if the
        wake listener were only deafened by the loop, SAM would hear itself say
        the phrase through any other path and answer itself.

        The end of speech is not instant -- resuming waits out the echo -- so
        this call can block for about a second on the way back down.
        """
        hook = self.on_speaking
        if hook is None:
            return
        try:
            hook(speaking)
        except Exception:  # noqa: BLE001 - a listener must not break speech
            pass

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
        """Read a Sorani reply sentence by sentence through one continuous stream.

        The whole reply used to go to KurdishTTS in one request, raw Markdown
        and all: over the free plan's 500-character cap it was refused and
        nothing was said, and under it nothing was heard until every sentence
        had been synthesised. Now each piece (speakable_chunks) is synthesised
        while the one before it plays, so the first words come out after one
        short request and there is no gap between pieces.
        """
        chunks = speakable_chunks(text)
        if not chunks:
            return {"engine": "sorani", "language": "ckb", "spoken": False, "interrupted": False,
                    "error": "There was nothing to say."}
        cancel = self.barge_in.should_stop_tts
        first = dict(self._synthesize_sorani(chunks[0]))
        audio = first.pop("audio", None)
        if not audio:
            error = first.get("error") or "KurdishTTS returned no audio."
            logger.warning("Sorani speech failed before anything was said: %s", error)
            return {**first, "spoken": False, "interrupted": False, "error": error,
                    "chunks": 0, "chunks_total": len(chunks)}
        try:
            opening, rate = sorani_speech.wav_to_frames(audio)
        except Exception as exc:  # noqa: BLE001
            return {**first, "spoken": False, "interrupted": False,
                    "error": f"Unreadable speech audio: {exc}"[:160]}

        def fitted(frames: Any, frames_rate: int) -> Any:
            if frames.shape[1] > 2:
                # Speech has nothing to say to a surround layout; fold it to mono.
                frames = frames.mean(axis=1, keepdims=True)
            # play_audio takes one rate for the whole reply.
            return frames if frames_rate == rate else _fit_audio(frames, frames_rate, rate, frames.shape[1])

        said = {"chunks": 1}
        failure: dict[str, Any] = {}

        def pieces() -> Iterator[Any]:
            yield fitted(opening, rate)
            if len(chunks) == 1:
                return
            # One worker: the next piece is synthesised while this one plays.
            # Never waited for on the way out -- after a barge-in the request in
            # flight finishes in the background and its audio is dropped.
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="sam-tts")
            try:
                pending = pool.submit(self._synthesize_sorani, chunks[1])
                for index in range(1, len(chunks)):
                    result = pending.result()
                    if cancel():
                        return
                    if index + 1 < len(chunks):
                        pending = pool.submit(self._synthesize_sorani, chunks[index + 1])
                    payload = result.get("audio")
                    if not payload:
                        failure.update(index=index, error=result.get("error") or "KurdishTTS returned no audio.")
                        return
                    frames, frames_rate = sorani_speech.wav_to_frames(payload)
                    said["chunks"] += 1
                    yield fitted(frames, frames_rate)
            finally:
                pool.shutdown(wait=False, cancel_futures=True)

        played = play_audio(pieces(), rate, cancel=cancel)
        outcome = {**first, "spoken": bool(played.get("played")),
                   "interrupted": bool(played.get("interrupted")),
                   "chunks": said["chunks"], "chunks_total": len(chunks)}
        for key in ("seconds", "underflows"):
            if key in played:
                outcome[key] = played[key]
        if played.get("error"):
            # Playback's failure is the caller's business too; it used to stop here.
            outcome["error"] = played["error"]
        elif failure:
            # What was said stays said; the reply as a whole did not get out.
            outcome["error"] = (f"Stopped after {failure['index']} of {len(chunks)} sentences: "
                                f"{failure['error']}")[:200]
        if outcome.get("error"):
            logger.warning("Sorani speech: %s", outcome["error"])
        return outcome

    def _play_wav(self, payload: bytes, *, cancel: Callable[[], bool] | None = None) -> dict[str, Any]:
        """Play a WAV clip through one continuous stream that barge-in can abort.

        It used to be 200 ms slices, each its own sounddevice.play()+wait():
        a stream opened and torn down five times a second, which on this
        machine left a quarter-second hole after every fifth of a second of
        voice (see PLAYBACK_BLOCK_SECONDS).
        """
        try:
            frames, rate = sorani_speech.wav_to_frames(payload)
        except Exception as exc:  # noqa: BLE001
            return {"played": False, "interrupted": False, "error": f"Unreadable speech audio: {exc}"[:160]}
        if frames.shape[1] > 2:
            # Speech has nothing to say to a surround layout; fold it to mono.
            frames = frames.mean(axis=1, keepdims=True)
        return play_audio([frames], rate, cancel=cancel)
