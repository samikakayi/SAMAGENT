"""Sorani (Central Kurdish) speech providers.

Whisper has no Kurdish model and Windows has no Kurdish voice, so Sorani speech
is served by KurdishTTS. The contract below was verified against the live API
rather than assumed:

    GET  /api/get-speakers?model_version=v4   -> {"speakers":[{id,name,dialect,gender,speaker_id}]}
    POST /api/tts-proxy                       -> audio/wav  {text, model_version, speaker_id}
    POST /api/stt-proxy                       -> JSON       multipart file + dialect
    POST /api/stt-stream-connect              -> websocket upgrade for live audio
    auth: x-api-key header; 401 body "Invalid API key"

The one rule that matters throughout: Sorani audio is never handed to an
English or Arabic recogniser to manufacture a result. If no Sorani provider is
available the caller is told so, in Sorani.
"""

from __future__ import annotations

import io
import time
import wave
from dataclasses import dataclass
from typing import Any, Iterator

import httpx

from .contracts import CapabilityState

KURDISHTTS_BASE = "https://www.kurdishtts.com/api"
DEFAULT_MODEL_VERSION = "v4"
SORANI_DIALECT = "sorani"
SORANI_CODES = {"ckb", "ckb-iq", "ku", "kur", "sorani"}

# Messages the user actually sees are written in Sorani, since that is the
# language they operate SAM in.
MESSAGE_NO_PROVIDER = "هیچ دابینکەرێکی دەنگی سۆرانی ڕێکنەخراوە. تکایە کلیلی KurdishTTS لە ڕێکخستنەکاندا دابنێ."
MESSAGE_AUTH_FAILED = "کلیلی KurdishTTS ڕەتکرایەوە. تکایە کلیلەکە نوێ بکەرەوە."
MESSAGE_RATE_LIMITED = "KurdishTTS سنووری داواکاری دانراوە. تکایە کەمێک چاوەڕێ بکە."
MESSAGE_UNREACHABLE = "ناتوانرێت پەیوەندی بە KurdishTTS بکرێت. ئینتەرنێت بپشکنە."
MESSAGE_EMPTY_AUDIO = "هیچ دەنگێک نەدۆزرایەوە بۆ ناسینەوە."
MESSAGE_TOO_SHORT = "دەنگەکە زۆر کورت بوو بۆ ناسینەوە. تکایە دووبارە بڵێوە."

# Measured against the live service: utterances of roughly half a second come
# back empty about three times in four, while utterances of about a second are
# recognised reliably. A one-word command can therefore be missed, so stopping
# SAM must never depend on recognising a word -- barge-in listens for speech
# energy instead, which does not need a transcript.
MIN_RELIABLE_SECONDS = 0.6


def is_sorani(language: str | None) -> bool:
    """Does this language tag mean Central Kurdish?"""
    if not language:
        return False
    code = language.strip().lower()
    return code in SORANI_CODES or code.split("-")[0] in {"ckb", "ku", "kur"}


def _classify(status_code: int, body: str = "") -> tuple[str, str]:
    """Map an HTTP status onto SAM's provider states and a Sorani message."""
    if status_code in (401, 403):
        return "AUTH_FAILED", MESSAGE_AUTH_FAILED
    if status_code == 429:
        return "RATE_LIMITED", MESSAGE_RATE_LIMITED
    if status_code >= 500:
        return "ERROR", f"KurdishTTS هەڵەیەکی ناوخۆیی گەڕاندەوە ({status_code})."
    if status_code >= 400:
        return "ERROR", f"KurdishTTS داواکارییەکەی ڕەتکردەوە ({status_code})."
    return "CONNECTED", ""


@dataclass(slots=True, frozen=True)
class SoraniSpeaker:
    id: str
    name: str
    gender: str
    dialect: str

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "gender": self.gender, "dialect": self.dialect}


def pcm_to_wav(samples: Any, sample_rate: int = 16_000) -> bytes:
    """Wrap float32 mono audio as a 16-bit PCM WAV, which the API accepts."""
    import numpy

    array = numpy.asarray(samples, dtype=numpy.float32)
    clipped = numpy.clip(array, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(numpy.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def wav_to_float32(payload: bytes) -> tuple[Any, int]:
    """Decode a WAV blob into float32 mono samples and its sample rate."""
    import numpy

    with wave.open(io.BytesIO(payload), "rb") as handle:
        channels, width, rate = handle.getnchannels(), handle.getsampwidth(), handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    dtype = {1: numpy.int8, 2: numpy.int16, 4: numpy.int32}.get(width)
    if dtype is None:
        raise ValueError(f"Unsupported WAV sample width: {width}")
    audio = numpy.frombuffer(frames, dtype=dtype).astype(numpy.float32) / float(numpy.iinfo(dtype).max)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, rate


class KurdishTTSClient:
    """Shared transport for both KurdishTTS providers."""

    def __init__(self, *, stt_key: str | None, tts_key: str | None, base_url: str = KURDISHTTS_BASE,
                 timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._stt_key = stt_key
        self._tts_key = tts_key
        self.timeout = timeout

    @property
    def stt_configured(self) -> bool:
        return bool(self._stt_key)

    @property
    def tts_configured(self) -> bool:
        return bool(self._tts_key)

    def _headers(self, key: str | None, content_type: str | None = None) -> dict[str, str]:
        headers = {"x-api-key": key or ""}
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    # --- Speakers ---------------------------------------------------------

    def speakers(self, model_version: str = DEFAULT_MODEL_VERSION) -> list[SoraniSpeaker]:
        if not self._tts_key:
            return []
        with httpx.Client(timeout=self.timeout, trust_env=False) as client:
            response = client.get(
                f"{self.base_url}/get-speakers",
                params={"model_version": model_version},
                headers=self._headers(self._tts_key),
            )
        response.raise_for_status()
        payload = response.json() or {}
        found: list[SoraniSpeaker] = []
        for item in payload.get("speakers") or []:
            identifier = item.get("speaker_id") or item.get("id")
            if not identifier:
                continue
            found.append(SoraniSpeaker(
                id=str(identifier), name=str(item.get("name") or identifier),
                gender=str(item.get("gender") or "unknown"), dialect=str(item.get("dialect") or "unknown"),
            ))
        return found

    def sorani_speakers(self, model_version: str = DEFAULT_MODEL_VERSION) -> list[SoraniSpeaker]:
        return [item for item in self.speakers(model_version) if item.dialect.lower().startswith("sor")]

    # --- Text to speech ---------------------------------------------------

    def synthesize(self, text: str, *, speaker_id: str,
                   model_version: str = DEFAULT_MODEL_VERSION) -> tuple[bytes, dict[str, Any]]:
        """Return WAV audio for Sorani text, plus timing metadata."""
        if not self._tts_key:
            raise PermissionError(MESSAGE_NO_PROVIDER)
        started = time.perf_counter()
        with httpx.Client(timeout=self.timeout, trust_env=False) as client:
            response = client.post(
                f"{self.base_url}/tts-proxy",
                headers=self._headers(self._tts_key, "application/json"),
                json={"text": text, "model_version": model_version, "speaker_id": speaker_id},
            )
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        if response.status_code != 200:
            _, message = _classify(response.status_code, response.text)
            raise RuntimeError(message)
        audio = response.content
        # A JSON error page or an empty body must never be treated as speech.
        if not audio or not audio.startswith(b"RIFF"):
            raise RuntimeError("KurdishTTS دەنگێکی دروستی نەگەڕاندەوە.")
        return audio, {
            "bytes": len(audio), "latency_ms": elapsed, "speaker_id": speaker_id,
            "model_version": model_version, "content_type": response.headers.get("content-type"),
        }

    # --- Speech to text ---------------------------------------------------

    def transcribe(self, wav_bytes: bytes, *, dialect: str = SORANI_DIALECT) -> dict[str, Any]:
        if not self._stt_key:
            raise PermissionError(MESSAGE_NO_PROVIDER)
        if not wav_bytes:
            raise ValueError(MESSAGE_EMPTY_AUDIO)
        started = time.perf_counter()
        with httpx.Client(timeout=self.timeout, trust_env=False) as client:
            response = client.post(
                f"{self.base_url}/stt-proxy",
                headers=self._headers(self._stt_key),
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                data={"dialect": dialect},
            )
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        if response.status_code != 200:
            _, message = _classify(response.status_code, response.text)
            raise RuntimeError(message)
        try:
            payload = response.json()
        except ValueError as exc:
            # An HTML error page decoded as a transcript would be worse than failing.
            raise RuntimeError("KurdishTTS وەڵامێکی نادروستی گەڕاندەوە.") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("KurdishTTS وەڵامێکی نادروستی گەڕاندەوە.")
        return {
            "text": str(payload.get("text") or "").strip(),
            "language": payload.get("language"),
            "detected_dialect": payload.get("detected_dialect"),
            "detected_script": payload.get("detected_script"),
            "duration_seconds": payload.get("duration_seconds"),
            "latency_ms": elapsed,
            "provider": "kurdishtts",
            "warnings": payload.get("warnings"),
        }


class KurdishTTSSTTProvider:
    """Primary Sorani speech recognition."""

    name = "kurdishtts"

    def __init__(self, client: KurdishTTSClient) -> None:
        self.client = client

    def health_check(self) -> dict[str, Any]:
        if not self.client.stt_configured:
            return {"provider": self.name, "role": "stt", "status": "UNCONFIGURED",
                    "state": CapabilityState.UNCONFIGURED.value, "detail": MESSAGE_NO_PROVIDER}
        # Do not transcribe silence as a probe. That round trip used to run
        # before every real utterance and doubled the wait for a spoken reply.
        # A missing key is reported here; a rejected key is reported when the
        # user actually speaks.
        return {"provider": self.name, "role": "stt", "status": "CONNECTED",
                "state": CapabilityState.AVAILABLE.value, "dialect": SORANI_DIALECT,
                "detail": "KurdishTTS ئامادەیە بۆ ناسینەوەی دەنگی سۆرانی."}

    def transcribe_audio(self, samples: Any, sample_rate: int = 16_000) -> dict[str, Any]:
        return self.client.transcribe(pcm_to_wav(samples, sample_rate))

    def transcribe_file(self, path: str) -> dict[str, Any]:
        from pathlib import Path

        return self.client.transcribe(Path(path).read_bytes())


class KurdishTTSTTSProvider:
    """Primary Sorani speech output."""

    name = "kurdishtts"

    def __init__(self, client: KurdishTTSClient, *, preferred_speaker: str | None = None) -> None:
        self.client = client
        self.preferred_speaker = preferred_speaker
        self._speakers: list[SoraniSpeaker] | None = None

    def available_speakers(self, refresh: bool = False) -> list[SoraniSpeaker]:
        if self._speakers is None or refresh:
            try:
                self._speakers = self.client.sorani_speakers()
            except Exception:
                self._speakers = []
        return self._speakers

    def resolve_speaker(self) -> str | None:
        """Pick a real Sorani voice rather than assuming an identifier."""
        speakers = self.available_speakers()
        if self.preferred_speaker and any(item.id == self.preferred_speaker for item in speakers):
            return self.preferred_speaker
        if self.preferred_speaker and not speakers:
            return self.preferred_speaker
        male = next((item.id for item in speakers if item.gender.lower() == "male"), None)
        return male or (speakers[0].id if speakers else None)

    def health_check(self) -> dict[str, Any]:
        if not self.client.tts_configured:
            return {"provider": self.name, "role": "tts", "status": "UNCONFIGURED",
                    "state": CapabilityState.UNCONFIGURED.value, "detail": MESSAGE_NO_PROVIDER}
        try:
            speakers = self.client.sorani_speakers()
        except httpx.HTTPStatusError as exc:
            status, message = _classify(exc.response.status_code)
            return {"provider": self.name, "role": "tts", "status": status,
                    "state": CapabilityState.UNAVAILABLE.value, "detail": message}
        except httpx.HTTPError:
            return {"provider": self.name, "role": "tts", "status": "ERROR",
                    "state": CapabilityState.UNAVAILABLE.value, "detail": MESSAGE_UNREACHABLE}
        except Exception as exc:
            return {"provider": self.name, "role": "tts", "status": "ERROR",
                    "state": CapabilityState.UNAVAILABLE.value, "detail": str(exc)[:160]}
        if not speakers:
            return {"provider": self.name, "role": "tts", "status": "ERROR",
                    "state": CapabilityState.UNAVAILABLE.value,
                    "detail": "هیچ دەنگێکی سۆرانی نەدۆزرایەوە."}
        self._speakers = speakers
        return {"provider": self.name, "role": "tts", "status": "CONNECTED",
                "state": CapabilityState.AVAILABLE.value,
                "speakers": len(speakers), "selected_speaker": self.resolve_speaker(),
                "detail": f"{len(speakers)} دەنگی سۆرانی بەردەستە."}

    def synthesize(self, text: str, *, speaker_id: str | None = None) -> tuple[bytes, dict[str, Any]]:
        chosen = speaker_id or self.resolve_speaker()
        if not chosen:
            raise RuntimeError("هیچ دەنگێکی سۆرانی بەردەست نییە.")
        return self.client.synthesize(text, speaker_id=chosen)


class GoogleSoraniSTTProvider:
    """Optional Central Kurdish fallback via Google Cloud Speech-to-Text."""

    name = "google"
    language_code = "ckb-IQ"

    def __init__(self, credentials_path: str | None = None) -> None:
        self.credentials_path = credentials_path

    def _library_available(self) -> bool:
        import importlib.util

        return importlib.util.find_spec("google.cloud.speech") is not None

    def health_check(self) -> dict[str, Any]:
        if not self._library_available():
            return {"provider": self.name, "role": "stt", "status": "UNCONFIGURED",
                    "state": CapabilityState.UNCONFIGURED.value,
                    "language": self.language_code,
                    "detail": "google-cloud-speech is not installed; the KurdishTTS provider is used instead."}
        import os

        credentials = self.credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not credentials:
            return {"provider": self.name, "role": "stt", "status": "UNCONFIGURED",
                    "state": CapabilityState.UNCONFIGURED.value, "language": self.language_code,
                    "detail": "No Google application credentials are configured."}
        try:
            from google.cloud import speech

            speech.SpeechClient()
        except Exception as exc:
            return {"provider": self.name, "role": "stt", "status": "AUTH_FAILED",
                    "state": CapabilityState.UNAVAILABLE.value, "language": self.language_code,
                    "detail": f"Google credentials were rejected: {str(exc)[:120]}"}
        return {"provider": self.name, "role": "stt", "status": "CONNECTED",
                "state": CapabilityState.AVAILABLE.value, "language": self.language_code,
                "detail": "Google Speech-to-Text is available as a Central Kurdish fallback."}

    def transcribe_audio(self, samples: Any, sample_rate: int = 16_000) -> dict[str, Any]:
        from google.cloud import speech

        client = speech.SpeechClient()
        audio = speech.RecognitionAudio(content=pcm_to_wav(samples, sample_rate))
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code=self.language_code,
            enable_automatic_punctuation=True,
        )
        response = client.recognize(config=config, audio=audio)
        text = " ".join(result.alternatives[0].transcript for result in response.results if result.alternatives)
        return {"text": text.strip(), "provider": self.name, "language": self.language_code,
                "detected_dialect": SORANI_DIALECT}


class SoraniSTTRouter:
    """KurdishTTS first, Google only on a genuine provider failure."""

    def __init__(self, primary: KurdishTTSSTTProvider, fallback: GoogleSoraniSTTProvider | None = None) -> None:
        self.primary = primary
        self.fallback = fallback
        self.last_route: str | None = None

    def transcribe(self, samples: Any, sample_rate: int = 16_000) -> dict[str, Any]:
        attempts: list[dict[str, str]] = []
        try:
            result = self.primary.transcribe_audio(samples, sample_rate)
            self.last_route = "kurdishtts"
            return {**result, "route": "kurdishtts", "fallbacks": attempts}
        except PermissionError as exc:
            attempts.append({"provider": "kurdishtts", "reason": "unconfigured"})
            primary_error: Exception = exc
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            # A transport or service failure is a fallback condition. A merely
            # surprising transcript is not, so nothing below re-runs on content.
            attempts.append({"provider": "kurdishtts", "reason": type(exc).__name__})
            primary_error = exc

        if self.fallback is not None and self.fallback.health_check()["status"] == "CONNECTED":
            try:
                result = self.fallback.transcribe_audio(samples, sample_rate)
                self.last_route = "google"
                return {**result, "route": "google", "fallbacks": attempts}
            except Exception as exc:
                attempts.append({"provider": "google", "reason": type(exc).__name__})
        self.last_route = None
        raise RuntimeError(str(primary_error))

    def status(self) -> dict[str, Any]:
        return {
            "primary": self.primary.health_check(),
            "fallback": self.fallback.health_check() if self.fallback else {
                "provider": "google", "role": "stt", "status": "UNCONFIGURED",
                "state": CapabilityState.UNCONFIGURED.value, "language": "ckb-IQ",
                "detail": "No Google fallback is configured.",
            },
            "last_route": self.last_route,
        }


class SoraniTTSRouter:
    """Sorani speech output. Never substitutes an English voice."""

    def __init__(self, primary: KurdishTTSTTSProvider) -> None:
        self.primary = primary
        self.last_route: str | None = None

    def synthesize(self, text: str, *, speaker_id: str | None = None) -> tuple[bytes, dict[str, Any]]:
        audio, metadata = self.primary.synthesize(text, speaker_id=speaker_id)
        self.last_route = "kurdishtts"
        return audio, {**metadata, "route": "kurdishtts", "language": "ckb"}

    def status(self) -> dict[str, Any]:
        return {"primary": self.primary.health_check(), "last_route": self.last_route,
                "note": "An English voice is never substituted for Sorani."}


# Kurdish is written in an extended Arabic script. These four letters are the
# ones Sorani uses that Arabic itself does not, so their presence is a positive
# signal rather than a guess based on the Arabic block alone.
KURDISH_SPECIFIC = "\u06a9\u06cc\u06d5\u06b5\u0695\u06be\u06c6\u0648\u200c"
ARABIC_BLOCK = ("\u0600", "\u06ff")


def looks_sorani(text: str) -> bool:
    """Whether a string is written in Kurdish/Arabic script.

    Used only to decide which voice to speak a reply with. It never changes what
    was recognised, and it never routes Sorani audio into an Arabic recogniser.
    """
    if not text:
        return False
    arabic = sum(1 for character in text if ARABIC_BLOCK[0] <= character <= ARABIC_BLOCK[1])
    letters = sum(1 for character in text if character.isalpha())
    if not letters:
        return False
    return arabic / letters > 0.5


def build_routers(
    *,
    stt_key: str | None,
    tts_key: str | None,
    google_credentials_path: str | None = None,
    preferred_speaker: str | None = None,
    base_url: str = KURDISHTTS_BASE,
) -> tuple["SoraniSTTRouter", "SoraniTTSRouter"]:
    """Assemble the Sorani speech stack from already-resolved credentials.

    Credentials are passed in rather than read here, so this module never
    touches the secret store and a key cannot leak into it by accident.
    """
    client = KurdishTTSClient(stt_key=stt_key, tts_key=tts_key, base_url=base_url)
    fallback = GoogleSoraniSTTProvider(google_credentials_path) if google_credentials_path else None
    return (
        SoraniSTTRouter(KurdishTTSSTTProvider(client), fallback),
        SoraniTTSRouter(KurdishTTSTTSProvider(client, preferred_speaker=preferred_speaker)),
    )


def stream_chunks(audio: bytes, chunk_bytes: int = 32_768) -> Iterator[bytes]:
    """Yield WAV audio in pieces so playback can start before the whole file."""
    for offset in range(0, len(audio), chunk_bytes):
        yield audio[offset:offset + chunk_bytes]
