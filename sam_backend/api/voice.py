"""Speech in and speech out, including the Sorani providers."""

from __future__ import annotations

import asyncio

from .. import sorani as sorani_speech
from ..schemas import VoiceListenRequest, VoiceSpeakRequest
from .services import AppServices
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.responses import Response
from typing import Any



VOICE_UPLOAD_MAX_BYTES = 8 * 1024 * 1024


def register_voice_routes(application: FastAPI, sv: AppServices) -> None:
    settings = sv.settings
    database = sv.database
    cancellation = sv.cancellation
    voice = sv.voice
    @application.get("/api/voice/handsfree")
    async def handsfree_status() -> dict[str, Any]:
        """What the hands-free loop is doing. Read-only, and never audio."""
        return sv.voice_session.describe()

    @application.post("/api/voice/handsfree/start")
    async def handsfree_start() -> dict[str, Any]:
        """Begin listening for the wake phrase.

        Refuses while the setting is off rather than quietly turning it on:
        switching a microphone on is the user's decision, made in Settings.
        """
        if not sv.settings.hands_free_enabled:
            raise HTTPException(409, {
                "error": "Hands-free voice is switched off. Enable it in Settings first.",
                "code": "NOT_CONFIGURED"})
        return await asyncio.to_thread(sv.voice_session.start)

    @application.post("/api/voice/handsfree/stop")
    async def handsfree_stop() -> dict[str, Any]:
        return await asyncio.to_thread(sv.voice_session.stop)

    @application.get("/api/voice/capabilities")
    async def voice_capabilities() -> dict[str, Any]:
        local = await asyncio.to_thread(voice.capabilities)
        return {
            **local,
            "browser_fallback": {
                "vad": True, "partial_transcript": True, "continuous": True,
                "barge_in": True, "streaming_tts_chunks": True,
            },
        }

    def _require_stt(language: str | None) -> None:
        spoken = language or settings.voice_language
        if sorani_speech.is_sorani(spoken):
            # A live health probe used to transcribe silence before the user's
            # audio. That cost a full KurdishTTS round trip for nothing.
            if not voice.sorani_input_configured():
                raise HTTPException(503, sorani_speech.MESSAGE_NO_PROVIDER)
            return
        support = voice.capabilities()
        if support["stt"]["state"] != "AVAILABLE":
            raise HTTPException(503, support["stt"].get("reason") or "Local speech recognition is unavailable.")

    @application.post("/api/voice/listen")
    async def voice_listen(payload: VoiceListenRequest) -> dict[str, Any]:
        await asyncio.to_thread(_require_stt, payload.language)
        result = await asyncio.to_thread(
            voice.listen_once,
            max_seconds=payload.max_seconds,
            device=payload.device,
            language=payload.language,
        )
        database.add_audit(
            "voice", "listen", "Captured one utterance", actor="user",
            details={"captured": result.get("captured"), "seconds": result.get("seconds")},
        )
        return result

    @application.post("/api/voice/transcribe")
    async def voice_transcribe(
        file: UploadFile = File(...),
        language: str | None = Form(None),
    ) -> dict[str, Any]:
        """Transcribe a browser-recorded WAV. Used for Sorani, which Web Speech cannot do."""
        await asyncio.to_thread(_require_stt, language)
        payload = await file.read()
        if len(payload) > VOICE_UPLOAD_MAX_BYTES:
            raise HTTPException(413, "دەنگەکە زۆر گەورەیە.")
        if not payload:
            raise HTTPException(400, sorani_speech.MESSAGE_EMPTY_AUDIO)
        try:
            result = await asyncio.to_thread(voice.transcribe_audio_bytes, payload, language)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        database.add_audit(
            "voice", "transcribe", "Transcribed an uploaded utterance", actor="user",
            details={"captured": result.get("captured"), "seconds": result.get("seconds"),
                     "engine": result.get("engine")},
        )
        return result

    @application.post("/api/voice/speak")
    async def voice_speak(payload: VoiceSpeakRequest) -> dict[str, Any]:
        return await asyncio.to_thread(voice.speak, payload.text, payload.language)

    @application.post("/api/voice/tts")
    async def voice_tts(payload: VoiceSpeakRequest) -> Response:
        """Return Sorani WAV for browser playback. Does not play on the server speakers."""
        result = await asyncio.to_thread(voice.synthesize, payload.text, payload.language)
        audio = result.get("audio")
        if not audio:
            raise HTTPException(503, result.get("error") or sorani_speech.MESSAGE_NO_PROVIDER)
        headers = {
            "X-SAM-Engine": str(result.get("engine") or ""),
            "X-SAM-Language": str(result.get("language") or ""),
        }
        if result.get("speaker_id"):
            headers["X-SAM-Speaker"] = str(result["speaker_id"])
        return Response(content=audio, media_type="audio/wav", headers=headers)

    @application.get("/api/voice/sorani/speakers")
    async def sorani_speakers() -> dict[str, Any]:
        """The Sorani voices available for replies. Never returns a credential."""
        def read() -> dict[str, Any]:
            _, tts_router = voice.sorani_stack()
            health = tts_router.status()["primary"]
            if health["status"] != "CONNECTED":
                return {"status": health["status"], "detail": health.get("detail"), "speakers": []}
            provider = tts_router.primary
            return {
                "status": "CONNECTED",
                "selected": provider.resolve_speaker(),
                "speakers": [speaker.as_dict() for speaker in provider.available_speakers()],
            }

        return await asyncio.to_thread(read)

    @application.post("/api/voice/interrupt")
    async def voice_interrupt() -> dict[str, Any]:
        was_speaking = voice.barge_in.interrupt("api")
        # Barge-in must also stop whatever work the spoken answer came from.
        cancelled = cancellation.emergency_stop("voice_barge_in") if was_speaking else {}
        return {"was_speaking": was_speaking, "interruptions": voice.barge_in.interruptions, "cancelled": cancelled}
