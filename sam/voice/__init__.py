"""SAM 2 voice: Gemini Live (primary) and an STT -> brain -> TTS cascade.

Entry module for ``sam.app.PACKAGES`` (contract 3.1): ``register`` creates
``app.voice`` and the ``voice_tts_cache`` table (no devices, no network, no
heavy imports -- google.genai, numpy and sounddevice load lazily); ``start``
subscribes to events and registers the hotkey in the background (the mic
stays closed until the user starts listening); ``stop`` releases everything.
"""

from __future__ import annotations

from typing import Any


def register(app: Any) -> None:
    from .engine import VOICE_DEFAULTS, VoiceEngine
    from .tts_cache import ensure_schema

    app.config.register_defaults(VOICE_DEFAULTS)
    ensure_schema(app.db)          # table voice_tts_cache (namespace "voice")
    app.voice = VoiceEngine(app)


async def start(app: Any) -> None:
    if app.voice is not None:
        await app.voice.start()


async def stop(app: Any) -> None:
    if app.voice is not None:
        await app.voice.stop()


__all__ = ["register", "start", "stop"]
