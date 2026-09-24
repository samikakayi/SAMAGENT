"""Start-up snapshot of component states for the UI (a late subscriber).

``sam.__main__`` awaits ``app.start()`` before ``sam.ui.run`` attaches the
UiAdapter, so every ``ComponentStatus`` / ``VoiceState`` published during
start-up (voice ready, OmniRoute running, MT5 connected, TradingView probed) is
gone by the time the UI listens, and the panel's status dots would stay
"unknown" until the next change. The bus keeps no history, so the UI reads the
same states back once, through cheap calls without side effects:

- voice: ``voice.status()`` and its STT/TTS ``configured()`` (what the engine
  itself publishes);
- OmniRoute: client key present + a 0.3 s local TCP probe of the gateway port;
- TradingView: ``tv.status()["connected"]`` (never launches or restarts);
- MT5: the feed's ``connected`` / ``offset_verified`` flags (never calls
  ``initialize()``, which could start the terminal).

The controller applies a snapshot entry only when no real event for that
component arrived first (real events always win).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import urlparse

from ..events import ComponentStatus, VoiceState

log = logging.getLogger("sam.ui.status")


def _voice_events(app: Any) -> list[Any]:
    voice = getattr(app, "voice", None)
    if voice is None:
        return []
    events: list[Any] = []
    try:
        status = voice.status() if callable(getattr(voice, "status", None)) else {}
        state = str(status.get("state") or getattr(voice, "state", "") or "idle")
        events.append(VoiceState(state=state, engine=str(getattr(voice, "engine_name", "") or "")))  # type: ignore[arg-type]
        stt, tts = getattr(voice, "stt", None), getattr(voice, "tts", None)
        stt_ok = stt.configured() if callable(getattr(stt, "configured", None)) else True
        tts_ok = tts.configured() if callable(getattr(tts, "configured", None)) else True
        comp = "degraded" if status.get("live_degraded") else "ok" if stt_ok and tts_ok else "unconfigured"
        events.append(ComponentStatus(component="voice", state=comp, detail="start-up snapshot"))  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001 - a snapshot must never break the UI
        log.debug("voice snapshot failed", exc_info=True)
    return events


async def _omniroute_event(app: Any) -> ComponentStatus | None:
    try:
        if not app.secrets.has("litellm_api_key"):
            return ComponentStatus(component="omniroute", state="unconfigured", detail="no client key")  # type: ignore[arg-type]
        url = urlparse(str(app.config.get("providers.omniroute.base_url", "") or ""))
        if not url.hostname:
            return None
        from ..omniroute import port_open

        port = url.port or (443 if url.scheme == "https" else 80)
        up = await asyncio.to_thread(port_open, url.hostname, port)
        return ComponentStatus(component="omniroute", state="ok" if up else "down",  # type: ignore[arg-type]
                               detail="start-up snapshot")
    except Exception:  # noqa: BLE001
        log.debug("omniroute snapshot failed", exc_info=True)
        return None


def _trading_events(app: Any) -> list[Any]:
    trading = getattr(app, "trading", None)
    events: list[Any] = []
    tv = getattr(trading, "tv", None)
    try:
        if tv is not None and callable(getattr(tv, "status", None)):
            connected = bool((tv.status() or {}).get("connected"))
            events.append(ComponentStatus(component="tradingview", state="ok" if connected else "down",  # type: ignore[arg-type]
                                          detail="connected" if connected else "not connected"))
    except Exception:  # noqa: BLE001
        log.debug("tradingview snapshot failed", exc_info=True)
    mt5 = getattr(trading, "mt5", None)
    if mt5 is not None and isinstance(getattr(mt5, "connected", None), bool):
        ok = mt5.connected
        state = "ok" if ok and getattr(mt5, "offset_verified", True) else "degraded" if ok else "down"
        events.append(ComponentStatus(component="mt5", state=state, detail="start-up snapshot"))  # type: ignore[arg-type]
    return events


def _brain_event(app: Any) -> ComponentStatus | None:
    """Cloud or local brain (``LLMClient.brain_mode``, sam/brain/llm_local.py):
    no request, only the client's own memory of who answered last."""
    llm = getattr(app, "llm", None)
    mode = getattr(llm, "brain_mode", None)
    if mode is None:
        return None
    if mode == "local":
        return ComponentStatus(component="brain", state="degraded", detail="local: start-up snapshot")  # type: ignore[arg-type]
    return ComponentStatus(component="brain", state="ok", detail="cloud: start-up snapshot")  # type: ignore[arg-type]


async def snapshot(app: Any) -> list[Any]:
    """Current ``VoiceState`` + ``ComponentStatus`` events (runs on the core loop)."""
    events = _voice_events(app)
    brain = _brain_event(app)
    if brain is not None:
        events.append(brain)
    omni = await _omniroute_event(app)
    if omni is not None:
        events.append(omni)
    return events + _trading_events(app)


__all__ = ["snapshot"]
