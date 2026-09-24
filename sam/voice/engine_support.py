"""Helpers of ``VoiceEngine`` split out of engine.py (kept under ~700 lines):
the Settings key test, the fixed-phrase TTS cache prewarm, connection warm-up
and the global hotkey registration. ``EngineSupport`` is a mixin; everything
it uses (``self.app``, ``self.tts``, ``self.stt``, ``self.cascade``, the hotkey
fields) is set by ``VoiceEngine.__init__``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ..events import Error
from . import kurdish_http, strings
from .stt import SttError
from .tts import TtsError

log = logging.getLogger("sam.voice")


class EngineSupport:
    app: Any
    stt: Any
    tts: Any
    cascade: Any
    _hotkey: Any
    _hotkey_lock: asyncio.Lock | None
    _hotkey_factory: Any
    _hotkey_note: str
    _prewarm_started: bool
    _stopped: bool
    hotkey_error: str | None

    async def test_key(self, name: str) -> dict[str, Any]:
        """Settings "Test" for the KurdishTTS keys (sam/ui/pages/settings.py
        calls it duck-typed; result shaped like ``llm.test_provider``). Only on
        a user click: it spends 4 TTS characters or 1 s of STT. v1 removed an
        automatic silence probe that ran before every utterance, so this is
        never called automatically."""
        started = time.perf_counter()

        def result(ok: bool, status: str, detail: str = "") -> dict[str, Any]:
            return {"ok": ok, "provider": name, "status": status, "detail": detail,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1)}

        providers = {**getattr(self.stt, "providers", {}), **{f"tts:{k}": v for k, v in
                                                             getattr(self.tts, "providers", {}).items()}}
        try:
            if name == "kurdishtts_tts_api_key" and "tts:kurdishtts" in providers:
                produced = 0
                async for chunk in providers["tts:kurdishtts"].stream("سڵاو"):
                    produced += len(chunk)
                return result(produced > 0, "connected" if produced else "error")
            if name == "kurdishtts_stt_api_key" and "kurdishtts" in providers:
                await providers["kurdishtts"].transcribe(bytes(32000))  # 1 s of silence
                return result(True, "connected")
        except (SttError, TtsError) as exc:
            status = {"unconfigured": "unconfigured", "auth": "auth_failed", "quota": "rate_limited",
                      "rate_limit": "rate_limited", "network": "unreachable"}.get(exc.kind, "error")
            return result(False, status, exc.kind)
        return result(False, "error", "no voice test for this key")

    @staticmethod
    def fixed_phrases() -> list[str]:
        """Sentences SAM speaks word for word: the brain's acknowledgements
        (spoken first in every tool-using voice turn) and the voice package's
        own. Only these go into the TTS phrase cache (sam/voice/tts_cache.py)."""
        phrases = [strings.STT_FAILED_SPOKEN]
        try:
            from ..brain.conversation import ACKS_DO, ACKS_LOOK  # optional: the brain may be missing
            phrases = [*ACKS_DO, *ACKS_LOOK, *phrases]
        except Exception:  # noqa: BLE001
            pass
        return phrases

    def _start_prewarm(self) -> None:
        """Once per run: synthesize the missing fixed phrases into the cache
        while nothing else is speaking (~100 KurdishTTS characters, once;
        measured: the acknowledgement's first audio went from 2.0 s to 4 ms)."""
        prewarm = getattr(self.tts, "prewarm", None)
        if self._prewarm_started or prewarm is None or not self.app.config.get("voice.tts_prewarm", True):
            return
        self._prewarm_started = True
        self.app.spawn(prewarm(self.fixed_phrases(), idle=lambda: not self.cascade.busy), "voice-tts-prewarm")

    async def _warm_connections(self) -> None:
        """Open the TLS connections the first cascade turn needs (KurdishTTS and
        the first text model) while the user is still speaking."""
        providers = [*getattr(self.stt, "providers", {}).values(), *getattr(self.tts, "providers", {}).values()]
        kurdish = any(getattr(p, "provider", "") == "kurdishtts" and p.configured() for p in providers)
        jobs = [kurdish_http.shared(self.app).warm()] if kurdish else []  # (test fakes: no network)
        try:
            backend = self.app.llm.backends.get("groq")
            if backend is not None and hasattr(backend, "warm"):
                jobs.append(backend.warm())
        except Exception:  # noqa: BLE001
            pass
        if jobs:
            await asyncio.gather(*jobs, return_exceptions=True)

    def _get_hotkey_lock(self) -> asyncio.Lock:
        if self._hotkey_lock is None:
            self._hotkey_lock = asyncio.Lock()
        return self._hotkey_lock

    async def _register_hotkey(self) -> None:
        """(Re)register the chord; serialised so a Settings change during the
        startup registration cannot leave two hotkey threads."""
        async with self._get_hotkey_lock():
            if self._stopped:
                return
            await self._register_hotkey_locked()

    async def _register_hotkey_locked(self) -> None:
        if self._hotkey is not None:
            await asyncio.to_thread(self._hotkey.stop)
            self._hotkey = None
        keys = str(self.app.config.get("voice.hotkey", "ctrl+alt+space") or "")
        if not keys:
            return
        loop = asyncio.get_running_loop()

        def pressed() -> None:  # hotkey thread -> core loop
            loop.call_soon_threadsafe(lambda: self.app.spawn(self.toggle_listening(), "voice-hotkey"))

        hotkey = self._hotkey_factory(keys, pressed)
        ok = await asyncio.to_thread(hotkey.start)
        self._hotkey = hotkey
        self.hotkey_error = None if ok else (getattr(hotkey, "error", None) or "failed")
        if ok:
            return
        log.warning("hotkey %s not registered: %s", keys, self.hotkey_error)
        # Measured 2026-09-24: another program on this PC already owns
        # Ctrl+Alt+Space (not SAM v1, which registers no hotkey). While the
        # user has not chosen a chord, take the first free fallback and store it
        # so Settings shows the key that really works.
        user_chose = self.app.db.query_one("SELECT 1 AS x FROM settings WHERE key='voice.hotkey'") is not None
        if not user_chose:
            for fallback in self.app.config.get("voice.hotkey_fallbacks", []) or []:
                candidate = self._hotkey_factory(str(fallback), pressed)
                if await asyncio.to_thread(candidate.start):
                    self._hotkey, self.hotkey_error = candidate, None
                    self._hotkey_note = f"{keys} -> {fallback}"
                    self.app.config.set("voice.hotkey", str(fallback))  # SettingsChanged: same chord, no-op
                    self.app.bus.publish(Error(where="voice.hotkey", message_ckb=strings.HOTKEY_FALLBACK.format(
                        hotkey=keys, fallback=fallback), detail=f"{keys}: {hotkey.error}"))
                    return
        self.app.bus.publish(Error(where="voice.hotkey", message_ckb=strings.HOTKEY_FAILED.format(hotkey=keys),
                                   detail=str(self.hotkey_error)))


__all__ = ["EngineSupport"]
