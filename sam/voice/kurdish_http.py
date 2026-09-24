"""One kept-alive HTTPS connection pool for KurdishTTS speech-to-text AND text-to-speech.

httpx closes idle connections after 5 s by default (0.28.1 ``DEFAULT_LIMITS``,
``keepalive_expiry=5.0``), and spoken turns are almost always more than 5 s
apart, so every turn paid a new TLS handshake for STT and again for TTS
(both on www.kurdishtts.com, in two separate pools). Measured by the repair
review on this PC (keepalive_probe.py): KurdishTTS cold 455 ms, reused after
8 s idle 448 ms with the default vs 217 ms with ``keepalive_expiry=120``;
Groq 476 vs 277 ms. So one pool, 120 s keep-alive, and a warm-up request when
listening starts (the first turn then skips the cold handshake too).
"""

from __future__ import annotations

import logging
import weakref
from typing import Any

log = logging.getLogger("sam.voice.http")

KEEPALIVE_S = 120.0
WARM_URL = "https://www.kurdishtts.com/"

_SHARED: "weakref.WeakKeyDictionary[Any, KurdishHttp]" = weakref.WeakKeyDictionary()


def limits() -> Any:
    import httpx

    return httpx.Limits(max_connections=10, max_keepalive_connections=4, keepalive_expiry=KEEPALIVE_S)


def new_client(transport: Any = None, *, read_s: float = 30.0) -> Any:
    import httpx

    return httpx.AsyncClient(timeout=httpx.Timeout(read_s, connect=10.0), trust_env=False, transport=transport,
                             limits=limits())


class KurdishHttp:
    """The shared client (created on first use, recreated after close)."""

    def __init__(self) -> None:
        self._client: Any = None

    def client(self) -> Any:
        if self._client is None or self._client.is_closed:
            self._client = new_client()
        return self._client

    async def warm(self, url: str = WARM_URL) -> bool:
        """Open the TLS connection before the first turn (a HEAD on the site
        root: no API call, no key, no quota)."""
        try:
            response = await self.client().head(url, timeout=5.0)
            return response.status_code < 500
        except Exception as exc:  # noqa: BLE001 - warming is best effort
            log.debug("kurdishtts warm-up failed: %s", type(exc).__name__)
            return False

    async def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001
                pass


def shared(app: Any) -> KurdishHttp:
    try:
        holder = _SHARED.get(app)
        if holder is None:
            holder = _SHARED[app] = KurdishHttp()
        return holder
    except TypeError:  # an object that cannot be weakly referenced (tests): a private holder
        return KurdishHttp()


__all__ = ["KurdishHttp", "shared", "new_client", "limits", "KEEPALIVE_S"]
