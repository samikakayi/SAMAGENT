"""Minimal Chrome DevTools Protocol client for TradingView Desktop (localhost only).

TradingView Desktop (Electron, Chrome 146 in 3.4.1) exposes a DevTools port when
it is started through Windows app activation with ``--remote-debugging-port=9222``
(verified 2026-09-24; see docs/DESIGN.md 2.4). This module only knows how to
find the chart page and talk to it; the chart logic lives in ``tradingview.py``.

Security rules enforced here (not left to callers):

- HTTP discovery and websockets go to ``127.0.0.1`` only. The websocket URL is
  rebuilt from the target id instead of trusting ``webSocketDebuggerUrl``.
- Proxies are disabled (``httpx`` ``trust_env=False``, ``websockets``
  ``proxy=None``): websockets 16 routes through env proxies by default.
- The port must belong to TradingView (``/json/version`` User-Agent says
  ``TradingView``) and a page is used only when its URL is an https
  ``tradingview.com`` chart. Page titles (they contain the account holder's
  name) are never kept or logged.
- No CDP domain is enabled (no Runtime/Network/Storage events): the client
  only sends ``Runtime.evaluate`` and ``Page.captureScreenshot``.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

log = logging.getLogger("sam.trading.cdp")

LOCAL_HOST = "127.0.0.1"
MAX_MESSAGE_BYTES = 64 * 1024 * 1024        # a 2697x1516 JPEG screenshot is ~180 KB; history is small
_TARGET_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
_CONTEXT_LOST = ("execution context was destroyed", "cannot find context", "cannot find default execution context",
                 "inspected target navigated or closed", "target closed")


class CdpError(Exception):
    """Base class for DevTools failures."""


class CdpUnavailable(CdpError):
    """Nothing (or not TradingView) answers on the DevTools port."""


class CdpClosed(CdpError):
    """The websocket closed (TradingView quit, tab closed or crashed)."""


class CdpTimeout(CdpError, TimeoutError):
    """No answer in time."""


class CdpProtocolError(CdpError):
    """The browser answered with ``{"error": ...}``."""

    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code

    @property
    def context_lost(self) -> bool:
        """True when the page reloaded/navigated under us (retry after re-inject)."""
        text = str(self).lower()
        return any(marker in text for marker in _CONTEXT_LOST)


class CdpJsError(CdpError):
    """The evaluated JavaScript threw."""


@dataclass(frozen=True)
class CdpTarget:
    """A chart page. ``url`` is kept for host checks; the title is deliberately dropped."""

    id: str
    url: str
    ws_url: str


def is_tradingview_url(url: str, *, chart_only: bool = True) -> bool:
    """https + host ``tradingview.com`` (or a subdomain) [+ ``/chart/`` path]."""
    try:
        parts = urlparse(url)
    except ValueError:
        return False
    host = (parts.hostname or "").lower()
    if parts.scheme != "https" or not (host == "tradingview.com" or host.endswith(".tradingview.com")):
        return False
    return "/chart/" in parts.path if chart_only else True


def is_tradingview_browser(version: dict[str, Any]) -> bool:
    """``/json/version`` of TradingView Desktop: 'User-Agent: ... TradingView/3.4.1 ... TVDesktop/3.4.1'."""
    agent = str(version.get("User-Agent", "")) + " " + str(version.get("Browser", ""))
    return "tradingview" in agent.lower() or "tvdesktop" in agent.lower()


def local_ws_url(port: int, target_id: str) -> str:
    if not _TARGET_ID.match(target_id):
        raise CdpError("unexpected DevTools target id")
    return f"ws://{LOCAL_HOST}:{int(port)}/devtools/page/{target_id}"


async def fetch_json(port: int, path: str, *, timeout_s: float = 2.0) -> Any:
    """GET ``http://127.0.0.1:<port><path>`` -> parsed JSON; CdpUnavailable when closed."""
    import httpx  # lazy: keeps register() fast

    url = f"http://{LOCAL_HOST}:{int(port)}{path}"
    try:
        async with httpx.AsyncClient(trust_env=False, timeout=timeout_s) as client:
            response = await client.get(url)
    except (httpx.HTTPError, OSError) as exc:
        raise CdpUnavailable(f"DevTools port {port} is not answering ({type(exc).__name__})") from None
    if response.status_code != 200:
        raise CdpUnavailable(f"DevTools port {port} answered HTTP {response.status_code}")
    try:
        return response.json()
    except ValueError:
        raise CdpUnavailable(f"DevTools port {port} did not return JSON") from None


async def browser_version(port: int, *, timeout_s: float = 2.0) -> dict[str, Any] | None:
    """``/json/version`` or None when nothing listens on the port."""
    try:
        data = await fetch_json(port, "/json/version", timeout_s=timeout_s)
    except CdpUnavailable:
        return None
    return data if isinstance(data, dict) else None


async def list_chart_targets(port: int, *, timeout_s: float = 2.0) -> list[CdpTarget]:
    """TradingView chart pages on the port (TradingView-only; see module doc).

    Raises CdpUnavailable when the port is closed or belongs to another app.
    """
    version = await browser_version(port, timeout_s=timeout_s)
    if version is None:
        raise CdpUnavailable(f"nothing answers on 127.0.0.1:{port}")
    if not is_tradingview_browser(version):
        raise CdpUnavailable(f"port {port} belongs to another application, not TradingView")
    raw = await fetch_json(port, "/json/list", timeout_s=timeout_s)
    targets: list[CdpTarget] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict) or item.get("type") != "page":
            continue
        url = str(item.get("url", ""))
        target_id = str(item.get("id", ""))
        if not is_tradingview_url(url) or not _TARGET_ID.match(target_id):
            continue
        targets.append(CdpTarget(id=target_id, url=url, ws_url=local_ws_url(port, target_id)))
    return targets


class CdpSession:
    """One websocket to one page target. Safe for concurrent ``call``s."""

    def __init__(self, ws_url: str, *, max_size: int = MAX_MESSAGE_BYTES) -> None:
        host = (urlparse(ws_url).hostname or "").lower()
        if host not in (LOCAL_HOST, "localhost"):
            raise CdpError("DevTools connections are allowed to 127.0.0.1 only")
        self.ws_url = ws_url
        self._max_size = max_size
        self._ws: Any = None
        self._reader: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._ids = itertools.count(1)
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    async def connect(self, *, timeout_s: float = 5.0) -> None:
        from websockets.asyncio.client import connect

        try:
            # ping_interval=None: the page's main thread can be busy for seconds
            # while TradingView loads history; per-call timeouts detect real loss.
            self._ws = await connect(self.ws_url, proxy=None, open_timeout=timeout_s, ping_interval=None,
                                     max_size=self._max_size, compression=None, close_timeout=2)
        except Exception as exc:  # noqa: BLE001 - refused, timeout, bad handshake: all mean "not reachable"
            raise CdpUnavailable(f"could not open the DevTools socket ({type(exc).__name__})") from None
        self._closed = False
        self._reader = asyncio.ensure_future(self._read_loop())

    async def _read_loop(self) -> None:
        try:
            async for raw in self._ws:
                try:
                    message = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                message_id = message.get("id") if isinstance(message, dict) else None
                if message_id is None:
                    continue  # events: no domain is enabled, ignore stragglers
                future = self._pending.pop(message_id, None)
                if future is None or future.done():
                    continue
                if "error" in message:
                    error = message.get("error") or {}
                    future.set_exception(CdpProtocolError(str(error.get("message", "DevTools error"))[:300],
                                                          error.get("code")))
                else:
                    future.set_result(message.get("result") or {})
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - ConnectionClosed and friends
            log.debug("DevTools socket ended: %s", type(exc).__name__)
        finally:
            self._closed = True
            pending, self._pending = self._pending, {}
            for future in pending.values():
                if not future.done():
                    future.set_exception(CdpClosed("TradingView closed the DevTools connection"))

    async def call(self, method: str, params: dict[str, Any] | None = None, *, timeout_s: float = 10.0) -> dict[str, Any]:
        """Send one command and wait for its result."""
        if self._closed or self._ws is None:
            raise CdpClosed("DevTools connection is closed")
        message_id = next(self._ids)
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[message_id] = future
        try:
            await self._ws.send(json.dumps({"id": message_id, "method": method, "params": params or {}}))
        except Exception:  # noqa: BLE001
            self._pending.pop(message_id, None)
            self._closed = True
            raise CdpClosed("DevTools connection is closed") from None
        try:
            return await asyncio.wait_for(future, timeout_s)
        except asyncio.TimeoutError:
            self._pending.pop(message_id, None)
            raise CdpTimeout(f"{method} got no answer within {timeout_s:.0f} s") from None

    async def evaluate(self, expression: str, *, await_promise: bool = False, timeout_s: float = 10.0) -> Any:
        """``Runtime.evaluate`` by value. JS exceptions become CdpJsError."""
        result = await self.call("Runtime.evaluate", {
            "expression": expression, "returnByValue": True, "awaitPromise": await_promise,
            "silent": True, "userGesture": False}, timeout_s=timeout_s)
        details = result.get("exceptionDetails")
        if details:
            exception = details.get("exception") or {}
            text = exception.get("description") or details.get("text") or "JavaScript error"
            raise CdpJsError(str(text).splitlines()[0][:300])
        value = result.get("result") or {}
        return value.get("value")

    async def close(self) -> None:
        self._closed = True
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                await asyncio.wait_for(ws.close(), 3)
            except Exception:  # noqa: BLE001
                pass
        if self._reader is not None:
            self._reader.cancel()
            try:
                await self._reader
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._reader = None


__all__ = ["CdpSession", "CdpTarget", "CdpError", "CdpUnavailable", "CdpClosed", "CdpTimeout", "CdpProtocolError",
           "CdpJsError", "browser_version", "list_chart_targets", "fetch_json", "is_tradingview_url",
           "is_tradingview_browser", "local_ws_url", "LOCAL_HOST"]
