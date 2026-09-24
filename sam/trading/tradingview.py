"""TradingView Desktop chart bridge over the Chrome DevTools Protocol (``app.trading.tv``).

What it does (docs/CONTRACTS.md 3.4): start TradingView with a local DevTools
port (or restart it once, after the user agrees), read the chart the user sees
(symbol, timeframe, bars), change symbol/timeframe, draw labelled lines, zones,
fibs, arrows and position tools, remove ONLY its own drawings, and take a chart
screenshot for vision without focusing the window.

Measured live on this PC (TradingView Desktop 3.4.1, 2026-09-24; acceptance
script ``acceptance/tradingview_live.py``): chart state 1.7 ms median round
trip, 300 bars 5-7 ms, five labelled drawings 62 ms (ten kinds, points read back
exactly), a clipped 1440 px chart screenshot 186-330 ms, timeframe change
0.8-3.3 s and symbol change ~2.2 s (TradingView loading history). v1 opened
TradingView 0 of 7 times and needed the mouse for 1-2 s per drawing
(reports/trading-intelligence.json); this bridge needs neither the mouse nor
the window in front.

Ownership: every drawing SAM creates is stored in the ``drawings`` table
(tv_id, kind, symbol, timeframe, points, text=label, tag, created_at).
``clear_my_drawings`` passes only those ids to the page, so user drawings are
never touched. TradingView keeps drawings per symbol, so a stored id that is
missing is considered only when its symbol is the one on the chart now -- and
then patiently, with re-adoption of drawings TradingView re-created under a
new id (``tv_owner.py``: after a symbol round trip SAM had lost its own lines).

This is an unofficial local automation of the user's own app; nothing leaves
127.0.0.1 and no account, cookie or storage data is read.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import Any, Awaitable, Callable

from .cdp import (CdpClosed, CdpError, CdpJsError, CdpProtocolError, CdpSession, CdpTarget, CdpTimeout,
                  CdpUnavailable, browser_version, is_tradingview_browser, is_tradingview_url, list_chart_targets)
from .common import Bar, canonical_symbol, from_tv_resolution
from .tv_app import TvProcess
from .tv_js import BUNDLE, LOCATION_EXPRESSION, VISIBILITY_EXPRESSION, call_expression
from .symbols import resolve_instrument
from .tv_owner import Ownership, match_shapes
from .tv_parse import (MAX_ITEMS, clean_tag, instrument_key, normalize_item, parse_tv_resolution,
                       resolution_label_ckb, same_resolution, tv_symbol_for)

log = logging.getLogger("sam.trading.tradingview")

CONFIRM_RESTART_CKB = ("TradingView دەبێت دابخرێت و دووبارە بکرێتەوە بۆ ئەوەی بتوانم لەسەر چارتەکە کار بکەم. "
                       "باشە؟")
Confirm = Callable[[str], Awaitable[bool]]


class TvError(Exception):
    """A chart operation failed; ``code`` is machine-readable (not_ready, timeout, ...)."""

    def __init__(self, message: str, *, code: str = "error") -> None:
        super().__init__(message)
        self.code = code


class TradingViewBridge:
    """``app.trading.tv``. All public coroutines run on the core loop."""

    poll_s = 0.5                 # tests shrink these
    launch_timeout_s = 60.0      # cold start of TradingView until a chart page exists
    ready_timeout_s = 30.0       # chart page until TradingViewApi + bars are loaded
    close_timeout_s = 10.0       # graceful close before terminating (restart only)
    call_timeout_s = 10.0
    change_timeout_s = 20.0      # setSymbol / setResolution incl. loading history

    def __init__(self, app: Any, *, port: int | None = None, proc: Any = None) -> None:
        self.app = app
        self._port = port
        self.proc = proc or TvProcess()
        self._session: CdpSession | None = None
        self._target: CdpTarget | None = None
        self._connect_lock = asyncio.Lock()
        self._op_lock = asyncio.Lock()
        self.injections = 0                  # bundle injections (once per page load)
        self.last_bars_source = ""
        self.last_screenshot: dict[str, Any] = {}
        self.owner = Ownership()
        self._sam_set = ""                   # the symbol SAM itself set last (not learned as the user's feed)
        self._learned_seen = ""

    # -- connection -------------------------------------------------------------------------------------------
    @property
    def port(self) -> int:
        if self._port:
            return int(self._port)
        try:
            return int(self.app.config.get("trading.tv_port", 9222))
        except (TypeError, ValueError):
            return 9222

    @property
    def connected(self) -> bool:
        return self._session is not None and not self._session.closed

    async def connect(self, *, wait_ready_s: float | None = None) -> bool:
        """Attach to the chart page and wait for its API. True = ready to use.
        Never starts TradingView (that is ``ensure_running``)."""
        try:
            await self._ensure_session()
        except (CdpError, TvError) as exc:
            self._status("down", str(exc))
            return False
        ready = await self._wait_ready(self.ready_timeout_s if wait_ready_s is None else wait_ready_s)
        self._status("ok" if ready else "degraded", "connected" if ready else "chart is still loading")
        return ready

    async def close(self) -> None:
        """Close the DevTools socket only; TradingView keeps running."""
        await self._drop()

    async def _drop(self) -> None:
        session, self._session, self._target = self._session, None, None
        if session is not None:
            await session.close()

    async def _ensure_session(self) -> CdpSession:
        async with self._connect_lock:
            if self._session is not None and not self._session.closed:
                return self._session
            await self._drop()
            try:
                targets = await list_chart_targets(self.port)
            except CdpError as exc:
                raise TvError(str(exc), code="unavailable") from None
            if not targets:
                raise TvError("TradingView has no chart open", code="no_chart")
            self._target, self._session = await self._open_best(targets)
            return self._session

    async def _open_best(self, targets: list[CdpTarget]) -> tuple[CdpTarget, CdpSession]:
        """One chart tab: use it. Several: prefer the visible one."""
        fallback: tuple[CdpTarget, CdpSession] | None = None
        for target in targets:
            session = CdpSession(target.ws_url)
            try:
                await session.connect()
            except CdpError:
                continue
            if len(targets) == 1:
                return target, session
            try:
                visible = await session.evaluate(VISIBILITY_EXPRESSION, timeout_s=3) == "visible"
            except CdpError:
                visible = False
            if visible:
                if fallback is not None:
                    await fallback[1].close()
                return target, session
            if fallback is None:
                fallback = (target, session)
            else:
                await session.close()
        if fallback is None:
            raise TvError("could not open the TradingView chart page", code="unavailable")
        return fallback

    async def _inject(self, session: CdpSession) -> None:
        """Install ``window.__sam`` once per page load, after re-checking the host."""
        try:
            location = await session.evaluate(LOCATION_EXPRESSION, timeout_s=5)
            if not is_tradingview_url(str(location or "")):
                await self._drop()
                raise TvError("the TradingView window is not showing a chart page", code="foreign_page")
            result = await session.evaluate(BUNDLE, timeout_s=5)
        except CdpClosed:
            await self._drop()
            raise TvError("TradingView closed the connection", code="unavailable") from None
        except CdpError as exc:
            raise TvError(f"could not prepare the chart page: {exc}") from None
        if result == "foreign_page":
            raise TvError("the TradingView window is not showing a chart page", code="foreign_page")
        self.injections += 1

    async def _call(self, fn: str, *args: Any, timeout_s: float | None = None) -> Any:
        """Call ``window.__sam[fn](*args)``; reconnects once on socket loss and
        re-injects after a page reload."""
        expression = call_expression(fn, *args)
        timeout = timeout_s or self.call_timeout_s
        last: BaseException | None = None
        for _attempt in range(4):
            session = await self._ensure_session()
            try:
                value = await session.evaluate(expression, await_promise=True, timeout_s=timeout)
            except CdpClosed as exc:
                last = exc
                await self._drop()
                continue
            except CdpProtocolError as exc:
                if exc.context_lost:
                    last = exc
                    await asyncio.sleep(0.3)
                    continue
                raise TvError(f"TradingView refused {fn}: {exc}") from None
            except CdpJsError as exc:
                raise TvError(f"chart script error in {fn}: {exc}") from None
            except CdpTimeout:
                raise TvError(f"TradingView did not answer {fn} within {timeout:.0f} s", code="timeout") from None
            if isinstance(value, dict) and value.get("__sam_missing"):
                await self._inject(session)
                continue
            if isinstance(value, dict) and "__sam_error" in value:
                error = str(value["__sam_error"])[:300]
                if error == "foreign_page":
                    await self._drop()
                code = error if error in ("not_ready", "foreign_page") else "chart_error"
                raise TvError(f"{fn}: {error}", code=code)
            return value
        raise TvError(f"TradingView chart did not answer {fn} ({type(last).__name__ if last else 'no reply'})",
                      code="unavailable")

    async def _wait_ready(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            try:
                if await self._call("ready", timeout_s=5) is True:
                    return True
            except (TvError, CdpError):
                pass
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(self.poll_s)

    # -- start / restart --------------------------------------------------------------------------------------
    async def ensure_running(self, *, allow_restart: bool = False, confirm: Confirm | None = None,
                             focus: bool = True) -> dict[str, Any]:
        """Make TradingView reachable over the DevTools port.

        -> {"ok", "state": connected|started|restarted|needs_restart|not_installed|failed, "detail", "ms"}.
        Running WITHOUT the port: ``needs_restart`` unless ``allow_restart``;
        then ``confirm(CONFIRM_RESTART_CKB)`` is asked first when given (a
        declined/expired answer keeps TradingView untouched). ``focus`` brings
        its window to the front (the user asked to open/see it); pass False
        for background work.
        """
        began = time.perf_counter()
        try:
            result = await self._ensure_running(allow_restart, confirm, focus)
        except Exception as exc:  # noqa: BLE001 - reported honestly, never raised to the tool
            log.warning("ensure_running failed: %s", self._redact(f"{type(exc).__name__}: {exc}"))
            result = {"ok": False, "state": "failed", "detail": self._redact(str(exc))[:300]}
        result["ms"] = round((time.perf_counter() - began) * 1000.0, 1)
        self._record("ensure_running", result["ms"], state=result["state"])
        state_map = {"connected": "ok", "started": "ok", "restarted": "ok", "needs_restart": "degraded"}
        self._status(state_map.get(result["state"], "down"), result.get("detail", ""))
        return result

    async def _ensure_running(self, allow_restart: bool, confirm: Confirm | None, focus: bool) -> dict[str, Any]:
        version = await browser_version(self.port)
        if version is not None:
            if not is_tradingview_browser(version):
                return {"ok": False, "state": "failed",
                        "detail": f"port {self.port} is used by another application, not TradingView"}
            if await self._wait_for_chart(self.launch_timeout_s):
                if focus:
                    await self._focus()
                return {"ok": True, "state": "connected", "detail": ""}
            return {"ok": False, "state": "failed", "detail": "TradingView is running but no chart finished loading"}
        pids = await asyncio.to_thread(self.proc.running_pids)
        state = "started"
        if pids:
            if not allow_restart:
                return {"ok": False, "state": "needs_restart",
                        "detail": "TradingView is running without the local DevTools port; it must be restarted once"}
            if confirm is not None and not await confirm(CONFIRM_RESTART_CKB):
                return {"ok": False, "state": "needs_restart", "declined": True,
                        "detail": "the user did not approve restarting TradingView"}
            closed = await asyncio.to_thread(self.proc.close, pids, self.close_timeout_s)
            self._activity("tradingview_restart", True, f"closed TradingView to add the DevTools port: {closed}")
            state = "restarted"
        launched = await self._launch()
        if not launched["ok"]:
            return {"ok": False, "state": launched["state"], "detail": launched["detail"]}
        if not await self._wait_for_chart(self.launch_timeout_s):
            return {"ok": False, "state": "failed", "detail": "TradingView started but no chart finished loading"}
        return {"ok": True, "state": state, "detail": ""}

    async def _wait_for_chart(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if await self.connect(wait_ready_s=max(0.0, min(self.ready_timeout_s, remaining))):
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(self.poll_s)

    async def _launch(self) -> dict[str, Any]:
        arguments = f"--remote-debugging-port={self.port}"
        configured = str(self.app.config.get("trading.tv_aumid", "") or "")
        tried: list[str] = []
        for aumid in [configured] if configured else []:
            tried.append(aumid)
            try:
                await asyncio.to_thread(self.proc.activate, aumid, arguments)
                return {"ok": True, "aumid": aumid}
            except OSError as exc:
                log.info("activation of the configured TradingView AUMID failed: %s", type(exc).__name__)
        discovered = await asyncio.to_thread(self.proc.discover_aumids)
        for aumid in discovered:
            if aumid in tried:
                continue
            try:
                await asyncio.to_thread(self.proc.activate, aumid, arguments)
                return {"ok": True, "aumid": aumid}
            except OSError as exc:
                log.info("activation of a discovered TradingView package failed: %s", type(exc).__name__)
        if not discovered:
            return {"ok": False, "state": "not_installed", "detail": "TradingView Desktop is not installed"}
        return {"ok": False, "state": "failed", "detail": "Windows could not start TradingView"}

    async def _focus(self) -> bool:
        try:
            pids = await asyncio.to_thread(self.proc.running_pids)
            return bool(await asyncio.to_thread(self.proc.focus, pids))
        except Exception:  # noqa: BLE001 - focusing is a nicety
            log.debug("focus failed", exc_info=True)
            return False

    async def probe(self) -> dict[str, Any]:
        """Start-up check: attach when the port is already open; never launches or restarts."""
        if await browser_version(self.port) is not None and await self.connect(wait_ready_s=10):
            return {"ok": True, "state": "connected"}
        pids = await asyncio.to_thread(self.proc.running_pids)
        if pids:
            self._status("degraded", "TradingView runs without the DevTools port")
            return {"ok": False, "state": "needs_restart"}
        self._status("down", "TradingView is not running")
        return {"ok": False, "state": "not_running"}

    # -- chart ------------------------------------------------------------------------------------------------
    async def chart_state(self) -> dict[str, Any]:
        """{"symbol","canonical","timeframe","resolution","timeframe_ckb","description","visible_range",
        "visible_price_range","last_bar": Bar,"price","bar_count","studies" (names = untrusted text),
        "my_drawings","all_drawings","user_drawings" (counts),"tick","timezone","visible","loading","failed"}."""
        began = time.perf_counter()
        raw = await self._call("state")
        shape_ids = {str(s.get("id")) for s in raw.get("shapes") or []}
        self._learn(str(raw.get("symbol", "")))
        mine = await self._reconcile(self._owned_rows(), shape_ids, str(raw.get("symbol", "")),
                                     loading=bool(raw.get("loading")))
        my_count = sum(1 for r in mine if r["tv_id"] in shape_ids)
        resolution = str(raw.get("resolution", ""))
        last = _bar(raw.get("last_bar"))
        self._record("chart_state", (time.perf_counter() - began) * 1000.0)
        return {"symbol": raw.get("symbol", ""), "canonical": canonical_symbol(str(raw.get("symbol", ""))),
                "timeframe": from_tv_resolution(resolution), "resolution": resolution,
                "timeframe_ckb": resolution_label_ckb(resolution), "description": raw.get("description", ""),
                "visible_range": raw.get("visible_range"), "visible_price_range": raw.get("visible_price_range"),
                "last_bar": last, "price": last["close"] if last else None, "bar_count": int(raw.get("bar_count") or 0),
                "studies": list(raw.get("studies") or []), "my_drawings": my_count, "all_drawings": len(shape_ids),
                "user_drawings": len(shape_ids) - my_count, "tick": raw.get("tick"), "timezone": raw.get("timezone", ""),
                "visible": raw.get("visibility") == "visible", "loading": bool(raw.get("loading")),
                "failed": bool(raw.get("failed"))}

    async def set_symbol(self, symbol: str) -> dict[str, Any]:
        """Any alias ('گۆڵد', 'بیتکۆین', 'OANDA:XAUUSD') -> chart symbol, verified by reading it back.
        Gold stays on the user's own gold feed when the chart already shows gold."""
        async with self._op_lock:
            began = time.perf_counter()
            current = str((await self._call("state")).get("symbol", ""))
            target, reason = tv_symbol_for(symbol, current=current,
                                           overrides=self.app.config.get("trading.symbol_map", {}) or {},
                                           learned=self.app.config.get("trading.tv_learned_symbols", {}) or {})
            base = {"requested": symbol, "previous": current, "reason": reason}
            if target is None:
                return {**base, "ok": False, "changed": False, "error": "unknown_symbol", "symbol": current}
            if reason == "same":
                return {**base, "ok": True, "changed": False, "symbol": current, "canonical": canonical_symbol(current)}
            res = await self._call("setSymbol", target, int(self.change_timeout_s * 1000),
                                   timeout_s=self.change_timeout_s * 2 + 5)
            self._record("set_symbol", (time.perf_counter() - began) * 1000.0, ok=bool(res.get("ok")))
            now = str(res.get("symbol", ""))
            self.owner.note_symbol(now)
            self._sam_set = now
            if not res.get("ok"):
                return {**base, "ok": False, "changed": False, "error": "symbol_failed", "tried": target,
                        "detail": str(res.get("error", ""))[:200], "restored": bool(res.get("restored")), "symbol": now}
            verified = instrument_key(now) == instrument_key(target)
            return {**base, "ok": verified, "changed": True, "symbol": now, "canonical": canonical_symbol(now),
                    "tried": target, "resolution": res.get("resolution"),
                    **({} if verified else {"error": "symbol_mismatch"})}

    async def set_timeframe(self, timeframe: str) -> dict[str, Any]:
        """'١٥ خولەک' / 'H1' / 'ڕۆژانە' -> chart resolution, verified by reading it back."""
        resolution = parse_tv_resolution(timeframe)
        if resolution is None:
            return {"ok": False, "changed": False, "error": "unknown_timeframe", "requested": timeframe}
        async with self._op_lock:
            began = time.perf_counter()
            res = await self._call("setResolution", resolution, int(self.change_timeout_s * 1000),
                                   timeout_s=self.change_timeout_s * 2 + 5)
            self._record("set_timeframe", (time.perf_counter() - began) * 1000.0, ok=bool(res.get("ok")))
        now = str(res.get("resolution", ""))
        base = {"requested": timeframe, "resolution": now, "timeframe": from_tv_resolution(now),
                "timeframe_ckb": resolution_label_ckb(now), "previous": res.get("before")}
        if not res.get("ok"):
            return {**base, "ok": False, "changed": False, "error": "timeframe_failed", "tried": resolution,
                    "detail": str(res.get("error", ""))[:200], "restored": bool(res.get("restored"))}
        verified = same_resolution(now, resolution)
        return {**base, "ok": verified, "changed": not same_resolution(str(res.get("before", "")), now),
                **({} if verified else {"error": "timeframe_mismatch"})}

    async def bars(self, count: int = 500) -> list[Bar]:
        """The chart's own loaded bars, oldest -> newest, UTC seconds (last bar may still be forming)."""
        count = max(1, min(int(count), 5000))
        began = time.perf_counter()
        raw = await self._call("bars", count, timeout_s=15)
        self.last_bars_source = str(raw.get("source", ""))
        out = [bar for bar in (_bar(r) for r in raw.get("rows") or []) if bar is not None]
        self._record("bars", (time.perf_counter() - began) * 1000.0, n=len(out), source=self.last_bars_source)
        return out

    # -- drawings ---------------------------------------------------------------------------------------------
    async def draw(self, kind: str, points: list[dict[str, Any]], *, text: str = "", color: str | None = None,
                   style: dict[str, Any] | None = None, tag: str = "") -> dict[str, Any]:
        """One drawing -> {"ok","id","db_id"} (+ "error")."""
        result = await self.draw_many([{"kind": kind, "points": list(points), "text": text, "color": color,
                                        "style": style}], tag=tag)
        if result["drawn"]:
            return {"ok": True, "id": result["ids"][0], "db_id": result["db_ids"][0]}
        errors = result.get("errors") or [{"error": "not drawn"}]
        return {"ok": False, "id": None, "db_id": None, "error": errors[0]["error"]}

    async def draw_many(self, items: list[dict[str, Any]], *, tag: str = "") -> dict[str, Any]:
        """Validate + draw -> {"ok","drawn","ids","db_ids","kinds","errors":[{"index","error"}],"symbol",
        "resolution","ms"}. ``ok`` = at least one drawn; per-item failures are listed honestly."""
        errors: list[dict[str, Any]] = []
        if not isinstance(items, list) or not items:
            return {"ok": False, "drawn": 0, "ids": [], "db_ids": [], "kinds": [],
                    "errors": [{"index": None, "error": "no items to draw"}]}
        if len(items) > MAX_ITEMS:
            errors.append({"index": None, "error": f"only the first {MAX_ITEMS} items were drawn"})
            items = items[:MAX_ITEMS]
        async with self._op_lock:
            began = time.perf_counter()
            state = await self._call("state")
            last_price = (state.get("last_bar") or {}).get("c")
            specs: list[tuple[int, dict[str, Any]]] = []
            for index, item in enumerate(items):
                spec, error = normalize_item(item, last_price=last_price)
                if error:
                    errors.append({"index": index, "error": error})
                else:
                    specs.append((index, spec))  # type: ignore[arg-type]
            if not specs:
                return {"ok": False, "drawn": 0, "ids": [], "db_ids": [], "kinds": [], "errors": errors,
                        "symbol": state.get("symbol")}
            page_specs = [{k: s[k] for k in ("kind", "points", "text", "lock", "overrides", "position") if k in s}
                          for _, s in specs]
            res = await self._call("draw", page_specs, timeout_s=30)
            symbol = str(res.get("symbol", ""))
            resolution = str(res.get("resolution", ""))
            timeframe = from_tv_resolution(resolution) or resolution
            clean = clean_tag(tag)
            ids: list[str] = []
            db_ids: list[int] = []
            kinds: list[str] = []
            for (index, spec), outcome in zip(specs, res.get("results") or []):
                if not outcome.get("ok") or not outcome.get("id"):
                    errors.append({"index": index, "error": str(outcome.get("error", "not drawn"))[:200]})
                    continue
                db_ids.append(self.app.db.insert("drawings", {
                    "tv_id": str(outcome["id"]), "kind": spec["kind"], "symbol": symbol, "timeframe": timeframe,
                    "points": outcome.get("points") or spec["points"], "text": spec["text"], "tag": clean,
                    "created_at": time.time()}))
                ids.append(str(outcome["id"]))
                kinds.append(spec["kind"])
            ms = (time.perf_counter() - began) * 1000.0
            self._record("draw", ms, n=len(ids), failed=len(errors))
        return {"ok": bool(ids), "drawn": len(ids), "ids": ids, "db_ids": db_ids, "kinds": kinds, "errors": errors,
                "symbol": symbol, "resolution": resolution, "tag": clean, "ms": round(ms, 1)}

    async def my_drawings(self, *, symbol: str | None = None, prune: bool = True) -> list[dict[str, Any]]:
        """SAM-owned drawings not yet removed (optionally for one instrument, any alias)."""
        rows = self._owned_rows()
        if prune and self.connected:
            try:
                info = await self._call("shapes")
                rows = await self._reconcile(rows, {str(s.get("id")) for s in info.get("shapes") or []},
                                             str(info.get("symbol")))
            except TvError:
                pass
        if symbol:
            key = instrument_key(symbol)
            rows = [r for r in rows if instrument_key(r["symbol"]) == key]
        return rows

    async def clear(self, *, tag: str | None = None) -> dict[str, Any]:
        """Remove SAM's own drawings -> {"removed","pruned","other_symbol","failed","remaining_on_chart"}.
        Only ids from the ``drawings`` table are ever sent to the page."""
        async with self._op_lock:
            rows = self._owned_rows(tag)
            if not rows:
                return {"removed": 0, "pruned": 0, "other_symbol": 0, "failed": 0}
            info = await self._call("shapes")
            symbol = str(info.get("symbol", ""))
            present = {str(s.get("id")) for s in info.get("shapes") or []}
            rows, present = await self._adopt_before_clear(rows, present, symbol)
            to_remove = [r for r in rows if r["tv_id"] in present]
            gone = [r for r in rows if r["tv_id"] not in present and r["symbol"] == symbol]
            other = [r for r in rows if r["tv_id"] not in present and r["symbol"] != symbol]
            outcome = {"removed": [], "failed": [], "missing": [], "remaining": len(present)}
            if to_remove:
                outcome = await self._call("remove", [r["tv_id"] for r in to_remove])
            finished = {str(i) for i in outcome.get("removed", []) + outcome.get("missing", [])}
            self._mark_removed([r["db_id"] for r in to_remove if r["tv_id"] in finished] + [r["db_id"] for r in gone])
        return {"removed": len(outcome.get("removed", [])), "pruned": len(gone) + len(outcome.get("missing", [])),
                "other_symbol": len(other), "failed": len(outcome.get("failed", [])),
                "remaining_on_chart": outcome.get("remaining")}

    async def clear_my_drawings(self, *, tag: str | None = None) -> int:
        """Contract form of ``clear``: the number of SAM drawings removed from the chart."""
        return int((await self.clear(tag=tag))["removed"])

    def _owned_rows(self, tag: str | None = None) -> list[dict[str, Any]]:
        rows = self.app.db.query("SELECT id, tv_id, kind, symbol, timeframe, points, text, tag, created_at "
                                 "FROM drawings WHERE removed_at IS NULL ORDER BY id")
        out = []
        for row in rows:
            if tag and not (row["tag"] == tag or row["tag"].startswith(tag + ":")):
                continue
            try:
                points = json.loads(row["points"]) if row["points"] else []
            except ValueError:
                points = []
            out.append({"db_id": row["id"], "tv_id": str(row["tv_id"]), "kind": row["kind"], "symbol": row["symbol"],
                        "timeframe": row["timeframe"], "points": points, "text": row["text"], "tag": row["tag"],
                        "created_at": row["created_at"]})
        return out

    async def _reconcile(self, rows: list[dict[str, Any]], present: set[str], symbol: str, *,
                         loading: bool = False) -> list[dict[str, Any]]:
        """Owned rows as they are now: re-adopt drawings re-created under a new
        id, forget one the user deleted by hand only after repeated misses
        (never right after a symbol change or while loading; tv_owner.py)."""
        missing, forget = self.owner.review(rows, present, symbol, loading=loading)
        if missing:
            adopted = await self._readopt(rows, missing)
            forget = [r for r in forget if r["db_id"] not in adopted]
        self._mark_removed([r["db_id"] for r in forget])
        gone = {r["db_id"] for r in forget}
        return [r for r in rows if r["db_id"] not in gone]

    async def _readopt(self, rows: list[dict[str, Any]], missing: list[dict[str, Any]]) -> dict[int, str]:
        try:
            info = await self._call("shapesDetailed")
        except TvError:
            return {}
        matches = match_shapes(missing, list(info.get("shapes") or []), {r["tv_id"] for r in rows})
        for row in rows:
            new_id = matches.get(row["db_id"])
            if new_id:
                self.app.db.execute("UPDATE drawings SET tv_id=? WHERE id=?", (new_id, row["db_id"]))
                row["tv_id"] = new_id
                self.owner.forget_now(row["db_id"])
        return matches

    async def _adopt_before_clear(self, rows: list[dict[str, Any]], present: set[str],
                                  symbol: str) -> tuple[list[dict[str, Any]], set[str]]:
        """Before removing: find re-created drawings, and right after a symbol
        change give TradingView up to 3 s to load the drawings of this symbol."""
        deadline = time.monotonic() + (3.0 if self.owner.settling() else 0.0)
        while True:
            missing = [r for r in rows if r["tv_id"] not in present and r["symbol"] == symbol]
            if not missing:
                break
            await self._readopt(rows, missing)
            info = await self._call("shapes")
            present = {str(s.get("id")) for s in info.get("shapes") or []}
            if all(r["tv_id"] in present for r in missing) or time.monotonic() >= deadline:
                break
            await asyncio.sleep(0.3)
        return rows, present

    def _learn(self, symbol: str) -> None:
        """Remember the feed the user's chart shows for an instrument (his gold is
        PEPPERSTONE:XAUUSD): ``tv_symbol_for`` then keeps it when SAM switches back."""
        if not symbol or ":" not in symbol or symbol == self._learned_seen or symbol == self._sam_set:
            return
        self._learned_seen = symbol
        canonical = resolve_instrument(symbol)
        if not canonical:
            return
        learned = dict(self.app.config.get("trading.tv_learned_symbols", {}) or {})
        if learned.get(canonical) != symbol:
            learned[canonical] = symbol
            self.app.config.set("trading.tv_learned_symbols", learned)

    def _mark_removed(self, db_ids: list[int]) -> None:
        if db_ids:
            now = time.time()
            self.app.db.executemany("UPDATE drawings SET removed_at=? WHERE id=?", [(now, i) for i in db_ids])

    # -- screenshot / raw -------------------------------------------------------------------------------------
    async def screenshot(self, *, fmt: str = "jpeg", max_width: int = 1440, clip_chart: bool = True) -> bytes:
        """Chart picture via Page.captureScreenshot (no focus needed), clipped to the chart
        canvas (excludes watchlist/trading panels) and scaled to <= max_width pixels."""
        fmt = "png" if fmt == "png" else "jpeg"
        began = time.perf_counter()
        rect = await self._call("chartRect")
        session = await self._ensure_session()
        dpr = float(rect.get("dpr") or 1.0)
        inner = rect.get("inner") or [rect.get("width"), rect.get("height")]
        x, y, width, height = ((rect["x"], rect["y"], rect["width"], rect["height"])
                               if clip_chart and rect.get("found") else (0, 0, inner[0], inner[1]))
        # Output width = clip width x scale x devicePixelRatio (measured: 1541 CSS px at dpr 1.75 -> 2697 px).
        scale = min(1.0, float(max_width) / max(1.0, float(width) * dpr)) if max_width else 1.0
        params: dict[str, Any] = {"format": fmt, "fromSurface": True, "captureBeyondViewport": False,
                                  "clip": {"x": x, "y": y, "width": width, "height": height, "scale": scale}}
        if fmt == "jpeg":
            params["quality"] = 80
        # A minimized window stops rendering (document hidden): fail fast instead of waiting 15 s.
        hidden = rect.get("visibility") != "visible"
        try:
            result = await session.call("Page.captureScreenshot", params, timeout_s=6 if hidden else 15)
        except CdpTimeout:
            raise TvError("TradingView did not render a picture (is its window minimized?)", code="timeout") from None
        except CdpError as exc:
            raise TvError(f"chart screenshot failed: {exc}") from None
        data = base64.b64decode(result.get("data") or b"")
        ms = (time.perf_counter() - began) * 1000.0
        self.last_screenshot = {"bytes": len(data), "ms": round(ms, 1), "scale": round(scale, 4),
                                "visible": rect.get("visibility") == "visible"}
        self._record("screenshot", ms, bytes=len(data))
        return data

    async def evaluate(self, js: str, *, await_promise: bool = False, timeout_s: float = 10) -> Any:
        """Raw evaluation for internal callers only (no tool exposes it); still host-checked."""
        session = await self._ensure_session()
        location = await session.evaluate(LOCATION_EXPRESSION, timeout_s=5)
        if not is_tradingview_url(str(location or "")):
            raise TvError("the TradingView window is not showing a chart page", code="foreign_page")
        return await session.evaluate(js, await_promise=await_promise, timeout_s=timeout_s)

    # -- plumbing ---------------------------------------------------------------------------------------------
    def _record(self, op: str, ms: float, **extra: Any) -> None:
        try:
            self.app.timing.record("tv_cdp", ms, kind="trading", op=op, **extra)
        except Exception:  # noqa: BLE001
            log.debug("timing failed", exc_info=True)

    def _status(self, state: str, detail: str = "") -> None:
        try:
            self.app.publish_status("tradingview", state, self._redact(detail)[:200])
        except Exception:  # noqa: BLE001
            log.debug("status publish failed", exc_info=True)

    def _activity(self, name: str, ok: bool, summary: str) -> None:
        try:
            self.app.db.log_activity("system", name, ok=ok, summary=self._redact(summary)[:500], source="tradingview")
        except Exception:  # noqa: BLE001
            log.debug("activity log failed", exc_info=True)

    def _redact(self, text: str) -> str:
        try:
            return self.app.redact(text)
        except Exception:  # noqa: BLE001
            return text

    def status(self) -> dict[str, Any]:
        return {"connected": self.connected, "port": self.port, "injections": self.injections,
                "bars_source": self.last_bars_source}


def _bar(raw: Any) -> Bar | None:
    """Page row {t,o,h,l,c,v} -> Bar (None when malformed)."""
    if not isinstance(raw, dict):
        return None
    try:
        return Bar(time=int(raw["t"]), open=float(raw["o"]), high=float(raw["h"]), low=float(raw["l"]),
                   close=float(raw["c"]), volume=float(raw.get("v") or 0.0))
    except (KeyError, TypeError, ValueError):
        return None


__all__ = ["TradingViewBridge", "TvError", "CONFIRM_RESTART_CKB", "CdpUnavailable"]
