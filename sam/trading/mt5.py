"""Read-only MetaTrader 5 feed: one persistent session on a dedicated thread.

Measured on this PC (2026-09-24, reports/trading-intelligence.json):
- v1 called ``initialize()``/``shutdown()`` around every request: 92 ms median,
  160 ms max per call. A persistent session answers ``symbol_info_tick`` in
  0.03 ms, so SAM 2 keeps ONE session and reconnects only on IPC errors.
- MT5 bar and tick times are broker SERVER time (UTC+3 today, UTC+2 in
  winter), not UTC. v1 treated them as UTC, flagged every fetch as "future"
  and blocked every setup. The offset is computed per connection as
  ``round((tick.time - time.time()) / 900) * 900`` and removed here, so
  everything that leaves this module is true UTC. The session is persistent,
  so the offset is also re-verified from live ticks every few minutes (and
  sooner when ticks look an hour off): a SAM left running across the broker's
  DST switch (UTC+3 -> UTC+2, US DST weekend) otherwise saw every bar as an
  hour old -- stale data, WAIT verdicts, nothing drawn (repair review dst.py).
  A new value is accepted only when verified (a live tick within 60 s of a
  900 s multiple), so a market frozen since Friday never changes it.
- The broker lists gold as XAUUSD, XAUUSD.crp and XAUUSD.m.e; TradingView
  names (TVC:GOLD, OANDA:XAUUSD) map through ``common.canonical_symbol`` and the
  user override ``trading.symbol_map``.

The MetaTrader5 package is not thread-safe and is synchronous, so every call
runs on the single ``sam-mt5`` thread. It is imported lazily there (it pulls
numpy, ~2 s here). SAM 2 NEVER trades: the module is wrapped in
:class:`ReadOnlyMT5`, which only exposes the market-data functions below.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import ctypes
import logging
import os
import time
from typing import Any, Callable, Iterable

from .common import canonical_symbol, normalize_timeframe

log = logging.getLogger("sam.trading.mt5")

# The only MetaTrader5 functions SAM may call. Order, position and account-
# changing functions are deliberately absent (docs/CONTRACTS.md: MT5 is READ-ONLY).
READ_ONLY_FUNCTIONS = frozenset({
    "initialize", "shutdown", "last_error", "terminal_info", "version", "symbol_info", "symbol_info_tick",
    "symbols_get", "symbols_total", "symbol_select", "copy_rates_from_pos", "copy_rates_from", "copy_rates_range",
    "copy_ticks_from", "copy_ticks_range",
})
# IPC failures of the terminal link (MetaTrader5 docs, "last_error" codes):
# -10001 send failed, -10002 receive failed, -10003 init failed, -10004 no IPC, -10005 timeout.
IPC_ERRORS = {-10001, -10002, -10003, -10004, -10005}
MAX_OFFSET_S = 14 * 3600
OFFSET_TOLERANCE_S = 60.0
OFFSET_RECHECK_S = 300.0        # re-verify the broker offset this often in a long session
OFFSET_DRIFT_RECHECK_S = 60.0   # ... or at most this often while ticks look off by >= 15 min
RECONNECT_BACKOFF_S = 20.0
# Liquid symbols whose newest tick tells the server clock (crypto trades at weekends).
CLOCK_SYMBOLS = ("XAUUSD", "EURUSD", "BTCUSD", "ETHUSD", "GBPUSD", "USDJPY")


class MT5Error(RuntimeError):
    """Feed problem with a short, key-free message."""


class ReadOnlyMT5:
    """Proxy around the MetaTrader5 module exposing constants and read-only
    functions only; anything else raises (so a future bug cannot trade)."""

    def __init__(self, module: Any) -> None:
        self._module = module

    def __getattr__(self, name: str) -> Any:
        if name in READ_ONLY_FUNCTIONS or name.isupper():
            return getattr(self._module, name)
        raise AttributeError(f"MetaTrader5.{name} is blocked: SAM 2 uses MetaTrader 5 read-only")


def compute_broker_offset(tick_times: Iterable[float], now: float,
                          tolerance_s: float = OFFSET_TOLERANCE_S) -> tuple[int | None, bool]:
    """(offset_s, verified) from the newest server tick time vs the local UTC clock.

    Broker offsets are whole quarter hours, so the raw difference is rounded to
    900 s. It is "verified" only when the raw difference sits within
    ``tolerance_s`` of that multiple -- true for a live tick, but not for a tick
    frozen since Friday's close (then the caller falls back to the last
    verified value instead of trusting a wrong offset).
    """
    times = [float(t) for t in tick_times if t]
    if not times:
        return None, False
    raw = max(times) - now
    offset = int(round(raw / 900.0) * 900)
    if abs(offset) > MAX_OFFSET_S:
        return None, False
    return offset, abs(raw - offset) <= tolerance_s


def rank_broker_symbols(names: Iterable[tuple[str, bool]], canonical: str) -> list[str]:
    """Order broker symbol candidates for a canonical name: exact, then
    prefix matches (XAUUSD.m.e), visible first, shortest suffix first."""
    base = canonical.upper()

    def score(item: tuple[str, bool]) -> tuple[int, int, int, int, str]:
        name, visible = item
        upper = name.upper()
        return (0 if upper == base else 1, 0 if upper.startswith(base) else 1, 0 if visible else 1, len(upper), upper)

    return [name for name, _ in sorted(names, key=score) if base in name.upper()]


def terminal_running() -> bool:
    """True when a MetaTrader 5 terminal process exists. ``initialize()`` would
    otherwise LAUNCH the terminal, which SAM must not do behind the user's back."""
    if os.name != "nt":
        return True
    try:
        psapi, kernel32 = ctypes.windll.psapi, ctypes.windll.kernel32  # type: ignore[attr-defined]
        pids = (ctypes.c_ulong * 4096)()
        needed = ctypes.c_ulong()
        if not psapi.EnumProcesses(ctypes.byref(pids), ctypes.sizeof(pids), ctypes.byref(needed)):
            return True
        buffer = ctypes.create_unicode_buffer(1024)
        for pid in pids[: needed.value // ctypes.sizeof(ctypes.c_ulong)]:
            handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not handle:
                continue
            try:
                size = ctypes.c_ulong(len(buffer))
                if kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
                    if os.path.basename(buffer.value).lower() in ("terminal64.exe", "terminal.exe"):
                        return True
            finally:
                kernel32.CloseHandle(handle)
        return False
    except Exception:  # noqa: BLE001 - never block the feed on the process check
        return True


def _load_module() -> Any:
    import MetaTrader5  # lazy: pulls numpy (~2 s on this PC)
    return MetaTrader5


class MT5Feed:
    """``app.trading.mt5`` (docs/CONTRACTS.md 3.5)."""

    def __init__(self, app: Any = None, *, module_loader: Callable[[], Any] = _load_module,
                 is_terminal_running: Callable[[], bool] = terminal_running,
                 clock: Callable[[], float] = time.time) -> None:
        self.app = app
        self._loader = module_loader
        self._is_running = is_terminal_running
        self._clock = clock
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="sam-mt5")
        self._mt5: ReadOnlyMT5 | None = None
        self.connected = False
        self.broker_offset_s = 0
        self.offset_verified = False
        self.offset_source = "none"
        self.last_error = ""
        self._symbols: dict[str, str] = {}
        self._info: dict[str, dict[str, Any]] = {}
        self._next_attempt = 0.0
        self.terminal_connected: bool | None = None
        self._offset_checked_at = 0.0
        self._offset_changed = False

    # -- threading ------------------------------------------------------------------
    async def _run(self, fn: Callable[..., Any], *args: Any) -> Any:
        return await asyncio.wrap_future(self._executor.submit(fn, *args))

    def _setting(self, key: str, default: Any) -> Any:
        try:
            return self.app.config.get(key, default) if self.app is not None else default
        except Exception:  # noqa: BLE001
            return default

    # -- connection (runs on the sam-mt5 thread) --------------------------------------
    def _connect_sync(self, force: bool = False) -> bool:
        if self.connected and not force:
            return True
        now = self._clock()
        if not force and now < self._next_attempt:
            return False
        self._next_attempt = now + RECONNECT_BACKOFF_S
        if not self._is_running():
            self.connected, self.last_error = False, "MetaTrader 5 is not running"
            return False
        try:
            mt5 = self._mt5 or ReadOnlyMT5(self._loader())
        except (ImportError, OSError) as exc:
            self.connected, self.last_error = False, f"MetaTrader5 package unavailable ({type(exc).__name__})"
            return False
        self._mt5 = mt5
        if self.connected:
            try:
                mt5.shutdown()
            except Exception:  # noqa: BLE001
                pass
        if not mt5.initialize():
            code = (mt5.last_error() or (0, ""))[0]
            self.connected, self.last_error = False, f"initialize failed ({code})"
            return False
        self.connected, self.last_error = True, ""
        self._symbols.clear()
        self._info.clear()
        terminal = mt5.terminal_info()
        self.terminal_connected = bool(getattr(terminal, "connected", False)) if terminal is not None else None
        self._compute_offset_sync()
        return True

    def _compute_offset_sync(self) -> None:
        assert self._mt5 is not None
        times = []
        for canonical in CLOCK_SYMBOLS:
            name = self._resolve_sync(canonical)
            tick = self._mt5.symbol_info_tick(name) if name else None
            if tick is not None and getattr(tick, "time", 0):
                times.append(float(tick.time_msc) / 1000.0 if getattr(tick, "time_msc", 0) else float(tick.time))
        offset, verified = compute_broker_offset(times, self._clock())
        self._offset_checked_at = self._clock()
        saved = self._setting("trading.mt5_offset_s", None)
        if offset is not None and verified:
            self.broker_offset_s, self.offset_verified, self.offset_source = offset, True, "tick"
        elif isinstance(saved, (int, float)):
            # Market closed (ticks frozen): the last verified offset is right.
            self.broker_offset_s, self.offset_verified, self.offset_source = int(saved), False, "saved"
        elif offset is not None:
            self.broker_offset_s, self.offset_verified, self.offset_source = offset, False, "unverified-tick"
        else:
            self.broker_offset_s, self.offset_verified, self.offset_source = 0, False, "none"

    def _recheck_offset_sync(self, server_time: float) -> None:
        """Re-verify the offset in a long-running session (see the module doc)."""
        now = self._clock()
        drift = abs(server_time - self.broker_offset_s - now)
        wait = OFFSET_DRIFT_RECHECK_S if drift >= 900 - OFFSET_TOLERANCE_S else OFFSET_RECHECK_S
        if now - self._offset_checked_at < wait or self._mt5 is None:
            return
        self._offset_checked_at = now
        times = []
        for canonical in CLOCK_SYMBOLS:
            name = self._resolve_sync(canonical)
            tick = self._mt5.symbol_info_tick(name) if name else None
            if tick is not None and getattr(tick, "time", 0):
                times.append(float(tick.time_msc) / 1000.0 if getattr(tick, "time_msc", 0) else float(tick.time))
        offset, verified = compute_broker_offset(times, now)
        if offset is not None and verified and offset != self.broker_offset_s:
            log.info("broker offset changed %+d -> %+d s (verified from live ticks)", self.broker_offset_s, offset)
            self.broker_offset_s, self.offset_verified, self.offset_source = offset, True, "tick"
            self._offset_changed = True

    def _call_sync(self, fn: Callable[[ReadOnlyMT5], Any]) -> Any:
        """Run ``fn(mt5)`` with one reconnect on an IPC failure."""
        if not self._connect_sync():
            raise MT5Error(self.last_error or "MetaTrader 5 is not connected")
        assert self._mt5 is not None
        result = fn(self._mt5)
        if result is None:
            code = (self._mt5.last_error() or (0, ""))[0]
            if code in IPC_ERRORS and self._connect_sync(force=True):
                result = fn(self._mt5)
        return result

    # -- symbols ----------------------------------------------------------------------
    def _resolve_sync(self, symbol: str) -> str | None:
        assert self._mt5 is not None
        raw = (symbol or "").strip()
        canonical = canonical_symbol(raw)
        if not canonical:
            return None
        bare = raw.split(":", 1)[-1]
        if "." in bare and bare in self._info:
            return bare
        if "." in bare:  # an explicit broker name such as XAUUSD.m.e wins over the canonical mapping
            info = self._mt5.symbol_info(bare)
            if info is not None:
                self._remember(canonical, info, cache=False)
                return str(info.name)
        if canonical in self._symbols:
            return self._symbols[canonical]
        overrides = self._setting("trading.symbol_map", {}) or {}
        candidates: list[str] = []
        override = (overrides.get(canonical) or {}).get("mt5") if isinstance(overrides.get(canonical), dict) else None
        if override:
            candidates.append(str(override))
        candidates.append(canonical)
        for name in candidates:
            info = self._mt5.symbol_info(name)
            if info is not None:
                return self._remember(canonical, info)
        found = list(self._mt5.symbols_get(f"*{canonical}*") or [])
        ranked = rank_broker_symbols([(str(i.name), bool(getattr(i, "visible", False))) for i in found], canonical)
        if not ranked:
            return None
        return self._remember(canonical, next(i for i in found if str(i.name) == ranked[0]))

    def _remember(self, canonical: str, info: Any, *, cache: bool = True) -> str:
        name = str(info.name)
        if not bool(getattr(info, "visible", True)):
            self._mt5.symbol_select(name, True)  # type: ignore[union-attr] # Market Watch only; no trading
        if cache:
            self._symbols[canonical] = name
        self._info[name] = {"digits": int(getattr(info, "digits", 2) or 2),
                            "point": float(getattr(info, "point", 0.01) or 0.01)}
        return name

    # -- data ---------------------------------------------------------------------------
    def _bars_sync(self, symbol: str, timeframe: str, count: int) -> list[dict[str, Any]]:
        if not self._connect_sync():
            raise MT5Error(self.last_error or "MetaTrader 5 is not connected")
        name = self._resolve_sync(symbol)
        if name is None:
            raise MT5Error(f"no MetaTrader symbol matches {symbol}")
        tf = normalize_timeframe(timeframe)
        constant = getattr(self._mt5, f"TIMEFRAME_{tf}", None) if tf else None
        if constant is None:
            raise MT5Error(f"unsupported timeframe {timeframe}")
        count = max(10, min(10_000, int(count)))
        rates = self._call_sync(lambda m: m.copy_rates_from_pos(name, constant, 0, count))
        if rates is None or len(rates) == 0:
            raise MT5Error(f"no {tf} bars for {name}")
        live = self._mt5.symbol_info_tick(name) if self._mt5 is not None else None
        if live is not None and getattr(live, "time", 0):
            self._recheck_offset_sync(float(live.time_msc) / 1000.0 if getattr(live, "time_msc", 0) else float(live.time))
        offset = self.broker_offset_s
        names = getattr(getattr(rates, "dtype", None), "names", None) or ()
        times, opens, highs, lows, closes = (rates[k].tolist() for k in ("time", "open", "high", "low", "close"))
        ticks = rates["tick_volume"].tolist()
        reals = rates["real_volume"].tolist() if "real_volume" in names else [0] * len(times)
        return [{"time": int(t) - offset, "open": float(o), "high": float(h), "low": float(lo), "close": float(c),
                 "volume": float(rv if rv else tv)}
                for t, o, h, lo, c, tv, rv in zip(times, opens, highs, lows, closes, ticks, reals)]

    def _tick_sync(self, symbol: str) -> dict[str, Any]:
        if not self._connect_sync():
            raise MT5Error(self.last_error or "MetaTrader 5 is not connected")
        name = self._resolve_sync(symbol)
        if name is None:
            raise MT5Error(f"no MetaTrader symbol matches {symbol}")
        tick = self._call_sync(lambda m: m.symbol_info_tick(name))
        if tick is None or not getattr(tick, "time", 0):
            raise MT5Error(f"no tick for {name}")
        server = float(tick.time_msc) / 1000.0 if getattr(tick, "time_msc", 0) else float(tick.time)
        self._recheck_offset_sync(server)
        bid, ask = float(tick.bid or 0.0), float(tick.ask or 0.0)
        info = self._info.get(name, {})
        return {"symbol": name, "canonical": canonical_symbol(symbol), "bid": bid or None, "ask": ask or None,
                "last": float(getattr(tick, "last", 0.0) or 0.0) or None,
                "spread": round(ask - bid, 10) if bid and ask else None, "time": server - self.broker_offset_s,
                "volume": float(getattr(tick, "volume", 0) or 0), "digits": info.get("digits"),
                "point": info.get("point")}

    def _status_sync(self) -> dict[str, Any]:
        ok = self._connect_sync()
        symbols: list[str] = []
        if ok and self._mt5 is not None:
            symbols = sorted(str(i.name) for i in (self._mt5.symbols_get("*XAU*") or []))[:12]
        return {"connected": ok, "broker_offset_s": self.broker_offset_s, "server_time_ok": self.offset_verified,
                "offset_source": self.offset_source, "terminal_connected": self.terminal_connected,
                "symbols": symbols, "resolved": dict(self._symbols), "error": self.last_error or None}

    # -- async API (docs/CONTRACTS.md) --------------------------------------------------
    async def connect(self) -> bool:
        ok = await self._run(self._connect_sync, True)
        if self.app is not None:
            if ok and self.offset_verified:
                try:
                    if self.app.config.get("trading.mt5_offset_s") != self.broker_offset_s:
                        self.app.config.set("trading.mt5_offset_s", self.broker_offset_s)
                except Exception:  # noqa: BLE001
                    log.exception("could not persist the broker offset")
            detail = (f"offset {self.broker_offset_s:+d}s ({self.offset_source})" if ok else self.last_error)
            self.app.publish_status("mt5", "ok" if ok and self.offset_verified else "degraded" if ok else "down", detail)
        return ok

    async def status(self) -> dict[str, Any]:
        return await self._run(self._status_sync)

    async def resolve_symbol(self, symbol: str) -> str | None:
        def work() -> str | None:
            return self._resolve_sync(symbol) if self._connect_sync() else None
        return await self._run(work)

    async def bars(self, symbol: str, timeframe: str, count: int = 500) -> list[dict[str, Any]]:
        result = await self._run(self._bars_sync, symbol, timeframe, count)
        self._persist_offset()
        return result

    async def tick(self, symbol: str) -> dict[str, Any]:
        result = await self._run(self._tick_sync, symbol)
        self._persist_offset()
        return result

    def _persist_offset(self) -> None:
        """On the core loop (config.set publishes SettingsChanged)."""
        if not self._offset_changed or self.app is None:
            return
        self._offset_changed = False
        try:
            self.app.config.set("trading.mt5_offset_s", self.broker_offset_s)
            self.app.publish_status("mt5", "ok", f"offset {self.broker_offset_s:+d}s (re-verified)")
        except Exception:  # noqa: BLE001
            log.exception("could not persist the broker offset")

    async def symbol_meta(self, symbol: str) -> dict[str, Any]:
        """{"name", "digits", "point"} of the broker symbol (for rounding)."""
        def work() -> dict[str, Any]:
            if not self._connect_sync():
                raise MT5Error(self.last_error or "MetaTrader 5 is not connected")
            name = self._resolve_sync(symbol)
            if name is None:
                raise MT5Error(f"no MetaTrader symbol matches {symbol}")
            return {"name": name, **self._info.get(name, {})}
        return await self._run(work)

    async def close(self) -> None:
        def shutdown() -> None:
            if self._mt5 is not None and self.connected:
                try:
                    self._mt5.shutdown()
                except Exception:  # noqa: BLE001
                    pass
            self.connected = False
        try:
            await asyncio.wait_for(self._run(shutdown), 5.0)
        except Exception:  # noqa: BLE001 - stop() must never raise
            pass
        self._executor.shutdown(wait=False, cancel_futures=True)


__all__ = ["MT5Feed", "MT5Error", "ReadOnlyMT5", "compute_broker_offset", "rank_broker_symbols",
           "terminal_running", "READ_ONLY_FUNCTIONS"]
