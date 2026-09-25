"""Shared fakes for the trading-engine tests: synthetic UTC bars, a fake MT5
feed, a fake TradingView bridge and a fake MetaTrader5 module (no terminal,
no network, no chart)."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from sam.trading.engine.types import Candle

TF_SECONDS = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600, "H4": 14400, "D1": 86400}
BASE = datetime(2025, 1, 6, 0, 0, tzinfo=UTC)


def v1_candles(count: int = 180, *, slope: float = 0.03, wave: float = 1.6, step_min: int = 1) -> list[Candle]:
    """v1's test series (tests/test_trading.py ``candles``)."""
    result = []
    for index in range(count):
        middle = 100 + index * slope + math.sin(index / 4) * wave
        opening = middle - math.sin(index / 3) * 0.25
        closing = middle + math.cos(index / 5) * 0.25
        result.append(Candle(BASE + timedelta(minutes=index * step_min), opening, max(opening, closing) + 0.45,
                             min(opening, closing) - 0.45, closing, 100 + index))
    return result


def wave_bars(count: int, tf: str, *, end: float, base: float = 2600.0, slope: float = 0.03,
              wave: float = 1.6, volume: float = 100.0) -> list[dict[str, Any]]:
    """UTC bars ending with the bar that contains ``end`` (the forming bar)."""
    step = TF_SECONDS[tf]
    last_open = int(end) // step * step
    out = []
    for i in range(count):
        mid = base + i * slope + math.sin(i / 4) * wave
        o = mid - math.sin(i / 3) * 0.25
        c = mid + math.cos(i / 5) * 0.25
        out.append({"time": last_open - (count - 1 - i) * step, "open": o, "high": max(o, c) + 0.45,
                    "low": min(o, c) - 0.45, "close": c, "volume": volume + i % 7})
    return out


def flat_bars(count: int, tf: str, *, end: float, price: float = 100.0, spread: float = 0.5,
              volume: float = 100.0) -> list[dict[str, Any]]:
    step = TF_SECONDS[tf]
    last_open = int(end) // step * step
    return [{"time": last_open - (count - 1 - i) * step, "open": price, "high": price + spread,
             "low": price - spread, "close": price, "volume": volume} for i in range(count)]


class FakeFeed:
    """Async MT5Feed stand-in: ``bars``/``ticks`` are dicts the test mutates."""

    def __init__(self, bars: dict[str, list[dict[str, Any]]] | None = None, *, price: float | None = None,
                 spread: float = 0.2, digits: int = 2, now: float | None = None) -> None:
        self.bars_by_tf = bars or {}
        self.price = price
        self.spread = spread
        self.digits = digits
        self.now = now
        self.calls: list[tuple[str, ...]] = []
        self.fail = False

    async def connect(self) -> bool:
        return not self.fail

    async def close(self) -> None:
        self.closed = True

    async def bars(self, symbol: str, timeframe: str, count: int = 500) -> list[dict[str, Any]]:
        self.calls.append(("bars", symbol, timeframe))
        if self.fail:
            raise RuntimeError("feed down")
        items = self.bars_by_tf.get(timeframe)
        if items is None:
            raise RuntimeError(f"no {timeframe} bars")
        return [dict(b) for b in items[-count:]]

    async def tick(self, symbol: str) -> dict[str, Any]:
        self.calls.append(("tick", symbol))
        if self.fail:
            raise RuntimeError("feed down")
        price = self.price
        if price is None:
            any_bars = next(iter(self.bars_by_tf.values()))
            price = any_bars[-1]["close"]
        return {"symbol": symbol, "canonical": symbol, "bid": price, "ask": price + self.spread, "last": None,
                "spread": self.spread, "time": self.now if self.now is not None else 0.0, "volume": 1,
                "digits": self.digits, "point": 10 ** -self.digits}


class FakeTV:
    """TradingViewBridge stand-in (contract methods only)."""

    def __init__(self, *, symbol: str = "OANDA:XAUUSD", timeframe: str = "M15",
                 bars: list[dict[str, Any]] | None = None, connected: bool = True) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self._bars = bars or []
        self.connected = connected
        self.drawn: list[dict[str, Any]] = []
        self.cleared: list[str | None] = []
        self.screenshots = 0

    async def connect(self) -> bool:
        return self.connected

    async def chart_state(self) -> dict[str, Any]:
        from sam.trading.common import canonical_symbol
        last = self._bars[-1] if self._bars else None
        return {"symbol": self.symbol, "canonical": canonical_symbol(self.symbol), "timeframe": self.timeframe,
                "last_bar": last, "price": last["close"] if last else None, "bar_count": len(self._bars)}

    async def bars(self, count: int = 500) -> list[dict[str, Any]]:
        return [dict(b) for b in self._bars[-count:]]

    async def draw_many(self, items: list[dict[str, Any]], *, tag: str = "") -> dict[str, Any]:
        ids = [f"tv{len(self.drawn) + i}" for i in range(len(items))]
        self.drawn.extend({**item, "tag": tag} for item in items)
        return {"ok": bool(ids), "drawn": len(ids), "ids": ids, "errors": []}

    async def clear_my_drawings(self, *, tag: str | None = None) -> int:
        self.cleared.append(tag)
        return 0

    async def screenshot(self, *, fmt: str = "jpeg", max_width: int = 1440) -> bytes:
        self.screenshots += 1
        return b"\xff\xd8fakejpeg"


class FakeMT5Module:
    """Minimal MetaTrader5 module: broker server clock = UTC + ``offset``."""

    TIMEFRAME_M1, TIMEFRAME_M5, TIMEFRAME_M15, TIMEFRAME_H1 = 1, 5, 15, 16385

    def __init__(self, *, now: float, offset: int = 10800, symbols: tuple[str, ...] = ("XAUUSD", "XAUUSD.crp", "XAUUSD.m.e", "EURUSD"),
                 tick_age: float = 1.0) -> None:
        self.now = now
        self.offset = offset
        self.names = symbols
        self.tick_age = tick_age
        self.initialized = 0
        self.fail_next_rates = 0
        self._last_error = (1, "Success")
        self.orders: list[Any] = []

    def initialize(self, *args: Any, **kwargs: Any) -> bool:
        self.initialized += 1
        return True

    def shutdown(self) -> None:
        pass

    def last_error(self) -> tuple[int, str]:
        return self._last_error

    def terminal_info(self) -> Any:
        return SimpleNamespace(connected=True)

    def symbol_info(self, name: str) -> Any:
        if name not in self.names:
            return None
        return SimpleNamespace(name=name, visible=name == "XAUUSD", digits=2, point=0.01)

    def symbols_get(self, pattern: str = "*") -> list[Any]:
        needle = pattern.strip("*").upper()
        return [self.symbol_info(n) for n in self.names if needle in n.upper()]

    def symbol_select(self, name: str, enable: bool) -> bool:
        return True

    def symbol_info_tick(self, name: str) -> Any:
        if name not in self.names:
            return None
        server = self.now - self.tick_age + self.offset
        return SimpleNamespace(time=int(server), time_msc=int(server * 1000), bid=2650.0, ask=2650.2, last=0.0, volume=3)

    def copy_rates_from_pos(self, name: str, timeframe: int, start: int, count: int) -> Any:
        import numpy as np
        if self.fail_next_rates:
            self.fail_next_rates -= 1
            self._last_error = (-10004, "No IPC connection")
            return None
        self._last_error = (1, "Success")
        step = {1: 60, 5: 300, 15: 900, 16385: 3600}[timeframe]
        last_open = int(self.now + self.offset) // step * step
        dtype = [("time", "<i8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"), ("close", "<f8"),
                 ("tick_volume", "<u8"), ("spread", "<i4"), ("real_volume", "<u8")]
        rows = [(last_open - (count - 1 - i) * step, 2650.0, 2651.0, 2649.0, 2650.5, 10 + i, 20, 0) for i in range(count)]
        return np.array(rows, dtype=dtype)

    def order_send(self, request: Any) -> Any:  # must never be reachable through the proxy
        self.orders.append(request)
        return None


__all__ = ["v1_candles", "wave_bars", "flat_bars", "FakeFeed", "FakeTV", "FakeMT5Module", "BASE", "TF_SECONDS"]
