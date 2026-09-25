"""Engine value types: candles, feed batches with freshness metadata, levels.

Ported from v1 ``sam_backend/trading/types.py`` and decoupled from v1's
``contracts`` module. The one behavioural change is where freshness comes
from: v1 read MT5 broker-server timestamps as if they were UTC, so every
fetch looked 3 h in the future (180 "future" M1 bars, future_tick=True,
data_quality_verified=False on every timeframe, measured 2026-09-24,
reports/trading-intelligence.json) and the analyst blocked every entry. In
SAM 2 the MT5 feed removes the broker offset before bars reach the engine,
and :func:`build_batch` computes freshness from true UTC times.
"""

from __future__ import annotations

import math
import time as _time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Iterable, Mapping

from ..common import normalize_timeframe as _common_normalize_timeframe


class Direction(StrEnum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class SetupDecision(StrEnum):
    NO_TRADE = "NO_TRADE"
    WAIT = "WAIT"
    WATCH = "WATCH"
    SETUP_FORMING = "SETUP_FORMING"
    ENTRY_READY = "ENTRY_READY"


class SetupState(StrEnum):
    NO_SETUP = "NO_SETUP"
    WATCH = "WATCH"
    APPROACHING_ZONE = "APPROACHING_ZONE"
    IN_ZONE = "IN_ZONE"
    SETUP_FORMING = "SETUP_FORMING"
    WAITING_FOR_LIQUIDITY = "WAITING_FOR_LIQUIDITY"
    WAITING_FOR_MSS = "WAITING_FOR_MSS"
    WAITING_FOR_BOS = "WAITING_FOR_BOS"
    WAITING_FOR_RETEST = "WAITING_FOR_RETEST"
    WAITING_FOR_TRIGGER = "WAITING_FOR_TRIGGER"
    ENTRY_READY = "ENTRY_READY"
    ENTRY_TRIGGERED = "ENTRY_TRIGGERED"
    INVALIDATED = "INVALIDATED"
    TP1 = "TP1"
    TP2 = "TP2"
    TP3 = "TP3"
    STOPPED = "STOPPED"
    EXPIRED = "EXPIRED"


class CapabilityState(StrEnum):
    """Honest capability labels (v1 ``contracts.CapabilityState``)."""

    AVAILABLE = "AVAILABLE"
    PARTIALLY_AVAILABLE = "PARTIALLY_AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    UNCONFIGURED = "UNCONFIGURED"


TIMEFRAME_SECONDS: dict[str, int] = {
    "S1": 1, "S5": 5, "S10": 10, "S15": 15, "S30": 30,
    "M1": 60, "M2": 120, "M3": 180, "M4": 240, "M5": 300, "M6": 360, "M10": 600, "M12": 720,
    "M15": 900, "M20": 1200, "M30": 1800,
    "H1": 3600, "H2": 7200, "H3": 10800, "H4": 14400, "H6": 21600, "H8": 28800, "H12": 43200,
    "D1": 86400, "W1": 604800, "MN1": 2592000,
}

_V1_ALIASES = {
    "1M": "M1", "3M": "M3", "5M": "M5", "15M": "M15", "30M": "M30", "45M": "M45",
    "1H": "H1", "2H": "H2", "4H": "H4", "1D": "D1", "D": "D1", "1W": "W1", "W": "W1",
    "1MO": "MN1", "MONTHLY": "MN1", "DAILY": "D1", "WEEKLY": "W1",
}


def normalize_timeframe(value: str) -> str:
    """Canonical timeframe name. Uses the shared parser (Sorani words, TV
    resolutions) first; unknown values fall back to v1's alias table and are
    returned upper-cased so callers can report them rather than guess."""
    common = _common_normalize_timeframe(value)
    if common:
        return common
    raw = str(value).strip().upper().replace(" ", "")
    return _V1_ALIASES.get(raw, raw)


@dataclass(slots=True, frozen=True)
class Candle:
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    spread: float | None = None
    real_volume: float | None = None

    def __post_init__(self) -> None:
        values = (self.open, self.high, self.low, self.close, self.volume)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Candle contains a non-finite value")
        if self.high < max(self.open, self.close, self.low) or self.low > min(self.open, self.close, self.high):
            raise ValueError("Candle OHLC values are inconsistent")
        if self.volume < 0:
            raise ValueError("Candle volume cannot be negative")

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def bullish(self) -> bool:
        return self.close > self.open

    @property
    def ts(self) -> int:
        """Bar open time as UTC unix seconds."""
        return int(self.time.timestamp())

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["time"] = self.time.astimezone(UTC).isoformat()
        return payload


def bars_to_candles(bars: Iterable[Mapping[str, Any]]) -> tuple[list[Candle], int, int]:
    """Convert ``common.Bar`` dicts (UTC seconds) into sorted, de-duplicated
    candles. Returns (candles, duplicates, rejected) -- a malformed bar is
    counted and skipped instead of failing the whole analysis."""
    by_time: dict[int, Candle] = {}
    duplicates = rejected = 0
    for bar in bars:
        try:
            stamp = int(bar["time"])
            candle = Candle(time=datetime.fromtimestamp(stamp, UTC), open=float(bar["open"]),
                            high=float(bar["high"]), low=float(bar["low"]), close=float(bar["close"]),
                            volume=float(bar.get("volume") or 0.0))
        except (KeyError, TypeError, ValueError, OverflowError, OSError):
            rejected += 1
            continue
        if stamp in by_time:
            duplicates += 1
        by_time[stamp] = candle
    return [by_time[key] for key in sorted(by_time)], duplicates, rejected


def candles_to_bars(candles: Iterable[Candle]) -> list[dict[str, Any]]:
    return [{"time": c.ts, "open": c.open, "high": c.high, "low": c.low, "close": c.close, "volume": c.volume}
            for c in candles]


@dataclass(slots=True)
class MarketDataBatch:
    provider: str
    requested_symbol: str
    resolved_symbol: str
    timeframe: str
    candles: list[Candle]
    precision: int
    point: float
    fetched_at: datetime
    bid: float | None = None
    ask: float | None = None
    tick_time: datetime | None = None
    tick_age_seconds: float | None = None
    tick_stale: bool = False
    future_tick: bool = False
    future_bar_count: int = 0
    latest_bar_age_seconds: float | None = None
    max_future_offset_seconds: float = 0.0
    quote_timestamp_verified: bool = False
    feed: str | None = None
    stale: bool = False
    missing_bars: int = 0
    duplicate_bars: int = 0
    volume_kind: str = "tick"
    market_closed: bool = False
    observations: list[str] = field(default_factory=list)

    @property
    def current_price(self) -> float | None:
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return self.candles[-1].close if self.candles else None

    @property
    def timestamp_quality_verified(self) -> bool:
        return (self.future_bar_count == 0 and not self.future_tick and self.max_future_offset_seconds <= 0
                and self.quote_timestamp_verified)

    @property
    def data_quality_verified(self) -> bool:
        return self.timestamp_quality_verified and not self.stale and not self.duplicate_bars

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": self.provider, "requested_symbol": self.requested_symbol,
            "resolved_symbol": self.resolved_symbol, "timeframe": self.timeframe, "bars": len(self.candles),
            "precision": self.precision, "point": self.point,
            "fetched_at": self.fetched_at.astimezone(UTC).isoformat(), "bid": self.bid, "ask": self.ask,
            "tick_time": self.tick_time.astimezone(UTC).isoformat() if self.tick_time else None,
            "tick_age_seconds": self.tick_age_seconds, "tick_stale": self.tick_stale,
            "future_tick": self.future_tick, "future_bar_count": self.future_bar_count,
            "latest_bar_age_seconds": self.latest_bar_age_seconds,
            "max_future_offset_seconds": self.max_future_offset_seconds,
            "quote_timestamp_verified": self.quote_timestamp_verified,
            "timestamp_quality_verified": self.timestamp_quality_verified,
            "data_quality_verified": self.data_quality_verified, "feed": self.feed, "stale": self.stale,
            "market_closed": self.market_closed, "missing_bars": self.missing_bars,
            "duplicate_bars": self.duplicate_bars, "volume_kind": self.volume_kind,
            "observations": self.observations,
        }


# A bar stamped this far ahead of the local clock is "future" (clock skew
# between the PC and the broker/TradingView is normally well under a second).
FUTURE_TOLERANCE_S = 5.0
TICK_FRESHNESS_S = 300.0


def is_weekend_close(at: float) -> bool:
    """FX/metals weekly close: Friday ~21:00 UTC to Sunday ~21:00 UTC (22:00
    in winter). Used only to explain stale data, never to fabricate it."""
    moment = datetime.fromtimestamp(at, UTC)
    weekday, hour = moment.weekday(), moment.hour
    return weekday == 5 or (weekday == 4 and hour >= 21) or (weekday == 6 and hour < 21)


def build_batch(*, provider: str, requested_symbol: str, resolved_symbol: str, timeframe: str,
                bars: Iterable[Mapping[str, Any]], feed: str | None = None, tick: Mapping[str, Any] | None = None,
                point: float = 0.01, precision: int = 2, now: float | None = None,
                volume_kind: str = "tick") -> MarketDataBatch:
    """Build a batch from UTC bars (+ optional UTC tick) and judge freshness.

    Bars MUST already be UTC (the MT5 feed removes the broker offset; TradingView
    bars are UTC). A bar is stale when the newest one opened more than
    max(4 intervals, 5 min) ago; on a weekend that is reported as a closed
    market rather than an error.
    """
    tf = normalize_timeframe(timeframe)
    now_s = float(now if now is not None else _time.time())
    candles, duplicates, rejected = bars_to_candles(bars)
    if not candles:
        raise ValueError(f"No usable {tf} bars for {resolved_symbol}")
    interval = TIMEFRAME_SECONDS.get(tf)
    missing = 0
    if interval:
        for left, right in zip(candles, candles[1:]):
            delta = right.ts - left.ts
            if interval * 1.5 < delta <= interval * 6:  # closures are not "missing"
                missing += max(0, round(delta / interval) - 1)
    latest_age = now_s - candles[-1].ts
    future_offsets = [c.ts - now_s for c in candles if c.ts - now_s > FUTURE_TOLERANCE_S]
    stale_candle = bool(interval and latest_age > max(interval * 4, 300))
    bid = ask = None
    tick_time: datetime | None = None
    tick_age: float | None = None
    if tick:
        bid = float(tick["bid"]) if tick.get("bid") else None
        ask = float(tick["ask"]) if tick.get("ask") else None
        if tick.get("time"):
            tick_time = datetime.fromtimestamp(float(tick["time"]), UTC)
            tick_age = now_s - float(tick["time"])
    future_tick = bool(tick_age is not None and tick_age < -FUTURE_TOLERANCE_S)
    tick_stale = bool(tick_age is not None and tick_age > TICK_FRESHNESS_S)
    quote_present = bid is not None or ask is not None
    quote_ok = bool(not quote_present or (tick_time is not None and not future_tick and not tick_stale))
    max_future = max([0.0, *future_offsets, (-tick_age if future_tick and tick_age is not None else 0.0)])
    stale = bool(stale_candle or future_offsets or future_tick or (quote_present and not quote_ok))
    closed = bool(stale and is_weekend_close(now_s))
    observations: list[str] = []
    if resolved_symbol.upper() != requested_symbol.upper():
        observations.append(f"Requested symbol {requested_symbol} was mapped to {resolved_symbol}.")
    if rejected:
        observations.append(f"{rejected} malformed bar(s) were skipped.")
    if missing:
        observations.append(f"Detected {missing} short in-session candle gap(s).")
    if future_offsets:
        observations.append(f"{len(future_offsets)} bar(s) are up to {max(future_offsets):.0f}s in the future "
                            "(clock or offset problem).")
    if future_tick:
        observations.append("The tick timestamp is in the future (clock or offset problem).")
    if tick_stale:
        observations.append(f"The last tick is {tick_age:.0f}s old.")
    if stale_candle:
        observations.append("The market is closed for the weekend." if closed
                            else f"The newest {tf} bar opened {latest_age:.0f}s ago (stale).")
    return MarketDataBatch(
        provider=provider, requested_symbol=requested_symbol, resolved_symbol=resolved_symbol, timeframe=tf,
        candles=candles, precision=precision, point=point, fetched_at=datetime.fromtimestamp(now_s, UTC),
        bid=bid, ask=ask, tick_time=tick_time, tick_age_seconds=round(tick_age, 3) if tick_age is not None else None,
        tick_stale=tick_stale, future_tick=future_tick, future_bar_count=len(future_offsets),
        latest_bar_age_seconds=round(latest_age, 3), max_future_offset_seconds=round(max_future, 3),
        quote_timestamp_verified=quote_ok, feed=feed, stale=stale, missing_bars=missing,
        duplicate_bars=duplicates, volume_kind=volume_kind, market_closed=closed, observations=observations)


@dataclass(slots=True)
class Evidence:
    source: str
    direction: Direction
    strength: float
    timeframe: str
    confidence: float
    observation: str
    freshness: float = 1.0
    dependency: str | None = None
    conflicts: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["direction"] = self.direction.value
        return payload


@dataclass(slots=True)
class PriceLevel:
    price: float
    kind: str
    score: float
    reactions: int
    last_reaction_time: datetime | None
    timeframe: str
    fresh: bool
    broken: bool = False
    role_reversal: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["last_reaction_time"] = self.last_reaction_time.isoformat() if self.last_reaction_time else None
        return payload


__all__ = ["Direction", "SetupDecision", "SetupState", "CapabilityState", "TIMEFRAME_SECONDS",
           "normalize_timeframe", "Candle", "MarketDataBatch", "Evidence", "PriceLevel", "build_batch",
           "bars_to_candles", "candles_to_bars", "is_weekend_close", "FUTURE_TOLERANCE_S"]
