from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from ..contracts import CapabilityState


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


TIMEFRAME_SECONDS: dict[str, int] = {
    "S1": 1,
    "S5": 5,
    "S10": 10,
    "S15": 15,
    "S30": 30,
    "M1": 60,
    "M2": 120,
    "M3": 180,
    "M4": 240,
    "M5": 300,
    "M6": 360,
    "M10": 600,
    "M12": 720,
    "M15": 900,
    "M20": 1200,
    "M30": 1800,
    "H1": 3600,
    "H2": 7200,
    "H3": 10800,
    "H4": 14400,
    "H6": 21600,
    "H8": 28800,
    "H12": 43200,
    "D1": 86400,
    "W1": 604800,
    "MN1": 2592000,
}


def normalize_timeframe(value: str) -> str:
    raw = value.strip().upper().replace(" ", "")
    aliases = {
        "1M": "M1", "3M": "M3", "5M": "M5", "15M": "M15", "30M": "M30", "45M": "M45",
        "1H": "H1", "2H": "H2", "4H": "H4", "1D": "D1", "D": "D1", "1W": "W1", "W": "W1",
        "1MO": "MN1", "MONTHLY": "MN1", "DAILY": "D1", "WEEKLY": "W1",
    }
    return aliases.get(raw, raw)


_INTERVAL_FOLD = re.compile(r"[^A-Z0-9]+")
_MINUTE_TOKENS = {"1m", "1min", "m1"}
_MONTH_TOKENS = {"1M", "MN", "MN1", "1mo", "1MO", "Mo", "monthly", "MONTHLY"}
_INTERVAL_TOKEN_MAP = {
    "1": "M1", "M1": "M1", "1MIN": "M1",
    "3": "M3", "M3": "M3", "3M": "M3",
    "5": "M5", "M5": "M5", "5M": "M5",
    "10": "M10", "M10": "M10", "10M": "M10",
    "15": "M15", "M15": "M15", "15M": "M15",
    "30": "M30", "M30": "M30", "30M": "M30",
    "45": "M45", "M45": "M45", "45M": "M45",
    "60": "H1", "H1": "H1", "1H": "H1",
    "120": "H2", "H2": "H2", "2H": "H2",
    "240": "H4", "H4": "H4", "4H": "H4",
    "D": "D1", "1D": "D1", "D1": "D1", "DAILY": "D1",
    "W": "W1", "1W": "W1", "W1": "W1", "WEEKLY": "W1",
    "MONTHLY": "MN1", "1MO": "MN1", "MN1": "MN1",
    "S1": "S1", "1S": "S1",
    "S5": "S5", "5S": "S5",
    "S15": "S15", "15S": "S15",
    "S30": "S30", "30S": "S30",
}


def parse_interval_token(text: str) -> str | None:
    """Map a TradingView toolbar token onto a canonical SAM timeframe.

    OCR returns fragments such as ``15m``, ``1h``, ``D``, or a bare ``15``.
    Monthly versus one-minute is kept distinct: a lone ``1`` is M1, while
    ``1M`` / ``MN`` / ``monthly`` are MN1.
    """
    raw = (text or "").strip()
    if not raw:
        return None
    compact = raw.replace(" ", "")
    if compact in _MINUTE_TOKENS:
        return "M1"
    if compact in _MONTH_TOKENS:
        return "MN1"
    folded = _INTERVAL_FOLD.sub("", compact.upper())
    if folded == "1M":
        return "MN1" if "M" in raw and "m" not in raw.replace("M", "") else "M1"
    return _INTERVAL_TOKEN_MAP.get(folded)


def pick_visible_interval(
    tokens: list[tuple[str, float]],
    *,
    symbol: str | None = None,
) -> str | None:
    """Choose the current chart interval from OCR tokens.

    A closed TradingView toolbar shows one interval button immediately to the
    right of the symbol. An open interval menu shows many; in that case the
    left-most interval still to the right of the symbol is the armed button.
    """
    symbol_x: float | None = None
    if symbol:
        wanted = symbol.strip().upper()
        for text, x in tokens:
            if text.strip().upper() == wanted:
                symbol_x = x
                break
    ranked: list[tuple[str, float]] = []
    for text, x in tokens:
        if symbol_x is not None and x <= symbol_x + 6:
            continue
        parsed = parse_interval_token(text)
        if parsed:
            ranked.append((parsed, x))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[1])
    unique = list(dict.fromkeys(item[0] for item in ranked))
    if len(unique) == 1:
        return unique[0]
    nearest, nearest_x = ranked[0]
    if symbol_x is None or nearest_x - symbol_x < 220:
        return nearest
    return None


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

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["time"] = self.time.astimezone(UTC).isoformat()
        return payload


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
    observations: list[str] = field(default_factory=list)

    @property
    def current_price(self) -> float | None:
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return self.candles[-1].close if self.candles else None

    @property
    def timestamp_quality_verified(self) -> bool:
        return (
            self.future_bar_count == 0
            and not self.future_tick
            and self.max_future_offset_seconds <= 0
            and self.quote_timestamp_verified
        )

    @property
    def data_quality_verified(self) -> bool:
        return self.timestamp_quality_verified and not self.stale and not self.duplicate_bars

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "requested_symbol": self.requested_symbol,
            "resolved_symbol": self.resolved_symbol,
            "timeframe": self.timeframe,
            "bars": len(self.candles),
            "precision": self.precision,
            "point": self.point,
            "fetched_at": self.fetched_at.astimezone(UTC).isoformat(),
            "bid": self.bid,
            "ask": self.ask,
            "tick_time": self.tick_time.astimezone(UTC).isoformat() if self.tick_time else None,
            "tick_age_seconds": self.tick_age_seconds,
            "tick_stale": self.tick_stale,
            "future_tick": self.future_tick,
            "future_bar_count": self.future_bar_count,
            "latest_bar_age_seconds": self.latest_bar_age_seconds,
            "max_future_offset_seconds": self.max_future_offset_seconds,
            "quote_timestamp_verified": self.quote_timestamp_verified,
            "timestamp_quality_verified": self.timestamp_quality_verified,
            "data_quality_verified": self.data_quality_verified,
            "feed": self.feed,
            "stale": self.stale,
            "missing_bars": self.missing_bars,
            "duplicate_bars": self.duplicate_bars,
            "volume_kind": self.volume_kind,
            "observations": self.observations,
        }


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


def capability(name: str, state: CapabilityState, source: str | None = None, reason: str | None = None) -> dict[str, Any]:
    return {"name": name, "state": state.value, "source": source, "reason": reason}
