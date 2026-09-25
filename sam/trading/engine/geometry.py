"""(Ported from v1 ``sam_backend/trading/geometry.py``.) Gann and pitchfork geometry.

Both families are angle-and-slope constructions, so they need a price-per-bar
scale to mean anything: a "1x1" Gann line is one price unit per one bar unit, and
that unit has to be derived from the instrument rather than assumed. Everything
here returns analytical anchors in (price, minutes) chart space; turning those
into pixels is the calibration layer's job, never this module's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .indicators import atr, find_swings, latest
from .types import Candle, Direction

# Classic Gann ratios expressed as (price units : time units).
GANN_ANGLES: tuple[tuple[int, int], ...] = (
    (1, 8), (1, 4), (1, 3), (1, 2), (1, 1), (2, 1), (3, 1), (4, 1), (8, 1),
)

# Pitchfork variants differ only in where the median line starts.
PITCHFORK_VARIANTS = ("andrews", "schiff", "modified_schiff")


@dataclass(slots=True, frozen=True)
class ChartPoint:
    """A point in chart space: a price and a minute-of-day on the time axis."""

    price: float
    minutes: float

    def as_dict(self) -> dict[str, float]:
        return {"price": self.price, "minutes": self.minutes}


def price_per_bar(candles: list[Candle], *, period: int = 14) -> float:
    """The price scale one bar represents, used as Gann's 1x1 unit.

    Average true range is the honest choice: it is the instrument's own measure
    of how far price travels per bar, so a 1x1 angle means the same thing on gold
    as it does on EURUSD instead of depending on an arbitrary constant.
    """
    value = latest(atr(candles, period))
    if value and math.isfinite(value) and value > 0:
        return float(value)
    if len(candles) >= 2:
        spans = [candle.range for candle in candles[-period:] if candle.range > 0]
        if spans:
            return sum(spans) / len(spans)
    return 0.0


def bar_minutes(candles: list[Candle]) -> float:
    """Minutes per bar, measured from the series rather than assumed."""
    if len(candles) < 2:
        return 0.0
    deltas = [
        (later.time - earlier.time).total_seconds() / 60
        for earlier, later in zip(candles[-11:], candles[-10:])
    ]
    positive = [delta for delta in deltas if delta > 0]
    if not positive:
        return 0.0
    positive.sort()
    return positive[len(positive) // 2]


# --- Gann ---------------------------------------------------------------------


def gann_fan(
    candles: list[Candle],
    *,
    pivot_index: int | None = None,
    direction: Direction | None = None,
    bars_forward: int = 60,
    angles: tuple[tuple[int, int], ...] = GANN_ANGLES,
) -> dict[str, Any]:
    """Build a Gann fan from a real swing pivot.

    Each ray is returned as two chart-space anchors so the drawing layer can draw
    it with the ordinary two-anchor trendline path; TradingView's native Gann Fan
    is not reliably placeable through automation, and a fan of verified trendlines
    is the honest equivalent rather than a pretend native object.
    """
    if len(candles) < 30:
        return {"available": False, "reason": "A Gann fan needs at least thirty candles."}
    unit_price = price_per_bar(candles)
    unit_minutes = bar_minutes(candles)
    if unit_price <= 0 or unit_minutes <= 0:
        return {"available": False, "reason": "Could not derive a price-per-bar scale from this series."}

    if pivot_index is None:
        swings = find_swings(candles, 3, 3)
        lows, highs = swings["lows"], swings["highs"]
        if direction is Direction.BEARISH and highs:
            pivot = highs[-1]
        elif direction is Direction.BULLISH and lows:
            pivot = lows[-1]
        elif lows and highs:
            pivot = lows[-1] if lows[-1]["index"] > highs[-1]["index"] else highs[-1]
        elif lows or highs:
            pivot = (lows or highs)[-1]
        else:
            return {"available": False, "reason": "No swing pivot was found to anchor the fan."}
        pivot_index = pivot["index"]
        rising = pivot["kind"] == "SWING_LOW"
    else:
        if not 0 <= pivot_index < len(candles):
            return {"available": False, "reason": "pivot_index is outside the series."}
        rising = direction is not Direction.BEARISH
        pivot = {"index": pivot_index, "price": candles[pivot_index].low if rising else candles[pivot_index].high}

    origin_candle = candles[pivot_index]
    origin_price = float(pivot["price"])
    origin_minutes = origin_candle.time.hour * 60 + origin_candle.time.minute

    rays: list[dict[str, Any]] = []
    for price_units, time_units in angles:
        slope = (price_units / time_units) * (unit_price / unit_minutes)
        if not rising:
            slope = -slope
        end_minutes = origin_minutes + bars_forward * unit_minutes
        end_price = origin_price + slope * (end_minutes - origin_minutes)
        rays.append({
            "label": f"{price_units}x{time_units}",
            "price_units": price_units,
            "time_units": time_units,
            "slope_price_per_minute": slope,
            "primary": price_units == time_units,
            "start": ChartPoint(origin_price, origin_minutes).as_dict(),
            "end": ChartPoint(end_price, end_minutes).as_dict(),
        })

    return {
        "available": True,
        "representation": "trendline_fan",
        "limitation": (
            "Drawn as a fan of individually verified trendlines. TradingView's native Gann Fan "
            "object cannot be placed reliably through automation, so this is the semantic "
            "equivalent rather than the native tool."
        ),
        "pivot": {"index": pivot_index, "price": origin_price, "minutes": origin_minutes,
                  "time": origin_candle.time.isoformat(), "rising": rising},
        "unit_price_per_bar": unit_price,
        "unit_minutes_per_bar": unit_minutes,
        "rays": rays,
        "subjective": True,
        "note": "Gann anchors are analyst-chosen; the pivot used is always reported alongside the rays.",
    }


def gann_box(candles: list[Candle], *, bars: int = 60) -> dict[str, Any]:
    """A price/time box over the recent range, divided on Gann's eighths."""
    if len(candles) < bars:
        return {"available": False, "reason": f"A Gann box needs at least {bars} candles."}
    window = candles[-bars:]
    high = max(candle.high for candle in window)
    low = min(candle.low for candle in window)
    span = high - low
    if span <= 0:
        return {"available": False, "reason": "The window has no price range."}
    start, end = window[0], window[-1]
    divisions = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    return {
        "available": True,
        "representation": "horizontal_levels",
        "limitation": "Rendered as the box's horizontal divisions; the native Gann Box object is not automatable.",
        "high": high,
        "low": low,
        "span": span,
        "start_time": start.time.isoformat(),
        "end_time": end.time.isoformat(),
        "levels": [
            {"ratio": ratio, "price": low + span * ratio, "label": f"{int(ratio * 8)}/8"}
            for ratio in divisions
        ],
        "subjective": True,
    }


def square_of_nine_levels(price: float, *, rings: int = 4, step_degrees: float = 45.0) -> dict[str, Any]:
    """Square-of-9 levels around a price, as horizontal levels.

    The construction walks the square's spiral: taking the root of the price and
    stepping it by a fraction of a full turn, then squaring the result.
    """
    if price <= 0:
        return {"available": False, "reason": "Square of 9 requires a positive price."}
    if step_degrees <= 0 or step_degrees > 360:
        return {"available": False, "reason": "step_degrees must fall between 0 and 360."}
    root = math.sqrt(price)
    increment = step_degrees / 180.0
    levels: list[dict[str, Any]] = []
    steps = int(round(360.0 / step_degrees)) * rings
    for index in range(-steps, steps + 1):
        if index == 0:
            continue
        candidate = (root + index * increment) ** 2
        if candidate <= 0:
            continue
        levels.append({
            "degrees": index * step_degrees,
            "price": candidate,
            "above": candidate > price,
            "ring": abs(index) * step_degrees / 360.0,
        })
    levels.sort(key=lambda item: item["price"])
    return {
        "available": True,
        "representation": "horizontal_levels",
        "anchor_price": price,
        "root": root,
        "step_degrees": step_degrees,
        "levels": levels,
        "subjective": True,
        "limitation": "Square-of-9 output is a set of horizontal price levels, not a native TradingView object.",
    }


# --- Pitchfork ----------------------------------------------------------------


def calculate_pitchfork_anchors(
    candles: list[Candle],
    *,
    variant: str = "andrews",
    p0_index: int | None = None,
    p1_index: int | None = None,
    p2_index: int | None = None,
) -> dict[str, Any]:
    """Derive P0/P1/P2 and the resulting median line and parallels.

    Andrews draws the median from P0 through the midpoint of P1-P2. Schiff moves
    the origin to the midpoint of P0 and that midpoint in price only; modified
    Schiff moves it in both price and time.
    """
    normalized = variant.strip().lower()
    if normalized not in PITCHFORK_VARIANTS:
        return {"available": False, "reason": f"Unknown pitchfork variant: {variant}"}
    if len(candles) < 30:
        return {"available": False, "reason": "A pitchfork needs at least thirty candles."}

    if None in (p0_index, p1_index, p2_index):
        swings = find_swings(candles, 3, 3)
        points = sorted(
            [*({**item, "type": "L"} for item in swings["lows"]),
             *({**item, "type": "H"} for item in swings["highs"])],
            key=lambda item: item["index"],
        )
        alternating: list[dict[str, Any]] = []
        for point in points:
            if alternating and alternating[-1]["type"] == point["type"]:
                continue
            alternating.append(point)
        if len(alternating) < 3:
            return {"available": False, "reason": "Three alternating swing pivots were not found."}
        chosen = alternating[-3:]
        p0_index, p1_index, p2_index = (item["index"] for item in chosen)
        prices = [item["price"] for item in chosen]
    else:
        for index in (p0_index, p1_index, p2_index):
            if not 0 <= index < len(candles):
                return {"available": False, "reason": "An anchor index is outside the series."}
        if not p0_index < p1_index < p2_index:
            return {"available": False, "reason": "Anchors must be ordered P0 < P1 < P2 in time."}
        prices = [candles[p0_index].close, candles[p1_index].close, candles[p2_index].close]

    def minutes_of(index: int) -> float:
        candle = candles[index]
        return candle.time.hour * 60 + candle.time.minute

    p0 = ChartPoint(float(prices[0]), minutes_of(p0_index))
    p1 = ChartPoint(float(prices[1]), minutes_of(p1_index))
    p2 = ChartPoint(float(prices[2]), minutes_of(p2_index))
    midpoint = ChartPoint((p1.price + p2.price) / 2, (p1.minutes + p2.minutes) / 2)

    if normalized == "andrews":
        origin = p0
    elif normalized == "schiff":
        # Price-only shift: half way between P0 and the P1/P2 midpoint.
        origin = ChartPoint((p0.price + midpoint.price) / 2, p0.minutes)
    else:
        origin = ChartPoint((p0.price + midpoint.price) / 2, (p0.minutes + midpoint.minutes) / 2)

    if midpoint.minutes == origin.minutes:
        return {"available": False, "reason": "The median line is vertical; these anchors cannot form a pitchfork."}
    slope = (midpoint.price - origin.price) / (midpoint.minutes - origin.minutes)

    # Parallels run through P1 and P2 at the median's slope.
    def parallel_through(point: ChartPoint) -> dict[str, Any]:
        intercept = point.price - slope * point.minutes
        return {
            "through": point.as_dict(),
            "slope_price_per_minute": slope,
            "intercept": intercept,
            "start": {"price": slope * origin.minutes + intercept, "minutes": origin.minutes},
            "end": {"price": slope * midpoint.minutes + intercept, "minutes": midpoint.minutes},
        }

    upper_point, lower_point = (p1, p2) if p1.price >= p2.price else (p2, p1)
    return {
        "available": True,
        "variant": normalized,
        "anchors": {"P0": p0.as_dict(), "P1": p1.as_dict(), "P2": p2.as_dict()},
        "anchor_indices": {"P0": p0_index, "P1": p1_index, "P2": p2_index},
        "midpoint": midpoint.as_dict(),
        "median_line": {
            "start": origin.as_dict(),
            "end": midpoint.as_dict(),
            "slope_price_per_minute": slope,
            "intercept": origin.price - slope * origin.minutes,
        },
        "upper_parallel": parallel_through(upper_point),
        "lower_parallel": parallel_through(lower_point),
        "warning_lines": [
            {"multiple": multiple, "slope_price_per_minute": slope,
             "intercept": (upper_point.price - slope * upper_point.minutes)
             + multiple * ((upper_point.price - lower_point.price))}
            for multiple in (0.5, 1.0)
        ],
        "subjective": True,
        "representation": "three_trendlines",
        "limitation": (
            "Drawn as the median line plus its two parallels, each verified independently. "
            "TradingView's native pitchfork tool needs a three-click sequence whose intermediate "
            "state cannot be confirmed from a screenshot, so the equivalent lines are drawn instead."
        ),
    }


def pitchfork_price_at(pitchfork: dict[str, Any], line: str, minutes: float) -> float | None:
    """Price of a pitchfork line at a given time, for verification."""
    if not pitchfork.get("available"):
        return None
    if line == "median":
        payload = pitchfork["median_line"]
        return payload["slope_price_per_minute"] * minutes + payload["intercept"]
    if line in {"upper", "lower"}:
        payload = pitchfork[f"{line}_parallel"]
        return payload["slope_price_per_minute"] * minutes + payload["intercept"]
    return None
