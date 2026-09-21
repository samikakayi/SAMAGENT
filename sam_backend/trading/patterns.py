"""Deterministic pattern engines: order blocks, Fibonacci, harmonics, Elliott, ORB.

Every function here is pure arithmetic over closed candles. Nothing calls a model,
nothing repaints, and each engine refuses to emit a pattern whose defining
conditions are not met — an order block needs displacement out of it, a harmonic
needs its ratios inside tolerance, and an Elliott count needs the three hard rules
to hold. Reporting "no pattern" is a valid and frequent answer.
"""

from __future__ import annotations

from datetime import UTC, datetime, time as clock_time, timedelta
from typing import Any

from .indicators import find_swings
from .types import Candle, Direction

# --- Order blocks -------------------------------------------------------------

# The candle that originates a move must be followed by a decisive expansion.
# Requiring the impulse to exceed a multiple of recent range is what separates an
# order block from "the last opposite candle", which is the usual mislabelling.
DISPLACEMENT_ATR_MULTIPLE = 1.5
DISPLACEMENT_LOOKAHEAD = 3
ATR_PERIOD = 14


def _average_range(candles: list[Candle], end: int, period: int = ATR_PERIOD) -> float:
    window = candles[max(0, end - period):end]
    if not window:
        return 0.0
    return sum(candle.range for candle in window) / len(window)


def _structure_break(candles: list[Candle], origin: int, direction: Direction, lookahead: int) -> dict[str, Any] | None:
    """Did price take out the prior swing after leaving this candle?"""
    window = candles[origin + 1:origin + 1 + lookahead]
    if not window:
        return None
    if direction is Direction.BULLISH:
        prior_high = max((candle.high for candle in candles[max(0, origin - 10):origin + 1]), default=None)
        if prior_high is None:
            return None
        for offset, candle in enumerate(window, start=1):
            if candle.close > prior_high:
                return {"broke_index": origin + offset, "level": prior_high}
        return None
    prior_low = min((candle.low for candle in candles[max(0, origin - 10):origin + 1]), default=None)
    if prior_low is None:
        return None
    for offset, candle in enumerate(window, start=1):
        if candle.close < prior_low:
            return {"broke_index": origin + offset, "level": prior_low}
    return None


def order_blocks(candles: list[Candle], timeframe: str, limit: int = 12) -> list[dict[str, Any]]:
    """Detect order blocks: the last opposing candle before a displacing break.

    A candidate is only promoted when the move away from it both exceeds the
    displacement threshold and breaks structure. Later interaction reclassifies it
    as mitigated, a breaker, or invalid.
    """
    if len(candles) < ATR_PERIOD + DISPLACEMENT_LOOKAHEAD + 2:
        return []
    blocks: list[dict[str, Any]] = []
    last_index = len(candles) - 1
    for index in range(ATR_PERIOD, last_index - DISPLACEMENT_LOOKAHEAD):
        candle = candles[index]
        average = _average_range(candles, index)
        if average <= 0:
            continue
        window = candles[index + 1:index + 1 + DISPLACEMENT_LOOKAHEAD]
        if not window:
            continue

        # Bullish OB: a down candle, then an expansion up that breaks structure.
        if not candle.bullish:
            impulse = max(item.high for item in window) - candle.low
            if impulse >= average * DISPLACEMENT_ATR_MULTIPLE:
                broke = _structure_break(candles, index, Direction.BULLISH, DISPLACEMENT_LOOKAHEAD)
                if broke:
                    blocks.append(_build_block(candles, index, Direction.BULLISH, timeframe, impulse / average, broke))

        # Bearish OB: an up candle, then an expansion down that breaks structure.
        if candle.bullish:
            impulse = candle.high - min(item.low for item in window)
            if impulse >= average * DISPLACEMENT_ATR_MULTIPLE:
                broke = _structure_break(candles, index, Direction.BEARISH, DISPLACEMENT_LOOKAHEAD)
                if broke:
                    blocks.append(_build_block(candles, index, Direction.BEARISH, timeframe, impulse / average, broke))

    blocks.sort(key=lambda item: item["index"], reverse=True)
    return blocks[:limit]


def _build_block(
    candles: list[Candle],
    index: int,
    direction: Direction,
    timeframe: str,
    displacement: float,
    broke: dict[str, Any],
) -> dict[str, Any]:
    candle = candles[index]
    proximal = candle.open if direction is Direction.BULLISH else candle.open
    top, bottom = candle.high, candle.low
    later = candles[index + 1:]
    state = "FRESH"
    mitigated_at: str | None = None
    invalidated = False
    for follower in later:
        if direction is Direction.BULLISH:
            if follower.close < bottom:
                invalidated = True
                break
            if follower.low <= top and state == "FRESH":
                state, mitigated_at = "MITIGATED", follower.time.isoformat()
        else:
            if follower.close > top:
                invalidated = True
                break
            if follower.high >= bottom and state == "FRESH":
                state, mitigated_at = "MITIGATED", follower.time.isoformat()
    if invalidated:
        # Price closed clean through the block: it may now act as a breaker.
        state = "BREAKER"
    return {
        "index": index,
        "kind": "BULLISH_OB" if direction is Direction.BULLISH else "BEARISH_OB",
        "direction": direction.value,
        "timeframe": timeframe,
        "top": top,
        "bottom": bottom,
        "proximal": proximal,
        "distal": bottom if direction is Direction.BULLISH else top,
        "time": candle.time.isoformat(),
        "state": state,
        "mitigated_at": mitigated_at,
        "displacement_atr": round(displacement, 3),
        "structure_break": broke,
        "fresh": state == "FRESH",
    }


# --- Fibonacci ----------------------------------------------------------------

RETRACEMENT_LEVELS = (0.236, 0.382, 0.5, 0.618, 0.705, 0.786, 0.886)
EXTENSION_LEVELS = (1.272, 1.414, 1.618, 2.0, 2.618)
OTE_BAND = (0.618, 0.79)


def fib_retracement(swing_high: float, swing_low: float, direction: Direction) -> dict[str, Any]:
    """Retracement levels for a leg. Direction states which way the leg ran."""
    if swing_high <= swing_low:
        raise ValueError("swing_high must be above swing_low")
    span = swing_high - swing_low
    if direction is Direction.BULLISH:
        levels = {str(ratio): swing_high - span * ratio for ratio in RETRACEMENT_LEVELS}
        anchor = {"from": swing_low, "to": swing_high}
    else:
        levels = {str(ratio): swing_low + span * ratio for ratio in RETRACEMENT_LEVELS}
        anchor = {"from": swing_high, "to": swing_low}
    return {
        "direction": direction.value,
        "anchor": anchor,
        "span": span,
        "levels": levels,
        "equilibrium": (swing_high + swing_low) / 2,
    }


def fib_extension(swing_high: float, swing_low: float, direction: Direction) -> dict[str, Any]:
    span = swing_high - swing_low
    if span <= 0:
        raise ValueError("swing_high must be above swing_low")
    if direction is Direction.BULLISH:
        levels = {str(ratio): swing_high + span * (ratio - 1) for ratio in EXTENSION_LEVELS}
    else:
        levels = {str(ratio): swing_low - span * (ratio - 1) for ratio in EXTENSION_LEVELS}
    return {"direction": direction.value, "span": span, "levels": levels}


def optimal_trade_entry(swing_high: float, swing_low: float, direction: Direction) -> dict[str, Any]:
    """The 0.618-0.79 retracement band."""
    span = swing_high - swing_low
    if span <= 0:
        raise ValueError("swing_high must be above swing_low")
    if direction is Direction.BULLISH:
        upper = swing_high - span * OTE_BAND[0]
        lower = swing_high - span * OTE_BAND[1]
    else:
        lower = swing_low + span * OTE_BAND[0]
        upper = swing_low + span * OTE_BAND[1]
    low, high = min(lower, upper), max(lower, upper)
    return {"direction": direction.value, "low": low, "high": high, "midpoint": (low + high) / 2}


def fib_confluence(sets: list[dict[str, Any]], tolerance: float) -> list[dict[str, Any]]:
    """Prices where levels from independent legs cluster."""
    points: list[tuple[float, str]] = []
    for index, item in enumerate(sets):
        for ratio, price in (item.get("levels") or {}).items():
            points.append((float(price), f"leg{index}:{ratio}"))
    points.sort()
    clusters: list[dict[str, Any]] = []
    current: list[tuple[float, str]] = []
    for point in points:
        if current and point[0] - current[-1][0] > tolerance:
            if len(current) > 1:
                clusters.append(_cluster(current))
            current = []
        current.append(point)
    if len(current) > 1:
        clusters.append(_cluster(current))
    clusters.sort(key=lambda item: item["count"], reverse=True)
    return clusters


def _cluster(points: list[tuple[float, str]]) -> dict[str, Any]:
    prices = [point[0] for point in points]
    return {
        "price": sum(prices) / len(prices),
        "low": min(prices),
        "high": max(prices),
        "count": len(points),
        "sources": [point[1] for point in points],
    }


# --- Harmonic patterns --------------------------------------------------------

# Ratio envelopes per pattern; None means that leg is unconstrained by definition.
# `cd_ab` is what actually makes an AB=CD an AB=CD, so it is enforced rather
# than assumed: without it any pair of alternating legs would qualify.
HARMONIC_RULES: dict[str, dict[str, tuple[float, float] | None]] = {
    "Gartley": {"ab_xa": (0.585, 0.650), "bc_ab": (0.382, 0.886), "cd_bc": (1.13, 1.618), "ad_xa": (0.746, 0.826)},
    "Bat": {"ab_xa": (0.382, 0.500), "bc_ab": (0.382, 0.886), "cd_bc": (1.618, 2.618), "ad_xa": (0.850, 0.920)},
    "AlternateBat": {"ab_xa": (0.352, 0.400), "bc_ab": (0.382, 0.886), "cd_bc": (2.0, 3.618), "ad_xa": (1.080, 1.180)},
    "Butterfly": {"ab_xa": (0.750, 0.820), "bc_ab": (0.382, 0.886), "cd_bc": (1.618, 2.618), "ad_xa": (1.240, 1.410)},
    "Crab": {"ab_xa": (0.382, 0.618), "bc_ab": (0.382, 0.886), "cd_bc": (2.240, 3.618), "ad_xa": (1.550, 1.650)},
    "DeepCrab": {"ab_xa": (0.850, 0.920), "bc_ab": (0.382, 0.886), "cd_bc": (2.0, 3.618), "ad_xa": (1.550, 1.650)},
    "Shark": {"ab_xa": (0.382, 0.618), "bc_ab": (1.130, 1.618), "cd_bc": (1.618, 2.240), "ad_xa": (0.850, 1.130)},
    "Cypher": {"ab_xa": (0.382, 0.618), "bc_ab": (1.130, 1.414), "cd_bc": (1.272, 2.0), "ad_xa": (0.746, 0.800)},
    "AB=CD": {"ab_xa": None, "bc_ab": (0.382, 0.886), "cd_bc": (1.130, 2.618), "ad_xa": None, "cd_ab": (0.90, 1.10)},
    "5-0": {"ab_xa": (1.130, 1.618), "bc_ab": (1.618, 2.240), "cd_bc": (0.480, 0.520), "ad_xa": None},
}


def _ratio(numerator: float, denominator: float) -> float | None:
    return abs(numerator) / abs(denominator) if denominator else None


def validate_harmonic(x: float, a: float, b: float, c: float, d: float) -> list[dict[str, Any]]:
    """Return every harmonic definition whose ratios the XABCD points satisfy."""
    xa, ab, bc, cd, ad = a - x, b - a, c - b, d - c, d - a
    # Legs must alternate direction; otherwise this is not an XABCD shape.
    if not (xa * ab < 0 and ab * bc < 0 and bc * cd < 0):
        return []
    measured = {
        "ab_xa": _ratio(ab, xa),
        "bc_ab": _ratio(bc, ab),
        "cd_bc": _ratio(cd, bc),
        "ad_xa": _ratio(ad, xa),
        "cd_ab": _ratio(cd, ab),
    }
    matches: list[dict[str, Any]] = []
    for name, rules in HARMONIC_RULES.items():
        satisfied = True
        for key, bounds in rules.items():
            if bounds is None:
                continue
            value = measured.get(key)
            if value is None or not (bounds[0] <= value <= bounds[1]):
                satisfied = False
                break
        if satisfied:
            matches.append({
                "pattern": name,
                "direction": (Direction.BULLISH if d < c else Direction.BEARISH).value,
                "ratios": {key: round(value, 4) if value is not None else None for key, value in measured.items()},
                "points": {"X": x, "A": a, "B": b, "C": c, "D": d},
                "prz": prz_from_points(x, a, b, c),
                "targets": harmonic_targets(a, c, d),
                "invalidation": x,
            })
    return matches


def prz_from_points(x: float, a: float, b: float, c: float) -> dict[str, float]:
    """Potential reversal zone: where the common completion ratios overlap."""
    xa, bc = a - x, c - b
    candidates = [a + xa * -ratio for ratio in (0.786, 0.886, 1.13, 1.618)]
    candidates += [c + bc * ratio for ratio in (1.272, 1.618, 2.0)]
    return {"low": min(candidates), "high": max(candidates)}


def harmonic_targets(a: float, c: float, d: float) -> dict[str, float]:
    leg = c - d
    return {"tp1": d + leg * 0.382, "tp2": d + leg * 0.618, "tp3": a}


def detect_harmonics(candles: list[Candle], timeframe: str, limit: int = 5) -> list[dict[str, Any]]:
    """Scan recent alternating swings for valid XABCD structures."""
    swings = find_swings(candles, left=3, right=3)
    points = sorted(
        [*({**item, "type": "H"} for item in swings["highs"]), *({**item, "type": "L"} for item in swings["lows"])],
        key=lambda item: item["index"],
    )
    # Keep only strictly alternating high/low pivots.
    alternating: list[dict[str, Any]] = []
    for point in points:
        if alternating and alternating[-1]["type"] == point["type"]:
            better = point["price"] > alternating[-1]["price"] if point["type"] == "H" else point["price"] < alternating[-1]["price"]
            if better:
                alternating[-1] = point
            continue
        alternating.append(point)
    found: list[dict[str, Any]] = []
    for start in range(max(0, len(alternating) - 12), len(alternating) - 4):
        window = alternating[start:start + 5]
        if len(window) < 5:
            break
        prices = [item["price"] for item in window]
        for match in validate_harmonic(*prices):
            found.append({
                **match,
                "timeframe": timeframe,
                "indices": [item["index"] for item in window],
                "times": [item["time"] for item in window],
            })
    return found[-limit:]


# --- Elliott wave -------------------------------------------------------------


def validate_impulse(points: list[float]) -> dict[str, Any]:
    """Check the three inviolable impulse rules for a 0-1-2-3-4-5 point set."""
    if len(points) != 6:
        return {"valid": False, "reason": "An impulse needs six points (0 through 5)."}
    origin, one, two, three, four, five = points
    up = one > origin
    violations: list[str] = []
    if up:
        if two <= origin:
            violations.append("Wave 2 retraced beyond the start of wave 1.")
        if three <= one:
            violations.append("Wave 3 did not exceed the end of wave 1.")
        length_one, length_three, length_five = one - origin, three - two, five - four
        if length_three < length_one and length_three < length_five:
            violations.append("Wave 3 is the shortest impulse leg.")
        if four <= one:
            violations.append("Wave 4 overlapped the territory of wave 1.")
    else:
        if two >= origin:
            violations.append("Wave 2 retraced beyond the start of wave 1.")
        if three >= one:
            violations.append("Wave 3 did not exceed the end of wave 1.")
        length_one, length_three, length_five = origin - one, two - three, four - five
        if length_three < length_one and length_three < length_five:
            violations.append("Wave 3 is the shortest impulse leg.")
        if four >= one:
            violations.append("Wave 4 overlapped the territory of wave 1.")
    return {
        "valid": not violations,
        "direction": (Direction.BULLISH if up else Direction.BEARISH).value,
        "violations": violations,
        "wave_lengths": {"1": abs(one - origin), "3": abs(three - two), "5": abs(five - four)},
        "extended_wave": _extended_wave(abs(one - origin), abs(three - two), abs(five - four)),
    }


def _extended_wave(one: float, three: float, five: float) -> str:
    longest = max(one, three, five)
    return {one: "1", three: "3", five: "5"}[longest]


def wave_fib_relationships(points: list[float]) -> dict[str, float | None]:
    if len(points) != 6:
        return {}
    origin, one, two, three, four, five = points
    length_one = abs(one - origin)
    return {
        "wave2_retracement": _ratio(two - one, one - origin),
        "wave3_of_wave1": _ratio(three - two, one - origin) if length_one else None,
        "wave4_retracement": _ratio(four - three, three - two),
        "wave5_of_wave1": _ratio(five - four, one - origin) if length_one else None,
    }


def rank_wave_counts(candles: list[Candle], timeframe: str, max_counts: int = 3) -> list[dict[str, Any]]:
    """Produce candidate impulse counts, ranked. Never forces a single answer."""
    swings = find_swings(candles, left=3, right=3)
    points = sorted(
        [*({**item, "type": "H"} for item in swings["highs"]), *({**item, "type": "L"} for item in swings["lows"])],
        key=lambda item: item["index"],
    )
    alternating: list[dict[str, Any]] = []
    for point in points:
        if alternating and alternating[-1]["type"] == point["type"]:
            continue
        alternating.append(point)
    candidates: list[dict[str, Any]] = []
    for start in range(max(0, len(alternating) - 14), max(0, len(alternating) - 5)):
        window = alternating[start:start + 6]
        if len(window) < 6:
            break
        prices = [item["price"] for item in window]
        validation = validate_impulse(prices)
        relationships = wave_fib_relationships(prices)
        # Prefer valid counts, then those closest to the classic 0.618/1.618 pair.
        score = 0.0
        if validation["valid"]:
            score += 10.0
        wave2 = relationships.get("wave2_retracement")
        wave3 = relationships.get("wave3_of_wave1")
        if wave2 is not None:
            score += max(0.0, 2.0 - abs(wave2 - 0.618) * 4)
        if wave3 is not None:
            score += max(0.0, 2.0 - abs(wave3 - 1.618) * 2)
        candidates.append({
            "timeframe": timeframe,
            "points": prices,
            "indices": [item["index"] for item in window],
            "times": [item["time"] for item in window],
            "validation": validation,
            "fib": {key: (round(value, 4) if value is not None else None) for key, value in relationships.items()},
            "score": round(score, 3),
            "label": "IMPULSE" if validation["valid"] else "INVALID_IMPULSE",
        })
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:max_counts]


# --- Opening range breakout ---------------------------------------------------

SESSION_OPENS: dict[str, tuple[int, int]] = {
    "tokyo": (0, 0),
    "london": (7, 0),
    "frankfurt": (6, 0),
    "new_york": (12, 0),
    "nyse": (13, 30),
}


def opening_range(
    candles: list[Candle],
    *,
    minutes: int = 15,
    session: str = "london",
    session_date: datetime | None = None,
) -> dict[str, Any]:
    """Build the opening range and classify what price did to it afterwards.

    Only candles that closed inside the range window define it, so the range can
    never be shaped by the very move it is used to trade.
    """
    if minutes <= 0:
        raise ValueError("minutes must be positive")
    if session not in SESSION_OPENS:
        raise ValueError(f"Unknown session: {session}")
    if not candles:
        return {"available": False, "reason": "No candles supplied."}
    hour, minute = SESSION_OPENS[session]
    reference = (session_date or candles[-1].time).astimezone(UTC)
    open_at = datetime.combine(reference.date(), clock_time(hour, minute), tzinfo=UTC)
    close_at = open_at + timedelta(minutes=minutes)

    in_range = [candle for candle in candles if open_at <= candle.time < close_at]
    if not in_range:
        return {
            "available": False,
            "session": session,
            "minutes": minutes,
            "window": {"open": open_at.isoformat(), "close": close_at.isoformat()},
            "reason": "No candles fall inside the opening-range window for this date.",
        }
    high = max(candle.high for candle in in_range)
    low = min(candle.low for candle in in_range)
    after = [candle for candle in candles if candle.time >= close_at]

    breakout: dict[str, Any] | None = None
    false_breakout = False
    retest = False
    for candle in after:
        if breakout is None:
            if candle.close > high:
                breakout = {"direction": Direction.BULLISH.value, "time": candle.time.isoformat(), "price": candle.close}
            elif candle.close < low:
                breakout = {"direction": Direction.BEARISH.value, "time": candle.time.isoformat(), "price": candle.close}
            continue
        if breakout["direction"] == Direction.BULLISH.value:
            if candle.close < low:
                false_breakout = True
            elif candle.low <= high:
                retest = True
        else:
            if candle.close > high:
                false_breakout = True
            elif candle.high >= low:
                retest = True

    size = high - low
    result: dict[str, Any] = {
        "available": True,
        "session": session,
        "minutes": minutes,
        "window": {"open": open_at.isoformat(), "close": close_at.isoformat()},
        "high": high,
        "low": low,
        "size": size,
        "midpoint": (high + low) / 2,
        "candles_in_range": len(in_range),
        "breakout": breakout,
        "false_breakout": false_breakout,
        "retest": retest,
        "state": "NO_BREAKOUT" if breakout is None else ("FALSE_BREAKOUT" if false_breakout else "BROKEN"),
    }
    if breakout and not false_breakout:
        bullish = breakout["direction"] == Direction.BULLISH.value
        result["entry_trigger"] = high if bullish else low
        result["invalidation"] = low if bullish else high
        result["targets"] = {
            "tp1": (high + size) if bullish else (low - size),
            "tp2": (high + size * 2) if bullish else (low - size * 2),
            "tp3": (high + size * 3) if bullish else (low - size * 3),
        }
    return result
