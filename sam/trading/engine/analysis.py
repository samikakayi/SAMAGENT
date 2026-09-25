"""Per-timeframe analysis primitives and the entry hunter.

Ported from v1 ``sam_backend/trading/analysis.py`` (pure functions over
closed candles). Changes: the stop/target construction of ``entry_hunter``
is shared as :func:`plan_from_levels` so analyze_market and strategy cards
can compute a plan for a direction without re-implementing it, and levels
closer than a fraction of ATR are skipped as invalidation/targets (a stop
0.2 below price on M1 noise is not a technical invalidation).
"""

from __future__ import annotations

import calendar
from datetime import UTC, datetime, timedelta, timezone
from statistics import fmean, pstdev
from typing import Any

from .indicators import atr, ema, find_swings, latest, rsi, volume_profile, vwap
from .patterns import fib_extension, fib_retracement, optimal_trade_entry, order_blocks
from .types import Candle, Direction, Evidence, MarketDataBatch, PriceLevel, SetupDecision, SetupState, TIMEFRAME_SECONDS


TIMEFRAME_WEIGHTS = {"MN1": 6.0, "W1": 5.0, "D1": 4.2, "H4": 3.5, "H1": 3.0, "M30": 2.4, "M15": 2.0, "M5": 1.5, "M3": 1.3, "M1": 1.0}


def market_structure(candles: list[Candle], timeframe: str) -> dict[str, Any]:
    if len(candles) < 10:
        return {"timeframe": timeframe, "trend": Direction.NEUTRAL.value, "labels": [], "bos": None, "choch": None, "reason": "Insufficient candles"}
    swings = find_swings(candles)
    labels: list[dict[str, Any]] = []
    for points, high_names in ((swings["highs"], ("LH", "HH")), (swings["lows"], ("LL", "HL"))):
        for previous, current in zip(points, points[1:]):
            label = high_names[1] if current["price"] > previous["price"] else high_names[0]
            labels.append({**current, "label": label})
    labels.sort(key=lambda item: item["index"])
    recent_highs = swings["highs"][-2:]
    recent_lows = swings["lows"][-2:]
    trend = Direction.NEUTRAL
    if len(recent_highs) == 2 and len(recent_lows) == 2:
        higher_high = recent_highs[-1]["price"] > recent_highs[-2]["price"]
        higher_low = recent_lows[-1]["price"] > recent_lows[-2]["price"]
        lower_high = recent_highs[-1]["price"] < recent_highs[-2]["price"]
        lower_low = recent_lows[-1]["price"] < recent_lows[-2]["price"]
        if higher_high and higher_low:
            trend = Direction.BULLISH
        elif lower_high and lower_low:
            trend = Direction.BEARISH
    closes = [candle.close for candle in candles]
    current_close = closes[-1]
    bos: dict[str, Any] | None = None
    choch: dict[str, Any] | None = None
    prior_highs = [item for item in swings["highs"] if item["index"] < len(candles) - 1]
    prior_lows = [item for item in swings["lows"] if item["index"] < len(candles) - 1]
    if prior_highs and current_close > prior_highs[-1]["price"]:
        event = {"direction": Direction.BULLISH.value, "level": prior_highs[-1]["price"], "time": candles[-1].time.isoformat(), "confirmed_by_close": True}
        if trend == Direction.BEARISH:
            choch = event
        else:
            bos = event
    elif prior_lows and current_close < prior_lows[-1]["price"]:
        event = {"direction": Direction.BEARISH.value, "level": prior_lows[-1]["price"], "time": candles[-1].time.isoformat(), "confirmed_by_close": True}
        if trend == Direction.BULLISH:
            choch = event
        else:
            bos = event
    internal = market_structure(candles[-max(40, len(candles) // 3):], timeframe) if len(candles) > 120 else None
    return {
        "timeframe": timeframe,
        "trend": trend.value,
        "labels": labels[-12:],
        "bos": bos,
        "choch": choch,
        "mss": choch,
        "last_swing_high": recent_highs[-1] if recent_highs else None,
        "last_swing_low": recent_lows[-1] if recent_lows else None,
        "internal_trend": internal["trend"] if internal else trend.value,
        "external_trend": trend.value,
    }


def _cluster_prices(points: list[dict[str, Any]], tolerance: float) -> list[list[dict[str, Any]]]:
    clusters: list[list[dict[str, Any]]] = []
    for point in sorted(points, key=lambda item: item["price"]):
        if not clusters:
            clusters.append([point])
            continue
        center = fmean(item["price"] for item in clusters[-1])
        if abs(point["price"] - center) <= tolerance:
            clusters[-1].append(point)
        else:
            clusters.append([point])
    return clusters


def snr_levels(candles: list[Candle], timeframe: str, max_levels: int = 16) -> list[PriceLevel]:
    if len(candles) < 20:
        return []
    swings = find_swings(candles, 2, 2)
    current = candles[-1].close
    atr_value = latest(atr(candles, 14)) or max(current * 0.001, 1e-9)
    tolerance = max(atr_value * 0.18, current * 0.00015)
    points = [{**item, "origin": "high"} for item in swings["highs"]] + [{**item, "origin": "low"} for item in swings["lows"]]
    levels: list[PriceLevel] = []
    timeframe_weight = TIMEFRAME_WEIGHTS.get(timeframe, 1.0)
    for cluster in _cluster_prices(points, tolerance):
        price = fmean(point["price"] for point in cluster)
        reactions = len(cluster)
        last_index = max(point["index"] for point in cluster)
        age = len(candles) - 1 - last_index
        recency = max(0.0, 1 - age / max(1, len(candles)))
        fresh = reactions <= 1 or age > max(10, len(candles) // 8)
        closes_beyond = sum(1 for candle in candles[last_index + 1:] if (price > current and candle.close > price + tolerance) or (price < current and candle.close < price - tolerance))
        broken = closes_beyond > 0
        origin_kinds = {point["origin"] for point in cluster}
        role_reversal = len(origin_kinds) > 1 or broken
        kind = "RESISTANCE" if price > current else "SUPPORT"
        if role_reversal:
            kind = "SBR" if price > current else "RBS"
        reaction_quality = min(2.0, fmean(
            (candles[point["index"]].range / atr_value) if atr_value else 0.0 for point in cluster
        ))
        score = timeframe_weight * (1 + min(reactions, 5) * 0.35 + recency * 0.5 + reaction_quality * 0.25 + (0.4 if role_reversal else 0))
        levels.append(PriceLevel(
            price=price,
            kind=kind,
            score=round(score, 4),
            reactions=reactions,
            last_reaction_time=candles[last_index].time,
            timeframe=timeframe,
            fresh=fresh,
            broken=broken,
            role_reversal=role_reversal,
            metadata={"tolerance": tolerance, "age_bars": age, "origins": sorted(origin_kinds)},
        ))
    return _rank_per_side(levels, current, max_levels) + anchor_levels(candles, timeframe, levels, tolerance)


def _rank_per_side(levels: list[PriceLevel], current: float, max_levels: int) -> list[PriceLevel]:
    """The best levels BELOW and ABOVE price separately, then the rest by score.
    One global top-16 kept only levels above price after gold ranged 4270-4360
    for days: the day's low 27 USD below (a swing low on H1, M15 and M5) was
    dropped and SAM said there was no support (repair review, live_levels.py)."""
    ranked = sorted(levels, key=lambda level: level.score, reverse=True)
    half = max(1, max_levels // 2)
    chosen = [lv for lv in ranked if lv.price < current][:half] + [lv for lv in ranked if lv.price >= current][:half]
    rest = [lv for lv in ranked if lv not in chosen][: max(0, max_levels - len(chosen))]
    return sorted(chosen + rest, key=lambda level: level.score, reverse=True)


_INTRADAY = {"M1", "M3", "M5", "M15", "M30", "H1", "H4"}
DAY_SHIFT_S = 2 * 3600   # the FX/metals day rolls over at ~22:00 UTC (17:00 New York)


def anchor_levels(candles: list[Candle], timeframe: str, levels: list[PriceLevel], tolerance: float) -> list[PriceLevel]:
    """Levels every trader looks at, kept regardless of score: today's and the
    previous trading day's high and low, and the last confirmed swing high/low.
    Skipped where a scored level already sits within ``tolerance``."""
    if timeframe not in _INTRADAY or len(candles) < 20:
        return []
    current = candles[-1].close
    days: dict[Any, list[Candle]] = {}
    for candle in candles:
        days.setdefault(datetime.fromtimestamp(candle.time.timestamp() + DAY_SHIFT_S, UTC).date(), []).append(candle)
    keys = sorted(days)[-2:]
    marks: list[tuple[str, float, Candle]] = []
    for name, key in zip(("prev_day", "day")[-len(keys):], keys):
        bars = days[key]
        high = max(bars, key=lambda c: c.high)
        low = min(bars, key=lambda c: c.low)
        marks += [(f"{name}_high", high.high, high), (f"{name}_low", low.low, low)]
    swings = find_swings(candles, 2, 2)
    for side, key in (("swing_high", "highs"), ("swing_low", "lows")):
        if swings[key]:
            point = swings[key][-1]
            marks.append((side, float(point["price"]), candles[int(point["index"])]))
    weight = TIMEFRAME_WEIGHTS.get(timeframe, 1.0)
    out: list[PriceLevel] = []
    for name, price, candle in marks:
        if any(abs(price - lv.price) <= tolerance for lv in [*levels, *out]):
            continue
        kind = "RESISTANCE" if price > current else "SUPPORT"
        out.append(PriceLevel(price=price, kind=kind, score=round(weight * 1.6, 4), reactions=1,
                              last_reaction_time=candle.time, timeframe=timeframe, fresh=True, broken=False,
                              role_reversal=False, metadata={"tolerance": tolerance, "anchor": name}))
    return out


def liquidity_analysis(candles: list[Candle], timeframe: str) -> dict[str, Any]:
    if len(candles) < 20:
        return {"timeframe": timeframe, "equal_highs": [], "equal_lows": [], "sweeps": []}
    swings = find_swings(candles)
    atr_value = latest(atr(candles, 14)) or max(candles[-1].close * 0.001, 1e-9)
    tolerance = atr_value * 0.12

    def equal_clusters(points: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
        return [
            {"type": label, "price": fmean(item["price"] for item in cluster), "touches": len(cluster), "points": cluster}
            for cluster in _cluster_prices(points, tolerance) if len(cluster) >= 2
        ]

    equal_highs = equal_clusters(swings["highs"], "EQUAL_HIGHS")
    equal_lows = equal_clusters(swings["lows"], "EQUAL_LOWS")
    latest_candle = candles[-1]
    sweeps: list[dict[str, Any]] = []
    for level in equal_highs[-5:]:
        if latest_candle.high > level["price"] + tolerance and latest_candle.close < level["price"]:
            sweeps.append({"type": "BUY_SIDE_SWEEP", "price": level["price"], "time": latest_candle.time.isoformat(), "confirmed_by_close": True})
    for level in equal_lows[-5:]:
        if latest_candle.low < level["price"] - tolerance and latest_candle.close > level["price"]:
            sweeps.append({"type": "SELL_SIDE_SWEEP", "price": level["price"], "time": latest_candle.time.isoformat(), "confirmed_by_close": True})
    return {
        "timeframe": timeframe,
        "equal_highs": equal_highs[-8:],
        "equal_lows": equal_lows[-8:],
        "buy_side_liquidity": [item["price"] for item in equal_highs[-8:]],
        "sell_side_liquidity": [item["price"] for item in equal_lows[-8:]],
        "sweeps": sweeps,
        "tolerance": tolerance,
    }


def fair_value_gaps(candles: list[Candle], timeframe: str, limit: int = 20) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    for index in range(2, len(candles)):
        first, middle, third = candles[index - 2], candles[index - 1], candles[index]
        if third.low > first.high:
            lower, upper, direction = first.high, third.low, Direction.BULLISH
        elif third.high < first.low:
            lower, upper, direction = third.high, first.low, Direction.BEARISH
        else:
            continue
        subsequent = candles[index + 1:]
        fully_filled = any(candle.low <= lower for candle in subsequent) if direction == Direction.BULLISH else any(candle.high >= upper for candle in subsequent)
        partially_filled = any(candle.low < upper for candle in subsequent) if direction == Direction.BULLISH else any(candle.high > lower for candle in subsequent)
        gaps.append({
            "type": f"{direction.value}_FVG",
            "direction": direction.value,
            "lower": lower,
            "upper": upper,
            "midpoint": (lower + upper) / 2,
            "time": middle.time.isoformat(),
            "index": index - 1,
            "partial_fill": partially_filled and not fully_filled,
            "full_fill": fully_filled,
            "active": not fully_filled,
            "timeframe": timeframe,
        })
    return gaps[-limit:]


def supply_demand_zones(candles: list[Candle], timeframe: str, limit: int = 12) -> list[dict[str, Any]]:
    if len(candles) < 20:
        return []
    atr_value = latest(atr(candles, 14)) or 0.0
    zones: list[dict[str, Any]] = []
    for index in range(2, len(candles) - 2):
        base = candles[index]
        future = candles[index + 1:index + 3]
        displacement_up = max(item.close for item in future) - base.high
        displacement_down = base.low - min(item.close for item in future)
        if atr_value and displacement_up >= atr_value * 1.2 and base.close <= base.open:
            zone_type = "DEMAND"
            proximal, distal = max(base.open, base.close), base.low
            pattern = "DBR" if candles[index - 1].close < candles[index - 1].open else "RBR"
        elif atr_value and displacement_down >= atr_value * 1.2 and base.close >= base.open:
            zone_type = "SUPPLY"
            proximal, distal = min(base.open, base.close), base.high
            pattern = "RBD" if candles[index - 1].close > candles[index - 1].open else "DBD"
        else:
            continue
        tested = any(candle.low <= max(proximal, distal) and candle.high >= min(proximal, distal) for candle in candles[index + 3:])
        invalidated = any(candle.close < distal for candle in candles[index + 3:]) if zone_type == "DEMAND" else any(candle.close > distal for candle in candles[index + 3:])
        zones.append({
            "type": zone_type,
            "pattern": pattern,
            "proximal": proximal,
            "distal": distal,
            "fresh": not tested,
            "tested": tested,
            "invalidated": invalidated,
            "time": base.time.isoformat(),
            "timeframe": timeframe,
        })
    return zones[-limit:]


def candlestick_patterns(candles: list[Candle], lookback: int = 10) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    start = max(1, len(candles) - lookback)
    for index in range(start, len(candles)):
        previous, candle = candles[index - 1], candles[index]
        body = candle.body
        candle_range = max(candle.range, 1e-12)
        upper_wick = candle.high - max(candle.open, candle.close)
        lower_wick = min(candle.open, candle.close) - candle.low
        name = None
        direction = Direction.NEUTRAL
        if body / candle_range <= 0.1:
            name = "DOJI"
        elif lower_wick >= body * 2 and upper_wick <= body:
            name = "HAMMER" if candle.close >= candle.open else "HANGING_MAN"
            direction = Direction.BULLISH if candle.close >= candle.open else Direction.BEARISH
        elif upper_wick >= body * 2 and lower_wick <= body:
            name = "SHOOTING_STAR" if candle.close <= candle.open else "INVERTED_HAMMER"
            direction = Direction.BEARISH if candle.close <= candle.open else Direction.BULLISH
        elif candle.bullish and not previous.bullish and candle.open <= previous.close and candle.close >= previous.open:
            name, direction = "BULLISH_ENGULFING", Direction.BULLISH
        elif not candle.bullish and previous.bullish and candle.open >= previous.close and candle.close <= previous.open:
            name, direction = "BEARISH_ENGULFING", Direction.BEARISH
        elif body / candle_range >= 0.85:
            name, direction = "MARUBOZU", Direction.BULLISH if candle.bullish else Direction.BEARISH
        if name:
            patterns.append({"name": name, "direction": direction.value, "index": index, "time": candle.time.isoformat(), "context_required": True})
    return patterns


def session_context(at: datetime | None = None) -> dict[str, Any]:
    at = (at or datetime.now(UTC)).astimezone(UTC)

    def last_sunday(year: int, month: int) -> int:
        last_day = calendar.monthrange(year, month)[1]
        weekday = datetime(year, month, last_day).weekday()
        return last_day - ((weekday + 1) % 7)

    def nth_sunday(year: int, month: int, occurrence: int) -> int:
        first_weekday = datetime(year, month, 1).weekday()
        first_sunday = 1 + ((6 - first_weekday) % 7)
        return first_sunday + 7 * (occurrence - 1)

    year = at.year
    london_start = datetime(year, 3, last_sunday(year, 3), 1, tzinfo=UTC)
    london_end = datetime(year, 10, last_sunday(year, 10), 1, tzinfo=UTC)
    london_offset = 1 if london_start <= at < london_end else 0
    new_york_start = datetime(year, 3, nth_sunday(year, 3, 2), 7, tzinfo=UTC)
    new_york_end = datetime(year, 11, nth_sunday(year, 11, 1), 6, tzinfo=UTC)
    new_york_offset = -4 if new_york_start <= at < new_york_end else -5
    tokyo_tz = timezone(timedelta(hours=9), "Asia/Tokyo")
    london_tz = timezone(timedelta(hours=london_offset), "Europe/London")
    new_york_tz = timezone(timedelta(hours=new_york_offset), "America/New_York")
    definitions = {
        "TOKYO": (tokyo_tz, 9, 18),
        "LONDON": (london_tz, 8, 17),
        "NEW_YORK": (new_york_tz, 8, 17),
        "NYSE": (new_york_tz, 9.5, 16),
    }
    sessions: list[dict[str, Any]] = []
    active: list[str] = []
    for name, (timezone_info, start_hour, end_hour) in definitions.items():
        local = at.astimezone(timezone_info)
        decimal_hour = local.hour + local.minute / 60
        is_active = local.weekday() < 5 and start_hour <= decimal_hour < end_hour
        if is_active:
            active.append(name)
        sessions.append({"name": name, "timezone": str(timezone_info), "local_time": local.isoformat(), "active": is_active})
    return {"timestamp_utc": at.isoformat(), "active": active, "sessions": sessions, "london_new_york_overlap": "LONDON" in active and "NEW_YORK" in active}


def statistical_summary(candles: list[Candle], lookback: int = 100) -> dict[str, Any]:
    sample = candles[-max(3, lookback):]
    closes = [candle.close for candle in sample]
    returns = [(right / left - 1) for left, right in zip(closes, closes[1:]) if left]
    mean_return = fmean(returns) if returns else 0.0
    volatility = pstdev(returns) if len(returns) > 1 else 0.0
    current = closes[-1]
    mean = fmean(closes)
    close_std = pstdev(closes) if len(closes) > 1 else 0.0
    zscore = (current - mean) / close_std if close_std else 0.0
    sorted_returns = sorted(returns)
    percentile_index = int(0.95 * (len(sorted_returns) - 1)) if sorted_returns else 0
    return {
        "observations": len(sample),
        "mean_return": mean_return,
        "volatility": volatility,
        "zscore": zscore,
        "return_95th_percentile": sorted_returns[percentile_index] if sorted_returns else 0.0,
        "rolling_range": max(candle.high for candle in sample) - min(candle.low for candle in sample),
    }


def _leg_fibonacci(candles: list[Candle], structure: dict[str, Any]) -> dict[str, Any] | None:
    """Fibonacci anchored on the latest swing leg, with its OTE band."""
    swings = find_swings(candles, left=3, right=3)
    highs, lows = swings["highs"], swings["lows"]
    if not highs or not lows:
        return None
    last_high, last_low = highs[-1], lows[-1]
    if last_high["price"] <= last_low["price"]:
        return None
    direction = Direction.BULLISH if last_low["index"] < last_high["index"] else Direction.BEARISH
    try:
        retracement = fib_retracement(last_high["price"], last_low["price"], direction)
        ote = optimal_trade_entry(last_high["price"], last_low["price"], direction)
        extension = fib_extension(last_high["price"], last_low["price"], direction)
    except ValueError:
        return None
    return {
        "leg": {"high": last_high["price"], "low": last_low["price"], "direction": direction.value},
        "retracement": retracement["levels"],
        "equilibrium": retracement["equilibrium"],
        "ote": ote,
        "extension": extension["levels"],
    }


def timeframe_analysis(batch: MarketDataBatch) -> dict[str, Any]:
    candles = batch.candles
    interval = TIMEFRAME_SECONDS.get(batch.timeframe)
    current_bar_excluded = False
    if interval and len(candles) > 20 and batch.fetched_at.timestamp() < candles[-1].time.timestamp() + interval:
        candles = candles[:-1]
        current_bar_excluded = True
    closes = [candle.close for candle in candles]
    structure = market_structure(candles, batch.timeframe)
    levels = snr_levels(candles, batch.timeframe)
    liquidity = liquidity_analysis(candles, batch.timeframe)
    gaps = fair_value_gaps(candles, batch.timeframe)
    zones = supply_demand_zones(candles, batch.timeframe)
    patterns = candlestick_patterns(candles)
    blocks = order_blocks(candles, batch.timeframe)
    # Anchor Fibonacci on the most recent significant leg rather than an
    # arbitrary window, and expose the OTE band the ICT/SMC modules rely on.
    fibonacci = _leg_fibonacci(candles, structure)
    ema20 = latest(ema(closes, 20))
    ema50 = latest(ema(closes, 50))
    rsi14 = latest(rsi(closes, 14))
    atr14 = latest(atr(candles, 14))
    vwap_value = latest(vwap(candles))
    direction = Direction(structure["trend"])
    evidence = Evidence(
        source="market_structure",
        direction=direction,
        strength=0.75 if direction != Direction.NEUTRAL else 0.25,
        timeframe=batch.timeframe,
        confidence=0.8 if len(candles) >= 100 and not batch.stale else 0.45,
        observation=f"{batch.timeframe} structure is {direction.value.lower()}.",
    )
    return {
        "metadata": {**batch.metadata(), "analysis_bars": len(candles), "current_forming_bar_excluded": current_bar_excluded},
        "current_price": batch.current_price,
        "structure": structure,
        "snr": [level.as_dict() for level in levels],
        "liquidity": liquidity,
        "fvg": gaps,
        "supply_demand": zones,
        "candlestick_patterns": patterns,
        "order_blocks": blocks,
        "fibonacci": fibonacci,
        "indicators": {"ema20": ema20, "ema50": ema50, "rsi14": rsi14, "atr14": atr14, "vwap": vwap_value},
        "volume_profile": volume_profile(candles[-min(250, len(candles)):]),
        "statistics": statistical_summary(candles),
        "evidence": [evidence.as_dict()],
    }


def _nearest_levels(analyses: dict[str, dict[str, Any]], price: float, side: str) -> list[dict[str, Any]]:
    levels: list[dict[str, Any]] = []
    for analysis in analyses.values():
        for level in analysis.get("snr", []):
            if side == "above" and level["price"] > price:
                levels.append(level)
            elif side == "below" and level["price"] < price:
                levels.append(level)
        for zone in analysis.get("supply_demand", []):
            zone_price = zone["proximal"]
            if side == "above" and zone_price > price:
                levels.append({"price": zone_price, "kind": zone["type"], "score": 1.0, "timeframe": zone["timeframe"]})
            elif side == "below" and zone_price < price:
                levels.append({"price": zone_price, "kind": zone["type"], "score": 1.0, "timeframe": zone["timeframe"]})
    unique: dict[int, dict[str, Any]] = {}
    for level in levels:
        key = round(float(level["price"]) * 1_000_000)
        existing = unique.get(key)
        if existing is None or float(level.get("score", 0)) > float(existing.get("score", 0)):
            unique[key] = level
    return sorted(unique.values(), key=lambda item: abs(float(item["price"]) - price))


# Levels nearer than this many ATRs of the execution timeframe are noise, not
# a technical invalidation or a target worth naming.
MIN_LEVEL_DISTANCE_ATR = 0.25


def plan_from_levels(analyses: dict[str, dict[str, Any]], direction: Direction, price: float,
                     ltf: dict[str, Any]) -> dict[str, Any]:
    """Stop and up to three targets for ``direction`` from detected levels.

    The stop sits beyond the nearest structural level on the losing side plus a
    market buffer (3 points, 1.5x spread or 0.08 ATR, whichever is largest);
    targets are the nearest levels on the winning side. Nothing is invented:
    without a level on either side the plan says what is missing.
    """
    bullish = direction == Direction.BULLISH
    atr_value = float((ltf.get("indicators") or {}).get("atr14") or 0.0)
    min_gap = atr_value * MIN_LEVEL_DISTANCE_ATR
    below = [lvl for lvl in _nearest_levels(analyses, price, "below") if price - float(lvl["price"]) >= min_gap]
    above = [lvl for lvl in _nearest_levels(analyses, price, "above") if float(lvl["price"]) - price >= min_gap]
    invalidation_candidates, target_candidates = (below, above) if bullish else (above, below)
    if not invalidation_candidates:
        return {"ok": False, "missing": "Technical invalidation level",
                "reason": "No defensible structural invalidation level is available."}
    invalidation = float(invalidation_candidates[0]["price"])
    metadata = ltf.get("metadata", {})
    spread = None
    if metadata.get("bid") is not None and metadata.get("ask") is not None:
        spread = abs(float(metadata["ask"]) - float(metadata["bid"]))
    buffer = max(float(metadata.get("point") or 0.0) * 3, (spread or 0.0) * 1.5, atr_value * 0.08)
    stop = invalidation - buffer if bullish else invalidation + buffer
    risk = abs(price - stop)
    if risk <= 0:
        return {"ok": False, "zero_risk": True, "missing": None, "reason": "Entry and invalidation produce zero risk."}
    target_prices = sorted({float(item["price"]) for item in target_candidates}, reverse=not bullish)
    targets = [{"price": target, "rr": abs(target - price) / risk,
                "technical_source": next((item.get("kind") for item in target_candidates
                                          if float(item["price"]) == target), "STRUCTURE")}
               for target in target_prices[:3]]
    if not targets:
        return {"ok": False, "missing": "Technical target", "invalidation": invalidation, "stop": stop,
                "stop_distance": risk,
                "reason": "No technical liquidity or structure target is available; SAM will not invent a target."}
    return {"ok": True, "invalidation": invalidation, "stop": stop, "stop_distance": risk, "targets": targets,
            "rr": targets[0]["rr"]}


def entry_hunter(
    analyses: dict[str, dict[str, Any]],
    *,
    minimum_rr: float = 1.5,
    execution_timeframes: tuple[str, ...] = ("M5", "M3", "M1"),
) -> dict[str, Any]:
    if not analyses:
        return {"decision": SetupDecision.NO_TRADE.value, "state": SetupState.NO_SETUP.value, "reasons": ["No market data was supplied."]}
    current_analysis = next((analyses[key] for key in execution_timeframes if key in analyses), next(iter(analyses.values())))
    price = current_analysis.get("current_price")
    if price is None:
        return {"decision": SetupDecision.NO_TRADE.value, "state": SetupState.NO_SETUP.value, "reasons": ["Current price is unavailable."]}
    htf_keys = [key for key in ("H4", "H1", "M30", "M15") if key in analyses]
    htf_directions = [Direction(analyses[key]["structure"]["trend"]) for key in htf_keys]
    bullish_votes = sum(direction == Direction.BULLISH for direction in htf_directions)
    bearish_votes = sum(direction == Direction.BEARISH for direction in htf_directions)
    if bullish_votes and bearish_votes:
        return {
            "decision": SetupDecision.WAIT.value,
            "state": SetupState.WATCH.value,
            "direction": Direction.NEUTRAL.value,
            "reasons": ["Higher timeframes conflict; no independent directional setup is confirmed."],
            "missing_confirmation": ["Aligned higher-timeframe structure"],
        }
    if bullish_votes == bearish_votes:
        return {
            "decision": SetupDecision.WAIT.value,
            "state": SetupState.WATCH.value,
            "direction": Direction.NEUTRAL.value,
            "reasons": ["Higher-timeframe structure is neutral or insufficient."],
            "missing_confirmation": ["Directional higher-timeframe structure"],
        }
    direction = Direction.BULLISH if bullish_votes > bearish_votes else Direction.BEARISH
    ltf = current_analysis
    ltf_structure = ltf["structure"]
    sweeps = ltf["liquidity"]["sweeps"]
    patterns = ltf["candlestick_patterns"]
    active_fvgs = [gap for gap in ltf["fvg"] if gap["active"] and gap["direction"] == direction.value]
    if direction == Direction.BULLISH:
        has_sweep = any(item["type"] == "SELL_SIDE_SWEEP" for item in sweeps)
        has_structure_trigger = (ltf_structure.get("bos") or ltf_structure.get("mss") or {}).get("direction") == direction.value
        has_candle_trigger = any(item["direction"] == direction.value and item["name"] in {"BULLISH_ENGULFING", "HAMMER"} for item in patterns[-3:])
    else:
        has_sweep = any(item["type"] == "BUY_SIDE_SWEEP" for item in sweeps)
        has_structure_trigger = (ltf_structure.get("bos") or ltf_structure.get("mss") or {}).get("direction") == direction.value
        has_candle_trigger = any(item["direction"] == direction.value and item["name"] in {"BEARISH_ENGULFING", "SHOOTING_STAR"} for item in patterns[-3:])
    confirmations = {
        "higher_timeframe_bias": True,
        "liquidity_sweep": has_sweep,
        "mss_or_bos": has_structure_trigger,
        "candle_trigger": has_candle_trigger,
        "active_fvg": bool(active_fvgs),
    }
    missing = [name for name, present in confirmations.items() if name != "active_fvg" and not present]
    if not has_sweep:
        state = SetupState.WAITING_FOR_LIQUIDITY
    elif not has_structure_trigger:
        state = SetupState.WAITING_FOR_MSS
    elif not has_candle_trigger:
        state = SetupState.WAITING_FOR_TRIGGER
    else:
        state = SetupState.ENTRY_READY
    if missing:
        return {
            "decision": SetupDecision.SETUP_FORMING.value if has_sweep or has_structure_trigger else SetupDecision.WATCH.value,
            "state": state.value,
            "direction": direction.value,
            "confirmations": confirmations,
            "missing_confirmation": missing,
            "reasons": ["An entry is not ready because required lower-timeframe confirmation is incomplete."],
            "entry": None,
            "stop": None,
            "targets": [],
        }
    plan = plan_from_levels(analyses, direction, float(price), ltf)
    if plan.get("zero_risk"):
        return {"decision": SetupDecision.NO_TRADE.value, "state": SetupState.NO_SETUP.value, "direction": direction.value, "reasons": ["Entry and invalidation produce zero or inverted risk."]}
    if not plan["ok"]:
        return {
            "decision": SetupDecision.WAIT.value,
            "state": SetupState.WAITING_FOR_TRIGGER.value,
            "direction": direction.value,
            "confirmations": confirmations,
            "missing_confirmation": [plan["missing"]],
            "reasons": [plan["reason"]],
        }
    invalidation, stop, risk, targets = plan["invalidation"], plan["stop"], plan["stop_distance"], plan["targets"]
    if targets[0]["rr"] < minimum_rr:
        return {
            "decision": SetupDecision.NO_TRADE.value,
            "state": SetupState.NO_SETUP.value,
            "direction": direction.value,
            "confirmations": confirmations,
            "entry": price,
            "stop": stop,
            "targets": targets,
            "rr": targets[0]["rr"],
            "reasons": ["SETUP_VALID_BUT_BAD_ENTRY_LOCATION", f"Nearest technical target offers {targets[0]['rr']:.2f}R, below the {minimum_rr:.2f}R minimum."],
        }
    return {
        "decision": SetupDecision.ENTRY_READY.value,
        "state": SetupState.ENTRY_READY.value,
        "direction": direction.value,
        "entry_type": "MARKET_CONFIRMATION",
        "entry": price,
        "entry_zone": active_fvgs[-1] if active_fvgs else None,
        "execution_timeframe": ltf["metadata"]["timeframe"],
        "technical_invalidation": invalidation,
        "stop": stop,
        "stop_distance": risk,
        "targets": targets,
        "rr": targets[0]["rr"],
        "confirmations": confirmations,
        "missing_confirmation": [],
        "reasons": ["Higher-timeframe bias, liquidity event, lower-timeframe structure shift, and candle trigger are confirmed."],
    }


def self_check(analyses: dict[str, dict[str, Any]], setup: dict[str, Any], requested_symbol: str, requested_timeframes: list[str]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    symbols = {item["metadata"]["resolved_symbol"] for item in analyses.values()}
    add("symbol_resolved", bool(symbols), f"Resolved feed symbol(s): {', '.join(sorted(symbols)) or 'none'}")
    present_timeframes = set(analyses)
    add("requested_timeframes", set(requested_timeframes).issubset(present_timeframes), f"Analyzed: {', '.join(sorted(present_timeframes))}")
    add("data_fresh", all(not item["metadata"].get("stale") for item in analyses.values()), "Freshness comes from provider timestamps.")
    future_bar_count = sum(int(item["metadata"].get("future_bar_count") or 0) for item in analyses.values())
    future_tick_timeframes = [
        timeframe
        for timeframe, item in analyses.items()
        if item["metadata"].get("future_tick")
    ]
    add(
        "timestamps_not_future",
        future_bar_count == 0 and not future_tick_timeframes,
        f"Future candles: {future_bar_count}; future tick timeframes: {', '.join(future_tick_timeframes) or 'none'}. Provider timestamps are never normalized silently.",
    )
    unverified_quote_timeframes = [
        timeframe
        for timeframe, item in analyses.items()
        if not item["metadata"].get("quote_timestamp_verified", False)
    ]
    add(
        "quote_timestamps_verified",
        not unverified_quote_timeframes,
        f"Unverified quote timestamps: {', '.join(unverified_quote_timeframes) or 'none'}.",
    )
    add("no_duplicates", all(not item["metadata"].get("duplicate_bars") for item in analyses.values()), "Duplicate timestamps are rejected/deduplicated.")
    add("feed_identified", all(bool(item["metadata"].get("feed")) for item in analyses.values()), "Every timeframe includes provider/feed metadata.")
    add("advanced_claims_guarded", True, "DOM, true delta, footprint, CVD, sentiment, and event claims require an AVAILABLE capability.")
    if setup.get("decision") == SetupDecision.ENTRY_READY.value:
        add("entry_confirmed", not setup.get("missing_confirmation"), "All mandatory deterministic entry confirmations are present.")
        add("technical_stop", setup.get("technical_invalidation") is not None and setup.get("stop") is not None, "Stop derives from a structural invalidation plus a market buffer.")
        add("technical_targets", bool(setup.get("targets")), "Targets derive from detected structure/levels.")
    critical_names = {
        "symbol_resolved", "requested_timeframes", "data_fresh", "timestamps_not_future",
        "quote_timestamps_verified", "no_duplicates", "feed_identified", "entry_confirmed",
        "technical_stop", "technical_targets",
    }
    failed_critical = [item for item in checks if item["name"] in critical_names and not item["passed"]]
    return {
        "passed": not failed_critical,
        "checks": checks,
        "critical_failures": [item["name"] for item in failed_critical],
        "requested_symbol": requested_symbol,
    }
