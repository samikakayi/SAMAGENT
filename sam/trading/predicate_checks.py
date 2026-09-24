"""The predicate library: every deterministic check a strategy card rule can
be tied to (registered into ``predicates.PREDICATES`` on import).

Each check returns ``(passed, detail)``; ``passed`` is None when the data to
decide is missing -- never a guess. See ``predicates.py`` for parameter
conventions (timeframe roles, direction words).
"""

from __future__ import annotations

import datetime as _dt
from statistics import fmean
from typing import Any

from ..textnorm import normalize_ckb
from .engine.analysis import market_structure, session_context
from .engine.indicators import adx, ema, find_swings, latest
from .engine.patterns import opening_range
from .engine.research import _divergence
from .engine.types import Candle, Direction, normalize_timeframe
from .predicates import (PredicateContext, Result, _analysis, _atr, _int, _missing, _num, _tf, predicate,
                         to_direction)


def _structure_breaks(candles: list[Candle], within: int) -> list[dict[str, Any]]:
    """Closes beyond the latest *confirmed* swing (two bars on the right)
    within the last ``within`` bars; labelled bos/choch by the prior trend."""
    events: list[dict[str, Any]] = []
    if len(candles) < 20:
        return events
    swings = find_swings(candles, 2, 2)
    start = max(10, len(candles) - within)
    for i in range(start, len(candles)):
        highs = [s for s in swings["highs"] if s["index"] + 2 < i]
        lows = [s for s in swings["lows"] if s["index"] + 2 < i]
        prior = market_structure(candles[:i], "X")["trend"] if i >= 10 else Direction.NEUTRAL.value
        close, prev_close = candles[i].close, candles[i - 1].close
        if highs and close > highs[-1]["price"] >= prev_close:
            kind = "choch" if prior == Direction.BEARISH.value else "bos"
            events.append({"index": i, "direction": Direction.BULLISH, "level": highs[-1]["price"], "kind": kind})
        if lows and close < lows[-1]["price"] <= prev_close:
            kind = "choch" if prior == Direction.BULLISH.value else "bos"
            events.append({"index": i, "direction": Direction.BEARISH, "level": lows[-1]["price"], "kind": kind})
    return events


def _break_predicate(ctx: PredicateContext, params: dict[str, Any], kinds: set[str]) -> Result:
    tf = _tf(ctx, params, "entry")
    candles = ctx.candles.get(tf)
    if not candles:
        return _missing(tf)
    wanted = to_direction(params.get("direction"), ctx)
    within = _int(params, "within_bars", 10)
    key = ("breaks", tf, within)
    if key not in ctx.cache:  # bos/choch/mss_or_bos on one card share the scan
        ctx.cache[key] = _structure_breaks(candles, within)
    for event in reversed(ctx.cache[key]):
        if event["kind"] in kinds and (wanted is None or event["direction"] == wanted):
            ago = len(candles) - 1 - event["index"]
            return True, f"{tf} {event['kind'].upper()} {event['direction'].value.lower()} through {event['level']:.5g} {ago} bar(s) ago"
    return False, f"no {'/'.join(sorted(kinds)).upper()} on {tf} in the last {within} bars"


def trading_day_start(at: float) -> float:
    """FX trading day rolls at 17:00 New York (the standard broker day)."""
    from zoneinfo import ZoneInfo
    ny = _dt.datetime.fromtimestamp(at, ZoneInfo("America/New_York"))
    roll = ny.replace(hour=17, minute=0, second=0, microsecond=0)
    if ny < roll:
        roll -= _dt.timedelta(days=1)
    return roll.timestamp()


def _day_range(candles: list[Candle], start: float, end: float) -> tuple[float, float] | None:
    inside = [c for c in candles if start <= c.ts < end]
    if not inside:
        return None
    return max(c.high for c in inside), min(c.low for c in inside)


def _reference_level(ctx: PredicateContext, candles: list[Candle], analysis: dict[str, Any], target: str,
                     now: float) -> tuple[float | None, float, str]:
    """(price, since_ts, label) of a liquidity reference for ``swept``."""
    day0 = trading_day_start(now)
    if target in ("asian_high", "asian_low"):
        midnight = _dt.datetime.fromtimestamp(now, _dt.UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        start = midnight.timestamp()
        asia = _day_range(candles, start, start + 7 * 3600)  # 00:00-07:00 UTC, before London
        if asia is None:
            return None, 0.0, "Asian range"
        return (asia[0] if target == "asian_high" else asia[1]), start + 7 * 3600, "Asian range"
    if target in ("pdh", "pdl"):
        prev = None
        for back in range(1, 5):  # Monday's previous trading day is Friday
            prev = _day_range(candles, day0 - 86400 * back, day0 - 86400 * (back - 1))
            if prev is not None:
                break
        if prev is None:
            return None, 0.0, "previous day"
        return (prev[0] if target == "pdh" else prev[1]), day0, "previous day"
    if target in ("equal_highs", "equal_lows"):
        pools = analysis["liquidity"].get("equal_highs" if target == "equal_highs" else "equal_lows") or []
        if not pools:
            return None, 0.0, "equal highs/lows"
        pool = min(pools, key=lambda p: abs(p["price"] - ctx.price))
        return pool["price"], 0.0, "equal highs/lows"
    swings = find_swings(candles[:-3] if len(candles) > 3 else candles, 2, 2)
    points = swings["highs"] if target == "swing_high" else swings["lows"]
    if not points:
        return None, 0.0, "swing"
    return points[-1]["price"], float(candles[points[-1]["index"]].ts), "swing"


# --- predicates ----------------------------------------------------------------------

@predicate("trend_is", "Market structure trend of a timeframe equals a direction.",
           "ترێندی کاتێک بەرەو ئاراستەیەکی دیاریکراوە.",
           tf={"type": "string"}, direction={"type": "string", "enum": ["up", "down", "range", "setup"]})
def trend_is(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    tf, analysis = _analysis(ctx, params, "bias")
    if analysis is None:
        return _missing(tf)
    wanted = to_direction(params.get("direction"), ctx)
    actual = Direction(analysis["structure"]["trend"])
    if wanted is None:
        return None, "no direction to compare"
    return actual == wanted, f"{tf} trend is {actual.value.lower()}"


@predicate("trend_aligned", "Every listed timeframe trends the same (non-neutral) way, optionally a given one.",
           "هەموو کاتە دیاریکراوەکان ترێندیان یەک ئاراستەیە.",
           tfs={"type": "string", "description": "comma separated, e.g. H4,H1"}, direction={"type": "string"})
def trend_aligned(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    tfs = [normalize_timeframe(t) for t in str(params.get("tfs") or ",".join(ctx.analyses)).split(",") if t.strip()]
    missing = [tf for tf in tfs if tf not in ctx.analyses]
    if missing:
        return _missing(",".join(missing))
    trends = {Direction(ctx.analyses[tf]["structure"]["trend"]) for tf in tfs}
    wanted = to_direction(params.get("direction"), ctx) if params.get("direction") else None
    aligned = len(trends) == 1 and Direction.NEUTRAL not in trends and (wanted is None or trends == {wanted})
    return aligned, f"{','.join(tfs)} trends: {', '.join(sorted(t.value.lower() for t in trends))}"


@predicate("mss_or_bos", "A close broke the latest confirmed swing (BOS or MSS/CHoCH) within N bars.",
           "شکانی پێکهاتە (BOS یان MSS) لە ماوەی چەند مۆمی دواییدا.",
           tf={"type": "string"}, direction={"type": "string"}, within_bars={"type": "integer", "default": 10})
def mss_or_bos(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    return _break_predicate(ctx, params, {"bos", "choch"})


@predicate("bos", "Break of structure in the direction of the prior trend within N bars.",
           "شکانی پێکهاتە بە ئاراستەی ترێندی پێشوو.",
           tf={"type": "string"}, direction={"type": "string"}, within_bars={"type": "integer", "default": 10})
def bos(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    return _break_predicate(ctx, params, {"bos"})


@predicate("choch", "Change of character / market structure shift against the prior trend within N bars.",
           "گۆڕانی ئاراستەی بازاڕ (CHoCH/MSS) دژی ترێندی پێشوو.",
           tf={"type": "string"}, direction={"type": "string"}, within_bars={"type": "integer", "default": 10})
def choch(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    return _break_predicate(ctx, params, {"choch"})


@predicate("swept", "Price traded through a liquidity reference and (optionally) closed back inside.",
           "نرخ لیکویدیتی ئاستێکی ڕاماڵی (نزمایی ئاسیا، PDH، PDL، لووتکەی یەکسان...) و گەڕایەوە.",
           tf={"type": "string"},
           level={"type": "string", "enum": ["asian_low", "asian_high", "pdh", "pdl", "equal_lows", "equal_highs",
                                             "swing_low", "swing_high"]},
           within_bars={"type": "integer", "default": 20}, reclaim={"type": "boolean", "default": True})
def swept(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    tf, analysis = _analysis(ctx, params, "setup")
    candles = ctx.candles.get(tf)
    if analysis is None or not candles:
        return _missing(tf)
    target = str(params.get("level") or ("swing_low" if ctx.direction == "long" else "swing_high"))
    level, since, label = _reference_level(ctx, candles, analysis, target, ctx.now or candles[-1].ts)
    if level is None:
        return None, f"no {label} reference on {tf}"
    low_side = target.endswith("low") or target in ("pdl", "equal_lows")
    window = [c for c in candles[-_int(params, "within_bars", 20):] if c.ts >= since]
    reclaim = params.get("reclaim", True) not in (False, "false", "0")
    for i, candle in enumerate(window):
        pierced = candle.low < level if low_side else candle.high > level
        if not pierced:
            continue
        later = window[i:]
        back = any((c.close > level) if low_side else (c.close < level) for c in later)
        if back or not reclaim:
            return True, f"{tf} swept {target} {level:.5g}"
    return False, f"{tf} {target} {level:.5g} not swept"


@predicate("liquidity_pool", "Equal highs (buy side) or equal lows (sell side) rest within 2 ATR of price.",
           "لووتکە یان نزمایی یەکسان (لیکویدیتی) نزیکی نرخن.",
           tf={"type": "string"}, side={"type": "string", "enum": ["buy", "sell"]})
def liquidity_pool(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    tf, analysis = _analysis(ctx, params, "setup")
    if analysis is None:
        return _missing(tf)
    key = "buy_side_liquidity" if str(params.get("side", "buy")).startswith("b") else "sell_side_liquidity"
    near = [p for p in analysis["liquidity"].get(key, []) if abs(p - ctx.price) <= 2 * max(_atr(analysis), 1e-9)]
    return bool(near), f"{tf} {key.split('_')[0]}-side pools near price: {len(near)}"


def _inside(price: float, low: float, high: float, pad: float = 0.0) -> bool:
    return min(low, high) - pad <= price <= max(low, high) + pad


@predicate("in_fvg", "Price is inside an active fair value gap (optionally of a direction).",
           "نرخ لە ناو بۆشایی نرخێکی (FVG) چالاکدایە.",
           tf={"type": "string"}, direction={"type": "string"})
def in_fvg(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    tf, analysis = _analysis(ctx, params, "setup")
    if analysis is None:
        return _missing(tf)
    wanted = to_direction(params.get("direction"), ctx)
    for gap in analysis.get("fvg") or []:
        if gap["active"] and (wanted is None or gap["direction"] == wanted.value) and _inside(ctx.price, gap["lower"], gap["upper"]):
            return True, f"inside {tf} {gap['direction'].lower()} FVG {gap['lower']:.5g}-{gap['upper']:.5g}"
    return False, f"not inside an active {tf} FVG"


@predicate("fvg_present", "An active fair value gap formed within the last N bars.",
           "بۆشایی نرخێکی چالاک لە مۆمە دواییەکاندا دروست بووە.",
           tf={"type": "string"}, direction={"type": "string"}, within_bars={"type": "integer", "default": 20})
def fvg_present(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    tf, analysis = _analysis(ctx, params, "setup")
    candles = ctx.candles.get(tf)
    if analysis is None or not candles:
        return _missing(tf)
    wanted = to_direction(params.get("direction"), ctx)
    first = len(candles) - _int(params, "within_bars", 20)
    found = [g for g in analysis.get("fvg") or []
             if g["active"] and g["index"] >= first and (wanted is None or g["direction"] == wanted.value)]
    return bool(found), f"{len(found)} recent active FVG(s) on {tf}"


@predicate("in_order_block", "Price is inside a live (fresh or mitigated, not breaker) order block.",
           "نرخ لە ناو ئۆردەر بلۆکێکی زیندوودایە.",
           tf={"type": "string"}, direction={"type": "string"})
def in_order_block(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    tf, analysis = _analysis(ctx, params, "setup")
    if analysis is None:
        return _missing(tf)
    wanted = to_direction(params.get("direction"), ctx)
    for block in analysis.get("order_blocks") or []:
        if block["state"] == "BREAKER" or (wanted is not None and block["direction"] != wanted.value):
            continue
        if _inside(ctx.price, block["bottom"], block["top"]):
            return True, f"inside {tf} {block['kind']} {block['bottom']:.5g}-{block['top']:.5g} ({block['state'].lower()})"
    return False, f"not inside a live {tf} order block"


@predicate("in_ote", "Price is in the optimal trade entry band (default 0.62-0.79) of the latest swing leg.",
           "نرخ لە ناوچەی OTE ی دوایین شەپۆلدایە.",
           tf={"type": "string"}, low={"type": "number", "default": 0.62}, high={"type": "number", "default": 0.79})
def in_ote(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    tf, analysis = _analysis(ctx, params, "setup")
    if analysis is None:
        return _missing(tf)
    fib = analysis.get("fibonacci")
    if not fib:
        return None, f"no swing leg on {tf}"
    leg = fib["leg"]
    span = leg["high"] - leg["low"]
    lo_r, hi_r = _num(params, "low", 0.62), _num(params, "high", 0.79)
    if leg["direction"] == Direction.BULLISH.value:
        band = (leg["high"] - span * hi_r, leg["high"] - span * lo_r)
    else:
        band = (leg["low"] + span * lo_r, leg["low"] + span * hi_r)
    return _inside(ctx.price, *band), f"{tf} OTE {band[0]:.5g}-{band[1]:.5g}"


@predicate("premium_discount", "Price is in the premium (above equilibrium) or discount half of the dealing range.",
           "نرخ لە نیوەی پرێمیەم یان دیسکاونتی مەودای مامەڵەدایە.",
           tf={"type": "string"}, zone={"type": "string", "enum": ["premium", "discount", "setup"]})
def premium_discount(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    tf, analysis = _analysis(ctx, params, "bias")
    if analysis is None:
        return _missing(tf)
    fib = analysis.get("fibonacci")
    if not fib:
        return None, f"no dealing range on {tf}"
    zone = str(params.get("zone") or "setup")
    if zone == "setup":
        zone = "discount" if ctx.direction == "long" else "premium" if ctx.direction == "short" else ""
    if not zone:
        return None, "no direction to pick premium or discount"
    eq = fib["equilibrium"]
    actual = "premium" if ctx.price > eq else "discount"
    return actual == zone, f"price is in {tf} {actual} (equilibrium {eq:.5g})"


@predicate("near_level", "Price is within a fraction of ATR of a scored support/resistance level.",
           "نرخ نزیکی ئاستێکی پشتگیری یان بەرگرییە.",
           tf={"type": "string"}, atr_frac={"type": "number", "default": 0.3},
           kind={"type": "string", "enum": ["any", "support", "resistance"]})
def near_level(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    tf, analysis = _analysis(ctx, params, "setup")
    if analysis is None:
        return _missing(tf)
    reach = _num(params, "atr_frac", 0.3) * max(_atr(analysis), 1e-9)
    kind = str(params.get("kind") or "any")
    for level in analysis.get("snr") or []:
        side = "support" if level["price"] <= ctx.price else "resistance"
        if (kind == "any" or kind == side) and abs(level["price"] - ctx.price) <= reach:
            return True, f"{tf} {side} {level['price']:.5g} within {reach:.3g}"
    return False, f"no {tf} level within {reach:.3g}"


@predicate("in_supply_demand", "Price is inside a valid supply or demand zone.",
           "نرخ لە ناو ناوچەی داواکاری یان خستنەڕوودایە.",
           tf={"type": "string"}, kind={"type": "string", "enum": ["demand", "supply", "setup", "any"]})
def in_supply_demand(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    tf, analysis = _analysis(ctx, params, "setup")
    if analysis is None:
        return _missing(tf)
    kind = str(params.get("kind") or "setup")
    if kind == "setup":
        kind = "demand" if ctx.direction == "long" else "supply" if ctx.direction == "short" else "any"
    for zone in analysis.get("supply_demand") or []:
        if zone["invalidated"] or (kind != "any" and zone["type"].lower() != kind):
            continue
        if _inside(ctx.price, zone["proximal"], zone["distal"]):
            return True, f"inside {tf} {zone['type'].lower()} {zone['proximal']:.5g}-{zone['distal']:.5g}"
    return False, f"not inside a {tf} {kind} zone"


def _rsi(ctx: PredicateContext, params: dict[str, Any]) -> tuple[str, float | None]:
    tf, analysis = _analysis(ctx, params, "entry")
    return tf, (analysis or {}).get("indicators", {}).get("rsi14")


@predicate("rsi_above", "RSI(14) of a timeframe is above a value.", "RSI لە ژمارەیەک بەرزترە.",
           tf={"type": "string"}, value={"type": "number", "default": 50})
def rsi_above(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    tf, value = _rsi(ctx, params)
    return (None, f"no RSI on {tf}") if value is None else (value > _num(params, "value", 50), f"{tf} RSI {value:.1f}")


@predicate("rsi_below", "RSI(14) of a timeframe is below a value.", "RSI لە ژمارەیەک نزمترە.",
           tf={"type": "string"}, value={"type": "number", "default": 50})
def rsi_below(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    tf, value = _rsi(ctx, params)
    return (None, f"no RSI on {tf}") if value is None else (value < _num(params, "value", 50), f"{tf} RSI {value:.1f}")


@predicate("rsi_divergence", "Regular RSI divergence at the last two swings.", "دایڤێرجێنسی RSI لە دوو لووتکە/نزمایی دواییدا.",
           tf={"type": "string"}, direction={"type": "string"})
def rsi_divergence(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    tf = _tf(ctx, params, "setup")
    candles = ctx.candles.get(tf)
    if not candles:
        return _missing(tf)
    wanted = to_direction(params.get("direction"), ctx)
    if wanted in (None, Direction.NEUTRAL):
        both = _divergence(candles, True) or _divergence(candles, False)
        return both, f"{tf} divergence {'found' if both else 'absent'}"
    found = _divergence(candles, wanted == Direction.BULLISH)
    return found, f"{tf} {wanted.value.lower()} divergence {'found' if found else 'absent'}"


@predicate("volume_spike", "The last closed bar's volume is at least k x the average of the previous n bars "
           "(tick volume on FX/metals).", "ڤۆلیۆمی دوایین مۆم چەند هێندەی تێکڕایە.",
           tf={"type": "string"}, k={"type": "number", "default": 2.0}, n={"type": "integer", "default": 20},
           within_bars={"type": "integer", "default": 1})
def volume_spike(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    tf = _tf(ctx, params, "entry")
    candles = ctx.candles.get(tf)
    n, k, within = _int(params, "n", 20), _num(params, "k", 2.0), max(1, _int(params, "within_bars", 1))
    if not candles or len(candles) < n + within:
        return _missing(tf)
    for offset in range(within):
        end = len(candles) - offset
        base = [c.volume for c in candles[end - 1 - n:end - 1]]
        average = fmean(base) if base else 0.0
        if average > 0 and candles[end - 1].volume >= k * average:
            return True, f"{tf} volume {candles[end - 1].volume:.0f} = {candles[end - 1].volume / average:.1f}x average"
    return False, f"no {tf} volume spike >= {k}x"


_SESSION_NAMES = {"asia": "TOKYO", "asian": "TOKYO", "tokyo": "TOKYO", "ئاسیا": "TOKYO", "london": "LONDON",
                  "لەندەن": "LONDON", "new_york": "NEW_YORK", "newyork": "NEW_YORK", "ny": "NEW_YORK",
                  "نیویۆرک": "NEW_YORK", "nyse": "NYSE"}


@predicate("session_is", "The current time is inside a trading session (asia, london, new_york, overlap).",
           "کاتی ئێستا لە ناو سیشنێکی دیاریکراودایە.",
           session={"type": "string", "enum": ["asia", "london", "new_york", "overlap"]})
def session_is(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    info = session_context(_dt.datetime.fromtimestamp(ctx.now, _dt.UTC) if ctx.now else None)
    name = normalize_ckb(str(params.get("session") or ""), strip_punct=True).replace(" ", "_")
    if name in ("overlap", "london_new_york", "london_ny"):
        return bool(info["london_new_york_overlap"]), f"active: {', '.join(info['active']) or 'none'}"
    wanted = _SESSION_NAMES.get(name)
    if wanted is None:
        return None, f"unknown session {name}"
    return wanted in info["active"], f"active: {', '.join(info['active']) or 'none'}"


# ICT killzones in New York local time (DST handled by zoneinfo).
KILLZONES = {"asia": (20.0, 24.0), "london": (2.0, 5.0), "ny_am": (7.0, 10.0), "ny_pm": (13.5, 16.0),
             "london_close": (10.0, 12.0)}


@predicate("killzone", "The current New York time is inside an ICT killzone (asia, london, ny_am, ny_pm, london_close).",
           "کاتی ئێستا لە ناو کیلزۆنێکی ICT دایە.", name={"type": "string", "enum": list(KILLZONES)})
def killzone(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    from zoneinfo import ZoneInfo
    name = str(params.get("name") or "").lower()
    if name not in KILLZONES:
        return None, f"unknown killzone {name}"
    ny = _dt.datetime.fromtimestamp(ctx.now or _dt.datetime.now().timestamp(), ZoneInfo("America/New_York"))
    hour = ny.hour + ny.minute / 60
    start, end = KILLZONES[name]
    return start <= hour < end, f"New York time {ny:%H:%M}"


@predicate("time_window", "Local time (default Asia/Baghdad) is between start and end (HH:MM).",
           "کاتی ناوخۆیی لە نێوان دوو کاتدایە.",
           start={"type": "string"}, end={"type": "string"}, tz={"type": "string", "default": "Asia/Baghdad"})
def time_window(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    from zoneinfo import ZoneInfo
    try:
        local = _dt.datetime.fromtimestamp(ctx.now, ZoneInfo(str(params.get("tz") or "Asia/Baghdad")))
        start = _dt.time.fromisoformat(str(params["start"]))
        end = _dt.time.fromisoformat(str(params["end"]))
    except (KeyError, ValueError):
        return None, "start/end must be HH:MM"
    now = local.time()
    inside = start <= now < end if start <= end else (now >= start or now < end)
    return inside, f"local time {local:%H:%M}"


@predicate("weekday_in", "Today (UTC) is one of the listed weekdays (mon,tue,wed,thu,fri).",
           "ئەمڕۆ یەکێکە لە ڕۆژە دیاریکراوەکان.", days={"type": "string"})
def weekday_in(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    today = names[_dt.datetime.fromtimestamp(ctx.now, _dt.UTC).weekday()]
    wanted = {d.strip().lower()[:3] for d in str(params.get("days") or "").split(",") if d.strip()}
    return (today in wanted, f"today is {today}") if wanted else (None, "no days given")


_PATTERN_ALIASES = {"engulfing": {"BULLISH_ENGULFING", "BEARISH_ENGULFING"}, "pin_bar": {"HAMMER", "SHOOTING_STAR", "INVERTED_HAMMER", "HANGING_MAN"},
                    "rejection": {"HAMMER", "SHOOTING_STAR"}, "hammer": {"HAMMER"}, "shooting_star": {"SHOOTING_STAR"},
                    "doji": {"DOJI"}, "marubozu": {"MARUBOZU"}}


@predicate("candle_pattern", "A candle pattern (engulfing, pin_bar, hammer, shooting_star, doji, marubozu) closed "
           "within the last N bars, optionally in a direction.", "شێوەیەکی مۆم لە مۆمە دواییەکاندا دروست بوو.",
           tf={"type": "string"}, name={"type": "string"}, direction={"type": "string"},
           within_bars={"type": "integer", "default": 3})
def candle_pattern(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    tf, analysis = _analysis(ctx, params, "entry")
    candles = ctx.candles.get(tf)
    if analysis is None or not candles:
        return _missing(tf)
    name = str(params.get("name") or "engulfing").lower().replace(" ", "_")
    names = _PATTERN_ALIASES.get(name, {name.upper()})
    wanted = to_direction(params.get("direction"), ctx)
    first = len(candles) - _int(params, "within_bars", 3)
    for item in reversed(analysis.get("candlestick_patterns") or []):
        if item["index"] >= first and item["name"] in names and (
                wanted is None or item["direction"] in (wanted.value, Direction.NEUTRAL.value)):
            return True, f"{tf} {item['name'].lower()} {len(candles) - 1 - item['index']} bar(s) ago"
    return False, f"no {name} on {tf} in the last bars"


@predicate("ema_cross", "EMA(fast) crossed EMA(slow) in a direction within the last N bars.",
           "بڕینی دوو مووڤینگ ئەڤرێج (EMA).",
           tf={"type": "string"}, fast={"type": "integer", "default": 20}, slow={"type": "integer", "default": 50},
           direction={"type": "string"}, within_bars={"type": "integer", "default": 3})
def ema_cross(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    tf = _tf(ctx, params, "entry")
    candles = ctx.candles.get(tf)
    fast_n, slow_n = _int(params, "fast", 20), _int(params, "slow", 50)
    if not candles or len(candles) < slow_n + 2:
        return _missing(tf)
    closes = [c.close for c in candles]
    fast, slow = ema(closes, fast_n), ema(closes, slow_n)
    wanted = to_direction(params.get("direction"), ctx)
    for i in range(len(closes) - 1, max(slow_n, len(closes) - _int(params, "within_bars", 3)) - 1, -1):
        a0, b0, a1, b1 = fast[i - 1], slow[i - 1], fast[i], slow[i]
        if None in (a0, b0, a1, b1):
            continue
        if a0 <= b0 and a1 > b1 and wanted in (None, Direction.BULLISH):
            return True, f"{tf} EMA{fast_n} crossed above EMA{slow_n}"
        if a0 >= b0 and a1 < b1 and wanted in (None, Direction.BEARISH):
            return True, f"{tf} EMA{fast_n} crossed below EMA{slow_n}"
    return False, f"no {tf} EMA{fast_n}/{slow_n} cross"


@predicate("price_vs_ema", "Price is above or below EMA(period).", "نرخ لە سەرەوە یان خوارەوەی EMA یە.",
           tf={"type": "string"}, period={"type": "integer", "default": 200},
           side={"type": "string", "enum": ["above", "below", "setup"]})
def price_vs_ema(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    tf = _tf(ctx, params, "bias")
    candles = ctx.candles.get(tf)
    period = _int(params, "period", 200)
    value = latest(ema([c.close for c in candles], period)) if candles else None
    if value is None:
        return None, f"not enough {tf} bars for EMA{period}"
    side = str(params.get("side") or "setup")
    if side == "setup":
        side = "above" if ctx.direction == "long" else "below" if ctx.direction == "short" else ""
    if side not in ("above", "below"):
        return None, "no side given"
    return (ctx.price > value) == (side == "above"), f"{tf} EMA{period} {value:.5g}"


@predicate("price_vs_vwap", "Price is above or below the VWAP of the analysed window.", "نرخ لە سەرەوە یان خوارەوەی VWAP ە.",
           tf={"type": "string"}, side={"type": "string", "enum": ["above", "below", "setup"]})
def price_vs_vwap(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    tf, analysis = _analysis(ctx, params, "setup")
    value = ((analysis or {}).get("indicators") or {}).get("vwap")
    if value is None:
        return _missing(tf)
    side = str(params.get("side") or "setup")
    if side == "setup":
        side = "above" if ctx.direction == "long" else "below" if ctx.direction == "short" else ""
    if side not in ("above", "below"):
        return None, "no side given"
    return (ctx.price > value) == (side == "above"), f"{tf} VWAP {value:.5g}"


@predicate("adx_above", "ADX(14) is above a value (trend strength).", "ADX لە ژمارەیەک بەرزترە (هێزی ترێند).",
           tf={"type": "string"}, value={"type": "number", "default": 25})
def adx_above(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    tf = _tf(ctx, params, "bias")
    candles = ctx.candles.get(tf)
    value = latest(adx(candles)) if candles else None
    if value is None:
        return None, f"not enough {tf} bars for ADX"
    return value > _num(params, "value", 25), f"{tf} ADX {value:.1f}"


@predicate("atr_at_least", "ATR(14) is at least a value in price units (enough volatility).",
           "ATR لانیکەم ئەوەندەیە (جووڵەی پێویست).", tf={"type": "string"}, value={"type": "number"})
def atr_at_least(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    tf, analysis = _analysis(ctx, params, "entry")
    if analysis is None:
        return _missing(tf)
    value = _atr(analysis)
    return value >= _num(params, "value", 0.0), f"{tf} ATR {value:.5g}"


@predicate("displacement", "A strong candle (body >= atr_mult x ATR) closed in a direction within N bars.",
           "مۆمێکی بەهێز (displacement) بە ئاراستەیەک.",
           tf={"type": "string"}, direction={"type": "string"}, atr_mult={"type": "number", "default": 1.5},
           within_bars={"type": "integer", "default": 5})
def displacement(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    tf, analysis = _analysis(ctx, params, "entry")
    candles = ctx.candles.get(tf)
    if analysis is None or not candles:
        return _missing(tf)
    wanted = to_direction(params.get("direction"), ctx)
    need = _num(params, "atr_mult", 1.5) * max(_atr(analysis), 1e-9)
    for candle in candles[-_int(params, "within_bars", 5):]:
        right_way = wanted is None or (candle.bullish if wanted == Direction.BULLISH else not candle.bullish)
        if right_way and candle.body >= need:
            return True, f"{tf} displacement candle body {candle.body:.5g}"
    return False, f"no {tf} displacement candle"


@predicate("opening_range_break", "The session opening range (default London, 15 min) broke in a direction and held.",
           "شکانی مەودای کردنەوەی سیشن.", tf={"type": "string"},
           session={"type": "string", "enum": ["london", "new_york", "tokyo"]},
           minutes={"type": "integer", "default": 15}, direction={"type": "string"})
def opening_range_break(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    tf = _tf(ctx, params, "entry")
    candles = ctx.candles.get(tf)
    if not candles:
        return _missing(tf)
    try:
        result = opening_range(candles, minutes=_int(params, "minutes", 15), session=str(params.get("session") or "london"))
    except ValueError as exc:
        return None, str(exc)
    if not result.get("available"):
        return None, result.get("reason", "no opening range")
    wanted = to_direction(params.get("direction"), ctx)
    broke = result["state"] == "BROKEN" and (wanted is None or result["breakout"]["direction"] == wanted.value)
    return broke, f"{result['session']} opening range {result['state'].lower()}"


@predicate("usdx_trend", "The dollar index trend on a timeframe (default: opposite to the setup, as for gold).",
           "ترێندی ئیندێکسی دۆلار.", tf={"type": "string"},
           direction={"type": "string", "enum": ["up", "down", "opposite"]})
def usdx_trend(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    tf = _tf(ctx, params, "bias")
    analysis = (ctx.extra.get("USDX") or {}).get(tf)
    if analysis is None:
        return None, f"USDX {tf} is not available"
    actual = Direction(analysis["structure"]["trend"])
    raw = str(params.get("direction") or "opposite")
    if raw == "opposite":
        if ctx.direction not in ("long", "short"):
            return None, "no setup direction"
        wanted = Direction.BEARISH if ctx.direction == "long" else Direction.BULLISH
    else:
        wanted = to_direction(raw)
    return actual == wanted, f"USDX {tf} trend is {actual.value.lower()}"


@predicate("rr_at_least", "The planned reward:risk to the first target is at least a value.",
           "ڕێژەی قازانج بە مەترسی لانیکەم ئەوەندەیە.", value={"type": "number", "default": 2.0})
def rr_at_least(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    rr = (ctx.plan or {}).get("rr")
    if rr is None:
        return None, "no stop/target plan yet"
    return float(rr) >= _num(params, "value", 2.0), f"R:R {float(rr):.2f}"


@predicate("spread_below", "The current spread is at most a value in price units.", "سپرێد لە ژمارەیەک کەمترە.",
           value={"type": "number"})
def spread_below(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    if ctx.spread is None:
        return None, "spread unknown"
    return ctx.spread <= _num(params, "value", 0.0), f"spread {ctx.spread:.5g}"


@predicate("no_news_blackout", "Now is outside every manual news blackout window (setting trading.news_blackouts).",
           "ئێستا لە کاتی هەواڵە گرنگەکان نییە.")
def no_news_blackout(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    windows = ctx.settings.get("news_blackouts") or []
    for window in windows:
        try:
            start, end = float(window["start"]), float(window["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if start <= ctx.now <= end:
            return False, f"news blackout: {window.get('note', '')}".strip()
    return True, f"{len(windows)} blackout window(s) configured"


@predicate("price_above", "Price is above a fixed level.", "نرخ لە سەرووی ئاستێکە.", level={"type": "number"})
def price_above(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    if params.get("level") is None:
        return None, "no level"
    return ctx.price > _num(params, "level", 0.0), f"price {ctx.price:.5g}"


@predicate("price_below", "Price is below a fixed level.", "نرخ لە خوارووی ئاستێکە.", level={"type": "number"})
def price_below(ctx: PredicateContext, params: dict[str, Any]) -> Result:
    if params.get("level") is None:
        return None, "no level"
    return ctx.price < _num(params, "level", 0.0), f"price {ctx.price:.5g}"


__all__ = ["KILLZONES", "trading_day_start"]
