"""Deterministic strategy-card predicates on synthetic bars."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from engine_helpers import flat_bars

from sam.trading.engine.analysis import timeframe_analysis
from sam.trading.engine.core import closed_candles
from sam.trading.engine.types import build_batch
from sam.trading.predicates import PREDICATES, PredicateContext, evaluate_card, trading_day_start

MONDAY_10 = datetime(2026, 9, 21, 10, 0, tzinfo=UTC).timestamp()   # London session, NY 06:00
MONDAY_13 = datetime(2026, 9, 21, 13, 0, tzinfo=UTC).timestamp()   # NY 09:00 (ny_am killzone)
MONDAY_03 = datetime(2026, 9, 21, 3, 0, tzinfo=UTC).timestamp()    # Tokyo


def path_bars(points: list[float], *, start: float, step: int = 900, per_leg: int = 4, wick: float = 0.1,
              volume: float = 100.0) -> list[dict]:
    """Bars walking linearly through price pivots (open = previous close)."""
    closes: list[float] = []
    for a, b in zip(points, points[1:]):
        closes += [a + (b - a) * (k + 1) / per_leg for k in range(per_leg)]
    out, prev = [], points[0]
    for i, close in enumerate(closes):
        out.append({"time": int(start) + i * step, "open": prev, "high": max(prev, close) + wick,
                    "low": min(prev, close) - wick, "close": close, "volume": volume})
        prev = close
    return out


def ctx_for(bars_by_tf: dict[str, list[dict]], *, now: float, price: float | None = None, direction: str | None = None,
            roles: dict[str, str] | None = None, plan: dict | None = None) -> PredicateContext:
    analyses, candles = {}, {}
    for tf, bars in bars_by_tf.items():
        batch = build_batch(provider="test", requested_symbol="XAUUSD", resolved_symbol="XAUUSD", timeframe=tf,
                            bars=bars, now=now)
        analyses[tf] = timeframe_analysis(batch)
        candles[tf] = closed_candles(batch)
    last = next(iter(bars_by_tf.values()))[-1]["close"]
    tf0 = next(iter(bars_by_tf))
    return PredicateContext(analyses=analyses, candles=candles, price=price if price is not None else last,
                            direction=direction, now=now, plan=plan,
                            roles=roles or {"bias": tf0, "setup": tf0, "entry": tf0})


def check(pred: str, ctx: PredicateContext, /, **params):
    return PREDICATES[pred].fn(ctx, params)


def test_the_library_has_the_planned_predicates():
    wanted = {"trend_is", "swept", "mss_or_bos", "in_fvg", "in_order_block", "in_ote", "near_level", "rsi_divergence",
              "volume_spike", "session_is", "candle_pattern", "ema_cross", "usdx_trend", "rr_at_least", "killzone",
              "premium_discount", "displacement", "no_news_blackout"}
    assert wanted <= set(PREDICATES)
    assert len(PREDICATES) >= 30
    assert all(spec.description and spec.description_ckb for spec in PREDICATES.values())


def test_an_asian_low_sweep_with_reclaim():
    midnight = datetime(2026, 9, 21, 0, 0, tzinfo=UTC).timestamp()
    asia = flat_bars(28, "M15", end=midnight + 27 * 900, price=100.0, spread=0.5)  # 00:00-07:00, low 99.5
    sweep = [{"time": int(midnight) + 28 * 900, "open": 100.0, "high": 100.2, "low": 99.0, "close": 100.1, "volume": 100},
             {"time": int(midnight) + 29 * 900, "open": 100.1, "high": 100.4, "low": 99.9, "close": 100.3, "volume": 100}]
    ctx = ctx_for({"M15": asia + sweep}, now=midnight + 30 * 900 + 60)
    passed, detail = check("swept", ctx, tf="M15", level="asian_low")
    assert passed is True and "99.5" in detail
    held_below = asia + [{**sweep[0], "close": 98.8}, {**sweep[1], "open": 98.8, "high": 99.2, "low": 98.5, "close": 98.9}]
    ctx = ctx_for({"M15": held_below}, now=midnight + 30 * 900 + 60)
    assert check("swept", ctx, tf="M15", level="asian_low")[0] is False
    assert check("swept", ctx, tf="M15", level="asian_low", reclaim=False)[0] is True


def test_a_bullish_market_structure_shift_after_a_downtrend():
    points = [112, 105, 110, 103, 108, 101, 106, 100, 109]
    bars = path_bars(points, start=MONDAY_10 - 40 * 900, per_leg=5)
    ctx = ctx_for({"M15": bars}, now=bars[-1]["time"] + 900 + 1)
    assert check("mss_or_bos", ctx, tf="M15", direction="up")[0] is True
    assert check("choch", ctx, tf="M15", direction="up")[0] is True
    assert check("bos", ctx, tf="M15", direction="up")[0] is False
    assert check("mss_or_bos", ctx, tf="M15", direction="down", within_bars=3)[0] is False


def test_price_inside_an_active_bullish_fvg():
    start = MONDAY_10 - 40 * 900
    bars = flat_bars(30, "M15", end=start + 29 * 900, price=100.0, spread=0.2)
    t = bars[-1]["time"]
    bars += [{"time": t + 900, "open": 100.0, "high": 100.5, "low": 99.9, "close": 100.4, "volume": 100},
             {"time": t + 1800, "open": 100.4, "high": 103.2, "low": 100.3, "close": 103.0, "volume": 100},
             {"time": t + 2700, "open": 103.0, "high": 103.6, "low": 102.0, "close": 103.5, "volume": 100},
             {"time": t + 3600, "open": 103.5, "high": 103.5, "low": 101.4, "close": 101.6, "volume": 100}]
    ctx = ctx_for({"M15": bars}, now=t + 4500 + 1, price=101.6)
    passed, detail = check("in_fvg", ctx, tf="M15", direction="up")
    assert passed is True and "FVG" in detail
    assert check("in_fvg", ctx, tf="M15", direction="down")[0] is False
    assert check("fvg_present", ctx, tf="M15", direction="up")[0] is True


def test_price_back_inside_a_mitigated_order_block():
    start = MONDAY_10 - 40 * 900
    bars = flat_bars(20, "M15", end=start + 19 * 900, price=100.0, spread=0.5)
    t = bars[-1]["time"]
    shape = [(100.0, 100.2, 99.0, 99.2), (99.2, 103.0, 99.1, 102.8), (102.8, 104.0, 102.5, 103.9)]
    shape += [(104.0, 104.5, 103.5, 104.0)] * 6 + [(104.0, 104.1, 99.6, 99.8)]
    bars += [{"time": t + (i + 1) * 900, "open": o, "high": h, "low": lo, "close": c, "volume": 100}
             for i, (o, h, lo, c) in enumerate(shape)]
    ctx = ctx_for({"M15": bars}, now=bars[-1]["time"] + 901, price=99.8)
    passed, detail = check("in_order_block", ctx, tf="M15", direction="up")
    assert passed is True and "mitigated" in detail
    assert check("in_order_block", ctx, tf="M15", direction="down")[0] is False


def test_volume_spike_against_the_average():
    bars = flat_bars(30, "M5", end=MONDAY_10 - 300, volume=100.0)
    bars[-1] = {**bars[-1], "volume": 350.0}
    ctx = ctx_for({"M5": bars}, now=bars[-1]["time"] + 301)
    passed, detail = check("volume_spike", ctx, tf="M5", k=3, n=20)
    assert passed is True and "3.5x" in detail
    assert check("volume_spike", ctx, tf="M5", k=4, n=20)[0] is False


def test_sessions_and_killzones_follow_the_clock():
    bars = {"M5": flat_bars(30, "M5", end=MONDAY_10 - 300)}
    assert check("session_is", ctx_for(bars, now=MONDAY_10), session="london")[0] is True
    assert check("session_is", ctx_for(bars, now=MONDAY_10), session="asia")[0] is False
    assert check("session_is", ctx_for(bars, now=MONDAY_03), session="ئاسیا")[0] is True
    assert check("killzone", ctx_for(bars, now=MONDAY_13), name="ny_am")[0] is True
    assert check("killzone", ctx_for(bars, now=MONDAY_10), name="ny_am")[0] is False
    assert check("time_window", ctx_for(bars, now=MONDAY_10), start="12:00", end="14:00")[0] is True  # 13:00 Baghdad
    assert check("weekday_in", ctx_for(bars, now=MONDAY_10), days="mon,tue")[0] is True


def test_rr_and_unknowns_are_honest():
    bars = {"M5": flat_bars(30, "M5", end=MONDAY_10 - 300)}
    assert check("rr_at_least", ctx_for(bars, now=MONDAY_10, plan={"rr": 2.2}), value=2)[0] is True
    assert check("rr_at_least", ctx_for(bars, now=MONDAY_10, plan={"rr": 2.2}), value=3)[0] is False
    assert check("rr_at_least", ctx_for(bars, now=MONDAY_10), value=2)[0] is None
    assert check("usdx_trend", ctx_for(bars, now=MONDAY_10, direction="long"))[0] is None
    assert check("trend_is", ctx_for(bars, now=MONDAY_10), tf="H4", direction="up")[0] is None


def test_a_card_mixes_predicates_chart_rules_and_unknown_names():
    bars = {"M5": flat_bars(40, "M5", end=MONDAY_10 - 300)}
    card = {"id": "x", "rules": [
        {"id": "r1", "kind": "filter", "text_ckb": "تەنها لەندەن", "check": {"predicate": "session_is", "params": {"session": "london"}}},
        {"id": "r2", "kind": "setup", "text_ckb": "شێوەی چارت", "check": None},
        {"id": "r3", "kind": "trigger", "check": {"predicate": "no_such_check", "params": {}}},
        {"id": "r4", "kind": "trigger", "check": {"predicate": "rsi_above", "params": {"value": "40"}}}]}
    result = evaluate_card(card, ctx_for(bars, now=MONDAY_10))
    rules = {r["id"]: r for r in result["rules"]}
    assert rules["r1"]["passed"] is True and rules["r1"]["how"] == "predicate"
    assert rules["r2"]["how"] == "llm" and rules["r2"]["passed"] is None
    assert rules["r3"]["how"] == "unsupported" and rules["r3"]["passed"] is None
    assert rules["r4"]["passed"] is True  # flat series RSI = 50 > 40 (param coerced from a string)
    assert result["all_passed"] is False and result["pending"] == ["r2"]


def test_trading_day_rolls_at_five_pm_new_york():
    start = trading_day_start(datetime(2026, 9, 21, 14, 13, tzinfo=UTC).timestamp())
    assert datetime.fromtimestamp(start, UTC) == datetime(2026, 9, 20, 21, 0, tzinfo=UTC)


@pytest.mark.parametrize("name", ["candle_pattern", "ema_cross", "price_vs_ema", "adx_above", "displacement",
                                  "in_ote", "premium_discount", "near_level", "rsi_divergence", "liquidity_pool",
                                  "opening_range_break", "in_supply_demand", "trend_aligned", "price_vs_vwap"])
def test_every_predicate_answers_without_raising(name):
    bars = path_bars([100, 104, 101, 106, 103, 108, 105, 110], start=MONDAY_10 - 200 * 300, step=300, per_leg=25)
    ctx = ctx_for({"M5": bars}, now=bars[-1]["time"] + 301, direction="long")
    passed, detail = PREDICATES[name].fn(ctx, {"tf": "M5"})
    assert passed in (True, False, None) and isinstance(detail, str) and detail
