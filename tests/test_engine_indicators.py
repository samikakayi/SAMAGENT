"""Indicators and per-timeframe analysis (ported from v1 tests/test_trading.py)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from engine_helpers import BASE, v1_candles

from sam.trading.engine.analysis import (candlestick_patterns, entry_hunter, fair_value_gaps, find_swings,
                                         liquidity_analysis, market_structure, session_context, snr_levels,
                                         statistical_summary, timeframe_analysis)
from sam.trading.engine.indicators import (adx, atr, bollinger, ema, ichimoku, keltner, macd, parabolic_sar,
                                           pivot_points, rsi, sma, stochastic, stochastic_rsi, volume_profile, vwap, wma)
from sam.trading.engine.types import Candle, CapabilityState, Direction, MarketDataBatch, SetupDecision, normalize_timeframe
from sam.trading.theories import TradingKnowledgeRegistry


def batch(items: list[Candle] | None = None, timeframe: str = "M1") -> MarketDataBatch:
    items = items or v1_candles()
    return MarketDataBatch(provider="test", requested_symbol="XAUUSD", resolved_symbol="XAUUSD", timeframe=timeframe,
                           candles=items, precision=2, point=0.01, fetched_at=BASE + timedelta(days=2),
                           bid=items[-1].close - 0.05, ask=items[-1].close + 0.05, feed="Synthetic Test Feed")


@pytest.mark.parametrize(("raw", "expected"), [("1m", "M1"), ("15m", "M15"), ("4h", "H4"), ("daily", "D1"),
                                               ("monthly", "MN1"), ("١٥ خولەک", "M15"), ("60", "H1")])
def test_timeframe_aliases(raw, expected):
    assert normalize_timeframe(raw) == expected


def test_candle_rejects_inconsistent_or_negative_data():
    with pytest.raises(ValueError, match="inconsistent"):
        Candle(BASE, 10, 9, 8, 10, 1)
    with pytest.raises(ValueError, match="negative"):
        Candle(BASE, 10, 11, 9, 10, -1)


def test_moving_averages_have_expected_seed_values():
    values = [1, 2, 3, 4, 5]
    assert sma(values, 3) == [None, None, 2, 3, 4]
    assert ema(values, 3)[2] == 2
    assert wma(values, 3)[2] == pytest.approx(14 / 6)


def test_rsi_handles_trending_and_flat_series():
    assert rsi(list(range(30)), 14)[-1] == 100
    assert rsi([5.0] * 30, 14)[-1] == 50


def test_atr_macd_and_adx_output_shapes():
    items = v1_candles()
    assert len(atr(items)) == len(items)
    assert atr(items)[-1] > 0
    assert set(macd([item.close for item in items])) == {"macd", "signal", "histogram"}
    assert len(adx(items)) == len(items)


def test_oscillators_and_channels_stay_well_formed():
    items = v1_candles()
    assert 0 <= stochastic(items)["k"][-1] <= 100
    srsi = stochastic_rsi([item.close for item in items])
    assert srsi[-1] is None or 0 <= srsi[-1] <= 100
    bands = bollinger([item.close for item in items])
    assert bands["lower"][-1] <= bands["middle"][-1] <= bands["upper"][-1]
    channels = keltner(items)
    assert channels["lower"][-1] < channels["middle"][-1] < channels["upper"][-1]


def test_sar_ichimoku_and_vwap_shapes():
    items = v1_candles()
    assert len(parabolic_sar(items)) == len(items)
    cloud = ichimoku(items)
    assert set(cloud) == {"conversion", "base", "span_a", "span_b", "lagging"}
    assert all(len(series) == len(items) for series in cloud.values())
    values = vwap(items)
    assert len(values) == len(items) and values[-1] is not None


def test_volume_profile_and_pivots_satisfy_invariants():
    profile = volume_profile(v1_candles(), bins=24)
    assert profile["val"] <= profile["poc"] <= profile["vah"]
    assert len(profile["bins"]) == 24
    standard = pivot_points(110, 90, 100)
    assert standard["P"] == 100
    assert standard["S1"] < standard["P"] < standard["R1"]
    with pytest.raises(ValueError, match="Unsupported"):
        pivot_points(110, 90, 100, "invented")


def test_swing_detection_uses_confirmed_neighbors():
    prices = [1, 2, 5, 2, 1, 2, 0, 2, 3]
    items = [Candle(BASE + timedelta(minutes=i), v, v + 0.2, v - 0.2, v, 10) for i, v in enumerate(prices)]
    swings = find_swings(items, 1, 1)
    assert swings["highs"][0]["index"] == 2
    assert swings["lows"][0]["index"] == 4


def test_market_structure_is_neutral_when_history_is_insufficient():
    result = market_structure(v1_candles(8), "M1")
    assert result["trend"] == Direction.NEUTRAL.value
    assert "Insufficient" in result["reason"]


def test_fair_value_gap_requires_a_real_three_candle_gap():
    items = [Candle(BASE, 100, 101, 99, 100, 10), Candle(BASE + timedelta(minutes=1), 101, 103, 100, 102, 10),
             Candle(BASE + timedelta(minutes=2), 104, 105, 103, 104, 10)]
    gaps = fair_value_gaps(items, "M1")
    assert gaps[0]["direction"] == Direction.BULLISH.value
    assert gaps[0]["lower"] == 101 and gaps[0]["upper"] == 103


def test_candlestick_engulfing_is_observation_not_setup():
    items = [Candle(BASE, 101, 101.2, 99.8, 100, 10), Candle(BASE + timedelta(minutes=1), 99.8, 101.5, 99.5, 101.2, 10)]
    pattern = candlestick_patterns(items)[-1]
    assert pattern["name"] == "BULLISH_ENGULFING"
    assert pattern["context_required"] is True


def test_snr_and_liquidity_return_scored_typed_evidence():
    items = v1_candles()
    levels = snr_levels(items, "H1")
    assert levels and len(levels) <= 16
    assert all(level.score > 0 and level.timeframe == "H1" for level in levels)
    assert {"equal_highs", "equal_lows", "sweeps"} <= set(liquidity_analysis(items, "H1"))


def test_session_context_applies_summer_dst_without_external_tzdata():
    result = session_context(datetime(2025, 6, 2, 12, 0, tzinfo=UTC))
    london = next(item for item in result["sessions"] if item["name"] == "LONDON")
    new_york = next(item for item in result["sessions"] if item["name"] == "NEW_YORK")
    assert london["local_time"].endswith("+01:00")
    assert new_york["local_time"].endswith("-04:00")


def test_statistical_summary_is_deterministic():
    first, second = statistical_summary(v1_candles()), statistical_summary(v1_candles())
    assert first == second
    assert first["observations"] == 100
    assert first["rolling_range"] > 0


def test_timeframe_analysis_retains_feed_metadata_and_evidence():
    result = timeframe_analysis(batch())
    assert result["metadata"]["feed"] == "Synthetic Test Feed"
    assert result["metadata"]["current_forming_bar_excluded"] is False
    assert result["evidence"][0]["source"] == "market_structure"
    assert result["current_price"] == pytest.approx((batch().bid + batch().ask) / 2)
    assert result["order_blocks"] is not None and "fibonacci" in result


def test_entry_hunter_defaults_to_wait_without_htf_direction():
    analysis = timeframe_analysis(batch())
    analysis["structure"] = {**analysis["structure"], "trend": Direction.NEUTRAL.value}
    result = entry_hunter({"H1": analysis, "M1": analysis})
    assert result["decision"] == SetupDecision.WAIT.value
    assert "Directional higher-timeframe structure" in result["missing_confirmation"]


def test_knowledge_registry_exposes_honest_capability_states():
    knowledge = TradingKnowledgeRegistry()
    assert knowledge.get("snr").health == CapabilityState.AVAILABLE
    assert knowledge.get("footprint").health == CapabilityState.UNAVAILABLE
    assert knowledge.get("wyckoff").health == CapabilityState.PARTIALLY_AVAILABLE
    with pytest.raises(KeyError):
        knowledge.get("no_such_theory")
