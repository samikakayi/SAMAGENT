from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from sam_backend.cancellation import CancellationManager
from sam_backend.contracts import CapabilityState, ExecutionStatus, StandardResult
from sam_backend.db import Database
from sam_backend.trading.analysis import (
    candlestick_patterns,
    entry_hunter,
    fair_value_gaps,
    find_swings,
    liquidity_analysis,
    market_structure,
    session_context,
    snr_levels,
    statistical_summary,
    timeframe_analysis,
)
from sam_backend.trading.indicators import (
    adx,
    atr,
    bollinger,
    ema,
    ichimoku,
    keltner,
    macd,
    parabolic_sar,
    pivot_points,
    rsi,
    sma,
    stochastic,
    stochastic_rsi,
    volume_profile,
    vwap,
    wma,
)
from sam_backend.trading.market_data import MarketDataService, MetaTrader5Provider
from sam_backend.trading.registry import TradingKnowledgeRegistry, build_skill_registry
from sam_backend.trading.service import TradingService
from sam_backend.trading.types import Candle, Direction, MarketDataBatch, SetupDecision, normalize_timeframe


BASE = datetime(2025, 1, 6, 0, 0, tzinfo=UTC)


def candles(count: int = 180, *, slope: float = 0.03, wave: float = 1.6) -> list[Candle]:
    result = []
    for index in range(count):
        middle = 100 + index * slope + math.sin(index / 4) * wave
        opening = middle - math.sin(index / 3) * 0.25
        closing = middle + math.cos(index / 5) * 0.25
        result.append(Candle(
            BASE + timedelta(minutes=index),
            opening,
            max(opening, closing) + 0.45,
            min(opening, closing) - 0.45,
            closing,
            100 + index,
        ))
    return result


def batch(items: list[Candle] | None = None, timeframe: str = "M1") -> MarketDataBatch:
    items = items or candles()
    return MarketDataBatch(
        provider="test",
        requested_symbol="XAUUSD",
        resolved_symbol="XAUUSD",
        timeframe=timeframe,
        candles=items,
        precision=2,
        point=0.01,
        fetched_at=BASE + timedelta(days=2),
        bid=items[-1].close - 0.05,
        ask=items[-1].close + 0.05,
        feed="Synthetic Test Feed",
    )


@pytest.mark.parametrize(("raw", "expected"), [("1m", "M1"), ("15m", "M15"), ("4h", "H4"), ("daily", "D1"), ("monthly", "MN1")])
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
    items = candles()
    assert len(atr(items)) == len(items)
    assert atr(items)[-1] > 0
    assert set(macd([item.close for item in items])) == {"macd", "signal", "histogram"}
    assert len(adx(items)) == len(items)


def test_oscillators_and_channels_stay_well_formed():
    items = candles()
    stochastic_values = stochastic(items)
    assert 0 <= stochastic_values["k"][-1] <= 100
    srsi = stochastic_rsi([item.close for item in items])
    assert srsi[-1] is None or 0 <= srsi[-1] <= 100
    bands = bollinger([item.close for item in items])
    assert bands["lower"][-1] <= bands["middle"][-1] <= bands["upper"][-1]
    channels = keltner(items)
    assert channels["lower"][-1] < channels["middle"][-1] < channels["upper"][-1]


def test_sar_ichimoku_and_vwap_shapes():
    items = candles()
    assert len(parabolic_sar(items)) == len(items)
    cloud = ichimoku(items)
    assert set(cloud) == {"conversion", "base", "span_a", "span_b", "lagging"}
    assert all(len(series) == len(items) for series in cloud.values())
    values = vwap(items)
    assert len(values) == len(items) and values[-1] is not None


def test_volume_profile_and_pivots_satisfy_invariants():
    profile = volume_profile(candles(), bins=24)
    assert profile["val"] <= profile["poc"] <= profile["vah"]
    assert len(profile["bins"]) == 24
    standard = pivot_points(110, 90, 100)
    assert standard["P"] == 100
    assert standard["S1"] < standard["P"] < standard["R1"]
    with pytest.raises(ValueError, match="Unsupported"):
        pivot_points(110, 90, 100, "invented")


def test_swing_detection_uses_confirmed_neighbors():
    prices = [1, 2, 5, 2, 1, 2, 0, 2, 3]
    items = [Candle(BASE + timedelta(minutes=i), value, value + 0.2, value - 0.2, value, 10) for i, value in enumerate(prices)]
    swings = find_swings(items, 1, 1)
    assert swings["highs"][0]["index"] == 2
    assert swings["lows"][0]["index"] == 4


def test_market_structure_is_neutral_when_history_is_insufficient():
    result = market_structure(candles(8), "M1")
    assert result["trend"] == Direction.NEUTRAL.value
    assert "Insufficient" in result["reason"]


def test_fair_value_gap_requires_a_real_three_candle_gap():
    items = [
        Candle(BASE, 100, 101, 99, 100, 10),
        Candle(BASE + timedelta(minutes=1), 101, 103, 100, 102, 10),
        Candle(BASE + timedelta(minutes=2), 104, 105, 103, 104, 10),
    ]
    gaps = fair_value_gaps(items, "M1")
    assert gaps[0]["direction"] == Direction.BULLISH.value
    assert gaps[0]["lower"] == 101 and gaps[0]["upper"] == 103


def test_candlestick_engulfing_is_observation_not_setup():
    items = [
        Candle(BASE, 101, 101.2, 99.8, 100, 10),
        Candle(BASE + timedelta(minutes=1), 99.8, 101.5, 99.5, 101.2, 10),
    ]
    pattern = candlestick_patterns(items)[-1]
    assert pattern["name"] == "BULLISH_ENGULFING"
    assert pattern["context_required"] is True


def test_snr_and_liquidity_return_scored_typed_evidence():
    items = candles()
    levels = snr_levels(items, "H1")
    assert levels and len(levels) <= 16
    assert all(level.score > 0 and level.timeframe == "H1" for level in levels)
    liquidity = liquidity_analysis(items, "H1")
    assert set(("equal_highs", "equal_lows", "sweeps")).issubset(liquidity)


def test_session_context_applies_summer_dst_without_external_tzdata():
    result = session_context(datetime(2025, 6, 2, 12, 0, tzinfo=UTC))
    london = next(item for item in result["sessions"] if item["name"] == "LONDON")
    new_york = next(item for item in result["sessions"] if item["name"] == "NEW_YORK")
    assert london["local_time"].endswith("+01:00")
    assert new_york["local_time"].endswith("-04:00")


def test_statistical_summary_is_deterministic():
    first = statistical_summary(candles())
    second = statistical_summary(candles())
    assert first == second
    assert first["observations"] == 100
    assert first["rolling_range"] > 0


def test_timeframe_analysis_retains_feed_metadata_and_evidence():
    result = timeframe_analysis(batch())
    assert result["metadata"]["feed"] == "Synthetic Test Feed"
    assert result["metadata"]["current_forming_bar_excluded"] is False
    assert result["evidence"][0]["source"] == "market_structure"
    assert result["current_price"] == pytest.approx((batch().bid + batch().ask) / 2)


def test_entry_hunter_defaults_to_wait_without_htf_direction():
    analysis = timeframe_analysis(batch())
    analysis["structure"] = {**analysis["structure"], "trend": Direction.NEUTRAL.value}
    result = entry_hunter({"H1": analysis, "M1": analysis})
    assert result["decision"] == SetupDecision.WAIT.value
    assert result["entry"] if "entry" in result else True
    assert "Directional higher-timeframe structure" in result["missing_confirmation"]


def test_standard_result_never_equates_execution_with_verification():
    partial = StandardResult(ExecutionStatus.PARTIAL, True, False, data={"attempted": True}, error="not independently observed")
    payload = partial.as_dict()
    assert payload["executed"] is True
    assert payload["verified"] is False
    assert payload["status"] == "PARTIAL"


def test_knowledge_and_skill_registries_expose_honest_capability_states():
    knowledge = TradingKnowledgeRegistry()
    assert knowledge.get("snr").health == CapabilityState.AVAILABLE
    assert knowledge.get("footprint").health == CapabilityState.UNAVAILABLE
    assert knowledge.get("wyckoff").health == CapabilityState.PARTIALLY_AVAILABLE
    skills = build_skill_registry().list()
    assert any(item["id"] == "entry_hunter" for item in skills)
    assert any(item["id"] == "tradingview_timeframe" and item["health"] == "PARTIALLY_AVAILABLE" for item in skills)


def test_custom_theory_validation_and_versioning(settings):
    settings.prepare()
    service = TradingService(settings, Database(settings.database_path), CancellationManager())
    bad = service.validate_custom_theory({"name": "unsafe", "conditions": [{"predicate": "execute_python"}]})
    assert bad["valid"] is False
    definition = {
        "name": "Confirmed M5 Structure",
        "description": "Requires an M5 bullish trend.",
        "conditions": [{"predicate": "trend_is", "timeframe": "M5", "value": "BULLISH"}],
        "invalidation": "M5 structural low",
        "targets": ["next resistance"],
    }
    first = service.save_custom_theory(definition)
    second = service.save_custom_theory(definition)
    assert second["version"] == first["version"] + 1


def test_setup_creation_requires_a_completed_analysis(settings):
    settings.prepare()
    service = TradingService(settings, Database(settings.database_path), CancellationManager())
    with pytest.raises(ValueError, match="Run a market analysis"):
        service.create_setup_from_last_analysis()


def test_market_data_unknown_provider_fails_closed():
    result = MarketDataService().get_ohlcv("XAUUSD", "M1", provider="invented")
    assert result.status == ExecutionStatus.FAILED
    assert result.error_code == "PROVIDER_NOT_FOUND"


def test_metatrader_symbol_ranking_prefers_exact_visible_name():
    assert MetaTrader5Provider._candidate_score("XAUUSD", "XAUUSD", True) < MetaTrader5Provider._candidate_score("XAUUSD.a", "XAUUSD", True)


# --- one owner for the latest report -------------------------------------------
# The chart draws from the latest analysis and a monitored setup is created
# from it. Both must read the same object from the one place that writes it.

def fresh_service(settings) -> TradingService:
    settings.prepare()
    return TradingService(settings, Database(settings.database_path), CancellationManager())


REPORT = {
    "symbol": "XAUUSD", "feed": "test", "setup_state": "NO_SETUP", "theories": {"default": {}},
    "long_scenario": {"direction": "BULLISH"}, "entry": 100.0, "stop": 99.0, "tp1": 101.0,
    "rr": 1.0, "invalidation": "x", "confidence": 0.5, "support": [], "resistance": [],
}


def test_the_facade_reads_the_latest_report_from_its_owner(settings):
    service = fresh_service(settings)
    assert service.analyst.latest is None

    service.analyst.latest = REPORT
    setup = service.create_setup_from_last_analysis("default")

    assert setup["symbol"] == "XAUUSD" and setup["theory"] == "default"
    assert not hasattr(service, "_last_report"), "the facade keeps no copy of its own"


def test_a_failed_fetch_does_not_replace_the_latest_report(settings, monkeypatch):
    """An analysis that cannot start leaves the last good view of the market."""
    from sam_backend.trading.market_data import MarketDataError

    service = fresh_service(settings)
    service.analyst.latest = REPORT

    def refuse(*args, **kwargs):
        raise MarketDataError("feed is down")

    monkeypatch.setattr(service.market_data.providers["metatrader5"], "fetch", refuse)
    result = service.analyze("XAUUSD", ["M15"])

    assert not result.verified
    assert service.analyst.latest is REPORT, "a hard failure must not clobber the last report"


def test_analysis_public_surface_is_unchanged():
    """Callers of TradingService keep working; the pipeline moved, the API did not."""
    import inspect

    methods = {name for name, _ in inspect.getmembers(TradingService, inspect.isfunction) if not name.startswith("_")}
    assert {
        "analyze", "market_snapshot", "route_natural_intent", "create_setup_from_last_analysis",
        "create_setup_from_analysis", "draw_analysis", "draw_annotation", "draw_two_anchor",
        "backtest", "list_entry_triggers", "poll_monitors", "status", "refresh_permissions",
        "gann_analysis", "pitchfork_analysis", "draw_gann_fan", "draw_pitchfork",
        "list_drawings", "clear_drawings", "set_layer_visibility", "calibrate_chart",
        "verify_calibration", "validate_custom_theory", "save_custom_theory",
    } <= methods
    # The defaults callers relied on survive the delegation.
    assert inspect.signature(TradingService.analyze).parameters["symbol"].default == "XAUUSD"
    assert inspect.signature(TradingService.market_snapshot).parameters["symbol"].default == "XAUUSD"


def test_a_full_analysis_builds_a_report_and_becomes_the_latest(settings, monkeypatch):
    """The whole pipeline on synthetic candles, through the public facade.

    This is the path that had no test when the pipeline moved, and the one a
    stale reference inside it would only have broken at runtime against a
    live feed. It also proves the analysis -> setup chain end to end.
    """
    service = fresh_service(settings)
    monkeypatch.setattr(service.market_data.providers["metatrader5"], "fetch",
                        lambda symbol, timeframe, count: batch(timeframe=timeframe))

    result = service.analyze("XAUUSD", ["M15", "M5"], ["default"])

    assert result.executed, result.error
    report = service.analyst.latest
    assert report is not None and report is result.data, "the report the caller got is the one that is kept"
    assert report["symbol"] == "XAUUSD" and "confidence" in report and "spoken_summary_ckb" in report

    # The record group reads the same object the analysis wrote.
    setup = service.create_setup_from_last_analysis("default")
    assert setup["symbol"] == "XAUUSD"
    # And the chart's source for drawing is that report too.
    assert service.draw_analysis().error_code != "NO_ANALYSIS"
