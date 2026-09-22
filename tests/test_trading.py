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
from sam_backend.trading.types import Candle, Direction, MarketDataBatch, SetupDecision, SetupState, normalize_timeframe


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


# --- a MetaTrader5 package that cannot be loaded ---------------------------------
# `import MetaTrader5` reads MetaTrader5/__init__.py and then loads the _core
# extension. A missing package or a failed DLL load arrives as ImportError; a
# package file the process cannot read arrives as a raw OSError, because the
# import system does not wrap I/O failures. Both mean the same thing here: the
# broker is unavailable. Neither is a reason for a 500, and neither should put
# a filesystem path into an API payload.

UNREADABLE_PACKAGE = r"C:\Users\someone\site-packages\MetaTrader5\__init__.py"


class UnreadableMetaTrader5:
    """A finder/loader pair whose package file raises on read."""

    def find_spec(self, name, path=None, target=None):
        import importlib.util

        if name == "MetaTrader5":
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise PermissionError(13, "Access is denied", UNREADABLE_PACKAGE)


@pytest.fixture()
def unreadable_metatrader(monkeypatch):
    import sys

    monkeypatch.delitem(sys.modules, "MetaTrader5", raising=False)
    monkeypatch.setattr(sys, "meta_path", [UnreadableMetaTrader5(), *sys.meta_path])


def test_an_unreadable_metatrader_package_is_an_unavailable_provider(unreadable_metatrader):
    health = MetaTrader5Provider().health()

    assert health["state"] == "UNAVAILABLE"
    assert "could not be loaded" in health["error"]
    assert "site-packages" not in health["error"] and "someone" not in health["error"]


def test_an_unreadable_metatrader_package_leaks_no_path_through_capabilities(unreadable_metatrader):
    report = MetaTrader5Provider().capabilities("XAUUSD")

    assert report["state"] == "UNAVAILABLE"
    assert "someone" not in report["error"] and "__init__" not in report["error"]


def test_a_missing_metatrader_package_still_reads_as_not_installed(monkeypatch):
    import sys

    class Absent:
        def find_spec(self, name, path=None, target=None):
            if name == "MetaTrader5":
                raise ImportError("No module named 'MetaTrader5'")
            return None

    monkeypatch.delitem(sys.modules, "MetaTrader5", raising=False)
    monkeypatch.setattr(sys, "meta_path", [Absent(), *sys.meta_path])

    assert "not installed" in MetaTrader5Provider().health()["error"]


def test_a_corrupt_metatrader_package_is_not_silently_called_unavailable(monkeypatch):
    """A programmer-class failure inside the package must still surface."""
    import sys

    class Corrupt(UnreadableMetaTrader5):
        def exec_module(self, module):
            raise SyntaxError("invalid syntax")

    monkeypatch.delitem(sys.modules, "MetaTrader5", raising=False)
    monkeypatch.setattr(sys, "meta_path", [Corrupt(), *sys.meta_path])

    with pytest.raises(SyntaxError):
        MetaTrader5Provider().health()


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


# --- the analysis pipeline, branch by branch ---------------------------------
# The pipeline moved into MarketAnalyst with four defects that only surfaced
# late, because most of its branches were reached by nothing. Each moved
# method is driven here through the public facade on synthetic candles, and
# every assertion is on a field the chart or setup code actually reads.

def analysing_service(settings, monkeypatch, *, verified: bool = False) -> TradingService:
    """A service whose only feed is the synthetic one; verified flips the
    quote-timestamp check that decides SUCCESS versus PARTIAL."""
    service = fresh_service(settings)

    def fetch(symbol, timeframe, count):
        result = batch(timeframe=timeframe)
        result.quote_timestamp_verified = verified
        return result

    monkeypatch.setattr(service.market_data.providers["metatrader5"], "fetch", fetch)
    return service


@pytest.mark.parametrize("theory, field, inner", [
    ("snr", "levels", None),                       # _execute_theory: snr branch
    ("smc", "imbalances", None),                   # smc/ict/liquidity branch
    ("ict", "session", None),                      # ict adds the session
    ("wyckoff", "wyckoff", "phase_candidate"),     # _wyckoff
    ("volume_profile", "profiles", None),
    ("vwap", "vwap", None),
    ("price_action", "patterns", None),
    ("order_blocks", "order_blocks", None),
    ("fibonacci", "fibonacci", None),
    ("harmonic", "patterns", None),                # _harmonics -> _candles_for
    ("elliott", "counts", None),                   # _elliott -> _candles_for
    ("sessions", "opening_ranges", None),          # _opening_ranges -> _candles_for
])
def test_each_theory_branch_adds_its_own_evidence_to_the_report(settings, monkeypatch, theory, field, inner):
    service = analysing_service(settings, monkeypatch)

    result = service.analyze("XAUUSD", ["H1", "M15"], [theory])

    assert result.executed, result.error
    output = result.data["theories"][theory]
    assert field in output, f"{theory} produced no {field!r}; keys: {sorted(output)}"
    assert output["interpretation"]["direction"] in {"BULLISH", "BEARISH", "NEUTRAL"}
    if inner:
        assert inner in output[field]
    # Every per-timeframe branch answers for each timeframe it was asked about.
    if theory == "ict":
        assert {"active", "sessions", "london_new_york_overlap"} <= set(output[field])
    elif isinstance(output[field], dict) and theory not in {"wyckoff", "sessions"}:
        assert set(output[field]) == {"H1", "M15"}


def test_wyckoff_names_a_phase_and_refuses_to_confirm_it(settings, monkeypatch):
    service = analysing_service(settings, monkeypatch)

    wyckoff = service.analyze("XAUUSD", ["H1"], ["wyckoff"]).data["theories"]["wyckoff"]["wyckoff"]

    assert wyckoff["timeframe"] == "H1"
    assert len(wyckoff["alternate_interpretations"]) >= 1
    assert wyckoff["phase_candidate"].endswith("UNCONFIRMED")
    assert 0 < wyckoff["confidence"] < 0.5, "an unconfirmed phase must not read as confident"


def test_the_session_study_covers_both_sessions_per_timeframe(settings, monkeypatch):
    service = analysing_service(settings, monkeypatch)

    ranges = service.analyze("XAUUSD", ["M15"], ["sessions"]).data["theories"]["sessions"]["opening_ranges"]

    assert set(ranges) == {"M15"} and set(ranges["M15"]) == {"london", "new_york"}


def test_a_custom_theory_is_evaluated_and_unsupported_predicates_are_not_guessed(settings, monkeypatch):
    """_execute_custom_theory: supported predicates run; unknown ones are named, never executed.

    save_custom_theory refuses an unsupported predicate outright, so the
    runtime branch exists for a row that reached the table some other way;
    it is written straight to the store here to prove the guard holds.
    """
    service = analysing_service(settings, monkeypatch)
    saved = service.database.save_custom_theory("Trend Check", {
        "name": "Trend Check", "description": "d",
        "conditions": [
            {"predicate": "trend_is", "timeframe": "M15", "value": "BULLISH"},
            {"predicate": "rsi_above", "timeframe": "M15", "value": 0},
            {"predicate": "launch_missiles", "timeframe": "M15"},
        ],
        "invalidation": "x", "targets": ["y"],
    })

    output = service.analyze("XAUUSD", ["M15"], ["Trend Check"]).data["theories"]["custom:Trend Check"]

    assert output["theory"]["custom"] is True and output["theory"]["version"] == saved["version"]
    statuses = [item["status"] for item in output["evaluated_conditions"]]
    assert statuses == ["EVALUATED", "EVALUATED", "UNSUPPORTED_PREDICATE"]
    assert output["evaluated_conditions"][1]["passed"] is True, "rsi is always above zero"
    assert output["status"] == "PARTIALLY_AVAILABLE" and "not guessed" in output["warning"]
    assert output["setup_match"] is False, "one unknown predicate means no match is claimed"


def test_a_custom_theory_asking_for_a_missing_timeframe_says_so(settings, monkeypatch):
    service = analysing_service(settings, monkeypatch)
    service.save_custom_theory({"name": "Needs H4", "description": "d", "invalidation": "x", "targets": ["y"],
                                "conditions": [{"predicate": "trend_is", "timeframe": "H4", "value": "BULLISH"}]})

    output = service.analyze("XAUUSD", ["M15"], ["Needs H4"]).data["theories"]["custom:Needs H4"]

    assert output["evaluated_conditions"][0]["status"] == "MISSING_TIMEFRAME"
    assert output["setup_match"] is False


def test_an_unknown_theory_is_reported_unavailable_rather_than_invented(settings, monkeypatch):
    service = analysing_service(settings, monkeypatch)

    output = service.analyze("XAUUSD", ["M15"], ["no_such_theory"]).data["theories"]["no_such_theory"]

    assert output["status"] == "UNAVAILABLE" and "no_such_theory" in output["error"]


def test_an_unverified_feed_completes_as_partial_and_still_becomes_the_latest(settings, monkeypatch):
    """The PARTIAL path: gaps are named in the status and the report is kept."""
    service = analysing_service(settings, monkeypatch, verified=False)

    result = service.analyze("XAUUSD", ["M15"], ["default"])

    assert result.status.value == "PARTIAL" and result.verified is False
    assert result.error == "Analysis completed with verification gaps."
    assert result.data["self_check"]["passed"] is False
    assert service.analyst.latest is result.data


def test_a_verified_feed_completes_as_success(settings, monkeypatch):
    service = analysing_service(settings, monkeypatch, verified=True)

    result = service.analyze("XAUUSD", ["M15"], ["default"])

    assert result.status.value == "SUCCESS", result.error
    assert result.data["self_check"]["passed"] is True
    assert result.error is None


def test_the_report_carries_what_the_chart_and_setup_code_read(settings, monkeypatch):
    """_report, _confidence and _sorani_summary, on the fields downstream consumes."""
    service = analysing_service(settings, monkeypatch)

    report = service.analyze("XAUUSD", ["H1", "M15"], ["default", "snr"]).data

    assert report["symbol"] == "XAUUSD" and report["requested_symbol"] == "XAUUSD"
    assert set(report["timeframes"]) == {"H1", "M15"}
    assert 0.0 <= report["confidence"] <= 1.0
    assert report["setup_state"] in {state.value for state in SetupState}
    assert isinstance(report["support"], list) and isinstance(report["resistance"], list)
    assert set(report["theories"]) == {"default", "snr"}
    # _sorani_summary: the spoken line carries the bias, the decision and
    # what is still missing -- so a listener knows why there is no entry.
    spoken = report["spoken_summary_ckb"]
    assert spoken.startswith("HTF ") and str(report["htf_bias"]).lower() in spoken.lower()
    assert str(report["decision"] or "WAIT") in spoken
    for missing in (report.get("missing_confirmation") or [])[:3]:
        assert str(missing) in spoken
    # Pattern engines rerun on the candles the analysis kept; those must not be serialised.
    assert "_candles" not in str(report)


def test_market_snapshot_reports_quotes_for_each_timeframe(settings, monkeypatch):
    service = analysing_service(settings, monkeypatch, verified=True)

    result = service.market_snapshot("XAUUSD", ["M15", "M5"])

    assert result.executed and result.verified is True
    assert result.data["symbol"] == "XAUUSD" and result.data["feed"] == "Synthetic Test Feed"
    assert result.data["spread"] == pytest.approx(0.10)
    assert set(result.data["timeframes"]) == {"M15", "M5"}
    m15 = result.data["timeframes"]["M15"]
    assert {"open", "high", "low", "close", "time", "metadata"} <= set(m15)
    assert m15["metadata"]["timeframe"] == "M15" and m15["metadata"]["bars"] == 180
    assert result.data["errors"] == {}


def test_an_unverified_snapshot_is_partial_and_says_why(settings, monkeypatch):
    service = analysing_service(settings, monkeypatch, verified=False)

    result = service.market_snapshot("XAUUSD", ["M15"])

    assert result.executed and result.verified is False
    assert result.status.value == "PARTIAL"
    assert "verification gaps" in result.error


def test_market_snapshot_fails_closed_when_no_timeframe_returns_data(settings, monkeypatch):
    from sam_backend.trading.market_data import MarketDataError

    service = fresh_service(settings)

    def refuse(*args, **kwargs):
        raise MarketDataError("feed down")

    monkeypatch.setattr(service.market_data.providers["metatrader5"], "fetch", refuse)
    result = service.market_snapshot("XAUUSD", ["M15"])

    assert not result.executed and result.error_code == "MARKET_DATA_UNAVAILABLE"
    assert "feed down" in " ".join(result.observations)


def test_analysis_stops_when_the_task_is_cancelled(settings, monkeypatch):
    """_raise_if_cancelled: a cancelled token ends the run before theories execute."""
    service = analysing_service(settings, monkeypatch)
    service.cancellation.create("task_x")
    service.cancellation.cancel("task_x", "test")

    with pytest.raises(RuntimeError, match="cancel"):
        service.analyze("XAUUSD", ["M15"], ["default"], task_id="task_x")


def test_a_theory_whose_data_sam_cannot_see_says_so_instead_of_pretending(settings, monkeypatch):
    """_execute_theory: no order-flow feed means no order-flow verdict."""
    service = analysing_service(settings, monkeypatch)

    output = service.analyze("XAUUSD", ["M15"], ["dom"]).data["theories"]["dom"]

    assert output["status"] == "UNAVAILABLE"
    assert output["interpretation"] is None and output["facts"] == []
    assert output["reason"], "the refusal names what is missing"


@pytest.mark.parametrize("slope, expected_candidates, expected_phase", [
    (0.03, ["MARKUP", "REACCUMULATION"], "D_OR_E_UNCONFIRMED"),      # bullish structure
    (-0.03, ["MARKDOWN", "REDISTRIBUTION"], "D_OR_E_UNCONFIRMED"),   # bearish structure
])
def test_wyckoff_reads_the_phase_from_the_structure_direction(settings, monkeypatch, slope, expected_candidates, expected_phase):
    """_wyckoff: each structural direction maps to its own pair of readings."""
    service = fresh_service(settings)
    monkeypatch.setattr(service.market_data.providers["metatrader5"], "fetch",
                        lambda symbol, timeframe, count: batch(candles(slope=slope), timeframe=timeframe))

    wyckoff = service.analyze("XAUUSD", ["H1"], ["wyckoff"]).data["theories"]["wyckoff"]["wyckoff"]

    assert wyckoff["alternate_interpretations"] == expected_candidates
    assert wyckoff["phase_candidate"] == expected_phase


def test_an_unreadable_metatrader_package_does_not_fail_the_status_endpoint(app, unreadable_metatrader):
    from fastapi.testclient import TestClient

    from sam_backend.trading.tradingview import TradingViewState

    absent = TradingViewState(False, [], None, None, None, None, None, False, None, None, None, False, False)
    app.state.trading.tradingview.observe = lambda: absent
    app.state.trading.drawing._observe = lambda: absent

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/trading/status")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["market_data"]["metatrader5"]["state"] == "UNAVAILABLE"
    assert payload["capabilities"]["state"] == "UNAVAILABLE"
    assert "someone" not in response.text and "site-packages" not in response.text
    # The rest of the payload is unaffected by the broker.
    assert "drawing" in payload and "tradingview" in payload
