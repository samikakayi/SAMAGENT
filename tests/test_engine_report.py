"""Engine pipeline and report (v1 analyst tests adapted) + the four audit fixes:
UTC freshness, real order blocks, entry/stop/tp keys read by the drawing
plan, Sorani summaries."""

from __future__ import annotations

import re
import time

import pytest
from engine_helpers import wave_bars

from sam.textnorm import is_arabic_script
from sam.trading.engine.core import Engine, apply_strategy
from sam.trading.engine.drawplan import build_draw_plan
from sam.trading.engine.sorani import full_text, spoken_summary, trend_sentence
from sam.trading.engine.types import SetupState, build_batch, is_weekend_close

NOW = 1_790_000_000.0  # Monday 2026-09-21 14:13 UTC -- fixed so tests are deterministic
TFS = ["H1", "M15"]


def bars_for(tfs=TFS, *, end=NOW, slope=0.03, base=100.0, shift=0):
    return {tf: [{**b, "time": b["time"] + shift} for b in wave_bars(180, tf, end=end, base=base, slope=slope)]
            for tf in tfs}


def analyse(tfs=TFS, *, theories=None, strategy=None, bars=None, tick=True, now=NOW, **kw):
    bars = bars or bars_for(tfs)
    last = bars[tfs[-1]][-1]["close"]
    tick_data = {"bid": last - 0.05, "ask": last + 0.05, "time": now - 1, "spread": 0.1} if tick else None
    return Engine().analyze_sync("XAUUSD", tfs, bars, strategy=strategy, theories=theories, data_source="mt5",
                                 sources={tf: "mt5" for tf in tfs}, tick=tick_data,
                                 meta={"name": "XAUUSD", "digits": 2, "point": 0.01}, now=now, **kw)


def test_true_utc_bars_pass_freshness_and_raw_broker_time_is_flagged():
    """v1 defect 1: broker server time (UTC+3) read as UTC made every fetch 'future'."""
    good = analyse()
    assert good["self_check_passed"] is True and good["stale"] is False
    broker_time = analyse(bars=bars_for(shift=10800))
    failed = {c["name"] for c in broker_time["checks"] if not c["passed"]}
    assert "timestamps_not_future" in failed
    assert broker_time["verdict"] == "WAIT" and broker_time["entry"] is None


def test_weekend_staleness_is_reported_as_a_closed_market():
    saturday = NOW + 5 * 86400  # Saturday 14:13 UTC
    assert is_weekend_close(saturday) and not is_weekend_close(NOW)
    friday_bars = bars_for(end=saturday - 18 * 3600)  # last bars Friday ~20:13 UTC
    report = analyse(bars=friday_bars, now=saturday, tick=False)
    assert report["stale"] is True and report["market_closed"] is True
    assert "داخراوە" in report["summary_ckb"]
    batch = build_batch(provider="mt5", requested_symbol="XAUUSD", resolved_symbol="XAUUSD", timeframe="M15",
                        bars=friday_bars["M15"], now=saturday)
    assert batch.market_closed and not batch.data_quality_verified


@pytest.mark.parametrize("theory, field, inner", [
    ("snr", "levels", None), ("smc", "imbalances", None), ("ict", "session", None),
    ("wyckoff", "wyckoff", "phase_candidate"), ("volume_profile", "profiles", None), ("vwap", "vwap", None),
    ("price_action", "patterns", None), ("order_blocks", "order_blocks", None), ("fibonacci", "fibonacci", None),
    ("harmonic", "patterns", None), ("elliott", "counts", None), ("sessions", "opening_ranges", None),
])
def test_each_theory_branch_adds_its_own_evidence_to_the_report(theory, field, inner):
    output = analyse(theories=[theory])["theories"][theory]
    assert field in output, f"{theory} produced no {field!r}; keys: {sorted(output)}"
    assert output["interpretation"]["direction"] in {"BULLISH", "BEARISH", "NEUTRAL"}
    if inner:
        assert inner in output[field]
    if theory == "ict":
        assert {"active", "sessions", "london_new_york_overlap"} <= set(output[field])
    elif isinstance(output[field], dict) and theory not in {"wyckoff", "sessions"}:
        assert set(output[field]) == set(TFS)


def test_theories_are_found_by_sorani_name():
    output = analyse(theories=["وایکۆف"])["theories"]
    assert "wyckoff" in output


def test_wyckoff_names_a_phase_and_refuses_to_confirm_it():
    wyckoff = analyse(["H1"], theories=["wyckoff"])["theories"]["wyckoff"]["wyckoff"]
    assert wyckoff["timeframe"] == "H1"
    assert wyckoff["phase_candidate"].endswith("UNCONFIRMED")
    assert 0 < wyckoff["confidence"] < 0.5


@pytest.mark.parametrize("slope, expected_candidates", [(0.03, ["MARKUP", "REACCUMULATION"]),
                                                        (-0.03, ["MARKDOWN", "REDISTRIBUTION"])])
def test_wyckoff_reads_the_phase_from_the_structure_direction(slope, expected_candidates):
    report = analyse(["H1"], theories=["wyckoff"], bars=bars_for(["H1"], slope=slope))
    assert report["theories"]["wyckoff"]["wyckoff"]["alternate_interpretations"] == expected_candidates


def test_the_session_study_covers_both_sessions_per_timeframe():
    ranges = analyse(["M15"], theories=["sessions"])["theories"]["sessions"]["opening_ranges"]
    assert set(ranges) == {"M15"} and set(ranges["M15"]) == {"london", "new_york"}


def test_unknown_and_unavailable_theories_say_so_instead_of_pretending():
    report = analyse(theories=["no_such_theory", "dom"])
    assert report["theories"]["no_such_theory"]["status"] == "UNAVAILABLE"
    dom = report["theories"]["dom"]
    assert dom["status"] == "UNAVAILABLE" and dom["interpretation"] is None and dom["reason"]


def test_the_report_carries_every_contract_key():
    report = analyse()
    for key in ("symbol", "timeframes", "price", "data_source", "verdict", "direction", "entry", "stop", "targets",
                "rr", "levels", "zones", "order_blocks", "trend", "strategy", "checks", "summary_ckb", "drawn",
                "invalidation", "tp1", "tp2", "tp3", "text_ckb"):
        assert key in report, key
    assert report["verdict"] in ("WAIT", "NO_TRADE", "SETUP")
    assert set(report["trend"]) == set(TFS) and set(report["trend"].values()) <= {"up", "down", "range"}
    assert report["setup_state"] in {s.value for s in SetupState}
    assert all(level["kind"] in ("support", "resistance") and 0 < level["strength"] <= 1 for level in report["levels"])
    assert "_candles" not in str(report)


def test_real_order_blocks_are_reported():
    """v1 defect 4: the report hard-coded an empty order-block list."""
    report = analyse()
    assert report["order_blocks"], "detection finds blocks on this series; the report must carry them"
    block = report["order_blocks"][0]
    assert block["kind"] in ("bullish", "bearish") and block["low"] <= block["high"] and block["tf"] in TFS
    assert any(zone["kind"] == "order_block" for zone in report["zones"]) or all(
        b["state"] == "BREAKER" for b in report["order_blocks"])


def test_the_summary_is_proper_sorani_without_english_jargon():
    report = analyse()
    spoken = report["summary_ckb"]
    assert is_arabic_script(spoken)
    for english in ("HTF", "WAIT", "bullish", "bearish", "Entry", "NO TRADE"):
        assert english not in spoken
    sentences = [s for s in re.split(r"\.(?:\s|$)", spoken) if s.strip()]  # decimals are not ends
    assert len(sentences) <= 3
    assert "ئەمە تەنها شیکارییە" in report["text_ckb"]


def test_trend_sentences_group_timeframes():
    assert trend_sentence({"H1": "down", "M15": "down"}) == "ترێند لە هەموو تایمفرەیمەکاندا بەرەو خوارەوەیە"
    # «لە هیچ کاتێکدا» reads as "never" (review 2026-09-24)
    assert trend_sentence({"H1": "range", "M15": "range"}) == "لە هیچ تایمفرەیمێکدا ترێندی ڕوون نییە"
    mixed = trend_sentence({"H1": "up", "M15": "range"})
    assert "یەک کاتژمێر" in mixed and "بێ ئاراستە" in mixed


SETUP_REPORT = {
    "symbol": "XAUUSD", "price": 2650.0, "digits": 2, "verdict": "SETUP", "direction": "long", "entry": 2650.0,
    "stop": 2645.0, "invalidation": 2645.5, "tp1": 2660.0, "tp2": 2668.0, "tp3": None, "rr": 2.0,
    "trend": {"H1": "up"}, "support": [{"price": 2644.0, "tf": "H1"}], "resistance": [{"price": 2661.0, "tf": "H1"}],
    "zones": [{"kind": "order_block", "tf": "H1", "low": 2640.0, "high": 2643.0, "direction": "bullish", "time": 1_789_990_000}],
    "stale": False, "missing_confirmation": [], "checks": [], "data_source": "mt5",
}


def test_the_drawing_plan_reads_entry_stop_and_targets():
    """v1 defect 2: the plan read report['setup'], so entry/stop/TP were never drawn."""
    items = build_draw_plan(SETUP_REPORT, "full")
    position = next(item for item in items if item["kind"] == "long_position")
    assert [p["price"] for p in position["points"]] == [2650.0, 2645.0, 2660.0]
    rays = {item["text"]: item["points"][0]["price"] for item in items if item["kind"] == "horizontal_ray"}
    assert rays["ستۆپ"] == 2645.0 and rays["ئامانجی 1"] == 2660.0 and rays["ئامانجی 2"] == 2668.0
    assert any(item["kind"] == "rectangle" and item["points"][0]["time"] == 1_789_990_000 for item in items)
    levels_only = build_draw_plan(SETUP_REPORT, "levels")
    assert {item["kind"] for item in levels_only} == {"horizontal_line"}
    shifted = build_draw_plan(SETUP_REPORT, "levels", price_offset=1.5)
    assert shifted[0]["points"][0]["price"] == 2661.0 + 1.5
    # Closed market / stale feed: historic levels and zones are drawn, never entry/stop/targets
    # (review 2026-09-24: a Saturday "draw S/R" drew nothing).
    closed = build_draw_plan({**SETUP_REPORT, "stale": True, "market_closed": True}, "full")
    assert {item["kind"] for item in closed} == {"horizontal_line", "rectangle"}
    wait = build_draw_plan({**SETUP_REPORT, "verdict": "WAIT"}, "full")
    assert not any(item["kind"].endswith("_position") for item in wait)


def test_setup_and_no_trade_summaries():
    setup = spoken_summary(SETUP_REPORT)
    assert "چوونەژوورەوە 2650" in setup and "ستۆپ 2645" in setup and "ئامانجی یەکەم 2660" in setup
    no_trade = spoken_summary({**SETUP_REPORT, "verdict": "NO_TRADE", "rr": 0.8})
    assert "کاتی مامەڵە نییە" in no_trade and "0.8" in no_trade
    assert "چوونەژوورەوە" in full_text(SETUP_REPORT)


def test_a_card_decides_the_verdict_and_unknown_rules_never_pass():
    card = {"id": "c1", "title_ckb": "تاقیکردنەوە", "risk": {"target_rr": 2.0},
            "rules": [{"id": "r1", "kind": "bias", "text_ckb": "ترێندی کاتژمێر سەرەوە", "text_en": "H1 up",
                       "check": {"predicate": "trend_is", "params": {"tf": "H1", "direction": "up"}}},
                      {"id": "r2", "kind": "setup", "text_ckb": "دیسپلەیسمێنتی پاک", "text_en": "clean displacement",
                       "check": None}]}
    report = analyse(strategy=card)
    rules = {r["id"]: r for r in report["strategy"]["rules"]}
    assert rules["r1"]["passed"] is True and rules["r1"]["how"] == "predicate"
    assert rules["r2"]["passed"] is None and rules["r2"]["how"] == "llm"
    assert report["verdict"] == "WAIT" and report["entry"] is None
    # A vision verdict for r2 turns it into a setup with the card's own 1:2 target.
    rules["r2"].update(passed=True, how="vision")
    apply_strategy(report, report["strategy"], report["strategy_plan"], 1.5)
    assert report["verdict"] == "SETUP" and report["direction"] == "long"
    assert report["rr"] == 2.0 and report["tp1"] == pytest.approx(report["entry"] + 2 * (report["entry"] - report["stop"]))
    # A failed filter rule forbids trading.
    rules["r2"].update(kind="filter", passed=False)
    apply_strategy(report, report["strategy"], report["strategy_plan"], 1.5)
    assert report["verdict"] == "NO_TRADE" and report["entry"] is None


def test_unsupported_predicates_and_missing_timeframes_are_not_guessed():
    card = {"id": "c2", "rules": [
        {"id": "r1", "kind": "setup", "check": {"predicate": "launch_missiles", "params": {}}},
        {"id": "r2", "kind": "bias", "check": {"predicate": "trend_is", "params": {"tf": "H4", "direction": "up"}}},
        {"id": "r3", "kind": "trigger", "check": {"predicate": "rsi_above", "params": {"tf": "M15", "value": 0}}}]}
    rules = {r["id"]: r for r in analyse(strategy=card)["strategy"]["rules"]}
    assert rules["r1"]["how"] == "unsupported" and rules["r1"]["passed"] is None
    assert rules["r2"]["passed"] is None and "H4" in rules["r2"]["detail"]
    assert rules["r3"]["passed"] is True


def test_engine_compute_stays_fast_for_four_timeframes():
    tfs = ["H1", "M15", "M5", "M1"]
    bars = {tf: wave_bars(600, tf, end=NOW, base=2600.0) for tf in tfs}
    started = time.perf_counter()
    analyse(tfs, bars=bars)
    assert time.perf_counter() - started < 1.5
