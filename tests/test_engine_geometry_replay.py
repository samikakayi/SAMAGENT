"""Gann/pitchfork geometry and replay anti-lookahead (ported from v1
tests/test_geometry_replay_secrets.py; the secret-store half is covered by
tests/test_core_secrets.py)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from sam.trading.engine.geometry import (
    GANN_ANGLES,
    PITCHFORK_VARIANTS,
    bar_minutes,
    calculate_pitchfork_anchors,
    gann_box,
    gann_fan,
    pitchfork_price_at,
    price_per_bar,
    square_of_nine_levels,
)
from sam.trading.engine.replay import BarReplayResearch, LookaheadError, ReplaySession, ReplayState
from sam.trading.engine.types import CapabilityState, Candle

START = datetime(2026, 7, 1, 8, 0, tzinfo=UTC)


def candle(index: int, open_: float, high: float, low: float, close: float) -> Candle:
    return Candle(time=START + timedelta(minutes=15 * index), open=open_, high=high, low=low, close=close, volume=100.0)


def zigzag(count: int = 120) -> list[Candle]:
    """Alternating swings so pivot detection has something real to find."""
    series, price = [], 100.0
    for index in range(count):
        move = 3.0 if (index // 7) % 2 == 0 else -3.0
        series.append(candle(index, price, price + abs(move) + 1, price - abs(move) - 1, price + move))
        price += move
    return series


# --- Gann ---------------------------------------------------------------------


def test_the_price_per_bar_unit_comes_from_the_instrument():
    series = zigzag()
    unit = price_per_bar(series)
    assert unit > 0
    # A 1x1 angle means one of these units per bar, so it must track real range.
    assert unit == pytest.approx(sum(c.range for c in series[-14:]) / 14, rel=0.6)


def test_bar_spacing_is_measured_not_assumed():
    assert bar_minutes(zigzag()) == pytest.approx(15.0)
    assert bar_minutes([candle(0, 1, 1, 1, 1)]) == 0.0


def test_a_gann_fan_returns_every_classic_angle_from_one_pivot():
    fan = gann_fan(zigzag())
    assert fan["available"] is True
    assert len(fan["rays"]) == len(GANN_ANGLES)
    labels = {ray["label"] for ray in fan["rays"]}
    assert "1x1" in labels
    assert sum(1 for ray in fan["rays"] if ray["primary"]) == 1
    # Every ray starts at the same pivot.
    starts = {(ray["start"]["price"], ray["start"]["minutes"]) for ray in fan["rays"]}
    assert len(starts) == 1


def test_gann_output_declares_that_it_is_not_the_native_object():
    fan = gann_fan(zigzag())
    assert fan["representation"] == "trendline_fan"
    assert "native" in fan["limitation"].lower()
    assert fan["subjective"] is True


def test_steeper_gann_angles_climb_faster_than_shallower_ones():
    rays = {ray["label"]: ray for ray in gann_fan(zigzag())["rays"]}
    assert abs(rays["4x1"]["slope_price_per_minute"]) > abs(rays["1x1"]["slope_price_per_minute"])
    assert abs(rays["1x1"]["slope_price_per_minute"]) > abs(rays["1x4"]["slope_price_per_minute"])


def test_a_short_series_will_not_produce_a_fan():
    assert gann_fan(zigzag(10))["available"] is False


def test_a_gann_box_divides_the_range_into_eighths():
    box = gann_box(zigzag(), bars=60)
    assert box["available"] is True
    assert len(box["levels"]) == 9
    assert box["levels"][0]["price"] == pytest.approx(box["low"])
    assert box["levels"][-1]["price"] == pytest.approx(box["high"])
    midpoint = next(level for level in box["levels"] if level["label"] == "4/8")
    assert midpoint["price"] == pytest.approx((box["high"] + box["low"]) / 2)


def test_square_of_nine_levels_straddle_the_anchor_price():
    result = square_of_nine_levels(100.0, rings=1)
    assert result["available"] is True
    assert result["root"] == pytest.approx(10.0)
    assert any(level["above"] for level in result["levels"])
    assert any(not level["above"] for level in result["levels"])
    prices = [level["price"] for level in result["levels"]]
    assert prices == sorted(prices)


def test_square_of_nine_rejects_impossible_input():
    assert square_of_nine_levels(0.0)["available"] is False
    assert square_of_nine_levels(100.0, step_degrees=0)["available"] is False


# --- Pitchfork ----------------------------------------------------------------


@pytest.mark.parametrize("variant", PITCHFORK_VARIANTS)
def test_every_pitchfork_variant_produces_three_anchors_and_three_lines(variant):
    result = calculate_pitchfork_anchors(zigzag(), variant=variant)
    assert result["available"] is True, result.get("reason")
    assert set(result["anchors"]) == {"P0", "P1", "P2"}
    for line in ("median_line", "upper_parallel", "lower_parallel"):
        assert line in result
    assert result["variant"] == variant


def test_the_parallels_share_the_median_slope():
    result = calculate_pitchfork_anchors(zigzag())
    slope = result["median_line"]["slope_price_per_minute"]
    assert result["upper_parallel"]["slope_price_per_minute"] == pytest.approx(slope)
    assert result["lower_parallel"]["slope_price_per_minute"] == pytest.approx(slope)


def test_the_upper_parallel_stays_above_the_lower_one():
    result = calculate_pitchfork_anchors(zigzag())
    minutes = result["anchors"]["P2"]["minutes"]
    upper = pitchfork_price_at(result, "upper", minutes)
    lower = pitchfork_price_at(result, "lower", minutes)
    median = pitchfork_price_at(result, "median", minutes)
    assert upper > lower
    assert lower <= median <= upper


def test_the_median_runs_through_the_midpoint_of_p1_and_p2():
    result = calculate_pitchfork_anchors(zigzag(), variant="andrews")
    midpoint = result["midpoint"]
    assert pitchfork_price_at(result, "median", midpoint["minutes"]) == pytest.approx(midpoint["price"], abs=1e-6)


def test_schiff_shifts_the_origin_away_from_p0():
    andrews = calculate_pitchfork_anchors(zigzag(), variant="andrews")
    schiff = calculate_pitchfork_anchors(zigzag(), variant="schiff")
    assert schiff["median_line"]["start"]["price"] != andrews["median_line"]["start"]["price"]


def test_unordered_or_unknown_anchors_are_refused():
    series = zigzag()
    assert calculate_pitchfork_anchors(series, variant="nonsense")["available"] is False
    out_of_order = calculate_pitchfork_anchors(series, p0_index=50, p1_index=20, p2_index=80)
    assert out_of_order["available"] is False
    assert "P0 < P1 < P2" in out_of_order["reason"]


def test_pitchfork_declares_its_drawing_limitation():
    result = calculate_pitchfork_anchors(zigzag())
    assert result["representation"] == "three_trendlines"
    assert "native" in result["limitation"].lower()


# --- Replay -------------------------------------------------------------------


def test_a_replay_session_only_reveals_candles_up_to_the_cursor():
    series = zigzag(200)
    session = ReplaySession(series, start_index=50)
    assert len(session.visible()) == 51
    session.advance(10)
    assert len(session.visible()) == 61
    assert session.visible()[-1] is series[60]


def test_reading_past_the_cursor_is_refused():
    session = ReplaySession(zigzag(200), start_index=50)
    session.peek(50)
    with pytest.raises(LookaheadError, match="ahead of the replay cursor"):
        session.peek(51)


def test_the_replay_state_machine_moves_through_its_states():
    session = ReplaySession(zigzag(200), start_index=50)
    assert session.state is ReplayState.LOADED
    session.advance(1)
    assert session.state is ReplayState.PLAYING
    assert session.pause()["state"] == ReplayState.PAUSED.value
    assert session.resume()["state"] == ReplayState.PLAYING.value
    assert session.stop()["state"] == ReplayState.IDLE.value


def test_the_cursor_stops_at_the_end_and_reports_finished():
    session = ReplaySession(zigzag(60), start_index=50)
    session.advance(500)
    assert session.finished is True
    assert session.state is ReplayState.FINISHED


def test_advancing_by_less_than_one_bar_is_refused():
    session = ReplaySession(zigzag(200), start_index=50)
    with pytest.raises(ValueError):
        session.advance(0)


def test_an_outcome_cannot_be_scored_before_the_cursor_passes_it():
    session = ReplaySession(zigzag(200), start_index=50)
    session.record_decision(theory="t", setup_state="ENTRY_READY", direction="BULLISH",
                            entry=100.0, stop=95.0, targets=[110.0])
    session.resolve_decisions()
    # The cursor has not moved, so the future is genuinely unknown.
    assert session.decisions[0].outcome is None
    session.advance(40)
    session.resolve_decisions()
    assert session.decisions[0].outcome in {None, "STOPPED", "TARGET"}
    if session.decisions[0].outcome:
        assert session.decisions[0].outcome_cursor > session.decisions[0].cursor


def test_a_resolved_decision_scores_from_bars_after_the_decision_only():
    series = [candle(index, 100, 101, 99, 100) for index in range(60)]
    # A clean drop after bar 30 so the stop is the only reachable level.
    series += [candle(60 + index, 100 - index * 2, 101 - index * 2, 98 - index * 2, 99 - index * 2) for index in range(20)]
    session = ReplaySession(series, start_index=59)
    session.record_decision(theory="t", setup_state="ENTRY_READY", direction="BULLISH",
                            entry=100.0, stop=96.0, targets=[130.0])
    session.advance(10)
    session.resolve_decisions()
    decision = session.decisions[0]
    assert decision.outcome == "STOPPED"
    assert decision.r_multiple == pytest.approx(-1.0)
    assert decision.outcome_cursor > decision.cursor


def test_a_session_needs_a_valid_start_index():
    with pytest.raises(ValueError):
        ReplaySession(zigzag(50), start_index=500)
    with pytest.raises(ValueError):
        ReplaySession([], start_index=0)


def test_native_tradingview_replay_is_reported_partial_not_verified():
    capability = BarReplayResearch(fetch=None).capability()
    assert capability["data_replay"]["state"] == CapabilityState.AVAILABLE.value
    native = capability["tradingview_native_replay"]
    assert native["state"] == CapabilityState.PARTIALLY_AVAILABLE.value
    assert "cannot be independently confirmed" in native["reason"]


def test_controlling_a_missing_session_is_refused():
    research = BarReplayResearch(fetch=None)
    result = research.control("advance")
    assert result["ok"] is False and result["error_code"] == "NO_REPLAY_SESSION"


def test_a_replay_scan_scores_decisions_only_after_the_cursor_passes():
    series = zigzag(400)
    research = BarReplayResearch(fetch=lambda symbol, timeframe, count: series)
    started = research.start("XAUUSD", "M15", start_offset=100)
    assert started["ok"] is True and started["cursor"] == 100
    report = research.run_scan("bullish_rejection", bars=150)
    assert report["ok"] is True
    for decision in report["recorded"]:
        if decision["outcome_cursor"] is not None:
            assert decision["outcome_cursor"] > decision["cursor"]


def test_a_replay_start_without_enough_history_is_refused():
    research = BarReplayResearch(fetch=lambda symbol, timeframe, count: zigzag(50))
    assert research.start(start_offset=100)["error_code"] == "INSUFFICIENT_HISTORY"
