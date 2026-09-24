"""(Ported from v1 tests/test_patterns.py.) Deterministic tests for the order-block, Fibonacci, harmonic, Elliott, and ORB engines."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from sam.trading.engine.patterns import (
    HARMONIC_RULES,
    detect_harmonics,
    fib_extension,
    fib_confluence,
    fib_retracement,
    opening_range,
    optimal_trade_entry,
    order_blocks,
    prz_from_points,
    rank_wave_counts,
    validate_harmonic,
    validate_impulse,
    wave_fib_relationships,
)
from sam.trading.engine.types import Candle, Direction

START = datetime(2026, 8, 24, 0, 0, tzinfo=UTC)


def candle(index: int, open_: float, high: float, low: float, close: float, minutes: int = 15, volume: float = 100.0) -> Candle:
    return Candle(time=START + timedelta(minutes=minutes * index), open=open_, high=high, low=low, close=close, volume=volume)


def flat_series(count: int, price: float = 100.0) -> list[Candle]:
    """Quiet candles that establish a small average range."""
    return [candle(i, price, price + 0.5, price - 0.5, price) for i in range(count)]


# --- Order blocks -------------------------------------------------------------


def test_a_quiet_market_produces_no_order_blocks():
    assert order_blocks(flat_series(40), "M15") == []


def test_a_down_candle_followed_by_a_displacing_break_is_a_bullish_order_block():
    series = flat_series(20)
    # The order block: a down candle after the quiet stretch.
    series.append(candle(20, 100.0, 100.2, 99.0, 99.2))
    # Displacement up through the prior highs.
    series.append(candle(21, 99.2, 103.0, 99.1, 102.8))
    series.append(candle(22, 102.8, 104.0, 102.5, 103.9))
    series += [candle(23 + i, 104.0, 104.5, 103.5, 104.0) for i in range(6)]

    blocks = order_blocks(series, "M15")
    bullish = [block for block in blocks if block["kind"] == "BULLISH_OB"]
    assert bullish, f"expected a bullish order block, got {blocks}"
    block = bullish[0]
    assert block["bottom"] == pytest.approx(99.0)
    assert block["top"] == pytest.approx(100.2)
    assert block["displacement_atr"] >= 1.5
    assert block["structure_break"] is not None


def test_an_opposite_candle_without_displacement_is_not_an_order_block():
    series = flat_series(20)
    series.append(candle(20, 100.0, 100.2, 99.0, 99.2))
    # Drift back up, but far too slowly to count as displacement.
    series += [candle(21 + i, 99.2 + i * 0.05, 99.4 + i * 0.05, 99.1 + i * 0.05, 99.3 + i * 0.05) for i in range(8)]
    assert order_blocks(series, "M15") == []


def test_a_block_price_closed_through_is_reported_as_a_breaker():
    series = flat_series(20)
    series.append(candle(20, 100.0, 100.2, 99.0, 99.2))
    series.append(candle(21, 99.2, 103.0, 99.1, 102.8))
    series.append(candle(22, 102.8, 104.0, 102.5, 103.9))
    # Price returns and closes decisively below the block's low.
    series += [candle(23 + i, 103.0 - i, 103.2 - i, 102.0 - i, 102.5 - i) for i in range(8)]
    blocks = order_blocks(series, "M15")
    bullish = [block for block in blocks if block["kind"] == "BULLISH_OB"]
    assert bullish and bullish[0]["state"] == "BREAKER"
    assert bullish[0]["fresh"] is False


def test_order_block_detection_stays_selective_on_a_long_series():
    # A trending random-ish series must not label a large share of candles.
    series = []
    price = 100.0
    for index in range(200):
        step = 0.6 if index % 3 else -0.4
        series.append(candle(index, price, price + abs(step) + 0.3, price - abs(step) - 0.3, price + step))
        price += step
    blocks = order_blocks(series, "M15", limit=100)
    assert len(blocks) < len(series) * 0.15, f"{len(blocks)} blocks from {len(series)} candles is not selective"


# --- Fibonacci ----------------------------------------------------------------


def test_bullish_retracement_levels_sit_below_the_swing_high():
    result = fib_retracement(110.0, 100.0, Direction.BULLISH)
    assert result["span"] == pytest.approx(10.0)
    assert result["levels"]["0.5"] == pytest.approx(105.0)
    assert result["levels"]["0.618"] == pytest.approx(103.82)
    assert result["equilibrium"] == pytest.approx(105.0)


def test_bearish_retracement_levels_sit_above_the_swing_low():
    result = fib_retracement(110.0, 100.0, Direction.BEARISH)
    assert result["levels"]["0.5"] == pytest.approx(105.0)
    assert result["levels"]["0.786"] == pytest.approx(107.86)


def test_extensions_project_beyond_the_leg():
    up = fib_extension(110.0, 100.0, Direction.BULLISH)
    assert up["levels"]["1.618"] == pytest.approx(116.18)
    down = fib_extension(110.0, 100.0, Direction.BEARISH)
    assert down["levels"]["1.618"] == pytest.approx(93.82)


def test_the_ote_band_spans_the_golden_pocket():
    band = optimal_trade_entry(110.0, 100.0, Direction.BULLISH)
    assert band["high"] == pytest.approx(103.82)
    assert band["low"] == pytest.approx(102.10)
    assert band["low"] < band["midpoint"] < band["high"]


def test_an_inverted_leg_is_rejected_rather_than_silently_flipped():
    with pytest.raises(ValueError):
        fib_retracement(100.0, 110.0, Direction.BULLISH)


def test_confluence_reports_where_independent_legs_agree():
    first = fib_retracement(110.0, 100.0, Direction.BULLISH)
    second = fib_retracement(112.0, 102.0, Direction.BULLISH)
    clusters = fib_confluence([first, second], tolerance=0.6)
    assert clusters
    assert clusters[0]["count"] >= 2
    assert clusters[0]["low"] <= clusters[0]["price"] <= clusters[0]["high"]


# --- Harmonics ----------------------------------------------------------------


def test_a_textbook_gartley_validates():
    # X=100 A=110 B=103.82 (0.618 XA) C=107.6 D=101.86 (0.786 XA)
    matches = validate_harmonic(100.0, 110.0, 103.82, 107.6, 101.86)
    assert any(match["pattern"] == "Gartley" for match in matches)
    gartley = next(match for match in matches if match["pattern"] == "Gartley")
    assert gartley["ratios"]["ab_xa"] == pytest.approx(0.618, abs=0.01)
    assert gartley["invalidation"] == pytest.approx(100.0)
    assert gartley["prz"]["low"] < gartley["prz"]["high"]


def test_ratios_outside_tolerance_produce_no_pattern():
    # BC retraces 90% of AB and CD is only a third of BC: outside every envelope.
    assert validate_harmonic(100.0, 110.0, 100.5, 109.1, 106.2) == []


def test_ab_equals_cd_requires_the_two_legs_to_actually_match():
    # AB = -2 and CD = -2: the defining equality holds.
    assert any(m["pattern"] == "AB=CD" for m in validate_harmonic(100.0, 110.0, 108.0, 109.0, 107.0))
    # Same shape but CD is three times AB, so it is not an AB=CD.
    assert not any(m["pattern"] == "AB=CD" for m in validate_harmonic(100.0, 110.0, 108.0, 109.0, 103.0))


def test_non_alternating_legs_are_not_an_xabcd_shape():
    assert validate_harmonic(100.0, 110.0, 120.0, 130.0, 140.0) == []


def test_every_harmonic_rule_set_is_well_formed():
    for name, rules in HARMONIC_RULES.items():
        for key, bounds in rules.items():
            if bounds is None:
                continue
            assert bounds[0] < bounds[1], f"{name}.{key} bounds are inverted"


def test_the_potential_reversal_zone_is_an_interval():
    zone = prz_from_points(100.0, 110.0, 103.82, 107.6)
    assert zone["low"] < zone["high"]


def test_harmonic_scan_returns_nothing_on_a_flat_series():
    assert detect_harmonics(flat_series(80), "M15") == []


# --- Elliott ------------------------------------------------------------------


def test_a_valid_impulse_passes_all_three_rules():
    result = validate_impulse([100.0, 110.0, 105.0, 130.0, 122.0, 140.0])
    assert result["valid"] is True
    assert result["violations"] == []
    assert result["direction"] == Direction.BULLISH.value
    assert result["extended_wave"] == "3"


def test_wave_two_may_not_retrace_past_the_start():
    result = validate_impulse([100.0, 110.0, 99.0, 130.0, 122.0, 140.0])
    assert result["valid"] is False
    assert any("Wave 2" in violation for violation in result["violations"])


def test_wave_four_may_not_overlap_wave_one():
    result = validate_impulse([100.0, 110.0, 105.0, 130.0, 108.0, 140.0])
    assert result["valid"] is False
    assert any("Wave 4" in violation for violation in result["violations"])


def test_wave_three_may_not_be_the_shortest():
    result = validate_impulse([100.0, 120.0, 110.0, 125.0, 122.0, 160.0])
    assert result["valid"] is False
    assert any("shortest" in violation for violation in result["violations"])


def test_a_bearish_impulse_is_validated_with_mirrored_rules():
    result = validate_impulse([140.0, 130.0, 135.0, 110.0, 118.0, 100.0])
    assert result["valid"] is True
    assert result["direction"] == Direction.BEARISH.value


def test_an_incomplete_point_set_is_rejected():
    assert validate_impulse([100.0, 110.0, 105.0])["valid"] is False


def test_wave_relationships_are_measured_against_wave_one():
    relationships = wave_fib_relationships([100.0, 110.0, 105.0, 130.0, 122.0, 140.0])
    assert relationships["wave2_retracement"] == pytest.approx(0.5)
    assert relationships["wave3_of_wave1"] == pytest.approx(2.5)


def test_wave_counts_are_ranked_and_never_forced():
    counts = rank_wave_counts(flat_series(80), "M15")
    # A flat series offers no impulse; whatever is returned must not claim validity.
    assert all(not item["validation"]["valid"] for item in counts)


# --- Opening range breakout ---------------------------------------------------


def orb_series(range_high: float, range_low: float, after: list[tuple[float, float, float, float]]) -> list[Candle]:
    """15 one-minute candles inside the London open, then the supplied follow-up."""
    series: list[Candle] = []
    open_at = datetime(2026, 8, 24, 7, 0, tzinfo=UTC)
    for index in range(15):
        series.append(Candle(
            time=open_at + timedelta(minutes=index),
            open=(range_high + range_low) / 2, high=range_high, low=range_low,
            close=(range_high + range_low) / 2, volume=50.0,
        ))
    for index, (open_, high, low, close) in enumerate(after):
        series.append(Candle(
            time=open_at + timedelta(minutes=15 + index),
            open=open_, high=high, low=low, close=close, volume=50.0,
        ))
    return series


def test_the_opening_range_is_built_only_from_candles_inside_the_window():
    series = orb_series(105.0, 100.0, [(102.0, 130.0, 101.0, 129.0)])
    result = opening_range(series, minutes=15, session="london")
    assert result["available"] is True
    assert result["candles_in_range"] == 15
    # The 130 high after the window must not widen the range.
    assert result["high"] == pytest.approx(105.0)
    assert result["low"] == pytest.approx(100.0)
    assert result["size"] == pytest.approx(5.0)


def test_a_clean_break_yields_a_trigger_invalidation_and_targets():
    series = orb_series(105.0, 100.0, [(104.0, 108.0, 103.5, 107.0), (107.0, 109.0, 106.0, 108.5)])
    result = opening_range(series, minutes=15, session="london")
    assert result["state"] == "BROKEN"
    assert result["breakout"]["direction"] == Direction.BULLISH.value
    assert result["entry_trigger"] == pytest.approx(105.0)
    assert result["invalidation"] == pytest.approx(100.0)
    assert result["targets"]["tp1"] == pytest.approx(110.0)


def test_a_break_that_reverses_through_the_range_is_a_false_breakout():
    series = orb_series(105.0, 100.0, [(104.0, 108.0, 103.5, 107.0), (107.0, 107.5, 98.0, 98.5)])
    result = opening_range(series, minutes=15, session="london")
    assert result["false_breakout"] is True
    assert result["state"] == "FALSE_BREAKOUT"
    # A false breakout must not hand back an entry trigger.
    assert "entry_trigger" not in result


def test_price_held_inside_the_range_reports_no_breakout():
    series = orb_series(105.0, 100.0, [(102.0, 104.0, 101.0, 103.0)])
    result = opening_range(series, minutes=15, session="london")
    assert result["state"] == "NO_BREAKOUT"
    assert result["breakout"] is None


def test_a_date_with_no_session_candles_is_reported_unavailable():
    result = opening_range(flat_series(30), minutes=15, session="new_york")
    assert result["available"] is False
    assert "reason" in result


def test_an_unknown_session_or_bad_window_is_rejected():
    with pytest.raises(ValueError):
        opening_range(flat_series(30), minutes=15, session="atlantis")
    with pytest.raises(ValueError):
        opening_range(flat_series(30), minutes=0, session="london")
