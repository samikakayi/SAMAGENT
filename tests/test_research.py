"""Entry-trigger and backtest tests, with anti-lookahead as the central concern."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from sam_backend.trading.research import (
    TRIGGER_REGISTRY,
    Trade,
    _resolve_trade,
    backtest_trigger,
    compare_triggers,
    summarize_trades,
)
from sam_backend.trading.types import Candle, Direction

START = datetime(2026, 7, 1, tzinfo=UTC)


def candle(index: int, open_: float, high: float, low: float, close: float) -> Candle:
    return Candle(time=START + timedelta(minutes=15 * index), open=open_, high=high, low=low, close=close, volume=100.0)


def drifting_series(count: int, *, start: float = 100.0, step: float = 0.4) -> list[Candle]:
    series, price = [], start
    for index in range(count):
        move = step if index % 3 else -step
        series.append(candle(index, price, price + abs(move) + 0.3, price - abs(move) - 0.3, price + move))
        price += move
    return series


# --- Trigger registry ---------------------------------------------------------


def test_every_trigger_is_fully_described():
    assert len(TRIGGER_REGISTRY) >= 14
    for trigger_id, trigger in TRIGGER_REGISTRY.items():
        payload = trigger.as_dict()
        assert payload["id"] == trigger_id
        assert payload["direction"] in {Direction.BULLISH.value, Direction.BEARISH.value}
        for field in ("requirements", "confirmation", "invalidation", "preferred_timeframes", "compatible_theories"):
            assert payload[field], f"{trigger_id} is missing {field}"


def test_a_trigger_never_raises_on_short_input():
    for trigger in TRIGGER_REGISTRY.values():
        assert trigger.evaluate([]) is False
        assert trigger.evaluate(drifting_series(3)) in {True, False}


def test_bullish_engulfing_fires_only_on_a_real_engulfing():
    engulfing = [candle(0, 100, 100.5, 99.0, 99.2), candle(1, 99.1, 101.5, 99.0, 101.2)]
    assert TRIGGER_REGISTRY["bullish_engulfing"].evaluate(engulfing) is True
    # Same shape but the close fails to clear the prior open.
    weak = [candle(0, 100, 100.5, 99.0, 99.2), candle(1, 99.1, 99.8, 99.0, 99.6)]
    assert TRIGGER_REGISTRY["bullish_engulfing"].evaluate(weak) is False


def test_a_sweep_requires_reclaiming_the_level():
    base = [candle(index, 100, 100.5, 99.5, 100) for index in range(11)]
    reclaimed = base + [candle(11, 100, 100.2, 98.0, 99.9)]
    assert TRIGGER_REGISTRY["bullish_sweep"].evaluate(reclaimed) is True
    # Took the low but closed below it: no reclaim, so no trigger.
    rejected = base + [candle(11, 100, 100.2, 98.0, 98.2)]
    assert TRIGGER_REGISTRY["bullish_sweep"].evaluate(rejected) is False


# --- Trade resolution ---------------------------------------------------------


def test_a_target_hit_resolves_as_a_win_at_exactly_the_reward():
    trade = Trade(index=0, time="t", direction=Direction.BULLISH.value, entry=100.0, stop=99.0, target=102.0)
    future = [candle(1, 100, 100.5, 99.5, 100.2), candle(2, 100.2, 102.5, 100.0, 102.3)]
    resolved = _resolve_trade(trade, future, max_bars=10)
    assert resolved.outcome == "WIN"
    assert resolved.r_multiple == pytest.approx(2.0)
    assert resolved.bars_held == 2


def test_a_stop_hit_resolves_as_exactly_minus_one_r():
    trade = Trade(index=0, time="t", direction=Direction.BULLISH.value, entry=100.0, stop=99.0, target=102.0)
    resolved = _resolve_trade(trade, [candle(1, 100, 100.2, 98.5, 98.8)], max_bars=10)
    assert resolved.outcome == "LOSS"
    assert resolved.r_multiple == pytest.approx(-1.0)


def test_a_candle_spanning_stop_and_target_is_scored_as_a_loss():
    """The pessimistic fill: assuming the win here is how backtests flatter themselves."""
    trade = Trade(index=0, time="t", direction=Direction.BULLISH.value, entry=100.0, stop=99.0, target=102.0)
    both = [candle(1, 100, 103.0, 98.0, 101.0)]
    assert _resolve_trade(trade, both, max_bars=10).outcome == "LOSS"


def test_a_short_trade_resolves_with_mirrored_logic():
    trade = Trade(index=0, time="t", direction=Direction.BEARISH.value, entry=100.0, stop=101.0, target=98.0)
    resolved = _resolve_trade(trade, [candle(1, 100, 100.2, 97.5, 97.8)], max_bars=10)
    assert resolved.outcome == "WIN"
    assert resolved.r_multiple == pytest.approx(2.0)


def test_an_unresolved_trade_expires_rather_than_staying_open():
    trade = Trade(index=0, time="t", direction=Direction.BULLISH.value, entry=100.0, stop=95.0, target=110.0)
    flat = [candle(index, 100, 100.4, 99.6, 100.1) for index in range(1, 6)]
    resolved = _resolve_trade(trade, flat, max_bars=3)
    assert resolved.outcome == "EXPIRED"
    assert resolved.bars_held == 3


# --- Anti-lookahead -----------------------------------------------------------


def test_a_decision_never_sees_a_candle_it_could_not_have_seen():
    """Record the longest history handed to the detector at each call."""
    series = drifting_series(400)
    seen: list[int] = []

    original = TRIGGER_REGISTRY["bullish_engulfing"].detector

    def spy(candles):
        seen.append(len(candles))
        return original(candles)

    TRIGGER_REGISTRY["bullish_engulfing"].detector = spy
    try:
        result = backtest_trigger(series, "bullish_engulfing", max_bars=20, warmup=80)
    finally:
        TRIGGER_REGISTRY["bullish_engulfing"].detector = original

    assert result["available"] is True
    # The scan must stop far enough from the end to resolve every trade.
    assert max(seen) <= len(series) - 20 - 1
    assert min(seen) >= 80


def test_entries_are_taken_at_the_next_bar_open_not_the_signal_close():
    series = drifting_series(400)
    result = backtest_trigger(series, "bullish_engulfing", max_bars=20, warmup=80)
    for trade in result["trades"]:
        index = next(i for i, c in enumerate(series) if c.time.isoformat() == trade["time"])
        assert trade["entry"] == pytest.approx(series[index].open)


def test_positions_never_overlap():
    series = drifting_series(600)
    result = backtest_trigger(series, "bullish_rejection", max_bars=20, warmup=80)
    trades = result["trades"]
    for previous, following in zip(trades, trades[1:]):
        previous_index = next(i for i, c in enumerate(series) if c.time.isoformat() == previous["time"])
        following_index = next(i for i, c in enumerate(series) if c.time.isoformat() == following["time"])
        assert following_index > previous_index + previous["bars_held"] - 1


def test_too_little_history_is_refused_rather_than_scored():
    result = backtest_trigger(drifting_series(40), "bullish_engulfing")
    assert result["available"] is False
    assert "reason" in result


def test_an_unknown_trigger_is_refused():
    assert backtest_trigger(drifting_series(400), "not_a_trigger")["available"] is False


# --- Statistics ---------------------------------------------------------------


def test_statistics_are_arithmetically_consistent():
    trades = [
        Trade(index=i, time="t", direction=Direction.BULLISH.value, entry=100, stop=99, target=102,
              outcome=outcome, r_multiple=r)
        for i, (outcome, r) in enumerate([("WIN", 2.0), ("LOSS", -1.0), ("WIN", 2.0), ("LOSS", -1.0), ("LOSS", -1.0)])
    ]
    stats = summarize_trades(trades)
    assert stats["total_setups"] == 5
    assert stats["wins"] == 2 and stats["losses"] == 3
    assert stats["win_rate"] == pytest.approx(0.4)
    assert stats["total_r"] == pytest.approx(1.0)
    assert stats["average_r"] == pytest.approx(0.2)
    assert stats["profit_factor"] == pytest.approx(4.0 / 3.0, abs=1e-4)
    # WIN, LOSS, WIN, LOSS, LOSS -> the longest loss run is the final pair.
    assert stats["max_consecutive_losses"] == 2
    assert stats["max_consecutive_wins"] == 1


def test_drawdown_is_measured_from_the_equity_peak():
    trades = [
        Trade(index=i, time="t", direction=Direction.BULLISH.value, entry=100, stop=99, target=102,
              outcome="WIN" if r > 0 else "LOSS", r_multiple=r)
        for i, r in enumerate([2.0, 2.0, -1.0, -1.0, -1.0, 2.0])
    ]
    stats = summarize_trades(trades)
    # Peak of +4 then down to +1 is a 3R drawdown.
    assert stats["max_drawdown_r"] == pytest.approx(3.0)


def test_an_empty_trade_list_produces_zeroed_statistics():
    stats = summarize_trades([])
    assert stats["total_setups"] == 0
    assert stats["win_rate"] == 0.0
    assert stats["average_r"] == 0.0


def test_comparing_triggers_ranks_them_on_identical_data():
    series = drifting_series(600)
    outcome = compare_triggers(series, ["bullish_engulfing", "bearish_engulfing"], max_bars=20, warmup=80)
    assert outcome["compared"] == 2
    assert {item["trigger"] for item in outcome["ranking"]} <= {"bullish_engulfing", "bearish_engulfing"}
    expectancies = [item["expectancy"] for item in outcome["ranking"]]
    assert expectancies == sorted(expectancies, reverse=True)
