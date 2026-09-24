"""(Ported from v1 ``sam_backend/trading/research.py``.) Historical research: entry triggers and strategy backtesting.

The single most important property here is that no decision may see a candle it
could not have seen live. Every scan walks the series forward one closed bar at a
time and is handed only `candles[:index]`; the outcome is then resolved on the
bars after that index. That is what makes the statistics mean anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from .analysis import market_structure
from .indicators import atr, ema, find_swings, latest, rsi
from .patterns import order_blocks
from .types import Candle, Direction

# --- Entry trigger registry ---------------------------------------------------


@dataclass(slots=True)
class TriggerDefinition:
    """One reusable entry condition, evaluated on closed candles only."""

    id: str
    name: str
    direction: Direction
    requirements: list[str]
    confirmation: str
    invalidation: str
    preferred_timeframes: list[str]
    compatible_theories: list[str]
    detector: Callable[[list[Candle]], bool] = field(repr=False, default=lambda candles: False)

    def evaluate(self, candles: list[Candle]) -> bool:
        try:
            return bool(self.detector(candles))
        except Exception:
            return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "direction": self.direction.value,
            "requirements": self.requirements,
            "confirmation": self.confirmation,
            "invalidation": self.invalidation,
            "preferred_timeframes": self.preferred_timeframes,
            "compatible_theories": self.compatible_theories,
        }


def _engulfing(candles: list[Candle], bullish: bool) -> bool:
    if len(candles) < 2:
        return False
    previous, current = candles[-2], candles[-1]
    if bullish:
        return (not previous.bullish and current.bullish
                and current.close > previous.open and current.open <= previous.close)
    return (previous.bullish and not current.bullish
            and current.close < previous.open and current.open >= previous.close)


def _swept_and_reclaimed(candles: list[Candle], bullish: bool) -> bool:
    """A liquidity sweep: price takes out a prior extreme then closes back inside."""
    if len(candles) < 12:
        return False
    window = candles[-12:-1]
    current = candles[-1]
    if bullish:
        prior_low = min(candle.low for candle in window)
        return current.low < prior_low and current.close > prior_low
    prior_high = max(candle.high for candle in window)
    return current.high > prior_high and current.close < prior_high


def _structure_shift(candles: list[Candle], bullish: bool) -> bool:
    if len(candles) < 30:
        return False
    structure = market_structure(candles, "X")
    event = structure.get("choch") or structure.get("bos")
    if not event:
        return False
    wanted = Direction.BULLISH.value if bullish else Direction.BEARISH.value
    return event["direction"] == wanted


def _rejection(candles: list[Candle], bullish: bool) -> bool:
    if not candles:
        return False
    candle = candles[-1]
    if candle.range <= 0:
        return False
    upper = candle.high - max(candle.open, candle.close)
    lower = min(candle.open, candle.close) - candle.low
    if bullish:
        return lower > candle.body * 1.5 and lower / candle.range > 0.5
    return upper > candle.body * 1.5 and upper / candle.range > 0.5


def _ma_cross(candles: list[Candle], golden: bool) -> bool:
    if len(candles) < 60:
        return False
    closes = [candle.close for candle in candles]
    fast, slow = ema(closes, 20), ema(closes, 50)
    if len(fast) < 2 or fast[-1] is None or slow[-1] is None or fast[-2] is None or slow[-2] is None:
        return False
    if golden:
        return fast[-2] <= slow[-2] and fast[-1] > slow[-1]
    return fast[-2] >= slow[-2] and fast[-1] < slow[-1]


def _order_block_tap(candles: list[Candle], bullish: bool) -> bool:
    """Price is entering a block that was still unmitigated before this bar.

    Freshness has to be judged on history *excluding* the current candle: a block
    is marked mitigated the moment price touches it, so evaluating both on the
    same series would mean a tap could never be observed.
    """
    if len(candles) < 3:
        return False
    blocks = order_blocks(candles[:-1], "X", limit=8)
    wanted = "BULLISH_OB" if bullish else "BEARISH_OB"
    current = candles[-1]
    for block in blocks:
        if block["kind"] != wanted or block["state"] != "FRESH":
            continue
        if bullish and current.low <= block["top"] and current.high >= block["bottom"]:
            return True
        if not bullish and current.high >= block["bottom"] and current.low <= block["top"]:
            return True
    return False


def _divergence(candles: list[Candle], bullish: bool) -> bool:
    """Price makes a new extreme while RSI does not."""
    if len(candles) < 40:
        return False
    closes = [candle.close for candle in candles]
    values = rsi(closes, 14)
    swings = find_swings(candles, 3, 3)
    points = swings["lows"] if bullish else swings["highs"]
    if len(points) < 2:
        return False
    first, second = points[-2], points[-1]
    if values[first["index"]] is None or values[second["index"]] is None:
        return False
    if bullish:
        return second["price"] < first["price"] and values[second["index"]] > values[first["index"]]
    return second["price"] > first["price"] and values[second["index"]] < values[first["index"]]


def build_trigger_registry() -> dict[str, TriggerDefinition]:
    def define(
        trigger_id: str, name: str, direction: Direction, detector: Callable[[list[Candle]], bool],
        *, requirements: list[str], confirmation: str, invalidation: str,
        timeframes: list[str] | None = None, theories: list[str] | None = None,
    ) -> TriggerDefinition:
        return TriggerDefinition(
            id=trigger_id, name=name, direction=direction,
            requirements=requirements, confirmation=confirmation, invalidation=invalidation,
            preferred_timeframes=timeframes or ["M15", "M5", "M1"],
            compatible_theories=theories or ["default"],
            detector=detector,
        )

    definitions = [
        define("bullish_engulfing", "Bullish Engulfing Entry", Direction.BULLISH,
               lambda candles: _engulfing(candles, True),
               requirements=["Two closed candles"], confirmation="The engulfing candle closes above the prior open.",
               invalidation="A close below the engulfing candle's low.", theories=["price_action", "candlesticks"]),
        define("bearish_engulfing", "Bearish Engulfing Entry", Direction.BEARISH,
               lambda candles: _engulfing(candles, False),
               requirements=["Two closed candles"], confirmation="The engulfing candle closes below the prior open.",
               invalidation="A close above the engulfing candle's high.", theories=["price_action", "candlesticks"]),
        define("bullish_sweep", "Sell-Side Liquidity Sweep Entry", Direction.BULLISH,
               lambda candles: _swept_and_reclaimed(candles, True),
               requirements=["A prior swing low within twelve bars"],
               confirmation="Price trades below the prior low and closes back above it.",
               invalidation="A close below the sweep's extreme.", theories=["smc", "ict", "liquidity"]),
        define("bearish_sweep", "Buy-Side Liquidity Sweep Entry", Direction.BEARISH,
               lambda candles: _swept_and_reclaimed(candles, False),
               requirements=["A prior swing high within twelve bars"],
               confirmation="Price trades above the prior high and closes back below it.",
               invalidation="A close above the sweep's extreme.", theories=["smc", "ict", "liquidity"]),
        define("bullish_mss", "Bullish MSS/BOS Entry", Direction.BULLISH,
               lambda candles: _structure_shift(candles, True),
               requirements=["At least thirty closed candles"],
               confirmation="A close beyond the prior swing high.",
               invalidation="A close back below the broken swing.", theories=["market_structure", "smc", "ict"]),
        define("bearish_mss", "Bearish MSS/BOS Entry", Direction.BEARISH,
               lambda candles: _structure_shift(candles, False),
               requirements=["At least thirty closed candles"],
               confirmation="A close beyond the prior swing low.",
               invalidation="A close back above the broken swing.", theories=["market_structure", "smc", "ict"]),
        define("bullish_rejection", "Bullish Rejection Wick Entry", Direction.BULLISH,
               lambda candles: _rejection(candles, True),
               requirements=["One closed candle with range"],
               confirmation="A lower wick well over half the candle's range.",
               invalidation="A close below the wick's low.", theories=["price_action", "snr"]),
        define("bearish_rejection", "Bearish Rejection Wick Entry", Direction.BEARISH,
               lambda candles: _rejection(candles, False),
               requirements=["One closed candle with range"],
               confirmation="An upper wick well over half the candle's range.",
               invalidation="A close above the wick's high.", theories=["price_action", "snr"]),
        define("golden_cross", "Golden Cross Entry", Direction.BULLISH,
               lambda candles: _ma_cross(candles, True),
               requirements=["Sixty closed candles"], confirmation="EMA20 crosses above EMA50.",
               invalidation="EMA20 crosses back below EMA50.", timeframes=["H1", "H4", "D1"], theories=["indicators"]),
        define("death_cross", "Death Cross Entry", Direction.BEARISH,
               lambda candles: _ma_cross(candles, False),
               requirements=["Sixty closed candles"], confirmation="EMA20 crosses below EMA50.",
               invalidation="EMA20 crosses back above EMA50.", timeframes=["H1", "H4", "D1"], theories=["indicators"]),
        define("bullish_ob_tap", "Bullish Order Block Tap", Direction.BULLISH,
               lambda candles: _order_block_tap(candles, True),
               requirements=["A fresh bullish order block"],
               confirmation="Price trades into an unmitigated bullish block.",
               invalidation="A close below the block's low.", theories=["smc", "ict", "order_blocks"]),
        define("bearish_ob_tap", "Bearish Order Block Tap", Direction.BEARISH,
               lambda candles: _order_block_tap(candles, False),
               requirements=["A fresh bearish order block"],
               confirmation="Price trades into an unmitigated bearish block.",
               invalidation="A close above the block's high.", theories=["smc", "ict", "order_blocks"]),
        define("bullish_divergence", "Bullish RSI Divergence Entry", Direction.BULLISH,
               lambda candles: _divergence(candles, True),
               requirements=["Two swing lows and RSI history"],
               confirmation="A lower price low against a higher RSI low.",
               invalidation="A close below the second low.", theories=["indicators"]),
        define("bearish_divergence", "Bearish RSI Divergence Entry", Direction.BEARISH,
               lambda candles: _divergence(candles, False),
               requirements=["Two swing highs and RSI history"],
               confirmation="A higher price high against a lower RSI high.",
               invalidation="A close above the second high.", theories=["indicators"]),
    ]
    return {definition.id: definition for definition in definitions}


TRIGGER_REGISTRY = build_trigger_registry()


# --- Backtesting --------------------------------------------------------------


@dataclass(slots=True)
class Trade:
    index: int
    time: str
    direction: str
    entry: float
    stop: float
    target: float
    exit_price: float | None = None
    exit_index: int | None = None
    exit_time: str | None = None
    outcome: str = "OPEN"
    r_multiple: float = 0.0
    bars_held: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "time": self.time, "direction": self.direction,
            "entry": self.entry, "stop": self.stop, "target": self.target,
            "exit_price": self.exit_price, "exit_time": self.exit_time,
            "outcome": self.outcome, "r_multiple": round(self.r_multiple, 4), "bars_held": self.bars_held,
        }


def _resolve_trade(trade: Trade, future: list[Candle], max_bars: int) -> Trade:
    """Walk forward from the bar after entry until stop, target, or expiry.

    When a single candle spans both the stop and the target, the stop is assumed
    to have been hit first. Assuming the favourable fill is the classic way a
    backtest flatters itself.
    """
    long = trade.direction == Direction.BULLISH.value
    risk = abs(trade.entry - trade.stop)
    for offset, candle in enumerate(future[:max_bars], start=1):
        hit_stop = candle.low <= trade.stop if long else candle.high >= trade.stop
        hit_target = candle.high >= trade.target if long else candle.low <= trade.target
        if hit_stop:
            trade.outcome, trade.exit_price = "LOSS", trade.stop
        elif hit_target:
            trade.outcome, trade.exit_price = "WIN", trade.target
        else:
            continue
        trade.exit_index = trade.index + offset
        trade.exit_time = candle.time.isoformat()
        trade.bars_held = offset
        trade.r_multiple = (
            (trade.exit_price - trade.entry) / risk if long else (trade.entry - trade.exit_price) / risk
        ) if risk else 0.0
        return trade
    # Never resolved inside the horizon: mark expired at the last available close.
    if future:
        last = future[min(max_bars, len(future)) - 1]
        trade.exit_price = last.close
        trade.exit_index = trade.index + min(max_bars, len(future))
        trade.exit_time = last.time.isoformat()
        trade.bars_held = min(max_bars, len(future))
        trade.r_multiple = (
            (last.close - trade.entry) / risk if long else (trade.entry - last.close) / risk
        ) if risk else 0.0
        trade.outcome = "EXPIRED"
    return trade


def _session_of(candle: Candle) -> str:
    hour = candle.time.hour
    if 0 <= hour < 7:
        return "TOKYO"
    if 7 <= hour < 12:
        return "LONDON"
    if 12 <= hour < 17:
        return "NEW_YORK"
    return "AFTER_HOURS"


def backtest_trigger(
    candles: list[Candle],
    trigger_id: str,
    *,
    timeframe: str = "M15",
    symbol: str = "",
    stop_atr_multiple: float = 1.5,
    reward_multiple: float = 2.0,
    max_bars: int = 60,
    warmup: int = 80,
) -> dict[str, Any]:
    """Scan history for one trigger and score the outcomes.

    Only closed candles up to each decision bar are visible to the detector, and
    the trade is resolved strictly on later bars.
    """
    trigger = TRIGGER_REGISTRY.get(trigger_id)
    if trigger is None:
        return {"available": False, "reason": f"Unknown trigger: {trigger_id}"}
    if len(candles) < warmup + max_bars + 5:
        return {
            "available": False,
            "reason": f"Need at least {warmup + max_bars + 5} candles; {len(candles)} supplied.",
        }

    trades: list[Trade] = []
    long = trigger.direction is Direction.BULLISH
    last_exit = -1
    for index in range(warmup, len(candles) - max_bars - 1):
        if index <= last_exit:
            # One position at a time; overlapping entries would double-count edge.
            continue
        history = candles[: index + 1]
        if not trigger.evaluate(history):
            continue
        atr_value = latest(atr(history, 14))
        if not atr_value or not math.isfinite(atr_value) or atr_value <= 0:
            continue
        # Enter at the next bar's open: the trigger bar must have closed first.
        entry = candles[index + 1].open
        distance = atr_value * stop_atr_multiple
        stop = entry - distance if long else entry + distance
        target = entry + distance * reward_multiple if long else entry - distance * reward_multiple
        trade = Trade(
            index=index + 1, time=candles[index + 1].time.isoformat(),
            direction=trigger.direction.value, entry=entry, stop=stop, target=target,
        )
        trade = _resolve_trade(trade, candles[index + 2:], max_bars)
        trades.append(trade)
        last_exit = trade.exit_index or index + 1

    return {
        "available": True,
        "trigger": trigger.as_dict(),
        "symbol": symbol,
        "timeframe": timeframe,
        "parameters": {
            "stop_atr_multiple": stop_atr_multiple, "reward_multiple": reward_multiple,
            "max_bars": max_bars, "warmup": warmup,
        },
        "candles_scanned": len(candles),
        **summarize_trades(trades, candles),
    }


def summarize_trades(trades: list[Trade], candles: list[Candle] | None = None) -> dict[str, Any]:
    """Standard performance statistics over a resolved trade list."""
    total = len(trades)
    wins = [trade for trade in trades if trade.outcome == "WIN"]
    losses = [trade for trade in trades if trade.outcome == "LOSS"]
    expired = [trade for trade in trades if trade.outcome == "EXPIRED"]
    breakeven = [trade for trade in trades if abs(trade.r_multiple) < 1e-9]
    r_values = [trade.r_multiple for trade in trades]
    gross_profit = sum(value for value in r_values if value > 0)
    gross_loss = abs(sum(value for value in r_values if value < 0))

    equity, peak, max_drawdown = 0.0, 0.0, 0.0
    for value in r_values:
        equity += value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)

    def streak(predicate: Callable[[Trade], bool]) -> int:
        best = current = 0
        for trade in trades:
            current = current + 1 if predicate(trade) else 0
            best = max(best, current)
        return best

    by_session: dict[str, dict[str, Any]] = {}
    if candles:
        for trade in trades:
            if trade.index < len(candles):
                session = _session_of(candles[trade.index])
                bucket = by_session.setdefault(session, {"trades": 0, "total_r": 0.0, "wins": 0})
                bucket["trades"] += 1
                bucket["total_r"] += trade.r_multiple
                bucket["wins"] += 1 if trade.outcome == "WIN" else 0
        for bucket in by_session.values():
            bucket["total_r"] = round(bucket["total_r"], 4)
            bucket["win_rate"] = round(bucket["wins"] / bucket["trades"], 4) if bucket["trades"] else 0.0

    return {
        "total_setups": total,
        "wins": len(wins),
        "losses": len(losses),
        "expired": len(expired),
        "breakeven": len(breakeven),
        "win_rate": round(len(wins) / total, 4) if total else 0.0,
        "average_r": round(sum(r_values) / total, 4) if total else 0.0,
        "expectancy": round(sum(r_values) / total, 4) if total else 0.0,
        "total_r": round(sum(r_values), 4),
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss else (None if not gross_profit else float("inf")),
        "max_drawdown_r": round(max_drawdown, 4),
        "max_consecutive_wins": streak(lambda trade: trade.outcome == "WIN"),
        "max_consecutive_losses": streak(lambda trade: trade.outcome == "LOSS"),
        "by_session": by_session,
        "trades": [trade.as_dict() for trade in trades[-50:]],
    }


def compare_triggers(
    candles: list[Candle], trigger_ids: list[str] | None = None, **options: Any
) -> dict[str, Any]:
    """Backtest several triggers on identical data so results are comparable."""
    selected = trigger_ids or list(TRIGGER_REGISTRY)
    results = {
        trigger_id: backtest_trigger(candles, trigger_id, **options)
        for trigger_id in selected
        if trigger_id in TRIGGER_REGISTRY
    }
    ranked = sorted(
        (
            {"trigger": key, "expectancy": value.get("expectancy", 0.0),
             "total_setups": value.get("total_setups", 0), "win_rate": value.get("win_rate", 0.0)}
            for key, value in results.items() if value.get("available")
        ),
        key=lambda item: (item["expectancy"], item["total_setups"]),
        reverse=True,
    )
    return {"results": results, "ranking": ranked, "compared": len(results)}
