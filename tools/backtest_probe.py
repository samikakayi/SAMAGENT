"""Run every entry trigger over real MT5 history and print the statistics."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam_backend.trading.market_data import MetaTrader5Provider  # noqa: E402
from sam_backend.trading.research import TRIGGER_REGISTRY, compare_triggers  # noqa: E402


def main() -> None:
    candles = MetaTrader5Provider().fetch("XAUUSD", "M15", 3000).candles
    print(f"REAL DATA: {len(candles)} M15 candles {candles[0].time.date()} -> {candles[-1].time.date()}")
    print(f"triggers registered: {len(TRIGGER_REGISTRY)}\n")

    outcome = compare_triggers(candles, timeframe="M15", symbol="XAUUSD")
    header = f"{'trigger':22s} {'setups':>7s} {'win%':>7s} {'avgR':>8s} {'PF':>7s} {'maxDD':>7s} {'maxL':>5s}"
    print(header)
    print("-" * len(header))
    for row in outcome["ranking"]:
        data = outcome["results"][row["trigger"]]
        factor = data["profit_factor"]
        factor_text = f"{factor:.2f}" if isinstance(factor, (int, float)) and factor == factor else "n/a"
        print(
            f"{row['trigger']:22s} {data['total_setups']:7d} {data['win_rate'] * 100:6.1f}% "
            f"{data['average_r']:+8.3f} {factor_text:>7s} {data['max_drawdown_r']:7.2f} "
            f"{data['max_consecutive_losses']:5d}"
        )

    if not outcome["ranking"]:
        print("\nNo trigger produced a resolvable setup on this history.")
        return

    best = outcome["ranking"][0]["trigger"]
    data = outcome["results"][best]
    print(f"\n--- {best} detail ---")
    for session, stats in sorted(data["by_session"].items()):
        print(f"  {session:12s} trades={stats['trades']:3d} total_r={stats['total_r']:+8.3f} win_rate={stats['win_rate']:.2f}")
    print("  sample trades:")
    for trade in data["trades"][:5]:
        print(
            f"    {trade['time'][:16]} {trade['direction']:7s} entry={trade['entry']:.2f} "
            f"stop={trade['stop']:.2f} target={trade['target']:.2f} -> {trade['outcome']:7s} "
            f"R={trade['r_multiple']:+.2f} bars={trade['bars_held']}"
        )


if __name__ == "__main__":
    main()
