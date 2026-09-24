"""Live acceptance for the trading engine (run by hand on this PC).

    .venv\\Scripts\\python.exe acceptance\\engine_live.py [--draw]

Read-only on MetaTrader 5 (no order functions exist in SAM 2's feed). Uses a
temporary SAM home under work/ so nothing in the user's data changes. With
``--draw`` and TradingView Desktop showing gold with its CDP port open, the
analysis is drawn and then EVERY drawing this script made is removed again;
the chart's symbol and timeframe are never changed. Prints no account data.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sam.app import App  # noqa: E402
from sam.trading import tools as trading_tools  # noqa: E402
from sam.trading.analyze import analyze_market, compact_report  # noqa: E402
from sam.trading.common import TIMEFRAME_SECONDS  # noqa: E402

TFS = ["H1", "M15", "M5", "M1"]


def iso(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


async def main(draw: bool) -> int:
    home = ROOT / "work" / "acceptance-engine-home"
    (home / "data").mkdir(parents=True, exist_ok=True)
    app = App(home, environ={})
    app.loop = asyncio.get_running_loop()
    app.bus.bind_loop(app.loop)
    trading_tools.register(app)
    try:
        from sam.trading import chart_tools  # the chart bridge (other builder); optional
        chart_tools.register(app)
    except Exception as exc:  # noqa: BLE001
        print("chart bridge unavailable:", type(exc).__name__)
    out: dict = {}
    feed = app.trading.mt5
    started = time.perf_counter()
    connected = await feed.connect()
    out["mt5_connect_ms"] = round((time.perf_counter() - started) * 1000, 1)
    status = await feed.status()
    out["mt5"] = {k: status[k] for k in ("connected", "broker_offset_s", "server_time_ok", "offset_source",
                                         "terminal_connected", "symbols")}
    if not connected:
        print(json.dumps(out, indent=1, ensure_ascii=False))
        return 1
    now = time.time()
    bars_report = {}
    for tf in TFS:
        began = time.perf_counter()
        bars = await feed.bars("XAUUSD", tf, 600)
        last = bars[-1]["time"]
        interval = TIMEFRAME_SECONDS[tf]
        bars_report[tf] = {"bars": len(bars), "ms": round((time.perf_counter() - began) * 1000, 1),
                           "last_bar_open": iso(last), "age_s": round(now - last, 1),
                           "forming_bar_contains_now": last <= now < last + interval,
                           "future_bars": sum(1 for b in bars if b["time"] > now + 5)}
    out["bars"] = bars_report
    tick = await feed.tick("TVC:GOLD")
    out["tick"] = {"symbol": tick["symbol"], "bid": tick["bid"], "ask": tick["ask"], "spread": tick["spread"],
                   "time": iso(tick["time"]), "age_s": round(time.time() - tick["time"], 2)}
    tv = app.trading.tv
    chart = None
    if tv is not None:
        try:
            if await asyncio.wait_for(tv.connect(), 5):
                state = await tv.chart_state()
                chart = {"symbol": state.get("symbol"), "timeframe": state.get("timeframe")}
        except Exception as exc:  # noqa: BLE001
            chart = {"error": type(exc).__name__}
    out["chart"] = chart
    try:
        report = await run_analyses(app, tv, draw, out)
    finally:
        if draw and tv is not None:
            # Safety net: remove every analysis drawing this script's (temporary) DB owns.
            out["final_cleanup_removed"] = await tv.clear_my_drawings(tag="analysis")
    out["text_ckb"] = report["text_ckb"]
    await trading_tools.stop(app)
    if tv is not None:
        await tv.close()
    app.close()
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


async def run_analyses(app: App, tv, draw: bool, out: dict) -> dict:
    report: dict = {}
    for label, mode in (("dry", "none"), ("draw", "full" if draw else None)):
        if mode is None:
            continue
        began = time.perf_counter()
        report = await analyze_market(app, "XAUUSD", draw=mode)
        compact = compact_report(report)
        out[f"analysis_{label}"] = {
            "wall_ms": round((time.perf_counter() - began) * 1000, 1), "engine_ms": report["engine_ms"],
            "total_ms": report["total_ms"], "data_source": report["data_source"], "sources": report["sources"],
            "feed_offset": report.get("feed_offset"),
            "feed_offset_measured": report.get("feed_offset_measured"), "verdict": report["verdict"], "direction": report["direction"],
            "stale": report["stale"], "failed_checks": [c["name"] for c in report["checks"] if not c["passed"]],
            "trend": report["trend"], "support": compact["support"], "resistance": compact["resistance"],
            "zones": compact["zones"], "order_blocks": len(report["order_blocks"]),
            "summary_ckb": report["summary_ckb"], "drawing": report.get("drawing", {}).get("plan"),
            "dry_run": report.get("drawing", {}).get("dry_run"), "drawn": len(report["drawn"]),
        }
        if report["drawn"]:
            mine = await tv.my_drawings()
            tag = f"analysis:{report['analysis_id']}"
            removed = await tv.clear_my_drawings(tag=tag)
            left = [d for d in await tv.my_drawings() if d["tag"] == tag]
            out[f"analysis_{label}"].update(owned_after_draw=len(mine), removed=removed, left_after_clear=len(left))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--draw", action="store_true", help="draw on the chart, then remove every drawing made")
    sys.exit(asyncio.run(main(parser.parse_args().draw)))
