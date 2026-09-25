"""Live: analyze_market draws on the real TradingView chart and clears only SAM's drawings.

    .venv\\Scripts\\python.exe acceptance\\engine_draw_live.py

Design acceptance 3/4: analysis on live gold returns in < 5 s (engine < 1.5 s),
its levels/zones are drawn on the chart the user sees, and «بیانسڕەوە» removes
only SAM's drawings. The chart's symbol and timeframe are never changed: the
analysis uses the timeframes the chart does NOT show (H1/M15/M5 minus the
chart's), so its bars come from MT5 -- on 2026-09-24 TradingView's own series
had formed no bar for 7.8 h and the stale guard (correctly) refused to draw
from it. The user's drawings stay untouched. MT5 is read only; a throw-away SAM
home under work/ holds the drawings table. No keys, no LLM (vision is off),
nothing spoken.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import time

from _common import ROOT, Acceptance

from sam.app import App
from sam.trading import chart_tools
from sam.trading import tools as trading_tools
from sam.trading.analyze import analyze_market


async def run() -> int:
    acc = Acceptance("engine_draw_live")
    home = ROOT / "work" / "acceptance-draw-home"
    shutil.rmtree(home, ignore_errors=True)
    (home / "data").mkdir(parents=True, exist_ok=True)
    app = App(home, environ={})
    app.loop = asyncio.get_running_loop()
    app.bus.bind_loop(app.loop)
    trading_tools.register(app)
    chart_tools.register(app)
    tv = app.trading.tv
    original: dict = {}
    try:
        with acc.check("TradingView connected and showing gold") as c:
            if not await asyncio.wait_for(tv.connect(), 8):
                c.skip("TradingView Desktop is not running with its local DevTools port")
            state = await tv.chart_state()
            original = {"symbol": state.get("symbol"), "timeframe": state.get("timeframe")}
            c.data = {"original": original, "user_drawings": state.get("user_drawings")}
            if "XAU" not in str(original["symbol"]).upper() and "GOLD" not in str(original["symbol"]).upper():
                c.skip(f"the chart shows {original['symbol']}, not gold (this check never switches the symbol)")
            await app.trading.mt5.connect()
        before = {str(s["id"]) for s in (await tv._call("shapes"))["shapes"]}  # noqa: SLF001 - the page's own list
        timeframes = [tf for tf in ("H1", "M15", "M5") if tf != str(original.get("timeframe") or "")]
        with acc.check(f"analysis {timeframes} is drawn (< 5 s, engine < 1.5 s)") as c:
            began = time.perf_counter()
            report = await analyze_market(app, "XAUUSD", timeframes=timeframes, draw="full", vision=False)
            wall = round((time.perf_counter() - began) * 1000)
            mine = await tv.my_drawings()
            c.data = {"wall_ms": wall, "engine_ms": report.get("engine_ms"), "verdict": report.get("verdict"),
                      "stale": report.get("stale"), "stale_timeframes": report.get("stale_timeframes"),
                      "sources": report.get("sources"), "feed_offset": report.get("feed_offset"),
                      "drawn": len(report.get("drawn") or []), "on_chart": len(mine),
                      "summary_ckb": report.get("summary_ckb")}
            assert wall < 5000 and (report.get("engine_ms") or 0) < 1500, c.data
            assert report.get("drawn") and mine, c.data
            c.detail = f"{len(mine)} drawings in {wall} ms: {report.get('summary_ckb')}"
        with acc.check("clearing removes only SAM's drawings") as c:
            removed = await tv.clear_my_drawings(tag="analysis")
            after = {str(s["id"]) for s in (await tv._call("shapes"))["shapes"]}  # noqa: SLF001
            c.data = {"removed": removed, "user_drawings_kept": before <= after, "left_over": sorted(after - before)}
            assert before <= after and not (after - before), c.data
    finally:
        try:
            if tv is not None and tv.connected:
                await tv.clear_my_drawings(tag="analysis")
                state = await tv.chart_state()
                print("chart now:", {"symbol": state.get("symbol"), "timeframe": state.get("timeframe")},
                      "original:", original)
        finally:
            await trading_tools.stop(app)
            if tv is not None:
                await tv.close()
            app.close()
            shutil.rmtree(home, ignore_errors=True)
    return acc.finish()


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    sys.exit(asyncio.run(run()))
