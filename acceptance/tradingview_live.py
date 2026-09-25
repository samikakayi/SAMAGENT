"""Live check of the TradingView chart bridge on this PC (run by hand).

Needs TradingView Desktop already running WITH the DevTools port (it never
restarts TradingView: ``allow_restart=False``). Steps: read the chart; set the
timeframe to 15 and back; keep-gold / explicit symbol switch and back; fetch
300 bars (timed); draw a horizontal line, trend line, rectangle, fib and text
labelled "SAM test" plus the other five kinds; verify via the page's shape
list; save chart screenshots; then remove ONLY those drawings and verify the
chart is back to its original symbol, timeframe and drawing count.

Uses a throw-away SAM home (the drawings table lives there), no keys, no LLM.

    .venv\\Scripts\\python.exe acceptance\\tradingview_live.py --out <folder>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam.app import App  # noqa: E402
from sam.trading import chart_tools  # noqa: E402
from sam.trading.tv_parse import same_resolution  # noqa: E402

TAG = "acceptance"


def ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 1)


async def shape_ids(tv: Any) -> set[str]:
    info = await tv._call("shapes")          # the page's own list (all drawings, SAM's and the user's)
    return {str(s["id"]) for s in info["shapes"]}


async def run(out: Path, home: Path) -> dict[str, Any]:
    (home / "data").mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    app = App(home, environ={})
    chart_tools.register(app)
    tv = app.trading.tv
    report: dict[str, Any] = {"checks": {}, "timings_ms": {}}
    checks, timings = report["checks"], report["timings_ms"]

    began = time.perf_counter()
    running = await tv.ensure_running(allow_restart=False, focus=False)
    timings["ensure_running_connect"] = ms(began)
    report["ensure_running"] = {k: running[k] for k in ("ok", "state")}
    if not running["ok"]:
        report["aborted"] = "TradingView is not reachable on the DevTools port"
        return report

    samples = []
    for _ in range(5):
        began = time.perf_counter()
        original = await tv.chart_state()
        samples.append(ms(began))
    timings["chart_state_median"] = statistics.median(samples)
    original_ids = await shape_ids(tv)
    report["original"] = {"symbol": original["symbol"], "resolution": original["resolution"],
                          "shapes": len(original_ids), "bars_loaded": original["bar_count"], "price": original["price"]}
    try:
        # timeframe -> 15 and back
        began = time.perf_counter()
        to15 = await tv.set_timeframe("١٥ خولەک")
        timings["set_timeframe_15"] = ms(began)
        checks["timeframe_15"] = bool(to15["ok"] and (await tv.chart_state())["resolution"] == "15")
        began = time.perf_counter()
        back = await tv.set_timeframe(original["resolution"])
        timings["set_timeframe_back"] = ms(began)
        checks["timeframe_back"] = bool(back["ok"] and same_resolution((await tv.chart_state())["resolution"],
                                                                       original["resolution"]))

        # symbol: 'گۆڵد' keeps the user's gold feed; an explicit exchange switches; then back
        keep = await tv.set_symbol("گۆڵد")
        checks["gold_keeps_current_feed"] = bool(keep["ok"] and not keep["changed"]) if original["canonical"] == "XAUUSD" \
            else None
        began = time.perf_counter()
        other = "OANDA:XAUUSD" if original["symbol"] != "OANDA:XAUUSD" else "TVC:GOLD"
        switched = await tv.set_symbol(other)
        timings["set_symbol_switch"] = ms(began)
        began = time.perf_counter()
        restored = await tv.set_symbol(original["symbol"])
        timings["set_symbol_back"] = ms(began)
        checks["symbol_switch_and_back"] = bool(switched["ok"] and switched["symbol"] == other and restored["ok"]
                                                and (await tv.chart_state())["symbol"] == original["symbol"])

        # bars
        began = time.perf_counter()
        bars = await tv.bars(300)
        timings["bars_300"] = ms(began)
        report["bars"] = {"count": len(bars), "source": tv.last_bars_source, "first": bars[0]["time"],
                          "last": bars[-1]["time"], "last_close": bars[-1]["close"],
                          "age_s": round(time.time() - bars[-1]["time"], 1)}
        checks["bars_300_ascending"] = len(bars) == 300 and all(a["time"] < b["time"] for a, b in zip(bars, bars[1:]))

        # draw the five requested kinds labelled "SAM test"
        window = bars[-60:]
        high, low = max(b["high"] for b in window), min(b["low"] for b in window)
        last = bars[-1]
        items = [
            {"kind": "horizontal_line", "points": [{"price": last["close"]}], "text": "SAM test", "color": "zone"},
            {"kind": "trend_line", "points": [{"price": window[0]["low"], "time": window[0]["time"]},
                                              {"price": last["low"], "time": last["time"]}], "text": "SAM test"},
            {"kind": "rectangle", "points": [{"price": high, "bars_ago": 30}, {"price": high - (high - low) * 0.25}],
             "text": "SAM test", "color": "resistance"},
            {"kind": "fib_retracement", "points": [{"price": low, "time": window[0]["time"]},
                                                   {"price": high, "time": last["time"]}]},
            {"kind": "text", "points": [{"price": (high + low) / 2, "bars_ago": 20}], "text": "SAM test"},
        ]
        began = time.perf_counter()
        drawn = await tv.draw_many(items, tag=TAG)
        timings["draw_5"] = ms(began)
        report["draw_5"] = {"drawn": drawn["drawn"], "kinds": drawn["kinds"], "errors": drawn["errors"],
                            "page_ms": drawn.get("ms")}
        present = await shape_ids(tv)
        checks["draw_5_verified"] = drawn["drawn"] == 5 and set(drawn["ids"]) <= present
        # read the created shapes' own properties back from TradingView (label + colour)
        props = await tv.evaluate(
            "(() => { const c = TradingViewApi.activeChart(); return " + json.dumps(drawn["ids"]) + ".map(id => {"
            " const p = c.getShapeById(id).getProperties();"
            " return {id, text: p.text || '', color: p.linecolor || p.color || ''}; }); })()")
        report["draw_5_properties"] = props
        wanted = {i: item.get("text", "") for i, item in zip(drawn["ids"], items)}
        checks["labels_read_back"] = all(p["text"] == wanted[p["id"]] for p in props)
        rows = app.db.query("SELECT tv_id, kind, symbol, timeframe, points, text FROM drawings WHERE tag=?", (TAG,))
        checks["ownership_rows"] = len(rows) == 5 and all(r["text"] in ("SAM test", "") for r in rows)
        report["drawn_points_sample"] = json.loads(rows[0]["points"]) if rows else None

        began = time.perf_counter()
        picture = await tv.screenshot(max_width=1440)
        timings["screenshot"] = ms(began)
        (out / "chart_sam_test.jpg").write_bytes(picture)
        report["screenshot"] = dict(tv.last_screenshot, file=str(out / "chart_sam_test.jpg"))

        # the remaining five kinds
        mid = (high + low) / 2
        more = [
            {"kind": "horizontal_ray", "points": [{"price": high, "bars_ago": 45}], "text": "SAM test"},
            {"kind": "arrow_up", "points": [{"price": low, "bars_ago": 10}], "text": "SAM"},
            {"kind": "arrow_down", "points": [{"price": high, "bars_ago": 5}], "text": "SAM"},
            {"kind": "long_position", "points": [{"price": mid, "bars_ago": 15}, {"price": low}, {"price": high}]},
            {"kind": "short_position", "points": [{"price": mid, "bars_ago": 40}, {"price": high}, {"price": low}]},
        ]
        began = time.perf_counter()
        drawn2 = await tv.draw_many(more, tag=TAG + ":more")
        timings["draw_other_5"] = ms(began)
        report["draw_other_5"] = {"drawn": drawn2["drawn"], "kinds": drawn2["kinds"], "errors": drawn2["errors"]}
        checks["draw_other_5_verified"] = drawn2["drawn"] == 5 and set(drawn2["ids"]) <= await shape_ids(tv)
        (out / "chart_all_kinds.jpg").write_bytes(await tv.screenshot(max_width=1440))
        state = await tv.chart_state()
        checks["state_counts_mine"] = state["my_drawings"] == 10 and state["user_drawings"] == len(original_ids)
    finally:
        began = time.perf_counter()
        cleared = await tv.clear(tag=TAG)
        timings["clear_my_drawings"] = ms(began)
        report["clear"] = cleared
        final = await tv.chart_state()
        if final["symbol"] != original["symbol"]:
            await tv.set_symbol(original["symbol"])
        if not same_resolution(final["resolution"], original["resolution"]):
            await tv.set_timeframe(original["resolution"])
        final = await tv.chart_state()
        final_ids = await shape_ids(tv)
        report["final"] = {"symbol": final["symbol"], "resolution": final["resolution"], "shapes": len(final_ids)}
        checks["restored_exactly"] = (final["symbol"] == original["symbol"] and final["resolution"] ==
                                      original["resolution"] and final_ids == original_ids)
        leftovers = app.db.query("SELECT COUNT(*) AS n FROM drawings WHERE removed_at IS NULL")[0]["n"]
        checks["no_owned_leftovers"] = leftovers == 0
        cdp = app.db.query("SELECT extra, ms FROM timings WHERE stage='tv_cdp' ORDER BY id")
        report["tv_cdp_rows"] = len(cdp)
        await tv.close()
        app.close()
    report["ok"] = all(v is not False for v in checks.values())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "work" / "tv-live"))
    parser.add_argument("--home", default="")
    args = parser.parse_args()
    home = Path(args.home) if args.home else Path(tempfile.mkdtemp(prefix="sam2-tv-live-", dir=ROOT / "work"))
    report = asyncio.run(run(Path(args.out), home))
    text = json.dumps(report, indent=1, ensure_ascii=False, default=str)
    (Path(args.out) / "report.json").write_text(text, encoding="utf-8")
    sys.stdout.buffer.write(text.encode("utf-8") + b"\n")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
