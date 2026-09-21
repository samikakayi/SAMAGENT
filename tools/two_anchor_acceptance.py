"""Two-anchor drawing acceptance test (spec section 29).

Calibrates both axes, drags a real trendline between two (price, time) chart
anchors, verifies BOTH endpoints changed on the chart, then removes only the
SAM-created object.

Run:  .venv\\Scripts\\python.exe tools\\two_anchor_acceptance.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import win32con  # noqa: E402

from sam_backend.cancellation import CancellationManager  # noqa: E402
from sam_backend.config import Settings  # noqa: E402
from sam_backend.db import Database  # noqa: E402
from sam_backend.dpi import ensure_dpi_awareness  # noqa: E402
from sam_backend.trading.calibration import calibration_from_row, geometry_hash  # noqa: E402
from sam_backend.trading.drawing import TwoAnchorRequest  # noqa: E402
from sam_backend.trading.service import TradingService  # noqa: E402

TEST_THEORY = "__two_anchor__"
results: list[tuple[str, bool, str]] = []


def step(name: str, passed: bool, detail: str = "") -> bool:
    results.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return passed


def main() -> int:
    print("=" * 74)
    print("TWO-ANCHOR DRAWING ACCEPTANCE TEST")
    print("=" * 74)
    ensure_dpi_awareness()

    root = PROJECT_ROOT / "work" / "acceptance"
    root.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        project_root=root, workspace_root=root / "ws", data_dir=root / "data",
        screen_access_enabled=True, computer_control_enabled=True,
    )
    settings.prepare()
    database = Database(settings.database_path)
    trading = TradingService(settings, database, CancellationManager())
    trading.refresh_permissions()

    print("\n1) Focus and calibrate both axes")
    if not step("TradingView focused", trading.tradingview.focus().verified):
        return summarize()
    time.sleep(0.6)
    calibration_result = trading.calibrate_chart()
    for observation in calibration_result.observations:
        print(f"   obs: {observation}")
    if not step("Price axis calibrated", calibration_result.verified, calibration_result.error or ""):
        return summarize()
    data = calibration_result.data
    if not step("Time axis calibrated", bool(data.get("time_axis_verified")),
                f"minutes/pixel={data.get('minutes_per_pixel')}"):
        return summarize()
    print(f"   1 bar spans {1 / data['minutes_per_pixel']:.2f}px at this zoom")

    state = trading.tradingview.observe()
    geometry = trading.drawing.chart_geometry(state)
    row = database.get_chart_calibration(
        window_handle=state.window_handle, symbol=(state.symbol or "").upper(),
        timeframe=state.timeframe or "UNKNOWN", geometry_hash=geometry_hash(geometry),
    )
    calibration = calibration_from_row(row)
    region = trading.calibrator.plot_region(geometry, calibration.axis_x)

    print("\n2) Choose two on-screen anchors")
    prices = [anchor["price"] for anchor in data["anchors"]]
    low, high = min(prices), max(prices)
    # Two points well inside the visible chart, on a clear diagonal.
    price_a = round(low + (high - low) * 0.30, 2)
    price_b = round(low + (high - low) * 0.70, 2)
    minutes_a = calibration.minutes_at(region["left"] + (region["right"] - region["left"]) * 0.30)
    minutes_b = calibration.minutes_at(region["left"] + (region["right"] - region["left"]) * 0.70)
    anchor_a = trading.drawing.anchor_to_screen(calibration, price_a, minutes_a, region)
    anchor_b = trading.drawing.anchor_to_screen(calibration, price_b, minutes_b, region)
    print(f"   A: {price_a} @ {int(minutes_a)//60:02d}:{int(minutes_a)%60:02d} -> ({anchor_a['x']:.0f}, {anchor_a['y']:.0f})")
    print(f"   B: {price_b} @ {int(minutes_b)//60:02d}:{int(minutes_b)%60:02d} -> ({anchor_b['x']:.0f}, {anchor_b['y']:.0f})")
    if not step("Both anchors are on-screen", anchor_a["on_screen"] and anchor_b["on_screen"]):
        return summarize()

    print("\n3) Drag the trendline and verify both endpoints")
    drawn = trading.drawing.draw_two_anchor(TwoAnchorRequest(
        annotation="trendline", price_a=price_a, minutes_a=minutes_a,
        price_b=price_b, minutes_b=minutes_b, label="SAM two-anchor test", theory=TEST_THEORY,
    ))
    verification = (drawn.data or {}).get("verification") or {}
    print(f"   status={drawn.status} verified={drawn.verified} code={drawn.error_code}")
    for name in ("anchor_a", "anchor_b"):
        item = verification.get(name) or {}
        print(f"   {name}: changed={item.get('changed')} ratio={item.get('ratio'):.4f}"
              if item.get("ratio") is not None else f"   {name}: {item}")
    step("Trendline drawn with both endpoints verified", drawn.verified, drawn.error or "")

    print("\n4) Ownership records the two anchors")
    owned = database.list_drawings(theory=TEST_THEORY)
    payload = owned[0]["payload"] if owned else {}
    step(
        "Both price anchors persisted",
        bool(owned) and owned[0]["price"] == price_a and owned[0]["price_secondary"] == price_b
        and payload.get("two_anchor") is True,
        f"rows={len(owned)} a={owned[0]['price'] if owned else None} b={owned[0]['price_secondary'] if owned else None}",
    )

    print("\n5) Remove only the SAM object")
    removed = 0
    for drawing in database.list_drawings(theory=TEST_THEORY):
        anchors = drawing["payload"].get("screen_a") or {}
        changed, _ = trading.drawing._select_and_delete(
            int(anchors.get("x", 0)), float(anchors.get("y", 0)),
            geometry=geometry, axis_x=calibration.axis_x, delete_key=win32con.VK_DELETE,
        )
        if changed:
            database.delete_drawings(drawing_id=drawing["id"])
            removed += 1
    step("SAM trendline removed from the chart", removed == len(owned) and not database.list_drawings(theory=TEST_THEORY),
         f"removed={removed}")
    return summarize()


def summarize() -> int:
    print("\n" + "=" * 74)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"RESULT: {passed}/{len(results)} checks passed")
    for name, ok, detail in results:
        if not ok:
            print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))
    print("=" * 74)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
