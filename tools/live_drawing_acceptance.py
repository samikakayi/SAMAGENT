"""Live TradingView drawing acceptance test (spec section 27).

Drives the real pointer against the installed TradingView Desktop:
focus -> calibrate -> draw a horizontal line at a safe visible price ->
screenshot -> verify -> measure placement error -> confirm persistence ->
delete only the SAM-created drawing -> verify cleanup.

Run:  .venv\\Scripts\\python.exe tools\\live_drawing_acceptance.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam_backend.cancellation import CancellationManager  # noqa: E402
from sam_backend.config import Settings  # noqa: E402
from sam_backend.contracts import ExecutionStatus  # noqa: E402
from sam_backend.db import Database  # noqa: E402
from sam_backend.dpi import ensure_dpi_awareness  # noqa: E402
from sam_backend.trading.service import TradingService  # noqa: E402

TEST_THEORY = "__acceptance__"
results: list[tuple[str, bool, str]] = []


def step(name: str, passed: bool, detail: str = "") -> bool:
    results.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return passed


def main() -> int:
    print("=" * 74)
    print("LIVE TRADINGVIEW DRAWING ACCEPTANCE TEST")
    print("=" * 74)
    dpi = ensure_dpi_awareness()
    print(f"DPI: {dpi['mode']} scale={dpi.get('scale')} consistent={dpi.get('coordinates_consistent')}")

    root = PROJECT_ROOT / "work" / "acceptance"
    root.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        project_root=root,
        workspace_root=root / "ws",
        data_dir=root / "data",
        screen_access_enabled=True,
        computer_control_enabled=True,
    )
    settings.prepare()
    database = Database(settings.database_path)
    trading = TradingService(settings, database, CancellationManager())
    trading.refresh_permissions()

    # --- 1. focus -----------------------------------------------------------
    print("\n1) Focus TradingView")
    focused = trading.tradingview.focus()
    if not step("TradingView focused and confirmed foreground", focused.verified, focused.error or ""):
        return summarize()
    time.sleep(0.6)

    state = trading.tradingview.observe()
    print(f"   symbol={state.symbol} title_price={state.current_price} client={state.client_geometry}")

    # --- 2. calibrate -------------------------------------------------------
    print("\n2) Calibrate the price axis")
    calibration = trading.calibrate_chart()
    for observation in calibration.observations:
        print(f"   obs: {observation}")
    if not step("Calibration verified", calibration.verified, calibration.error or ""):
        return summarize()
    data = calibration.data
    slope = data["slope"]
    precision = data.get("precision") or 0.0
    print(f"   slope={slope:.6f} price/px  precision=+/-{precision:.4f}  axis_x={data.get('axis_x')}")

    # --- 3. choose a safe visible test price --------------------------------
    print("\n3) Choose a safe visible test price")
    anchors = data["anchors"]
    prices = [anchor["price"] for anchor in anchors]
    # Mid-range of the fitted anchors: guaranteed on screen, away from the edges.
    test_price = round((max(prices) + min(prices)) / 2, 2)
    placement = trading.drawing.price_to_screen(test_price)
    if not step(
        f"Test price {test_price} maps on-screen",
        bool(placement.verified and placement.data.get("on_screen")),
        placement.error or "",
    ):
        return summarize()
    expected_y = placement.data["y"]
    print(f"   {test_price} -> y={expected_y:.1f}px")

    # --- 4/5/6. draw, capture, verify ---------------------------------------
    print("\n4-6) Draw the horizontal line, capture, and verify")
    drawn = trading.draw_annotation(
        "support", test_price, label="SAM acceptance test", theory=TEST_THEORY
    )
    print(f"   status={drawn.status} verified={drawn.verified} code={drawn.error_code}")
    if drawn.error:
        print(f"   error: {drawn.error}")
    verification = (drawn.data or {}).get("verification") or {}
    print(f"   verification: {verification}")
    if not step("Line drawn and visually verified on the chart", drawn.verified, drawn.error or ""):
        # Still attempt cleanup of anything recorded before returning.
        cleanup(trading, database)
        return summarize()

    # --- 7. measure placement error -----------------------------------------
    print("\n7) Measure placement error")
    matched_row = verification.get("matched_row")
    expected_local = verification.get("expected_local_y")
    if matched_row is not None and expected_local is not None:
        error_px = abs(matched_row - expected_local)
        error_price = error_px * abs(slope)
        step(
            "Placement error within calibration precision",
            error_price <= max(precision, abs(slope) * 4),
            f"{error_px:.1f}px = {error_price:.4f} price units (precision +/-{precision:.4f})",
        )
    else:
        step("Placement error measurable", False, "verification did not report a matched row")

    # --- 8. persistence ------------------------------------------------------
    print("\n8) Confirm ownership persistence")
    owned = database.list_drawings(theory=TEST_THEORY)
    step(
        "Drawing persisted with full provenance",
        len(owned) == 1 and owned[0]["verified"] and owned[0]["price"] == test_price,
        f"rows={len(owned)}"
        + (
            f" symbol={owned[0]['symbol']} layer={owned[0]['layer']} tf={owned[0]['timeframe']} verified={owned[0]['verified']}"
            if owned
            else ""
        ),
    )

    # A second, unrelated record stands in for a user drawing SAM must not touch.
    foreign = database.record_drawing(
        symbol=owned[0]["symbol"] if owned else "XAUUSD",
        layer="NOTES",
        drawing_type="support",
        theory="__not_the_test__",
        price=test_price + 500,
    )

    # --- 9/10. delete only the SAM test drawing, verify cleanup -------------
    print("\n9-10) Delete only the acceptance drawing and verify cleanup")
    cleared = trading.clear_drawings(theory=TEST_THEORY)
    print(f"   status={cleared.status} removed={(cleared.data or {}).get('count')} skipped={(cleared.data or {}).get('skipped')}")
    remaining = database.list_drawings(theory=TEST_THEORY)
    step(
        "SAM acceptance drawing removed from the chart and from ownership",
        cleared.status is ExecutionStatus.SUCCESS and not remaining,
        cleared.error or f"remaining={len(remaining)}",
    )
    step(
        "The unrelated drawing was left untouched",
        database.get_drawing(foreign["id"]) is not None,
        "a non-matching record must never be selected or deleted",
    )
    database.delete_drawings(drawing_id=foreign["id"])
    return summarize()


def cleanup(trading: TradingService, database: Database) -> None:
    """Remove the acceptance annotation from the chart.

    Ownership rows are deliberately left in place when the chart deletion fails:
    dropping the record would orphan a real object on the user's chart that SAM
    could never find again.
    """
    try:
        trading.clear_drawings(theory=TEST_THEORY)
    except Exception:
        pass
    remaining = database.list_drawings(theory=TEST_THEORY)
    if remaining:
        print(f"   NOTE: {len(remaining)} acceptance drawing(s) remain on the chart and stay tracked for cleanup.")


def summarize() -> int:
    print("\n" + "=" * 74)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"RESULT: {passed}/{len(results)} checks passed")
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail and not ok else ""))
    print("=" * 74)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
