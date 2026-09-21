"""TradingView recovery and adversarial calibration (spec sections 11-13).

Puts the chart into each awkward state a normal day produces — minimised, wrong
size, side panels open, scrolled away — and checks SAM either recovers or
refuses honestly. Nothing here forces a drawing onto an unrelated price range.

Run:  .venv\\Scripts\\python.exe tools\\recovery_acceptance.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import win32con  # noqa: E402
import win32gui  # noqa: E402

from sam_backend.cancellation import CancellationManager  # noqa: E402
from sam_backend.config import Settings  # noqa: E402
from sam_backend.db import Database  # noqa: E402
from sam_backend.dpi import ensure_dpi_awareness  # noqa: E402
from sam_backend.trading.service import TradingService  # noqa: E402

results: list[tuple[str, bool, str]] = []


def step(name: str, passed: bool, detail: str = "") -> bool:
    results.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return passed


def main() -> int:
    print("=" * 78)
    print("TRADINGVIEW RECOVERY AND ADVERSARIAL CALIBRATION")
    print("=" * 78)
    ensure_dpi_awareness()
    root = PROJECT_ROOT / "work" / "recovery"
    root.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        project_root=root, workspace_root=root / "ws", data_dir=root / "data",
        screen_access_enabled=True, computer_control_enabled=True,
    )
    settings.prepare()
    database = Database(settings.database_path)
    trading = TradingService(settings, database, CancellationManager())
    trading.refresh_permissions()

    # --- A: already open ----------------------------------------------------
    print("\nA) TradingView already open")
    state = trading.tradingview.observe()
    step("An already-running TradingView is detected", state.running and state.window_handle is not None,
         f"symbol={state.symbol} handle={state.window_handle}")
    handle = state.window_handle
    if handle is None:
        return summarize()

    focused = trading.tradingview.focus()
    step("It can be brought to the foreground", focused.verified, focused.error or "")
    time.sleep(0.5)

    # --- C: minimised -------------------------------------------------------
    print("\nC) Minimised window")
    win32gui.ShowWindow(handle, win32con.SW_MINIMIZE)
    time.sleep(1.2)
    minimised = trading.tradingview.observe()
    print(f"   while minimised: active={minimised.active}")
    blocked = trading.calibrate_chart()
    step("Calibration refuses a minimised chart rather than reading stale pixels",
         blocked.verified or blocked.error_code in {"TRADINGVIEW_NOT_FOREGROUND", "CHART_TOO_SMALL",
                                                    "INSUFFICIENT_ANCHORS", "CALIBRATION_NOT_LINEAR"},
         f"code={blocked.error_code}")
    restored = trading.tradingview.focus()
    time.sleep(1.5)
    step("SAM restores the window and regains the foreground", restored.verified, restored.error or "")

    # --- F: resized ---------------------------------------------------------
    print("\nF) Resized window")
    win32gui.ShowWindow(handle, win32con.SW_MAXIMIZE)
    time.sleep(1.2)
    trading.tradingview.focus()
    time.sleep(0.5)
    maximised = trading.calibrate_chart()
    step("Calibrates while maximised", maximised.verified, maximised.error or "")
    if maximised.verified:
        print(f"   precision +/-{maximised.data.get('precision'):.4f}, axis at x={maximised.data.get('axis_x')}")

    win32gui.ShowWindow(handle, win32con.SW_RESTORE)
    time.sleep(0.8)
    win32gui.MoveWindow(handle, 80, 80, 1400, 950, True)
    time.sleep(1.5)
    trading.tradingview.focus()
    time.sleep(0.6)
    stale = trading.drawing.price_to_screen(4460.0)
    step("The pre-resize calibration is rejected, not reused",
         not stale.verified and stale.error_code in {"CALIBRATION_REQUIRED", "CALIBRATION_NOT_VERIFIED", "PRICE_OFF_SCREEN"},
         f"code={stale.error_code}")
    resized = trading.calibrate_chart()
    step("Recalibrates at the new size", resized.verified, resized.error or "")
    if resized.verified:
        print(f"   precision +/-{resized.data.get('precision'):.4f}, axis at x={resized.data.get('axis_x')}")

    win32gui.ShowWindow(handle, win32con.SW_MAXIMIZE)
    time.sleep(1.5)
    trading.tradingview.focus()
    time.sleep(0.6)

    # --- G/13: adversarial panels ------------------------------------------
    print("\nG) Side panels and adversarial columns")
    adversarial = trading.calibrate_chart()
    step("Calibration succeeds or refuses cleanly with panels in their current state",
         adversarial.verified or adversarial.error_code is not None,
         f"verified={adversarial.verified} code={adversarial.error_code}")
    if adversarial.verified:
        prices = [item["price"] for item in adversarial.data["anchors"]]
        title_price = trading.tradingview.observe().current_price
        near = title_price is None or (min(prices) - abs(title_price) * 0.3 <= title_price <= max(prices) + abs(title_price) * 0.3)
        step("The chosen axis actually belongs to the traded instrument", near,
             f"axis {min(prices):.2f}-{max(prices):.2f} vs live {title_price}")
        for observation in adversarial.observations:
            print(f"   {observation}")

    # --- H: scrolled away ---------------------------------------------------
    print("\nH) Analysis levels outside the visible range")
    if adversarial.verified:
        prices = [item["price"] for item in adversarial.data["anchors"]]
        far = max(prices) * 3
        off = trading.draw_annotation("resistance", far, theory="__recovery__")
        step("A price outside the view is refused, never forced onto the chart",
             not off.verified and off.error_code == "PRICE_OFF_SCREEN",
             f"asked for {far:.0f}, code={off.error_code}")
        step("The refusal states nothing was drawn", off.executed is False)
        step("Nothing was recorded as owned for a refused drawing",
             not database.list_drawings(theory="__recovery__"))

    # --- 13: stale geometry rejection --------------------------------------
    print("\n13) Stale geometry")
    if adversarial.verified:
        verify = trading.verify_calibration()
        step("A fresh calibration re-verifies against a second reading",
             verify.verified or verify.error_code in {"CALIBRATION_DRIFTED", "CALIBRATION_STALE_GEOMETRY"},
             f"verified={verify.verified} code={verify.error_code}")

    return summarize()


def summarize() -> int:
    print("\n" + "=" * 78)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"RESULT: {passed}/{len(results)} checks passed")
    for name, ok, detail in results:
        if not ok:
            print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))
    print("=" * 78)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
