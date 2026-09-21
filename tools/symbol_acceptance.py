"""TradingView symbol-switching acceptance test (spec section 35).

Switches the live chart between real instruments, verifies each change from the
window title rather than from the fact that keys were sent, checks that a stale
calibration cannot survive a symbol change, and always puts the user's original
symbol back before exiting -- including when a step fails.

Two findings from bringing this path up are asserted here as regressions:

  * Chromium ignores synthetic `KEYEVENTF_UNICODE` input, so the symbol has to
    be entered through the clipboard. The user's clipboard must be restored.
  * The top toolbar scrolls horizontally. Clicking whatever token sits left-most
    once hit the "4m" interval button and silently changed the user's timeframe,
    so a symbol click now requires the token to actually read as the symbol.

Run:  .venv\\Scripts\\python.exe tools\\symbol_acceptance.py
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

# Instruments every TradingView account can resolve, so the test does not depend
# on the user's watchlist or subscription.
PROBE_SYMBOLS = ("XAUUSD", "EURUSD")
results: list[tuple[str, bool, str]] = []


def step(name: str, passed: bool, detail: str = "") -> bool:
    results.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return passed


def main() -> int:
    print("=" * 74)
    print("TRADINGVIEW SYMBOL SWITCHING ACCEPTANCE TEST")
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
    controller = trading.tradingview

    print("\n1) Establish the starting state")
    if not step("TradingView focused", controller.focus().verified):
        return summarize()
    time.sleep(0.6)
    original = (controller.observe().symbol or "").upper()
    if not step("The current symbol is readable", bool(original), original or "no title symbol"):
        return summarize()

    try:
        print("\n2) Switch to instruments the chart is not currently showing")
        for symbol in PROBE_SYMBOLS:
            if symbol == original:
                continue
            result = controller.set_symbol(symbol)
            step(f"set_symbol({symbol}) reports success",
                 result.status == ExecutionStatus.SUCCESS and result.verified,
                 result.error_code or f"status={result.status.value}")
            shown = (controller.observe().symbol or "").upper()
            step(f"The chart actually shows {symbol}", shown == symbol, f"title reads {shown!r}")
            step(f"verify_symbol({symbol}) agrees",
                 controller.verify_symbol(symbol).status == ExecutionStatus.SUCCESS)
            time.sleep(0.8)

        print("\n3) A symbol already displayed is a no-op, not a re-entry")
        current = (controller.observe().symbol or "").upper()
        repeat = controller.set_symbol(current)
        step("Re-selecting the displayed symbol succeeds without touching the chart",
             repeat.status == ExecutionStatus.SUCCESS and repeat.verified)

        print("\n4) A bad symbol is refused before anything is typed")
        bad = controller.set_symbol("not a symbol!")
        step("An unsupported symbol is rejected", bad.error_code == "INVALID_SYMBOL", bad.error_code or "")
        step("The rejected attempt did not execute anything", not bad.executed)
        step("The chart is untouched by the rejected attempt",
             (controller.observe().symbol or "").upper() == current)

        print("\n5) Calibration cannot outlive a symbol change")
        handle = controller.observe().window_handle
        controller._last_verified_timeframe[handle] = "M1"
        target = next((s for s in PROBE_SYMBOLS if s != current), PROBE_SYMBOLS[0])
        controller.set_symbol(target)
        step("The cached timeframe was dropped when the instrument changed",
             handle not in controller._last_verified_timeframe)

        print("\n6) The clipboard belongs to the user, not to SAM")
        sentinel = "sam-clipboard-sentinel"
        controller._clipboard_write(sentinel)
        controller.set_symbol(current)
        step("The user's clipboard was restored after the paste",
             controller._clipboard_read() == sentinel,
             f"clipboard now {controller._clipboard_read()!r}")

        print("\n7) The toolbar symbol button is matched by text, never by position")
        # A positional fallback is what previously clicked the interval button.
        source = (PROJECT_ROOT / "sam_backend" / "trading" / "tradingview.py").read_text(encoding="utf-8")
        step("No left-most-token fallback remains in the symbol click",
             "min(candidates, key=lambda word: word.center_x)" not in source)
        step("The symbol click is bounded to the symbol slot",
             "SYMBOL_SLOT_WIDTH" in source)
    finally:
        print("\n8) Restore the user's original symbol")
        restored = controller.set_symbol(original)
        step(f"The chart was returned to {original}",
             restored.status == ExecutionStatus.SUCCESS
             and (controller.observe().symbol or "").upper() == original,
             restored.error_code or "")

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
