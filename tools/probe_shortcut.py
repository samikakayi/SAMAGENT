"""Probe which TradingView keyboard route actually creates a horizontal line."""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import win32api  # noqa: E402
import win32con  # noqa: E402
from PIL import ImageGrab  # noqa: E402

from sam_backend.dpi import ensure_dpi_awareness  # noqa: E402
from sam_backend.trading.drawing import DrawingEngine  # noqa: E402
from sam_backend.trading.tradingview import TradingViewController  # noqa: E402


def grab(geometry):
    return ImageGrab.grab(
        bbox=(geometry["left"], geometry["top"], geometry["right"], geometry["bottom"]), all_screens=True
    )


def chord(modifier_codes, key):
    virtual = win32api.VkKeyScan(key)
    code = virtual & 0xFF
    for modifier in modifier_codes:
        win32api.keybd_event(modifier, 0, 0, 0)
    win32api.keybd_event(code, 0, 0, 0)
    win32api.keybd_event(code, 0, win32con.KEYEVENTF_KEYUP, 0)
    for modifier in reversed(modifier_codes):
        win32api.keybd_event(modifier, 0, win32con.KEYEVENTF_KEYUP, 0)


def main() -> None:
    ensure_dpi_awareness()
    controller = TradingViewController(PROJECT_ROOT / "work" / "acceptance" / "data",
                                       computer_control=True, screen_access=True)
    focused = controller.focus()
    print("focus:", focused.status, focused.verified)
    time.sleep(0.8)
    state = controller.observe()
    geometry = state.client_geometry
    print("client:", geometry)

    # Point at an unambiguous spot inside the candle area.
    target_x = int(geometry["left"] + (geometry["right"] - geometry["left"]) * 0.45)
    target_y = int(geometry["top"] + (geometry["bottom"] - geometry["top"]) * 0.55)
    print(f"pointer -> ({target_x}, {target_y})")

    candidates = [
        ("alt+h", [win32con.VK_MENU], "h"),
        ("alt+H(shift)", [win32con.VK_MENU, win32con.VK_SHIFT], "h"),
        ("ctrl+alt+h", [win32con.VK_CONTROL, win32con.VK_MENU], "h"),
    ]

    for name, modifiers, key in candidates:
        win32api.SetCursorPos((target_x, target_y))
        time.sleep(0.35)
        before = grab(geometry)
        chord(modifiers, key)
        time.sleep(1.0)
        after = grab(geometry)
        rows = DrawingEngine.changed_rows(before, after)
        wide = [row for row in rows if row[1] > 0.6]
        print(f"\n{name}: changed_rows={len(rows)} wide_rows={len(wide)}")
        if wide:
            print("   wide rows (local y, ratio):", wide[:8])
            expected_local = target_y - geometry["top"]
            nearest = min(wide, key=lambda row: abs(row[0] - expected_local))
            print(f"   pointer local y={expected_local}  nearest wide row={nearest[0]} (delta {nearest[0]-expected_local})")
        after.save(str(PROJECT_ROOT / "work" / f"probe_{name.replace('+','_')}.png"))
        # Undo whatever happened so the chart is left as found.
        chord([win32con.VK_CONTROL], "z")
        time.sleep(0.6)


if __name__ == "__main__":
    main()
