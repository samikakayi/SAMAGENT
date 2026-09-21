"""Does Alt+H follow the pointer? Draw at several heights and measure."""

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


def move(x: int, y: int) -> None:
    """Move and jiggle so the app definitely receives a move event."""
    win32api.SetCursorPos((x, y - 3))
    time.sleep(0.12)
    win32api.SetCursorPos((x, y))
    time.sleep(0.45)


def alt_h() -> None:
    code = win32api.VkKeyScan("h") & 0xFF
    win32api.keybd_event(win32con.VK_MENU, 0, 0, 0)
    win32api.keybd_event(code, 0, 0, 0)
    win32api.keybd_event(code, 0, win32con.KEYEVENTF_KEYUP, 0)
    win32api.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)


def undo() -> None:
    code = win32api.VkKeyScan("z") & 0xFF
    win32api.keybd_event(win32con.VK_CONTROL, 0, 0, 0)
    win32api.keybd_event(code, 0, 0, 0)
    win32api.keybd_event(code, 0, win32con.KEYEVENTF_KEYUP, 0)
    win32api.keybd_event(win32con.VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)
    time.sleep(0.7)


def main() -> None:
    ensure_dpi_awareness()
    controller = TradingViewController(PROJECT_ROOT / "work" / "acceptance" / "data",
                                       computer_control=True, screen_access=True)
    controller.focus()
    time.sleep(0.8)
    geometry = controller.observe().client_geometry
    height = geometry["bottom"] - geometry["top"]
    target_x = int(geometry["left"] + (geometry["right"] - geometry["left"]) * 0.45)

    print(f"client={geometry}")
    print(f"{'pointer_y':>10} {'line_row':>10} {'delta':>8}  ratio")
    observations = []
    for fraction in (0.35, 0.50, 0.65):
        pointer_y = int(geometry["top"] + height * fraction)
        move(target_x, pointer_y)
        before = grab(geometry)
        alt_h()
        time.sleep(1.1)
        after = grab(geometry)
        wide = [row for row in DrawingEngine.changed_rows(before, after) if row[1] > 0.6]
        if wide:
            row = max(wide, key=lambda item: item[1])
            delta = row[0] - (pointer_y - geometry["top"])
            observations.append((pointer_y, row[0], delta))
            print(f"{pointer_y:>10} {row[0]:>10} {delta:>8}  {row[1]:.3f}")
        else:
            print(f"{pointer_y:>10} {'none':>10} {'-':>8}  no wide row detected")
        undo()

    if len(observations) >= 2:
        deltas = [item[2] for item in observations]
        spread = max(deltas) - min(deltas)
        print(f"\ndeltas={deltas} spread={spread}")
        if spread <= 6:
            print("=> The line TRACKS the pointer with a constant offset; correctable.")
        else:
            rows = [item[1] for item in observations]
            if max(rows) - min(rows) <= 6:
                print("=> The line is placed at a FIXED location and ignores the pointer.")
            else:
                print("=> The line moves but not in step with the pointer.")


if __name__ == "__main__":
    main()
