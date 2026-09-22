"""The desktop as a drawing tool sees it: a pointer, a keyboard and the screen.

Every OS call the drawing engine makes goes through here, which is what lets
the engine be driven without a window: hand it a stand-in and the gates, the
price mapping, the verification maths and the ownership records all run for
real. The Windows imports are deferred so the engine can be constructed and
its pure paths tested anywhere; only actually moving the pointer needs win32.
"""

from __future__ import annotations

import os
import time
from typing import Any

# Pointer approach distance and settle time that make the crosshair follow.
MOUSE_SETTLE_OFFSET = 3
MOUSE_SETTLE_SECONDS = 0.45
# Intermediate positions sent during a drag so the app tracks the motion.
DRAG_STEPS = 12


class DesktopInput:
    """Sends input to whatever is on screen. Knows nothing about charts."""

    # Fixed Windows virtual-key codes; the engine names keys by intent, not number.
    ESCAPE = 0x1B
    DELETE = 0x2E

    @staticmethod
    def modules() -> tuple[Any, Any]:
        if os.name != "nt":
            raise RuntimeError("TradingView drawing is available only on Windows")
        import win32api
        import win32con

        return win32api, win32con

    def move_mouse(self, x: int, y: int) -> None:
        """Move the pointer so the chart's crosshair actually follows it.

        A single `SetCursorPos` to a position the cursor may already occupy does
        not reliably produce a move event, leaving TradingView's crosshair — and
        therefore any shortcut-placed drawing — at a stale price. Approaching the
        target from a few pixels away guarantees the move is delivered.
        """
        win32api, _ = self.modules()
        win32api.SetCursorPos((int(x), int(y) - MOUSE_SETTLE_OFFSET))
        time.sleep(0.12)
        win32api.SetCursorPos((int(x), int(y)))
        time.sleep(MOUSE_SETTLE_SECONDS)

    @staticmethod
    def virtual_key(key: str) -> int:
        """The virtual-key code for a shortcut, independent of keyboard layout.

        `VkKeyScan` asks the *current* layout which key produces a character, so
        with a Kurdish or Arabic layout selected it returns -1 for plain Latin
        letters and every drawing shortcut fails. Shortcuts are dispatched by
        virtual key, not by character, and for ASCII letters and digits that
        code is fixed ('A' is 0x41 on every layout), so it can be taken directly.
        """
        if len(key) == 1 and (key.isascii() and (key.isalpha() or key.isdigit())):
            return ord(key.upper())
        import win32api

        virtual = win32api.VkKeyScan(key)
        if virtual == -1:
            raise RuntimeError(f"Cannot map shortcut key {key!r}")
        return virtual & 0xFF

    def press_chord(self, modifier: str, key: str) -> None:
        win32api, win32con = self.modules()
        modifier_code = {"alt": win32con.VK_MENU, "ctrl": win32con.VK_CONTROL, "shift": win32con.VK_SHIFT}[modifier]
        key_code = self.virtual_key(key)
        win32api.keybd_event(modifier_code, 0, 0, 0)
        win32api.keybd_event(key_code, 0, 0, 0)
        win32api.keybd_event(key_code, 0, win32con.KEYEVENTF_KEYUP, 0)
        win32api.keybd_event(modifier_code, 0, win32con.KEYEVENTF_KEYUP, 0)
        time.sleep(0.2)

    def click(self, x: int, y: int) -> None:
        win32api, win32con = self.modules()
        self.move_mouse(x, y)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        time.sleep(0.12)

    def press_key(self, key_code: int) -> None:
        win32api, win32con = self.modules()
        win32api.keybd_event(key_code, 0, 0, 0)
        win32api.keybd_event(key_code, 0, win32con.KEYEVENTF_KEYUP, 0)
        time.sleep(0.15)

    def drag(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        """Press, move through intermediate points, release.

        A single jump from press to release is often treated as a click, so the
        motion is delivered in steps the way a hand would move the pointer.
        """
        win32api, win32con = self.modules()
        self.move_mouse(*start)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(0.12)
        for step in range(1, DRAG_STEPS + 1):
            ratio = step / DRAG_STEPS
            x = int(start[0] + (end[0] - start[0]) * ratio)
            y = int(start[1] + (end[1] - start[1]) * ratio)
            win32api.SetCursorPos((x, y))
            time.sleep(0.03)
        time.sleep(0.15)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        time.sleep(0.2)

    def grab(self, bbox: tuple[int, int, int, int]) -> Any:
        """The pixels currently on screen inside that rectangle, all monitors."""
        from PIL import ImageGrab

        from ..dpi import ensure_dpi_awareness

        ensure_dpi_awareness()
        return ImageGrab.grab(bbox=bbox, all_screens=True)
