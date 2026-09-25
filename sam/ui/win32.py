"""Win32 window-style helpers (cosmetic; no-ops on other platforms/offscreen).

Kept apart from the widgets so ``island.py`` and ``panel.py`` stay about
behaviour and layout. Both helpers never raise: a failed style call must not
break the UI.
"""

from __future__ import annotations

import sys

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QWidget


def apply_no_activate(widget: QWidget) -> bool:
    """Win32: WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW, no WS_EX_APPWINDOW, so a click
    on the island never takes focus away from the user's app. Qt's
    ``WindowDoesNotAcceptFocus`` covers most of it; setting the style directly
    makes it independent of the Qt version."""
    if sys.platform != "win32" or QGuiApplication.platformName() != "windows":
        return False
    try:
        import ctypes

        user32 = ctypes.windll.user32
        hwnd = int(widget.winId())
        get_style = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
        set_style = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
        get_style.restype = ctypes.c_ssize_t
        get_style.argtypes = [ctypes.c_void_p, ctypes.c_int]
        set_style.restype = ctypes.c_ssize_t
        set_style.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ssize_t]
        GWL_EXSTYLE, WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW, WS_EX_APPWINDOW = -20, 0x08000000, 0x80, 0x40000
        style = get_style(hwnd, GWL_EXSTYLE)
        new = (style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW) & ~WS_EX_APPWINDOW
        if new != style:
            set_style(hwnd, GWL_EXSTYLE, new)
        return True
    except Exception:  # noqa: BLE001 - cosmetic; never break the UI over it
        return False


def dark_title_bar(widget: QWidget) -> bool:
    """Windows 11: ask DWM for a dark caption bar (DWMWA_USE_IMMERSIVE_DARK_MODE=20)."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        if QGuiApplication.platformName() != "windows":
            return False
        value = ctypes.c_int(1)
        hr = ctypes.windll.dwmapi.DwmSetWindowAttribute(ctypes.c_void_p(int(widget.winId())), 20,
                                                         ctypes.byref(value), ctypes.sizeof(value))
        return hr == 0
    except Exception:  # noqa: BLE001
        return False


def bring_to_front(widget: QWidget) -> bool:
    """Win32: make a window the user asked for (the panel) the foreground window.

    Windows' foreground lock lets a process activate its window only in a few
    cases (it is the foreground process, it got the last input, ...). SAM's
    panel is opened by a launch, a second launch, the tray or the island while
    another app is in front, so ``activateWindow`` alone may only flash the
    taskbar button and leave the panel behind that app. Order: restore if
    minimised, SetForegroundWindow, and when Windows refuses, attach to the
    foreground thread's input for the call (only ever for a window the user
    asked to see). Returns True when the panel is the foreground window."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        if QGuiApplication.platformName() != "windows":
            return False
        user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
        user32.GetForegroundWindow.restype = wintypes.HWND
        user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.c_void_p]
        user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        own = int(widget.winId())
        hwnd = wintypes.HWND(own)
        SW_RESTORE, SW_SHOW = 9, 5
        user32.ShowWindow(hwnd, SW_RESTORE if user32.IsIconic(hwnd) else SW_SHOW)

        def front() -> bool:
            current = user32.GetForegroundWindow()
            return bool(current) and int(current) == own

        if front() or (user32.SetForegroundWindow(hwnd) and front()):
            return True
        foreground = user32.GetForegroundWindow()
        other = user32.GetWindowThreadProcessId(foreground, None) if foreground else 0
        mine = kernel32.GetCurrentThreadId()
        if other and other != mine and user32.AttachThreadInput(mine, other, True):
            try:
                user32.BringWindowToTop(hwnd)
                user32.SetForegroundWindow(hwnd)
            finally:
                user32.AttachThreadInput(mine, other, False)
        return front()
    except Exception:  # noqa: BLE001 - never break the UI over window order
        return False


def allow_any_foreground() -> bool:
    """Let another process (the running SAM) take the foreground once. Called
    by a second launch before it signals: Windows gave that launch foreground
    rights because the user started it (AllowSetForegroundWindow(ASFW_ANY))."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        return bool(ctypes.windll.user32.AllowSetForegroundWindow(-1))
    except Exception:  # noqa: BLE001
        return False


__all__ = ["apply_no_activate", "dark_title_bar", "bring_to_front", "allow_any_foreground"]
