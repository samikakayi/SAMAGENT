"""Per-monitor DPI awareness for SAM's screen-coordinate work.

A DPI-unaware Windows process is lied to: `GetWindowRect` returns virtualized
logical coordinates while `ImageGrab` captures real physical pixels. On a 175%
display those differ by a factor of 1.75, so a chart region captured from a
window rectangle would show entirely the wrong part of the screen, and a mouse
click computed from a screenshot would land somewhere else.

Declaring per-monitor awareness makes both APIs speak physical pixels, which is
what every calibration, verification, and pointer calculation assumes.
"""

from __future__ import annotations

import ctypes
import os
import threading
from typing import Any

_lock = threading.Lock()
_state: dict[str, Any] | None = None

# DPI_AWARENESS_CONTEXT values (winuser.h). The handles are negative sentinels.
_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)
_PER_MONITOR_AWARE = ctypes.c_void_p(-3)
_SYSTEM_AWARE = ctypes.c_void_p(-2)


def ensure_dpi_awareness() -> dict[str, Any]:
    """Make this process per-monitor DPI aware. Safe to call repeatedly.

    Awareness can only be set once per process and fails if the host already
    chose a mode, so a failure here is reported rather than raised.
    """
    global _state
    with _lock:
        if _state is not None:
            return _state
        if os.name != "nt":
            _state = {"applied": False, "mode": "not-windows", "scale": 1.0}
            return _state
        user32 = ctypes.windll.user32
        applied = False
        mode = "unchanged"
        for context, name in (
            (_PER_MONITOR_AWARE_V2, "per-monitor-v2"),
            (_PER_MONITOR_AWARE, "per-monitor"),
            (_SYSTEM_AWARE, "system"),
        ):
            try:
                if user32.SetProcessDpiAwarenessContext(context):
                    applied, mode = True, name
                    break
            except (AttributeError, OSError):
                continue
        if not applied:
            # Windows 8.1 fallback, and the path taken when awareness was
            # already set by the host process.
            try:
                applied = ctypes.windll.shcore.SetProcessDpiAwareness(2) == 0
                mode = "shcore-per-monitor" if applied else mode
            except (AttributeError, OSError):
                pass
        _state = {"applied": applied, "mode": mode, **measure_scale()}
        return _state


def measure_scale() -> dict[str, Any]:
    """Compare the reported virtual screen with a real capture to expose scaling."""
    if os.name != "nt":
        return {"scale": 1.0, "logical_width": None, "physical_width": None}
    try:
        user32 = ctypes.windll.user32
        logical_width = user32.GetSystemMetrics(78) or user32.GetSystemMetrics(0)
        from PIL import ImageGrab

        physical_width = ImageGrab.grab(all_screens=True).width
    except Exception:
        return {"scale": 1.0, "logical_width": None, "physical_width": None}
    scale = physical_width / logical_width if logical_width else 1.0
    return {
        "scale": round(scale, 4),
        "logical_width": logical_width,
        "physical_width": physical_width,
        "coordinates_consistent": abs(scale - 1.0) < 0.01,
    }


def status() -> dict[str, Any]:
    return ensure_dpi_awareness()
