"""Global push-to-talk hotkey via Win32 ``RegisterHotKey`` on its own thread.

Why not a low-level keyboard hook: a hook sees every key the user types
(privacy) and Windows removes slow hooks; ``RegisterHotKey`` only delivers the
one chord and works under any keyboard layout because it uses virtual keys
(the user types with a Kurdish/Arabic layout). The research recommends a
push-to-talk key plus a conversation window instead of an always-on wake word
(v1's Whisper wake check cost 2.6-3.3 s of CPU per sound, reports/audit-latency.json).

WM_HOTKEY is posted to the queue of the thread that registered the key, so
registration, the message loop and unregistration all run on one dedicated
thread; ``stop()`` posts WM_QUIT to it.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import threading
from ctypes import wintypes
from typing import Callable

log = logging.getLogger("sam.voice.hotkey")

MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN, MOD_NOREPEAT = 0x0001, 0x0002, 0x0004, 0x0008, 0x4000
WM_HOTKEY, WM_QUIT = 0x0312, 0x0012
PM_NOREMOVE = 0x0000

_MODS = {"ctrl": MOD_CONTROL, "control": MOD_CONTROL, "alt": MOD_ALT, "shift": MOD_SHIFT,
         "win": MOD_WIN, "super": MOD_WIN, "meta": MOD_WIN}
_KEYS = {"space": 0x20, "enter": 0x0D, "return": 0x0D, "tab": 0x09, "esc": 0x1B, "escape": 0x1B,
         "backspace": 0x08, "pause": 0x13, "insert": 0x2D, "delete": 0x2E, "home": 0x24, "end": 0x23,
         "pageup": 0x21, "pagedown": 0x22, "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
         "capslock": 0x14, "scrolllock": 0x91, "printscreen": 0x2C, "`": 0xC0, "backquote": 0xC0,
         "-": 0xBD, "=": 0xBB, "[": 0xDB, "]": 0xDD, ";": 0xBA, "'": 0xDE, ",": 0xBC, ".": 0xBE, "/": 0xBF,
         "\\": 0xDC}
_KEYS.update({f"f{n}": 0x6F + n for n in range(1, 25)})
_KEYS.update({f"num{n}": 0x60 + n for n in range(10)})


def parse_hotkey(text: str) -> tuple[int, int]:
    """"ctrl+alt+space" -> (MOD_CONTROL|MOD_ALT, VK_SPACE). Raises ValueError."""
    parts = [p.strip().lower() for p in (text or "").split("+") if p.strip()]
    if not parts:
        raise ValueError("empty hotkey")
    mods, key = 0, None
    for part in parts:
        if part in _MODS:
            mods |= _MODS[part]
        elif key is None:
            if part in _KEYS:
                key = _KEYS[part]
            elif len(part) == 1 and part.isalnum() and part.isascii():
                key = ord(part.upper())
            else:
                raise ValueError(f"unknown key {part!r}")
        else:
            raise ValueError("only one non-modifier key is allowed")
    if key is None:
        raise ValueError("hotkey needs a key besides modifiers")
    if mods == 0:
        raise ValueError("hotkey needs at least one modifier (ctrl/alt/shift/win)")
    return mods, key


class GlobalHotkey:
    """Calls ``callback()`` (on the hotkey thread) each time the chord is pressed.

    The callback must be quick and thread-safe -- the voice engine passes a
    function that schedules work on the core loop with call_soon_threadsafe."""

    _ids = 0x5A00

    def __init__(self, hotkey: str, callback: Callable[[], None]) -> None:
        self.hotkey = hotkey
        self.callback = callback
        self.registered = False
        self.error: str | None = None
        self.presses = 0
        self._thread: threading.Thread | None = None
        self._thread_id: int | None = None
        self._ready = threading.Event()
        GlobalHotkey._ids += 1
        self._id = GlobalHotkey._ids

    def start(self, timeout: float = 2.0) -> bool:
        """Register on a new thread; True when the chord is ours."""
        if sys.platform != "win32":
            self.error = "not windows"
            return False
        try:
            self._mods, self._vk = parse_hotkey(self.hotkey)
        except ValueError as exc:
            self.error = str(exc)
            return False
        self._ready.clear()
        self._thread = threading.Thread(target=self._run, name="sam-hotkey", daemon=True)
        self._thread.start()
        self._ready.wait(timeout)
        return self.registered

    def _run(self) -> None:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
        user32.RegisterHotKey.restype = wintypes.BOOL
        user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
        user32.GetMessageW.restype = ctypes.c_int
        user32.PeekMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT,
                                        wintypes.UINT, wintypes.UINT]
        msg = wintypes.MSG()
        # Create this thread's message queue before anyone posts WM_QUIT to it.
        user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_NOREMOVE)
        self._thread_id = kernel32.GetCurrentThreadId()
        if not user32.RegisterHotKey(None, self._id, self._mods | MOD_NOREPEAT, self._vk):
            code = ctypes.get_last_error()
            # 1409 = ERROR_HOTKEY_ALREADY_REGISTERED (another app owns the chord)
            self.error = "already in use by another program" if code == 1409 else f"RegisterHotKey failed ({code})"
            self._ready.set()
            return
        self.registered = True
        self._ready.set()
        try:
            while True:
                result = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if result <= 0:  # 0 = WM_QUIT, -1 = error
                    break
                if msg.message == WM_HOTKEY and msg.wParam == self._id:
                    self.presses += 1
                    try:
                        self.callback()
                    except Exception:  # noqa: BLE001 - never kill the hotkey thread
                        log.exception("hotkey callback failed")
        finally:
            user32.UnregisterHotKey(None, self._id)
            self.registered = False

    def stop(self, timeout: float = 2.0) -> None:
        thread, tid = self._thread, self._thread_id
        if thread is None:
            return
        if tid is not None and thread.is_alive():
            ctypes.windll.user32.PostThreadMessageW(tid, WM_QUIT, 0, 0)
        thread.join(timeout)
        self._thread = None
        self._thread_id = None

    def simulate_press(self) -> bool:
        """Post a WM_HOTKEY to our own thread (tests; no key press needed)."""
        if self._thread_id is None:
            return False
        return bool(ctypes.windll.user32.PostThreadMessageW(self._thread_id, WM_HOTKEY, self._id, 0))


__all__ = ["GlobalHotkey", "parse_hotkey"]
