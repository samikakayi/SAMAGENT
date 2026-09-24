"""Keyboard, mouse and clipboard (SendInput via ctypes).

Keyboard layout: the user may type with a Kurdish or Arabic layout. v1's
``desktop_input.py`` learned that ``VkKeyScan`` asks the *current* layout which
key makes a character, so with a Kurdish layout it returns -1 for Latin
letters and every shortcut fails. Shortcuts are therefore sent as fixed
virtual-key codes ('S' is 0x53 on every layout; punctuation uses the US
physical OEM keys), never as characters.

Text: Sorani/Unicode text is typed by clipboard paste (Ctrl+V), which works
in Win32, UWP and Electron apps alike; the user's previous clipboard (every
memory-backed format, not only text) is restored afterwards, and SAM's text
is marked so Windows keeps it out of clipboard history and cloud sync.
``KEYEVENTF_UNICODE`` typing is the fallback when the clipboard is locked.

Mouse coordinates are physical pixels: callers run in a per-monitor-v2 DPI
thread (``_win.dpi_aware``), so SetCursorPos matches captures and UIA rects.
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes
from typing import Any, Callable, Protocol

from . import _win

# --- virtual keys --------------------------------------------------------------
VK = {
    "backspace": 0x08, "tab": 0x09, "enter": 0x0D, "return": 0x0D, "shift": 0x10, "ctrl": 0x11, "control": 0x11,
    "alt": 0x12, "menu": 0x12, "pause": 0x13, "capslock": 0x14, "esc": 0x1B, "escape": 0x1B, "space": 0x20,
    "pageup": 0x21, "pgup": 0x21, "pagedown": 0x22, "pgdn": 0x22, "end": 0x23, "home": 0x24, "left": 0x25,
    "up": 0x26, "right": 0x27, "down": 0x28, "printscreen": 0x2C, "prtsc": 0x2C, "insert": 0x2D, "ins": 0x2D,
    "delete": 0x2E, "del": 0x2E, "win": 0x5B, "windows": 0x5B, "lwin": 0x5B, "rwin": 0x5C, "apps": 0x5D,
    "contextmenu": 0x5D, "numlock": 0x90, "scrolllock": 0x91,
    "browser_back": 0xA6, "browser_forward": 0xA7, "browser_refresh": 0xA8, "browser_home": 0xAC,
    "volume_mute": 0xAD, "mute": 0xAD, "volume_down": 0xAE, "volume_up": 0xAF,
    "next_track": 0xB0, "next": 0xB0, "prev_track": 0xB1, "previous_track": 0xB1, "previous": 0xB1, "prev": 0xB1,
    "media_stop": 0xB2, "stop_media": 0xB2, "play_pause": 0xB3, "playpause": 0xB3, "play": 0xB3,
    # US physical punctuation keys (layout independent as shortcuts)
    ";": 0xBA, "semicolon": 0xBA, "=": 0xBB, "plus": 0xBB, "equals": 0xBB, ",": 0xBC, "comma": 0xBC,
    "-": 0xBD, "minus": 0xBD, ".": 0xBE, "period": 0xBE, "/": 0xBF, "slash": 0xBF, "`": 0xC0, "backtick": 0xC0,
    "[": 0xDB, "\\": 0xDC, "backslash": 0xDC, "]": 0xDD, "'": 0xDE, "quote": 0xDE,
}
for _n in range(1, 25):
    VK[f"f{_n}"] = 0x6F + _n
# Spoken Sorani key names the model may pass through.
VK.update({"ئینتەر": 0x0D, "ئێنتەر": 0x0D, "سپەیس": 0x20, "تاب": 0x09, "ئێسکەیپ": 0x1B, "دیلیت": 0x2E})
MODIFIERS = {0x10, 0x11, 0x12, 0x5B, 0x5C}
# Keys that need KEYEVENTF_EXTENDEDKEY (arrows, navigation block, right-hand
# modifiers, media keys); without it some apps see numpad keys instead.
EXTENDED = {0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2C, 0x2D, 0x2E, 0x5B, 0x5C, 0x5D, 0x6F, 0x90,
            0xA6, 0xA7, 0xA8, 0xAC, 0xAD, 0xAE, 0xAF, 0xB0, 0xB1, 0xB2, 0xB3}

KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP, KEYEVENTF_UNICODE = 0x0001, 0x0002, 0x0004
MOUSEEVENTF = {"left": (0x0002, 0x0004), "right": (0x0008, 0x0010), "middle": (0x0020, 0x0040)}
MOUSEEVENTF_WHEEL = 0x0800
INPUT_MOUSE, INPUT_KEYBOARD = 0, 1
ULONG_PTR = ctypes.c_size_t


def key_code(name: str) -> int:
    """Virtual-key code for one key name, independent of the keyboard layout."""
    key = name.strip().lower().replace(" ", "_")
    if key in VK:
        return VK[key]
    if len(key) == 1 and key.isascii() and (key.isalpha() or key.isdigit()):
        return ord(key.upper())  # 'A'..'Z' = 0x41.., '0'..'9' = 0x30.. on every layout
    compact = key.replace("_", "")
    if compact in VK:
        return VK[compact]
    raise ValueError(f"unknown key '{name}'")


def parse_keys(spec: str) -> list[list[int]]:
    """'ctrl+shift+esc, alt+tab' -> [[0x11, 0x10, 0x1B], [0x12, 0x09]].

    Chords are joined with '+'; several chords are separated by commas or
    spaces. A literal '+' key is written 'plus'."""
    chords: list[list[int]] = []
    text = spec.strip().replace("،", ",")
    for part in [p for chunk in text.split(",") for p in chunk.split()]:
        keys = [k for k in part.split("+") if k]
        if not keys:
            continue
        chords.append([key_code(k) for k in keys])
    if not chords:
        raise ValueError("no keys given")
    return chords


def describe_chord(chord: list[int]) -> str:
    names = {v: k for k, v in reversed(list(VK.items())) if k.isascii()}
    return "+".join(names.get(vk, chr(vk) if 0x30 <= vk <= 0x5A else hex(vk)) for vk in chord)


# --- SendInput structures -------------------------------------------------------
class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", _KEYBDINPUT), ("mi", _MOUSEINPUT), ("hi", _HARDWAREINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


class InputBackend(Protocol):
    def key(self, vk: int, up: bool) -> None: ...
    def unicode(self, unit: int, up: bool) -> None: ...
    def move(self, x: int, y: int) -> None: ...
    def button(self, button: str, up: bool) -> None: ...
    def wheel(self, delta: int) -> None: ...
    def cursor(self) -> tuple[int, int]: ...


class SendInputBackend:
    """Real input through user32.SendInput (call inside ``_win.dpi_aware``)."""

    def __init__(self) -> None:
        self.u = _win.user32()
        self.u.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(_INPUT), ctypes.c_int]

    def _send(self, item: _INPUT) -> None:
        if self.u.SendInput(1, ctypes.byref(item), ctypes.sizeof(_INPUT)) != 1:
            raise OSError("SendInput was blocked (a higher-privilege window may have focus)")

    def key(self, vk: int, up: bool) -> None:
        flags = (KEYEVENTF_KEYUP if up else 0) | (KEYEVENTF_EXTENDEDKEY if vk in EXTENDED else 0)
        scan = self.u.MapVirtualKeyW(vk, 0) & 0xFF
        self._send(_INPUT(INPUT_KEYBOARD, _INPUTUNION(ki=_KEYBDINPUT(vk, scan, flags, 0, 0))))

    def unicode(self, unit: int, up: bool) -> None:
        flags = KEYEVENTF_UNICODE | (KEYEVENTF_KEYUP if up else 0)
        self._send(_INPUT(INPUT_KEYBOARD, _INPUTUNION(ki=_KEYBDINPUT(0, unit, flags, 0, 0))))

    def move(self, x: int, y: int) -> None:
        self.u.SetCursorPos(int(x), int(y))

    def button(self, button: str, up: bool) -> None:
        down_flag, up_flag = MOUSEEVENTF[button]
        self._send(_INPUT(INPUT_MOUSE, _INPUTUNION(mi=_MOUSEINPUT(0, 0, 0, up_flag if up else down_flag, 0, 0))))

    def wheel(self, delta: int) -> None:
        self._send(_INPUT(INPUT_MOUSE, _INPUTUNION(mi=_MOUSEINPUT(0, 0, ctypes.c_uint32(delta).value,
                                                                  MOUSEEVENTF_WHEEL, 0, 0))))

    def cursor(self) -> tuple[int, int]:
        point = wintypes.POINT()
        self.u.GetCursorPos(ctypes.byref(point))
        return (point.x, point.y)


# --- clipboard ------------------------------------------------------------------
CF_UNICODETEXT = 13
# Handles that are not HGLOBAL memory (GDI objects, metafiles) or that Windows
# synthesises from another format; they are skipped when saving/restoring.
_NON_MEMORY_FORMATS = {2, 3, 9, 14, 0x80, 0x81, 0x82, 0x83, 0x8E}
_SYNTHESIZED = {1: CF_UNICODETEXT, 7: CF_UNICODETEXT, 16: CF_UNICODETEXT, 2: 8}
MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024


class Clipboard:
    """Clipboard read/write/snapshot via ctypes (thread-agnostic)."""

    def __init__(self) -> None:
        u = _win.user32()
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        u.GetClipboardData.restype = wintypes.HANDLE
        u.GetClipboardData.argtypes = [wintypes.UINT]
        u.SetClipboardData.restype = wintypes.HANDLE
        u.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
        u.EnumClipboardFormats.argtypes = [wintypes.UINT]
        u.RegisterClipboardFormatW.argtypes = [wintypes.LPCWSTR]
        k.GlobalAlloc.restype = wintypes.HGLOBAL
        k.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
        k.GlobalLock.restype = ctypes.c_void_p
        k.GlobalLock.argtypes = [wintypes.HGLOBAL]
        k.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
        k.GlobalSize.restype = ctypes.c_size_t
        k.GlobalSize.argtypes = [wintypes.HGLOBAL]
        k.GlobalFree.argtypes = [wintypes.HGLOBAL]
        self.u, self.k = u, k

    def _open(self, attempts: int = 20) -> None:
        for _ in range(attempts):
            if self.u.OpenClipboard(None):
                return
            time.sleep(0.03)  # another app holds the clipboard for a moment
        raise OSError("the clipboard is locked by another app")

    def _read(self, fmt: int) -> bytes | None:
        handle = self.u.GetClipboardData(fmt)
        if not handle:
            return None
        size = self.k.GlobalSize(handle)
        pointer = self.k.GlobalLock(handle)
        if not pointer:
            return None
        try:
            return ctypes.string_at(pointer, size)
        finally:
            self.k.GlobalUnlock(handle)

    def _write(self, fmt: int, data: bytes) -> None:
        handle = self.k.GlobalAlloc(0x0002, max(1, len(data)))  # GMEM_MOVEABLE
        if not handle:
            raise MemoryError("GlobalAlloc failed")
        pointer = self.k.GlobalLock(handle)
        ctypes.memmove(pointer, data, len(data))
        self.k.GlobalUnlock(handle)
        if not self.u.SetClipboardData(fmt, handle):
            self.k.GlobalFree(handle)
            raise OSError(f"SetClipboardData({fmt}) failed")

    def get_text(self) -> str | None:
        self._open()
        try:
            raw = self._read(CF_UNICODETEXT)
        finally:
            self.u.CloseClipboard()
        if raw is None:
            return None
        return raw.decode("utf-16-le", errors="replace").split("\x00", 1)[0]

    def set_text(self, text: str, *, private: bool = True) -> None:
        """Put ``text`` on the clipboard. ``private`` keeps it out of the
        Win+V history and cloud clipboard (documented clipboard formats)."""
        self._open()
        try:
            self.u.EmptyClipboard()
            self._write(CF_UNICODETEXT, (text + "\x00").encode("utf-16-le"))
            if private:
                exclude = self.u.RegisterClipboardFormatW("ExcludeClipboardContentFromMonitorProcessing")
                history = self.u.RegisterClipboardFormatW("CanIncludeInClipboardHistory")
                cloud = self.u.RegisterClipboardFormatW("CanUploadToCloudClipboard")
                self._write(exclude, b"\x00")
                self._write(history, (0).to_bytes(4, "little"))
                self._write(cloud, (0).to_bytes(4, "little"))
        finally:
            self.u.CloseClipboard()

    def snapshot(self) -> list[tuple[int, bytes]]:
        """Every memory-backed format currently on the clipboard."""
        self._open()
        saved: list[tuple[int, bytes]] = []
        total = 0
        try:
            formats: list[int] = []
            fmt = self.u.EnumClipboardFormats(0)
            while fmt:
                formats.append(fmt)
                fmt = self.u.EnumClipboardFormats(fmt)
            present = set(formats)
            for fmt in formats:
                if fmt in _NON_MEMORY_FORMATS or (fmt in _SYNTHESIZED and _SYNTHESIZED[fmt] in present):
                    continue
                data = self._read(fmt)
                if data is None:
                    continue
                total += len(data)
                if total > MAX_SNAPSHOT_BYTES:
                    break
                saved.append((fmt, data))
        finally:
            self.u.CloseClipboard()
        return saved

    def restore(self, saved: list[tuple[int, bytes]]) -> None:
        self._open()
        try:
            self.u.EmptyClipboard()
            for fmt, data in saved:
                try:
                    self._write(fmt, data)
                except (OSError, MemoryError):
                    continue
        finally:
            self.u.CloseClipboard()


# --- high-level input ----------------------------------------------------------
class Input:
    """Synchronous input helpers; call from a worker thread (``Hands.run_input``)."""

    def __init__(self, backend: InputBackend | None = None, clipboard: Any = None,
                 sleep: Callable[[float], None] = time.sleep, paste_settle_s: float = 0.35) -> None:
        self._backend = backend
        self._clipboard = clipboard
        self.sleep = sleep
        self.paste_settle_s = paste_settle_s

    @property
    def backend(self) -> InputBackend:
        if self._backend is None:
            self._backend = SendInputBackend()
        return self._backend

    @property
    def clipboard(self) -> Any:
        if self._clipboard is None:
            self._clipboard = Clipboard()
        return self._clipboard

    # keyboard
    def chord(self, keys: list[int]) -> None:
        pressed: list[int] = []
        try:
            for vk in keys:
                self.backend.key(vk, up=False)
                pressed.append(vk)
        finally:
            for vk in reversed(pressed):
                self.backend.key(vk, up=True)
        self.sleep(0.03)

    def press_keys(self, spec: str, repeat: int = 1) -> list[str]:
        chords = parse_keys(spec)
        for _ in range(max(1, min(int(repeat or 1), 50))):
            for chord in chords:
                self.chord(chord)
        return [describe_chord(c) for c in chords]

    def type_unicode(self, text: str) -> None:
        for char in text:
            if char == "\n":
                self.chord([0x0D])
                continue
            data = char.encode("utf-16-le")
            units = [int.from_bytes(data[i:i + 2], "little") for i in range(0, len(data), 2)]
            for unit in units:
                self.backend.unicode(unit, up=False)
            for unit in units:
                self.backend.unicode(unit, up=True)

    def paste_text(self, text: str) -> dict[str, Any]:
        """Type ``text`` by Ctrl+V and restore the previous clipboard.

        The target app reads the clipboard asynchronously while it handles
        the paste, so the old content is restored only after a short settle
        time (350 ms default; Electron apps were the slowest readers)."""
        clipboard = self.clipboard
        try:
            saved = clipboard.snapshot()
        except OSError:
            self.type_unicode(text)
            return {"method": "unicode", "restored": None}
        clipboard.set_text(text)
        restored = False
        try:
            self.chord([0x11, 0x56])  # Ctrl+V by virtual key: layout independent
            self.sleep(self.paste_settle_s)
        finally:
            try:
                clipboard.restore(saved)
                restored = True
            except OSError:
                restored = False
        return {"method": "paste", "restored": restored, "saved_formats": len(saved)}

    def type_text(self, text: str, *, press_enter: bool = False) -> dict[str, Any]:
        result = self.paste_text(text) if text else {"method": "none"}
        if press_enter:
            self.sleep(0.05)
            self.chord([0x0D])
        return result

    # mouse
    def move(self, x: int, y: int) -> None:
        """Approach from 3 px away: a SetCursorPos to the position the pointer
        already has produces no move event (v1 desktop_input lesson)."""
        self.backend.move(int(x), int(y) - 3)
        self.sleep(0.03)
        self.backend.move(int(x), int(y))
        self.sleep(0.05)

    def click(self, x: int, y: int, *, button: str = "left", double: bool = False) -> None:
        self.move(x, y)
        for _ in range(2 if double else 1):
            self.backend.button(button, up=False)
            self.backend.button(button, up=True)
            self.sleep(0.06)

    def scroll(self, x: int, y: int, clicks: int) -> None:
        self.move(x, y)
        self.backend.wheel(int(clicks) * 120)

    def cursor(self) -> tuple[int, int]:
        return self.backend.cursor()


__all__ = ["Clipboard", "Input", "InputBackend", "SendInputBackend", "VK", "describe_chord", "key_code", "parse_keys"]
