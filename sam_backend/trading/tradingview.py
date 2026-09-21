from __future__ import annotations

import base64
import hashlib
import io
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..contracts import CapabilityState, ExecutionStatus, StandardResult
from .types import normalize_timeframe, pick_visible_interval


@dataclass(slots=True)
class TradingViewState:
    running: bool
    process_ids: list[int]
    window_handle: int | None
    title: str | None
    symbol: str | None
    feed: str | None
    timeframe: str | None
    timeframe_verified: bool
    current_price: float | None
    window_geometry: dict[str, int] | None
    monitor: dict[str, Any] | None
    active: bool
    interactive: bool
    client_geometry: dict[str, int] | None = None
    chart_type: str | None = None
    visible_price_range: dict[str, float] | None = None
    visible_time_range: dict[str, str] | None = None
    chart_geometry: dict[str, int] | None = None
    price_scale_geometry: dict[str, int] | None = None
    time_scale_geometry: dict[str, int] | None = None
    visible_indicators: list[str] = field(default_factory=list)
    layout: str | None = None
    observations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class TradingViewController:
    """Semantic TradingView Desktop observer with conservative verified controls."""

    def __init__(self, data_dir: Path, *, computer_control: bool = False, screen_access: bool = False) -> None:
        self.data_dir = data_dir
        self.computer_control = computer_control
        self.screen_access = screen_access
        self._lock = threading.RLock()
        self._last_verified_timeframe: dict[int, str] = {}
        self._calibration: dict[int, dict[str, float]] = {}

    @staticmethod
    def _modules():
        if os.name != "nt":
            raise RuntimeError("TradingView Desktop control is available only on Windows")
        try:
            import psutil
            import win32api
            import win32con
            import win32gui
            import win32process
        except ImportError as exc:
            raise RuntimeError("TradingView control requires psutil and pywin32") from exc
        return psutil, win32api, win32con, win32gui, win32process

    @staticmethod
    def _parse_title(title: str) -> dict[str, Any]:
        match = re.match(r"^\s*([A-Z0-9][A-Z0-9._:/-]{1,31})\s+[▼▲]?\s*([0-9][0-9,]*(?:\.[0-9]+)?)?", title, re.I)
        symbol = match.group(1).upper() if match else None
        price = float(match.group(2).replace(",", "")) if match and match.group(2) else None
        feed = None
        # Some layouts include "Symbol · Exchange" in the native title; do not infer a feed otherwise.
        feed_match = re.search(r"(?:·|:)\s*([A-Za-z][A-Za-z0-9 _.-]{2,40})(?:\s*[-/]|$)", title)
        if feed_match:
            feed = feed_match.group(1).strip()
        return {"symbol": symbol, "price": price, "feed": feed}

    def _windows(self) -> tuple[list[int], list[dict[str, Any]]]:
        psutil, _, _, win32gui, win32process = self._modules()
        process_ids: list[int] = []
        for process in psutil.process_iter(["pid", "name"]):
            try:
                if (process.info.get("name") or "").lower() == "tradingview.exe":
                    process_ids.append(int(process.info["pid"]))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        windows: list[dict[str, Any]] = []

        def callback(hwnd: int, _: Any) -> None:
            if not win32gui.IsWindowVisible(hwnd):
                return
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            if pid not in process_ids:
                return
            title = win32gui.GetWindowText(hwnd).strip()
            if title:
                windows.append({"hwnd": int(hwnd), "pid": int(pid), "title": title})

        win32gui.EnumWindows(callback, None)
        return sorted(set(process_ids)), windows

    @staticmethod
    def _client_geometry(hwnd: int, win32gui: Any) -> dict[str, int] | None:
        """Screen rectangle of the drawable client area.

        A maximized window's frame rectangle extends past the visible desktop by
        the invisible resize border, so capturing from it samples off-screen
        pixels. The client area is what the chart is actually painted into.
        """
        try:
            _, _, width, height = win32gui.GetClientRect(hwnd)
            left, top = win32gui.ClientToScreen(hwnd, (0, 0))
        except Exception:
            return None
        if width <= 0 or height <= 0:
            return None
        return {
            "left": int(left), "top": int(top),
            "right": int(left) + int(width), "bottom": int(top) + int(height),
            "width": int(width), "height": int(height),
        }

    @staticmethod
    def _monitor_for_window(hwnd: int, win32api: Any, win32con: Any) -> dict[str, Any] | None:
        try:
            monitor_handle = win32api.MonitorFromWindow(hwnd, win32con.MONITOR_DEFAULTTONEAREST)
            info = win32api.GetMonitorInfo(monitor_handle)
            return {
                "device": info.get("Device"),
                "monitor_rect": dict(zip(("left", "top", "right", "bottom"), info["Monitor"])),
                "work_rect": dict(zip(("left", "top", "right", "bottom"), info["Work"])),
                "primary": bool(info.get("Flags")),
            }
        except Exception:
            return None

    def observe(self) -> TradingViewState:
        try:
            _, win32api, win32con, win32gui, _ = self._modules()
            process_ids, windows = self._windows()
        except Exception as exc:
            return TradingViewState(False, [], None, None, None, None, None, False, None, None, None, False, False, observations=[str(exc)])
        if not windows:
            return TradingViewState(bool(process_ids), process_ids, None, None, None, None, None, False, None, None, None, False, False, observations=["TradingView has no visible targetable chart window."])
        foreground = int(win32gui.GetForegroundWindow())
        windows.sort(key=lambda item: (item["hwnd"] != foreground, -len(item["title"])))
        selected = windows[0]
        hwnd = selected["hwnd"]
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        client = self._client_geometry(hwnd, win32gui)
        parsed = self._parse_title(selected["title"])
        timeframe = self._last_verified_timeframe.get(hwnd)
        observations = []
        if len(windows) > 1:
            observations.append(f"{len(windows)} TradingView windows are open; the foreground/primary chart window was selected.")
        if parsed["feed"] is None:
            observations.append("The native window title does not expose a verified market feed; use market-data provider metadata or chart observation.")
        if timeframe is None:
            observations.append("The native title does not expose the selected timeframe; toolbar OCR verifies it after a timeframe command when Screen Access is on.")
        else:
            observations.append(f"Timeframe {timeframe} was verified from the TradingView toolbar.")
        return TradingViewState(
            running=True,
            process_ids=process_ids,
            window_handle=hwnd,
            title=selected["title"],
            symbol=parsed["symbol"],
            feed=parsed["feed"],
            timeframe=timeframe,
            timeframe_verified=timeframe is not None,
            current_price=parsed["price"],
            window_geometry={"left": left, "top": top, "right": right, "bottom": bottom, "width": right - left, "height": bottom - top},
            client_geometry=client,
            monitor=self._monitor_for_window(hwnd, win32api, win32con),
            active=hwnd == foreground,
            interactive=True,
            observations=observations,
        )

    @staticmethod
    def _raise_window(hwnd: int, win32con: Any, win32gui: Any, win32process: Any) -> None:
        """Bring a window forward despite Windows' foreground lock.

        `SetForegroundWindow` fails whenever the caller is not already the
        foreground process, reporting a misleading "invalid handle". Attaching to
        the current foreground thread's input queue is the documented way to make
        the request legitimate; the attachment is always undone.
        """
        import ctypes

        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        foreground = win32gui.GetForegroundWindow()
        if int(foreground) == int(hwnd):
            return
        current_thread = ctypes.windll.kernel32.GetCurrentThreadId()
        target_thread, _ = win32process.GetWindowThreadProcessId(hwnd)
        foreground_thread = win32process.GetWindowThreadProcessId(foreground)[0] if foreground else 0
        attached: list[int] = []
        try:
            for thread in {foreground_thread, target_thread} - {0, current_thread}:
                if ctypes.windll.user32.AttachThreadInput(current_thread, thread, True):
                    attached.append(thread)
            # Windows only honours a foreground change from a process that has
            # recently received input. A no-op ALT tap satisfies that rule; it is
            # released immediately so no menu is left open.
            ctypes.windll.user32.keybd_event(0x12, 0, 0, 0)
            ctypes.windll.user32.keybd_event(0x12, 0, 2, 0)
            for attempt in range(3):
                win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
                try:
                    win32gui.BringWindowToTop(hwnd)
                    win32gui.SetForegroundWindow(hwnd)
                except Exception:
                    # A refusal here is expected while the lock is still held;
                    # the loop retries and the caller verifies the real outcome.
                    pass
                time.sleep(0.12 * (attempt + 1))
                if int(win32gui.GetForegroundWindow()) == int(hwnd):
                    return
        finally:
            for thread in attached:
                ctypes.windll.user32.AttachThreadInput(current_thread, thread, False)

    def focus(self) -> StandardResult:
        started = time.perf_counter()
        if not self.computer_control:
            return StandardResult.failure(
                "Computer Control is off, so SAM cannot focus or drive TradingView. Nothing was changed. "
                "Turn Computer Control on in Settings to allow it.",
                error_code="COMPUTER_CONTROL_DISABLED", started_at=started,
            )
        state = self.observe()
        if not state.window_handle:
            return StandardResult.failure("TradingView window was not found", error_code="WINDOW_NOT_FOUND", started_at=started)
        _, _, win32con, win32gui, win32process = self._modules()
        try:
            self._raise_window(state.window_handle, win32con, win32gui, win32process)
            verified = int(win32gui.GetForegroundWindow()) == state.window_handle
            return StandardResult(
                ExecutionStatus.SUCCESS if verified else ExecutionStatus.PARTIAL,
                True,
                verified,
                data={"window_handle": state.window_handle, "title": state.title},
                error=None if verified else "Windows did not confirm TradingView as the foreground window.",
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        except Exception as exc:
            return StandardResult.failure(str(exc), executed=True, error_code="FOCUS_FAILED", started_at=started)

    def launch(self, timeout_seconds: float = 12.0) -> StandardResult:
        started = time.perf_counter()
        if not self.computer_control:
            return StandardResult.failure(
                "Computer Control is off, so SAM cannot focus or drive TradingView. Nothing was changed. "
                "Turn Computer Control on in Settings to allow it.",
                error_code="COMPUTER_CONTROL_DISABLED", started_at=started,
            )
        state = self.observe()
        if state.window_handle:
            return self.focus()
        try:
            os.startfile("shell:AppsFolder\\TradingView.Desktop_n534cwy3pjxzj!TradingView.Desktop")
            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                state = self.observe()
                if state.window_handle:
                    focused = self.focus()
                    focused.duration_ms = round((time.perf_counter() - started) * 1000, 2)
                    return focused
                time.sleep(0.25)
            return StandardResult.failure(
                "TradingView launch was requested but no visible window appeared before timeout.",
                executed=True,
                error_code="LAUNCH_NOT_VERIFIED",
                started_at=started,
            )
        except Exception as exc:
            return StandardResult.failure(str(exc), executed=True, error_code="LAUNCH_FAILED", started_at=started)

    @staticmethod
    def _send_unicode(character: str) -> None:
        """Type one character independently of the active keyboard layout.

        `VkKeyScan` can only express characters the current layout can produce,
        so with a Kurdish or Arabic layout selected it returns -1 for plain Latin
        letters and no symbol could be typed at all. Injecting the character as a
        Unicode key event sidesteps the layout entirely.
        """
        import ctypes
        from ctypes import wintypes

        KEYEVENTF_UNICODE = 0x0004
        KEYEVENTF_KEYUP = 0x0002
        INPUT_KEYBOARD = 1

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
            ]

        class INPUT(ctypes.Structure):
            class _UNION(ctypes.Union):
                _fields_ = [("ki", KEYBDINPUT)]

            _anonymous_ = ("union",)
            _fields_ = [("type", wintypes.DWORD), ("union", _UNION)]

        events = (INPUT * 2)()
        for index, flags in enumerate((KEYEVENTF_UNICODE, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP)):
            events[index].type = INPUT_KEYBOARD
            events[index].ki = KEYBDINPUT(0, ord(character), flags, 0, None)
        ctypes.windll.user32.SendInput(2, ctypes.byref(events), ctypes.sizeof(INPUT))

    @staticmethod
    def _clipboard_read() -> str | None:
        """The current clipboard text, or None if it holds something else."""
        import win32clipboard
        import win32con

        try:
            win32clipboard.OpenClipboard()
        except Exception:
            return None
        try:
            return win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
        except Exception:
            return None
        finally:
            try:
                win32clipboard.CloseClipboard()
            except Exception:
                pass

    @staticmethod
    def _clipboard_write(text: str) -> bool:
        import win32clipboard
        import win32con

        for _ in range(5):
            try:
                win32clipboard.OpenClipboard()
            except Exception:
                # Another process holds the clipboard; it is normally released fast.
                time.sleep(0.08)
                continue
            try:
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardData(win32con.CF_UNICODETEXT, text)
                return True
            except Exception:
                return False
            finally:
                try:
                    win32clipboard.CloseClipboard()
                except Exception:
                    pass
        return False

    def _paste_text(self, text: str) -> bool:
        """Replace a focused field's contents with `text` via the clipboard.

        TradingView is an Electron app, and Chromium ignores the synthetic
        `KEYEVENTF_UNICODE` events that work elsewhere on Windows: the search
        dialog stayed on the old symbol while every character was sent. A real
        Ctrl+V is honoured, so the text goes via the clipboard instead. The
        user's own clipboard is put back afterwards.
        """
        _, win32api, win32con, _, _ = self._modules()
        saved = self._clipboard_read()
        if not self._clipboard_write(text):
            return False
        try:
            for key in (0x41, 0x56):  # Ctrl+A then Ctrl+V
                win32api.keybd_event(win32con.VK_CONTROL, 0, 0, 0)
                win32api.keybd_event(key, 0, 0, 0)
                win32api.keybd_event(key, 0, win32con.KEYEVENTF_KEYUP, 0)
                win32api.keybd_event(win32con.VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)
                time.sleep(0.35)
        finally:
            if saved is not None:
                # Give the paste time to be read before the clipboard is restored.
                time.sleep(0.4)
                self._clipboard_write(saved)
        return True

    def _send_text_and_enter(self, text: str) -> None:
        _, win32api, win32con, _, _ = self._modules()
        for character in text:
            self._send_unicode(character)
            time.sleep(0.03)
        time.sleep(0.15)
        win32api.keybd_event(win32con.VK_RETURN, 0, 0, 0)
        win32api.keybd_event(win32con.VK_RETURN, 0, win32con.KEYEVENTF_KEYUP, 0)

    # The symbol button lives in the top toolbar strip; it is located by reading
    # the toolbar rather than assuming a fixed position for a given layout.
    TOOLBAR_TOP_INSET = 60
    TOOLBAR_HEIGHT = 90
    TOOLBAR_WIDTH = 700
    # The symbol slot is always the toolbar's left-most control; anything beyond
    # this offset is an interval or a tool and must never be clicked for a symbol.
    SYMBOL_SLOT_WIDTH = 400

    def _toolbar_band(self, state: Any) -> tuple[int, int, int, int] | None:
        geometry = state.client_geometry or state.window_geometry
        if not geometry:
            return None
        return (
            geometry["left"], geometry["top"] + self.TOOLBAR_TOP_INSET,
            min(geometry["right"], geometry["left"] + self.TOOLBAR_WIDTH),
            geometry["top"] + self.TOOLBAR_TOP_INSET + self.TOOLBAR_HEIGHT,
        )

    def _read_toolbar_words(self, state: Any) -> tuple[list[Any], dict[str, Any]]:
        from PIL import ImageGrab

        from ..dpi import ensure_dpi_awareness
        from .ocr import WindowsOcrEngine

        ensure_dpi_awareness()
        band = self._toolbar_band(state)
        if not band:
            return [], {"reason": "no chart geometry"}
        try:
            words = WindowsOcrEngine().recognize(ImageGrab.grab(bbox=band, all_screens=True))
        except Exception as exc:
            return [], {"reason": f"toolbar could not be read: {exc}"}
        return words, {
            "band": {"left": band[0], "top": band[1], "right": band[2], "bottom": band[3]},
            "tokens": [word.text for word in words if word.text.strip()][:20],
        }

    def _read_toolbar_timeframe(self, state: Any) -> tuple[str | None, dict[str, Any]]:
        """Read the current interval from the toolbar. Needs Screen Access."""
        if not self.screen_access:
            return None, {"reason": "Screen Access is off, so the interval cannot be verified independently."}
        words, meta = self._read_toolbar_words(state)
        if not words and meta.get("reason"):
            return None, meta
        picked = pick_visible_interval(
            [(word.text, word.center_x) for word in words],
            symbol=state.symbol,
        )
        meta["picked"] = picked
        return picked, meta

    def _open_symbol_search(self, state: Any) -> tuple[bool, dict[str, Any]]:
        """Click the toolbar symbol button and confirm the search dialog opened.

        Typing a symbol straight at the chart does not open the search on this
        build, and TradingView is an Electron app whose inner controls are not
        exposed to UI Automation, so the toolbar button is the reliable route.
        """
        import numpy
        from PIL import ImageGrab

        words, meta = self._read_toolbar_words(state)
        if not words:
            return False, {"reason": meta.get("reason") or "toolbar could not be read"}
        band_info = meta.get("band") or {}
        geometry = state.client_geometry or state.window_geometry
        if not geometry or not band_info:
            return False, {"reason": "no chart geometry"}
        band = (band_info["left"], band_info["top"], band_info["right"], band_info["bottom"])
        current = (state.symbol or "").upper()
        # Only the token that actually reads as the current symbol is clickable.
        # There is deliberately no positional fallback: the toolbar scrolls
        # horizontally, and clicking whatever happens to sit left-most once hit
        # the "4m" interval button and silently changed the user's timeframe.
        target = next(
            (word for word in words
             if word.text.strip().upper() == current and word.center_x <= self.SYMBOL_SLOT_WIDTH),
            None,
        )
        if target is None:
            return False, {
                "reason": f"the {current or 'symbol'} button was not visible in the toolbar "
                          f"(it scrolls horizontally); nothing was clicked",
                "toolbar_tokens": [word.text for word in words if word.text.strip()][:12],
            }

        x = int(band[0] + target.center_x)
        y = int(band[1] + target.center_y)
        full = (geometry["left"], geometry["top"], geometry["right"], geometry["bottom"])
        before = numpy.asarray(ImageGrab.grab(bbox=full, all_screens=True).convert("L"), dtype=numpy.int16)
        _, win32api, win32con, _, _ = self._modules()
        win32api.SetCursorPos((x, y - 3))
        time.sleep(0.1)
        win32api.SetCursorPos((x, y))
        time.sleep(0.25)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        time.sleep(1.4)
        after = numpy.asarray(ImageGrab.grab(bbox=full, all_screens=True).convert("L"), dtype=numpy.int16)
        if after.shape != before.shape:
            return False, {"reason": "the window changed size while opening the search"}
        difference = numpy.abs(after - before) > 18
        changed = float(difference.mean())
        # A modal dialog repaints a large share of the window; a missed click does not.
        opened = changed > 0.03
        detail: dict[str, Any] = {"clicked": {"x": x, "y": y}, "screen_change": round(changed, 4),
                                  "button_text": target.text}
        if opened:
            rows = numpy.flatnonzero(difference.any(axis=1))
            columns = numpy.flatnonzero(difference.any(axis=0))
            if rows.size and columns.size:
                detail["dialog"] = {
                    "left": geometry["left"] + int(columns[0]), "right": geometry["left"] + int(columns[-1]),
                    "top": geometry["top"] + int(rows[0]), "bottom": geometry["top"] + int(rows[-1]),
                }
        return opened, detail

    def set_symbol(self, symbol: str, timeout_seconds: float = 12.0, feed_hint: str | None = None) -> StandardResult:
        started = time.perf_counter()
        symbol = symbol.strip().upper()
        if not re.fullmatch(r"[A-Z0-9._:/-]{2,32}", symbol):
            return StandardResult.failure("Symbol contains unsupported characters", error_code="INVALID_SYMBOL", started_at=started)
        focused = self.focus()
        if focused.status != ExecutionStatus.SUCCESS:
            return focused
        state = self.observe()
        if (state.symbol or "").upper() == symbol:
            return StandardResult.success(state.as_dict(), verified=True, started_at=started,
                                          observations=[f"The chart already shows {symbol}."])
        opened, detail = self._open_symbol_search(state)
        if not opened:
            return StandardResult.failure(
                "The TradingView symbol search did not open, so the symbol was not changed. "
                f"({detail.get('reason', 'no dialog appeared')})",
                executed=True, error_code="SYMBOL_SEARCH_NOT_OPENED", started_at=started,
            )
        try:
            _, win32api, win32con, _, _ = self._modules()
            # The dialog opens with the search field focused and its current text
            # selected, so the paste replaces it. Do not click "into" the field
            # first: the modal dims the entire client area, so its measured
            # bounds are the whole window and such a click lands on the backdrop,
            # which dismisses the dialog.
            if not self._paste_text(symbol):
                return StandardResult.failure(
                    "The Windows clipboard could not be used to enter the symbol, so nothing was typed.",
                    executed=True, error_code="SYMBOL_CLIPBOARD_UNAVAILABLE", started_at=started,
                )
            # Let the result list settle before committing to the top match.
            time.sleep(1.4)
            win32api.keybd_event(win32con.VK_RETURN, 0, 0, 0)
            win32api.keybd_event(win32con.VK_RETURN, 0, win32con.KEYEVENTF_KEYUP, 0)
        except Exception as exc:
            return StandardResult.failure(str(exc), executed=True, error_code="SYMBOL_CONTROL_FAILED", started_at=started)

        deadline = time.monotonic() + timeout_seconds
        last_state = self.observe()
        while time.monotonic() < deadline:
            last_state = self.observe()
            shown = (last_state.symbol or "").upper()
            if shown and (shown == symbol or shown.endswith(symbol) or symbol in shown):
                # Any stored calibration belongs to the previous instrument.
                if last_state.window_handle:
                    self._last_verified_timeframe.pop(last_state.window_handle, None)
                return StandardResult.success(
                    {**last_state.as_dict(), "search": detail}, verified=True, started_at=started,
                    observations=[f"The window title confirms {shown}."],
                )
            time.sleep(0.3)
        # Leave no dialog open if the change did not take.
        try:
            _, win32api, win32con, _, _ = self._modules()
            win32api.keybd_event(win32con.VK_ESCAPE, 0, 0, 0)
            win32api.keybd_event(win32con.VK_ESCAPE, 0, win32con.KEYEVENTF_KEYUP, 0)
        except Exception:
            pass
        return StandardResult(
            ExecutionStatus.PARTIAL, True, False,
            data={**last_state.as_dict(), "search": detail},
            error=f"TradingView did not confirm {symbol} in its title before the timeout; the chart still shows "
                  f"{last_state.symbol}. Nothing was analysed against the wrong instrument.",
            error_code="SYMBOL_NOT_VERIFIED",
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    def verify_symbol(self, symbol: str) -> StandardResult:
        """Confirm the chart is showing the instrument the caller expects."""
        state = self.observe()
        shown = (state.symbol or "").upper()
        wanted = symbol.strip().upper()
        matches = bool(shown) and (shown == wanted or shown.endswith(wanted) or wanted in shown)
        if matches:
            return StandardResult.success({"symbol": shown, "expected": wanted}, verified=True)
        return StandardResult.failure(
            f"The chart is showing {shown or 'nothing'}, not {wanted}.",
            error_code="SYMBOL_MISMATCH",
        )

    def set_timeframe(self, timeframe: str, timeout_seconds: float = 8.0) -> StandardResult:
        started = time.perf_counter()
        normalized = normalize_timeframe(timeframe)
        shortcut = {
            "S1": "1S", "S5": "5S", "S15": "15S", "S30": "30S",
            "M1": "1", "M3": "3", "M5": "5", "M15": "15", "M30": "30", "M45": "45",
            "H1": "60", "H2": "120", "H4": "240", "D1": "1D", "W1": "1W", "MN1": "1M",
        }.get(normalized)
        if not shortcut:
            return StandardResult.failure(f"Unsupported TradingView interval: {timeframe}", error_code="INVALID_TIMEFRAME", started_at=started)
        focused = self.focus()
        if focused.status != ExecutionStatus.SUCCESS:
            return focused
        state = self.observe()
        if self.screen_access:
            current, preview = self._read_toolbar_timeframe(state)
            if current == normalized and state.window_handle:
                self._last_verified_timeframe[state.window_handle] = normalized
                state.timeframe = normalized
                state.timeframe_verified = True
                return StandardResult.success(
                    {**state.as_dict(), "toolbar": preview},
                    verified=True, started_at=started,
                    observations=[f"The toolbar already shows {normalized}."],
                )
        try:
            self._send_text_and_enter(shortcut)
        except Exception as exc:
            return StandardResult.failure(str(exc), executed=True, error_code="TIMEFRAME_CONTROL_FAILED", started_at=started)

        last_meta: dict[str, Any] = {}
        if self.screen_access:
            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                after = self.observe()
                picked, last_meta = self._read_toolbar_timeframe(after)
                if picked == normalized and after.window_handle:
                    self._last_verified_timeframe[after.window_handle] = normalized
                    after.timeframe = normalized
                    after.timeframe_verified = True
                    return StandardResult.success(
                        {**after.as_dict(), "toolbar": last_meta},
                        verified=True, started_at=started,
                        observations=[f"Toolbar OCR confirmed {normalized}."],
                    )
                time.sleep(0.4)
            try:
                _, win32api, win32con, _, _ = self._modules()
                win32api.keybd_event(win32con.VK_ESCAPE, 0, 0, 0)
                win32api.keybd_event(win32con.VK_ESCAPE, 0, win32con.KEYEVENTF_KEYUP, 0)
            except Exception:
                pass
            after = self.observe()
            if after.window_handle:
                self._last_verified_timeframe.pop(after.window_handle, None)
            after.timeframe = None
            after.timeframe_verified = False
            after.observations.append(
                "The interval command was sent, but toolbar OCR did not independently confirm the selected timeframe."
            )
            return StandardResult(
                ExecutionStatus.PARTIAL, True, False,
                data={**after.as_dict(), "toolbar": last_meta},
                error="Timeframe command executed but remains unverified.",
                error_code="TIMEFRAME_NOT_VERIFIED",
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )

        if state.window_handle:
            self._last_verified_timeframe.pop(state.window_handle, None)
        after = self.observe()
        after.timeframe = normalized
        after.timeframe_verified = False
        after.observations.append(
            "The interval command was sent, but Screen Access is off so the timeframe cannot be verified."
        )
        return StandardResult(
            ExecutionStatus.PARTIAL, True, False,
            data=after.as_dict(),
            error="Timeframe command executed but remains unverified.",
            error_code="TIMEFRAME_NOT_VERIFIED",
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    def capture(self) -> StandardResult:
        started = time.perf_counter()
        if not self.screen_access:
            return StandardResult.failure(
                "Screen Access is off, so SAM cannot capture the TradingView window. Nothing was captured. "
                "Turn Screen Access on in Settings to allow it.",
                error_code="SCREEN_ACCESS_DISABLED", started_at=started,
            )
        state = self.observe()
        if not state.window_handle or not state.window_geometry:
            return StandardResult.failure("TradingView window was not found", error_code="WINDOW_NOT_FOUND", started_at=started)
        try:
            from PIL import ImageGrab

            from ..dpi import ensure_dpi_awareness

            ensure_dpi_awareness()
            geometry = state.window_geometry
            image = ImageGrab.grab(bbox=(geometry["left"], geometry["top"], geometry["right"], geometry["bottom"]), all_screens=True)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            raw = buffer.getvalue()
            digest = hashlib.sha256(raw).hexdigest()
            screenshots = self.data_dir / "screenshots"
            screenshots.mkdir(parents=True, exist_ok=True)
            path = screenshots / f"tradingview-{int(time.time() * 1000)}-{digest[:12]}.png"
            path.write_bytes(raw)
            return StandardResult.success(
                {"path": str(path), "sha256": digest, "width": image.width, "height": image.height, "image_b64": base64.b64encode(raw).decode("ascii")},
                verified=image.width > 100 and image.height > 100,
                started_at=started,
            )
        except Exception as exc:
            return StandardResult.failure(str(exc), executed=True, error_code="SCREEN_CAPTURE_FAILED", started_at=started)

    def calibrate(self, price_a: float, y_a: float, price_b: float, y_b: float) -> StandardResult:
        started = time.perf_counter()
        state = self.observe()
        if not state.window_handle:
            return StandardResult.failure("TradingView window was not found", error_code="WINDOW_NOT_FOUND", started_at=started)
        if not all(map(lambda value: isinstance(value, (int, float)), (price_a, y_a, price_b, y_b))) or y_a == y_b or price_a == price_b:
            return StandardResult.failure("Two distinct price/Y anchors are required", error_code="INVALID_CALIBRATION", started_at=started)
        slope = (price_b - price_a) / (y_b - y_a)
        intercept = price_a - slope * y_a
        round_trip_a = slope * y_a + intercept
        round_trip_b = slope * y_b + intercept
        verified = abs(round_trip_a - price_a) < 1e-9 and abs(round_trip_b - price_b) < 1e-9
        self._calibration[state.window_handle] = {"slope": slope, "intercept": intercept, "created_at": time.time()}
        return StandardResult.success({"window_handle": state.window_handle, "slope": slope, "intercept": intercept}, verified=verified, started_at=started)

    def price_to_screen(self, price: float) -> StandardResult:
        state = self.observe()
        calibration = self._calibration.get(state.window_handle or -1)
        if calibration is None:
            return StandardResult.failure("Chart is not calibrated", error_code="CALIBRATION_REQUIRED")
        y = (price - calibration["intercept"]) / calibration["slope"]
        return StandardResult.success({"price": price, "y": y}, verified=math_is_finite(y))

    def drawing_capability(self) -> dict[str, Any]:
        state = self.observe()
        calibrated = bool(state.window_handle and state.window_handle in self._calibration)
        return {
            "state": CapabilityState.PARTIALLY_AVAILABLE.value if state.interactive else CapabilityState.UNAVAILABLE.value,
            "semantic_commands": ["horizontal_line", "trend_line", "vertical_line", "rectangle", "fibonacci", "undo", "redo", "save_layout", "go_to_date"],
            "calibrated": calibrated,
            "verified_price_drawing": calibrated and self.screen_access and self.computer_control,
            "reason": None if calibrated else "Price-specific drawing is blocked until chart calibration is verified.",
            "timeframe_ocr": bool(self.screen_access),
        }


def math_is_finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}
