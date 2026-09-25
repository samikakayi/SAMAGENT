"""Top-level windows: list, find by name, focus/minimize/maximize/restore/
close/snap -- every action verified afterwards (ported from v1
``windows_control.py``, which verified the same way).

Measured on this PC: EnumWindows + titles + process names for ~20 windows
takes 1.2-2.5 ms, so listing is done fresh for every command (no cache).

All Win32 calls go through ``WinApi`` (swappable in tests) and run in a
worker thread in per-monitor-v2 DPI mode, so rectangles are physical pixels
(this screen: 2880x1800 at 175 %).
"""

from __future__ import annotations

import asyncio
import ctypes
import os
import re
import time
from ctypes import wintypes
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from ..textnorm import normalize_ckb
from . import _win
from .aliases import match_alias

Rect = tuple[int, int, int, int]  # left, top, right, bottom (physical px)

SW_MINIMIZE, SW_MAXIMIZE, SW_RESTORE, SW_SHOWNOACTIVATE = 6, 3, 9, 4
WM_CLOSE = 0x0010
WS_EX_TOOLWINDOW, WS_EX_APPWINDOW, GWL_EXSTYLE = 0x00000080, 0x00040000, -20
DWMWA_CLOAKED, DWMWA_EXTENDED_FRAME_BOUNDS = 14, 9
# Shell windows SAM must never close or move.
PROTECTED_CLASSES = frozenset({"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd"})
# "this window", "ئەم پەنجەرەیە": the window the user is looking at.
CURRENT_WORDS = frozenset(normalize_ckb(w) for w in (
    "", "this", "this window", "current", "current window", "active", "active window", "foreground", "it",
    "ئەمە", "ئەم پەنجەرەیە", "ئەو پەنجەرەیە", "پەنجەرەکە", "ئەم پەنجەرە", "پەنجەرەی ئێستا", "ئەمەیان"))


MT5_PROCESSES = frozenset({"terminal64.exe", "terminal.exe", "metatrader.exe"})
_ACCOUNT_DIGITS = re.compile(r"\d{6,12}")


def scrub_title(process: str, title: str) -> str:
    """MetaTrader 5 titles carry the account number ("<9 digits> - Broker-Server:
    Demo Account - Hedge - Broker"). Screenshots already blank the title bar;
    the text went to cloud models through window lists, open_app/screen_look
    results and the activity log (review 2026-09-24, window_list_probe.py)."""
    if (process or "").lower() in MT5_PROCESSES:
        return _ACCOUNT_DIGITS.sub("######", title or "")
    return title


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    process: str          # exe basename, e.g. "chrome.exe"
    pid: int
    rect: Rect
    minimized: bool
    visible: bool
    maximized: bool = False
    cls: str = ""
    foreground: bool = False

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["rect"] = list(self.rect)
        return data

    def brief(self) -> dict[str, Any]:
        return {"hwnd": self.hwnd, "title": scrub_title(self.process, self.title)[:120], "app": self.process,
                "state": "minimized" if self.minimized else "maximized" if self.maximized else "normal",
                "active": self.foreground}


class WinApi:
    """Thin ctypes Win32 layer (sync). Replace with a fake in tests."""

    def __init__(self) -> None:
        self.u = _win.user32()
        self.u.GetForegroundWindow.restype = wintypes.HWND
        self.u.GetWindow.restype = wintypes.HWND
        self.u.MonitorFromWindow.restype = wintypes.HANDLE
        self.dwm = ctypes.WinDLL("dwmapi")

    def _title(self, hwnd: int) -> str:
        length = self.u.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return ""
        buffer = ctypes.create_unicode_buffer(length + 1)
        self.u.GetWindowTextW(hwnd, buffer, length + 1)
        return buffer.value

    def _class(self, hwnd: int) -> str:
        buffer = ctypes.create_unicode_buffer(256)
        self.u.GetClassNameW(hwnd, buffer, 256)
        return buffer.value

    def _cloaked(self, hwnd: int) -> bool:
        value = wintypes.DWORD(0)
        try:
            self.dwm.DwmGetWindowAttribute(wintypes.HWND(hwnd), DWMWA_CLOAKED, ctypes.byref(value), 4)
        except OSError:
            return False
        return bool(value.value)

    def rect(self, hwnd: int) -> Rect:
        r = wintypes.RECT()
        self.u.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(r))
        return (r.left, r.top, r.right, r.bottom)

    def frame_rect(self, hwnd: int) -> Rect:
        """Visible frame (without the invisible resize borders)."""
        r = wintypes.RECT()
        try:
            if self.dwm.DwmGetWindowAttribute(wintypes.HWND(hwnd), DWMWA_EXTENDED_FRAME_BOUNDS,
                                              ctypes.byref(r), ctypes.sizeof(r)) == 0:
                return (r.left, r.top, r.right, r.bottom)
        except OSError:
            pass
        return self.rect(hwnd)

    def enum(self, *, include_hidden: bool = False) -> list[WindowInfo]:
        foreground = int(self.u.GetForegroundWindow() or 0)
        found: list[WindowInfo] = []
        images: dict[int, str] = {}
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def callback(hwnd: Any, _: Any) -> bool:
            handle = int(hwnd or 0)
            visible = bool(self.u.IsWindowVisible(hwnd))
            if not visible and not include_hidden:
                return True
            title = self._title(handle)
            if not title:
                return True
            ex_style = self.u.GetWindowLongW(hwnd, GWL_EXSTYLE) & 0xFFFFFFFF
            owner = self.u.GetWindow(hwnd, 4)  # GW_OWNER
            if (ex_style & WS_EX_TOOLWINDOW) and not (ex_style & WS_EX_APPWINDOW):
                return True
            if owner and not (ex_style & WS_EX_APPWINDOW) and not include_hidden:
                return True
            if visible and self._cloaked(handle):
                return True
            pid = wintypes.DWORD()
            self.u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value not in images:
                images[pid.value] = os.path.basename(_win.process_image(pid.value) or "")
            found.append(WindowInfo(
                hwnd=handle, title=scrub_title(images[pid.value], title), process=images[pid.value],
                pid=int(pid.value), rect=self.rect(handle),
                minimized=bool(self.u.IsIconic(hwnd)), visible=visible, maximized=bool(self.u.IsZoomed(hwnd)),
                cls=self._class(handle), foreground=handle == foreground))
            return True

        self.u.EnumWindows(callback_type(callback), 0)
        return found

    def foreground(self) -> int:
        return int(self.u.GetForegroundWindow() or 0)

    def is_window(self, hwnd: int) -> bool:
        return bool(self.u.IsWindow(wintypes.HWND(hwnd)))

    def is_iconic(self, hwnd: int) -> bool:
        return bool(self.u.IsIconic(wintypes.HWND(hwnd)))

    def is_zoomed(self, hwnd: int) -> bool:
        return bool(self.u.IsZoomed(wintypes.HWND(hwnd)))

    def show(self, hwnd: int, command: int) -> None:
        self.u.ShowWindow(wintypes.HWND(hwnd), command)

    def set_foreground(self, hwnd: int) -> bool:
        """SetForegroundWindow with the usual work-arounds for the focus lock.

        A background process may not steal focus. Attaching to the input of
        the current foreground thread lifts the lock without injecting the
        Alt key (Alt would open the menu bar of classic apps)."""
        handle = wintypes.HWND(hwnd)
        if self.u.SetForegroundWindow(handle) and self.foreground() == hwnd:
            return True
        kernel32 = ctypes.windll.kernel32
        current = kernel32.GetCurrentThreadId()
        fg_thread = self.u.GetWindowThreadProcessId(self.u.GetForegroundWindow(), None)
        target_thread = self.u.GetWindowThreadProcessId(handle, None)
        attached = []
        for other in {fg_thread, target_thread}:
            if other and other != current and self.u.AttachThreadInput(current, other, True):
                attached.append(other)
        try:
            self.u.BringWindowToTop(handle)
            self.u.SetForegroundWindow(handle)
            self.u.SetFocus(handle)
        finally:
            for other in attached:
                self.u.AttachThreadInput(current, other, False)
        if self.foreground() != hwnd:
            try:
                self.u.SwitchToThisWindow(handle, True)
            except (AttributeError, OSError):
                pass
        return self.foreground() == hwnd

    def post_close(self, hwnd: int) -> None:
        self.u.PostMessageW(wintypes.HWND(hwnd), WM_CLOSE, 0, 0)

    def work_area(self, hwnd: int) -> Rect:
        class MONITORINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT),
                        ("dwFlags", wintypes.DWORD)]
        monitor = self.u.MonitorFromWindow(wintypes.HWND(hwnd), 2)  # MONITOR_DEFAULTTONEAREST
        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        self.u.GetMonitorInfoW(monitor, ctypes.byref(info))
        r = info.rcWork
        return (r.left, r.top, r.right, r.bottom)

    def move(self, hwnd: int, rect: Rect) -> None:
        left, top, right, bottom = rect
        self.u.SetWindowPos(wintypes.HWND(hwnd), None, left, top, right - left, bottom - top, 0x0004 | 0x0010)

    def owned_popups(self, hwnd: int) -> list[str]:
        """Titles of visible windows owned by ``hwnd`` (e.g. a save prompt)."""
        titles: list[str] = []
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def callback(child: Any, _: Any) -> bool:
            if int(self.u.GetWindow(child, 4) or 0) == hwnd and self.u.IsWindowVisible(child):
                titles.append(self._title(int(child)))
            return True

        self.u.EnumWindows(callback_type(callback), 0)
        return titles


def _score_window(query: str, window: WindowInfo, alias_processes: tuple[str, ...]) -> float:
    from rapidfuzz import fuzz

    title = normalize_ckb(window.title)
    process = window.process.lower().removesuffix(".exe")
    if alias_processes and window.process.lower() in alias_processes:
        return 100.0 if not window.minimized else 99.0
    if query == process:
        return 97.0
    if query and query in title:
        return 92.0
    return max(float(fuzz.partial_ratio(query, title)) if len(query) >= 4 else 0.0,
               float(fuzz.ratio(query, process)))


class Windows:
    """Async facade used by the tools (all Win32 work off the event loop)."""

    def __init__(self, api: Any = None, *, own_pid: int | None = None, settle_s: float = 1.2) -> None:
        self._api = api
        self.own_pid = own_pid if own_pid is not None else os.getpid()
        self.settle_s = settle_s

    @property
    def api(self) -> Any:
        if self._api is None:
            self._api = WinApi()
        return self._api

    async def _run(self, fn: Any, *args: Any) -> Any:
        def call() -> Any:
            with _win.dpi_aware():
                return fn(*args)
        return await asyncio.to_thread(call)

    # -- reading ---------------------------------------------------------------
    async def list(self, *, include_own: bool = False) -> list[WindowInfo]:
        windows = await self._run(self.api.enum)
        return [w for w in windows
                if (include_own or w.pid != self.own_pid) and w.cls not in ("Progman", "WorkerW")]

    async def foreground(self) -> WindowInfo | None:
        """The window the user is working in -- skipping SAM's own windows
        (when the user typed into SAM's panel, the panel is the foreground)."""
        windows = await self._run(self.api.enum)
        fg = next((w for w in windows if w.foreground), None)
        if fg is not None and fg.pid != self.own_pid and fg.cls not in PROTECTED_CLASSES:
            return fg
        # EnumWindows returns top-level windows in Z order: the first visible,
        # non-minimised window that is not SAM's is the one under SAM's panel.
        return next((w for w in windows if w.pid != self.own_pid and not w.minimized
                     and w.cls not in PROTECTED_CLASSES), None)

    def foreground_sync(self) -> WindowInfo | None:
        """Same as ``foreground`` but synchronous (~2 ms): for risk
        classifiers, which run on the event loop and must not await."""
        with _win.dpi_aware():
            windows = self.api.enum()
        fg = next((w for w in windows if w.foreground), None)
        if fg is not None and fg.pid != self.own_pid and fg.cls not in PROTECTED_CLASSES:
            return fg
        return next((w for w in windows if w.pid != self.own_pid and not w.minimized
                     and w.cls not in PROTECTED_CLASSES), None)

    async def find(self, query: str | int | None) -> WindowInfo | None:
        """Window by hwnd, title words, app name (English/Sorani) or 'this'."""
        if isinstance(query, int) or (isinstance(query, str) and query.strip().isdigit() and len(query.strip()) > 4):
            hwnd = int(query)
            return next((w for w in await self.list(include_own=True) if w.hwnd == hwnd), None)
        text = normalize_ckb(str(query or ""), strip_punct=True)
        if text in CURRENT_WORDS:
            return await self.foreground()
        alias = match_alias(str(query))
        processes = alias[0].processes if alias else ()
        candidates = await self.list()
        scored = sorted(((_score_window(text, w, processes), not w.minimized, w.foreground, w) for w in candidates),
                        key=lambda item: item[:3], reverse=True)
        if scored and scored[0][0] >= 72.0:
            return scored[0][3]
        return None

    def find_sync(self, query: str) -> WindowInfo | None:
        """``find`` for risk classifiers (synchronous, ~2 ms): the window a
        tool with ``window=query`` will act on (hwnd, 'this', alias, title)."""
        text = normalize_ckb(str(query or ""), strip_punct=True)
        with _win.dpi_aware():
            windows = self.api.enum()
        if str(query).strip().isdigit() and len(str(query).strip()) > 4:
            return next((w for w in windows if w.hwnd == int(str(query).strip())), None)
        if not text or text in CURRENT_WORDS:
            return self.foreground_sync()
        alias = match_alias(str(query))
        processes = alias[0].processes if alias else ()
        candidates = [w for w in windows if w.pid != self.own_pid and w.cls not in ("Progman", "WorkerW")]
        scored = sorted(((_score_window(text, w, processes), not w.minimized, w.foreground, w) for w in candidates),
                        key=lambda item: item[:3], reverse=True)
        return scored[0][3] if scored and scored[0][0] >= 72.0 else None

    async def find_by_process(self, processes: Iterable[str]) -> WindowInfo | None:
        """The best visible window of any of these exe names (foreground first)."""
        wanted = {p.lower() for p in processes}
        windows = await self._run(self.api.enum)
        matches = [w for w in windows if w.process.lower() in wanted and w.pid != self.own_pid]
        matches.sort(key=lambda w: (w.foreground, not w.minimized), reverse=True)
        return matches[0] if matches else None

    # -- actions (each verified) ---------------------------------------------------
    async def _wait(self, check: Any, timeout: float | None = None) -> bool:
        deadline = time.monotonic() + (self.settle_s if timeout is None else timeout)
        while True:
            if await self._run(check):
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(0.08)

    def _guard(self, window: WindowInfo) -> str | None:
        if window.pid == self.own_pid:
            return "That is SAM's own window."
        if window.cls in PROTECTED_CLASSES:
            return "That is part of the Windows shell (desktop/taskbar)."
        return None

    async def focus(self, target: WindowInfo | int) -> bool:
        hwnd = target.hwnd if isinstance(target, WindowInfo) else int(target)
        api = self.api

        def act() -> None:
            if api.is_iconic(hwnd):
                api.show(hwnd, SW_RESTORE)
            api.set_foreground(hwnd)
        await self._run(act)
        return await self._wait(lambda: api.foreground() == hwnd)

    async def act(self, action: str, window: WindowInfo) -> dict[str, Any]:
        """Run one window action with verification. Returns a result dict."""
        api = self.api
        hwnd = window.hwnd
        name = window.title[:80]
        guard = self._guard(window)
        if guard and action != "focus":
            return {"ok": False, "summary": f"Refused: {guard}"}
        if action == "focus":
            done = await self.focus(hwnd)
            return {"ok": done, "summary": f"Brought '{name}' to the front." if done else
                    f"Windows did not let SAM bring '{name}' to the front."}
        if action == "minimize":
            await self._run(api.show, hwnd, SW_MINIMIZE)
            done = await self._wait(lambda: api.is_iconic(hwnd))
            return {"ok": done, "summary": f"Minimized '{name}'." if done else f"'{name}' did not minimize."}
        if action == "maximize":
            await self._run(api.show, hwnd, SW_MAXIMIZE)
            done = await self._wait(lambda: api.is_zoomed(hwnd))
            if done:
                await self.focus(hwnd)
            return {"ok": done, "summary": f"Maximized '{name}'." if done else f"'{name}' did not maximize."}
        if action == "restore":
            await self._run(api.show, hwnd, SW_RESTORE)
            done = await self._wait(lambda: not api.is_iconic(hwnd) and not api.is_zoomed(hwnd))
            return {"ok": done, "summary": f"Restored '{name}'." if done else f"'{name}' was not restored."}
        if action == "close":
            await self._run(api.post_close, hwnd)
            done = await self._wait(lambda: not api.is_window(hwnd), timeout=3.0)
            if done:
                return {"ok": True, "summary": f"Closed '{name}'."}
            popups = await self._run(api.owned_popups, hwnd)
            return {"ok": False, "pending_dialog": popups[:3],
                    "summary": f"'{name}' is still open" + (f"; it is asking: {popups[0]!r} (maybe to save changes)."
                                                           if popups else " (it may be asking to save changes).")}
        if action in ("snap_left", "snap_right"):
            return await self._snap(window, left=action == "snap_left")
        return {"ok": False, "summary": f"Unknown window action '{action}'."}

    # -- contract shorthands (CONTRACTS 3.3): target = WindowInfo, hwnd or name ----------
    async def _act_on(self, action: str, target: WindowInfo | int | str | None) -> dict[str, Any]:
        window = target if isinstance(target, WindowInfo) else await self.find(target)
        if window is None:
            return {"ok": False, "summary": f"No open window matches '{target}'."}
        return await self.act(action, window)

    async def minimize(self, target: WindowInfo | int | str | None = None) -> dict[str, Any]:
        return await self._act_on("minimize", target)

    async def maximize(self, target: WindowInfo | int | str | None = None) -> dict[str, Any]:
        return await self._act_on("maximize", target)

    async def restore(self, target: WindowInfo | int | str | None = None) -> dict[str, Any]:
        return await self._act_on("restore", target)

    async def close(self, target: WindowInfo | int | str | None = None) -> dict[str, Any]:
        """Politely close (WM_CLOSE). Callers outside the tool must have asked
        the user first: the ``window_control`` tool's classifier does that."""
        return await self._act_on("close", target)

    async def snap(self, target: WindowInfo | int | str | None = None, *, side: str = "left") -> dict[str, Any]:
        return await self._act_on("snap_right" if side == "right" else "snap_left", target)

    async def _snap(self, window: WindowInfo, *, left: bool) -> dict[str, Any]:
        api = self.api
        hwnd = window.hwnd

        def act() -> tuple[Rect, Rect]:
            if api.is_iconic(hwnd) or api.is_zoomed(hwnd):
                api.show(hwnd, SW_RESTORE)
            work = api.work_area(hwnd)
            half = (work[2] - work[0]) // 2
            wanted = (work[0], work[1], work[0] + half, work[3]) if left else (work[0] + half, work[1], work[2], work[3])
            # Windows 10/11 windows have invisible resize borders: grow the
            # outer rect by them so the visible frame fills the half exactly.
            outer, frame = api.rect(hwnd), api.frame_rect(hwnd)
            border = (frame[0] - outer[0], frame[1] - outer[1], outer[2] - frame[2], outer[3] - frame[3])
            api.move(hwnd, (wanted[0] - border[0], wanted[1] - border[1], wanted[2] + border[2], wanted[3] + border[3]))
            return wanted, api.frame_rect(hwnd)

        wanted, got = await self._run(act)
        close = all(abs(a - b) <= 24 for a, b in zip(wanted, got))
        await self.focus(hwnd)
        side = "left" if left else "right"
        return {"ok": close, "rect": list(got),
                "summary": f"Moved '{window.title[:80]}' to the {side} half." if close else
                f"Tried to move '{window.title[:80]}' to the {side} half, but it kept a different size."}


__all__ = ["WinApi", "WindowInfo", "Windows", "Rect", "CURRENT_WORDS", "PROTECTED_CLASSES"]
