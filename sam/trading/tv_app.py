"""TradingView Desktop as a Windows app: find, start with the DevTools port,
close gracefully, focus. All methods are synchronous (call them through
``asyncio.to_thread``); the bridge talks to them through ``TvProcess`` so tests
can swap in a fake.

Measured on this PC (2026-09-24):
- Starting ``TradingView.exe`` from WindowsApps directly exits at once; the
  working start is ``IApplicationActivationManager::ActivateApplication`` with
  the AUMID and ``--remote-debugging-port=<port>`` (``sam.winapp.activate_aumid``).
- Two packages are installed: ``TradingView.Desktop`` (3.4.1.8194, the real app,
  AUMID ``TradingView.Desktop_n534cwy3pjxzj!TradingView.Desktop``) and the Store
  entry ``31178TradingViewInc.TradingView`` (3.4.1.0). ``Get-AppxPackage`` +
  manifest lookup takes ~0.8 s, so it runs only when the configured AUMID fails.
- TradingView runs ~10 ``TradingView.exe`` processes; only one owns a titled
  top-level window, and its title contains the account holder's name, so
  titles are never logged or returned.
"""

from __future__ import annotations

import ctypes
import logging
import os
import subprocess
import time
from ctypes import wintypes
from typing import Any

log = logging.getLogger("sam.trading.tv_app")

EXE_NAME = "tradingview.exe"
PACKAGE_NAMES = ("TradingView.Desktop", "31178TradingViewInc.TradingView")   # preference order
TH32CS_SNAPPROCESS = 0x00000002
PROCESS_TERMINATE = 0x0001
SYNCHRONIZE = 0x00100000
WM_CLOSE = 0x0010
SW_RESTORE = 9
GW_OWNER = 4
CREATE_NO_WINDOW = 0x08000000


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260)]


def _kernel32() -> Any:
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    k32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    return k32


def _user32() -> Any:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    hwnd = wintypes.HWND
    user32.GetWindow.restype = hwnd
    user32.GetWindow.argtypes = [hwnd, wintypes.UINT]
    user32.GetWindowThreadProcessId.argtypes = [hwnd, ctypes.POINTER(wintypes.DWORD)]
    user32.IsWindowVisible.argtypes = [hwnd]
    user32.GetWindowTextLengthW.argtypes = [hwnd]
    user32.GetWindowRect.argtypes = [hwnd, ctypes.POINTER(wintypes.RECT)]
    user32.PostMessageW.argtypes = [hwnd, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.IsIconic.argtypes = [hwnd]
    user32.ShowWindow.argtypes = [hwnd, ctypes.c_int]
    user32.SetForegroundWindow.argtypes = [hwnd]
    user32.GetForegroundWindow.restype = hwnd
    return user32


def running_pids(exe: str = EXE_NAME) -> list[int]:
    """PIDs of every process named ``exe`` (Toolhelp snapshot, ~ms, no subprocess)."""
    if os.name != "nt":
        return []
    k32 = _kernel32()
    snapshot = k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if not snapshot or snapshot == wintypes.HANDLE(-1).value:
        return []
    pids: list[int] = []
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        ok = k32.Process32FirstW(snapshot, ctypes.byref(entry))
        while ok:
            if entry.szExeFile.lower() == exe:
                pids.append(int(entry.th32ProcessID))
            ok = k32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        k32.CloseHandle(snapshot)
    return pids


def main_windows(pids: list[int]) -> list[int]:
    """Visible, titled, unowned top-level windows of ``pids`` (largest first)."""
    if os.name != "nt" or not pids:
        return []
    user32 = _user32()
    wanted = set(pids)
    found: list[tuple[int, int]] = []
    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def callback(hwnd: int, _lparam: int) -> bool:
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value in wanted and user32.IsWindowVisible(hwnd) and user32.GetWindowTextLengthW(hwnd) > 0 \
                and not user32.GetWindow(hwnd, GW_OWNER):
            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            found.append(((rect.right - rect.left) * (rect.bottom - rect.top), int(hwnd or 0)))
        return True

    user32.EnumWindows(enum_proc(callback), 0)
    return [hwnd for _, hwnd in sorted(found, reverse=True)]


def close_gracefully(pids: list[int], *, timeout_s: float = 10.0) -> dict[str, Any]:
    """WM_CLOSE to TradingView's windows, wait for every process to exit; only
    if it is still running after ``timeout_s`` terminate the rest (the user has
    already approved the restart). Returns {"graceful", "terminated"}."""
    if os.name != "nt":
        return {"graceful": False, "terminated": 0}
    user32 = _user32()
    for hwnd in main_windows(pids):
        user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not running_pids():
            return {"graceful": True, "terminated": 0}
        time.sleep(0.25)
    k32 = _kernel32()
    terminated = 0
    for pid in running_pids():
        handle = k32.OpenProcess(PROCESS_TERMINATE | SYNCHRONIZE, False, pid)
        if handle:
            try:
                if k32.TerminateProcess(handle, 0):
                    terminated += 1
            finally:
                k32.CloseHandle(handle)
    for _ in range(40):
        if not running_pids():
            break
        time.sleep(0.25)
    return {"graceful": False, "terminated": terminated}


def focus(pids: list[int]) -> bool:
    """Restore + bring TradingView's main window to the front."""
    if os.name != "nt":
        return False
    windows = main_windows(pids)
    if not windows:
        return False
    hwnd = windows[0]
    user32 = _user32()
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)
    if user32.SetForegroundWindow(hwnd):
        return True
    # Foreground lock (SAM is usually not the foreground app when it acts on a
    # voice command): SwitchToThisWindow is the documented-for-this-purpose
    # fallback that does not inject keystrokes.
    try:
        user32.SwitchToThisWindow.argtypes = [wintypes.HWND, wintypes.BOOL]
        user32.SwitchToThisWindow(hwnd, True)
    except AttributeError:
        return False
    return int(user32.GetForegroundWindow() or 0) == hwnd


def discover_aumids(timeout_s: float = 20.0) -> list[str]:
    """AUMIDs of installed TradingView packages (TradingView.Desktop first)."""
    if os.name != "nt":
        return []
    script = ("Get-AppxPackage *TradingView* | ForEach-Object { $m = Get-AppxPackageManifest $_; "
              "$_.Name + '|' + $_.PackageFamilyName + '!' + @($m.Package.Applications.Application)[0].Id }")
    try:
        output = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                                capture_output=True, text=True, timeout=timeout_s, creationflags=CREATE_NO_WINDOW,
                                check=False).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("TradingView package lookup failed: %s", type(exc).__name__)
        return []
    found: dict[str, str] = {}
    for line in output.splitlines():
        name, _, aumid = line.strip().partition("|")
        if "!" in aumid:
            found[name] = aumid
    ordered = [found[n] for n in PACKAGE_NAMES if n in found]
    return ordered + [a for n, a in found.items() if n not in PACKAGE_NAMES]


class TvProcess:
    """The OS side of TradingView, injectable for tests."""

    def running_pids(self) -> list[int]:
        return running_pids()

    def activate(self, aumid: str, arguments: str) -> int:
        from ..winapp import activate_aumid

        return activate_aumid(aumid, arguments)

    def discover_aumids(self) -> list[str]:
        return discover_aumids()

    def close(self, pids: list[int], timeout_s: float = 10.0) -> dict[str, Any]:
        return close_gracefully(pids, timeout_s=timeout_s)

    def focus(self, pids: list[int]) -> bool:
        return focus(pids)


__all__ = ["TvProcess", "running_pids", "main_windows", "close_gracefully", "focus", "discover_aumids",
           "PACKAGE_NAMES"]
