"""Small Win32 helpers shared by the hands modules (ctypes only, no Qt).

Why per-thread DPI awareness instead of a process-wide switch:
this laptop runs at 175 % (GetDpiForSystem = 168; a DPI-unaware thread sees a
1646x1029 screen, a per-monitor-v2 thread sees the real 2880x1800 -- measured
2026-09-24). Every coordinate SAM reads (window rects, UIA rectangles, mss
captures) and writes (SetCursorPos) must be in the same physical pixels, so
every worker thread that touches coordinates runs inside ``dpi_aware()``.
The process-wide mode is left to Qt (it sets per-monitor-v2 when the UI
starts); importing ``uiautomation`` or creating an ``mss`` instance would set a
process mode as a side effect, so hands only does either on worker threads
after start-up.

``Worker`` is a one-thread executor: COM objects (UI Automation, Shell) and
WinRT OCR must stay on the thread (apartment) that created them.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import ctypes
import functools
import logging
import os
import threading
from ctypes import wintypes
from typing import Any, Callable, Iterator, TypeVar

log = logging.getLogger("sam.hands")

IS_WINDOWS = os.name == "nt"
T = TypeVar("T")

# DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 (winuser.h sentinel handle).
_PER_MONITOR_V2 = -4
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

# Known-folder GUIDs used in Start-menu AppIDs ("{6D809377-...}\\MetaTrader 5\\terminal64.exe")
# and for the user's folders (Desktop may be redirected to OneDrive, so never
# assume %USERPROFILE%\Desktop).
KNOWN_FOLDERS = {
    "desktop": "{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}",
    "documents": "{FDD39AD0-238F-46AF-ADB4-6C85480369C7}",
    "downloads": "{374DE290-123F-4565-9164-39C4925E467B}",
    "pictures": "{33E28130-4E1E-4676-835A-98395C3BC3BB}",
    "music": "{4BD8D571-6D19-48D3-BE97-422220080E43}",
    "videos": "{18989B1D-99B5-455B-841C-AB7C74E4DDFC}",
}


def user32() -> Any:
    lib = ctypes.WinDLL("user32", use_last_error=True)
    lib.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p
    lib.SetThreadDpiAwarenessContext.argtypes = [ctypes.c_void_p]
    return lib


@contextlib.contextmanager
def dpi_aware() -> Iterator[None]:
    """Run the block with this thread in per-monitor-v2 DPI mode (physical px)."""
    if not IS_WINDOWS:
        yield
        return
    lib = user32()
    try:
        previous = lib.SetThreadDpiAwarenessContext(ctypes.c_void_p(_PER_MONITOR_V2))
    except (AttributeError, OSError):
        previous = None
    try:
        yield
    finally:
        if previous:
            try:
                lib.SetThreadDpiAwarenessContext(ctypes.c_void_p(previous))
            except (AttributeError, OSError):
                pass


def set_thread_dpi_aware() -> None:
    """Permanently switch the current (worker) thread to physical pixels."""
    if IS_WINDOWS:
        try:
            user32().SetThreadDpiAwarenessContext(ctypes.c_void_p(_PER_MONITOR_V2))
        except (AttributeError, OSError):
            pass


def process_image(pid: int) -> str | None:
    """Full path of the process image, or None (no psutil needed)."""
    if not IS_WINDOWS or pid <= 0:
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        return None
    try:
        size = wintypes.DWORD(1024)
        buffer = ctypes.create_unicode_buffer(size.value)
        kernel32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                                        ctypes.POINTER(wintypes.DWORD)]
        if kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return buffer.value
        return None
    finally:
        kernel32.CloseHandle(handle)


def pid_alive(pid: int) -> bool:
    if not IS_WINDOWS or pid <= 0:
        return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        return code.value == 259  # STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


class _PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_wchar * 260)]


def running_processes() -> dict[int, str]:
    """{pid: exe basename} of every process (Toolhelp snapshot, a few ms)."""
    if not IS_WINDOWS:
        return {}
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)  # TH32CS_SNAPPROCESS
    if not snapshot or snapshot == wintypes.HANDLE(-1).value:
        return {}
    found: dict[int, str] = {}
    try:
        entry = _PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(_PROCESSENTRY32W)
        ok = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while ok:
            found[int(entry.th32ProcessID)] = entry.szExeFile
            ok = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return found


class _GUID(ctypes.Structure):
    _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                ("Data4", ctypes.c_ubyte * 8)]


def _guid(text: str) -> _GUID:
    value = text.strip("{}")
    parts = value.split("-")
    tail = bytes.fromhex(parts[3] + parts[4])
    return _GUID(int(parts[0], 16), int(parts[1], 16), int(parts[2], 16), (ctypes.c_ubyte * 8)(*tail))


@functools.lru_cache(maxsize=64)
def known_folder(guid: str) -> str | None:
    """SHGetKnownFolderPath for a GUID string; None if unknown."""
    if not IS_WINDOWS:
        return None
    try:
        shell32 = ctypes.WinDLL("shell32")
        ole32 = ctypes.WinDLL("ole32")
        path = ctypes.c_wchar_p()
        folder_id = _guid(guid)
        shell32.SHGetKnownFolderPath.argtypes = [ctypes.POINTER(_GUID), wintypes.DWORD, wintypes.HANDLE,
                                                 ctypes.POINTER(ctypes.c_wchar_p)]
        if shell32.SHGetKnownFolderPath(ctypes.byref(folder_id), 0, None, ctypes.byref(path)) != 0:
            return None
        try:
            return path.value
        finally:
            ole32.CoTaskMemFree(path)
    except (OSError, ValueError, IndexError):
        return None


def expand_known_folder_path(path: str) -> str:
    """'{6D809377-...}\\MetaTrader 5\\terminal64.exe' -> 'C:\\Program Files\\MetaTrader 5\\terminal64.exe'."""
    if path.startswith("{") and "}" in path:
        guid, _, rest = path.partition("}")
        base = known_folder(guid + "}")
        if base:
            return os.path.join(base, rest.lstrip("\\/"))
    return path


class Worker:
    """A single dedicated thread. ``initializer`` runs once on it (COM
    apartment, DPI mode). Calls already on the worker run inline, so helpers
    can nest without deadlocking a one-thread pool (v1 ocr.py lesson)."""

    def __init__(self, name: str, initializer: Callable[[], None] | None = None) -> None:
        self.name = name
        self._initializer = initializer
        self._pool: concurrent.futures.ThreadPoolExecutor | None = None
        self._lock = threading.Lock()
        self._local = threading.local()

    def _init(self) -> None:
        self._local.active = True
        set_thread_dpi_aware()
        if self._initializer is not None:
            try:
                self._initializer()
            except Exception:  # noqa: BLE001 - a broken initializer must not kill the pool
                log.exception("worker %s initializer failed", self.name)

    def _executor(self) -> concurrent.futures.ThreadPoolExecutor:
        with self._lock:
            if self._pool is None:
                self._pool = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix=self.name, initializer=self._init)
            return self._pool

    @property
    def on_worker(self) -> bool:
        return bool(getattr(self._local, "active", False))

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run ``fn`` on the worker and wait (from any non-loop thread)."""
        if self.on_worker:
            return fn(*args, **kwargs)
        return self._executor().submit(fn, *args, **kwargs).result()

    async def run(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Await ``fn`` on the worker without blocking the event loop."""
        if self.on_worker:
            return fn(*args, **kwargs)
        future = self._executor().submit(fn, *args, **kwargs)
        return await asyncio.wrap_future(future)

    def shutdown(self) -> None:
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)


# Variables an Electron host leaves in the environment of processes it starts.
# Measured 2026-09-24: SAM run from a VS Code terminal/extension inherits
# ELECTRON_RUN_AS_NODE=1 and VSCODE_ESM_ENTRYPOINT & co.; Code.exe started with
# them runs as plain Node, exits with code 1 and no window appears (the first
# live build_project run never saw its VS Code window). The same breaks every
# Electron app SAM launches (Cursor, Discord, Slack...). With them removed
# the VS Code window appeared in 1.2 s.
_HOST_ENV_NAMES = frozenset({"ELECTRON_RUN_AS_NODE", "ELECTRON_NO_ATTACH_CONSOLE", "ELECTRON_NO_ASAR"})
_HOST_ENV_PREFIXES = ("VSCODE_",)


def is_host_variable(name: str) -> bool:
    upper = name.upper()
    return upper in _HOST_ENV_NAMES or upper.startswith(_HOST_ENV_PREFIXES)


def launch_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    """A copy of ``base`` (default ``os.environ``) without Electron/VS Code host variables."""
    source = os.environ if base is None else base
    return {k: v for k, v in source.items() if not is_host_variable(k)}


def scrub_host_environment() -> list[str]:
    """Remove Electron/VS Code host variables from SAM's own environment, so
    apps started through ShellExecute (``os.startfile``, ``shell:AppsFolder``),
    which cannot take an explicit environment, start normally too. SAM itself
    never reads them. Returns the removed names (values are never logged)."""
    removed = [name for name in list(os.environ) if is_host_variable(name)]
    for name in removed:
        os.environ.pop(name, None)
    return removed


def com_sta_initializer() -> None:
    """Initialise COM (STA) on a worker thread. comtypes initialises the thread
    that first imports it as STA too, so STA avoids RPC_E_CHANGED_MODE."""
    try:
        import pythoncom  # pywin32

        pythoncom.CoInitialize()
    except ImportError:
        pass


__all__ = ["IS_WINDOWS", "KNOWN_FOLDERS", "Worker", "com_sta_initializer", "dpi_aware", "expand_known_folder_path",
           "is_host_variable", "known_folder", "launch_environment", "pid_alive", "process_image", "running_processes",
           "scrub_host_environment", "set_thread_dpi_aware", "user32"]
