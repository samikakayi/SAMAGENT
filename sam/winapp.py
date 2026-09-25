"""Small Windows helpers shared by several packages.

- ``activate_aumid``: start/activate a packaged (MSIX/Store) or Start-menu app
  by its AppUserModelID WITH command-line arguments, through
  ``IApplicationActivationManager::ActivateApplication``. Verified by the lead
  on 2026-09-24 to be the working way to start TradingView Desktop with
  ``--remote-debugging-port=9222`` (launching the WindowsApps exe directly
  exits at once). Used by hands.apps (open_app) and trading.tradingview.
- single-instance named mutex + "show" event (a second ``python -m sam`` asks
  the running one to show its panel and exits) + "quit" event (``python -m sam
  --quit`` / ``SAM.pyw --quit`` asks the running one to shut down cleanly:
  stop listening, close CDP/MT5, flush the DB -- the installer and the
  integration smoke test need a stop that is not a process kill).
- per-monitor-v2 DPI awareness for headless runs (Qt sets it itself when the
  UI starts).
"""

from __future__ import annotations

import ctypes
import logging
import os
import threading
from ctypes import wintypes
from typing import Any, Callable

log = logging.getLogger("sam.winapp")

MUTEX_NAME = "Local\\SAM2.SingleInstance"
SHOW_EVENT_NAME = "Local\\SAM2.ShowPanel"
QUIT_EVENT_NAME = "Local\\SAM2.Quit"
ERROR_ALREADY_EXISTS = 183
SYNCHRONIZE = 0x00100000

CLSID_APPLICATION_ACTIVATION_MANAGER = "{45BA127D-10A8-46EA-8AB7-56EA9078943C}"
IID_APPLICATION_ACTIVATION_MANAGER = "{2e941141-7f97-4756-ba1d-9decde894a3d}"
AO_NONE, AO_DESIGNMODE, AO_NOERRORUI, AO_NOSPLASHSCREEN = 0, 1, 2, 4

_mutex_handle: Any = None


def _interface() -> Any:
    import comtypes
    from comtypes import COMMETHOD, GUID, HRESULT, IUnknown

    class IApplicationActivationManager(IUnknown):
        _iid_ = GUID(IID_APPLICATION_ACTIVATION_MANAGER)
        _methods_ = [
            COMMETHOD([], HRESULT, "ActivateApplication",
                      (["in"], wintypes.LPCWSTR, "appUserModelId"),
                      (["in"], wintypes.LPCWSTR, "arguments"),
                      (["in"], ctypes.c_int, "options"),
                      (["out"], ctypes.POINTER(wintypes.DWORD), "processId")),
        ]
    return comtypes, GUID, IApplicationActivationManager


def activate_aumid(aumid: str, arguments: str = "", options: int = AO_NONE, timeout_s: float = 15.0) -> int:
    """Activate ``aumid`` with ``arguments``; return the process id.

    Runs on a short-lived thread with its own COM apartment so it is safe to
    call from any thread (use ``await asyncio.to_thread(activate_aumid, ...)``
    from the core loop). Raises OSError on failure.
    """
    result: dict[str, Any] = {}

    def worker() -> None:
        try:
            comtypes, GUID, interface = _interface()
            comtypes.CoInitializeEx(comtypes.COINIT_APARTMENTTHREADED)
            try:
                from comtypes.client import CreateObject
                manager = CreateObject(GUID(CLSID_APPLICATION_ACTIVATION_MANAGER), interface=interface)
                result["pid"] = int(manager.ActivateApplication(aumid, arguments or None, options))
            finally:
                comtypes.CoUninitialize()
        except Exception as exc:  # noqa: BLE001
            result["error"] = exc

    thread = threading.Thread(target=worker, name="sam-activate", daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise OSError(f"activation of {aumid} did not return within {timeout_s:.0f} s")
    if "error" in result:
        raise OSError(f"could not activate {aumid}: {result['error']}")
    return result.get("pid", 0)


def acquire_single_instance(name: str = MUTEX_NAME) -> bool:
    """True if this is the only SAM 2 process (keeps the mutex for life)."""
    global _mutex_handle
    if os.name != "nt":
        return True
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    handle = kernel32.CreateMutexW(None, False, name)
    already = ctypes.get_last_error() == ERROR_ALREADY_EXISTS
    if not handle:
        return True  # cannot tell; do not block start-up
    if already:
        kernel32.CloseHandle(handle)
        return False
    _mutex_handle = handle
    return True


def release_single_instance() -> None:
    """Drop the mutex at the end of a clean shutdown, so a follow-up launch
    (tray Restart, ``--quit`` waiters) sees SAM gone before the process exits."""
    global _mutex_handle
    handle, _mutex_handle = _mutex_handle, None
    if handle and os.name == "nt":
        ctypes.WinDLL("kernel32").CloseHandle(wintypes.HANDLE(handle))


def _event(name: str, create: bool) -> Any:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if create:
        kernel32.CreateEventW.restype = wintypes.HANDLE
        kernel32.CreateEventW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
        return kernel32, kernel32.CreateEventW(None, False, False, name)
    kernel32.OpenEventW.restype = wintypes.HANDLE
    kernel32.OpenEventW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    return kernel32, kernel32.OpenEventW(0x0002, False, name)  # EVENT_MODIFY_STATE


def _signal(name: str) -> bool:
    if os.name != "nt":
        return False
    kernel32, handle = _event(name, create=False)
    if not handle:
        return False
    try:
        return bool(kernel32.SetEvent(handle))
    finally:
        kernel32.CloseHandle(handle)


def _watch(name: str, callback: Callable[[], None], thread_name: str) -> threading.Thread | None:
    if os.name != "nt":
        return None
    kernel32, handle = _event(name, create=True)
    if not handle:
        return None
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]

    def loop() -> None:
        while True:
            if kernel32.WaitForSingleObject(handle, 0xFFFFFFFF) == 0:
                try:
                    callback()
                except Exception:  # noqa: BLE001
                    log.exception("%s callback failed", thread_name)

    thread = threading.Thread(target=loop, name=thread_name, daemon=True)
    thread.start()
    return thread


def signal_show(name: str = SHOW_EVENT_NAME) -> bool:
    """Ask the running instance to show itself (second launch)."""
    return _signal(name)


def watch_show_requests(callback: Callable[[], None], name: str = SHOW_EVENT_NAME) -> threading.Thread | None:
    """Call ``callback`` (on a daemon thread) whenever ``signal_show`` fires.
    The UI passes a function that emits a Qt signal."""
    return _watch(name, callback, "sam-show-watch")


def signal_quit(name: str = QUIT_EVENT_NAME) -> bool:
    """Ask the running instance to shut down cleanly. False = nobody listens
    (not running, or still starting before its watcher exists)."""
    return _signal(name)


def watch_quit_requests(callback: Callable[[], None], name: str = QUIT_EVENT_NAME) -> threading.Thread | None:
    """Call ``callback`` (on a daemon thread) whenever ``signal_quit`` fires."""
    return _watch(name, callback, "sam-quit-watch")


def instance_running(name: str = MUTEX_NAME) -> bool:
    """True while a SAM 2 process holds the single-instance mutex (never creates it)."""
    if os.name != "nt":
        return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenMutexW.restype = wintypes.HANDLE
    kernel32.OpenMutexW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    handle = kernel32.OpenMutexW(SYNCHRONIZE, False, name)
    if handle:
        kernel32.CloseHandle(handle)
        return True
    return ctypes.get_last_error() == 5  # ERROR_ACCESS_DENIED: exists at another integrity level


def set_dpi_awareness() -> bool:
    """Per-monitor v2 DPI awareness (call before any window exists; headless
    runs only -- Qt 6 sets the same mode when QApplication starts)."""
    if os.name != "nt":
        return False
    try:
        return bool(ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)))
    except (AttributeError, OSError):
        return False


__all__ = ["activate_aumid", "acquire_single_instance", "release_single_instance", "signal_show",
           "watch_show_requests", "signal_quit", "watch_quit_requests", "instance_running", "set_dpi_awareness",
           "MUTEX_NAME", "SHOW_EVENT_NAME", "QUIT_EVENT_NAME"]
