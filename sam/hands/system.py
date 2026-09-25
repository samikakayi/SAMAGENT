"""System status and control: battery, time, disks, memory, volume
(Core Audio), media keys and Windows toast notifications.

Volume uses the Core Audio endpoint (IAudioEndpointVolume through comtypes)
on a short-lived COM thread, so SAM can say and set the exact level
("دەنگەکە بکە بە پەنجا"), not only press the volume keys. Media keys go
through ``input`` (layout-independent virtual keys).
"""

from __future__ import annotations

import ctypes
import datetime as _dt
import os
import shutil
import string
import threading
import time
from ctypes import wintypes
from typing import Any, Callable


class _SYSTEM_POWER_STATUS(ctypes.Structure):
    _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", wintypes.DWORD), ("BatteryFullLifeTime", wintypes.DWORD)]


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD), ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64), ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64), ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64), ("ullAvailExtendedVirtual", ctypes.c_uint64)]


def battery() -> dict[str, Any]:
    status = _SYSTEM_POWER_STATUS()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return {"present": None}
    if status.BatteryFlag == 128 or status.BatteryLifePercent == 255:
        return {"present": False, "plugged_in": status.ACLineStatus == 1}
    minutes = None if status.BatteryLifeTime == 0xFFFFFFFF else round(status.BatteryLifeTime / 60)
    return {"present": True, "percent": int(status.BatteryLifePercent), "plugged_in": status.ACLineStatus == 1,
            "charging": bool(status.BatteryFlag & 8), "minutes_left": minutes,
            "saver": bool(status.SystemStatusFlag & 1)}


def memory() -> dict[str, Any]:
    status = _MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return {"used_percent": int(status.dwMemoryLoad), "total_gb": round(status.ullTotalPhys / 2**30, 1),
            "free_gb": round(status.ullAvailPhys / 2**30, 1)}


def disks() -> list[dict[str, Any]]:
    found = []
    mask = ctypes.windll.kernel32.GetLogicalDrives()
    for index, letter in enumerate(string.ascii_uppercase):
        if not mask & (1 << index):
            continue
        root = f"{letter}:\\"
        if ctypes.windll.kernel32.GetDriveTypeW(root) != 3:  # DRIVE_FIXED
            continue
        try:
            usage = shutil.disk_usage(root)
        except OSError:
            continue
        found.append({"drive": root, "total_gb": round(usage.total / 2**30, 1), "free_gb": round(usage.free / 2**30, 1),
                      "used_percent": round(100 * usage.used / usage.total) if usage.total else None})
    return found


def uptime_hours() -> float:
    ctypes.windll.kernel32.GetTickCount64.restype = ctypes.c_uint64
    return round(ctypes.windll.kernel32.GetTickCount64() / 3_600_000, 1)


# --- Core Audio (master volume) -------------------------------------------------------
def _audio_interfaces() -> tuple[Any, Any, Any]:
    import comtypes
    from comtypes import COMMETHOD, GUID, HRESULT, IUnknown

    class IAudioEndpointVolume(IUnknown):
        _iid_ = GUID("{5CDF2C82-841E-4546-9722-0CF74078229A}")
        _methods_ = [
            COMMETHOD([], HRESULT, "RegisterControlChangeNotify", (["in"], ctypes.c_void_p, "pNotify")),
            COMMETHOD([], HRESULT, "UnregisterControlChangeNotify", (["in"], ctypes.c_void_p, "pNotify")),
            COMMETHOD([], HRESULT, "GetChannelCount", (["out"], ctypes.POINTER(ctypes.c_uint), "pnChannelCount")),
            COMMETHOD([], HRESULT, "SetMasterVolumeLevel", (["in"], ctypes.c_float, "fLevelDB"),
                      (["in"], ctypes.POINTER(GUID), "pguidEventContext")),
            COMMETHOD([], HRESULT, "SetMasterVolumeLevelScalar", (["in"], ctypes.c_float, "fLevel"),
                      (["in"], ctypes.POINTER(GUID), "pguidEventContext")),
            COMMETHOD([], HRESULT, "GetMasterVolumeLevel", (["out"], ctypes.POINTER(ctypes.c_float), "pfLevelDB")),
            COMMETHOD([], HRESULT, "GetMasterVolumeLevelScalar", (["out"], ctypes.POINTER(ctypes.c_float), "pfLevel")),
            COMMETHOD([], HRESULT, "SetChannelVolumeLevel", (["in"], ctypes.c_uint, "nChannel"),
                      (["in"], ctypes.c_float, "fLevelDB"), (["in"], ctypes.POINTER(GUID), "pguidEventContext")),
            COMMETHOD([], HRESULT, "SetChannelVolumeLevelScalar", (["in"], ctypes.c_uint, "nChannel"),
                      (["in"], ctypes.c_float, "fLevel"), (["in"], ctypes.POINTER(GUID), "pguidEventContext")),
            COMMETHOD([], HRESULT, "GetChannelVolumeLevel", (["in"], ctypes.c_uint, "nChannel"),
                      (["out"], ctypes.POINTER(ctypes.c_float), "pfLevelDB")),
            COMMETHOD([], HRESULT, "GetChannelVolumeLevelScalar", (["in"], ctypes.c_uint, "nChannel"),
                      (["out"], ctypes.POINTER(ctypes.c_float), "pfLevel")),
            COMMETHOD([], HRESULT, "SetMute", (["in"], wintypes.BOOL, "bMute"),
                      (["in"], ctypes.POINTER(GUID), "pguidEventContext")),
            COMMETHOD([], HRESULT, "GetMute", (["out"], ctypes.POINTER(wintypes.BOOL), "pbMute")),
        ]

    class IMMDevice(IUnknown):
        _iid_ = GUID("{D666063F-1587-4E43-81F1-B948E807363F}")
        _methods_ = [
            COMMETHOD([], HRESULT, "Activate", (["in"], ctypes.POINTER(GUID), "iid"), (["in"], wintypes.DWORD, "dwClsCtx"),
                      (["in"], ctypes.c_void_p, "pActivationParams"),
                      (["out"], ctypes.POINTER(ctypes.POINTER(IUnknown)), "ppInterface")),
        ]

    class IMMDeviceEnumerator(IUnknown):
        _iid_ = GUID("{A95664D2-9614-4F35-A746-DE8DB63617E6}")
        _methods_ = [
            COMMETHOD([], HRESULT, "EnumAudioEndpoints", (["in"], ctypes.c_int, "dataFlow"),
                      (["in"], wintypes.DWORD, "dwStateMask"), (["out"], ctypes.POINTER(ctypes.c_void_p), "ppDevices")),
            COMMETHOD([], HRESULT, "GetDefaultAudioEndpoint", (["in"], ctypes.c_int, "dataFlow"),
                      (["in"], ctypes.c_int, "role"), (["out"], ctypes.POINTER(ctypes.POINTER(IMMDevice)), "ppEndpoint")),
        ]
    return comtypes, IMMDeviceEnumerator, IAudioEndpointVolume


def _with_endpoint(action: Callable[[Any], Any]) -> Any:
    """Run ``action(IAudioEndpointVolume)`` on a short-lived STA thread."""
    result: dict[str, Any] = {}

    def worker() -> None:
        try:
            comtypes, enumerator_iface, volume_iface = _audio_interfaces()
            comtypes.CoInitializeEx(comtypes.COINIT_APARTMENTTHREADED)
            try:
                from comtypes import GUID
                from comtypes.client import CreateObject

                enumerator = CreateObject(GUID("{BCDE0395-E52F-467C-8E3D-C4579291692E}"), interface=enumerator_iface)
                device = enumerator.GetDefaultAudioEndpoint(0, 1)  # eRender, eMultimedia
                raw = device.Activate(ctypes.byref(volume_iface._iid_), 23, None)  # CLSCTX_ALL
                result["value"] = action(raw.QueryInterface(volume_iface))
            finally:
                comtypes.CoUninitialize()
        except Exception as exc:  # noqa: BLE001
            result["error"] = exc

    thread = threading.Thread(target=worker, name="sam-audio", daemon=True)
    thread.start()
    thread.join(10)
    if "error" in result:
        raise OSError(f"volume control failed: {result['error']}")
    return result.get("value")


def get_volume() -> dict[str, Any]:
    def read(endpoint: Any) -> dict[str, Any]:
        return {"level": round(endpoint.GetMasterVolumeLevelScalar() * 100), "muted": bool(endpoint.GetMute())}
    return _with_endpoint(read)


def set_volume(level: int | None = None, mute: bool | None = None) -> dict[str, Any]:
    def write(endpoint: Any) -> dict[str, Any]:
        if level is not None:
            endpoint.SetMasterVolumeLevelScalar(max(0.0, min(1.0, level / 100.0)), None)
        if mute is not None:
            endpoint.SetMute(bool(mute), None)
        return {"level": round(endpoint.GetMasterVolumeLevelScalar() * 100), "muted": bool(endpoint.GetMute())}
    return _with_endpoint(write)


def info(timezone: str = "Asia/Baghdad") -> dict[str, Any]:
    from zoneinfo import ZoneInfo

    now = _dt.datetime.now(ZoneInfo(timezone))
    data: dict[str, Any] = {"time": now.strftime("%H:%M"), "date": now.strftime("%Y-%m-%d"),
                            "weekday": now.strftime("%A"), "timezone": timezone}
    for key, fn in (("battery", battery), ("memory", memory), ("disks", disks), ("uptime_hours", uptime_hours),
                    ("volume", get_volume)):
        try:
            data[key] = fn()
        except Exception as exc:  # noqa: BLE001 - one missing part must not hide the rest
            data[key] = {"error": type(exc).__name__}
    data["computer"] = os.environ.get("COMPUTERNAME", "")
    return data


def toast_script(title: str, body: str) -> str:
    """PowerShell for a Windows toast via the WinRT ToastNotificationManager
    (Windows PowerShell 5.1 has the WinRT projection; pwsh 7 does not)."""
    from xml.sax.saxutils import escape

    xml = (f"<toast><visual><binding template='ToastGeneric'><text>{escape(title)}</text>"
           f"<text>{escape(body)}</text></binding></visual></toast>")
    xml = xml.replace("'", "''")
    app_id = r"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"
    return ("[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null; "
            "[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] > $null; "
            f"$x = New-Object Windows.Data.Xml.Dom.XmlDocument; $x.LoadXml('{xml}'); "
            "$t = [Windows.UI.Notifications.ToastNotification]::new($x); "
            f"[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('{app_id}').Show($t)")


def notify(title: str, body: str, *, runner: Callable[..., Any] | None = None) -> bool:
    """Show a Windows toast (hidden Windows PowerShell 5.1). True if shown."""
    import subprocess

    script = toast_script(title[:120], body[:400])
    run = runner or subprocess.run
    started = time.perf_counter()
    completed = run(["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=0x08000000, timeout=20, check=False)
    return getattr(completed, "returncode", 1) == 0 and time.perf_counter() - started < 20


__all__ = ["battery", "disks", "get_volume", "info", "memory", "notify", "set_volume", "toast_script", "uptime_hours"]
