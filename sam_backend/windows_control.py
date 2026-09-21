from __future__ import annotations

import ctypes
import io
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .contracts import CapabilityState, ExecutionStatus, StandardResult


class WindowsController:
    def __init__(self, data_dir: Path, *, computer_control: bool = False, screen_access: bool = False) -> None:
        self.data_dir = data_dir
        self.computer_control = computer_control
        self.screen_access = screen_access

    @staticmethod
    def _modules():
        if os.name != "nt":
            raise RuntimeError("Windows control is available only on Windows")
        try:
            import psutil
            import win32api
            import win32clipboard
            import win32con
            import win32gui
            import win32process
        except ImportError as exc:
            raise RuntimeError("Windows control requires psutil and pywin32") from exc
        return psutil, win32api, win32clipboard, win32con, win32gui, win32process

    def refresh_permissions(self, *, computer_control: bool, screen_access: bool) -> None:
        self.computer_control = computer_control
        self.screen_access = screen_access

    def list_processes(self, query: str = "", names: tuple[str, ...] = ()) -> StandardResult:
        """List processes, optionally narrowed to a substring or a set of names.

        `username` and `memory_info` each require opening the process, which on
        a machine with a few hundred of them costs seconds. They are read only
        for the rows that survive the filter, so a narrow query is fast.
        """
        started = time.perf_counter()
        try:
            psutil, *_ = self._modules()
            wanted = tuple(name.lower() for name in names)
            items = []
            for process in psutil.process_iter(["pid", "name"]):
                try:
                    name = process.info.get("name") or ""
                    lowered = name.lower()
                    if query and query.lower() not in lowered:
                        continue
                    if wanted and not any(candidate in lowered for candidate in wanted):
                        continue
                    detail = process.as_dict(attrs=["status", "username", "create_time", "memory_info"])
                    memory_info = detail.get("memory_info")
                    items.append({
                        "pid": int(process.info["pid"]),
                        "name": name,
                        "status": detail.get("status"),
                        "username": detail.get("username"),
                        "created_at": detail.get("create_time"),
                        "memory_bytes": int(memory_info.rss) if memory_info else None,
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return StandardResult.success({"processes": items[:2000], "truncated": len(items) > 2000}, started_at=started)
        except Exception as exc:
            return StandardResult.failure(str(exc), error_code="PROCESS_LIST_FAILED", started_at=started)

    def process_info(self, pid: int) -> StandardResult:
        started = time.perf_counter()
        try:
            psutil, *_ = self._modules()
            process = psutil.Process(int(pid))
            data = {
                "pid": process.pid,
                "name": process.name(),
                "status": process.status(),
                "exe": process.exe(),
                "cwd": process.cwd(),
                "create_time": process.create_time(),
                "children": [{"pid": child.pid, "name": child.name()} for child in process.children(recursive=True)],
            }
            return StandardResult.success(data, started_at=started)
        except Exception as exc:
            return StandardResult.failure(str(exc), error_code="PROCESS_INFO_FAILED", started_at=started)

    def stop_process(self, pid: int) -> StandardResult:
        started = time.perf_counter()
        if not self.computer_control:
            return StandardResult.failure("Computer Control is OFF.", error_code="COMPUTER_CONTROL_DISABLED", started_at=started)
        if int(pid) in {os.getpid(), os.getppid()}:
            return StandardResult.failure("SAM refuses to terminate itself or its launcher.", error_code="PROTECTED_PROCESS", started_at=started)
        try:
            psutil, *_ = self._modules()
            process = psutil.Process(int(pid))
            children = process.children(recursive=True)
            for child in reversed(children):
                child.terminate()
            process.terminate()
            _, alive = psutil.wait_procs([*children, process], timeout=4)
            for item in alive:
                item.kill()
            verified = not psutil.pid_exists(int(pid))
            return StandardResult(
                ExecutionStatus.SUCCESS if verified else ExecutionStatus.PARTIAL,
                True,
                verified,
                data={"pid": int(pid), "terminated_children": [item.pid for item in children]},
                error=None if verified else "Process still exists after termination request.",
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        except Exception as exc:
            return StandardResult.failure(str(exc), executed=True, error_code="PROCESS_STOP_FAILED", started_at=started)

    def enumerate_windows(self) -> StandardResult:
        started = time.perf_counter()
        try:
            psutil, _, _, _, win32gui, win32process = self._modules()
            foreground = int(win32gui.GetForegroundWindow())
            windows: list[dict[str, Any]] = []

            def callback(hwnd: int, _: Any) -> None:
                if not win32gui.IsWindowVisible(hwnd):
                    return
                title = win32gui.GetWindowText(hwnd).strip()
                if not title:
                    return
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                try:
                    process = psutil.Process(pid)
                    process_name = process.name()
                except Exception:
                    process_name = None
                left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                windows.append({
                    "hwnd": int(hwnd), "pid": int(pid), "process": process_name, "title": title,
                    "active": int(hwnd) == foreground,
                    "geometry": {"left": left, "top": top, "right": right, "bottom": bottom, "width": right - left, "height": bottom - top},
                })

            win32gui.EnumWindows(callback, None)
            return StandardResult.success({"windows": windows, "active_window": next((item for item in windows if item["active"]), None)}, started_at=started)
        except Exception as exc:
            return StandardResult.failure(str(exc), error_code="WINDOW_ENUMERATION_FAILED", started_at=started)

    def window_action(self, hwnd: int, action: str, *, x: int | None = None, y: int | None = None, width: int | None = None, height: int | None = None) -> StandardResult:
        started = time.perf_counter()
        if not self.computer_control:
            return StandardResult.failure("Computer Control is OFF.", error_code="COMPUTER_CONTROL_DISABLED", started_at=started)
        try:
            _, _, _, win32con, win32gui, _ = self._modules()
            hwnd = int(hwnd)
            if not win32gui.IsWindow(hwnd):
                return StandardResult.failure("Window handle is no longer valid", error_code="WINDOW_NOT_FOUND", started_at=started)
            normalized = action.strip().lower()
            if normalized == "focus":
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(hwnd)
                verified = int(win32gui.GetForegroundWindow()) == hwnd
            elif normalized == "minimize":
                win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
                verified = bool(win32gui.IsIconic(hwnd))
            elif normalized == "maximize":
                win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
                verified = bool(win32gui.IsZoomed(hwnd))
            elif normalized == "restore":
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                verified = not win32gui.IsIconic(hwnd)
            elif normalized == "close":
                win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
                time.sleep(0.2)
                verified = not bool(win32gui.IsWindow(hwnd))
            elif normalized == "move":
                if None in {x, y, width, height} or min(int(width), int(height)) < 50:
                    return StandardResult.failure("move requires x, y, width, and height (minimum 50px)", error_code="INVALID_GEOMETRY", started_at=started)
                win32gui.MoveWindow(hwnd, int(x), int(y), int(width), int(height), True)
                left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                verified = (left, top, right - left, bottom - top) == (int(x), int(y), int(width), int(height))
            else:
                return StandardResult.failure(f"Unsupported window action: {action}", error_code="INVALID_ACTION", started_at=started)
            return StandardResult.success({"hwnd": hwnd, "action": normalized}, verified=verified, started_at=started)
        except Exception as exc:
            return StandardResult.failure(str(exc), executed=True, error_code="WINDOW_ACTION_FAILED", started_at=started)

    def list_installed_apps(self, query: str = "") -> StandardResult:
        started = time.perf_counter()
        if os.name != "nt":
            return StandardResult.failure("Installed-app discovery is available only on Windows", error_code="UNSUPPORTED_PLATFORM", started_at=started)
        try:
            import winreg

            roots = [
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
                (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
            ]
            apps: dict[tuple[str, str], dict[str, Any]] = {}
            for hive, path in roots:
                try:
                    with winreg.OpenKey(hive, path) as root:
                        count = winreg.QueryInfoKey(root)[0]
                        for index in range(count):
                            try:
                                with winreg.OpenKey(root, winreg.EnumKey(root, index)) as key:
                                    name = str(winreg.QueryValueEx(key, "DisplayName")[0]).strip()
                                    if not name or (query and query.lower() not in name.lower()):
                                        continue
                                    def value(field: str) -> str | None:
                                        try:
                                            return str(winreg.QueryValueEx(key, field)[0])
                                        except OSError:
                                            return None
                                    item = {"name": name, "version": value("DisplayVersion"), "publisher": value("Publisher"), "location": value("InstallLocation")}
                                    apps[(name.lower(), item["version"] or "")] = item
                            except OSError:
                                continue
                except OSError:
                    continue
            return StandardResult.success({"apps": sorted(apps.values(), key=lambda item: item["name"].lower())}, started_at=started)
        except Exception as exc:
            return StandardResult.failure(str(exc), error_code="APP_DISCOVERY_FAILED", started_at=started)

    def read_clipboard(self) -> StandardResult:
        started = time.perf_counter()
        try:
            _, _, clipboard, win32con, _, _ = self._modules()
            clipboard.OpenClipboard()
            try:
                if not clipboard.IsClipboardFormatAvailable(win32con.CF_UNICODETEXT):
                    return StandardResult.success({"text": None, "format": "non-text"}, started_at=started)
                text = clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
            finally:
                clipboard.CloseClipboard()
            return StandardResult.success({"text": str(text)[:100_000], "truncated": len(str(text)) > 100_000}, started_at=started)
        except Exception as exc:
            return StandardResult.failure(str(exc), error_code="CLIPBOARD_READ_FAILED", started_at=started)

    def write_clipboard(self, text: str) -> StandardResult:
        started = time.perf_counter()
        if not self.computer_control:
            return StandardResult.failure("Computer Control is OFF.", error_code="COMPUTER_CONTROL_DISABLED", started_at=started)
        try:
            _, _, clipboard, win32con, _, _ = self._modules()
            clipboard.OpenClipboard()
            try:
                clipboard.EmptyClipboard()
                clipboard.SetClipboardData(win32con.CF_UNICODETEXT, text[:100_000])
            finally:
                clipboard.CloseClipboard()
            return StandardResult.success({"characters": min(len(text), 100_000)}, started_at=started)
        except Exception as exc:
            return StandardResult.failure(str(exc), executed=True, error_code="CLIPBOARD_WRITE_FAILED", started_at=started)

    def capture_screen(self, *, monitor: int | None = None) -> StandardResult:
        started = time.perf_counter()
        if not self.screen_access:
            return StandardResult.failure("Screen Access is OFF.", error_code="SCREEN_ACCESS_DISABLED", started_at=started)
        try:
            from PIL import ImageGrab

            image = ImageGrab.grab(all_screens=True)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            raw = buffer.getvalue()
            import hashlib
            digest = hashlib.sha256(raw).hexdigest()
            directory = self.data_dir / "screenshots"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"screen-{int(time.time() * 1000)}-{digest[:12]}.png"
            path.write_bytes(raw)
            return StandardResult.success({"path": str(path), "sha256": digest, "width": image.width, "height": image.height}, verified=image.width > 100 and image.height > 100, started_at=started)
        except Exception as exc:
            return StandardResult.failure(str(exc), executed=True, error_code="SCREEN_CAPTURE_FAILED", started_at=started)

    def inspect_ui_tree(self, hwnd: int) -> StandardResult:
        started = time.perf_counter()
        try:
            from pywinauto import Desktop
        except ImportError:
            return StandardResult(
                ExecutionStatus.PARTIAL,
                False,
                False,
                data={"capability": CapabilityState.UNCONFIGURED.value},
                error="pywinauto is not installed; UI Automation inspection is unavailable.",
                error_code="UIA_UNCONFIGURED",
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        try:
            window = Desktop(backend="uia").window(handle=int(hwnd))
            descendants = []
            for element in window.descendants()[:2000]:
                info = element.element_info
                descendants.append({"control_type": info.control_type, "name": info.name, "automation_id": info.automation_id, "enabled": element.is_enabled(), "visible": element.is_visible()})
            return StandardResult.success({"hwnd": int(hwnd), "elements": descendants, "truncated": len(descendants) >= 2000}, started_at=started)
        except Exception as exc:
            return StandardResult.failure(str(exc), error_code="UIA_INSPECTION_FAILED", started_at=started)

