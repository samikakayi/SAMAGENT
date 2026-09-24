"""Live: start SAM 2 exactly like the sign-in shortcut and measure it (design
acceptance 2: island visible < 3 s, RAM < 400 MB idle, no console window).

    pythonw SAM.pyw --background --home <home>

<home> is a fresh TEMP folder by default: no keys, no v1 data, no alerts, so
nothing can be spoken or imported. Set SAM_ACCEPTANCE_REAL_HOME=1 to use
SAM_HOME instead. Then a second ``--background`` launch must exit at once
without a second SAM, and a second normal launch must make the running SAM
show its panel. The script only ever terminates the processes it started;
if SAM 2 is already running it skips (it never touches the user's SAM).
"""

from __future__ import annotations

import ctypes
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from ctypes import wintypes
from pathlib import Path

from _common import ROOT, Acceptance, sam_home

ISLAND_TARGET_MS = 3000
RAM_TARGET_MB = 400
IDLE_S = 10

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
psapi = ctypes.WinDLL("psapi", use_last_error=True)


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_void_p), ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", wintypes.LONG), ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260)]


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t), ("PrivateUsage", ctypes.c_size_t)]


kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
kernel32.OpenProcess.restype = wintypes.HANDLE
psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]


def process_tree(root_pid: int) -> dict[int, str]:
    """{pid: exe} of root_pid and all its descendants."""
    snapshot = kernel32.CreateToolhelp32Snapshot(0x2, 0)
    entry = PROCESSENTRY32W()
    entry.dwSize = ctypes.sizeof(entry)
    parents: dict[int, tuple[int, str]] = {}
    if kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
        while True:
            parents[entry.th32ProcessID] = (entry.th32ParentProcessID, entry.szExeFile)
            if not kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                break
    kernel32.CloseHandle(snapshot)
    tree = {root_pid: parents.get(root_pid, (0, "?"))[1]}
    changed = True
    while changed:
        changed = False
        for pid, (parent, exe) in parents.items():
            if parent in tree and pid not in tree and pid != parent:
                tree[pid] = exe
                changed = True
    return tree


def visible_windows(pids: set[int]) -> list[dict]:
    found: list[dict] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _lparam):
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value in pids and user32.IsWindowVisible(hwnd):
            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            name = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, name, 256)
            if rect.right - rect.left > 0 and rect.bottom - rect.top > 0:
                found.append({"hwnd": int(hwnd), "pid": pid.value, "class": name.value,
                              "size": [rect.right - rect.left, rect.bottom - rect.top]})
        return True
    user32.EnumWindows(callback, 0)
    return found


def memory_mb(pid: int) -> dict[str, float]:
    handle = kernel32.OpenProcess(0x1000 | 0x0010, False, pid)   # QUERY_LIMITED_INFORMATION | VM_READ
    if not handle:
        return {}
    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(counters)
    try:
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return {}
        return {"working_set_mb": round(counters.WorkingSetSize / 2**20, 1),
                "private_mb": round(counters.PrivateUsage / 2**20, 1)}
    finally:
        kernel32.CloseHandle(handle)


def terminate(pids: list[int]) -> None:
    for pid in pids:
        handle = kernel32.OpenProcess(0x0001, False, pid)   # PROCESS_TERMINATE
        if handle:
            kernel32.TerminateProcess(handle, 0)
            kernel32.CloseHandle(handle)


def _load_launcher():
    import importlib.machinery
    import importlib.util

    loader = importlib.machinery.SourceFileLoader("sam_launcher_probe", str(ROOT / "SAM.pyw"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def main() -> int:
    acc = Acceptance("launcher_startup")
    launcher = _load_launcher()
    if launcher.instance_running():
        acc.skip("SAM 2 is already running on this PC; quit it first (this check never touches it)")
        return acc.finish()
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    real_home = os.environ.get("SAM_ACCEPTANCE_REAL_HOME") == "1"
    temp = Path(tempfile.mkdtemp(prefix="sam2-startup-"))
    home = sam_home() if real_home else temp / "home"
    (home / "data").mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "SAM_LOG_DIR": str(temp / "logs")}
    command = [str(pythonw), str(ROOT / "SAM.pyw"), "--home", str(home)]
    started_pids: list[int] = []
    try:
        began = time.perf_counter()
        first = subprocess.Popen(command + ["--background"], env=env, cwd=str(ROOT))
        started_pids.append(first.pid)

        with acc.check(f"island visible < {ISLAND_TARGET_MS} ms") as c:
            windows: list[dict] = []
            while time.perf_counter() - began < 60 and first.poll() is None:
                windows = visible_windows(set(process_tree(first.pid)))
                if windows:
                    break
                time.sleep(0.05)
            c.data["ms"] = round((time.perf_counter() - began) * 1000)
            c.data["windows"] = windows
            c.data["home"] = "SAM_HOME" if real_home else "temp"
            assert first.poll() is None, f"SAM exited with code {first.returncode}"
            assert windows, "no visible SAM window within 60 s"
            c.detail = f"{c.data['ms']} ms"
            assert c.data["ms"] < ISLAND_TARGET_MS, f"{c.data['ms']} ms"

        with acc.check("core ready (startup:core_ready timing)") as c:
            db = home / "data" / "sam2.sqlite3"
            deadline = time.monotonic() + 30
            value = None
            while time.monotonic() < deadline and value is None:
                try:
                    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True, timeout=2)
                    row = conn.execute("SELECT ms FROM timings WHERE stage='startup:core_ready' ORDER BY id DESC LIMIT 1").fetchone()
                    conn.close()
                    value = row[0] if row else None
                except sqlite3.Error:
                    value = None
                if value is None:
                    time.sleep(0.25)
            c.data["core_ready_ms"] = value
            assert value is not None, "no startup:core_ready timing recorded"
            c.detail = f"{value:.0f} ms"

        with acc.check("no console window") as c:
            tree = process_tree(first.pid)
            consoles = [w for w in visible_windows(set(tree))
                        if w["class"] in ("ConsoleWindowClass", "CASCADIA_HOSTING_WINDOW_CLASS")]
            c.data = {"processes": sorted(set(tree.values())), "consoles": consoles}
            assert not consoles, consoles

        with acc.check(f"RAM < {RAM_TARGET_MB} MB after {IDLE_S} s idle") as c:
            time.sleep(IDLE_S)
            tree = process_tree(first.pid)
            usage = {pid: memory_mb(pid) for pid, exe in tree.items() if exe.lower().startswith("python")}
            main_pid = max(usage, key=lambda p: usage[p].get("working_set_mb", 0)) if usage else first.pid
            c.data = {"per_process": {str(k): v for k, v in usage.items()}, "sam_pid": main_pid}
            working = usage.get(main_pid, {}).get("working_set_mb", 0.0)
            c.detail = f"working set {working} MB, private {usage.get(main_pid, {}).get('private_mb')} MB"
            assert 0 < working < RAM_TARGET_MB, c.detail

        with acc.check("second background launch exits without a second SAM") as c:
            before = set(process_tree(first.pid))
            t0 = time.perf_counter()
            second = subprocess.run(command + ["--background"], env=env, cwd=str(ROOT), timeout=30)
            c.data = {"exit_code": second.returncode, "ms": round((time.perf_counter() - t0) * 1000)}
            assert second.returncode == 0
            assert first.poll() is None and set(process_tree(first.pid)) == before

        with acc.check("startup breakdown (informational)") as c:
            db = home / "data" / "sam2.sqlite3"
            conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True, timeout=2)
            try:
                rows = conn.execute("SELECT stage, ms FROM timings WHERE stage LIKE 'startup:%' ORDER BY id").fetchall()
            finally:
                conn.close()
            c.data = {stage: round(ms) for stage, ms in rows}
            slow = sorted(((ms, stage) for stage, ms in rows if not stage.endswith((":app", ":core_ready"))), reverse=True)[:4]
            c.detail = ", ".join(f"{stage} {ms:.0f} ms" for ms, stage in slow)

        with acc.check("second normal launch shows the running SAM's panel") as c:
            before_windows = {w["hwnd"] for w in visible_windows(set(process_tree(first.pid)))}
            subprocess.run(command, env=env, cwd=str(ROOT), timeout=30)
            deadline = time.monotonic() + 10
            new: list[dict] = []
            while time.monotonic() < deadline and not new:
                new = [w for w in visible_windows(set(process_tree(first.pid))) if w["hwnd"] not in before_windows]
                time.sleep(0.1)
            c.data["new_windows"] = new
            assert new, "no panel appeared (does the UI call sam.winapp.watch_show_requests?)"
    finally:
        # Only SAM's own python processes: a gateway SAM may have started
        # (node/cmd under the tree) must survive -- SAM never stops OmniRoute.
        tree = process_tree(started_pids[0]) if started_pids else {}
        terminate([pid for pid, exe in tree.items() if exe.lower().startswith("python")])
        time.sleep(1.0)
        shutil.rmtree(temp, ignore_errors=True)
    return acc.finish()


if __name__ == "__main__":
    sys.exit(main())
