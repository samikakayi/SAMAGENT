"""run_powershell: one hidden PowerShell process per command, with a
timeout, process-tree kill, output caps and secret masking.

- ``-NoProfile -NonInteractive``, ``CREATE_NO_WINDOW``: no console flashes, no
  user profile scripts, never waits for input.
- Output is UTF-8 (the console code page would mangle Sorani file names).
- Child processes get the user's normal environment (v1 stripped APPDATA &
  co., which breaks tools like winget/npm -- reports/computer-control.json)
  minus variables whose names look like credentials. ``.env`` values never
  reach ``os.environ`` in SAM 2, so they cannot leak to children either.
- On timeout or stop_all the whole process tree is killed (``taskkill /T /F``).
- stdout/stderr are redacted by the caller (``app.redact``) and capped.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from ._win import launch_environment

_SECRET_ENV = re.compile(r"(?i)(api[_-]?key|token|secret|passw|credential|private[_-]?key|^litellm_|^openrouter|"
                         r"^groq|^gemini|^google_api|^openai|^anthropic|^kurdishtts|^n8n_api)")
OUTPUT_CAP = 4000
CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
PRELUDE = ("$ProgressPreference='SilentlyContinue'; "
           "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); "
           "$OutputEncoding=[System.Text.UTF8Encoding]::new($false); ")


@dataclass
class ShellResult:
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    cancelled: bool = False
    duration_ms: float = 0.0
    truncated: bool = False


def powershell_exe() -> str:
    """pwsh 7 when installed (faster, UTF-8 by default), else Windows PowerShell."""
    return shutil.which("pwsh") or shutil.which("powershell") or "powershell.exe"


def child_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    """The user's environment minus credential-looking names and the
    Electron/VS Code host variables (``_win.launch_environment``)."""
    source = launch_environment(dict(os.environ if base is None else base))
    return {k: v for k, v in source.items() if not _SECRET_ENV.search(k)}


def cap(text: str, limit: int = OUTPUT_CAP) -> tuple[str, bool]:
    """Keep the head and the tail (errors often come last)."""
    if len(text) <= limit:
        return text, False
    head = int(limit * 0.7)
    return text[:head] + f"\n…[{len(text) - limit} characters cut]…\n" + text[-(limit - head):], True


def _kill_tree(pid: int) -> None:
    try:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW,
                       check=False, timeout=10)
    except (OSError, subprocess.SubprocessError):
        pass


def run_sync(command: str, *, timeout_s: float = 45.0, cwd: str | None = None,
             cancel: threading.Event | None = None, exe: str | None = None,
             popen: Callable[..., Any] = subprocess.Popen) -> ShellResult:
    """Run one PowerShell command (blocking; call from a worker thread)."""
    started = time.perf_counter()
    argv = [exe or powershell_exe(), "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", PRELUDE + command]
    process = popen(argv, cwd=cwd or os.path.expanduser("~"), env=child_environment(), stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP)
    deadline = time.monotonic() + max(1.0, float(timeout_s))
    timed_out = cancelled = False
    out = err = b""
    while True:
        try:
            out, err = process.communicate(timeout=0.2)
            break
        except subprocess.TimeoutExpired:
            if cancel is not None and cancel.is_set():
                cancelled = True
            elif time.monotonic() >= deadline:
                timed_out = True
            else:
                continue
            _kill_tree(process.pid)
            try:
                process.kill()
            except OSError:
                pass
            try:
                out, err = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                out, err = b"", b""
            break
    stdout, cut_out = cap((out or b"").decode("utf-8", errors="replace").strip())
    stderr, cut_err = cap((err or b"").decode("utf-8", errors="replace").strip(), 2000)
    return ShellResult(exit_code=process.returncode, stdout=stdout, stderr=stderr, timed_out=timed_out,
                       cancelled=cancelled, duration_ms=round((time.perf_counter() - started) * 1000, 1),
                       truncated=cut_out or cut_err)


async def run_powershell(command: str, *, timeout_s: float = 45.0, cwd: str | None = None,
                         cancel: asyncio.Event | None = None) -> ShellResult:
    """Async wrapper: the process runs on a worker thread; ``cancel`` (the
    tool's stop_all event) kills the process tree."""
    stop = threading.Event()
    task = asyncio.ensure_future(asyncio.to_thread(run_sync, command, timeout_s=timeout_s, cwd=cwd, cancel=stop))
    try:
        if cancel is None:
            return await asyncio.shield(task)
        waiter = asyncio.ensure_future(cancel.wait())
        done, _ = await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
        if task not in done:
            stop.set()
        waiter.cancel()
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        stop.set()
        raise


__all__ = ["ShellResult", "cap", "child_environment", "powershell_exe", "run_powershell", "run_sync"]
