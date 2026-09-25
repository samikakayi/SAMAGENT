"""run_powershell plumbing: caps, secret-free child environment, timeout and
cancel kill the process tree; one real hidden PowerShell run proves UTF-8
Sorani output on this PC."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from typing import Any

import pytest

from sam.hands.shell import cap, child_environment, run_powershell, run_sync


def test_cap_keeps_head_and_tail() -> None:
    text = "".join(f"line {i}\n" for i in range(2000))
    capped, cut = cap(text, 400)
    assert cut and capped.startswith("line 0") and capped.rstrip().endswith("line 1999")
    assert "characters cut" in capped
    assert cap("short") == ("short", False)


def test_child_environment_drops_credentials_but_keeps_normal_variables() -> None:
    env = child_environment({"APPDATA": r"C:\Users\x\AppData\Roaming", "PATH": "p", "GROQ_API_KEY": "gsk_x",
                             "LITELLM_API_KEY": "k", "MY_SECRET": "s", "GITHUB_TOKEN": "t", "USERPROFILE": "u"})
    assert env == {"APPDATA": r"C:\Users\x\AppData\Roaming", "PATH": "p", "USERPROFILE": "u"}


def test_child_environment_drops_electron_host_variables() -> None:
    # SAM started from a VS Code terminal/extension inherits these; Electron
    # apps (VS Code itself, Cursor...) started with them exit at once.
    env = child_environment({"PATH": "p", "ELECTRON_RUN_AS_NODE": "1", "VSCODE_ESM_ENTRYPOINT": "x",
                             "VSCODE_IPC_HOOK": "y", "ELECTRON_ENABLE_LOGGING": "1"})
    assert env == {"PATH": "p", "ELECTRON_ENABLE_LOGGING": "1"}


class FakeProc:
    def __init__(self, finish_after: int | None) -> None:
        self.pid = 4321
        self.returncode: int | None = None
        self.polls = 0
        self.finish_after = finish_after
        self.killed = False

    def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
        self.polls += 1
        if self.killed or (self.finish_after is not None and self.polls >= self.finish_after):
            self.returncode = 1 if self.killed else 0
            return ("سڵاو\n".encode("utf-8"), b"")
        raise subprocess.TimeoutExpired("powershell", timeout or 0)

    def kill(self) -> None:
        self.killed = True


def test_timeout_kills_the_tree(monkeypatch) -> None:
    import sam.hands.shell as shell

    killed: list[int] = []
    monkeypatch.setattr(shell, "_kill_tree", killed.append)
    proc = FakeProc(finish_after=None)
    result = run_sync("Start-Sleep 99", timeout_s=1.0, popen=lambda *a, **k: proc, exe="pwsh")
    assert result.timed_out and killed == [4321] and proc.killed


def test_cancel_kills_the_tree(monkeypatch) -> None:
    import sam.hands.shell as shell

    killed: list[int] = []
    monkeypatch.setattr(shell, "_kill_tree", killed.append)
    stop = threading.Event()
    stop.set()
    result = run_sync("Start-Sleep 99", timeout_s=60, cancel=stop, popen=lambda *a, **k: FakeProc(None), exe="pwsh")
    assert result.cancelled and killed == [4321]


def test_command_line_is_hidden_and_profile_free() -> None:
    seen: dict[str, Any] = {}

    def popen(argv: list[str], **kwargs: Any) -> FakeProc:
        seen["argv"], seen["kwargs"] = argv, kwargs
        return FakeProc(finish_after=1)
    result = run_sync("Get-Date", popen=popen, exe="pwsh")
    assert result.exit_code == 0 and result.stdout == "سڵاو"
    assert seen["argv"][:5] == ["pwsh", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command"]
    assert seen["argv"][5].endswith("Get-Date") and "UTF8Encoding" in seen["argv"][5]
    assert seen["kwargs"]["creationflags"] & 0x08000000  # CREATE_NO_WINDOW


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell")
async def test_real_powershell_returns_sorani_utf8() -> None:
    # Double quotes inside the command must survive the Windows command line.
    result = await run_powershell("Write-Output ('سڵاو' + ' ' + (2+3)); Write-Output 'say \"hi\" to SAM'",
                                  timeout_s=60)
    assert result.exit_code == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == "سڵاو 5"
    assert lines[1] == 'say "hi" to SAM'
    assert not os.environ.get("NEVER_SET_VAR")
