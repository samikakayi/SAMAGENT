"""The desktop program: shortcuts start SAM quietly and open it in its own window.

Nothing here starts SAM, OmniRoute or a browser: every process the launcher
would spawn is recorded instead.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "desktop" / "sam_desktop.pyw"


@pytest.fixture()
def desktop(monkeypatch, tmp_path):
    loader = importlib.machinery.SourceFileLoader("sam_desktop_under_test", str(LAUNCHER))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    monkeypatch.setattr(module, "APP_DIR", tmp_path / "SAM")
    monkeypatch.setattr(module, "LOG_PATH", tmp_path / "SAM" / "sam-desktop.log")
    monkeypatch.setattr(module, "OMNIROUTE_HOME", tmp_path / "omniroute-home")
    spawned: list[list[str]] = []
    monkeypatch.setattr(module, "_spawn", lambda command, cwd, env=None: spawned.append(list(command)))
    module.spawned = spawned
    return module


def test_the_icon_is_written_as_a_windows_icon(desktop, tmp_path):
    target = tmp_path / "sam.ico"
    desktop.write_icon(target)

    assert target.read_bytes()[:4] == b"\x00\x00\x01\x00", "not an .ico file"
    image = desktop.make_icon_image(64)
    assert image.size == (64, 64) and image.mode == "RGBA"


def test_sam_is_started_through_start_ps1_hidden_and_without_a_browser(desktop, monkeypatch):
    monkeypatch.setattr(desktop, "sam_running", lambda: False)
    desktop.start_sam()

    (command,) = desktop.spawned
    assert command[0].lower() == "powershell.exe"
    assert str(ROOT / "start.ps1") in command
    assert ["-Port", str(desktop.PORT)] == command[command.index("-Port"):command.index("-Port") + 2]
    assert "-NoBrowser" in command and "Hidden" in command


def test_a_running_sam_is_reused_not_started_twice(desktop, monkeypatch):
    monkeypatch.setattr(desktop, "sam_running", lambda: True)
    desktop.start_sam()

    assert desktop.spawned == []


def test_omniroute_is_left_alone_when_it_is_not_set_up(desktop):
    desktop.start_omniroute()

    assert desktop.spawned == []


def test_omniroute_starts_from_its_own_folder_when_set_up(desktop, monkeypatch, tmp_path):
    home = tmp_path / "omniroute-home"
    home.mkdir()
    (home / ".env").write_text("OMNIROUTE_SERVER_HOST=127.0.0.1\n", encoding="utf-8")
    command_path = tmp_path / "omniroute.cmd"
    command_path.write_text("@echo off\n", encoding="utf-8")
    monkeypatch.setattr(desktop, "omniroute_running", lambda: False)
    monkeypatch.setattr(desktop.shutil, "which", lambda name: str(command_path))
    calls = []
    monkeypatch.setattr(desktop, "_spawn", lambda command, cwd, env=None: calls.append((command, cwd, env)))
    desktop.start_omniroute()

    (command, cwd, env) = calls[0]
    assert command == [str(command_path), "serve", "--no-open", "--no-tray"]
    assert Path(cwd) == home, "run elsewhere it would read SAM's .env"
    assert env["OMNIROUTE_CLI_SKIP_REPO_ENV"] == "1"


def test_the_window_is_an_app_window_with_its_own_profile(desktop, monkeypatch):
    opened = []
    monkeypatch.setattr(desktop, "_edge", lambda: r"C:\Edge\msedge.exe")
    monkeypatch.setattr(desktop.subprocess, "Popen", lambda args, **kw: opened.append(args))
    desktop.open_window()

    (args,) = opened
    assert f"--app={desktop.URL}" in args
    assert f"--user-data-dir={desktop.APP_DIR / 'window'}" in args


def test_without_edge_the_default_browser_opens_sam(desktop, monkeypatch):
    urls = []
    monkeypatch.setattr(desktop, "_edge", lambda: None)
    monkeypatch.setattr(desktop.webbrowser, "open", urls.append)
    desktop.open_window()

    assert urls == [desktop.URL]


def test_background_start_opens_no_window(desktop, monkeypatch):
    windows = []
    monkeypatch.setattr(desktop, "sam_running", lambda: True)
    monkeypatch.setattr(desktop, "open_window", lambda: windows.append(1))
    assert desktop.ensure_running(show_window=False) is True

    assert windows == []


@pytest.mark.skipif(shutil.which("powershell") is None, reason="needs Windows PowerShell")
@pytest.mark.parametrize("script", ["install-desktop.ps1", "uninstall-desktop.ps1"])
def test_the_install_scripts_parse(script):
    path = ROOT / script
    check = (
        "$errors = $null; "
        f"[System.Management.Automation.Language.Parser]::ParseFile('{path}', [ref]$null, [ref]$errors) | Out-Null; "
        "if ($errors.Count) { $errors | ForEach-Object { $_.Message }; exit 1 }"
    )
    result = subprocess.run(["powershell", "-NoProfile", "-Command", check], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
