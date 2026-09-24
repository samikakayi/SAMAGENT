"""scripts/*.ps1: they parse, stay ASCII (Windows PowerShell 5.1 reads BOM-less
files as ANSI), the installer's dry run changes nothing, and a real run into
temp folders makes exactly the three shortcuts (the real Desktop, Start menu
and Startup folders are never touched here)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.launcher_helpers import ROOT

SCRIPTS = ROOT / "scripts"
POWERSHELL = shutil.which("powershell")
needs_powershell = pytest.mark.skipif(POWERSHELL is None or os.name != "nt", reason="needs Windows PowerShell")


def _ps(*args: str, timeout: float = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", *args],
                          capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")


@pytest.mark.parametrize("script", ["install.ps1", "uninstall.ps1", "dev.ps1"])
def test_scripts_are_ascii(script):
    data = (SCRIPTS / script).read_bytes()
    assert all(b < 128 for b in data), f"{script} has non-ASCII bytes; PowerShell 5.1 would misread them"


@needs_powershell
@pytest.mark.parametrize("script", ["install.ps1", "uninstall.ps1", "dev.ps1"])
def test_scripts_parse(script):
    path = SCRIPTS / script
    check = ("$errors = $null; "
             f"[System.Management.Automation.Language.Parser]::ParseFile('{path}', [ref]$null, [ref]$errors) | Out-Null; "
             "if ($errors.Count) { $errors | ForEach-Object { $_.Message }; exit 1 }")
    result = _ps("-Command", check)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def folders(tmp_path) -> dict[str, Path]:
    dirs = {name: tmp_path / name for name in ("desktop", "programs", "startup", "home", "icons")}
    for path in dirs.values():
        path.mkdir()
    return dirs


def _install(folders: dict[str, Path], *extra: str) -> subprocess.CompletedProcess[str]:
    return _ps("-File", str(SCRIPTS / "install.ps1"), "-SamHome", str(folders["home"]),
               "-DesktopDir", str(folders["desktop"]), "-ProgramsDir", str(folders["programs"]),
               "-StartupDir", str(folders["startup"]), "-IconPath", str(folders["icons"] / "sam.ico"), *extra)


def _read_link(path: Path) -> dict[str, str]:
    """Read a .lnk through WScript.Shell in PowerShell (in-process COM from
    pytest makes faulthandler report handled access violations)."""
    import json

    result = _ps("-Command", f"$l = (New-Object -ComObject WScript.Shell).CreateShortcut('{path}'); "
                             "@{target=$l.TargetPath; args=$l.Arguments; cwd=$l.WorkingDirectory; "
                             "icon=$l.IconLocation} | ConvertTo-Json -Compress")
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@needs_powershell
def test_dry_run_changes_nothing(folders):
    result = _install(folders, "-DryRun")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "DRY RUN" in result.stdout and "would write" in result.stdout
    assert "SAM (background).lnk" in result.stdout and "--background" in result.stdout
    for path in folders.values():
        assert list(path.iterdir()) == [], f"dry run wrote into {path}"


@needs_powershell
def test_install_makes_three_shortcuts_to_pythonw_sam_pyw_and_is_idempotent(folders):
    for _ in range(2):
        result = _install(folders, "-SkipVenv", "-SkipPip", "-SkipCheck")
        assert result.returncode == 0, result.stdout + result.stderr

    assert [p.name for p in folders["desktop"].iterdir()] == ["SAM.lnk"]
    assert [p.name for p in folders["programs"].iterdir()] == ["SAM.lnk"]
    assert [p.name for p in folders["startup"].iterdir()] == ["SAM (background).lnk"]
    assert (folders["icons"] / "sam.ico").read_bytes()[:4] == b"\x00\x00\x01\x00"
    desktop = _read_link(folders["desktop"] / "SAM.lnk")
    startup = _read_link(folders["startup"] / "SAM (background).lnk")
    assert Path(desktop["target"]) == ROOT / ".venv" / "Scripts" / "pythonw.exe"
    assert desktop["args"] == f'"{ROOT / "SAM.pyw"}" --home "{folders["home"]}"'
    assert startup["args"] == desktop["args"] + " --background"
    assert Path(desktop["cwd"]) == ROOT
    assert desktop["icon"].lower().startswith(str(folders["icons"] / "sam.ico").lower())


@needs_powershell
def test_no_autostart_removes_the_startup_shortcut(folders):
    assert _install(folders, "-SkipVenv", "-SkipPip", "-SkipCheck").returncode == 0
    result = _install(folders, "-SkipVenv", "-SkipPip", "-SkipCheck", "-NoAutostart")

    assert result.returncode == 0, result.stdout + result.stderr
    assert list(folders["startup"].iterdir()) == []
    assert (folders["desktop"] / "SAM.lnk").exists()


@needs_powershell
def test_uninstall_removes_only_sam2_shortcuts(folders):
    assert _install(folders, "-SkipVenv", "-SkipPip", "-SkipCheck").returncode == 0
    # A foreign "SAM.lnk" (e.g. something else named SAM) must survive a default uninstall.
    foreign = _ps("-Command", f"$l = (New-Object -ComObject WScript.Shell).CreateShortcut('{folders['desktop'] / 'SAM.lnk'}'); "
                              "$l.TargetPath = 'C:\\Windows\\notepad.exe'; $l.Arguments = ''; $l.Save()")
    assert foreign.returncode == 0, foreign.stderr
    common = ["-File", str(SCRIPTS / "uninstall.ps1"), "-DesktopDir", str(folders["desktop"]),
              "-ProgramsDir", str(folders["programs"]), "-StartupDir", str(folders["startup"])]

    dry = _ps(*common, "-DryRun")
    assert dry.returncode == 0 and "would remove" in dry.stdout
    assert (folders["programs"] / "SAM.lnk").exists()
    result = _ps(*common)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (folders["programs"] / "SAM.lnk").exists()
    assert not (folders["startup"] / "SAM (background).lnk").exists()
    assert (folders["desktop"] / "SAM.lnk").exists() and "Kept" in result.stdout
    assert _ps(*common, "-All").returncode == 0
    assert not (folders["desktop"] / "SAM.lnk").exists()
    assert (folders["icons"] / "sam.ico").exists(), "uninstall removes shortcuts only"


@needs_powershell
def test_dev_check_runs_from_source_with_the_given_home(tmp_path):
    home = tmp_path / "home"
    (home / "data").mkdir(parents=True)
    result = _ps("-File", str(SCRIPTS / "dev.ps1"), "-Check", "-SamHome", str(home), timeout=240)

    assert result.returncode in (0, 1), result.stdout + result.stderr   # 1 = another package still failing
    assert f"SAM_HOME = {home}" in result.stdout
    assert '"keys"' in result.stdout and '"load"' in result.stdout
    assert (home / "data" / "sam2.sqlite3").exists()
