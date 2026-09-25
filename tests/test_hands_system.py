"""System status and toast helpers (read-only; nothing changes the PC)."""

from __future__ import annotations

import sys

import pytest

from sam.hands import system


def test_toast_script_escapes_xml_and_quotes() -> None:
    script = system.toast_script("ئاگاداری <زێڕ>", "نرخ گەیشتە 2700 & it's up")
    assert "&lt;زێڕ&gt;" in script and "&amp;" in script
    assert "it''s up" in script  # single quotes doubled inside the PowerShell literal
    assert "ToastNotificationManager" in script


def test_notify_runs_hidden_windows_powershell() -> None:
    seen: dict = {}

    class Done:
        returncode = 0

    def runner(argv, **kwargs):
        seen["argv"], seen["kwargs"] = argv, kwargs
        return Done()
    assert system.notify("SAM", "test", runner=runner)
    assert seen["argv"][0] == "powershell.exe" and "-NoProfile" in seen["argv"]
    assert seen["kwargs"]["creationflags"] == 0x08000000


def test_info_survives_a_failing_part(monkeypatch) -> None:
    monkeypatch.setattr(system, "get_volume", lambda: (_ for _ in ()).throw(OSError("no audio device")))
    data = system.info("Asia/Baghdad")
    assert data["volume"] == {"error": "OSError"}
    assert data["timezone"] == "Asia/Baghdad" and len(data["time"]) == 5


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_read_only_status_on_this_pc() -> None:
    assert 0 <= system.memory()["used_percent"] <= 100
    assert any(d["drive"].startswith("C") for d in system.disks())
    assert system.uptime_hours() >= 0
    assert "present" in system.battery()
