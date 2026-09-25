"""SAM.pyw: single instance, background mode, OmniRoute kick-off, logging
under pythonw. Nothing here starts SAM's UI: ``run_app`` is recorded, except
the last test, which runs the real launcher under pythonw with ``--check``."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.conftest import FAKE_GEMINI_AQ, FAKE_GROQ
from tests.launcher_helpers import LAUNCHER, ROOT, load_launcher


@pytest.fixture(autouse=True)
def restore_launcher_env(monkeypatch):
    """SAM.pyw writes these into os.environ; make monkeypatch restore them."""
    for name in ("SAM_BACKGROUND", "SAM_SHOW_PANEL", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)


@pytest.fixture
def launcher(monkeypatch, tmp_path):
    module = load_launcher()
    monkeypatch.setattr(module, "LOG_PATH", tmp_path / "SAM" / "sam.log")
    monkeypatch.setattr(module, "CRASH_LOG_PATH", tmp_path / "SAM" / "sam-crash.log")
    for handler in list(module._log.handlers):
        module._log.removeHandler(handler)
    calls: dict[str, list] = {"run_app": [], "show": [], "omniroute": [], "panel": [], "error": []}
    monkeypatch.setattr(module, "instance_running", lambda name=None: False)
    monkeypatch.setattr(module, "run_app", lambda argv: calls["run_app"].append(list(argv)) or 0)
    monkeypatch.setattr(module, "signal_show", lambda: calls["show"].append(1) or True)
    monkeypatch.setattr(module, "start_omniroute", lambda: calls["omniroute"].append(1) or "already_running")
    monkeypatch.setattr(module, "request_panel_when_ready", lambda *a, **k: calls["panel"].append(1))
    monkeypatch.setattr(module, "show_error", lambda text: calls["error"].append(text))
    module.calls = calls
    yield module
    for handler in list(module._log.handlers):
        handler.close()
        module._log.removeHandler(handler)


def _log_text(module) -> str:
    for handler in module._log.handlers:
        handler.flush()
    return module.LOG_PATH.read_text(encoding="utf-8") if module.LOG_PATH.exists() else ""


def test_a_normal_launch_runs_sam_opens_the_panel_and_starts_omniroute(launcher):
    assert launcher.launch(["--home", r"C:\SAMHOME"]) == 0

    assert launcher.calls["run_app"] == [["--home", r"C:\SAMHOME"]]
    # The UI opens the panel itself after the island's first frame (acceptance 2026-09-24:
    # waiting for the launcher's show request left ~8 s without a window).
    assert launcher.calls["omniroute"] == [1] and launcher.calls["panel"] == []
    assert os.environ["SAM_BACKGROUND"] == "0" and os.environ["SAM_SHOW_PANEL"] == "1"
    assert launcher.calls["error"] == []


def test_background_launch_starts_hidden(launcher):
    launcher.launch(["--background", "--home", "H"])

    assert launcher.calls["run_app"] == [["--home", "H"]]
    assert launcher.calls["panel"] == [], "sign-in start shows the island only"
    assert os.environ["SAM_BACKGROUND"] == "1" and os.environ["SAM_SHOW_PANEL"] == "0"


def test_second_launch_asks_the_running_sam_to_show_its_panel(launcher, monkeypatch):
    monkeypatch.setattr(launcher, "instance_running", lambda name=None: True)

    assert launcher.launch([]) == 0
    assert launcher.calls["show"] == [1]
    assert launcher.calls["run_app"] == [] and launcher.calls["omniroute"] == []


def test_second_background_launch_does_nothing(launcher, monkeypatch):
    monkeypatch.setattr(launcher, "instance_running", lambda name=None: True)

    assert launcher.launch(["--background"]) == 0
    assert launcher.calls["show"] == [] and launcher.calls["run_app"] == []


def test_restart_waits_for_the_old_instance_instead_of_exiting(launcher, monkeypatch):
    monkeypatch.setattr(launcher, "instance_running", lambda name=None: True)
    launcher.launch(["--after-pid", "1234"])

    assert launcher.calls["run_app"] == [["--after-pid", "1234"]]


def test_check_and_headless_runs_skip_panel_and_gateway(launcher):
    launcher.launch(["--check"])
    launcher.launch(["--no-ui", "--no-omniroute"])

    assert launcher.calls["run_app"] == [["--check"], ["--no-ui"]]
    assert launcher.calls["omniroute"] == [] and launcher.calls["panel"] == []


def test_a_crash_is_logged_redacted_and_shown(launcher, monkeypatch):
    def boom(argv):
        raise RuntimeError(f"provider said no for key {FAKE_GROQ}")
    monkeypatch.setattr(launcher, "run_app", boom)

    assert launcher.launch([]) == 1
    text = _log_text(launcher)
    assert "SAM crashed" in text and "RuntimeError" in text
    assert FAKE_GROQ not in text and "[REDACTED]" in text
    (message,) = launcher.calls["error"]
    assert "SAM نەیتوانی دەست پێبکات" in message and str(launcher.LOG_PATH) in message


def test_a_failing_exit_code_is_reported(launcher, monkeypatch):
    monkeypatch.setattr(launcher, "run_app", lambda argv: 3)

    assert launcher.launch([]) == 3
    assert len(launcher.calls["error"]) == 1


def test_pythonw_output_goes_to_the_log_redacted(launcher, monkeypatch):
    import faulthandler

    import logging

    enabled = []
    monkeypatch.setattr(faulthandler, "enable", lambda file=None, **kw: enabled.append(file))
    monkeypatch.setattr(logging, "raiseExceptions", logging.raiseExceptions)
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    launcher.setup_logging(launcher.LOG_PATH)
    print("hello from pythonw", FAKE_GEMINI_AQ)
    sys.stderr.write("Traceback line one\npartial")
    sys.stderr.flush()

    text = _log_text(launcher)
    assert "hello from pythonw [REDACTED]" in text and FAKE_GEMINI_AQ not in text
    assert "Traceback line one" in text and "partial" in text
    assert enabled and Path(enabled[0].name).name == "sam-crash.log"
    enabled[0].close()


def test_the_log_stream_survives_a_failing_handler(launcher):
    import logging

    class Broken(logging.Handler):
        def emit(self, record):          # like FileHandler.emit on a full disk
            try:
                raise OSError("disk full")
            except OSError:
                self.handleError(record)

        def handleError(self, record):   # logging reports handler errors to sys.stderr = the stream
            stream.write("handler failed\n")

    logger = logging.getLogger("sam.launcher.test-broken")
    logger.propagate = False
    logger.addHandler(Broken())
    stream = launcher._LogStream(logger, logging.ERROR)
    assert stream.write("line\n") == 5     # no RecursionError


def test_write_icon(launcher, tmp_path):
    target = tmp_path / "icons" / "sam.ico"
    assert launcher.launch(["--write-icon", str(target)]) == 0

    assert target.read_bytes()[:4] == b"\x00\x00\x01\x00"
    assert launcher.calls["run_app"] == []


def test_panel_request_polls_until_the_ui_listens(launcher):
    module = load_launcher("sam_launcher_panel")
    answers = iter([False, False, True])
    sent = []

    def fake_signal():
        sent.append(1)
        return next(answers)
    thread = module.request_panel_when_ready(5, poll_s=0.01, signal=fake_signal)
    thread.join(5)

    assert not thread.is_alive() and len(sent) == 3


def test_panel_request_gives_up_quietly(launcher):
    module = load_launcher("sam_launcher_panel_timeout")
    thread = module.request_panel_when_ready(0.05, poll_s=0.01, signal=lambda: False)
    thread.join(5)

    assert not thread.is_alive()


@pytest.mark.skipif(os.name != "nt", reason="Windows named mutex")
def test_instance_detection_uses_the_real_named_mutex():
    module = load_launcher("sam_launcher_mutex")
    name = f"Local\\SAM2.Test.{os.getpid()}.{time.monotonic_ns()}"
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
    assert module.instance_running(name) is False
    handle = kernel32.CreateMutexW(None, False, name)
    try:
        assert module.instance_running(name) is True
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(handle))
    assert module.instance_running(name) is False


def test_launcher_uses_the_same_mutex_as_sam_main():
    from sam import winapp

    source = LAUNCHER.read_text(encoding="utf-8")
    assert "from sam.winapp import MUTEX_NAME" in source
    assert "subprocess" not in source and "creationflags" not in source   # spawning lives in sam.omniroute
    assert winapp.MUTEX_NAME == "Local\\SAM2.SingleInstance"


@pytest.mark.skipif(os.name != "nt", reason="pythonw is Windows-only")
def test_real_pythonw_run_logs_to_localappdata_without_a_console(tmp_path):
    """Run SAM.pyw with the venv's pythonw (no stdout/stderr at all) in --check
    mode against a temp home: the status JSON and the launcher lines must land
    in %LOCALAPPDATA%\\SAM\\sam.log, and nothing may leak a key."""
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    if not pythonw.is_file():
        pytest.skip("pythonw.exe not next to the test interpreter")
    home = tmp_path / "home"
    (home / "data").mkdir(parents=True)
    env = {**os.environ, "LOCALAPPDATA": str(tmp_path / "localappdata"), "SAM_LOG_DIR": str(tmp_path / "logs"),
           "SAM_HOME": str(home), "GROQ_API_KEY": FAKE_GROQ}
    # No stdio handles at all, as from a shortcut (capturing would give pythonw real ones).
    result = subprocess.run([str(pythonw), str(LAUNCHER), "--check", "--home", str(home)], env=env,
                            cwd=str(ROOT), timeout=180)
    log_path = tmp_path / "localappdata" / "SAM" / "sam.log"

    assert log_path.is_file(), "pythonw run left no launcher log"
    text = log_path.read_text(encoding="utf-8")
    assert "launch: background=False" in text
    assert '"keys"' in text and '"groq_api_key"' in text      # --check's JSON went to the log
    assert FAKE_GROQ not in text
    assert result.returncode in (0, 1)   # 1 = some other package is still failing to load


def test_openblas_threads_default_to_two_but_respect_the_user(monkeypatch):
    load_launcher("sam_launcher_blas_default")
    assert os.environ["OPENBLAS_NUM_THREADS"] == "2"
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")
    load_launcher("sam_launcher_blas_user")
    assert os.environ["OPENBLAS_NUM_THREADS"] == "8"
