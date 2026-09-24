"""Integration: ``python -m sam`` start-up order, clean shutdown and ``--quit``.

The real packages are NOT started here (no MT5, CDP, hotkeys or network):
``App.load_packages`` / ``start`` / ``stop`` are replaced by probes, and the
UI entry point by a fake that records what had happened when it was called.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time

import pytest

import sam.__main__ as sam_main
import sam.app
import sam.ui
from sam import winapp


@pytest.fixture
def isolated_main(tmp_path, monkeypatch):
    """Run ``sam.__main__.main`` on a temp home with probe packages."""
    monkeypatch.setenv("SAM_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(winapp, "acquire_single_instance", lambda *a, **k: True)
    calls: dict = {"order": []}
    release = threading.Event()

    def load_packages(self, names=()):
        calls["order"].append("registered")
        return {}

    async def start(self):
        self.loop = asyncio.get_running_loop()
        calls["order"].append("start-begin")
        await asyncio.to_thread(release.wait, 10)
        calls["order"].append("start-end")

    async def stop(self):
        calls["order"].append("stop")

    monkeypatch.setattr(sam.app.App, "load_packages", load_packages)
    monkeypatch.setattr(sam.app.App, "start", start)
    monkeypatch.setattr(sam.app.App, "stop", stop)
    root = logging.getLogger()
    before = list(root.handlers)
    yield calls, release, tmp_path / "home"
    release.set()
    for handler in list(root.handlers):
        if handler not in before:
            root.removeHandler(handler)
            handler.close()


def test_ui_is_shown_before_app_start_finishes(isolated_main, monkeypatch):
    calls, release, home = isolated_main

    def fake_run(app, core, *, started=None):
        # The island must not wait for package start-up (voice alone took
        # 0.56-5.4 s in the launcher stage's measurements).
        deadline = time.monotonic() + 5
        while "start-begin" not in calls["order"] and time.monotonic() < deadline:
            time.sleep(0.01)
        calls["order"].append("ui")
        calls["started_arg"] = started
        calls["bus_bound"] = app.bus.loop is core.loop
        release.set()
        return 0

    monkeypatch.setattr(sam.ui, "run", fake_run)
    assert sam_main.main(["--home", str(home)]) == 0
    order = calls["order"]
    assert order.index("ui") < order.index("start-end"), order
    assert order[-1] == "stop" and order.index("start-end") < order.index("stop")
    assert isinstance(calls["started_arg"], float)
    assert calls["bus_bound"] is True


def test_quit_during_start_waits_for_start_then_stops(isolated_main, monkeypatch):
    calls, release, home = isolated_main

    def fake_run(app, core, *, started=None):
        threading.Timer(0.2, release.set).start()   # start() finishes after the UI quit
        return 0

    monkeypatch.setattr(sam.ui, "run", fake_run)
    assert sam_main.main(["--home", str(home)]) == 0
    assert calls["order"][-2:] == ["start-end", "stop"]


def test_quit_flag_is_a_no_op_when_sam_is_not_running(monkeypatch):
    monkeypatch.setattr(winapp, "instance_running", lambda *a, **k: False)
    assert sam_main.main(["--quit"]) == 0


def test_quit_flag_signals_and_waits_for_the_instance(monkeypatch):
    state = {"running": True, "signals": 0}

    def signal_quit(*_a, **_k):
        state["signals"] += 1
        state["running"] = False
        return True

    monkeypatch.setattr(winapp, "instance_running", lambda *a, **k: state["running"])
    monkeypatch.setattr(winapp, "signal_quit", signal_quit)
    assert sam_main.request_quit(timeout_s=2) == 0
    assert state["signals"] == 1


@pytest.mark.skipif(os.name != "nt", reason="named events are Windows-only")
def test_quit_event_round_trip_with_a_real_named_event():
    name = f"Local\\SAM2.Test.Quit.{os.getpid()}.{time.monotonic_ns()}"
    assert winapp.signal_quit(name) is False          # nobody listens yet
    fired = threading.Event()
    assert winapp.watch_quit_requests(fired.set, name) is not None
    assert winapp.signal_quit(name) is True
    assert fired.wait(2)


@pytest.mark.skipif(os.name != "nt", reason="named mutexes are Windows-only")
def test_instance_running_never_creates_the_mutex():
    name = f"Local\\SAM2.Test.Mutex.{os.getpid()}.{time.monotonic_ns()}"
    assert winapp.instance_running(name) is False
    assert winapp.instance_running(name) is False     # still absent: the probe did not create it
