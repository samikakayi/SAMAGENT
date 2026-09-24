"""sam.omniroute: start the gateway only when installed and down, the way v1
learned to (its folder, SKIP_REPO_ENV, no DETACHED_PROCESS). Nothing here
starts the real OmniRoute: spawns are recorded, except the regression tests
that run a tiny local script with the exact flags."""

from __future__ import annotations

import http.server
import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest

from sam import omniroute


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch, tmp_path):
    omniroute._reset_for_tests()
    monkeypatch.setattr(omniroute, "omniroute_home", lambda: tmp_path / "omniroute-home")
    yield
    omniroute._reset_for_tests()


@pytest.fixture
def spawned(monkeypatch):
    calls: list[dict] = []

    def fake_spawn(command, cwd, env, log_path):
        calls.append({"command": list(command), "cwd": Path(cwd), "env": env, "log": Path(log_path)})
        return 4242
    monkeypatch.setattr(omniroute, "_spawn", fake_spawn)
    return calls


@pytest.fixture
def installed(tmp_path, monkeypatch) -> Path:
    home = tmp_path / "omniroute-home"
    home.mkdir()
    (home / ".env").write_text("PORT=20128\n", encoding="utf-8")
    command = tmp_path / "omniroute.cmd"
    command.write_text("@echo off\n", encoding="utf-8")
    monkeypatch.setattr(omniroute.shutil, "which", lambda name: str(command))
    return command


def test_left_alone_when_not_set_up(spawned):
    assert omniroute.start_if_needed(probe=lambda: False) == "not_installed"
    assert spawned == []


def test_a_running_gateway_is_reused_not_started_twice(installed, spawned):
    assert omniroute.start_if_needed(probe=lambda: True) == "already_running"
    assert spawned == []


def test_starts_from_its_own_folder_with_skip_repo_env(installed, spawned, tmp_path):
    state = omniroute.start_if_needed(probe=lambda: False, log_dir=tmp_path / "logs")

    assert state == "starting"
    (call,) = spawned
    assert call["command"] == [str(installed), "serve", "--no-open", "--no-tray"]
    assert call["cwd"] == tmp_path / "omniroute-home", "run elsewhere it would read SAM's .env"
    assert call["env"]["OMNIROUTE_CLI_SKIP_REPO_ENV"] == "1"
    assert call["log"] == tmp_path / "logs" / "omniroute.log"


def test_flags_hide_the_window_and_never_detach():
    assert omniroute.BACKGROUND_FLAGS == omniroute.CREATE_NO_WINDOW | omniroute.CREATE_NEW_PROCESS_GROUP
    assert not omniroute.BACKGROUND_FLAGS & omniroute.DETACHED_PROCESS


def test_the_child_gets_no_sam_dotenv_values(installed, spawned, monkeypatch):
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    omniroute.start_if_needed(probe=lambda: False)

    assert "LITELLM_API_KEY" not in spawned[0]["env"]   # Config never copies .env into os.environ


def test_early_start_then_app_start_spawns_once(installed, spawned):
    assert omniroute.start_if_needed(probe=lambda: False) == "starting"
    assert omniroute.start_if_needed(probe=lambda: False) == "starting"

    assert len(spawned) == 1


def test_missing_command_is_reported_not_fatal(tmp_path, monkeypatch, spawned):
    home = tmp_path / "omniroute-home"
    home.mkdir()
    (home / ".env").write_text("", encoding="utf-8")
    monkeypatch.setattr(omniroute.shutil, "which", lambda name: None)
    monkeypatch.setenv("APPDATA", str(tmp_path / "no-appdata"))

    assert omniroute.start_if_needed(probe=lambda: False) == "not_installed"
    assert spawned == []


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        self.send_response(401)   # OmniRoute's auth endpoint refusing still means "up"
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def local_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_is_running_counts_any_http_answer(local_server):
    assert omniroute.is_running(local_server) is True
    assert omniroute.is_running(f"http://127.0.0.1:{_closed_port()}", timeout=1) is False


def test_port_probe(local_server):
    port = int(local_server.rsplit(":", 1)[1])
    assert omniroute.port_open("127.0.0.1", port) is True
    assert omniroute.port_open("127.0.0.1", _closed_port()) is False


class _StatusApp:
    """Just enough App for ensure_running."""

    def __init__(self, tmp_path: Path, base_url: str, has_key: bool = True) -> None:
        self.statuses: list[tuple[str, str]] = []
        self.config = type("C", (), {"log_dir": tmp_path / "logs",
                                     "get": staticmethod(lambda key, default=None: base_url)})()
        self.secrets = type("S", (), {"has": staticmethod(lambda name: has_key and name == "litellm_api_key")})()

    def publish_status(self, component, state, detail=""):
        self.statuses.append((component, state))


async def test_ensure_running_reports_already_running(installed, spawned, tmp_path, local_server):
    app = _StatusApp(tmp_path, local_server + "/v1")
    assert await omniroute.ensure_running(app) == "already_running"
    assert spawned == [] and app.statuses == [("omniroute", "ok")]


async def test_ensure_running_waits_for_the_started_gateway(installed, spawned, tmp_path, monkeypatch):
    app = _StatusApp(tmp_path, "http://127.0.0.1:20128/v1")
    answers = iter([False, False, True])
    monkeypatch.setattr(omniroute, "is_running", lambda url, timeout=2.0: next(answers))
    monkeypatch.setattr(omniroute, "port_open", lambda host=None, port=None, timeout=0.3: False)

    assert await omniroute.ensure_running(app, wait_s=5, poll_s=0.01) == "started"
    assert len(spawned) == 1
    assert app.statuses[-1] == ("omniroute", "ok")


async def test_ensure_running_gives_up_and_says_so(installed, spawned, tmp_path, monkeypatch):
    app = _StatusApp(tmp_path, "http://127.0.0.1:20128/v1")
    monkeypatch.setattr(omniroute, "is_running", lambda url, timeout=2.0: False)
    monkeypatch.setattr(omniroute, "port_open", lambda host=None, port=None, timeout=0.3: False)

    assert await omniroute.ensure_running(app, wait_s=0.05, poll_s=0.01) == "failed"
    assert app.statuses[-1] == ("omniroute", "down")


async def test_without_the_gateway_key_nothing_is_probed_or_started(installed, spawned, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("must not touch the network")
    monkeypatch.setattr(omniroute, "is_running", forbidden)
    monkeypatch.setattr(omniroute, "port_open", forbidden)
    app = _StatusApp(tmp_path, "http://127.0.0.1:20128/v1", has_key=False)

    assert await omniroute.ensure_running(app) == "not_installed"
    assert spawned == [] and app.statuses == [("omniroute", "unconfigured")]


async def test_a_test_app_never_reaches_the_real_gateway(make_app, spawned, monkeypatch):
    """Every package's tests call App.start() on a temp home without keys."""
    def forbidden(*args, **kwargs):
        raise AssertionError("must not touch the network")
    monkeypatch.setattr(omniroute, "is_running", forbidden)
    monkeypatch.setattr(omniroute, "port_open", forbidden)

    assert await omniroute.ensure_running(make_app()) == "not_installed"
    assert spawned == []


async def test_ensure_running_when_not_installed(spawned, tmp_path):
    app = _StatusApp(tmp_path, "http://127.0.0.1:20128/v1")
    assert await omniroute.ensure_running(app) == "not_installed"
    assert app.statuses == [("omniroute", "unconfigured")]


async def test_a_remote_gateway_is_never_started_here(installed, spawned, tmp_path, monkeypatch):
    app = _StatusApp(tmp_path, "http://10.0.0.5:20128/v1")
    monkeypatch.setattr(omniroute, "is_running", lambda url, timeout=2.0: False)

    assert await omniroute.ensure_running(app) == "failed"
    assert spawned == []


def test_app_start_picks_the_module_up(make_app, monkeypatch):
    """App._start_background_services spawns ensure_running when sam.omniroute exists."""
    import asyncio

    calls = []

    async def fake(app):
        calls.append(app)
        return "already_running"
    monkeypatch.setattr(omniroute, "ensure_running", fake)
    app = make_app()

    async def go():
        await app.start()
        await asyncio.sleep(0)
        await app.stop()
    asyncio.run(go())
    assert calls == [app]


# --- real processes: the v1 regression ---------------------------------------------------

def _wait_for(path: Path, seconds: float = 60.0, needle: str = "") -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            if path.is_file() and needle in path.read_text(encoding="utf-8", errors="replace"):
                return True
        except OSError:
            pass  # still being written by the child
        time.sleep(0.2)
    return False


@pytest.mark.skipif(shutil.which("powershell") is None, reason="needs Windows PowerShell")
def test_a_hidden_powershell_script_really_runs_with_these_flags(tmp_path):
    """v1's first install opened a window onto a SAM that never started: with
    DETACHED_PROCESS, powershell.exe running a script exited at once with code
    0 and wrote nothing. Run a real script with the flags SAM uses."""
    marker = tmp_path / "ran.txt"
    script = tmp_path / "probe.ps1"
    script.write_text(f"Set-Content -LiteralPath '{marker}' -Value 'ran'\n", encoding="utf-8")
    process = subprocess.Popen(  # noqa: S603
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-WindowStyle", "Hidden", "-File", str(script)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=omniroute.BACKGROUND_FLAGS)
    process.wait(timeout=90)

    assert marker.is_file(), "the hidden script never ran"


@pytest.mark.skipif(os.name != "nt", reason="Windows .cmd shim")
def test_a_cmd_shim_like_omniroutes_really_runs_through_spawn(tmp_path):
    """OmniRoute's entry point is an npm ``.cmd`` shim: run a real one through
    the same _spawn (flags, cwd, env, log file) and check it ran in its folder
    with SKIP_REPO_ENV set and its output landed in the log."""
    home = tmp_path / "gateway-home"
    home.mkdir()
    marker = home / "ran.txt"
    shim = tmp_path / "fake-omniroute.cmd"
    shim.write_text("@echo off\r\n"
                    "echo started %1 %2 %3\r\n"
                    # redirect first: "...%VAR%> file" with VAR=1 would read as "1>"
                    ">ran.txt echo %CD%^|%OMNIROUTE_CLI_SKIP_REPO_ENV%\r\n", encoding="ascii")
    log_path = tmp_path / "logs" / "omniroute.log"
    env = {**os.environ, "OMNIROUTE_CLI_SKIP_REPO_ENV": "1"}
    pid = omniroute._spawn([str(shim), "serve", "--no-open", "--no-tray"], home, env, log_path)

    assert pid > 0
    assert _wait_for(marker, needle="|"), "the hidden .cmd never ran"
    cwd, skip = marker.read_text(encoding="ascii").strip().split("|")
    assert Path(cwd) == home and skip == "1"
    assert _wait_for(log_path, 15, "started serve --no-open --no-tray"), "child output did not reach omniroute.log"
