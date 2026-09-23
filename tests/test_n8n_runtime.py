"""The managed n8n runtime: one installation, and only one SAM will touch.

A "start this binary" endpoint on a local agent is remote code execution with
a friendly name, so most of these tests are about what the feature refuses.
It takes no path, no port and no command from anybody; it starts one fixed
argument vector; and before it kills anything it insists on evidence that the
process is the installation it made.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sam_backend.db import Database
from sam_backend.n8n_runtime import (
    HOST,
    RECORD_SETTING_KEY,
    ManagedN8nRuntime,
    RuntimeState,
)


def install(tmp_path: Path, version: str = "2.40.5") -> tuple[Path, Path]:
    """A directory shaped like a real npm install of n8n."""
    runtime = tmp_path / "n8n-runtime"
    package = runtime / "node_modules" / "n8n"
    (package / "bin").mkdir(parents=True, exist_ok=True)
    (package / "bin" / "n8n").write_text("#!/usr/bin/env node\n", encoding="utf-8")
    (package / "package.json").write_text(json.dumps({"name": "n8n", "version": version}),
                                          encoding="utf-8")
    data = tmp_path / "n8n-data"
    data.mkdir(exist_ok=True)
    return runtime, data


def runtime(tmp_path: Path, *, installed: bool = True, port: int = 65123) -> ManagedN8nRuntime:
    if installed:
        runtime_dir, data_dir = install(tmp_path)
    else:
        runtime_dir, data_dir = tmp_path / "absent", tmp_path / "n8n-data"
    return ManagedN8nRuntime(Database(tmp_path / "runtime.db"),
                             runtime_dir=runtime_dir, data_dir=data_dir, port=port)


# --- what is installed ---------------------------------------------------------


def test_an_absent_runtime_is_reported_not_invented(tmp_path):
    status = runtime(tmp_path, installed=False).status()

    assert status.state is RuntimeState.NOT_INSTALLED
    assert "No managed n8n is installed" in status.detail
    assert status.record.pid is None


def test_starting_something_that_is_not_installed_does_nothing(tmp_path):
    engine = runtime(tmp_path, installed=False)

    status = engine.start()

    assert status.state is RuntimeState.NOT_INSTALLED


def test_an_installed_stopped_runtime_reports_its_version(tmp_path):
    status = runtime(tmp_path).status()

    assert status.state is RuntimeState.STOPPED
    assert status.record.version == "2.40.5"
    assert status.record.runtime_path.endswith("n8n-runtime")


def test_the_url_is_always_loopback(tmp_path):
    engine = runtime(tmp_path, port=65123)

    assert engine.url == "http://127.0.0.1:65123"
    assert HOST == "127.0.0.1"


def test_the_launch_environment_never_binds_beyond_loopback(tmp_path):
    """The one setting that would expose n8n to the LAN is pinned here."""
    environment = runtime(tmp_path)._environment()

    assert environment["N8N_LISTEN_ADDRESS"] == "127.0.0.1"
    assert environment["N8N_HOST"] == "127.0.0.1"
    assert "0.0.0.0" not in json.dumps(environment)
    assert environment["N8N_DIAGNOSTICS_ENABLED"] == "false"


def test_the_data_directory_is_outside_the_repository(tmp_path):
    engine = runtime(tmp_path)

    assert engine._environment()["N8N_USER_FOLDER"] == str(engine.data_dir)
    assert "SAM-Agent" not in str(engine.data_dir) or str(engine.data_dir).startswith(str(tmp_path))


# --- ownership is what permits a kill ------------------------------------------


def test_a_pid_alone_is_not_ownership(tmp_path):
    """PIDs get reused. The command line has to name this runtime."""
    engine = runtime(tmp_path)

    assert engine._owned_process(os.getpid()) is None, "this test process is not n8n"
    assert engine._owned_process(None) is None
    assert engine._owned_process(999_999_999) is None


def test_a_process_whose_command_names_the_entrypoint_is_ours(tmp_path):
    engine = runtime(tmp_path)
    # A real child process whose command line contains n8n's own bin script,
    # which is exactly the evidence `stop` requires.
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", str(engine.entrypoint)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert engine._owned_process(child.pid) is not None
    finally:
        child.kill()
        child.wait(timeout=10)


def test_naming_the_runtime_directory_alone_is_not_ownership(tmp_path):
    """Otherwise a broad SAM_N8N_RUNTIME_DIR would make unrelated processes killable.

    Ownership decides what may be terminated, so it matches the full path to
    n8n's bin script rather than a directory substring an operator chose.
    """
    engine = runtime(tmp_path)
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", str(engine.runtime_dir)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert engine._owned_process(child.pid) is None
    finally:
        child.kill()
        child.wait(timeout=10)


def test_stopping_when_nothing_is_running_is_not_an_error(tmp_path):
    status = runtime(tmp_path).stop()

    assert status.state is RuntimeState.STOPPED
    assert "not running" in status.detail


def test_sam_refuses_to_stop_a_process_it_cannot_prove_is_its_own(tmp_path):
    """The refusal is the feature: SAM stops what it started, not what it found."""
    engine = runtime(tmp_path)

    class Foreign:
        pid = os.getpid()

    engine._port_owner = lambda: Foreign()
    engine.healthy = lambda timeout=4.0: True

    status = engine.stop()

    assert status.state is RuntimeState.UNKNOWN_PROCESS
    assert "will not stop it" in status.detail


def test_a_recorded_pid_that_is_now_someone_else_is_not_killed(tmp_path):
    """A reused PID must not become permission to kill a stranger."""
    engine = runtime(tmp_path)
    engine.database.update_settings({RECORD_SETTING_KEY: {"pid": os.getpid()}})
    engine._port_owner = lambda: None

    status = engine.stop()

    assert status.state is RuntimeState.STOPPED
    assert status.record.pid is None, "the stale pid is forgotten, not acted on"


# --- port conflicts -------------------------------------------------------------


def test_a_port_held_by_someone_else_is_a_conflict_not_a_start(tmp_path):
    engine = runtime(tmp_path)

    class Foreign:
        pid = os.getpid()

    engine._port_owner = lambda: Foreign()
    engine.healthy = lambda timeout=4.0: False

    status = engine.status()
    assert status.state is RuntimeState.PORT_CONFLICT

    # And start refuses rather than fighting for the socket.
    assert engine.start().state is RuntimeState.PORT_CONFLICT


def test_something_unidentifiable_answering_the_port_is_not_claimed(tmp_path):
    engine = runtime(tmp_path)

    class Foreign:
        pid = os.getpid()

    engine._port_owner = lambda: Foreign()
    engine.healthy = lambda timeout=4.0: True

    status = engine.status()

    assert status.state is RuntimeState.UNKNOWN_PROCESS
    assert status.healthy is True, "it is answering -- SAM just will not claim it"


def test_a_healthy_managed_process_is_running(tmp_path):
    engine = runtime(tmp_path)
    engine.database.update_settings({RECORD_SETTING_KEY: {"pid": 4242}})
    engine.healthy = lambda timeout=4.0: True
    engine._owned_process = lambda pid: object() if pid == 4242 else None

    status = engine.status()

    assert status.state is RuntimeState.RUNNING
    assert status.healthy is True
    assert status.url == engine.url


def test_an_alive_but_silent_process_is_unhealthy_not_running(tmp_path):
    engine = runtime(tmp_path)
    engine.database.update_settings({RECORD_SETTING_KEY: {"pid": 4242}})
    engine.healthy = lambda timeout=4.0: False
    engine._owned_process = lambda pid: object() if pid == 4242 else None

    assert engine.status().state is RuntimeState.UNHEALTHY


def test_starting_something_already_running_is_a_no_op(tmp_path):
    engine = runtime(tmp_path)
    engine.database.update_settings({RECORD_SETTING_KEY: {"pid": 4242}})
    engine.healthy = lambda timeout=4.0: True
    engine._owned_process = lambda pid: object() if pid == 4242 else None

    assert engine.start().state is RuntimeState.RUNNING


# --- the record is non-secret ----------------------------------------------------


def test_the_persisted_record_holds_no_credential(tmp_path):
    engine = runtime(tmp_path)
    engine._write_record(pid=1234, last_started_at="2026-09-23T00:00:00+00:00")

    stored = engine.database.get_settings()[RECORD_SETTING_KEY]

    assert set(stored) == {"pid", "last_started_at", "last_stopped_at"}
    assert engine.status().as_dict().keys() >= {"state", "version", "port", "host", "url"}
    assert "api_key" not in json.dumps(engine.status().as_dict())
    assert "password" not in json.dumps(engine.status().as_dict())


# --- the API surface takes nothing --------------------------------------------------


def test_the_runtime_routes_accept_no_path_or_command(client):
    """There is no field to inject into; that is the whole design."""
    from sam_backend.api.n8n_runtime import RuntimeActionRequest

    assert set(RuntimeActionRequest.model_fields) == {"approval_id"}

    smuggled = RuntimeActionRequest.model_validate({
        "approval_id": None, "command": "calc.exe", "runtime_path": "C:/evil", "port": 1,
    })
    assert smuggled.model_dump() == {"approval_id": None}


def test_status_is_readable_without_approval(client):
    response = client.get("/api/n8n/runtime")

    assert response.status_code == 200
    assert response.json()["state"] in {state.value for state in RuntimeState}


@pytest.mark.parametrize("action", ["start", "stop"])
def test_starting_and_stopping_require_an_approval(client, action):
    response = client.post(f"/api/n8n/runtime/{action}", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["approval_required"] is True
    assert payload["risk_level"] == "medium"
    assert payload["approval_id"]


def test_an_approval_for_starting_cannot_be_used_to_stop(client):
    granted = client.post("/api/n8n/runtime/start", json={}).json()

    refused = client.post("/api/n8n/runtime/stop",
                          json={"approval_id": granted["approval_id"]})

    assert refused.status_code == 409


def test_a_workflow_import_approval_cannot_start_a_server(client, app):
    """Approvals are bound to their action, across features as well as within one."""
    from sam_backend.workflows import prepare
    from sam_backend.workflows.models import WorkflowProvenance

    artifact = prepare(
        {"name": "x", "nodes": [{"name": "S", "type": "n8n-nodes-base.manualTrigger",
                                 "parameters": {}}], "connections": {}},
        WorkflowProvenance(source="library", source_repository="test/library"))
    app.state.workflows.remember_artifact(artifact)
    granted = client.post("/api/workflows/import",
                          json={"workflow_sha256": artifact.sha256}).json()

    refused = client.post("/api/n8n/runtime/start",
                          json={"approval_id": granted["approval_id"]})

    assert refused.status_code == 409


def test_starting_while_an_owned_process_is_still_coming_up_does_not_spawn_a_second(tmp_path):
    """A duplicate would die on the taken port and report the healthy one as STOPPED."""
    engine = runtime(tmp_path)
    engine.database.update_settings({RECORD_SETTING_KEY: {"pid": 4242}})
    engine._owned_process = lambda pid: object() if pid == 4242 else None
    engine._port_owner = lambda: None
    spawned: list[str] = []

    import subprocess as sp
    original = sp.Popen
    sp.Popen = lambda *a, **k: spawned.append("launched") or original(*a, **k)
    answers = iter([False, True])
    engine.healthy = lambda timeout=4.0: next(answers, True)
    try:
        status = engine.start(timeout=20)
    finally:
        sp.Popen = original

    assert spawned == [], "no second process was launched"
    assert status.state is RuntimeState.RUNNING
    assert "already starting" in status.detail


def test_readiness_means_the_public_api_not_merely_healthz(tmp_path):
    """n8n answers /healthz while its API is still mounting; SAM needs the API."""
    import httpx

    engine = runtime(tmp_path)
    seen: list[str] = []

    def handler(request):
        seen.append(request.url.path)
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"status": "ok"}, request=request)
        return httpx.Response(404, request=request)

    import sam_backend.n8n_runtime as module
    original = module.httpx.Client
    module.httpx.Client = lambda **kwargs: original(transport=httpx.MockTransport(handler))
    try:
        assert engine.healthy() is False, "healthz alone is not readiness"
    finally:
        module.httpx.Client = original
    assert "/api/v1/workflows" in seen


# --- SAM's own instance, while it is still coming up ----------------------------
#
# n8n answers /healthz well before it mounts its public API, so there is a real
# window where the managed process is alive and `healthy()` is still False. If
# the recorded pid does not match in that window -- a cleared record, a restart,
# a launch whose pid was never written -- SAM looked at the port, saw a
# listener, and called its own instance somebody else's.


def test_its_own_starting_instance_is_not_called_a_port_conflict(tmp_path):
    """The healthy path reattaches to an owned listener; this path forgot to."""
    engine = runtime(tmp_path)
    engine.healthy = lambda timeout=4.0: False  # alive, API not mounted yet

    class Ours:
        pid = 4242

    engine._port_owner = lambda: Ours()
    # Nothing recorded -- but the listener is demonstrably the managed n8n.
    engine._owned_process = lambda pid: object() if pid == 4242 else None

    status = engine.status()

    assert status.state is RuntimeState.UNHEALTHY, "it is SAM's own n8n, still starting"
    assert status.state is not RuntimeState.PORT_CONFLICT
    assert status.record.pid == 4242, "and the record reattaches to it"


def test_starting_while_its_own_instance_comes_up_waits_instead_of_refusing(tmp_path):
    """PORT_CONFLICT makes start() refuse, so the false verdict blocked the feature."""
    engine = runtime(tmp_path)

    class Ours:
        pid = 4242

    engine._port_owner = lambda: Ours()
    engine._owned_process = lambda pid: object() if pid == 4242 else None
    answers = iter([False, False, True])
    engine.healthy = lambda timeout=4.0: next(answers, True)

    status = engine.start(timeout=20)

    assert status.state is RuntimeState.RUNNING
    assert "already starting" in status.detail


def test_a_listener_that_is_not_ours_is_still_a_port_conflict(tmp_path):
    """The refusal has to survive: a stranger on the port is still a stranger."""
    engine = runtime(tmp_path)
    engine.healthy = lambda timeout=4.0: False

    class Foreign:
        pid = os.getpid()

    engine._port_owner = lambda: Foreign()

    status = engine.status()

    assert status.state is RuntimeState.PORT_CONFLICT
    assert status.record.pid is None, "SAM does not adopt a pid it cannot vouch for"
    assert engine.start().state is RuntimeState.PORT_CONFLICT
