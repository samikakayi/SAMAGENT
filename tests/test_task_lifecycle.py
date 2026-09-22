"""Lifecycle control of autonomous runs, through the real HTTP surface.

Three properties an autonomous agent must hold or it is not safe to run:
it can be stopped, its approvals resolve from wherever they are shown, and
one task is only ever driven by one driver.
"""

from __future__ import annotations

import asyncio
import collections
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sam_backend.app import create_app
from sam_backend.config import Settings
from sam_backend.models import AssistantTurn, ToolCall


class ScriptedAdapter:
    """Plans N steps, then answers each executor turn with one tool call."""

    def __init__(self, *, steps: int, tool: str, arguments, delay: float = 0.0) -> None:
        self.steps = steps
        self.tool = tool
        self.arguments = arguments
        self.delay = delay
        self.turns = 0

    async def list_models(self):
        return [{"id": "fake", "name": "fake", "provider": "ollama"}]

    async def complete(self, messages, tools, model):
        joined = "\n".join(str(item.get("content", "")) for item in messages)
        if "planning stage" in joined:
            plan = ",".join('{"text":"step %d","kind":"edit"}' % index for index in range(self.steps))
            return AssistantTurn("```json\n{\"steps\":[%s]}\n```" % plan)
        self.turns += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        arguments = self.arguments(self.turns) if callable(self.arguments) else self.arguments
        return AssistantTurn("working", [ToolCall(f"call_{self.turns}", self.tool, arguments)])


class Registry:
    def __init__(self, adapter) -> None:
        self.adapter = adapter

    def get(self, provider):
        return self.adapter


def make_client(tmp_path: Path, adapter, *, mode: str = "trusted") -> TestClient:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    settings = Settings(
        project_root=tmp_path, workspace_root=workspace, data_dir=tmp_path / "data",
        default_provider="ollama", default_model="fake", permission_mode=mode,
    )
    return TestClient(create_app(settings, Registry(adapter)))


def wait_for(client: TestClient, task_id: str, *, state: str | None = None, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    task: dict = {}
    while time.time() < deadline:
        task = client.get(f"/api/tasks/{task_id}").json()["task"]
        if task["terminal"] or (state and task["state"] == state):
            return task
        time.sleep(0.15)
    return task


def pending_approval_id(task: dict) -> str:
    return next(event["detail"]["approval_id"] for event in reversed(task["events"]) if event["kind"] == "approval")


# -- stop ----------------------------------------------------------------

def test_a_running_task_can_be_stopped_and_stops_doing_work(tmp_path: Path):
    adapter = ScriptedAdapter(
        steps=12, tool="write_file", delay=0.5,
        arguments=lambda turn: {"path": f"f{turn}.txt", "content": "x"},
    )
    with make_client(tmp_path, adapter) as client:
        task_id = client.post("/api/tasks", json={"goal": "long run"}).json()["task_id"]
        wait_for(client, task_id, state="EXECUTING")
        time.sleep(1.2)

        response = client.post(f"/api/tasks/{task_id}/cancel")
        assert response.status_code == 200 and response.json()["cancelled"] is True

        task = wait_for(client, task_id)
        assert task["state"] == "CANCELLED"
        assert task["completion_status"] == "cancelled"
        written = len(list((tmp_path / "workspace").glob("f*.txt")))
        assert 0 < written < 12, "stopped part-way, not at the end"

        # Nothing runs after the stop.
        time.sleep(1.5)
        assert len(list((tmp_path / "workspace").glob("f*.txt"))) == written
        again = client.post(f"/api/tasks/{task_id}/cancel").json()
        assert again["cancelled"] is False and "no longer running" in again["reason"]


def test_a_stop_issued_before_the_driver_starts_still_lands(tmp_path: Path):
    """The POST returns before the background driver registers its token."""
    adapter = ScriptedAdapter(steps=3, tool="write_file", delay=0.4, arguments={"path": "a.txt", "content": "x"})
    with make_client(tmp_path, adapter) as client:
        task_id = client.post("/api/tasks", json={"goal": "race"}).json()["task_id"]
        assert client.post(f"/api/tasks/{task_id}/cancel").json()["cancelled"] is True
        assert wait_for(client, task_id)["state"] == "CANCELLED"


def test_a_task_paused_for_approval_can_be_stopped_and_a_late_approve_does_nothing(tmp_path: Path):
    adapter = ScriptedAdapter(steps=1, tool="run_terminal", arguments={"command": "echo hi", "cwd": "."})
    with make_client(tmp_path, adapter, mode="guarded") as client:
        task_id = client.post("/api/tasks", json={"goal": "needs approval"}).json()["task_id"]
        task = wait_for(client, task_id, state="WAITING_FOR_APPROVAL")
        assert task["state"] == "WAITING_FOR_APPROVAL"
        approval_id = pending_approval_id(task)

        assert client.post(f"/api/tasks/{task_id}/cancel").json()["cancelled"] is True
        task = wait_for(client, task_id)
        assert task["state"] == "CANCELLED", "a paused task has no driver; Stop must finalise it itself"
        assert client.post(f"/api/tasks/{task_id}/resume").json()["resumed"] is False

        # A stale Approve click on a stopped task must not run the command.
        client.post(f"/api/approvals/{approval_id}/decision", json={"decision": "approved"})
        time.sleep(1.0)
        after = client.get(f"/api/tasks/{task_id}").json()["task"]
        assert after["state"] == "CANCELLED" and after["tool_calls"] == task["tool_calls"]


def test_stopping_an_unknown_task_is_a_404(tmp_path: Path):
    with make_client(tmp_path, ScriptedAdapter(steps=1, tool="read_file", arguments={"path": "x"})) as client:
        assert client.post("/api/tasks/task_nope/cancel").status_code == 404


# -- approvals from either panel ------------------------------------------

@pytest.mark.parametrize("via", ["approvals_panel", "autopilot_panel"])
def test_an_orchestrator_approval_resolves_from_either_panel(tmp_path: Path, via: str):
    adapter = ScriptedAdapter(steps=1, tool="run_terminal", arguments={"command": "echo hi", "cwd": "."})
    with make_client(tmp_path, adapter, mode="guarded") as client:
        task_id = client.post("/api/tasks", json={"goal": "run a command"}).json()["task_id"]
        task = wait_for(client, task_id, state="WAITING_FOR_APPROVAL")
        approval_id = pending_approval_id(task)

        if via == "approvals_panel":
            response = client.post(f"/api/approvals/{approval_id}/decision", json={"decision": "approved"})
            assert response.json()["task_id"] == task_id
        else:
            response = client.post(
                f"/api/tasks/{task_id}/approvals", json={"approval_id": approval_id, "decision": "approved"},
            )
        assert response.status_code == 200

        task = wait_for(client, task_id)
        assert task["state"] == "COMPLETED", task.get("summary")
        assert task["tool_calls"] == 1


def test_a_denial_from_the_approvals_panel_reaches_the_run(tmp_path: Path):
    adapter = ScriptedAdapter(steps=1, tool="run_terminal", arguments={"command": "echo hi", "cwd": "."})
    with make_client(tmp_path, adapter, mode="guarded") as client:
        task_id = client.post("/api/tasks", json={"goal": "run a command"}).json()["task_id"]
        approval_id = pending_approval_id(wait_for(client, task_id, state="WAITING_FOR_APPROVAL"))

        assert client.post(f"/api/approvals/{approval_id}/decision", json={"decision": "denied"}).status_code == 200

        task = wait_for(client, task_id)
        assert task["terminal"]
        assert any("denied" in observation.lower() for observation in task["observations"])


def test_a_chat_approval_still_resolves_through_the_shared_endpoint(client: TestClient, settings: Settings):
    """The dispatcher must not break the pre-existing chat path."""
    (Path(settings.workspace_root)).mkdir(parents=True, exist_ok=True)
    (Path(settings.workspace_root) / "keep.txt").write_text("old", encoding="utf-8")
    response = client.post("/api/chat", json={"message": "overwrite keep.txt new"})
    approval = response.json()["approvals"][0]

    decided = client.post(f"/api/approvals/{approval['id']}/decision", json={"decision": "approved"})

    assert decided.status_code == 200
    assert "agent_response" in decided.json()
    assert (Path(settings.workspace_root) / "keep.txt").read_text(encoding="utf-8") == "new"


# -- single driver ---------------------------------------------------------

def test_a_live_task_refuses_a_second_driver_and_is_not_corrupted(tmp_path: Path):
    adapter = ScriptedAdapter(
        steps=6, tool="write_file", delay=0.3,
        arguments=lambda turn: {"path": f"f{turn}.txt", "content": "x"},
    )
    with make_client(tmp_path, adapter) as client:
        task_id = client.post("/api/tasks", json={"goal": "concurrent"}).json()["task_id"]
        wait_for(client, task_id, state="EXECUTING")

        refusals = [client.post(f"/api/tasks/{task_id}/resume").json() for _ in range(4)]
        assert all(item["resumed"] is False and "already running" in item["reason"] for item in refusals)

        task = wait_for(client, task_id)
        assert task["state"] == "COMPLETED"
        assert task["tool_calls"] == 6
        assert len(list((tmp_path / "workspace").glob("f*.txt"))) == 6
        assert collections.Counter(step["status"] for step in task["plan"]) == {"done": 6}


def test_concurrent_drivers_at_the_orchestrator_level_are_refused(tmp_path: Path):
    from sam_backend.autonomy import TaskAlreadyRunning

    adapter = ScriptedAdapter(
        steps=2, tool="write_file", delay=0.4,
        arguments=lambda turn: {"path": f"f{turn}.txt", "content": "x"},
    )
    with make_client(tmp_path, adapter) as client:
        orchestrator = client.app.state.orchestrator
        task = orchestrator.store.create("direct")

        async def race():
            first = asyncio.create_task(orchestrator.run(task))
            await asyncio.sleep(0.2)
            with pytest.raises(TaskAlreadyRunning):
                await orchestrator.resume(task.id)
            return await first

        finished = client.portal.call(race) if hasattr(client, "portal") else asyncio.run(race())
        assert finished.terminal


def test_an_approval_cannot_be_resolved_through_an_unrelated_task(tmp_path: Path):
    """An approval is permission for one action in one run.

    The executed call is hash-bound, so another task cannot redirect it -- but
    routing it through a second task would still mark that task's step done
    and let a guarded run step past its own pending gate, while the run that
    actually asked waits forever.
    """
    adapter = ScriptedAdapter(steps=1, tool="run_terminal", arguments={"command": "echo hi", "cwd": "."})
    with make_client(tmp_path, adapter, mode="guarded") as client:
        first = client.post("/api/tasks", json={"goal": "needs approval"}).json()["task_id"]
        approval_id = pending_approval_id(wait_for(client, first, state="WAITING_FOR_APPROVAL"))
        assert approval_id, "the run never paused for approval"

        other = client.post("/api/tasks", json={"goal": "an unrelated run"}).json()["task_id"]
        response = client.post(f"/api/tasks/{other}/approvals",
                               json={"approval_id": approval_id, "decision": "approved"})

        assert response.status_code == 404, "another task's approval must not be usable here"
        # The approval the owner raised is untouched and still pending.
        assert client.get(f"/api/approvals/{approval_id}").json()["approval"]["status"] == "pending"
