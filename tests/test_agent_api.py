"""The autonomous-task HTTP surface and live activity stream."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sam_backend.app import create_app
from sam_backend.config import Settings
from sam_backend.models import AssistantTurn, ToolCall


class ScriptedAdapter:
    """Plans once, then writes one file, then reports done."""

    def __init__(self) -> None:
        self.executor_turns = 0

    async def list_models(self):
        return [{"id": "fake", "name": "fake", "provider": "ollama"}]

    async def complete(self, messages, tools, model):
        joined = "\n".join(str(item.get("content", "")) for item in messages)
        if "planning stage" in joined:
            payload = {
                "understanding": "Create the requested file.",
                "steps": [{"text": "Create report.txt", "kind": "edit"}],
            }
            return AssistantTurn(f"```json\n{json.dumps(payload)}\n```")
        self.executor_turns += 1
        if self.executor_turns == 1:
            return AssistantTurn(
                "Creating it.",
                [ToolCall("call_write", "write_file", {"path": "report.txt", "content": "done"})],
            )
        return AssistantTurn("Finished.")


class ScriptedRegistry:
    def __init__(self) -> None:
        self.adapter = ScriptedAdapter()

    def get(self, provider):
        return self.adapter


@pytest.fixture()
def agent_settings(tmp_path: Path) -> Settings:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (workspace / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    return Settings(
        project_root=tmp_path,
        workspace_root=workspace,
        data_dir=tmp_path / "data",
        default_provider="ollama",
        default_model="fake",
        permission_mode="trusted",
        cors_origins=["http://127.0.0.1:8765"],
    )


@pytest.fixture()
def agent_client(agent_settings: Settings):
    with TestClient(create_app(agent_settings, ScriptedRegistry())) as client:
        yield client


def wait_for_state(client: TestClient, task_id: str, *, timeout: float = 90.0) -> dict:
    """Poll until the run reaches a terminal state or pauses for approval."""
    deadline = time.time() + timeout
    task = {}
    while time.time() < deadline:
        task = client.get(f"/api/tasks/{task_id}").json()["task"]
        if task["terminal"] or task["state"] == "WAITING_FOR_APPROVAL":
            return task
        time.sleep(0.2)
    return task


def test_a_task_can_be_started_and_followed_to_completion(agent_client: TestClient, agent_settings: Settings):
    response = agent_client.post("/api/tasks", json={"goal": "Create a report file"})
    assert response.status_code == 202
    task_id = response.json()["task_id"]

    task = wait_for_state(agent_client, task_id)

    assert task["state"] == "COMPLETED", task.get("summary")
    assert task["completion_status"] == "completed_verified"
    assert (Path(agent_settings.workspace_root) / "report.txt").read_text(encoding="utf-8") == "done"


def test_a_running_task_appears_in_the_task_list(agent_client: TestClient):
    task_id = agent_client.post("/api/tasks", json={"goal": "Create a report file"}).json()["task_id"]
    wait_for_state(agent_client, task_id)

    listing = agent_client.get("/api/tasks").json()["tasks"]
    assert any(item["id"] == task_id for item in listing)
    row = next(item for item in listing if item["id"] == task_id)
    assert row["steps_total"] >= 1
    assert "events" not in row, "the list view must stay compact"


def test_a_task_exposes_its_timeline_for_the_activity_ui(agent_client: TestClient):
    task_id = agent_client.post("/api/tasks", json={"goal": "Create a report file"}).json()["task_id"]
    task = wait_for_state(agent_client, task_id)

    kinds = {event["kind"] for event in task["events"]}
    assert {"state", "plan", "tool", "result"} <= kinds
    assert task["plan"], "the plan must be visible to the user"
    assert any(event["kind"] == "state" for event in task["events"])


def test_an_empty_goal_is_rejected(agent_client: TestClient):
    assert agent_client.post("/api/tasks", json={"goal": "   "}).status_code == 422


def test_an_unknown_task_is_a_404(agent_client: TestClient):
    assert agent_client.get("/api/tasks/task_nope").status_code == 404
    assert agent_client.post(
        "/api/tasks/task_nope/approvals", json={"approval_id": "a", "decision": "approved"},
    ).status_code == 404


def test_the_project_map_endpoint_describes_the_workspace(agent_client: TestClient):
    payload = agent_client.get("/api/project/map").json()["map"]
    assert payload["commands"]["test"] == "python -m pytest"
    assert "summary" in payload
    assert "tree" not in payload, "the summary view must stay compact"


def test_the_environment_endpoint_reports_tools_and_providers(agent_client: TestClient):
    payload = agent_client.get("/api/environment").json()
    assert "python" in payload["tools"]["capabilities"]
    assert payload["checks"]["test"] == "python -m pytest"
    assert any(provider["name"] == "ollama" for provider in payload["providers"])


def test_no_secret_value_is_exposed_through_the_project_map(agent_client: TestClient, agent_settings: Settings):
    (Path(agent_settings.workspace_root) / ".env").write_text(
        "OPENAI_API_KEY=sk-live-must-not-appear-anywhere-1234567890\n", encoding="utf-8",
    )
    body = agent_client.get("/api/project/map?refresh=true").text

    assert "OPENAI_API_KEY" in body, "names are useful context"
    assert "sk-live-must-not-appear" not in body, "values must never leave the machine's disk"


def test_live_activity_is_broadcast_to_connected_clients(agent_client: TestClient):
    """The UI learns what the agent is doing without polling."""
    with agent_client.websocket_connect("/ws/live") as socket:
        assert socket.receive_json()["type"] == "state"
        task_id = agent_client.post("/api/tasks", json={"goal": "Create a report file"}).json()["task_id"]

        seen: list[dict] = []
        deadline = time.time() + 60
        while time.time() < deadline:
            message = socket.receive_json()
            seen.append(message)
            if message.get("type") == "task_state" and message.get("state") in {"COMPLETED", "FAILED"}:
                break

        assert any(item.get("type") == "task_state" for item in seen)
        assert any(item.get("type") == "task_event" for item in seen)
        assert all(item.get("task_id") == task_id for item in seen if item.get("task_id"))
        states = [item.get("state") for item in seen if item.get("type") == "task_state"]
        assert "PLANNING" in states and "COMPLETED" in states
