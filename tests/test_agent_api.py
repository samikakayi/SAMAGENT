"""The autonomous-task HTTP surface and live activity stream."""

from __future__ import annotations

import dataclasses
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
        return [{"id": "fake", "name": "fake", "provider": "ollama"},
                {"id": "spare", "name": "spare", "provider": "ollama"}]

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
    # The API hands the panel the evidence, not just the label: a verified
    # task can never serialise its verdict as null.
    assert task["verification"] is not None and task["verification"]["verified"] is True
    # The list stays a headline projection; its label must not disagree with
    # the evidence the detail carries.
    listed = next(item for item in agent_client.get("/api/tasks").json()["tasks"] if item["id"] == task_id)
    assert listed["completion_status"] == "completed_verified"


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


# -- which model will run --------------------------------------------------

def test_the_resolution_endpoint_names_a_usable_model(agent_client: TestClient):
    """What the Autopilot panel shows before a task starts."""
    body = agent_client.get("/api/providers/resolution").json()["resolution"]

    assert body["blocked"] is False and body["fallback_engaged"] is False
    assert body["active"]["model"] == "fake" and body["active"]["provider"] == "ollama"
    assert body["primary"]["usable"] is True
    assert body["fallback"] is None and body["fallback_enabled"] is False


@pytest.mark.parametrize("enabled, blocked, active", [(False, True, None), (True, False, "spare")])
def test_a_quota_blocked_primary_is_reported_with_its_fallback(agent_client: TestClient, agent_settings: Settings,
                                                               enabled, blocked, active):
    from sam_backend.provider_health import Availability, ModelCapability

    health = agent_client.app.state.orchestrator.health
    health.remember(ModelCapability("ollama", "fake", Availability.UNAVAILABLE_QUOTA, "no paid credit",
                                    supports_tools=True, context_length=200_000, cost_class="paid"))
    health.remember(ModelCapability("ollama", "spare", Availability.AVAILABLE,
                                    supports_tools=True, context_length=200_000, cost_class="free"))
    agent_settings.fallback_model = "spare"
    agent_settings.fallback_enabled = enabled

    body = agent_client.get("/api/providers/resolution").json()["resolution"]

    assert body["blocked"] is blocked
    assert (body["active"] or {}).get("model") == active
    assert body["primary"]["availability"] == "UNAVAILABLE_QUOTA" and "no paid credit" in body["primary"]["reason"]
    assert body["fallback"]["model"] == "spare" and body["fallback"]["usable"] is True
    assert "fake is unavailable" in body["fallback_reason"]


# -- fallback configured at runtime, not at startup ------------------------

def quota_block_primary(client: TestClient) -> None:
    """Make the configured model unavailable the way a spent account would."""
    from sam_backend.provider_health import Availability, ModelCapability

    client.app.state.orchestrator.health.remember(ModelCapability(
        "ollama", "fake", Availability.UNAVAILABLE_QUOTA, "no paid credit",
        supports_tools=True, context_length=200_000, cost_class="paid",
    ))


def resolution_of(client: TestClient) -> dict:
    quota_block_primary(client)
    return client.get("/api/providers/resolution").json()["resolution"]


def test_fallback_is_switched_on_and_off_through_the_api_with_no_restart(agent_client: TestClient):
    """The whole point: the running process changes which model a run uses."""
    assert agent_client.put("/api/settings", json={"fallback_model": "spare"}).status_code == 200
    disabled = resolution_of(agent_client)
    assert disabled["blocked"] is True and "fallback is disabled" in disabled["fallback_reason"]

    assert agent_client.put("/api/settings", json={"fallback_enabled": True}).status_code == 200
    engaged = resolution_of(agent_client)
    assert engaged["blocked"] is False and engaged["fallback_engaged"] is True
    assert engaged["active"]["model"] == "spare"

    assert agent_client.put("/api/settings", json={"fallback_enabled": False}).status_code == 200
    assert resolution_of(agent_client)["blocked"] is True


def test_a_run_started_after_the_change_uses_the_newly_configured_fallback(agent_client: TestClient):
    agent_client.put("/api/settings", json={"fallback_model": "spare", "fallback_enabled": True})
    quota_block_primary(agent_client)

    task_id = agent_client.post("/api/tasks", json={"goal": "Create a report file"}).json()["task_id"]
    task = wait_for_state(agent_client, task_id)

    assert task["state"] == "COMPLETED"
    switch = [event for event in task["events"] if event["kind"] == "fallback"]
    assert len(switch) == 1 and "spare" in switch[0]["message"]


def test_renaming_the_fallback_drops_the_verdict_cached_for_the_old_one(agent_client: TestClient):
    from sam_backend.provider_health import Availability, ModelCapability

    health = agent_client.app.state.orchestrator.health
    stale = ModelCapability("ollama", "spare", Availability.UNAVAILABLE_QUOTA, "spent an hour ago")
    health.remember(stale)
    agent_client.put("/api/settings", json={"fallback_model": "spare"})
    assert health.cached("ollama", "spare") is None, "a changed model is re-checked, not inherited"

    # Toggling the switch changes no verdict, so nothing usable is thrown away.
    health.remember(ModelCapability("ollama", "spare", Availability.AVAILABLE,
                                    supports_tools=True, context_length=200_000))
    agent_client.put("/api/settings", json={"fallback_enabled": True})
    assert health.cached("ollama", "spare") is not None


@pytest.mark.parametrize("payload, status", [
    ({"fallback_model": "nvidia/nemotron-3-ultra-550b-a55b:free"}, 200),
    ({"fallback_model": "   "}, 200),               # blank clears it
    ({"fallback_model": "not a model id"}, 422),    # not an identifier
    ({"fallback_model": "vendor/mock-model"}, 422),  # a test double, not a model
    ({"fallback_model": "fake"}, 422),
    ({"fallback_enabled": "sometimes"}, 422),
    ({"fallback_enabled": True}, 422),              # armed with nothing behind it
])
def test_fallback_settings_are_validated_before_they_take_effect(agent_client: TestClient, payload, status):
    assert agent_client.put("/api/settings", json=payload).status_code == status


def test_fallback_settings_are_persisted_and_survive_a_restart(agent_settings: Settings):
    with TestClient(create_app(agent_settings, ScriptedRegistry())) as client:
        client.put("/api/settings", json={"fallback_model": "spare", "fallback_enabled": True})
        body = client.get("/api/settings").json()
        assert body["overrides"]["fallback_model"] == "spare"
        assert body["runtime"]["fallback_enabled"] is True
        assert not any("api_key" in key for key in body["runtime"]), "no credential is returned"

    # A second process reading the same store: the persisted value wins over
    # the environment default the Settings object started with.
    restarted = dataclasses.replace(agent_settings, fallback_model="", fallback_enabled=False)
    with TestClient(create_app(restarted, ScriptedRegistry())) as client:
        assert client.get("/api/settings").json()["runtime"]["fallback_model"] == "spare"
    assert restarted.fallback_enabled is True


def test_the_resolution_endpoint_is_served_from_cache_unless_refreshed(agent_client: TestClient):
    adapter = agent_client.app.state.adapters.adapter
    calls = {"n": 0}
    original = adapter.list_models

    async def counted():
        calls["n"] += 1
        return await original()

    adapter.list_models = counted
    agent_client.get("/api/providers/resolution")
    first = calls["n"]
    agent_client.get("/api/providers/resolution")
    assert calls["n"] == first, "a cached verdict is not re-probed"
    agent_client.get("/api/providers/resolution?refresh=true")
    assert calls["n"] == first + 1, "an explicit refresh is"


def test_the_autopilot_panel_parses_and_consults_the_resolution_endpoint():
    import shutil
    import subprocess

    panel = Path(__file__).resolve().parents[1] / "frontend" / "agent-panel.js"
    source = panel.read_text(encoding="utf-8")
    assert "/api/providers/resolution" in source
    assert "fallback:" in source, "a model switch has its own timeline presentation"
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    subprocess.run([node, "--check", str(panel)], check=True, capture_output=True)


class QuotaRegistry:
    """Every request fails on credit; counts them."""

    def __init__(self) -> None:
        self.requests = 0

    def get(self, provider):
        registry = self

        class Adapter:
            async def list_models(self):
                return [{"id": "fake", "name": "fake", "provider": "ollama"}]

            async def complete(self, messages, tools, model):
                from sam_backend.models import ErrorCategory, ModelError

                registry.requests += 1
                raise ModelError("Insufficient credits", ErrorCategory.QUOTA)

        return Adapter()


def test_a_mid_run_quota_failure_is_the_providers_and_is_not_repeated(agent_settings: Settings):
    """Preflight passed (the model was listed), then the first real request
    hit a credit wall. The run says the provider is unavailable, sends
    nothing further, and the next run is stopped before spending anything."""
    registry = QuotaRegistry()
    with TestClient(create_app(agent_settings, registry)) as client:
        first = wait_for_state(client, client.post("/api/tasks", json={"goal": "Create a report file"}).json()["task_id"])
        second = wait_for_state(client, client.post("/api/tasks", json={"goal": "Create a report file"}).json()["task_id"])

    assert registry.requests == 1, "the planner's failure was enough; nothing was re-sent"
    assert first["completion_status"] == second["completion_status"] == "provider_unavailable"
    assert "credit" in first["summary"].lower()
    assert "did not start" in second["summary"]
    assert not (Path(agent_settings.workspace_root) / "report.txt").exists()
