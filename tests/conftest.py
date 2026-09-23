from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam_backend.app import create_app  # noqa: E402
from sam_backend.config import Settings  # noqa: E402
from sam_backend.models import AssistantTurn, ToolCall  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def supported_runtime():
    """Tests describe SAM running the way it is supported: not elevated.

    A Windows CI runner is an Administrator, and `Settings.prepare()` rightly
    refuses to start there, so every test that builds an application would fail
    for a reason that has nothing to do with what it is testing. This states the
    runtime being described rather than weakening the guard: the tests that own
    that guard set this back to True themselves and still prove the refusal.
    """
    import sam_backend.config as config

    # Session scope: module-scoped fixtures build applications before any
    # function-scoped patch would apply.
    patch = pytest.MonkeyPatch()
    patch.setattr(config, "is_elevated_windows_process", lambda: False)
    yield
    patch.undo()


class FakeAdapter:
    async def list_models(self):
        return [{"id": "fake", "name": "fake", "provider": "ollama"}]

    async def complete(self, messages, tools, model):
        if messages and messages[-1].get("role") == "tool":
            return AssistantTurn("The approved action finished." if "denied" not in messages[-1].get("content", "").lower() else "I respected the denial.")
        last_user = next((message.get("content", "") for message in reversed(messages) if message.get("role") == "user"), "")
        if last_user.startswith("overwrite "):
            _, path, content = last_user.split(" ", 2)
            return AssistantTurn("I need approval to overwrite that file.", [ToolCall("call_overwrite", "write_file", {"path": path, "content": content})])
        if last_user == "run python":
            return AssistantTurn("I need approval to run Python.", [ToolCall("call_python", "run_python", {"code": "print('ran')"})])
        if last_user == "read secret":
            return AssistantTurn("I need approval to access that sensitive file.", [ToolCall("call_secret", "read_file", {"path": ".env"})])
        if last_user.startswith("create "):
            _, path, content = last_user.split(" ", 2)
            return AssistantTurn("Creating it.", [ToolCall("call_create", "write_file", {"path": path, "content": content})])
        return AssistantTurn(f"SAM heard: {last_user}")


class FakeRegistry:
    def __init__(self):
        self.adapter = FakeAdapter()

    def get(self, provider):
        return self.adapter


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        project_root=tmp_path,
        workspace_root=tmp_path / "workspace",
        data_dir=tmp_path / "data",
        default_provider="ollama",
        default_model="fake",
        cors_origins=["http://127.0.0.1:8765"],
    )


@pytest.fixture()
def app(settings: Settings):
    return create_app(settings, FakeRegistry())


@pytest.fixture()
def client(app):
    with TestClient(app) as test_client:
        yield test_client
