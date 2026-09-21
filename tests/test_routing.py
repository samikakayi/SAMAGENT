from __future__ import annotations

from pathlib import Path

import pytest

from sam_backend.config import Settings
from sam_backend.db import Database
from sam_backend.models import AssistantTurn, ModelError
from sam_backend.routing import ModelRouter


class RouteAdapter:
    def __init__(self, provider: str, available: bool = True):
        self.provider = provider
        self.available = available

    async def list_models(self):
        return [{"id": f"{self.provider}-model", "provider": self.provider}] if self.available else []

    async def complete(self, messages, tools, model):
        return AssistantTurn("ok", raw={"model": model, "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.01}})


class RouteRegistry:
    def __init__(self, availability=None):
        availability = availability or {}
        self.adapters = {name: RouteAdapter(name, availability.get(name, False)) for name in ("ollama", "litellm", "openrouter", "openai")}

    def get(self, provider):
        return self.adapters[provider]


def router(tmp_path: Path, *, availability=None, **overrides):
    settings = Settings(project_root=tmp_path, workspace_root=tmp_path / "workspace", data_dir=tmp_path / "data", **overrides)
    settings.prepare()
    database = Database(settings.database_path)
    return ModelRouter(settings, RouteRegistry(availability), database), settings, database


def test_task_profile_marks_vision_privacy_and_complexity():
    profile = ModelRouter.profile("Analyze this chart screenshot and compare ICT with Wyckoff; do not expose my password")
    assert profile.needs_vision is True
    assert profile.privacy_sensitive is True
    assert profile.complexity == "strong"


def test_task_profile_omits_tools_for_chat_and_enables_them_for_actions():
    assert ModelRouter.profile("What is your name?").needs_tools is False
    assert ModelRouter.profile("ناوت چییە؟").needs_tools is False
    assert ModelRouter.profile("Create a file in the workspace").needs_tools is True
    assert ModelRouter.profile("فایلێک لە workspace دروست بکە").needs_tools is True


@pytest.mark.asyncio
async def test_auto_route_uses_available_local_model(tmp_path):
    model_router, _, _ = router(tmp_path, availability={"ollama": True})
    _, choices = await model_router.route("hello", provider="auto")
    assert [(item.provider, item.model) for item in choices] == [("ollama", "qwen3.5:4b")]


@pytest.mark.asyncio
async def test_explicit_provider_does_not_silently_change_boundary(tmp_path):
    model_router, _, _ = router(tmp_path, availability={"ollama": True}, openrouter_api_key="configured-test-key")
    _, choices = await model_router.route("hello", provider="openrouter", model="vendor/model")
    assert len(choices) == 1
    assert choices[0].provider == "openrouter"
    assert choices[0].model == "vendor/model"


@pytest.mark.asyncio
async def test_cloud_only_uses_saved_sonnet_instead_of_openrouter_auto(tmp_path):
    model_router, _, _ = router(
        tmp_path,
        availability={"ollama": True},
        openrouter_api_key="configured-test-key",
        model_mode="CLOUD_ONLY",
        default_model="anthropic/claude-sonnet-4.5",
        openrouter_fast_model="openrouter/auto",
    )
    _, choices = await model_router.route("hello", provider="auto")
    assert choices[0].provider == "openrouter"
    assert choices[0].model == "anthropic/claude-sonnet-4.5"


@pytest.mark.asyncio
async def test_openrouter_ignores_a_local_ollama_model_id(tmp_path):
    model_router, _, _ = router(
        tmp_path,
        availability={"openrouter": True},
        openrouter_api_key="configured-test-key",
        default_model="anthropic/claude-sonnet-4.5",
        openrouter_fast_model="openrouter/auto",
    )
    _, choices = await model_router.route("hello", provider="openrouter", model="qwen3.5:4b")
    assert choices[0].provider == "openrouter"
    assert choices[0].model == "anthropic/claude-sonnet-4.5"


@pytest.mark.asyncio
async def test_no_available_route_returns_clear_error(tmp_path):
    model_router, _, _ = router(tmp_path)
    with pytest.raises(ModelError, match="No AI model route"):
        await model_router.route("ordinary chat", provider="auto")


@pytest.mark.asyncio
async def test_budget_hard_limit_forces_local_only(tmp_path):
    model_router, settings, database = router(
        tmp_path,
        availability={"ollama": True, "openrouter": True},
        openrouter_api_key="configured-test-key",
        daily_budget_usd=1,
        monthly_budget_usd=10,
    )
    database.add_model_usage(provider="openrouter", model="test", route_mode="AUTO", cost_usd=1.25)
    assert model_router.budget_state()["mode"] == "LOCAL_ONLY"
    _, choices = await model_router.route("compare two complex theories", provider="auto")
    assert all(item.provider == "ollama" for item in choices)
    assert settings.model_mode == "AUTO"


@pytest.mark.asyncio
async def test_successful_completion_records_usage_and_route(tmp_path):
    model_router, _, database = router(tmp_path, availability={"ollama": True})
    turn, choice, failures = await model_router.complete(
        message="hello",
        messages=[{"role": "user", "content": "hello"}],
        tools=[],
        provider="auto",
        model=None,
        conversation_id=None,
        task_id="task-test",
    )
    assert turn.content == "ok"
    assert choice.provider == "ollama"
    assert failures == []
    assert database.model_cost_summary()["today"]["requests"] == 1
