"""Credential containment and model-routing behaviour (spec sections 98-100)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from sam_backend.config import Settings
from sam_backend.db import Database
from sam_backend.models import AdapterRegistry, AssistantTurn, ModelError
from sam_backend.routing import ModelRouter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SENTINEL = "sk-or-v1-THIS-IS-A-TEST-SENTINEL-0123456789"


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    created = Settings(
        project_root=tmp_path,
        workspace_root=tmp_path / "workspace",
        data_dir=tmp_path / "data",
        openrouter_api_key=SENTINEL,
    )
    created.prepare()
    return created


# --- Credential containment ---------------------------------------------------


def test_the_public_settings_view_never_carries_the_key(settings: Settings):
    public = settings.public_dict()
    assert "openrouter_api_key" not in public
    assert SENTINEL not in json.dumps(public)
    # It reports only that a key is configured.
    assert public["openrouter_configured"] is True


def test_saved_ui_overrides_cannot_persist_a_credential(settings: Settings):
    settings.save_public_overrides({"openrouter_api_key": SENTINEL, "model_mode": "AUTO"})
    saved = json.loads((settings.data_dir / "settings.json").read_text(encoding="utf-8"))
    assert "openrouter_api_key" not in saved
    assert SENTINEL not in json.dumps(saved)


def test_no_frontend_file_contains_a_credential_pattern():
    for path in (PROJECT_ROOT / "frontend").rglob("*"):
        if path.suffix.lower() not in {".js", ".html", ".css"}:
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        assert "sk-or-" not in content, f"{path.name} contains an OpenRouter key pattern"
        assert "OPENROUTER_API_KEY" not in content, f"{path.name} references the key variable"


def test_no_backend_source_hardcodes_a_credential():
    """The policy engine names key prefixes to detect them; that is not a leak.

    Only a full key-shaped literal counts, so the scan looks for the prefix
    followed by real key characters rather than the bare prefix.
    """
    literal = re.compile(r"sk-or-v1-[A-Za-z0-9_-]{12,}")
    for path in (PROJECT_ROOT / "sam_backend").rglob("*.py"):
        content = path.read_text(encoding="utf-8", errors="ignore")
        assert not literal.findall(content), f"{path} contains a literal OpenRouter key"


def test_the_key_is_read_from_the_environment_not_committed_files():
    example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    # The example may name the variable but must never assign it a value.
    for line in example.splitlines():
        stripped = line.strip()
        if stripped.startswith("OPENROUTER_API_KEY="):
            assert stripped == "OPENROUTER_API_KEY=" or stripped.startswith("#")


def test_gitignore_excludes_the_env_file():
    ignored = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in [line.strip() for line in ignored]


def test_the_adapter_sends_the_key_only_as_a_bearer_header(settings: Settings):
    adapter = AdapterRegistry(settings).get("openrouter")
    headers = adapter._headers()
    assert headers["Authorization"] == f"Bearer {SENTINEL}"
    # Nothing else may carry it, and the title header must not leak it either.
    assert all(SENTINEL not in value for key, value in headers.items() if key != "Authorization")


def test_audit_records_never_contain_the_key(settings: Settings):
    database = Database(settings.database_path)
    database.add_audit("model", "call", "routed a request", details={"provider": "openrouter", "model": "auto"})
    entries = json.dumps(database.list_audit(50))
    assert SENTINEL not in entries


# --- Routing and degradation --------------------------------------------------


class StubAdapter:
    def __init__(self, models: list[str], *, fail: bool = False) -> None:
        self._models = models
        self.fail = fail
        self.calls = 0

    async def list_models(self):
        return [{"id": name, "name": name} for name in self._models]

    async def complete(self, messages, tools, model):
        self.calls += 1
        if self.fail:
            raise ModelError("provider is unavailable")
        return AssistantTurn(f"answered by {model}", raw={"model": model, "usage": {}})


class StubRegistry:
    def __init__(self, adapters: dict[str, StubAdapter]) -> None:
        self.adapters = adapters

    def get(self, provider: str):
        if provider not in self.adapters:
            raise ModelError(f"Unsupported provider: {provider}")
        return self.adapters[provider]


@pytest.mark.asyncio
async def test_a_cloud_outage_falls_back_to_the_local_model(settings: Settings):
    settings.model_mode = "AUTO"
    database = Database(settings.database_path)
    adapters = {
        "openrouter": StubAdapter(["openrouter/auto"], fail=True),
        "litellm": StubAdapter([], fail=True),
        "ollama": StubAdapter(["qwen3.5:4b"]),
        "openai": StubAdapter([]),
    }
    router = ModelRouter(settings, StubRegistry(adapters), database)

    turn, choice, failures = await router.complete(
        message="compare snr and wyckoff and ict analysis in depth",
        messages=[{"role": "user", "content": "compare"}],
        tools=[], provider=None, model=None, conversation_id=None,
    )
    assert choice.provider == "ollama", f"expected local fallback, routed to {choice.provider}"
    assert adapters["ollama"].calls == 1
    # The degradation must be visible, not silent.
    assert failures, "a cloud failure should be recorded on the turn"
    assert turn.raw["fallbacks"] == failures


@pytest.mark.asyncio
async def test_every_route_failing_raises_a_clear_error(settings: Settings):
    database = Database(settings.database_path)
    adapters = {
        "openrouter": StubAdapter(["openrouter/auto"], fail=True),
        "litellm": StubAdapter([], fail=True),
        "ollama": StubAdapter(["qwen3.5:4b"], fail=True),
        "openai": StubAdapter([]),
    }
    router = ModelRouter(settings, StubRegistry(adapters), database)
    with pytest.raises(ModelError, match="All model routes failed"):
        await router.complete(
            message="analyze", messages=[{"role": "user", "content": "x"}],
            tools=[], provider=None, model=None, conversation_id=None,
        )


@pytest.mark.asyncio
async def test_local_only_mode_never_reaches_a_cloud_provider(settings: Settings):
    settings.model_mode = "LOCAL_ONLY"
    database = Database(settings.database_path)
    adapters = {
        "openrouter": StubAdapter(["openrouter/auto"]),
        "litellm": StubAdapter(["sam-fast"]),
        "ollama": StubAdapter(["qwen3.5:4b"]),
        "openai": StubAdapter([]),
    }
    router = ModelRouter(settings, StubRegistry(adapters), database)
    _, choices = await router.route("analyze gold")
    assert {choice.provider for choice in choices} == {"ollama"}


@pytest.mark.asyncio
async def test_exhausting_the_budget_forces_local_only(settings: Settings):
    settings.daily_budget_usd = 1.0
    database = Database(settings.database_path)
    database.add_model_usage(
        provider="openrouter", model="x", route_mode="AUTO",
        input_tokens=1000, output_tokens=1000, cost_usd=5.0,
    )
    adapters = {
        "openrouter": StubAdapter(["openrouter/auto"]),
        "litellm": StubAdapter(["sam-fast"]),
        "ollama": StubAdapter(["qwen3.5:4b"]),
        "openai": StubAdapter([]),
    }
    router = ModelRouter(settings, StubRegistry(adapters), database)
    assert router.budget_state()["mode"] == "LOCAL_ONLY"
    _, choices = await router.route("compare wyckoff and ict in depth")
    assert {choice.provider for choice in choices} == {"ollama"}


def test_task_profiling_separates_simple_from_complex_work():
    simple = ModelRouter.profile("open tradingview")
    complex_task = ModelRouter.profile("compare snr and wyckoff and ict analysis and strategy in depth")
    vision = ModelRouter.profile("look at the screen and tell me what the chart shows")

    assert simple.complexity == "fast"
    assert complex_task.complexity == "strong"
    assert vision.needs_vision is True
    # A credential question must be treated as privacy sensitive.
    assert ModelRouter.profile("read my api key from .env").privacy_sensitive is True
