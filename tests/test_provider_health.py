"""Provider preflight, model capability, and the real-model fallback policy.

No network: OpenRouter's catalogue and key endpoints are answered by a local
httpx transport, and other providers by a fake adapter registry. The
real-provider path is verified separately.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time

import httpx
import pytest

from sam_backend.config import Settings
from sam_backend.models import ErrorCategory, ModelError
from sam_backend.provider_health import (
    VERDICT_TTL_SECONDS,
    Availability,
    ModelCapability,
    ProviderHealth,
)
from sam_backend.tasks import TaskState
from tests.test_autonomy import ScriptedRouter, build_orchestrator, done_turn, plan_turn

PAID_MODEL = "anthropic/claude-sonnet-4.5"
FREE_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
PROBE_KEY = "sk-or-v1-" + "a" * 48


def catalogue_entry(model: str, *, paid: bool, tools: bool = True, context: int = 1_000_000) -> dict:
    return {
        "id": model,
        "context_length": context,
        "supported_parameters": ["tools"] if tools else ["temperature"],
        "pricing": {"prompt": "0.000003" if paid else "0", "completion": "0"},
    }


@pytest.fixture()
def openrouter(monkeypatch):
    """Answer OpenRouter's two free endpoints locally. Returns a factory:
    openrouter(free_tier=..., models=..., ...) -> (ProviderHealth, request counter)."""

    def make(*, free_tier: bool, limit_remaining=None, models=None, key_status: int = 200,
             transport_error: bool = False, settings: Settings | None = None):
        catalogue = models if models is not None else [
            catalogue_entry(PAID_MODEL, paid=True), catalogue_entry(FREE_MODEL, paid=False),
        ]
        calls = {"n": 0}

        def handle(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if transport_error:
                raise httpx.ConnectError("no route to host", request=request)
            if request.url.path.endswith("/key"):
                return httpx.Response(key_status, request=request, json={
                    "data": {"is_free_tier": free_tier, "limit_remaining": limit_remaining},
                })
            return httpx.Response(200, json={"data": catalogue}, request=request)

        transport = httpx.MockTransport(handle)
        original = httpx.AsyncClient

        class Patched(original):  # type: ignore[misc,valid-type]
            def __init__(self, *args, **kwargs):
                kwargs["transport"] = transport
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", Patched)
        settings = settings or Settings(openrouter_api_key=PROBE_KEY, default_provider="openrouter",
                                        default_model=PAID_MODEL)
        return ProviderHealth(settings), calls

    return make


class FakeRegistry:
    """A non-OpenRouter provider that lists exactly these models."""

    def __init__(self, names: list[str], *, fail: bool = False) -> None:
        self.names = names
        self.fail = fail

    def get(self, provider: str):
        registry = self

        class Adapter:
            async def list_models(self):
                if registry.fail:
                    raise RuntimeError("connection refused")
                return [{"id": name, "name": name} for name in registry.names]

        return Adapter()


def check(health: ProviderHealth, model: str, provider: str = "openrouter", registry=None) -> ModelCapability:
    return asyncio.run(health.capability(provider, model, registry or FakeRegistry([])))


# -- preflight verdicts ----------------------------------------------------

def test_a_capability_is_usable_only_when_available_and_able():
    base = dict(provider="openrouter", model=FREE_MODEL, availability=Availability.AVAILABLE,
                supports_tools=True, context_length=16_000)
    assert ModelCapability(**base).usable

    assert not ModelCapability(**{**base, "availability": Availability.UNAVAILABLE_QUOTA}).usable
    assert not ModelCapability(**{**base, "supports_tools": False}).usable, "a run needs tool calling"
    assert not ModelCapability(**{**base, "context_length": 4096}).usable, "a run needs room to think"


@pytest.mark.parametrize("free_tier, model, availability, cost_class", [
    (False, PAID_MODEL, Availability.AVAILABLE, "paid"),
    # The whole point: knowing this costs zero tokens.
    (True, PAID_MODEL, Availability.UNAVAILABLE_QUOTA, "paid"),
    (True, FREE_MODEL, Availability.AVAILABLE, "free"),
])
def test_the_account_decides_whether_a_billed_model_is_usable(openrouter, free_tier, model, availability, cost_class):
    health, _calls = openrouter(free_tier=free_tier, limit_remaining=None if free_tier else 25.0)

    capability = check(health, model)

    assert capability.availability is availability
    assert capability.cost_class == cost_class
    assert capability.usable is (availability is Availability.AVAILABLE)


@pytest.mark.parametrize("kwargs, availability", [
    ({"key_status": 401}, Availability.UNAVAILABLE_AUTH),
    ({"transport_error": True}, Availability.UNAVAILABLE_NETWORK),
])
def test_a_provider_problem_is_classified_not_mistaken_for_the_model(openrouter, kwargs, availability):
    health, _calls = openrouter(free_tier=True, **kwargs)

    assert check(health, FREE_MODEL).availability is availability


def test_a_missing_credential_is_auth_before_any_request():
    health = ProviderHealth(Settings(openrouter_api_key=None, default_provider="openrouter"))

    capability = check(health, PAID_MODEL)

    assert capability.availability is Availability.UNAVAILABLE_AUTH
    assert "credential" in capability.reason


@pytest.mark.parametrize("model, models, expected_fragment", [
    ("does/not-exist", None, "catalogue"),
    ("no/tools", [catalogue_entry("no/tools", paid=False, tools=False)], "tool calling"),
    ("tiny/context", [catalogue_entry("tiny/context", paid=False, context=4096)], "context"),
])
def test_a_model_that_cannot_do_the_job_is_unsupported(openrouter, model, models, expected_fragment):
    """Responding to chat is not the bar; an autonomous run needs more."""
    health, _calls = openrouter(free_tier=True, models=models)

    capability = check(health, model)

    assert capability.availability is Availability.UNSUPPORTED
    assert expected_fragment in capability.reason


def test_a_local_provider_is_checked_against_what_it_serves():
    settings = Settings(default_provider="ollama", default_model="qwen3.5:4b")

    served = check(ProviderHealth(settings), "qwen3.5:4b", "ollama", FakeRegistry(["qwen3.5:4b"]))
    missing = check(ProviderHealth(settings), "absent:1b", "ollama", FakeRegistry(["qwen3.5:4b"]))
    down = check(ProviderHealth(settings), "qwen3.5:4b", "ollama", FakeRegistry([], fail=True))

    assert served.availability is Availability.AVAILABLE and served.cost_class == "free"
    assert missing.availability is Availability.UNSUPPORTED
    assert down.availability is Availability.UNAVAILABLE_NETWORK


# -- the fallback policy ---------------------------------------------------

def resolve(openrouter, *, free_tier: bool, fallback: str = "", enabled: bool = False, models=None):
    settings = Settings(
        openrouter_api_key=PROBE_KEY, default_provider="openrouter", default_model=PAID_MODEL,
        fallback_model=fallback, fallback_enabled=enabled,
    )
    health, _calls = openrouter(free_tier=free_tier, limit_remaining=None if free_tier else 25.0,
                                settings=settings, models=models)
    return asyncio.run(health.resolve(FakeRegistry([])))


def test_a_usable_primary_is_used_and_no_fallback_is_engaged(openrouter):
    resolution = resolve(openrouter, free_tier=False, fallback=FREE_MODEL, enabled=True)

    assert resolution.active.model == PAID_MODEL
    assert not resolution.fallback_engaged and not resolution.blocked


def test_a_quota_blocked_primary_switches_to_the_configured_real_fallback(openrouter):
    resolution = resolve(openrouter, free_tier=True, fallback=FREE_MODEL, enabled=True)

    assert resolution.primary.availability is Availability.UNAVAILABLE_QUOTA
    assert resolution.active.model == FREE_MODEL and resolution.fallback_engaged
    assert resolution.active.provider == resolution.primary.provider, "a real model on the same provider"
    assert PAID_MODEL in resolution.fallback_reason and "credit" in resolution.fallback_reason


def test_with_fallback_disabled_the_run_is_blocked_rather_than_switched(openrouter):
    """Silently changing model is exactly what this policy prevents."""
    resolution = resolve(openrouter, free_tier=True, fallback=FREE_MODEL, enabled=False)

    assert resolution.blocked
    assert "fallback is disabled" in resolution.fallback_reason
    assert resolution.fallback.usable, "still assessed, so the UI can offer it"


def test_no_configured_fallback_blocks_even_when_enabled(openrouter):
    resolution = resolve(openrouter, free_tier=True, enabled=True)

    assert resolution.blocked
    assert "no fallback is configured" in resolution.fallback_reason


def test_an_unusable_fallback_is_named_rather_than_used(openrouter):
    resolution = resolve(
        openrouter, free_tier=True, fallback="no/tools", enabled=True,
        models=[catalogue_entry(PAID_MODEL, paid=True), catalogue_entry("no/tools", paid=False, tools=False)],
    )

    assert resolution.blocked, "a fallback that cannot call tools is not a fallback"
    assert "no/tools is also unavailable" in resolution.fallback_reason


def test_automatic_routing_has_nothing_to_preflight_and_is_not_blocked():
    """default_provider may be "auto": the router picks per request, so a
    preflight cannot name a model and must not invent a fault."""
    settings = Settings(default_provider="auto", default_model="anything")

    resolution = asyncio.run(ProviderHealth(settings).resolve(FakeRegistry([], fail=True)))

    assert resolution.primary.availability is Availability.UNKNOWN
    assert not resolution.blocked


# -- the cache -------------------------------------------------------------

def test_a_confirmed_quota_failure_is_not_re_probed_immediately(openrouter):
    """Re-probing a paid model after a credit failure is the waste to avoid."""
    health, calls = openrouter(free_tier=True)

    first = check(health, PAID_MODEL)
    after_first = calls["n"]
    second = check(health, PAID_MODEL)

    assert first.availability is second.availability is Availability.UNAVAILABLE_QUOTA
    assert calls["n"] == after_first, "the second check was served from cache"


def test_a_cached_verdict_expires_and_is_checked_again():
    health = ProviderHealth(Settings())
    stale = health.remember(ModelCapability("openrouter", FREE_MODEL, Availability.UNAVAILABLE_NETWORK, "blip"))
    assert health.cached("openrouter", FREE_MODEL) is not None

    stale.checked_at = time.time() - VERDICT_TTL_SECONDS[Availability.UNAVAILABLE_NETWORK] - 1

    assert health.cached("openrouter", FREE_MODEL) is None, "a transient verdict must not stick"


def test_a_real_request_failure_teaches_the_cache_without_forgetting_the_model():
    health = ProviderHealth(Settings())
    health.remember(ModelCapability("openrouter", PAID_MODEL, Availability.AVAILABLE,
                                    supports_tools=True, context_length=1_000_000, cost_class="paid"))

    health.record_failure("openrouter", PAID_MODEL, ModelError("needs more credits", ErrorCategory.QUOTA))
    health.record_failure("openrouter", FREE_MODEL, ModelError("weird", ErrorCategory.MALFORMED))

    cached = health.cached("openrouter", PAID_MODEL)
    assert cached.availability is Availability.UNAVAILABLE_QUOTA
    assert cached.supports_tools and cached.cost_class == "paid"
    assert health.cached("openrouter", FREE_MODEL) is None, "an uninformative failure teaches nothing"


# -- the policy applied to a run -------------------------------------------
# The real orchestrator with a scripted router whose verdicts are pre-seeded.

@pytest.fixture()
def run_settings(tmp_path) -> Settings:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return Settings(
        project_root=tmp_path, workspace_root=workspace, data_dir=tmp_path / "data",
        default_provider="openrouter", default_model=PAID_MODEL, permission_mode="trusted",
        fallback_model=FREE_MODEL,
    )


def run_with(settings: Settings, primary: Availability, turns: list) -> tuple:
    health = ProviderHealth(settings)
    health.remember(ModelCapability("openrouter", PAID_MODEL, primary, "This account has no credit.",
                                    supports_tools=True, context_length=1_000_000, cost_class="paid"))
    health.remember(ModelCapability("openrouter", FREE_MODEL, Availability.AVAILABLE,
                                    supports_tools=True, context_length=1_000_000, cost_class="free"))
    router = ScriptedRouter(turns)
    router.adapters = FakeRegistry([])  # a router with adapters gets preflighted
    orchestrator = build_orchestrator(settings, router)
    orchestrator.health = health
    task = orchestrator.store.create("Say hello")
    return orchestrator, router, asyncio.run(orchestrator.run(task))


def test_a_run_on_a_usable_primary_records_which_model_it_used(run_settings: Settings):
    orchestrator, _router, task = run_with(run_settings, Availability.AVAILABLE, [plan_turn(("Report", "report")), done_turn()])

    assert task.state is not TaskState.FAILED
    assert orchestrator.active_route == ("openrouter", PAID_MODEL)
    assert any(PAID_MODEL in event.message for event in task.events if event.kind == "thought")
    assert not any(event.kind == "fallback" for event in task.events)


def test_a_run_switches_to_the_fallback_and_names_both_models(run_settings: Settings):
    settings = dataclasses.replace(run_settings, fallback_enabled=True)

    orchestrator, _router, task = run_with(settings, Availability.UNAVAILABLE_QUOTA, [plan_turn(("Report", "report")), done_turn()])

    assert orchestrator.active_route == ("openrouter", FREE_MODEL)
    switch = [event for event in task.events if event.kind == "fallback"]
    assert len(switch) == 1, "exactly one visible fallback event"
    assert PAID_MODEL in switch[0].message and FREE_MODEL in switch[0].message and "no credit" in switch[0].message
    assert task.state is not TaskState.FAILED


def test_a_run_with_fallback_disabled_is_blocked_before_any_model_call(run_settings: Settings):
    orchestrator, router, task = run_with(run_settings, Availability.UNAVAILABLE_QUOTA, [plan_turn(("Report", "report")), done_turn()])

    assert task.state is TaskState.FAILED
    assert task.completion_status == "provider_unavailable"
    assert "did not start" in task.summary and "nothing was changed" in task.summary
    assert router.prompts == [], "not one token was spent"
    assert orchestrator.active_route is None
    assert any(event.kind == "error" and "fallback is disabled" in event.message for event in task.events)
