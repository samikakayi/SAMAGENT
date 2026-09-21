"""Provider preflight, model capability, and the real-model fallback policy.

No network: the OpenRouter catalogue and key endpoints are served by an
injected httpx transport, and non-OpenRouter providers go through a fake
adapter registry. The real-provider path is verified separately.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time

import httpx
import pytest

from sam_backend.config import Settings
from sam_backend.models import ErrorCategory, ModelError
from sam_backend.provider_health import (
    MINIMUM_CONTEXT_TOKENS,
    VERDICT_TTL_SECONDS,
    Availability,
    ModelCapability,
    ProviderHealth,
)

PAID_MODEL = "anthropic/claude-sonnet-4.5"
FREE_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
PROBE_KEY = "sk-or-v1-" + "a" * 48


def catalogue_entry(model: str, *, paid: bool, tools: bool = True, context: int = 1_000_000) -> dict:
    return {
        "id": model,
        "context_length": context,
        "supported_parameters": (["tools", "structured_outputs"] if tools else ["temperature"]),
        "pricing": {"prompt": "0.000003" if paid else "0", "completion": "0.000015" if paid else "0"},
    }


def openrouter_health(
    *, free_tier: bool, limit_remaining=None, models=None, key_status: int = 200,
    models_status: int = 200, transport_error: bool = False, settings: Settings | None = None,
) -> ProviderHealth:
    """A ProviderHealth whose HTTP calls are answered locally."""
    catalogue = models if models is not None else [
        catalogue_entry(PAID_MODEL, paid=True), catalogue_entry(FREE_MODEL, paid=False),
    ]

    def handle(request: httpx.Request) -> httpx.Response:
        if transport_error:
            raise httpx.ConnectError("no route to host", request=request)
        if request.url.path.endswith("/key"):
            if key_status != 200:
                return httpx.Response(key_status, json={"error": "nope"}, request=request)
            return httpx.Response(200, request=request, json={
                "data": {"is_free_tier": free_tier, "limit_remaining": limit_remaining, "usage": 0.07},
            })
        if models_status != 200:
            return httpx.Response(models_status, json={"error": "nope"}, request=request)
        return httpx.Response(200, json={"data": catalogue}, request=request)

    settings = settings or Settings(openrouter_api_key=PROBE_KEY, default_provider="openrouter",
                                    default_model=PAID_MODEL)
    health = ProviderHealth(settings)
    transport = httpx.MockTransport(handle)
    original = httpx.AsyncClient

    class PatchedClient(original):  # type: ignore[misc,valid-type]
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    health._client_factory = PatchedClient  # type: ignore[attr-defined]
    httpx.AsyncClient = PatchedClient  # type: ignore[assignment]
    health._restore = lambda: setattr(httpx, "AsyncClient", original)  # type: ignore[attr-defined]
    return health


@pytest.fixture()
def restore_httpx():
    original = httpx.AsyncClient
    yield
    httpx.AsyncClient = original


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


# -- capability representation --------------------------------------------

def test_a_capability_is_usable_only_when_available_and_able():
    base = dict(provider="openrouter", model=FREE_MODEL, availability=Availability.AVAILABLE,
                supports_tools=True, context_length=MINIMUM_CONTEXT_TOKENS)
    assert ModelCapability(**base).usable

    assert not ModelCapability(**{**base, "availability": Availability.UNAVAILABLE_QUOTA}).usable
    assert not ModelCapability(**{**base, "supports_tools": False}).usable, "a run needs tool calling"
    assert not ModelCapability(**{**base, "context_length": 4096}).usable, "a run needs room to think"


def test_a_capability_serialises_without_leaking_anything(restore_httpx):
    health = openrouter_health(free_tier=False, limit_remaining=10.0)
    capability = asyncio.run(health.capability("openrouter", FREE_MODEL, FakeRegistry([])))

    payload = json.dumps(capability.as_dict())

    assert PROBE_KEY not in payload
    assert capability.as_dict()["availability"] == "AVAILABLE"
    assert capability.as_dict()["usable"] is True


# -- preflight verdicts ----------------------------------------------------

def test_a_paid_model_on_a_funded_account_is_available(restore_httpx):
    health = openrouter_health(free_tier=False, limit_remaining=25.0)

    capability = asyncio.run(health.capability("openrouter", PAID_MODEL, FakeRegistry([])))

    assert capability.availability is Availability.AVAILABLE
    assert capability.cost_class == "paid"
    assert capability.supports_tools and capability.usable


def test_a_paid_model_on_a_free_tier_key_is_quota_blocked_without_spending(restore_httpx):
    """The whole point: knowing this costs zero tokens."""
    health = openrouter_health(free_tier=True, limit_remaining=None)

    capability = asyncio.run(health.capability("openrouter", PAID_MODEL, FakeRegistry([])))

    assert capability.availability is Availability.UNAVAILABLE_QUOTA
    assert "credit" in capability.reason
    assert not capability.usable


def test_a_free_model_on_a_free_tier_key_is_available(restore_httpx):
    health = openrouter_health(free_tier=True, limit_remaining=None)

    capability = asyncio.run(health.capability("openrouter", FREE_MODEL, FakeRegistry([])))

    assert capability.availability is Availability.AVAILABLE
    assert capability.cost_class == "free"
    assert capability.usable


def test_a_rejected_credential_is_auth_not_network(restore_httpx):
    health = openrouter_health(free_tier=True, key_status=401)

    capability = asyncio.run(health.capability("openrouter", FREE_MODEL, FakeRegistry([])))

    assert capability.availability is Availability.UNAVAILABLE_AUTH


def test_a_missing_credential_is_auth_before_any_request(restore_httpx):
    settings = Settings(openrouter_api_key=None, default_provider="openrouter", default_model=PAID_MODEL)
    health = ProviderHealth(settings)

    capability = asyncio.run(health.capability("openrouter", PAID_MODEL, FakeRegistry([])))

    assert capability.availability is Availability.UNAVAILABLE_AUTH
    assert "credential" in capability.reason


def test_an_unreachable_provider_is_network(restore_httpx):
    health = openrouter_health(free_tier=True, transport_error=True)

    capability = asyncio.run(health.capability("openrouter", FREE_MODEL, FakeRegistry([])))

    assert capability.availability is Availability.UNAVAILABLE_NETWORK


@pytest.mark.parametrize("model, models, expected_fragment", [
    ("does/not-exist", None, "catalogue"),
    ("no/tools", [catalogue_entry("no/tools", paid=False, tools=False)], "tool calling"),
    ("tiny/context", [catalogue_entry("tiny/context", paid=False, context=4096)], "context"),
])
def test_a_model_that_cannot_do_the_job_is_unsupported(restore_httpx, model, models, expected_fragment):
    """Responding to chat is not the bar; an autonomous run needs more."""
    health = openrouter_health(free_tier=True, models=models)

    capability = asyncio.run(health.capability("openrouter", model, FakeRegistry([])))

    assert capability.availability is Availability.UNSUPPORTED
    assert expected_fragment in capability.reason


def test_a_local_provider_is_checked_against_what_it_serves(restore_httpx):
    settings = Settings(default_provider="ollama", default_model="qwen3.5:4b")
    health = ProviderHealth(settings)

    served = asyncio.run(health.capability("ollama", "qwen3.5:4b", FakeRegistry(["qwen3.5:4b"])))
    missing = asyncio.run(health.capability("ollama", "absent:1b", FakeRegistry(["qwen3.5:4b"])))
    # A fresh instance: the first verdict above is (correctly) still cached.
    down = asyncio.run(ProviderHealth(settings).capability("ollama", "qwen3.5:4b", FakeRegistry([], fail=True)))

    assert served.availability is Availability.AVAILABLE and served.cost_class == "free"
    assert missing.availability is Availability.UNSUPPORTED
    assert down.availability is Availability.UNAVAILABLE_NETWORK


# -- the fallback policy ---------------------------------------------------

def resolution_for(*, free_tier: bool, fallback: str, enabled: bool, restore=None):
    settings = Settings(
        openrouter_api_key=PROBE_KEY, default_provider="openrouter", default_model=PAID_MODEL,
        fallback_model=fallback, fallback_enabled=enabled,
    )
    health = openrouter_health(free_tier=free_tier, settings=settings)
    return asyncio.run(health.resolve(FakeRegistry([])))


def test_a_usable_primary_is_used_and_no_fallback_is_engaged(restore_httpx):
    settings = Settings(openrouter_api_key=PROBE_KEY, default_provider="openrouter",
                        default_model=PAID_MODEL, fallback_model=FREE_MODEL, fallback_enabled=True)
    health = openrouter_health(free_tier=False, limit_remaining=25.0, settings=settings)

    resolution = asyncio.run(health.resolve(FakeRegistry([])))

    assert resolution.active.model == PAID_MODEL
    assert resolution.as_dict()["fallback_engaged"] is False
    assert not resolution.blocked


def test_a_quota_blocked_primary_switches_to_the_configured_real_fallback(restore_httpx):
    resolution = resolution_for(free_tier=True, fallback=FREE_MODEL, enabled=True)

    assert resolution.primary.availability is Availability.UNAVAILABLE_QUOTA
    assert resolution.active is not None and resolution.active.model == FREE_MODEL
    assert resolution.as_dict()["fallback_engaged"] is True
    # The reason names the primary and why it could not be used.
    assert PAID_MODEL in resolution.fallback_reason and "credit" in resolution.fallback_reason
    assert not resolution.blocked


def test_with_fallback_disabled_the_run_is_blocked_rather_than_switched(restore_httpx):
    """Silently changing model is exactly what this policy prevents."""
    resolution = resolution_for(free_tier=True, fallback=FREE_MODEL, enabled=False)

    assert resolution.blocked and resolution.active is None
    assert "fallback is disabled" in resolution.fallback_reason
    # The usable fallback was still assessed, so the UI can offer it.
    assert resolution.fallback is not None and resolution.fallback.usable


def test_no_configured_fallback_blocks_even_when_enabled(restore_httpx):
    resolution = resolution_for(free_tier=True, fallback="", enabled=True)

    assert resolution.blocked
    assert "no fallback is configured" in resolution.fallback_reason


def test_an_unusable_fallback_does_not_rescue_the_run(restore_httpx):
    settings = Settings(
        openrouter_api_key=PROBE_KEY, default_provider="openrouter", default_model=PAID_MODEL,
        fallback_model="no/tools", fallback_enabled=True,
    )
    health = openrouter_health(
        free_tier=True, settings=settings,
        models=[catalogue_entry(PAID_MODEL, paid=True), catalogue_entry("no/tools", paid=False, tools=False)],
    )

    resolution = asyncio.run(health.resolve(FakeRegistry([])))

    assert resolution.blocked, "a fallback that cannot call tools is not a fallback"


def test_an_unknown_verdict_proceeds_rather_than_inventing_an_outage(restore_httpx):
    """Preflight avoids wasted work; it must never become a false blocker."""
    settings = Settings(default_provider="mystery", default_model="mystery/model", fallback_enabled=False)
    health = ProviderHealth(settings)
    health.remember(ModelCapability("mystery", "mystery/model", Availability.UNKNOWN, "could not tell"))

    resolution = asyncio.run(health.resolve(FakeRegistry([])))

    assert not resolution.blocked
    assert resolution.active is not None and resolution.active.model == "mystery/model"


# -- the cache -------------------------------------------------------------

def test_a_confirmed_quota_failure_is_not_re_probed_immediately(restore_httpx):
    """Re-probing a paid model after a credit failure is the waste to avoid."""
    calls = {"count": 0}

    def handle(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if request.url.path.endswith("/key"):
            return httpx.Response(200, request=request,
                                  json={"data": {"is_free_tier": True, "limit_remaining": None}})
        return httpx.Response(200, request=request, json={"data": [catalogue_entry(PAID_MODEL, paid=True)]})

    settings = Settings(openrouter_api_key=PROBE_KEY, default_provider="openrouter", default_model=PAID_MODEL)
    health = ProviderHealth(settings)
    original = httpx.AsyncClient
    transport = httpx.MockTransport(handle)

    class Patched(original):  # type: ignore[misc,valid-type]
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    httpx.AsyncClient = Patched  # type: ignore[assignment]
    first = asyncio.run(health.capability("openrouter", PAID_MODEL, FakeRegistry([])))
    after_first = calls["count"]
    second = asyncio.run(health.capability("openrouter", PAID_MODEL, FakeRegistry([])))

    assert first.availability is Availability.UNAVAILABLE_QUOTA
    assert second.availability is Availability.UNAVAILABLE_QUOTA
    assert calls["count"] == after_first, "the second check was served from cache"


def test_transient_verdicts_expire_sooner_than_quota_and_auth():
    assert VERDICT_TTL_SECONDS[Availability.UNAVAILABLE_RATE_LIMIT] < VERDICT_TTL_SECONDS[Availability.UNAVAILABLE_QUOTA]
    assert VERDICT_TTL_SECONDS[Availability.UNAVAILABLE_NETWORK] < VERDICT_TTL_SECONDS[Availability.UNAVAILABLE_AUTH]
    assert all(ttl > 0 for ttl in VERDICT_TTL_SECONDS.values()), "nothing is cached forever"


def test_a_cached_verdict_expires_and_is_checked_again():
    health = ProviderHealth(Settings())
    stale = health.remember(ModelCapability("openrouter", FREE_MODEL, Availability.UNAVAILABLE_NETWORK, "blip"))
    assert health.cached("openrouter", FREE_MODEL) is not None

    stale.checked_at = time.time() - VERDICT_TTL_SECONDS[Availability.UNAVAILABLE_NETWORK] - 1

    assert health.cached("openrouter", FREE_MODEL) is None, "a transient verdict must not stick"


def test_a_real_request_failure_teaches_the_cache():
    health = ProviderHealth(Settings())
    health.remember(ModelCapability("openrouter", PAID_MODEL, Availability.AVAILABLE,
                                    supports_tools=True, context_length=1_000_000, cost_class="paid"))

    health.record_failure("openrouter", PAID_MODEL, ModelError("needs more credits", ErrorCategory.QUOTA))

    cached = health.cached("openrouter", PAID_MODEL)
    assert cached is not None and cached.availability is Availability.UNAVAILABLE_QUOTA
    # What the catalogue already taught us is kept; only the verdict changed.
    assert cached.supports_tools and cached.cost_class == "paid"


def test_an_uninformative_failure_does_not_poison_the_cache():
    health = ProviderHealth(Settings())
    health.remember(ModelCapability("openrouter", FREE_MODEL, Availability.AVAILABLE,
                                    supports_tools=True, context_length=1_000_000))

    health.record_failure("openrouter", FREE_MODEL, ModelError("weird", ErrorCategory.MALFORMED))

    cached = health.cached("openrouter", FREE_MODEL)
    assert cached is not None and cached.availability is Availability.AVAILABLE


def test_the_cache_can_be_cleared_wholesale(restore_httpx):
    health = ProviderHealth(Settings())
    health.remember(ModelCapability("openrouter", FREE_MODEL, Availability.AVAILABLE))

    health.invalidate()

    assert health.cached("openrouter", FREE_MODEL) is None


# -- the policy applied to a run -------------------------------------------
# These drive the real orchestrator with a scripted router whose health
# verdicts are pre-seeded, so the run's own behaviour is what is tested.

from sam_backend.tasks import TaskState  # noqa: E402
from tests.test_autonomy import ScriptedRouter, build_orchestrator, done_turn, plan_turn  # noqa: E402


@pytest.fixture()
def run_settings(tmp_path) -> Settings:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return Settings(
        project_root=tmp_path, workspace_root=workspace, data_dir=tmp_path / "data",
        default_provider="openrouter", default_model=PAID_MODEL, permission_mode="trusted",
    )


def with_fallback(settings: Settings, enabled: bool) -> Settings:
    return dataclasses.replace(settings, fallback_model=FREE_MODEL, fallback_enabled=enabled)


def seeded_health(settings: Settings, primary: Availability, primary_reason: str = "") -> ProviderHealth:
    health = ProviderHealth(settings)
    health.remember(ModelCapability(
        settings.default_provider, settings.default_model, primary, primary_reason,
        supports_tools=True, context_length=1_000_000, cost_class="paid",
    ))
    if settings.fallback_model:
        health.remember(ModelCapability(
            settings.default_provider, settings.fallback_model, Availability.AVAILABLE,
            supports_tools=True, context_length=1_000_000, cost_class="free",
        ))
    return health


def run_with(settings: Settings, health: ProviderHealth, turns: list) -> tuple:
    router = ScriptedRouter(turns)
    router.adapters = FakeRegistry([])  # a router with real adapters gets preflighted
    orchestrator = build_orchestrator(settings, router)
    orchestrator.health = health
    task = orchestrator.store.create("Say hello")
    return orchestrator, router, asyncio.run(orchestrator.run(task))


def test_a_run_on_a_usable_primary_records_which_model_it_used(run_settings: Settings):
    settings = run_settings
    health = seeded_health(settings, Availability.AVAILABLE)

    orchestrator, _router, task = run_with(settings, health, [plan_turn(("Report", "report")), done_turn()])

    assert task.state is not TaskState.FAILED
    assert orchestrator.active_route == ("openrouter", PAID_MODEL)
    assert any(PAID_MODEL in event.message for event in task.events if event.kind == "thought")
    assert not any(event.kind == "fallback" for event in task.events), "no fallback was engaged"


def test_a_run_switches_to_the_fallback_and_names_both_models(run_settings: Settings):
    settings = with_fallback(run_settings, enabled=True)
    health = seeded_health(settings, Availability.UNAVAILABLE_QUOTA, "This account has no credit.")

    orchestrator, router, task = run_with(settings, health, [plan_turn(("Report", "report")), done_turn()])

    assert orchestrator.active_route == ("openrouter", FREE_MODEL)
    switch = [event for event in task.events if event.kind == "fallback"]
    assert len(switch) == 1, "exactly one visible fallback event"
    assert PAID_MODEL in switch[0].message and FREE_MODEL in switch[0].message
    assert "no credit" in switch[0].message
    assert task.state is not TaskState.FAILED


def test_a_run_with_fallback_disabled_is_blocked_before_any_model_call(run_settings: Settings):
    settings = with_fallback(run_settings, enabled=False)
    health = seeded_health(settings, Availability.UNAVAILABLE_QUOTA, "This account has no credit.")

    orchestrator, router, task = run_with(settings, health, [plan_turn(("Report", "report")), done_turn()])

    assert task.state is TaskState.FAILED
    assert task.completion_status == "provider_unavailable"
    assert "did not start" in task.summary and "nothing was changed" in task.summary
    assert router.prompts == [], "not one token was spent"
    assert orchestrator.active_route is None
    blocked = [event for event in task.events if event.kind == "error"]
    assert blocked and "fallback is disabled" in blocked[0].message


def test_an_auth_failure_during_a_run_blocks_the_next_run_without_a_probe(run_settings: Settings):
    """A rejected credential is remembered: the second run does not re-probe
    and does not spend a token finding out the same thing again."""
    settings = run_settings
    health = seeded_health(settings, Availability.AVAILABLE)
    failure = ModelError("401 Unauthorized", ErrorCategory.AUTH)

    _orch, router, first = run_with(settings, health, [plan_turn(("Do a thing", "edit")), failure])
    assert first.state is TaskState.FAILED and len(router.prompts) == 2, "the first run really tried"

    cached = health.cached("openrouter", PAID_MODEL)
    assert cached is not None and cached.availability is Availability.UNAVAILABLE_AUTH

    _orch, router, second = run_with(settings, health, [plan_turn(("Report", "report")), done_turn()])
    assert second.completion_status == "provider_unavailable"
    assert router.prompts == [], "the cached auth verdict stopped the run before any request"


def test_the_fallback_is_a_real_model_on_the_same_provider(run_settings: Settings):
    """The fallback is whatever the operator configured on the SAME real
    provider. There is no code path that substitutes a scripted model: the
    resolution only ever holds capabilities built from provider adapters."""
    from sam_backend.models import AdapterRegistry

    settings = with_fallback(run_settings, enabled=True)
    health = seeded_health(settings, Availability.UNAVAILABLE_QUOTA)

    resolution = asyncio.run(health.resolve(AdapterRegistry(settings)))

    assert resolution.active is not None
    assert resolution.active.provider == resolution.primary.provider == "openrouter"
    assert resolution.active.model == FREE_MODEL


def test_automatic_routing_has_nothing_to_preflight_and_is_not_blocked():
    """default_provider may be "auto": the router picks per request, so a
    preflight cannot name a model to check and must not report a fault."""
    settings = Settings(default_provider="auto", default_model="anything", fallback_enabled=False)

    resolution = asyncio.run(ProviderHealth(settings).resolve(FakeRegistry([], fail=True)))

    assert resolution.primary.availability is Availability.UNKNOWN
    assert not resolution.blocked


def test_a_daily_cap_is_remembered_as_quota_not_a_burst_limit():
    """Seen live: OpenRouter answers 429 "free-models-per-day". Trusting that
    for only sixty seconds would let a new run fail the same way every minute."""
    health = ProviderHealth(Settings())

    health.record_failure("openrouter", FREE_MODEL,
                          ModelError("Rate limit exceeded: free-models-per-day. Add 10 credits", ErrorCategory.RATE_LIMIT))
    daily = health.cached("openrouter", FREE_MODEL)
    health.record_failure("openrouter", PAID_MODEL, ModelError("Rate limit exceeded: burst", ErrorCategory.RATE_LIMIT))
    burst = health.cached("openrouter", PAID_MODEL)

    assert daily is not None and daily.availability is Availability.UNAVAILABLE_QUOTA
    assert burst is not None and burst.availability is Availability.UNAVAILABLE_RATE_LIMIT


def test_an_unavailable_fallback_is_named_as_such_not_called_missing(run_settings: Settings):
    settings = with_fallback(run_settings, enabled=True)
    health = seeded_health(settings, Availability.UNAVAILABLE_QUOTA, "no credit")
    health.record_failure("openrouter", FREE_MODEL, ModelError("free-models-per-day", ErrorCategory.RATE_LIMIT))

    resolution = asyncio.run(health.resolve(FakeRegistry([])))

    assert resolution.blocked
    assert FREE_MODEL in resolution.fallback_reason and "also unavailable" in resolution.fallback_reason
    assert "no fallback is configured" not in resolution.fallback_reason
