"""Free-tier routing: which models a run may reach, and what FREE refuses.

The promise under test is narrow and worth stating exactly: SAM will not
choose a model it knows to be paid while FREE is active, and when FREE runs
out of candidates it stops rather than quietly spending money. Everything
here is deterministic -- no test reaches a provider or the network.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from sam_backend.config import Settings
from sam_backend.models import AdapterRegistry, ErrorCategory, ModelError
from sam_backend.routing import ModelRouter, RouteChoice
from sam_backend.routing_profiles import (
    DEFAULT_PROFILE,
    Eligibility,
    eligibility,
    normalise_profile,
    parse_candidates,
    usable_free_candidates,
)
from sam_backend.schemas import SettingsUpdate

# Shaped like the real thing so the store accepts them; neither is a real key.
GROQ_SENTINEL = "gsk_FREEMODETESTSENTINELabcdefghij0123456789"
GEMINI_SENTINEL = "AIzaFREEMODETESTSENTINELabcdefghij01234"

CONFIGURED_CHAIN = [
    RouteChoice("openrouter", "anthropic/claude-sonnet-4", "configured primary"),
    RouteChoice("ollama", "qwen3.5:4b", "local"),
]


def router_with(profile: str, candidates: list[str], chain=None) -> ModelRouter:
    """A router with only the state `apply_profile` reads."""
    settings = Settings()
    settings.routing_profile = profile
    settings.free_candidates = candidates
    router = ModelRouter.__new__(ModelRouter)
    router.settings = settings
    router.failures = {}
    return router


def references(choices) -> list[str]:
    return [f"{choice.provider}/{choice.model}" for choice in choices]


# --- what "free" is allowed to mean -------------------------------------------


def test_free_eligibility_is_only_claimed_where_a_contract_says_so():
    # OpenRouter publishes the :free suffix, so both answers are knowable.
    assert eligibility("openrouter", "meta-llama/llama-3.3-70b-instruct:free") is Eligibility.FREE
    assert eligibility("openrouter", "anthropic/claude-sonnet-4") is Eligibility.PAID
    # Groq and Gemini meter free use by account, not by model, so SAM does not
    # pretend to know. A name that merely contains "free" proves nothing.
    assert eligibility("groq", "llama-3.3-70b-versatile") is Eligibility.UNVERIFIED
    assert eligibility("gemini", "gemini-2.0-flash") is Eligibility.UNVERIFIED
    assert eligibility("groq", "totally-free-model") is Eligibility.UNVERIFIED


def test_a_known_paid_model_is_refused_from_free_even_if_configured():
    """The one guarantee: FREE never selects a model published as paid."""
    usable, refused = usable_free_candidates(parse_candidates([
        "openrouter/anthropic/claude-sonnet-4", "groq/llama-3.3-70b-versatile",
    ]))

    assert [c.reference for c in usable] == ["groq/llama-3.3-70b-versatile"]
    assert len(refused) == 1 and "paid" in refused[0]["reason"]


def test_candidate_parsing_keeps_order_and_drops_duplicates():
    parsed = parse_candidates(["groq/a", "gemini/b", "GROQ/a", {"provider": "openrouter", "model": "c:free"}])

    assert [c.reference for c in parsed] == ["groq/a", "gemini/b", "openrouter/c:free"]


def test_malformed_candidate_configuration_does_not_break_loading():
    """An unreadable settings row must not stop SAM from starting."""
    assert parse_candidates(["", "no-slash", None, 7, {"provider": "groq"}]) == []
    assert parse_candidates("not a list") == []
    assert normalise_profile("nonsense") == DEFAULT_PROFILE
    assert normalise_profile(None) == DEFAULT_PROFILE
    assert normalise_profile("free") == "FREE"


# --- the three profiles -------------------------------------------------------


def test_premium_routes_exactly_as_before_profiles_existed():
    """Installing this feature must not move an existing user's route."""
    router = router_with("PREMIUM", ["groq/llama-3.3-70b-versatile"])

    assert router.apply_profile(CONFIGURED_CHAIN) == CONFIGURED_CHAIN


def test_the_backward_compatible_default_is_premium():
    settings = Settings()
    assert settings.routing_profile == "PREMIUM"
    assert settings.free_candidates == []
    # An older settings row has neither field; loading must not invent one.
    stale = Settings()
    del stale.routing_profile
    router = ModelRouter.__new__(ModelRouter)
    router.settings, router.failures = stale, {}
    assert router.apply_profile(CONFIGURED_CHAIN) == CONFIGURED_CHAIN


def test_free_uses_only_free_candidates_in_the_configured_order():
    router = router_with("FREE", ["gemini/gemini-2.0-flash", "groq/llama-3.3-70b-versatile"])

    chosen = router.apply_profile(CONFIGURED_CHAIN)

    assert references(chosen) == ["gemini/gemini-2.0-flash", "groq/llama-3.3-70b-versatile"]
    assert all(choice.provider not in {"openrouter", "ollama"} for choice in chosen)


def test_a_failing_provider_does_not_reorder_the_free_chain():
    """Order is the operator's, not a ranking SAM invented."""
    router = router_with("FREE", ["groq/a", "gemini/b"])
    router.failures = {"groq": 9}

    assert references(router.apply_profile(CONFIGURED_CHAIN)) == ["groq/a", "gemini/b"]


def test_balanced_puts_free_first_then_the_configured_chain():
    router = router_with("BALANCED", ["groq/llama-3.3-70b-versatile"])

    assert references(router.apply_profile(CONFIGURED_CHAIN)) == [
        "groq/llama-3.3-70b-versatile", "openrouter/anthropic/claude-sonnet-4", "ollama/qwen3.5:4b",
    ]


def test_balanced_does_not_call_the_same_model_twice():
    """A model in both lists is one candidate, not two attempts."""
    router = router_with("BALANCED", ["ollama/qwen3.5:4b", "groq/a"])

    assert references(router.apply_profile(CONFIGURED_CHAIN)) == [
        "ollama/qwen3.5:4b", "groq/a", "openrouter/anthropic/claude-sonnet-4",
    ]


# --- FREE fails closed --------------------------------------------------------


def test_free_with_no_candidates_refuses_rather_than_using_the_paid_chain():
    router = router_with("FREE", [])

    assert router.apply_profile(CONFIGURED_CHAIN) == []
    error = router._no_route_error()
    assert error.category is ErrorCategory.NOT_CONFIGURED
    assert "no free-tier candidates are configured" in str(error)
    assert "will not fall back to a paid model" in str(error)


def test_free_with_only_paid_candidates_says_why_it_stopped():
    router = router_with("FREE", ["openrouter/anthropic/claude-sonnet-4"])

    assert router.apply_profile(CONFIGURED_CHAIN) == []
    assert "published by its provider as paid" in str(router._no_route_error())


def test_the_premium_message_is_unchanged_when_nothing_is_configured():
    router = router_with("PREMIUM", [])
    assert "No AI model route is available" in str(router._no_route_error())


def test_profile_state_explains_the_next_run_without_secrets():
    router = router_with("FREE", ["groq/a", "openrouter/paid-model"])

    state = router.profile_state()

    assert state["routing_profile"] == "FREE"
    assert state["paid_fallback_allowed"] is False
    assert [c["reference"] for c in state["free_candidates"]] == ["groq/a"]
    assert state["refused_candidates"][0]["reference"] == "openrouter/paid-model"
    assert GROQ_SENTINEL not in str(state)


def test_balanced_reports_that_paid_fallback_is_allowed():
    assert router_with("BALANCED", ["groq/a"]).profile_state()["paid_fallback_allowed"] is True
    assert router_with("PREMIUM", []).profile_state()["paid_fallback_allowed"] is True


# --- the two new adapters -----------------------------------------------------


def answering(monkeypatch, provider: str, handler):
    """An adapter of this provider whose every request gets this answer."""
    calls = {"n": 0, "paths": [], "payloads": []}

    def handle(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        calls["paths"].append(str(request.url))
        try:
            import json as _json
            calls["payloads"].append(_json.loads(request.content or b"{}"))
        except ValueError:
            calls["payloads"].append({})
        return handler(request)

    transport = httpx.MockTransport(handle)
    original = httpx.AsyncClient

    class Patched(original):  # type: ignore[misc,valid-type]
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", Patched)
    monkeypatch.setattr("sam_backend.models.PROVIDER_RETRY_BACKOFF_SECONDS", 0.0)
    settings = Settings(groq_api_key=GROQ_SENTINEL, gemini_api_key=GEMINI_SENTINEL)
    return AdapterRegistry(settings).get(provider), calls


def ok_body(content="done", tool_calls=None, usage=None, model="the-model"):
    message = {"content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "cmpl-1", "model": model,
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": usage or {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


@pytest.mark.parametrize("provider", ["groq", "gemini"])
def test_the_new_providers_are_registered_and_require_a_key(provider):
    registry = AdapterRegistry(Settings())
    adapter = registry.get(provider)

    assert provider in registry.providers
    assert adapter.key_required is True
    assert adapter.base_url.startswith("https://")
    with pytest.raises(ModelError) as raised:
        asyncio.run(adapter.complete([{"role": "user", "content": "hi"}], [], "any-model"))
    assert raised.value.category is ErrorCategory.NOT_CONFIGURED


@pytest.mark.parametrize("provider", ["groq", "gemini"])
def test_a_successful_completion_carries_usage_and_the_model_back(monkeypatch, provider):
    adapter, calls = answering(monkeypatch, provider, lambda request: httpx.Response(
        200, json=ok_body(model="configured-model-id"), request=request))

    turn = asyncio.run(adapter.complete([{"role": "user", "content": "hi"}], [], "configured-model-id"))

    assert turn.content == "done"
    assert turn.raw["provider"] == provider
    assert turn.raw["model"] == "configured-model-id"
    assert turn.raw["usage"]["total_tokens"] == 18
    # The configured model reaches the wire; nothing is hardcoded.
    assert calls["payloads"][0]["model"] == "configured-model-id"


@pytest.mark.parametrize("provider", ["groq", "gemini"])
@pytest.mark.parametrize("status, category, retried", [
    (401, ErrorCategory.AUTH, False),
    (402, ErrorCategory.QUOTA, False),
    (429, ErrorCategory.RATE_LIMIT, True),
    (503, ErrorCategory.NETWORK, True),
])
def test_provider_failures_map_onto_one_taxonomy(monkeypatch, provider, status, category, retried):
    adapter, calls = answering(monkeypatch, provider, lambda request: httpx.Response(
        status, json={"error": {"message": "upstream said no"}}, request=request))

    with pytest.raises(ModelError) as raised:
        asyncio.run(adapter.complete([{"role": "user", "content": "hi"}], [], "m"))

    assert raised.value.category is category
    # Auth and quota do not clear by trying again; transient faults may.
    assert (calls["n"] > 1) is retried


@pytest.mark.parametrize("provider", ["groq", "gemini"])
def test_a_daily_limit_is_quota_not_a_burst_to_retry(monkeypatch, provider):
    adapter, calls = answering(monkeypatch, provider, lambda request: httpx.Response(
        429, json={"error": {"message": "rate limit reached for requests per-day"}}, request=request))

    with pytest.raises(ModelError) as raised:
        asyncio.run(adapter.complete([{"role": "user", "content": "hi"}], [], "m"))

    assert raised.value.category is ErrorCategory.QUOTA
    assert calls["n"] == 1, "a daily quota was retried as though it were transient"


@pytest.mark.parametrize("provider", ["groq", "gemini"])
def test_a_timeout_is_classified_and_bounded(monkeypatch, provider):
    def handler(request):
        raise httpx.ReadTimeout("too slow", request=request)

    adapter, calls = answering(monkeypatch, provider, handler)

    with pytest.raises(ModelError) as raised:
        asyncio.run(adapter.complete([{"role": "user", "content": "hi"}], [], "m"))

    assert raised.value.category is ErrorCategory.TIMEOUT
    assert calls["n"] <= 3, "retries are bounded by the existing policy"


@pytest.mark.parametrize("provider", ["groq", "gemini"])
def test_a_response_with_no_choices_is_malformed_not_an_empty_answer(monkeypatch, provider):
    adapter, _ = answering(monkeypatch, provider, lambda request: httpx.Response(
        200, json={"id": "x", "choices": []}, request=request))

    with pytest.raises(ModelError) as raised:
        asyncio.run(adapter.complete([{"role": "user", "content": "hi"}], [], "m"))

    assert raised.value.category is ErrorCategory.MALFORMED


@pytest.mark.parametrize("provider", ["groq", "gemini"])
def test_a_tool_call_becomes_sams_own_representation(monkeypatch, provider):
    """Provider tool-call shape must not leak past the adapter."""
    adapter, _ = answering(monkeypatch, provider, lambda request: httpx.Response(200, json=ok_body(
        content="", tool_calls=[{
            "id": "call_1", "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "README.md"}'},
        }]), request=request))

    turn = asyncio.run(adapter.complete([{"role": "user", "content": "read it"}], [], "m"))

    assert len(turn.tool_calls) == 1
    call = turn.tool_calls[0]
    assert (call.name, call.arguments) == ("read_file", {"path": "README.md"})
    assert not hasattr(call, "function"), "the provider's own shape escaped"


@pytest.mark.parametrize("provider", ["groq", "gemini"])
def test_model_discovery_fails_quietly_rather_than_breaking_status(monkeypatch, provider):
    adapter, _ = answering(monkeypatch, provider, lambda request: httpx.Response(500, request=request))

    assert asyncio.run(adapter.list_models()) == []


@pytest.mark.parametrize("provider", ["groq", "gemini"])
def test_a_credential_never_appears_in_an_error(monkeypatch, provider):
    adapter, _ = answering(monkeypatch, provider, lambda request: httpx.Response(
        401, json={"error": {"message": f"invalid key {GROQ_SENTINEL} {GEMINI_SENTINEL}"}}, request=request))

    with pytest.raises(ModelError) as raised:
        asyncio.run(adapter.complete([{"role": "user", "content": "hi"}], [], "m"))

    body = str(raised.value)
    assert GROQ_SENTINEL not in body and GEMINI_SENTINEL not in body


# --- settings, migration and runtime updates ----------------------------------


def test_the_settings_api_accepts_a_profile_and_an_ordered_candidate_list():
    update = SettingsUpdate(routing_profile="FREE", free_candidates=["Groq/a", "gemini/b", "groq/a"])

    assert update.routing_profile == "FREE"
    # The provider is case-insensitive, so the repeat collapses. The model id
    # is not: providers treat those as distinct names.
    assert update.free_candidates == ["groq/a", "gemini/b"]
    assert SettingsUpdate(free_candidates=["groq/A", "groq/a"]).free_candidates == ["groq/A", "groq/a"]


def test_a_malformed_candidate_is_rejected_rather_than_silently_dropped():
    with pytest.raises(ValueError, match="provider/model"):
        SettingsUpdate(free_candidates=["groq/a", "oops"])


def test_an_installation_that_never_sets_the_profile_loads_unchanged(tmp_path):
    settings = Settings(project_root=tmp_path, workspace_root=tmp_path / "w", data_dir=tmp_path / "d")
    settings.prepare()

    assert settings.routing_profile == "PREMIUM"
    assert settings.free_candidates == []


def test_a_stored_value_is_normalised_where_it_is_used(tmp_path):
    """Stored settings are applied after construction, so the router normalises.

    `create_app` writes saved settings onto the object with setattr, which runs
    no validator, so a hand-edited row could hold "free" or a malformed
    reference. Routing must still be correct, which is why the profile is read
    through the normalisers rather than trusted as stored.
    """
    settings = Settings(project_root=tmp_path, workspace_root=tmp_path / "w", data_dir=tmp_path / "d")
    settings.routing_profile = "free"
    settings.free_candidates = ["GROQ/a", "groq/a", "broken"]

    router = ModelRouter.__new__(ModelRouter)
    router.settings, router.failures = settings, {}

    assert router.profile_state()["routing_profile"] == "FREE"
    assert references(router.apply_profile(CONFIGURED_CHAIN)) == ["groq/a"]


def test_changing_the_profile_takes_effect_on_the_next_route_without_a_restart():
    """The same router object, asked again after settings changed."""
    router = router_with("PREMIUM", ["groq/a"])

    assert references(router.apply_profile(CONFIGURED_CHAIN)) == references(CONFIGURED_CHAIN)
    router.settings.routing_profile = "FREE"
    assert references(router.apply_profile(CONFIGURED_CHAIN)) == ["groq/a"]
    router.settings.routing_profile = "BALANCED"
    assert references(router.apply_profile(CONFIGURED_CHAIN))[0] == "groq/a"
    assert len(router.apply_profile(CONFIGURED_CHAIN)) == 3
    router.settings.free_candidates = ["gemini/b"]
    assert references(router.apply_profile(CONFIGURED_CHAIN))[0] == "gemini/b"


def test_the_new_credentials_use_the_existing_secret_path():
    from sam_backend.secrets import SUPPORTED_KEYS, KEY_PATTERNS

    assert "groq_api_key" in SUPPORTED_KEYS and "gemini_api_key" in SUPPORTED_KEYS
    assert KEY_PATTERNS["groq_api_key"].match(GROQ_SENTINEL)
    assert KEY_PATTERNS["gemini_api_key"].match(GEMINI_SENTINEL)
    # Shape checking is there to catch a paste error, not to be decorative.
    assert not KEY_PATTERNS["groq_api_key"].match("not-a-groq-key")
    assert not KEY_PATTERNS["gemini_api_key"].match("not-a-gemini-key")


# --- free web search ----------------------------------------------------------
#
# Search results are untrusted text from the open internet. They are bounded on
# every axis and returned as data; nothing here may become an instruction.

HTML_PAGE = """
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fopenai.com%2Fabout&amp;rut=x">About <b>OpenAI</b></a>
  <a class="result__snippet">OpenAI is an AI research <b>laboratory</b>.</a>
</div>
<div class="result">
  <a class="result__a" href="https://en.wikipedia.org/wiki/OpenAI">OpenAI - Wikipedia</a>
  <a class="result__snippet">An American company founded in 2015.</a>
</div>
"""


class FakeResponse:
    def __init__(self, *, text="", payload=None, status=200):
        self.text = text
        self._payload = payload if payload is not None else {}
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=None)  # type: ignore[arg-type]

    def json(self):
        return self._payload


class FakeClient:
    """Records what was asked and answers from a script."""

    def __init__(self, *, post=None, get=None):
        self._post, self._get = post, get
        self.calls: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, **kwargs):
        self.calls.append(f"POST {url}")
        if self._post is None:
            raise httpx.ConnectError("no html backend")
        return self._post

    def get(self, url, **kwargs):
        self.calls.append(f"GET {url}")
        if self._get is None:
            raise httpx.ConnectError("no json backend")
        return self._get


def searcher(**kwargs):
    from sam_backend.search import WebSearch

    client = FakeClient(**kwargs)
    return WebSearch(client_factory=lambda: client), client


def test_search_returns_structured_results_with_real_destinations():
    engine, _ = searcher(post=FakeResponse(text=HTML_PAGE))

    found = engine.search("OpenAI")

    assert [item.title for item in found] == ["About OpenAI", "OpenAI - Wikipedia"]
    # The redirect wrapper is unwrapped, so the model sees where it actually goes.
    assert found[0].url == "https://openai.com/about"
    assert found[0].snippet == "OpenAI is an AI research laboratory."
    assert all(item.url.startswith("https://") for item in found)


def test_search_bounds_the_result_count():
    engine, _ = searcher(post=FakeResponse(text=HTML_PAGE * 20))

    assert len(engine.search("OpenAI", limit=3)) == 3
    # A caller asking for more than the ceiling gets the ceiling, not a refusal.
    assert len(engine.search("OpenAI", limit=999)) == 10


def test_search_bounds_the_length_of_every_field():
    from sam_backend.search import MAX_SNIPPET_CHARS, MAX_TITLE_CHARS

    huge = (
        f'<a class="result__a" href="https://example.com/x">{"T" * 5000}</a>'
        f'<a class="result__snippet">{"S" * 9000}</a>'
    )
    engine, _ = searcher(post=FakeResponse(text=huge))

    item = engine.search("anything")[0]

    assert len(item.title) <= MAX_TITLE_CHARS
    assert len(item.snippet) <= MAX_SNIPPET_CHARS


def test_result_markup_is_stripped_so_it_cannot_arrive_as_markup():
    engine, _ = searcher(post=FakeResponse(
        text='<a class="result__a" href="https://example.com">x<script>alert(1)</script></a>'))

    item = engine.search("q")[0]

    assert "<script>" not in item.title and "alert" in item.title
    assert "<" not in item.title


def test_a_non_http_destination_is_dropped():
    """A javascript: or file: target is not a search result SAM will report."""
    engine, _ = searcher(post=FakeResponse(
        text='<a class="result__a" href="javascript:alert(1)">bad</a>'
             '<a class="result__a" href="https://ok.example">good</a>'))

    assert [item.url for item in engine.search("q")] == ["https://ok.example"]


def test_search_falls_back_to_the_json_backend_when_html_fails():
    engine, client = searcher(post=None, get=FakeResponse(payload={
        "Heading": "OpenAI", "AbstractText": "An AI lab.", "AbstractURL": "https://openai.com",
        "RelatedTopics": [{"FirstURL": "https://example.com/t", "Text": "A related topic"}],
    }))

    found = engine.search("OpenAI")

    assert [item.url for item in found] == ["https://openai.com", "https://example.com/t"]
    assert any(call.startswith("POST") for call in client.calls), "the html backend was not tried first"


def test_search_says_it_could_not_reach_a_backend_rather_than_returning_nothing():
    from sam_backend.search import SearchUnavailable

    engine, _ = searcher(post=None, get=None)

    with pytest.raises(SearchUnavailable):
        engine.search("OpenAI")


def test_an_empty_query_is_refused():
    engine, _ = searcher(post=FakeResponse(text=HTML_PAGE))

    with pytest.raises(ValueError, match="needs a query"):
        engine.search("   ")


def test_the_search_tool_is_a_network_read_not_browser_control(tmp_path):
    """Search fetches results; it does not open pages or run anything."""
    from sam_backend.db import Database
    from sam_backend.tools import ToolRegistry

    settings = Settings(project_root=tmp_path, workspace_root=tmp_path / "w", data_dir=tmp_path / "d")
    settings.prepare()
    registry = ToolRegistry(settings, Database(settings.database_path))

    manifest = next(item for item in registry.manifests if item["name"] == "web_search")

    assert manifest["permission_class"] == "network"
    assert manifest["permission_class"] != "browser"
    assert "web_search" not in {"delete_path", "run_terminal", "run_python"}


def test_the_search_tool_returns_results_as_plain_data(tmp_path):
    from sam_backend.db import Database
    from sam_backend.tools import ToolRegistry

    settings = Settings(project_root=tmp_path, workspace_root=tmp_path / "w", data_dir=tmp_path / "d")
    settings.prepare()
    engine, _ = searcher(post=FakeResponse(text=HTML_PAGE))
    registry = ToolRegistry(settings, Database(settings.database_path), search=engine)

    result = registry._web_search({"query": "OpenAI", "limit": 2}, approved=False)

    assert result.ok is True
    assert result.output["count"] == 2
    assert set(result.output["results"][0]) == {"title", "url", "snippet"}
    # Plain data: no field a caller could mistake for control metadata.
    for key in ("status", "approved", "completion_status", "verified"):
        assert key not in result.output


def test_a_search_outage_is_reported_as_a_failed_tool_call(tmp_path):
    from sam_backend.db import Database
    from sam_backend.tools import ToolRegistry

    settings = Settings(project_root=tmp_path, workspace_root=tmp_path / "w", data_dir=tmp_path / "d")
    settings.prepare()
    engine, _ = searcher(post=None, get=None)
    registry = ToolRegistry(settings, Database(settings.database_path), search=engine)

    result = registry._web_search({"query": "OpenAI"}, approved=False)

    assert result.ok is False and result.output is None
    assert "No search backend answered" in result.error


# --- the API and the page -----------------------------------------------------


def test_the_settings_api_switches_profile_without_a_restart(client):
    """The same running app, asked again after the profile changed."""
    assert client.get("/api/providers/status").json()["routing"]["routing_profile"] == "PREMIUM"

    saved = client.put("/api/settings", json={
        "routing_profile": "FREE", "free_candidates": ["groq/llama-3.3-70b-versatile"],
    })
    assert saved.status_code == 200

    routing = client.get("/api/providers/status").json()["routing"]
    assert routing["routing_profile"] == "FREE"
    assert routing["paid_fallback_allowed"] is False
    assert [c["reference"] for c in routing["free_candidates"]] == ["groq/llama-3.3-70b-versatile"]

    client.put("/api/settings", json={"routing_profile": "BALANCED"})
    after = client.get("/api/providers/status").json()["routing"]
    assert after["routing_profile"] == "BALANCED"
    assert after["paid_fallback_allowed"] is True


def test_the_status_endpoint_reports_the_new_providers_without_a_key(client):
    payload = client.get("/api/providers/status").json()

    assert payload["groq"]["status"] == "UNCONFIGURED"
    assert payload["gemini"]["status"] == "UNCONFIGURED"
    # Not configured is a state, not an error, and carries no credential.
    for provider in ("groq", "gemini"):
        assert "api_key" not in str(payload[provider]).lower().replace("_api_key", "")


def test_the_resolution_endpoint_explains_which_models_are_permitted(client):
    client.put("/api/settings", json={"routing_profile": "FREE", "free_candidates": ["groq/a"]})

    routing = client.get("/api/providers/resolution").json()["routing"]

    assert routing["routing_profile"] == "FREE"
    assert routing["paid_fallback_allowed"] is False


def test_a_stored_credential_is_never_returned_to_the_page(client):
    stored = client.post("/api/providers/credentials", json={"name": "groq_api_key", "value": GROQ_SENTINEL})
    assert stored.status_code == 200
    assert GROQ_SENTINEL not in stored.text

    for path in ("/api/providers/status", "/api/settings", "/api/providers/resolution"):
        body = client.get(path).text
        assert GROQ_SENTINEL not in body, f"{path} echoed the credential"


def test_the_page_does_not_decide_which_models_are_free():
    """Eligibility is the backend's answer; the panel only displays it."""
    from pathlib import Path

    panel = (Path(__file__).resolve().parents[1] / "frontend" / "routing-panel.js").read_text(encoding="utf-8")

    # No client-side list of "free" providers or models.
    for invented in ("freeModels", "FREE_MODELS", "isFree(", ":free\")"):
        assert invented not in panel, f"the page encodes free eligibility via {invented}"
    assert "eligibility" in panel, "the panel should render the backend's verdict"
    assert "/api/providers/status" in panel


def test_the_page_does_not_keep_a_typed_key():
    from pathlib import Path

    panel = (Path(__file__).resolve().parents[1] / "frontend" / "routing-panel.js").read_text(encoding="utf-8")

    assert 'input.value = ""' in panel, "the key field must not retain the credential"
    assert "localStorage" not in panel and "sessionStorage" not in panel


def test_no_credential_field_can_reach_the_page_by_being_forgotten():
    """Adding a provider used to mean its key shipped until a list caught up.

    `public_dict` dropped a hardcoded three; a fourth and fifth were added by
    this feature and went straight to /api/settings in plaintext. The rule is
    now the field name, so the next provider cannot repeat it.
    """
    settings = Settings(
        openai_api_key="sk-LEAKCANARY", openrouter_api_key="sk-or-LEAKCANARY",
        litellm_api_key="LEAKCANARY", groq_api_key="gsk_LEAKCANARY", gemini_api_key="AIzaLEAKCANARY",
    )

    public = settings.public_dict()

    assert not [key for key, value in public.items() if isinstance(value, str) and "LEAKCANARY" in value]
    assert not [key for key in public if key.endswith("_api_key")]
    # Whether a key is set is still answerable, which is all the page needs.
    assert public["groq_configured"] is True and public["gemini_configured"] is True
    # Names the page already reads keep working.
    assert public["openrouter_configured"] is True and public["litellm_key_configured"] is True


# --- FREE is a ceiling, not a preference --------------------------------------
#
# `route()` short-circuits when a caller names a provider explicitly, and
# AutonomousOrchestrator always does: it resolves `active_route` from
# ProviderHealth and passes that provider and model into every step. So the
# short-circuit was the path every autonomous run took, and FREE never reached
# it -- the profile only constrained the chat path that omits a provider.

async def routed(router, **kwargs):
    _, chain = await router.route("do the thing", **kwargs)
    return references(chain)


def live_router(tmp_path, profile, candidates, **overrides):
    """A real ModelRouter, so the explicit path is exercised as it ships."""
    from sam_backend.db import Database

    settings = Settings(project_root=tmp_path, workspace_root=tmp_path / "w", data_dir=tmp_path / "d",
                        openrouter_api_key="sk-or-ROUTEPROBE", **overrides)
    settings.prepare()
    settings.routing_profile, settings.free_candidates = profile, candidates
    return ModelRouter(settings, AdapterRegistry(settings), Database(settings.database_path))


def test_free_refuses_an_explicitly_named_paid_model(tmp_path):
    """The orchestrator names its provider every step; FREE must still hold."""
    router = live_router(tmp_path, "FREE", ["groq/llama-3.3-70b-versatile"])

    chain = asyncio.run(routed(router, provider="openrouter", model="anthropic/claude-sonnet-4"))

    assert "openrouter/anthropic/claude-sonnet-4" not in chain, "a paid model was reached under FREE"
    assert chain == ["groq/llama-3.3-70b-versatile"]


def test_free_with_no_candidates_refuses_an_explicit_paid_model_outright(tmp_path):
    router = live_router(tmp_path, "FREE", [])

    with pytest.raises(ModelError) as raised:
        asyncio.run(routed(router, provider="openrouter", model="anthropic/claude-sonnet-4"))

    assert raised.value.category is ErrorCategory.NOT_CONFIGURED
    assert "will not fall back to a paid model" in str(raised.value)


def test_free_still_honours_an_explicit_choice_that_is_free_eligible(tmp_path):
    """Naming a free model explicitly is allowed -- it costs nothing."""
    router = live_router(tmp_path, "FREE", ["groq/a"])

    chain = asyncio.run(routed(router, provider="openrouter", model="some/model:free"))

    assert chain == ["openrouter/some/model:free"]


def test_free_honours_an_explicit_choice_the_operator_listed_as_free(tmp_path):
    """An unverified candidate the operator configured is theirs to name."""
    router = live_router(tmp_path, "FREE", ["groq/llama-3.3-70b-versatile"])

    chain = asyncio.run(routed(router, provider="groq", model="llama-3.3-70b-versatile"))

    assert chain == ["groq/llama-3.3-70b-versatile"]


@pytest.mark.parametrize("profile", ["BALANCED", "PREMIUM"])
def test_an_explicit_choice_is_untouched_outside_free(tmp_path, profile):
    """Only FREE is a hard ceiling; the others keep the previous behaviour."""
    router = live_router(tmp_path, profile, ["groq/a"])

    chain = asyncio.run(routed(router, provider="openrouter", model="anthropic/claude-sonnet-4"))

    assert chain == ["openrouter/anthropic/claude-sonnet-4"]
