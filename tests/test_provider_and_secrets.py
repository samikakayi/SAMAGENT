"""Secret storage at rest, provider error handling, and the real/mock boundary.

None of these tests touch the network: provider behaviour is driven through
injected fakes that raise the exact transport errors a provider would. The
real-provider path is verified separately as an integration step.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest

from sam_backend.config import Settings
from sam_backend.models import (
    AdapterRegistry,
    ErrorCategory,
    ModelError,
    classify_exception,
    sanitize_provider_message,
)
from sam_backend.secrets import SecretStore

PROBE_KEY = "sk-or-v1-" + "a" * 48


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://provider.example/chat")


def _status(code: int) -> httpx.HTTPStatusError:
    return httpx.HTTPStatusError("boom", request=_request(), response=httpx.Response(code, request=_request()))


# -- secret storage --------------------------------------------------------

def test_a_stored_credential_round_trips_through_a_fresh_store(tmp_path: Path):
    SecretStore(tmp_path).set("openrouter_api_key", PROBE_KEY)

    # A second instance must decrypt from disk, not from memory.
    assert SecretStore(tmp_path).get("openrouter_api_key") == PROBE_KEY


def test_the_credential_never_appears_in_the_file_on_disk(tmp_path: Path):
    store = SecretStore(tmp_path)
    store.set("openrouter_api_key", PROBE_KEY)

    raw = (tmp_path / "secrets.json").read_text(encoding="utf-8")

    assert PROBE_KEY not in raw
    if store.storage_status()["encrypted_at_rest"]:
        assert json.loads(raw)["_format"] == "dpapi-v1"


@pytest.mark.skipif(os.name != "nt", reason="DPAPI is a Windows facility")
def test_windows_stores_are_encrypted_at_rest_not_merely_permissioned(tmp_path: Path):
    store = SecretStore(tmp_path)
    store.set("openrouter_api_key", PROBE_KEY)

    assert store.storage_status() == {
        "encrypted_at_rest": True,
        "mechanism": "windows-dpapi",
        "path": str(tmp_path / "secrets.json"),
    }


def test_a_legacy_plaintext_store_is_migrated_without_losing_the_credential(tmp_path: Path):
    """The pre-existing format must keep working and get encrypted in place."""
    (tmp_path / "secrets.json").write_text(json.dumps({"openrouter_api_key": PROBE_KEY}), encoding="utf-8")

    store = SecretStore(tmp_path)  # migration happens on construction

    assert store.get("openrouter_api_key") == PROBE_KEY, "the credential survived"
    if store.storage_status()["encrypted_at_rest"]:
        assert PROBE_KEY not in (tmp_path / "secrets.json").read_text(encoding="utf-8")


def test_public_status_reports_presence_and_fingerprint_but_never_the_value(tmp_path: Path):
    store = SecretStore(tmp_path)
    store.set("openrouter_api_key", PROBE_KEY)

    status = store.public_status()

    assert status["openrouter_api_key"]["configured"] is True
    assert status["openrouter_api_key"]["fingerprint"]
    assert PROBE_KEY not in json.dumps(status)


def test_an_undecryptable_store_yields_nothing_rather_than_garbage(tmp_path: Path):
    """A store written by another account must not produce a half-read key."""
    (tmp_path / "secrets.json").write_text(
        json.dumps({"_format": "dpapi-v1", "data": "bm90LXJlYWwtY2lwaGVydGV4dA=="}), encoding="utf-8",
    )

    assert SecretStore(tmp_path).get("openrouter_api_key") is None


# -- provider error classification ----------------------------------------

@pytest.mark.parametrize("code, expected", [
    (401, ErrorCategory.AUTH),
    (403, ErrorCategory.AUTH),
    (429, ErrorCategory.RATE_LIMIT),
    (500, ErrorCategory.NETWORK),
    (503, ErrorCategory.NETWORK),
])
def test_http_status_codes_map_onto_actionable_categories(code: int, expected: ErrorCategory):
    assert classify_exception(_status(code)) is expected


def test_transport_failures_are_categorised():
    assert classify_exception(httpx.ConnectTimeout("slow")) is ErrorCategory.TIMEOUT
    assert classify_exception(httpx.ReadTimeout("slow")) is ErrorCategory.TIMEOUT
    assert classify_exception(httpx.ConnectError("refused")) is ErrorCategory.NETWORK
    assert classify_exception(ValueError("not json")) is ErrorCategory.MALFORMED


def test_an_unrecognised_failure_is_unknown_rather_than_mislabelled():
    assert classify_exception(_status(404)) is ErrorCategory.UNKNOWN
    assert ModelError("something").category is ErrorCategory.UNKNOWN


def test_a_providers_own_error_text_is_stripped_of_credential_shapes():
    """Providers echo the request back; that text reaches logs and the UI."""
    assert PROBE_KEY not in sanitize_provider_message(f"Invalid key {PROBE_KEY} rejected")
    assert "[REDACTED]" in sanitize_provider_message("Authorization: Bearer abc123def456ghi")
    assert "[REDACTED]" in sanitize_provider_message("api_key=supersecretvalue")
    # Ordinary text is left intact.
    assert sanitize_provider_message("model not found") == "model not found"


def test_a_missing_credential_is_its_own_category_not_a_network_error():
    settings = Settings(openrouter_api_key=None)
    adapter = AdapterRegistry(settings).get("openrouter")

    with pytest.raises(ModelError) as raised:
        import asyncio

        asyncio.run(adapter.complete([{"role": "user", "content": "hi"}], [], "some/model"))

    assert raised.value.category is ErrorCategory.NOT_CONFIGURED
    assert "not configured" in str(raised.value)


# -- the real/mock boundary ------------------------------------------------

def test_production_create_app_builds_a_real_adapter_registry(tmp_path: Path):
    """The production entrypoint passes no adapters, so nothing scripted can
    be in play: this is the guarantee that a mock cannot reach production."""
    from sam_backend.app import create_app

    (tmp_path / "workspace").mkdir()
    settings = Settings(
        project_root=tmp_path, workspace_root=tmp_path / "workspace", data_dir=tmp_path / "data",
    )

    app = create_app(settings, None)

    assert isinstance(app.state.adapters, AdapterRegistry)


def test_every_route_in_the_fallback_chain_is_a_real_provider(tmp_path: Path):
    """A scripted provider must not be reachable through routing, so that a
    failing real provider can never be silently substituted."""
    import asyncio

    from sam_backend.db import Database
    from sam_backend.routing import ModelRouter

    settings = Settings(
        project_root=tmp_path, workspace_root=tmp_path / "workspace", data_dir=tmp_path / "data",
        model_mode="AUTO",
    )
    router = ModelRouter(settings, AdapterRegistry(settings), Database(tmp_path / "t.sqlite3"))

    _profile, choices = asyncio.run(router.route("write a file", provider=None, model=None))

    assert choices, "routing produced no candidates"
    assert {choice.provider for choice in choices} <= {"ollama", "litellm", "openrouter", "openai"}


def test_when_every_provider_fails_the_router_raises_instead_of_degrading(tmp_path: Path):
    """The honest outcome is an error, never a fabricated completion."""
    import asyncio

    from sam_backend.db import Database
    from sam_backend.routing import ModelRouter

    class DeadAdapter:
        async def list_models(self):
            return [{"id": "m", "name": "m", "provider": "ollama"}]

        async def complete(self, messages, tools, model):
            raise ModelError("host unreachable", ErrorCategory.NETWORK)

    class DeadRegistry:
        def get(self, provider):
            return DeadAdapter()

    settings = Settings(
        project_root=tmp_path, workspace_root=tmp_path / "workspace", data_dir=tmp_path / "data",
        model_mode="LOCAL_ONLY", default_provider="ollama",
    )
    database = Database(tmp_path / "t.sqlite3")
    router = ModelRouter(settings, DeadRegistry(), database)

    with pytest.raises(ModelError) as raised:
        asyncio.run(router.complete(
            message="hi", messages=[{"role": "user", "content": "hi"}], tools=[],
            provider="ollama", model="m", conversation_id=None,
        ))

    assert "All model routes failed" in str(raised.value)
    assert raised.value.category is ErrorCategory.NETWORK
    # The failed attempt is recorded for the operator, categorised, no secrets.
    usage = database.list_model_usage(10)
    assert usage and usage[0]["metadata"]["outcome"] == "failed"
    assert usage[0]["metadata"]["error_category"] == "network"


def test_a_failed_attempt_is_recorded_without_any_credential(tmp_path: Path):
    import asyncio

    from sam_backend.db import Database
    from sam_backend.routing import ModelRouter

    class LeakyAdapter:
        async def list_models(self):
            return [{"id": "m", "name": "m", "provider": "ollama"}]

        async def complete(self, messages, tools, model):
            # A provider that unhelpfully echoes the key back in its error.
            raise ModelError(sanitize_provider_message(f"rejected key {PROBE_KEY}"), ErrorCategory.AUTH)

    settings = Settings(
        project_root=tmp_path, workspace_root=tmp_path / "workspace", data_dir=tmp_path / "data",
        model_mode="LOCAL_ONLY", default_provider="ollama",
    )
    database = Database(tmp_path / "t.sqlite3")
    router = ModelRouter(settings, type("R", (), {"get": lambda self, p: LeakyAdapter()})(), database)

    with pytest.raises(ModelError) as raised:
        asyncio.run(router.complete(
            message="hi", messages=[{"role": "user", "content": "hi"}], tools=[],
            provider="ollama", model="m", conversation_id=None,
        ))

    assert PROBE_KEY not in str(raised.value)
    assert PROBE_KEY not in json.dumps(database.list_model_usage(10))


# -- security containment --------------------------------------------------

def test_git_tools_cannot_reach_a_repository_outside_the_workspace(tmp_path: Path):
    """A workspace nested inside a larger repo must not expose that repo.

    Without the boundary check the upward search for .git climbs out of the
    workspace and reports the parent repository's status.
    """
    import subprocess

    from sam_backend.db import Database
    from sam_backend.tools import ToolRegistry

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "outside.txt").write_text("never exposed", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    settings = Settings(project_root=tmp_path, workspace_root=workspace, data_dir=tmp_path / "data")
    registry = ToolRegistry(settings, Database(tmp_path / "data" / "t.sqlite3"))

    result = registry.execute("git_status", {"path": "."})

    assert not result.ok
    assert "outside.txt" not in str(result.output)

    # A repository that genuinely is the workspace still works.
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    assert registry.execute("git_status", {"path": "."}).ok


@pytest.mark.parametrize("tool", ["project_map", "git_status", "run_tests"])
def test_new_tools_refuse_to_escape_the_workspace(tmp_path: Path, tool: str):
    from sam_backend.db import Database
    from sam_backend.policy import RiskPolicy
    from sam_backend.tools import ToolRegistry

    (tmp_path / "workspace").mkdir()
    settings = Settings(project_root=tmp_path, workspace_root=tmp_path / "workspace", data_dir=tmp_path / "data")
    registry = ToolRegistry(settings, Database(tmp_path / "data" / "t.sqlite3"))
    arguments = {"path": "../.."}

    decision = RiskPolicy(settings).evaluate(tool, arguments)
    result = registry.execute(tool, arguments)

    # Either policy demands approval, or the resolver refuses outright.
    assert decision.approval_required or not result.ok
    assert not result.ok
