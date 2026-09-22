"""The shape of the HTTP surface, and who owns the services behind it.

create_app is a composition root: it builds the services once and hands the
same objects to every route group. These tests pin both halves -- that no
endpoint moved or vanished when the groups were split out, and that a group
did not quietly get a private copy of something the runtime shares.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sam_backend.app import create_app
from sam_backend.config import Settings


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    root = tmp_path_factory.mktemp("composition")
    return create_app(Settings(
        project_root=root, workspace_root=root / "workspace", data_dir=root / "data",
        default_provider="ollama", default_model="fake",
    ))


def surface(app) -> set[tuple[str, str]]:
    """Every (path, method) the app answers, including included sub-routers."""
    found: set[tuple[str, str]] = set()

    def walk(routes) -> None:
        for route in routes:
            path = getattr(route, "path", None)
            if path is None:
                walk(getattr(getattr(route, "original_router", None), "routes", []))
                continue
            for method in sorted(getattr(route, "methods", None) or ["WS"]):
                if method not in {"HEAD", "OPTIONS"}:
                    found.add((path, method))

    walk(app.routes)
    return found


# One row per route group, naming an endpoint only that group registers. If a
# group stops being registered, or its prefix drifts, exactly one row fails
# and names the culprit.
@pytest.mark.parametrize("path, method", [
    ("/api/health", "GET"),                      # system
    ("/api/config", "GET"),
    ("/api/capabilities", "GET"),
    ("/api/workspace/tree", "GET"),
    ("/api/tools/manifests", "GET"),
    ("/api/conversations", "GET"),               # conversations
    ("/api/conversations", "POST"),
    ("/api/chat", "POST"),
    ("/ws/chat", "WS"),
    ("/api/approvals", "GET"),                   # oversight
    ("/api/audit", "GET"),
    ("/api/memories", "POST"),
    ("/api/settings", "GET"),                    # settings
    ("/api/settings", "PUT"),
    ("/api/models", "GET"),                      # providers
    ("/api/router/status", "GET"),
    ("/api/cost", "GET"),
    ("/api/providers/status", "GET"),
    ("/api/providers/credentials", "POST"),
    ("/api/trading/status", "GET"),              # trading
    ("/api/tradingview/action", "POST"),
    ("/api/research/replay/start", "POST"),
    ("/api/trading/strategies", "GET"),
    ("/api/voice/capabilities", "GET"),          # voice
    ("/api/voice/speak", "POST"),
    ("/api/mt5", "GET"),                         # desktop
    ("/api/desktop/status", "GET"),
    ("/api/abort", "POST"),
    ("/ws/live", "WS"),
    ("/api/tasks", "GET"),                       # agent_api's own router
    ("/api/providers/resolution", "GET"),
])
def test_every_route_group_is_registered(app, path, method):
    assert (path, method) in surface(app)


def test_the_surface_is_whole(app):
    """A count, so an endpoint cannot disappear unnoticed between the rows above."""
    paths = {path for path, _ in surface(app)}
    assert len(paths) >= 60, f"only {len(paths)} distinct paths registered"
    assert all(p.startswith(("/api/", "/ws/", "/", "/assets")) for p in paths)


# -- one set of services, shared ------------------------------------------

def test_the_route_groups_and_the_runtime_share_one_set_of_services(app):
    """The objects the endpoints act on must be the ones the agent runs on.

    A duplicate ProviderHealth or TaskStore would not fail any single-request
    test: resolution would simply stop reflecting what a run just learned.
    """
    state = app.state
    assert state.orchestrator.health is state.router.health, "resolution and routing share one health cache"
    assert state.orchestrator.store is state.tasks, "task routes and the runtime share one store"
    assert state.agent_api.orchestrator is state.orchestrator
    assert state.orchestrator.scanner is state.scanner
    assert state.orchestrator.capabilities is state.capabilities
    assert state.agent.adapters is state.adapters


def test_a_settings_change_invalidates_the_health_the_resolution_endpoint_reads(app):
    """The settings route and the resolution route must not hold separate caches."""
    from fastapi.testclient import TestClient
    from sam_backend.provider_health import Availability, ModelCapability

    health = app.state.orchestrator.health
    health.remember(ModelCapability("ollama", "stale-model", Availability.AVAILABLE))
    with TestClient(app) as client:
        assert client.put("/api/settings", json={"default_model": "something-else"}).status_code == 200
    assert health.cached("ollama", "stale-model") is None, "the endpoint reached the shared cache"
