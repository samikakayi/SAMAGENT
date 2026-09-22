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
    ("/api/trading/status", "GET"),              # trading: market
    ("/api/trading/analyze", "POST"),
    ("/api/trading/gann", "GET"),
    ("/api/trading/setups", "GET"),              # trading: record
    ("/api/trading/journal", "POST"),
    ("/api/tradingview/action", "POST"),         # trading: chart
    ("/api/tradingview/drawings", "GET"),
    ("/api/tradingview/layers", "POST"),
    ("/api/research/replay/start", "POST"),      # trading: replay
    ("/api/research/replay/scan", "POST"),
    ("/api/trading/strategies", "GET"),          # trading: strategies
    ("/api/trading/backtest", "POST"),
    ("/api/trading/triggers", "GET"),
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


def test_no_endpoint_is_registered_twice(app):
    """Splitting a route group is exactly how a duplicate registration happens."""
    seen = [(route.path, method)
            for route in app.routes
            for method in sorted(getattr(route, "methods", None) or [])
            if getattr(route, "path", "").startswith(("/api/trading", "/api/tradingview", "/api/research"))]
    duplicates = {pair for pair in seen if seen.count(pair) > 1}
    assert not duplicates, f"registered more than once: {sorted(duplicates)}"
    assert len(seen) >= 34, f"only {len(seen)} trading route entries survived the split"


def test_the_trading_groups_share_one_service(app):
    """Creating a setup reads analysis state the market group wrote.

    Separate TradingService instances would make POST /setups refuse work
    that had just succeeded, with nothing in the response to explain it.
    """
    assert app.state.trading is app.state.tools.trading
    assert app.state.replay.market_data is app.state.trading.market_data,         "replay research reads the same market data the analysis routes do"


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


# -- one live adapter registry --------------------------------------------

def registry_identities(app) -> dict[str, int]:
    """Who each consumer would reach for an adapter, by object identity."""
    state = app.state
    return {
        "app.state": id(state.adapters),
        "router": id(state.router.adapters),
        "agent": id(state.agent.adapters),
        "agent_api": id(state.agent_api._adapters()),
        "orchestrator": id(state.orchestrator.router.adapters),
    }


def test_every_adapter_consumer_tracks_a_runtime_rebuild(tmp_path):
    """A new credential rebuilds the registry; nothing may keep the old one.

    A consumer left on the startup registry would report on adapters built
    before the key existed -- the model list and the routing status would
    disagree with what a run actually uses, with no error to show for it.
    """
    from fastapi.testclient import TestClient

    root = tmp_path
    app = create_app(Settings(
        project_root=root, workspace_root=root / "workspace", data_dir=root / "data",
        default_provider="ollama", default_model="fake",
    ))
    before = registry_identities(app)
    assert len(set(before.values())) == 1, f"consumers disagree before any rebuild: {before}"
    original = next(iter(before.values()))

    with TestClient(app) as client:
        stored = client.post("/api/providers/credentials", json={
            "name": "openrouter_api_key", "value": "sk-or-v1-REBUILD-TEST-SENTINEL-0123456789",
        })
        assert stored.status_code == 200, stored.text

        after = registry_identities(app)
        assert len(set(after.values())) == 1, f"consumers split after the rebuild: {after}"
        assert next(iter(after.values())) != original, "the rebuild really did replace the registry"

        # Identity alone would not have caught the real defect: two routes
        # held the startup registry in a closure. Mark the registry the app
        # now considers current and check the endpoints actually report on it.
        class MarkerAdapter:
            async def list_models(self):
                return [{"id": "FROM-THE-CURRENT-REGISTRY", "name": "marker"}]

        current = app.state.adapters
        for provider in current.providers:
            current.adapters[provider] = MarkerAdapter()

        listed = client.get("/api/models").json()["providers"]
        seen = {item.get("id") for items in listed.values() for item in items}
        assert seen == {"FROM-THE-CURRENT-REGISTRY"}, f"/api/models read a stale registry: {seen}"

        # These two disagreed before the fix: one read the startup registry,
        # the other application.state, in the same moment.
        status = client.get("/api/router/status").json()["configured"]
        probed = client.get("/api/providers/status").json()
        assert status["litellm"] is True
        assert probed["litellm"]["status"] == "CONNECTED", "status routes must agree with discovery"


def test_a_failed_rebuild_leaves_every_consumer_on_the_working_registry(tmp_path, monkeypatch):
    """A bad credential update must not strand half the app on a new object."""
    from fastapi.testclient import TestClient
    import sam_backend.app as app_module

    root = tmp_path
    app = create_app(Settings(
        project_root=root, workspace_root=root / "workspace", data_dir=root / "data",
        default_provider="ollama", default_model="fake",
    ))
    before = registry_identities(app)

    def refuse(_settings):
        raise RuntimeError("adapter construction failed")

    monkeypatch.setattr(app_module, "AdapterRegistry", refuse)
    with TestClient(app, raise_server_exceptions=False) as client:
        client.post("/api/providers/credentials", json={
            "name": "openrouter_api_key", "value": "sk-or-v1-FAILED-REBUILD-SENTINEL-0123456789",
        })

    after = registry_identities(app)
    assert after == before, f"a failed rebuild split ownership: {before} -> {after}"
    assert len(set(after.values())) == 1
