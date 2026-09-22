"""API wiring for credentials, providers, replay, strategies, and the new panels."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SENTINEL = "sk-or-v1-APITEST-SENTINEL-abcdefghij0123456789"


# --- Credential endpoint ------------------------------------------------------


def test_storing_a_credential_never_returns_it(client, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    response = client.post("/api/providers/credentials",
                           json={"name": "openrouter_api_key", "value": SENTINEL})
    assert response.status_code == 200
    body = response.text
    assert SENTINEL not in body, "the credential was echoed back to the client"
    payload = response.json()
    assert payload["stored"] is True
    assert payload["fingerprint"] and payload["fingerprint"] not in SENTINEL
    assert payload["reloaded_without_restart"] is True


def test_the_stored_credential_is_absent_from_every_read_endpoint(client, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client.post("/api/providers/credentials", json={"name": "openrouter_api_key", "value": SENTINEL})
    for path in ("/api/config", "/api/settings", "/api/providers/status", "/api/health", "/api/audit"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert SENTINEL not in response.text, f"{path} leaked the credential"


def test_the_audit_entry_records_the_fingerprint_not_the_key(client, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client.post("/api/providers/credentials", json={"name": "openrouter_api_key", "value": SENTINEL})
    audit = client.get("/api/audit").text
    assert SENTINEL not in audit
    assert "credentials" in audit


def test_a_malformed_credential_is_rejected(client):
    response = client.post("/api/providers/credentials",
                           json={"name": "openrouter_api_key", "value": "obviously-not-a-key"})
    assert response.status_code == 400


def test_an_unsupported_credential_name_is_rejected(client):
    response = client.post("/api/providers/credentials", json={"name": "aws_secret", "value": SENTINEL})
    assert response.status_code == 422


def test_a_credential_can_be_cleared(client, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client.post("/api/providers/credentials", json={"name": "openrouter_api_key", "value": SENTINEL})
    response = client.delete("/api/providers/credentials/openrouter_api_key")
    assert response.status_code == 200
    assert response.json()["cleared"] is True
    assert response.json()["credentials"]["openrouter_api_key"]["configured"] is False


# --- Provider status ----------------------------------------------------------


def test_provider_status_classifies_each_provider(client):
    payload = client.get("/api/providers/status").json()
    for provider in ("openrouter", "ollama", "litellm"):
        assert provider in payload
    assert payload["openrouter"]["status"] in {"CONNECTED", "UNCONFIGURED", "AUTH_FAILED", "RATE_LIMITED", "ERROR"}
    assert payload["ollama"]["status"] in {"CONNECTED", "DOWN", "NO_MODELS", "ERROR"}
    assert "credentials" in payload


# --- Replay -------------------------------------------------------------------


def test_replay_capability_is_reported_honestly(client):
    payload = client.get("/api/research/replay/capability").json()
    assert payload["data_replay"]["state"] == "AVAILABLE"
    assert payload["tradingview_native_replay"]["state"] == "PARTIALLY_AVAILABLE"
    assert "anti_lookahead" in payload


def test_controlling_a_replay_without_a_session_is_refused(client):
    response = client.post("/api/research/replay/control", json={"action": "advance", "bars": 1})
    assert response.status_code == 200
    assert response.json()["error_code"] == "NO_REPLAY_SESSION"


def test_an_unknown_replay_action_is_rejected_by_the_schema(client):
    response = client.post("/api/research/replay/control", json={"action": "teleport"})
    assert response.status_code == 422


# --- Triggers and backtest ----------------------------------------------------


def test_the_trigger_registry_is_served_to_the_ui(client):
    payload = client.get("/api/trading/triggers").json()
    triggers = payload["data"]["triggers"]
    assert payload["data"]["count"] >= 14
    for trigger in triggers:
        for field in ("id", "name", "direction", "requirements", "confirmation",
                      "invalidation", "preferred_timeframes", "compatible_theories"):
            assert field in trigger


def test_a_backtest_for_an_unknown_trigger_is_refused(client):
    response = client.post("/api/trading/backtest", json={"trigger": "not_a_trigger", "count": 400})
    assert response.status_code == 200
    assert response.json()["error_code"] in {"UNKNOWN_TRIGGER", "MARKET_DATA_UNAVAILABLE"}


def test_backtest_parameters_are_range_checked(client):
    assert client.post("/api/trading/backtest", json={"count": 5}).status_code == 422
    assert client.post("/api/trading/backtest", json={"reward_multiple": 0}).status_code == 422


# --- Strategy composer --------------------------------------------------------


def test_a_composed_strategy_is_saved_and_versioned(client):
    body = {
        "name": "Composer Test", "context": "1H Demand", "setup": "Liquidity Sweep",
        "confirmation": "5m MSS", "entry_trigger": "bullish_mss",
        "invalidation": "Close below the sweep low", "stop": "Below the sweep",
        "targets": ["External liquidity"], "timeframes": ["H1", "M15", "M5"], "direction": "LONG",
    }
    first = client.post("/api/trading/strategies", json=body)
    assert first.status_code == 201
    assert first.json()["version"] == 1
    second = client.post("/api/trading/strategies", json={**body, "confirmation": "5m CHOCH"})
    assert second.status_code == 201
    # Re-saving must version rather than overwrite.
    assert second.json()["version"] == 2


def test_a_saved_strategy_is_executable_not_just_prose(client):
    client.post("/api/trading/strategies", json={
        "name": "Executable Test", "context": "1H bullish", "setup": "liquidity sweep",
        "confirmation": "5m mss", "entry_trigger": "bullish_sweep", "direction": "LONG",
    })
    listed = client.get("/api/trading/strategies").json()
    saved = next(item for item in listed["strategies"] if item["name"] == "Executable Test")
    conditions = saved["definition"]["conditions"]
    assert conditions, "a composed strategy must carry executable predicates"
    allowed = {"trend_is", "has_liquidity_sweep", "has_bos", "has_mss", "has_active_fvg", "rsi_above", "rsi_below"}
    assert all(condition["predicate"] in allowed for condition in conditions)


def test_a_strategy_naming_an_unknown_trigger_is_refused(client):
    response = client.post("/api/trading/strategies",
                           json={"name": "Bad", "entry_trigger": "does_not_exist"})
    assert response.status_code == 400


def test_saved_strategies_survive_being_listed_again(client):
    client.post("/api/trading/strategies", json={"name": "Reload Me", "context": "1H bullish"})
    names = [item["name"] for item in client.get("/api/trading/strategies").json()["strategies"]]
    assert "Reload Me" in names


# --- Gann and pitchfork endpoints ---------------------------------------------


def test_gann_and_pitchfork_endpoints_respond(client):
    for path in ("/api/trading/gann", "/api/trading/pitchfork"):
        response = client.get(path)
        assert response.status_code == 200
        payload = response.json()
        # Without a market provider in tests these report failure rather than fake data.
        assert "status" in payload


def test_drawing_endpoints_are_permission_gated(client):
    for path, body in (
        ("/api/tradingview/drawings/gann", {"symbol": "XAUUSD"}),
        ("/api/tradingview/drawings/pitchfork", {"symbol": "XAUUSD", "variant": "andrews"}),
    ):
        payload = client.post(path, json=body).json()
        assert payload["verified"] is False
        assert payload["error_code"] is not None


def test_an_unknown_pitchfork_variant_is_rejected(client):
    response = client.post("/api/tradingview/drawings/pitchfork",
                           json={"symbol": "XAUUSD", "variant": "nonsense"})
    assert response.status_code == 422


# --- Frontend wiring ----------------------------------------------------------


def test_the_frontend_never_embeds_a_credential():
    for path in (PROJECT_ROOT / "frontend").rglob("*"):
        if path.suffix.lower() not in {".js", ".html", ".css"}:
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        assert "sk-or-" not in content, path.name
        assert "OPENROUTER_API_KEY" not in content, path.name


def test_the_new_panels_exist_in_the_markup():
    markup = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    for element_id in (
        "card-providers", "card-triggers", "card-backtest", "card-composer", "card-journal",
        "bt-run-btn", "bt-cancel-btn", "bt-trigger", "sc-save-btn", "trigger-list",
        "backtest-results", "openrouter-key-input", "ollama-start-btn", "journal-list",
    ):
        assert f'id="{element_id}"' in markup, f"missing #{element_id}"


def test_the_openrouter_input_is_masked():
    markup = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    field = re.search(r'<input[^>]*id="openrouter-key-input"[^>]*>', markup)
    assert field and 'type="password"' in field.group(0)


def test_the_panels_call_the_real_backend_endpoints():
    script = (PROJECT_ROOT / "frontend" / "panels.js").read_text(encoding="utf-8")
    for endpoint in (
        "/api/providers/status", "/api/providers/credentials", "/api/providers/ollama/start",
        "/api/trading/triggers", "/api/trading/backtest", "/api/trading/strategies", "/api/trading/journal",
    ):
        assert endpoint in script, f"panels.js never calls {endpoint}"


def test_the_frontend_does_not_recompute_backtest_statistics():
    """Statistics must be rendered from the backend payload, never derived here."""
    script = (PROJECT_ROOT / "frontend" / "panels.js").read_text(encoding="utf-8")
    for invented in ("winRate =", "profitFactor =", "expectancy =", "computeExpectancy", "calculateWinRate"):
        assert invented not in script, f"panels.js appears to compute {invented} client-side"
    # It must read the backend's own fields instead.
    for field in ("win_rate", "profit_factor", "expectancy", "max_drawdown_r"):
        assert field in script


def test_the_key_input_is_cleared_after_submission():
    script = (PROJECT_ROOT / "frontend" / "panels.js").read_text(encoding="utf-8")
    assert 'input.value = ""' in script, "the key field must not retain the credential"


# --- Regressions found by driving the page in a real browser -----------------


def test_the_main_navigation_is_actually_bound():
    """Every content panel was unreachable: nothing toggled the active class.

    The stylesheet declares `.content-panel.active { display:flex }`, so without
    a handler the eleven main views could never be opened.
    """
    script = (PROJECT_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    assert "function selectView(" in script, "no view switcher exists"
    assert "[data-nav]" in script, "nothing binds the navigation buttons"
    assert 'classList.toggle("active"' in script, "the active class is never applied"


def test_every_navigation_target_has_a_panel_to_open():
    markup = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    targets = set(re.findall(r'data-nav="([a-z-]+)"', markup))
    panels = set(re.findall(r'class="content-panel[^"]*" id="panel-([a-z-]+)"', markup))
    assert targets, "no navigation targets found"
    assert targets <= panels, f"navigation points at views that do not exist: {sorted(targets - panels)}"


def test_no_handler_binds_to_an_undefined_dom_entry():
    """A throw here aborts the rest of wire(), silently unbinding later controls."""
    script = (PROJECT_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    defined = set(re.findall(r"^\s*([A-Za-z0-9_]+)\s*:", script, re.M))
    used = set(re.findall(r"dom\.([A-Za-z0-9_]+)\.addEventListener", script))
    missing = sorted(name for name in used if name not in defined)
    assert not missing, f"handlers bind to undefined dom entries: {missing}"


def test_the_page_makes_no_third_party_requests():
    """SAM is local-first, and its own CSP blocks external styles anyway."""
    markup = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    external = re.findall(r'(?:href|src)="(https?://[^"]+)"', markup)
    assert not external, f"the page loads third-party resources: {external}"


def test_the_font_stack_has_local_fallbacks():
    styles = (PROJECT_ROOT / "frontend" / "styles.css").read_text(encoding="utf-8")
    assert "Segoe UI" in styles or "system-ui" in styles or "ui-sans-serif" in styles


def test_the_manual_calibrate_endpoint_feeds_the_store_drawing_reads(tmp_path: Path):
    """End to end through the unchanged route: POST anchors, then drawing is unblocked.

    The endpoint, its method and its response keys are exactly as before; only
    where the values land has changed.
    """
    from fastapi.testclient import TestClient

    from sam_backend.app import create_app
    from sam_backend.config import Settings
    from sam_backend.trading.calibration import geometry_hash

    geometry = {"left": 0, "top": 0, "right": 1200, "bottom": 800}

    class Window:
        window_handle = 7
        symbol = "XAUUSD"
        timeframe = "M15"
        client_geometry = geometry
        window_geometry = geometry
        active = True
        title = "TradingView"
        current_price = 2500.0

        def as_dict(self):
            return {"symbol": self.symbol, "timeframe": self.timeframe}

    settings = Settings(project_root=tmp_path, workspace_root=tmp_path / "workspace",
                        data_dir=tmp_path / "data", default_provider="ollama", default_model="fake")
    app = create_app(settings)
    trading = app.state.trading
    # The engine is handed its observer at construction, so that is where a
    # stand-in window goes; patching the controller afterwards would not reach it.
    trading.drawing._observe = lambda: Window()

    with TestClient(app) as client:
        response = client.post("/api/tradingview/action", json={
            "action": "calibrate", "price_a": 3000.0, "y_a": 100.0, "price_b": 2000.0, "y_b": 600.0,
        })

    assert response.status_code == 200
    body = response.json()
    assert body["verified"] is True
    assert {"window_handle", "slope", "intercept"} <= set(body["data"]), "response shape preserved"
    assert body["data"]["slope"] == pytest.approx(-2.0)

    # The calibration the drawing engine consults is now present.
    calibration, failure = trading.drawing.active_calibration(Window())
    assert failure is None and calibration.verified
    assert calibration.price_at(350.0) == pytest.approx(2500.0)
    stored = trading.database.get_chart_calibration(
        window_handle=7, symbol="XAUUSD", timeframe="M15", geometry_hash=geometry_hash(geometry))
    assert stored["method"] == "manual_anchors"


def test_the_manual_calibrate_endpoint_still_validates_its_anchors(tmp_path: Path):
    from fastapi.testclient import TestClient

    from sam_backend.app import create_app
    from sam_backend.config import Settings

    settings = Settings(project_root=tmp_path, workspace_root=tmp_path / "workspace",
                        data_dir=tmp_path / "data", default_provider="ollama", default_model="fake")
    with TestClient(create_app(settings)) as client:
        missing = client.post("/api/tradingview/action", json={"action": "calibrate", "price_a": 3000.0})

    assert missing.status_code == 400 and "required" in missing.json()["detail"]
