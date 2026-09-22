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


# --- The TradingView panel's drawing row -------------------------------------
#
# Found live: the API reported calibrated=true and verified_price_drawing=true
# for a real XAUUSD chart while the panel still read "Unavailable until verified
# chart calibration". The panel gated on `state.chart_geometry`, a field of
# TradingViewState that observe() never assigns, so it was always null.

DRAWING_ROW = "tv-drawing-state"
UNAVAILABLE = "Unavailable until verified chart calibration"
PANEL_GEOMETRY = {"left": 0, "top": 0, "right": 1200, "bottom": 800}


def observed_state(**overrides):
    """A real TradingViewState, so the payload keys are the production ones."""
    from sam_backend.trading.tradingview import TradingViewState

    fields = dict(
        running=True, process_ids=[4242], window_handle=919736, title="XAUUSD ▼ 4,310.55",
        symbol="XAUUSD", feed="OANDA", timeframe="M15", timeframe_verified=True,
        current_price=4310.55, window_geometry=PANEL_GEOMETRY, monitor={"name": r"\.\DISPLAY1"},
        active=True, interactive=True, client_geometry=PANEL_GEOMETRY,
    )
    fields.update(overrides)
    return TradingViewState(**fields)


def panel_payload(tmp_path, *, calibrated=True, permitted=True, viewport_moved=False, window=True):
    """The `/api/trading/status` payload for one real chart situation.

    Both halves come from production code: the observation from TradingViewState,
    the drawing row's answer from DrawingEngine.capability() over a real database.
    """
    from sam_backend.db import Database
    from sam_backend.trading.calibration import ChartCalibrator, geometry_hash
    from sam_backend.trading.drawing import DrawingEngine

    state = observed_state(**({} if window else {"window_handle": None, "active": False}))
    database = Database(tmp_path / "panel.sqlite3")
    engine = DrawingEngine(
        database=database, calibrator=ChartCalibrator(), observe=lambda: state,
        focus=lambda: None, computer_control=permitted, screen_access=permitted,
    )
    if calibrated:
        # Calibrated for the viewport the chart had; optionally it has since moved.
        stored = dict(PANEL_GEOMETRY, right=1400) if viewport_moved else PANEL_GEOMETRY
        database.save_chart_calibration(
            window_handle=919736, symbol="XAUUSD", timeframe="M15",
            geometry_hash=geometry_hash(stored), slope=-0.2556, intercept=4485.04,
            method="price_axis_ocr", verified=True, axis_x=1100.0,
        )
    return {"tradingview": state.as_dict(), "drawing": engine.capability()}


def render_panel(tmp_path, payload):
    """Run the real panels.js against the payload and report what it displayed."""
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    payload_file = tmp_path / "payload.json"
    payload_file.write_text(json.dumps(payload), encoding="utf-8")
    finished = subprocess.run(
        [node, str(Path(__file__).parent / "render_tradingview_panel.js"),
         str(PROJECT_ROOT / "frontend" / "panels.js"), str(payload_file)],
        check=True, capture_output=True, text=True, encoding="utf-8",
    )
    return json.loads(finished.stdout)


def test_a_calibrated_chart_is_not_reported_as_uncalibrated(tmp_path):
    """The live defect: the backend said yes and the panel said no."""
    payload = panel_payload(tmp_path)
    assert payload["drawing"]["calibrated"] is True
    assert payload["drawing"]["verified_price_drawing"] is True

    shown = render_panel(tmp_path, payload)["rendered"][DRAWING_ROW]

    assert shown != UNAVAILABLE, "the panel contradicted a calibrated backend"
    assert "available" in shown.lower()


def test_an_uncalibrated_chart_still_reads_unavailable(tmp_path):
    payload = panel_payload(tmp_path, calibrated=False)
    assert payload["drawing"]["calibrated"] is False

    assert render_panel(tmp_path, payload)["rendered"][DRAWING_ROW] == UNAVAILABLE


def test_calibration_without_permission_never_claims_verified_drawing(tmp_path):
    """Calibrated, but screen access and computer control are off."""
    payload = panel_payload(tmp_path, permitted=False)
    assert payload["drawing"]["calibrated"] is True
    assert payload["drawing"]["verified_price_drawing"] is False

    shown = render_panel(tmp_path, payload)["rendered"][DRAWING_ROW]

    assert not re.search(r"(?<!un)available", shown, re.I), "claimed drawing it cannot perform"
    assert "permission" in shown.lower() or "required" in shown.lower()


def test_a_moved_viewport_invalidates_the_panel_row(tmp_path):
    """The stored calibration belongs to a chart geometry that no longer holds."""
    payload = panel_payload(tmp_path, viewport_moved=True)
    assert payload["drawing"]["calibrated"] is False

    assert render_panel(tmp_path, payload)["rendered"][DRAWING_ROW] == UNAVAILABLE


def test_no_chart_window_reads_unavailable(tmp_path):
    payload = panel_payload(tmp_path, window=False)
    assert payload["drawing"]["calibrated"] is False

    assert render_panel(tmp_path, payload)["rendered"][DRAWING_ROW] == UNAVAILABLE


def test_the_panel_asks_the_backend_instead_of_guessing_from_geometry(tmp_path):
    """The drawing row must consume the engine's own answer, not re-derive one."""
    requested = render_panel(tmp_path, panel_payload(tmp_path))["requested"]
    assert any("status" in path for path in requested), requested

    script = (PROJECT_ROOT / "frontend" / "panels.js").read_text(encoding="utf-8")
    assert "chart_geometry" not in script, "the panel still gates on a field that is never sent"


def test_the_observed_state_declares_no_chart_geometry_of_its_own():
    """It was never assigned, and it shadowed DrawingEngine.chart_geometry().

    A reader who found `state.chart_geometry` reasonably assumed it held the
    engine's answer; it held null, and the panel believed it.
    """
    import dataclasses

    from sam_backend.trading.tradingview import TradingViewState

    names = {field.name for field in dataclasses.fields(TradingViewState)}
    assert "chart_geometry" not in names
    assert "client_geometry" in names, "the geometry observe() does populate stays"


# --- The card must not depend on the broker feed ------------------------------
#
# The drawing row reads /api/trading/status, which also reaches MetaTrader5.
# A chart verdict must not be decided, delayed, or erased by an unrelated broker.

UNREACHABLE = "Unavailable; the drawing engine could not be reached"


def calibrated_service(tmp_path):
    """A real TradingService over a calibrated fake window, MT5 untouched."""
    from sam_backend.cancellation import CancellationManager
    from sam_backend.config import Settings
    from sam_backend.db import Database
    from sam_backend.trading.calibration import geometry_hash
    from sam_backend.trading.service import TradingService

    settings = Settings(project_root=tmp_path, workspace_root=tmp_path / "workspace",
                        data_dir=tmp_path / "data")
    settings.prepare()
    database = Database(settings.database_path)
    trading = TradingService(settings, database, CancellationManager())

    state = observed_state()
    trading.tradingview.observe = lambda: state
    trading.drawing._observe = lambda: state
    trading.drawing.computer_control = trading.drawing.screen_access = True
    database.save_chart_calibration(
        window_handle=state.window_handle, symbol="XAUUSD", timeframe="M15",
        geometry_hash=geometry_hash(PANEL_GEOMETRY), slope=-0.2556, intercept=4485.04,
        method="price_axis_ocr", verified=True, axis_x=1100.0,
    )
    return trading


def test_a_broken_metatrader_does_not_erase_the_chart_verdict(tmp_path):
    """MT5 failing is reported as data; the drawing engine still answers."""
    from sam_backend.trading.market_data import MarketDataError

    trading = calibrated_service(tmp_path)
    provider = trading.market_data.providers["metatrader5"]

    def dead(*args, **kwargs):
        # What the provider raises when the terminal is absent or unreachable.
        raise MarketDataError("MetaTrader 5 terminal is not running")

    provider.fetch = dead
    provider._module = dead

    payload = trading.status()

    assert payload["market_data"]["metatrader5"]["state"] == "UNAVAILABLE"
    assert payload["capabilities"]["state"] == "UNAVAILABLE"
    assert "not running" in payload["capabilities"]["error"]
    # The part the panel needs survived untouched.
    assert payload["drawing"]["calibrated"] is True
    assert payload["drawing"]["verified_price_drawing"] is True

    shown = render_panel(tmp_path, json.loads(json.dumps(payload, default=str)))["rendered"]
    assert shown[DRAWING_ROW] == "Verified price drawing available"
    assert shown["tv-observed-symbol"] == "XAUUSD"


def test_a_failing_status_endpoint_does_not_claim_the_chart_is_uncalibrated(tmp_path):
    """Not being able to ask is not the same answer as "not calibrated"."""
    scenario = {"responses": {
        "/api/tradingview/state": {"body": observed_state().as_dict()},
        "/api/trading/status": {"status": 500, "body": {"detail": "market data blew up"}},
    }}

    shown = render_panel(tmp_path, scenario)["rendered"]

    assert shown[DRAWING_ROW] != UNAVAILABLE, "claimed a calibration verdict it never received"
    assert "available" not in shown[DRAWING_ROW].lower() or "Unavailable" in shown[DRAWING_ROW]
    # The window observation needs no broker, so it must still be shown.
    assert shown["tv-observed-symbol"] == "XAUUSD"
    assert shown["tradingview-status"] == "RUNNING"


def test_a_status_payload_without_a_drawing_verdict_invents_none(tmp_path):
    scenario = {"responses": {
        "/api/tradingview/state": {"body": observed_state().as_dict()},
        "/api/trading/status": {"body": {"tradingview": observed_state().as_dict()}},
    }}

    shown = render_panel(tmp_path, scenario)["rendered"]

    assert shown[DRAWING_ROW] == UNREACHABLE
    assert shown["tv-observed-symbol"] == "XAUUSD"


def test_a_stalled_broker_does_not_freeze_the_rest_of_the_card(tmp_path):
    """A hung MetaTrader5 call holds /api/trading/status open indefinitely."""
    scenario = {"responses": {
        "/api/tradingview/state": {"body": observed_state().as_dict()},
        "/api/trading/status": {"hang": True},
    }}

    shown = render_panel(tmp_path, scenario)["rendered"]

    assert shown["tv-observed-symbol"] == "XAUUSD", "the observation waited on the broker"
    assert shown["tv-observed-price"] == "4310.55"
    assert shown["tradingview-status"] == "RUNNING"


def test_losing_the_chart_observation_retires_the_previous_verdict(tmp_path):
    """A verdict from an earlier tick describes a chart no longer being observed."""
    scenario = {"responses": {
        "/api/tradingview/state": {"status": 500, "body": {"detail": "window gone"}},
        "/api/trading/status": {"body": {"drawing": {"calibrated": True, "verified_price_drawing": True}}},
    }}

    shown = render_panel(tmp_path, scenario)["rendered"]

    assert shown["tradingview-status"] == "UNAVAILABLE"
    assert shown[DRAWING_ROW] == UNREACHABLE, "left a stale availability claim standing"


# --- Broker-status polling ------------------------------------------------------
#
# The page refreshes the card every ten seconds. /api/trading/status reaches the
# broker; if it hangs, each tick must not add another unanswered request on top.

STATUS, STATE = "/api/trading/status", "/api/tradingview/state"
AVAILABLE = "Verified price drawing available"


def observation(price):
    return {"body": observed_state(current_price=price).as_dict()}


def verdict(calibrated):
    return {"body": {"drawing": {"calibrated": calibrated, "verified_price_drawing": calibrated}}}


def poll_panel(tmp_path, polls):
    return render_panel(tmp_path, {"polls": polls})


def test_a_hung_broker_never_has_more_than_one_status_request_in_flight(tmp_path):
    """Three ticks against a broker that never answers."""
    hung = {STATE: observation(1.0), STATUS: {"hang": True}}

    result = poll_panel(tmp_path, [{"responses": hung}] * 3)

    fired = [poll["fired"]["requested"].count(STATUS) for poll in result["polls"]]
    assert sum(fired) >= 1, "the broker was never asked at all"
    assert result["maxInFlight"][STATUS] == 1, f"status requests piled up: {result['maxInFlight']}"


def test_observation_keeps_refreshing_while_the_broker_hangs(tmp_path):
    polls = [{"responses": {STATE: observation(price), STATUS: {"hang": True}}} for price in (1.0, 2.0, 3.0)]

    result = poll_panel(tmp_path, polls)

    for poll, price in zip(result["polls"], ("1.00", "2.00", "3.00")):
        assert STATE in poll["fired"]["requested"], "the observation stopped being asked for"
        assert poll["afterInterval"]["rendered"]["tv-observed-price"] == price
    # A stall is reported as not reached, not as a stale verdict or a calibration claim.
    assert result["rendered"][DRAWING_ROW] == UNREACHABLE
    assert result["errors"] == [], f"console noise: {result['errors']}"


def test_a_slow_old_answer_cannot_overwrite_a_newer_one(tmp_path):
    """Request A is slow and says "available"; B, asked later, says "not calibrated"."""
    polls = [
        {"responses": {STATE: observation(1.0), STATUS: dict(verdict(True), delayMs=15_000)}},
        {"responses": {STATE: observation(2.0), STATUS: verdict(False)}},
    ]

    result = poll_panel(tmp_path, polls)

    assert result["rendered"][DRAWING_ROW] != AVAILABLE, "a stale broker answer overwrote the newer one"
    assert result["polls"][1]["afterInterval"]["rendered"]["tv-observed-price"] == "2.00"


def test_the_broker_recovers_without_a_reload(tmp_path):
    """Two hung ticks, then the broker answers; no lock may outlive the stall."""
    hung = {STATE: observation(1.0), STATUS: {"hang": True}}
    polls = [{"responses": hung}, {"responses": hung},
             {"responses": {STATE: observation(3.0), STATUS: verdict(True)}}]

    result = poll_panel(tmp_path, polls)

    assert result["rendered"][DRAWING_ROW] == AVAILABLE, result["rendered"][DRAWING_ROW]
    assert result["inFlight"].get(STATUS, 0) == 0, "an unanswered request was left in flight"
    assert result["errors"] == []


def test_a_controlled_broker_error_releases_the_next_poll(tmp_path):
    polls = [{"responses": {STATE: observation(1.0), STATUS: {"status": 500, "body": {"detail": "broker down"}}}},
             {"responses": {STATE: observation(2.0), STATUS: verdict(True)}}]

    result = poll_panel(tmp_path, polls)

    assert result["polls"][0]["afterInterval"]["rendered"][DRAWING_ROW] == UNREACHABLE
    assert result["rendered"][DRAWING_ROW] == AVAILABLE
    assert result["polls"][1]["fired"]["requested"].count(STATUS) == 1


def test_a_refresh_inside_the_timeout_window_reuses_the_pending_request(tmp_path):
    """tradingViewAction refreshes immediately; it must not open a second broker request."""
    hung = {STATE: observation(1.0), STATUS: {"hang": True}}
    polls = [{"responses": hung, "advanceMs": 3_000}, {"responses": hung, "advanceMs": 3_000},
             {"responses": hung}]

    result = poll_panel(tmp_path, polls)

    assert result["maxInFlight"][STATUS] == 1, result["maxInFlight"]
    assert result["requested"].count(STATUS) == 1, "each early refresh opened its own request"
    assert result["requested"].count(STATE) == 3, "the observation itself must still refresh"


# --- One owner for the broker-status request ------------------------------------
# Page load runs refreshTradingView and loadToolStatus together. Both need a
# slice of /api/trading/status; they must share one request, not race two.

PAGE_LOAD = ["tradingView", "toolStatus"]
TOOL_PAGES = {
    "/api/tools/manifests": {"body": {"tools": [{"name": "read_file"}, {"name": "write_file"}, {"name": "analyze_market"}]}},
    "/api/health": {"body": {"computer_control": True}},
    "/api/desktop/status": {"body": {"tradingview": {"interactive": True, "running": True}}},
}


def full_status(mt5_state="AVAILABLE", calibrated=True):
    return {"body": {"market_data": {"metatrader5": {"state": mt5_state}},
                     "drawing": {"calibrated": calibrated, "verified_price_drawing": calibrated}}}


def test_page_load_asks_the_broker_once_for_both_the_badge_and_the_verdict(tmp_path):
    responses = {**TOOL_PAGES, STATE: observation(1.0), STATUS: full_status()}

    result = poll_panel(tmp_path, [{"responses": responses, "fire": PAGE_LOAD}])

    assert result["requested"].count(STATUS) == 1, result["requested"]
    assert result["maxInFlight"][STATUS] == 1
    # Both consumers were fed from that one answer.
    assert result["rendered"]["tool-status-mt5"] == "● Connected read-only"
    assert result["rendered"][DRAWING_ROW] == AVAILABLE
    # The tool badges that never needed the broker rendered from their own sources.
    assert result["rendered"]["tool-status-files"] == "● Available"


def test_a_hung_broker_at_page_load_still_yields_one_request_and_honest_rows(tmp_path):
    hung = {**TOOL_PAGES, STATE: observation(1.0), STATUS: {"hang": True}}
    polls = [{"responses": hung, "fire": PAGE_LOAD}, {"responses": hung}, {"responses": hung}]

    result = poll_panel(tmp_path, polls)

    assert result["maxInFlight"][STATUS] == 1, result["maxInFlight"]
    assert result["requested"].count(STATUS) == 3, "one request per tick, including the load"
    assert result["rendered"][DRAWING_ROW] == UNREACHABLE
    assert result["rendered"]["tool-status-mt5"] == "Unknown", "a badge claimed a broker state it never received"
    assert result["rendered"]["tool-status-files"] == "● Available", "an unreachable broker blanked unrelated badges"
    assert result["errors"] == []


def test_a_broker_that_recovers_after_load_updates_the_polled_row(tmp_path):
    """Only the card is on the interval; the tool badges are drawn once, at load."""
    hung = {**TOOL_PAGES, STATE: observation(1.0), STATUS: {"hang": True}}
    polls = [{"responses": hung, "fire": PAGE_LOAD},
             {"responses": {**TOOL_PAGES, STATE: observation(2.0), STATUS: full_status()}}]

    result = poll_panel(tmp_path, polls)

    assert result["requested"].count(STATUS) == 2
    assert result["rendered"][DRAWING_ROW] == AVAILABLE
    assert result["inFlight"].get(STATUS, 0) == 0
    # The load-time badge is not re-polled, so it keeps the honest answer it got.
    assert result["rendered"]["tool-status-mt5"] == "Unknown"
