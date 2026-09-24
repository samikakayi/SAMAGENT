"""The UI on the REAL packages (registered, not started: no devices, no
network, no MT5/CDP), talking to the core only through the bridge.

The other UI tests use small fakes; these check that the shapes the pages rely
on (StrategyStore rows, monitor, memory turns, voice status, secret status,
the brain's typed-text path) match what the other builders actually shipped.
"""

from __future__ import annotations

import logging

import pytest

from brain_helpers import ScriptedBackend
from sam.textnorm import is_arabic_script
from sam.ui.strings import TOOL_LABELS, learn_tool_labels, tool_label
from ui_helpers import core, pump, qapp, wait_until  # noqa: F401 - fixtures

UI_ERRORS = ("UI handler failed", "UI result callback failed", "UI command failed", "UI error callback failed")


@pytest.fixture
def full_app(make_app, core, qapp):
    backend = ScriptedBackend(default="باشە، ئامادەم. چی بکەم؟")
    app = make_app(backends={"groq": backend})
    status = app.load_packages()
    app.bus.bind_loop(core.loop)
    app.test_backend = backend
    app.test_load_status = status
    return app


@pytest.fixture
def full_controller(full_app, core, qapp):
    import sam.ui as ui

    ctrl = ui.build(full_app, core, show=False)
    yield ctrl
    ctrl.shutdown()
    if ctrl.panel is not None:
        ctrl.panel.deleteLater()
    ctrl.island.deleteLater()
    ctrl.tray.icon.deleteLater()
    pump(20)


def _ui_errors(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if any(r.getMessage().startswith(e) for e in UI_ERRORS)]


def test_every_registered_tool_has_a_sorani_label(full_app):
    assert all(v == "ok" for v in full_app.test_load_status.values()), full_app.test_load_status
    learn_tool_labels(full_app.tools)
    names = full_app.tools.names()
    assert len(names) >= 29
    missing = [n for n in names if n not in TOOL_LABELS or not is_arabic_script(tool_label(n))]
    assert missing == []


def test_learn_tool_labels_uses_the_tools_own_sorani_line(full_app):
    from sam.brain.tools import ok, tool

    @tool("ui_test_new_tool", description="A tool added after the UI table was written.",
          description_ckb="ئامرازێکی تاقیکردنەوە")
    async def _new_tool(ctx):  # pragma: no cover - never dispatched
        return ok("done")

    full_app.tools.add(_new_tool, owner="test")
    try:
        assert learn_tool_labels(full_app.tools) >= 1
        assert tool_label("ui_test_new_tool") == "ئامرازێکی تاقیکردنەوە"
        assert learn_tool_labels(full_app.tools) == 0          # idempotent
    finally:
        TOOL_LABELS.pop("ui_test_new_tool", None)
        full_app.tools.remove("ui_test_new_tool")


def test_panel_and_island_run_on_the_real_packages(full_controller, full_app, caplog):
    caplog.set_level(logging.WARNING)
    ctrl = full_controller
    panel = ctrl.ensure_panel()
    panel.show()
    pump(50)

    # Start-up snapshot from the real voice engine / TradingView bridge / MT5 feed (none started).
    assert wait_until(lambda: panel.components["voice"].dot.state != "unknown", 5.0)
    assert wait_until(lambda: panel.components["tradingview"].dot.state == "down", 5.0)
    assert panel.components["mt5"].dot.state == "down"

    # Typed text: the real Conversation answers through the scripted Groq backend.
    chat = panel.pages["chat"]
    chat.input.setPlainText("سڵاو سام")
    chat.send()
    assert wait_until(lambda: any(b.role == "assistant" and "ئامادەم" in b.text() for b in chat.bubbles), 10.0)
    assert [b.text() for b in chat.bubbles if b.role == "user"] == ["سڵاو سام"]      # shown once, no duplicate
    assert full_app.test_backend.requests, "the text brain was not called"
    assert wait_until(lambda: len(full_app.db.query("SELECT id FROM turns")) >= 2, 5.0)   # conversation persisted

    # Every page refreshes against the real objects without errors.
    for key in ("strategies", "monitor", "activity", "settings", "chat"):
        panel.show_page(key)
        pump(250)
    strategies = panel.pages["strategies"]
    assert wait_until(lambda: strategies.list_empty.isVisibleTo(strategies) or strategies.list.count() > 0, 5.0)
    monitor = panel.pages["monitor"]
    assert monitor.active_rows == [] and not monitor.cancel_all_button.isEnabled()
    settings = panel.pages["settings"]
    assert wait_until(lambda: settings.key_rows["gemini_api_key"].status.text() == "دانەنراوە", 5.0)
    assert settings.tv_button.isEnabled() and settings.mt5_button.isEnabled()

    # History comes back from the real memory after a new panel is built.
    chat2_loaded: list = []
    ctrl.bridge.on_core(full_app.memory.recent_turns, limit=10, on_ok=chat2_loaded.append)
    assert wait_until(lambda: bool(chat2_loaded), 5.0)
    assert any(t.get("text") == "سڵاو سام" for t in chat2_loaded[0])

    # The island followed the turn (thinking -> back to ready) and shows the reply.
    assert wait_until(lambda: ctrl.island.state in ("idle", "sleeping"), 5.0)
    assert _ui_errors(caplog) == []


def test_mt5_check_and_tradingview_status_use_the_real_objects(full_controller, full_app, monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    panel = full_controller.ensure_panel()
    panel.show_page("settings")
    panel.show()
    settings = panel.pages["settings"]

    async def fake_status():            # never initialise the real MT5 terminal from a test
        return {"connected": True, "broker_offset_s": 10800}

    monkeypatch.setattr(full_app.trading.mt5, "status", fake_status)
    # The start-up snapshot (MT5 not started -> "down") lands first; the click result is newer.
    assert wait_until(lambda: panel.components["mt5"].dot.state == "down", 5.0)
    pump(50)
    settings.check_mt5()
    ok_ = wait_until(lambda: "+٣" in settings.mt5_status.text(), 5.0)
    assert ok_, settings.mt5_status.text()
    assert settings.mt5_dot.state == "ok"
    assert _ui_errors(caplog) == []
