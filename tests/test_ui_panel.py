"""Panel pages, settings key safety, chat routing, tray, bridge."""

from __future__ import annotations

import logging
import time

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLineEdit, QScrollArea

from conftest import FAKE_GEMINI_AQ, FAKE_GROQ
from sam.events import Alert, Caption, ComponentStatus, ToolFinished, ToolStarted, Transcript, VoiceState
from sam.ui.panel import PAGES
from sam.ui.strings import STRINGS, tr
from ui_helpers import (FakeConversation, FakeMonitor, FakeStrategies, FakeVoice, all_texts, controller,  # noqa: F401
                        core, pump, qapp, ui_app, wait_until)


@pytest.fixture
def panel(controller):
    p = controller.ensure_panel()
    p.show()
    pump(30)
    return p


def test_panel_builds_all_five_sorani_tabs(panel):
    assert list(panel.pages) == list(PAGES)
    assert [panel.nav[k].text() for k in PAGES] == ["گفتوگۆ", "ستراتیژییەکان", "چاودێری", "چالاکی", "ڕێکخستنەکان"]
    assert panel.layoutDirection() == Qt.LayoutDirection.RightToLeft
    for key in PAGES:
        panel.show_page(key)
        pump(20)
        assert panel.stack.currentWidget() is panel.pages[key]
        assert panel.nav[key].isChecked()


def test_close_hides_instead_of_quitting(panel):
    panel.close()
    pump(20)
    assert not panel.isVisible()


def test_sidebar_follows_voice_state_and_component_status(panel, controller):
    controller.bridge.deliver(VoiceState(state="listening"))
    controller.bridge.deliver(ComponentStatus(component="tradingview", state="degraded", detail="no port"))
    assert panel.brand.state == "listening"
    row = panel.components["tradingview"]
    assert row.dot.state == "degraded" and row.state_label.text() == tr("status.degraded")


# -- settings: keys ---------------------------------------------------------------------------------------
def test_key_save_calls_secrets_set_and_never_shows_the_value(panel, ui_app, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    calls: list[tuple[str, str]] = []

    def fake_set(name: str, value: str) -> dict:
        calls.append((name, value))
        return {"name": name, "stored": True, "fingerprint": "ab12cd34ef56"}

    monkeypatch.setattr(ui_app.secrets, "set", fake_set)
    real_status = ui_app.secrets.status
    monkeypatch.setattr(ui_app.secrets, "status", lambda: {
        **real_status(), "groq_api_key": {"configured": True, "source": "secret_store", "fingerprint": "ab12cd34ef56"}})
    panel.show_page("settings")
    page = panel.pages["settings"]
    row = page.key_rows["groq_api_key"]
    assert row.edit.echoMode() == QLineEdit.EchoMode.Password
    row.edit.setText(FAKE_GROQ)
    row.save_button.click()
    assert row.edit.text() == ""                        # cleared before the save even runs
    assert wait_until(lambda: bool(calls), 5.0)
    assert calls == [("groq_api_key", FAKE_GROQ)]
    assert wait_until(lambda: row.result.text() == tr("set.saved"), 5.0)
    assert wait_until(lambda: "#ab12cd" in row.status.text(), 5.0)
    row.edit.undo()                                     # Ctrl+Z must not bring the key back
    assert row.edit.text() == ""
    texts = all_texts(panel) + all_texts(controller_island(panel))
    assert not any(FAKE_GROQ in t or FAKE_GROQ[:12] in t for t in texts)
    assert FAKE_GROQ not in caplog.text


def controller_island(panel):
    from PySide6.QtWidgets import QApplication

    from sam.ui.island import Island
    return next(w for w in QApplication.topLevelWidgets() if isinstance(w, Island))


def test_bad_key_shows_sorani_error_without_the_value(panel, ui_app):
    page = panel.pages["settings"]
    row = page.key_rows["gemini_api_key"]
    bad = "AQ.short"
    row.edit.setText(bad)
    row.save()
    assert wait_until(lambda: row.result.text() == tr("set.bad_key"), 5.0)
    assert not any(bad in t for t in all_texts(panel))
    assert not ui_app.secrets.has("gemini_api_key")


def test_real_store_round_trip_keeps_value_out_of_the_ui(panel, ui_app):
    page = panel.pages["settings"]
    row = page.key_rows["gemini_api_key"]
    row.edit.setText(FAKE_GEMINI_AQ)
    row.save()
    assert wait_until(lambda: ui_app.secrets.has("gemini_api_key"), 20.0)
    assert wait_until(lambda: row.status.text().startswith(tr("set.configured")), 10.0)
    assert not any(FAKE_GEMINI_AQ in t for t in all_texts(panel))


def test_test_button_reports_provider_result(panel, ui_app, monkeypatch):
    async def fake_test(provider: str) -> dict:
        return {"ok": True, "provider": provider, "status": "connected", "latency_ms": 320.0}

    monkeypatch.setattr(ui_app.llm, "test_provider", fake_test)
    row = panel.pages["settings"].key_rows["groq_api_key"]
    row.test()
    assert wait_until(lambda: row.result.text().startswith(tr("set.test_ok")), 5.0)
    assert "٣٢٠" in row.result.text() and row.dot.state == "ok"


def test_test_button_maps_auth_failure(panel, ui_app, monkeypatch):
    async def fake_test(provider: str) -> dict:
        return {"ok": False, "provider": provider, "status": "auth_failed"}

    monkeypatch.setattr(ui_app.llm, "test_provider", fake_test)
    panel.pages["settings"].test_omniroute()
    page = panel.pages["settings"]
    assert wait_until(lambda: page.omni_status.text() == tr("set.test.auth_failed"), 5.0)


def test_kurdishtts_test_without_voice_reports_presence_only(panel):
    row = panel.pages["settings"].key_rows["kurdishtts_stt_api_key"]
    row.test()
    assert row.result.text() == tr("set.test.presence")


def test_gemini_row_links_to_ai_studio(panel):
    from sam.ui.pages.settings import GEMINI_KEY_URL

    assert GEMINI_KEY_URL == "https://aistudio.google.com/apikey"
    row = panel.pages["settings"].key_rows["gemini_api_key"]
    assert any(GEMINI_KEY_URL == t for t in all_texts(row))


def test_voice_settings_are_saved(panel, ui_app):
    page = panel.pages["settings"]
    page.engine_buttons["cascade"].click()
    assert wait_until(lambda: ui_app.config.get("voice.engine") == "cascade", 5.0)
    page.always.click()
    assert wait_until(lambda: ui_app.config.get("voice.always_listening") is True, 5.0)
    page.hotkey.setText("CTRL+SHIFT+K")
    page.hotkey.editingFinished.emit()
    assert wait_until(lambda: ui_app.config.get("voice.hotkey") == "ctrl+shift+k", 5.0)
    page.hotkey.setText("banana")
    page.hotkey.editingFinished.emit()
    assert page.hotkey_hint.text() == tr("set.voice.hotkey_bad")
    idx = page.voice_name.findData("Puck")
    page.voice_name.setCurrentIndex(idx)
    assert wait_until(lambda: ui_app.config.get("voice.voice_name") == "Puck", 5.0)


def test_selftest_button_runs_voice_selftest(panel, ui_app):
    ui_app.voice = FakeVoice()
    page = panel.pages["settings"]
    page.selftest_button.setEnabled(True)
    page.run_selftest()
    # Plain Sorani for a non-expert; the numbers are in the tooltip.
    assert wait_until(lambda: page.selftest_result.text() in (tr("set.voice.selftest_good"),
                                                              tr("set.voice.selftest_weak")), 5.0)
    assert "CER 8%" in page.selftest_result.toolTip()
    assert ("selftest", None) in ui_app.voice.calls


def test_trading_buttons_disabled_when_packages_missing(panel):
    panel.show_page("settings")
    page = panel.pages["settings"]
    page.on_shown()
    assert not page.tv_button.isEnabled() and not page.mt5_button.isEnabled()
    assert page.tv_status.text() == tr("status.unavailable")


def test_tradingview_connect_passes_confirm_and_shows_state(panel, ui_app):
    seen: dict = {}

    class FakeTV:
        async def ensure_running(self, *, allow_restart=False, confirm=None):
            seen.update(allow_restart=allow_restart, confirm=confirm)
            return {"ok": True, "state": "started", "detail": ""}

    ui_app.trading.tv = FakeTV()
    page = panel.pages["settings"]
    page.tv_button.setEnabled(True)
    page.connect_tradingview()
    assert wait_until(lambda: page.tv_status.text() == tr("tv.state.started"), 5.0)
    assert seen["allow_restart"] is True and seen["confirm"] == ui_app.confirm.confirm


# -- chat --------------------------------------------------------------------------------------------------------
def test_typed_text_goes_to_the_conversation_and_renders_bubbles(panel, ui_app):
    ui_app.conversation = FakeConversation(ui_app)
    chat = panel.pages["chat"]
    chat.input.setPlainText("ترەیدینگ ڤیو بکەرەوە")
    chat.send()
    assert chat.input.toPlainText() == ""
    assert wait_until(lambda: ui_app.conversation.texts == ["ترەیدینگ ڤیو بکەرەوە"], 5.0)
    assert wait_until(lambda: any(b.role == "assistant" and b.text() == "باشە، کرایەوە." for b in chat.bubbles), 5.0)
    pump(100)
    users = [b for b in chat.bubbles if b.role == "user"]
    assistants = [b for b in chat.bubbles if b.role == "assistant"]
    assert [b.text() for b in users] == ["ترەیدینگ ڤیو بکەرەوە"]      # no duplicate from Caption/Transcript
    assert [b.text() for b in assistants] == ["باشە، کرایەوە."]
    assert not assistants[0].streaming


def test_live_session_without_brain_sends_to_voice(panel, ui_app):
    ui_app.voice = FakeVoice(engine_name="live", state="listening")
    chat = panel.pages["chat"]
    chat.send("سڵاو")
    assert wait_until(lambda: ("send_text", "سڵاو") in ui_app.voice.calls, 5.0)


def test_no_brain_no_live_shows_a_sorani_note(panel):
    chat = panel.pages["chat"]
    chat.send("سڵاو")
    from PySide6.QtWidgets import QLabel
    assert any(lab.text() == tr("chat.unavailable") for lab in chat.list_widget.findChildren(QLabel))


def test_voice_captions_stream_into_one_bubble(panel, controller):
    chat = panel.pages["chat"]
    deliver = controller.bridge.deliver
    deliver(Caption(text="گۆڵد لەسەر", role="user", final=False))
    deliver(Caption(text="گۆڵد لەسەر ١٥ خولەک", role="user", final=False))
    deliver(Transcript(role="user", text="گۆڵد لەسەر ١٥ خولەک پیشان بدە", source="live"))
    deliver(ToolStarted(call_id="t1", name="tv_set_chart", args={}, source="live"))
    deliver(ToolFinished(call_id="t1", name="tv_set_chart", ok=True, summary="ok", duration_ms=310, source="live"))
    deliver(Caption(text="باشە", role="assistant", final=False))
    deliver(Caption(text="باشە، گۆڕدرا.", role="assistant", final=True))
    deliver(Transcript(role="assistant", text="باشە، چارتەکە گۆڕدرا.", source="live"))
    assert [(b.role, b.text()) for b in chat.bubbles] == [
        ("user", "گۆڵد لەسەر ١٥ خولەک پیشان بدە"), ("assistant", "باشە، چارتەکە گۆڕدرا.")]
    assert all(not b.streaming for b in chat.bubbles)
    assert "t1" not in chat._chips


def test_chat_loads_history_from_memory(panel, ui_app):
    class FakeMemory:
        def recent_turns(self, conversation_id=None, limit=12):
            return [{"role": "user", "text": "سڵاو", "source": "text", "at": time.time() - 60},
                    {"role": "assistant", "text": "سڵاو، فەرموو.", "source": "text", "at": time.time() - 59},
                    {"role": "tool", "text": "{}", "source": "text", "at": time.time() - 58}]

    ui_app.memory = FakeMemory()
    chat = panel.pages["chat"]
    chat._history_loaded = False
    chat.on_shown()
    assert wait_until(lambda: len(chat.bubbles) == 2, 5.0)
    assert [b.text() for b in chat.bubbles] == ["سڵاو", "سڵاو، فەرموو."]


# -- strategies / monitor / activity ------------------------------------------------------------------------------
def test_strategies_list_detail_and_activate(panel, ui_app):
    store = FakeStrategies()
    ui_app.trading.strategies = store
    panel.show_page("strategies")
    page = panel.pages["strategies"]
    assert wait_until(lambda: page.list.count() == 2, 5.0)
    page.list.setCurrentRow(1)                      # the draft card (active cards sort first)
    assert page.selected == "ob"
    page._set_status("active")
    assert wait_until(lambda: store.status_calls == [("ob", "active")], 5.0)


def test_selecting_a_strategy_loads_the_whole_card(panel, ui_app):
    from PySide6.QtWidgets import QLabel

    store = FakeStrategies()
    ui_app.trading.strategies = store
    panel.show_page("strategies")
    page = panel.pages["strategies"]
    assert wait_until(lambda: page.list.count() == 2, 5.0)
    page.list.setCurrentRow(0)                      # the active card, listed with a rules *count* only

    def texts() -> list[str]:
        return [lab.text() for lab in page.detail_inner.findChildren(QLabel)]

    assert wait_until(lambda: "ئاراستەی H4 سەرەوە" in texts(), 5.0)   # rule text from get()
    assert store.get_calls[-1] == "asia-fvg"
    assert any("M15" in t for t in texts()) and "Asia sweep then FVG" in texts()


def test_strategy_save_failure_is_reported_in_sorani(panel, ui_app, monkeypatch):
    async def fake_dispatch(name, args, *, source="live", call_id=None):
        return {"ok": False, "summary": "Could not read the strategy (model unavailable: exhausted).", "data": {}}

    monkeypatch.setattr(ui_app.tools, "dispatch", fake_dispatch)
    page = panel.pages["strategies"]
    page._toggle_add()
    page.paste.setPlainText("buy the dip")
    page._save_new()
    assert wait_until(lambda: page.add_status.text() == tr("strat.save_failed"), 5.0)
    assert page.paste.toPlainText() == "buy the dip"        # nothing lost on failure
    assert "exhausted" in page.add_status.toolTip()


def test_strategies_page_without_store_says_so(panel):
    panel.show_page("strategies")
    page = panel.pages["strategies"]
    assert page.list_empty.label.text() == tr("strat.unavailable")


def test_strategy_paste_goes_through_strategy_save_tool(panel, ui_app, monkeypatch):
    seen: list = []

    async def fake_dispatch(name, args, *, source="live", call_id=None):
        seen.append((name, args, source))
        return {"ok": True, "summary": "saved", "data": {"id": "new-one", "readback_ckb": "ستراتیژییەکە تۆمار کرا."}}

    monkeypatch.setattr(ui_app.tools, "dispatch", fake_dispatch)
    page = panel.pages["strategies"]
    page._toggle_add()
    page.paste.setPlainText("London sweep then FVG entry")
    page._save_new()
    assert wait_until(lambda: page.add_status.text() == "ستراتیژییەکە تۆمار کرا.", 5.0)
    assert seen == [("strategy_save", {"text": "London sweep then FVG entry"}, "ui")]


def test_monitor_lists_alerts_and_cancels(panel, ui_app):
    now = time.time()
    ui_app.db.insert("alerts", {"kind": "price_cross", "symbol": "XAUUSD", "params": {"level": 2700, "direction": "up"},
                                "status": "active", "created_at": now})
    ui_app.db.insert("alerts", {"kind": "volume_spike", "symbol": "XAUUSD", "params": {"k": 2, "n": 20},
                                "status": "fired", "created_at": now - 100, "fired_at": now - 50,
                                "last_text_ckb": "قەبارە بەرز بووەوە"})
    monitor = FakeMonitor()
    ui_app.trading.monitor = monitor
    panel.show_page("monitor")
    page = panel.pages["monitor"]
    assert wait_until(lambda: len(page.active_rows) == 1 and len(page.history_rows) == 1, 5.0)
    assert "٢" in page.active_title.text() or "١" in page.active_title.text()
    page.cancel(page.active_rows[0]["id"])
    assert wait_until(lambda: monitor.cancelled == [page.active_rows[0]["id"]], 5.0)


def test_alert_condition_text_is_sorani():
    from sam.ui.pages.monitor import condition_text

    assert condition_text({"kind": "price_cross", "params": '{"level": 2700.5, "direction": "up"}',
                           "timeframe": "M15"}) == "2,700.5 · سەرەوە · M15"
    assert "هێندەی تێکڕای" in condition_text({"kind": "volume_spike", "params": {"k": 2.5, "n": 20}})


def test_activity_page_shows_actions_and_stage_timings(panel, ui_app):
    ui_app.db.log_activity("tool", "open_app", ok=True, summary="Opened TradingView", duration_ms=820, source="live")
    ui_app.timing.record("stt", 420.0, kind="cascade", turn_id="t-1")
    ui_app.timing.record("first_audio", 1380.0, kind="cascade", turn_id="t-1")
    panel.show_page("activity")
    page = panel.pages["activity"]
    assert wait_until(lambda: page.table.rowCount() >= 1 and len(page.timings.rows) >= 2, 5.0)
    assert page.table.item(0, 1).text() == "کردنەوەی بەرنامە"
    assert "یەکەم دەنگ" in page.last_turn.text()          # first_audio, labelled in Sorani
    assert page.table.item(0, 4).text() == "‪Opened TradingView‬"   # LTR summary kept in order


def test_text_helpers_for_the_rtl_panel():
    from sam.ui.strings import stage_label
    from sam.ui.widgets import bidi_text, when_text

    assert bidi_text("The user did not approve the action.") == "‪The user did not approve the action.‬"
    assert bidi_text("ترەیدینگ ڤیو کرایەوە.") == "ترەیدینگ ڤیو کرایەوە."
    assert bidi_text("") == ""
    assert stage_label("first_audio") == "یەکەم دەنگ"
    assert stage_label("tool:draw_on_chart") == "کێشان لەسەر چارت"
    assert stage_label("some_new_stage") == "some_new_stage"
    now = time.mktime((2026, 9, 24, 15, 0, 0, 0, 0, -1))
    assert when_text(now - 3600, now=now) == "١٤:٠٠"                     # today: time only
    assert when_text(now - 86400 * 3, now=now) == "٢١/٩ ١٥:٠٠"           # "/" keeps day/month in order
    assert when_text(None) == ""


def test_startup_snapshot_fills_status_dots_and_real_events_win(ui_app, core, qapp):
    """States published during app.start() predate the UI: the snapshot reads them back."""
    import sam.ui as ui

    class TV:
        def status(self) -> dict:
            return {"connected": True, "port": 9222}

    class MT5:
        connected = True
        offset_verified = True

    ui_app.voice = FakeVoice(engine_name="cascade", state="idle")
    ui_app.trading.tv = TV()
    ui_app.trading.mt5 = MT5()
    ctrl = ui.build(ui_app, core, show=False)
    ctrl.bridge.deliver(ComponentStatus(component="mt5", state="down", detail="terminal closed"))  # real, newer
    try:
        panel = ctrl.ensure_panel()
        rows = panel.components
        assert wait_until(lambda: rows["tradingview"].dot.state == "ok", 5.0)
        assert rows["voice"].dot.state == "ok"
        assert rows["omniroute"].dot.state == "unconfigured"        # temp home: no OmniRoute client key
        assert rows["mt5"].dot.state == "down"                      # the real event was not overwritten
        assert panel.brand.state == "idle"
    finally:
        ctrl.shutdown()
        ctrl.island.deleteLater()
        ctrl.tray.icon.deleteLater()
        panel.deleteLater()
        pump(20)


def test_background_start_never_opens_the_panel(ui_app, core, qapp, monkeypatch):
    import sam.ui as ui

    ui_app.config.set("ui.panel_on_start", True)
    monkeypatch.setenv("SAM_BACKGROUND", "1")
    ctrl = ui.build(ui_app, core, show=False)
    try:
        pump(500)
        assert ctrl.panel is not None and not ctrl.panel.isVisible()      # built (warm) but hidden
    finally:
        ctrl.shutdown()
        ctrl.island.deleteLater()
        ctrl.tray.icon.deleteLater()
        if ctrl.panel is not None:
            ctrl.panel.deleteLater()
        pump(20)


# -- tray / bridge / strings --------------------------------------------------------------------------------------
def test_tray_menu_and_alert_balloon(controller, monkeypatch):
    tray = controller.tray
    assert [a.text() for a in tray.actions] == [tr("tray.open"), tr("menu.mute"), tr("tray.restart"), tr("menu.quit")]
    shown: list = []
    monkeypatch.setattr(tray.icon, "showMessage", lambda *a: shown.append(a))
    controller.bridge.deliver(Alert(alert_id=1, kind="price_cross", symbol="XAUUSD", text_ckb="زێڕ گەیشتە ٢٧٠٠"))
    assert shown and shown[0][0] == tr("tray.alert_title") and shown[0][1] == "زێڕ گەیشتە ٢٧٠٠"
    tray.show()                                   # icon pixmaps are drawn on show (after the island)
    assert not tray.icon.icon().isNull()
    controller.bridge.deliver(VoiceState(state="listening"))
    assert not tray.icon.icon().isNull()


def test_restart_goes_through_the_launcher_in_background_mode(tmp_path):
    from sam.config import REPO_ROOT
    from sam.ui.tray import restart_command

    cmd = restart_command(tmp_path, pid=4242)
    assert cmd[1] == str(REPO_ROOT / "SAM.pyw")
    assert cmd[2:5] == ["--background", "--after-pid", "4242"]   # island only; waits for this pid
    assert cmd[-2:] == ["--home", str(tmp_path)]
    assert cmd[0].lower().endswith(("pythonw.exe", "python.exe"))


def test_restart_without_launcher_falls_back_to_the_module(tmp_path):
    from sam.ui.tray import restart_command

    cmd = restart_command(None, pid=7, launcher=tmp_path / "missing.pyw")
    assert cmd[1:] == ["-m", "sam", "--after-pid", "7"]


def test_mute_from_tray_reaches_voice_and_island(controller, ui_app):
    ui_app.voice = FakeVoice()
    controller.tray.mute_action.trigger()
    assert wait_until(lambda: ("mute", True) in ui_app.voice.calls, 5.0)
    assert controller.island.muted and controller.island.status_text() == "بێدەنگ"


def test_app_ui_handle_shows_panel_from_another_thread(controller, ui_app, core):
    core.call_soon(ui_app.ui.show_panel, "activity")
    assert wait_until(lambda: controller.panel is not None and controller.panel.isVisible(), 5.0)
    assert controller.panel.current == "activity"


def test_bridge_delivers_results_errors_and_events_on_the_gui_thread(controller, ui_app, core):
    import threading

    from PySide6.QtCore import QThread

    bridge = controller.bridge
    got: list = []

    async def ok():
        return 42

    async def bad():
        raise ValueError("nope")

    bridge.call(ok(), on_ok=lambda v: got.append(("ok", v, QThread.currentThread() is controller.thread())))
    bridge.call(bad(), on_err=lambda e: got.append(("err", type(e).__name__, True)))
    bridge.run(lambda: threading.current_thread().name, on_ok=lambda v: got.append(("run", v != "MainThread", True)))
    assert wait_until(lambda: len(got) == 3, 5.0)
    assert ("ok", 42, True) in got and ("err", "ValueError", True) in got and ("run", True, True) in got
    events: list = []
    unsubscribe = bridge.subscribe(VoiceState, events.append)
    core.call_soon(ui_app.bus.publish, VoiceState(state="thinking"))
    assert wait_until(lambda: bool(events), 5.0)
    unsubscribe()


def test_bridge_without_core_fails_softly(qapp, make_app):
    from sam.ui.qtbridge import QtBridge

    app = make_app()
    bridge = QtBridge(app, None)
    errors: list = []

    async def never():
        return 1

    assert bridge.call(never(), on_err=errors.append) is None
    assert errors and isinstance(errors[0], RuntimeError)


def test_sorani_strings_use_kurdish_letters_only():
    for key, (ckb, en) in STRINGS.items():
        assert "ي" not in ckb and "ك" not in ckb, key
        assert ckb.strip() and en.strip(), key


def test_ui_build_sets_app_ui_and_registers_defaults(controller, ui_app):
    assert ui_app.ui is not None and hasattr(ui_app.ui, "show_panel")
    assert ui_app.config.get("ui.panel_on_start") is False


def test_few_messages_need_no_scroll_bar(panel, controller):
    """Bubbles pin their wrapped height: three short exchanges fit without a
    scroll bar or a blank gap (a QLabel's own hint made the list ~30% taller)."""
    chat = panel.pages["chat"]
    panel.resize(1180, 900)
    long_reply = ("نرخی زێڕ ئێستا ٢٦٧٤ـە. ئاراستەی H4 سەرەوەیە و نزمیی ئاسیا ڕاماڵراوە، بەڵام هێشتا "
                  "شکاندنی پێکهاتە لە M15 نەبووە")
    for _ in range(3):
        controller.bridge.deliver(Transcript(role="user", text="ترەیدینگ ڤیو بکەرەوە", source="live"))
        controller.bridge.deliver(Transcript(role="assistant", text=long_reply, source="live"))
    pump(200)
    content = sum(b.height() for b in chat.bubbles) + 10 * (len(chat.bubbles) - 1)
    assert content + 20 < chat.scroll.viewport().height()          # the case under test: it all fits
    assert chat.scroll.verticalScrollBar().maximum() == 0
    assert chat.list_widget.minimumSizeHint().height() <= content + 40   # was ~165 px taller before the fix
    for bubble in chat.bubbles:
        assert bubble.label.height() == bubble.label.heightForWidth(bubble.label.width())


def test_scrolled_pages_keep_their_gutter_on_the_scroll_bar_side(panel):
    from sam.ui.pages import SCROLL_GUTTER

    for key in ("settings", "monitor"):
        page = panel.pages[key]
        area = page.findChild(QScrollArea)
        margins = area.widget().layout().contentsMargins()
        # RTL: the vertical scroll bar is on the left; Qt margins are not mirrored.
        assert (margins.left(), margins.right()) == (SCROLL_GUTTER, 0), key


def test_tradingview_and_mt5_status_read_as_connections(panel, controller):
    settings = panel.pages["settings"]
    controller.bridge.deliver(ComponentStatus(component="tradingview", state="ok", detail="connected"))
    controller.bridge.deliver(ComponentStatus(component="mt5", state="down", detail=""))
    assert settings.tv_status.text() == tr("status.connected")
    assert settings.mt5_status.text() == tr("status.not_connected")


def test_hotkey_validation_follows_the_voice_parser():
    from sam.ui.pages.settings import valid_hotkey

    for good in ("ctrl+alt+space", "ctrl+shift+k", "alt+f9", "win+shift+f12"):
        assert valid_hotkey(good), good
    for bad in ("f9", "banana", "ctrl+alt", "ctrl+a+b", ""):
        assert not valid_hotkey(bad), bad            # RegisterHotKey needs a modifier and one key


def test_hotkey_fallback_and_errors_from_the_voice_engine_are_shown(panel, controller):
    from sam.events import Error, SettingsChanged

    page = panel.pages["settings"]
    controller.bridge.deliver(SettingsChanged(key="voice.hotkey", value="ctrl+alt+k"))
    assert page.hotkey.text() == "ctrl+alt+k"
    controller.bridge.deliver(Error(where="voice.hotkey", message_ckb="کورتەڕێگای ctrl+alt+space گیراوە."))
    assert page.hotkey_hint.text() == "کورتەڕێگای ctrl+alt+space گیراوە."


def test_settings_shows_a_hotkey_the_engine_could_not_register(panel, ui_app):
    class Voice:
        def status(self):
            return {"hotkey": {"keys": "ctrl+alt+space", "registered": False, "error": "hotkey already registered"}}

    ui_app.voice = Voice()
    panel.show_page("settings")
    page = panel.pages["settings"]
    assert wait_until(lambda: page.hotkey_hint.text() == tr("set.voice.hotkey_taken"), 5.0)


def test_bubble_height_follows_its_width_both_ways(qapp):
    from sam.ui.pages.chat import Bubble

    text = ("نرخی زێڕ ئێستا ٢٦٧٤ـە. ئاراستەی H4 سەرەوەیە و نزمیی ئاسیا ڕاماڵراوە، بەڵام هێشتا شکاندنی "
            "پێکهاتە لە M15 نەبووە — بۆیە چاوەڕێ دەکەین.")
    bubble = Bubble("assistant", text)
    bubble.fit(260)
    narrow = bubble.label.height()
    bubble.fit(900)                     # the panel was widened: the bubble must shrink back
    wide = bubble.label.height()
    assert wide < narrow
    assert wide == bubble.label.heightForWidth(bubble.width() - 28)
    bubble.deleteLater()
