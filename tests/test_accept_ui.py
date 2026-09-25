"""Acceptance fixes 2026-09-24 (UI): the settings page is operable through
accessibility (unique Sorani names; toggles work without a mouse click), and a
normal launch opens the panel itself instead of waiting for the launcher's
show request (the user's evening test: ~8 s without a window)."""

from __future__ import annotations

import re

import pytest
from PySide6.QtGui import QAccessible
from PySide6.QtWidgets import QAbstractButton, QComboBox, QLineEdit, QSpinBox

from sam.ui.pages.settings import KEY_ROWS
from sam.ui.strings import tr
from ui_helpers import controller, core, pump, qapp, ui_app, wait_until  # noqa: F401

ARABIC_ONLY = re.compile("[يك]")    # Arabic yeh/kaf: never in Sorani text


@pytest.fixture
def settings(controller):
    panel = controller.ensure_panel()
    panel.show()
    panel.show_page("settings")
    pump(30)
    return panel.pages["settings"]


def _name(widget) -> str:
    return QAccessible.queryAccessibleInterface(widget).text(QAccessible.Text.Name)


def _do(widget, action: str) -> None:
    QAccessible.queryAccessibleInterface(widget).actionInterface().doAction(action)


def test_every_settings_control_has_a_unique_sorani_accessible_name(settings):
    controls = [w for w in settings.findChildren(QAbstractButton) + settings.findChildren(QLineEdit)
                + settings.findChildren(QComboBox) + settings.findChildren(QSpinBox)
                if w.isVisibleTo(settings) and not isinstance(w.parent(), (QComboBox, QSpinBox))]   # their inner edits
    names = [_name(w) for w in controls]
    named = [n for n in names if n]
    assert len(named) == len(controls), [type(w).__name__ for w, n in zip(controls, names) if not n]
    buttons = [n for w, n in zip(controls, names) if isinstance(w, QAbstractButton)]
    assert len(set(buttons)) == len(buttons), "two buttons with the same name cannot be told apart"
    for name in named:
        assert not ARABIC_ONLY.search(name), name
    row = settings.key_rows["groq_api_key"]
    assert _name(row.save_button) == tr("a11y.key.save", name="Groq") == "پاشەکەوتکردنی کلیلی Groq"
    assert _name(row.edit) == "کلیلی Groq" and _name(row.test_button) == "تاقیکردنەوەی کلیلی Groq"
    assert _name(settings.always) == "هەمیشە گوێ بگرە"
    assert _name(settings.engine_buttons["cascade"]) == "بزوێنەری دەنگ: دەنگی ئاسایی"
    assert QAccessible.queryAccessibleInterface(row.edit).state().passwordEdit       # the key never reads back
    assert {r.objectName() for r in settings.key_rows.values()} == {f"KeyRow_{name}" for name, _, _ in KEY_ROWS}


def test_settings_toggles_work_through_accessibility_without_a_click(settings, ui_app):
    """UI Automation's Toggle checks a button without a click: the choice must
    still be saved (the engine chips listened to 'clicked' only)."""
    _do(settings.engine_buttons["cascade"], "Toggle")
    assert wait_until(lambda: ui_app.config.get("voice.engine") == "cascade", 5.0)
    _do(settings.always, "Toggle")
    assert wait_until(lambda: ui_app.config.get("voice.always_listening") is True, 5.0)
    _do(settings.engine_buttons["auto"], "Toggle")
    assert wait_until(lambda: ui_app.config.get("voice.engine") == "auto", 5.0)
    assert settings.engine_buttons["auto"].isChecked() and not settings.engine_buttons["cascade"].isChecked()


def test_nav_items_switch_pages_through_accessibility(controller):
    panel = controller.ensure_panel()
    panel.show()
    pump(20)
    _do(panel.nav["monitor"], "Toggle")
    pump(20)
    assert panel.current == "monitor" and panel.stack.currentWidget() is panel.pages["monitor"]
    panel.show_page("chat")                       # programmatic switches do not loop through 'toggled'
    pump(20)
    assert panel.current == "chat" and panel.nav["chat"].isChecked()


def test_saving_a_key_clears_that_providers_rests(settings, ui_app, monkeypatch):
    monkeypatch.setattr(ui_app.secrets, "set", lambda name, value: {"name": name, "stored": True})
    ui_app.llm._cooldown["gemini:*"] = 10 ** 12                                      # noqa: SLF001
    ui_app.llm._load_health()                                                         # noqa: SLF001
    row = settings.key_rows["gemini_api_key"]
    row.edit.setText("AIza" + "FakeKey0123456789abcdefXYZ")
    row.save_button.click()
    assert wait_until(lambda: not ui_app.llm.cooling("gemini:gemini-3.5-flash-lite"), 5.0)


def test_a_normal_launch_opens_the_panel_itself(ui_app, core, qapp, monkeypatch):
    import sam.ui as ui

    monkeypatch.setenv("SAM_SHOW_PANEL", "1")
    monkeypatch.setenv("SAM_BACKGROUND", "0")
    ctrl = ui.build(ui_app, core, show=False)
    try:
        assert wait_until(lambda: ctrl.panel is not None and ctrl.panel.isVisible(), 5.0)
    finally:
        ctrl.shutdown()
        ctrl.island.deleteLater()
        ctrl.tray.icon.deleteLater()
        if ctrl.panel is not None:
            ctrl.panel.deleteLater()
        pump(20)


def test_background_launch_keeps_the_panel_hidden_even_when_asked(ui_app, core, qapp, monkeypatch):
    import sam.ui as ui

    monkeypatch.setenv("SAM_SHOW_PANEL", "1")
    monkeypatch.setenv("SAM_BACKGROUND", "1")
    ctrl = ui.build(ui_app, core, show=False)
    try:
        pump(400)
        assert ctrl.panel is not None and not ctrl.panel.isVisible()
    finally:
        ctrl.shutdown()
        ctrl.island.deleteLater()
        ctrl.tray.icon.deleteLater()
        if ctrl.panel is not None:
            ctrl.panel.deleteLater()
        pump(20)


def test_bring_to_front_is_harmless_offscreen(controller):
    from sam.ui.win32 import bring_to_front

    panel = controller.ensure_panel()
    panel.show_and_raise("chat")
    assert bring_to_front(panel) is False and panel.in_front is False          # offscreen: nothing to raise
