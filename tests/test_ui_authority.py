"""Settings: «دەسەڵاتی تەواو — بێ پرسیار» (safety.full_authority), on by default."""

from __future__ import annotations

from sam.events import SettingsChanged
from sam.ui.strings import tr
from ui_helpers import controller, core, pump, qapp, ui_app, wait_until  # noqa: F401


def test_the_full_authority_switch(controller, ui_app):
    panel = controller.ensure_panel()
    panel.show()
    pump(30)
    page = panel.pages["settings"]
    switch = page.authority
    assert switch.isChecked()                                                  # default: no questions
    assert switch.accessibleName() == tr("a11y.authority") and switch.objectName() == "full_authority"
    assert tr("set.authority") == "دەسەڵاتی تەواو — بێ پرسیار"
    switch.toggle()                                                            # UI Automation's Toggle works too
    assert wait_until(lambda: ui_app.config.get("safety.full_authority") is False, 5.0)
    controller.bridge.deliver(SettingsChanged(key="safety.full_authority", value=True))
    assert switch.isChecked()
