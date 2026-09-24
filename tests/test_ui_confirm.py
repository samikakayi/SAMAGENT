"""Confirmation end to end: real ConfirmBroker on the core thread -> bus ->
UiAdapter -> queued Qt signal -> island card -> click -> broker.resolve."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from ui_helpers import controller, core, pump, qapp, ui_app, wait_until  # noqa: F401 - fixtures


def _ask(ui_app, core, timeout_s: float = 10.0):
    return core.submit(ui_app.confirm.confirm("فایلەکانی Downloads بسڕمەوە؟", "Remove-Item *.tmp",
                                              tool_name="run_powershell", timeout_s=timeout_s))


@pytest.mark.parametrize("button, expected", [("yes_button", True), ("no_button", False)])
def test_card_click_resolves_the_broker(controller, ui_app, core, button, expected):
    card = controller.island.card
    controller.island.show()
    future = _ask(ui_app, core)
    assert wait_until(card.isVisible, 3.0)
    assert card.question.text() == "فایلەکانی Downloads بسڕمەوە؟"
    QTest.mouseClick(getattr(card, button), Qt.MouseButton.LeftButton)
    assert future.result(timeout=5) is expected
    assert wait_until(lambda: not card.isVisible(), 2.0)
    rows = ui_app.db.query("SELECT ok, source FROM activity WHERE kind='confirm'")
    assert rows and rows[-1]["source"] == "click" and bool(rows[-1]["ok"]) is expected


def test_broker_timeout_is_no_and_hides_the_card(controller, ui_app, core):
    card = controller.island.card
    controller.island.show()
    future = _ask(ui_app, core, timeout_s=0.6)
    assert wait_until(card.isVisible, 3.0)
    assert future.result(timeout=5) is False
    assert wait_until(lambda: not card.isVisible(), 3.0)


def test_voice_answer_elsewhere_hides_the_card(controller, ui_app, core):
    card = controller.island.card
    future = _ask(ui_app, core)
    assert wait_until(lambda: bool(card.pending_ids), 3.0)
    core.call_soon(ui_app.confirm.offer_transcript, "بەڵێ")
    assert future.result(timeout=5) is True
    assert wait_until(lambda: not card.pending_ids, 3.0)


def test_two_requests_stack_newest_first(controller, ui_app, core):
    card = controller.island.card
    controller.island.show()
    first = _ask(ui_app, core)
    assert wait_until(lambda: len(card.pending_ids) == 1, 3.0)
    second = core.submit(ui_app.confirm.confirm("پەیامەکە بنێرم؟", tool_name="type_text", timeout_s=10))
    assert wait_until(lambda: len(card.pending_ids) == 2, 3.0)
    assert card.question.text() == "پەیامەکە بنێرم؟"
    QTest.mouseClick(card.no_button, Qt.MouseButton.LeftButton)
    assert second.result(timeout=5) is False
    assert wait_until(lambda: card.question.text() == "فایلەکانی Downloads بسڕمەوە؟", 3.0)
    QTest.mouseClick(card.yes_button, Qt.MouseButton.LeftButton)
    assert first.result(timeout=5) is True
    pump(50)
