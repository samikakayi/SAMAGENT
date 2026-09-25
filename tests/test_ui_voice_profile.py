"""Island notices (quota / listening closed) and the Settings «گوێگرتن و دەنگی من»
card with its enrollment dialog (Qt offscreen; the voice engine is a fake)."""

from __future__ import annotations

import time
from typing import Any

import pytest

from sam.events import Transcript, VoiceState
from sam.ui.island import Island
from sam.ui.island_hints import MODELS_WORD, IslandHints
from sam.ui.strings import state_word
from sam.voice import strings as voice_strings
from sam.voice.notices import VoiceEnrollRequest, VoiceNotice
from ui_helpers import core, pump, qapp, ui_app, wait_until  # noqa: F401 - fixtures


class FakeEnrollVoice:
    """The voice engine's enrollment API, scripted."""

    def __init__(self, record_results: list[dict[str, Any]] | None = None) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.enrolled = False
        self.record_results = list(record_results or [])
        self.finish_results: list[dict[str, Any]] = []
        self.usable = True
        self.reset = 0

    async def enroll_begin(self):
        self.calls.append(("begin", None))
        return {"ok": True, "sentences": list(voice_strings.ENROLL_SENTENCES[:3])}

    async def enroll_record(self, index):
        self.calls.append(("record", index))
        return self.record_results.pop(0) if self.record_results else {"ok": True, "speech_ms": 2400}

    async def enroll_finish(self):
        self.calls.append(("finish", None))
        if self.finish_results:
            return self.finish_results.pop(0)
        self.enrolled = True
        return {"ok": True, "message_ckb": voice_strings.ENROLL_SAVED}

    async def enroll_cancel(self):
        self.calls.append(("cancel", None))

    async def voiceprint_delete(self):
        self.calls.append(("delete", None))
        self.enrolled = False
        return {"ok": True, "existed": True}

    def voiceprint_status(self):
        return {"enrolled": self.enrolled, "enabled": self.enrolled, "sherpa": True, "usable": self.usable}

    def reset_user_level(self):
        self.reset += 1
        return {"ok": True}


@pytest.fixture
def island(qapp, make_app):
    app = make_app()
    widget = Island(app.config)
    yield widget
    widget.hide()
    widget.deleteLater()
    pump(10)


def test_hints_models_word_sticks_until_reset_or_a_real_answer():
    now = [1000.0]
    hints = IslandHints(clock=lambda: now[0])
    shown = hints.on_notice(VoiceNotice(kind="models", text_ckb="سنووری ئەمڕۆ پڕە — دوای کاتژمێر ١٠ی بەیانی",
                                        until=5000.0))
    assert shown == ("سنووری ئەمڕۆ پڕە — دوای کاتژمێر ١٠ی بەیانی", "danger")
    assert hints.status_override("idle") == MODELS_WORD and hints.status_override("listening") is None
    hints.on_answer("assistant", "نرخی زێڕ ٢٦٥٠ دۆلارە.")
    assert hints.status_override("idle") is None
    hints.on_notice(VoiceNotice(kind="models", text_ckb="x", until=0.0))  # unknown reset: 10 minutes
    now[0] += 601
    assert hints.status_override("idle") is None
    assert hints.on_notice(VoiceNotice(kind="closed", text_ckb=voice_strings.LISTEN_CLOSED)) == (
        voice_strings.LISTEN_CLOSED, "system")


def test_island_shows_quota_and_closed_notices(island):
    island.handle_event(VoiceState(state="idle"))
    island.handle_event(VoiceNotice(kind="closed", text_ckb=voice_strings.LISTEN_NO_SPEECH, detail="no_speech"))
    assert island.caption is not None and island.caption.text == voice_strings.LISTEN_NO_SPEECH
    assert island.status_text() == state_word("idle")
    text = voice_strings.MODELS_EXHAUSTED_DAILY.format(time="کاتژمێر ١٠ی بەیانی")
    island.handle_event(VoiceNotice(kind="models", text_ckb=text, until=time.time() + 3600))
    assert island.caption.text == text and island.caption.tone == "danger"
    assert island.status_text() == MODELS_WORD
    island.handle_event(VoiceState(state="thinking"))
    assert island.status_text() == state_word("thinking")        # while working, the real state
    island.handle_event(VoiceState(state="idle"))
    island.handle_event(Transcript(role="assistant", text="باشە، کرایەوە.", source="cascade"))
    assert island.status_text() == state_word("idle")


@pytest.fixture
def card(ui_app, core, qapp):
    from sam.ui.qtbridge import QtBridge
    from sam.ui.voice_profile import VoiceProfileCard

    ui_app.voice = FakeEnrollVoice()
    bridge = QtBridge(ui_app, core)
    bridge.attach()
    widget = VoiceProfileCard(ui_app, bridge)
    widget.show()
    pump(30)
    yield widget, ui_app, bridge
    if widget.dialog is not None:
        widget.dialog.close()
    bridge.detach()
    widget.deleteLater()
    pump(20)


def test_card_controls_have_sorani_accessible_names_and_save_settings(card):
    widget, app, _bridge = card
    names = {w.accessibleName() for w in (widget.followup, widget.strict, widget.only, widget.sens,
                                          widget.enroll_button, widget.delete_button)}
    assert {"ناساندنی دەنگی من", "سڕینەوەی دەنگی من", "تەنها دەنگی من"} <= names
    widget.followup.setValue(9)
    widget.sens.setCurrentIndex(2)
    widget.only.setChecked(False)
    widget.strict.setCurrentIndex(2)
    assert app.config.get("voice.followup_s") == 9
    assert app.config.get("voice.only_my_voice_sensitivity") == "high"
    assert app.config.get("voice.only_my_voice") is False and app.config.get("voice.gate_margin_db") == 18.0


def test_enrollment_dialog_records_each_sentence_then_saves(card):
    widget, app, _bridge = card
    app.voice.record_results = [{"ok": False, "reason": "too_short", "message_ckb": voice_strings.ENROLL_TOO_SHORT}]
    dialog = widget.open_enrollment()
    pump(30)
    assert app.voice.calls == [] and dialog.start_button.isVisible()      # nothing recorded before Start
    assert dialog.start_button.accessibleName() == "دەست پێبکە"
    dialog.start_button.click()
    assert wait_until(lambda: dialog.retry_button.isVisible(), timeout=5)
    assert dialog.status.text() == voice_strings.ENROLL_TOO_SHORT       # the first try was too short
    dialog.retry_button.click()
    assert wait_until(lambda: ("finish", None) in app.voice.calls, timeout=8)
    assert wait_until(lambda: dialog.status.text() == voice_strings.ENROLL_SAVED, timeout=5)
    records = [c for c in app.voice.calls if c[0] == "record"]
    assert [i for _, i in records] == [0, 0, 1, 2]
    dialog.close()
    pump(20)
    assert ("cancel", None) not in app.voice.calls                     # a finished enrollment is not cancelled
    assert wait_until(lambda: widget.delete_button.isEnabled(), timeout=5)   # the card shows the voiceprint


def test_enrollment_dialog_reads_only_the_rejected_sentence_again(card):
    widget, app, _bridge = card
    app.voice.finish_results = [{"ok": False, "reason": "inconsistent", "retry_index": 1,
                                 "message_ckb": voice_strings.ENROLL_REPEAT_ONE}]
    dialog = widget.open_enrollment()
    dialog.start_button.click()
    assert wait_until(lambda: dialog.retry_button.isVisible(), timeout=8)
    assert dialog.status.text() == voice_strings.ENROLL_REPEAT_ONE
    dialog.retry_button.click()
    assert wait_until(lambda: dialog.status.text() == voice_strings.ENROLL_SAVED, timeout=8)
    records = [i for c, i in app.voice.calls if c == "record"]
    assert records == [0, 1, 2, 1] and [c for c, _ in app.voice.calls].count("finish") == 2


def test_card_shows_the_level_resets_it_and_says_when_the_voiceprint_cannot_run(card):
    widget, app, _bridge = card
    app.config.set("voice.gate_user_level_db", -26.0)
    app.voice.enrolled, app.voice.usable = True, False
    widget.refresh()
    assert wait_until(lambda: "ئامادە نییە" in widget.state.text(), timeout=5)
    assert "-26 dB" in widget.level.text() and widget.level_reset.isEnabled()
    assert widget.level_reset.accessibleName() == "ئاستی دەنگم لەبیر بکە"
    widget.level_reset.click()
    assert wait_until(lambda: app.voice.reset == 1, timeout=5)


def test_spoken_request_opens_the_dialog_and_cancel_stops_recording(card):
    widget, app, bridge = card
    bridge.deliver(VoiceEnrollRequest(source="voice"))
    assert wait_until(lambda: widget.dialog is not None and widget.dialog.isVisible(), timeout=5)
    pump(30)
    assert ("begin", None) not in app.voice.calls                          # the Start step first
    widget.dialog.start_button.click()
    assert wait_until(lambda: ("begin", None) in app.voice.calls, timeout=5)
    widget.dialog.cancel_button.click()
    assert wait_until(lambda: ("cancel", None) in app.voice.calls, timeout=5)


def test_delete_button_removes_the_voiceprint(card):
    widget, app, _bridge = card
    app.voice.enrolled = True
    widget.refresh()
    assert wait_until(lambda: widget.delete_button.isEnabled(), timeout=5)
    widget.delete_button.click()
    assert wait_until(lambda: ("delete", None) in app.voice.calls, timeout=5)
    assert wait_until(lambda: not widget.delete_button.isEnabled(), timeout=5)


def test_settings_page_contains_the_card(ui_app, core, qapp):
    import sam.ui as ui

    ctrl = ui.build(ui_app, core, show=False)
    try:
        panel = ctrl.ensure_panel()
        page = panel.pages["settings"]
        assert page.voice_profile.enroll_button.text() == "ناساندنی دەنگی من"
    finally:
        ctrl.shutdown()
        if ctrl.panel is not None:
            ctrl.panel.deleteLater()
        ctrl.island.deleteLater()
        ctrl.tray.icon.deleteLater()
        pump(20)
