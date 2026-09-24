"""Island pill: window flags, states, captions, confirmation card, clicks, drag, CPU."""

from __future__ import annotations

import time

import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QFontMetricsF
from PySide6.QtTest import QTest

from sam.events import (Caption, ConfirmRequest, ConfirmResult, LevelMeter, ToolFinished, ToolStarted, Transcript,
                        VoiceState, WorkerProgress)
from sam.ui import theme
from sam.ui.island import Island
from sam.ui.strings import tr
from ui_helpers import FakeVoice, controller, core, pump, qapp, ui_app, wait_until  # noqa: F401 - fixtures

LONG_CKB = ("باشە، چوار هێڵم لەسەر چارتی زێڕ کێشا: دوو پشتگیری لە ٢٦٥١ و ٢٦٤٠ و دوو بەرگری لە ٢٦٨٧ و ٢٦٩٥، "
            "ئەگەر بتەوێت ئاگادارکەرەوەش دادەنێم کاتێک نرخ دەگاتە یەکێک لەو هێڵانە")


@pytest.fixture
def island(qapp, make_app):
    app = make_app()
    widget = Island(app.config)
    yield widget
    widget.hide()
    widget.deleteLater()
    pump(10)


def test_island_is_a_frameless_on_top_tool_window_that_never_takes_focus(island):
    flags = island.windowFlags()
    for flag in (Qt.WindowType.FramelessWindowHint, Qt.WindowType.WindowStaysOnTopHint, Qt.WindowType.Tool,
                 Qt.WindowType.WindowDoesNotAcceptFocus):
        assert flags & flag, flag
    assert island.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    assert island.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    assert island.focusPolicy() == Qt.FocusPolicy.NoFocus
    pill = island._pill_rect()
    assert (round(pill.width()), round(pill.height())) == theme.PILL_COMPACT


def test_island_starts_top_centre_of_the_primary_screen(island):
    from PySide6.QtGui import QGuiApplication

    area = QGuiApplication.primaryScreen().availableGeometry()
    island.show()
    pump(20)
    geo = island.geometry()
    assert abs(geo.center().x() - area.center().x()) <= 2
    pill_top = geo.top() + theme.SHADOW_MARGIN[1]
    assert area.top() <= pill_top <= area.top() + 20


@pytest.mark.parametrize("state, word", [
    ("idle", "ئامادە"), ("listening", "گوێ دەگرم"), ("thinking", "بیردەکەمەوە"), ("speaking", "قسە دەکەم"),
    ("working", "کار دەکەم"), ("error", "هەڵە"), ("muted", "بێدەنگ"), ("sleeping", "ئامادە"),
])
def test_state_changes_update_status_word_and_colour(island, state, word):
    island.handle_event(VoiceState(state=state, engine="cascade"))
    assert island.status_text() == word
    assert island.state_color() == theme.STATE_COLORS[state]


def test_unknown_state_falls_back_to_ready(island):
    island.handle_event(VoiceState(state="bogus"))  # type: ignore[arg-type]
    assert island.state == "idle"
    assert island.status_text() == "ئامادە"


def test_state_colours_are_distinct_for_active_states():
    active = [theme.STATE_COLORS[s] for s in ("idle", "listening", "thinking", "speaking", "working", "error")]
    assert len(set(active)) == len(active)


def test_caption_elides_long_rtl_text_and_the_pill_grows(island):
    island.show()
    island.handle_event(Caption(text=LONG_CKB, role="assistant", final=True))
    shown = island.displayed_caption()
    assert shown != LONG_CKB and "…" in shown
    rect = island._caption_rect(island._pill_rect(*theme.PILL_EXPANDED))
    assert QFontMetricsF(island.f_caption).horizontalAdvance(shown) <= rect.width() + 0.5
    assert shown.startswith("باشە")              # final text: keep the beginning
    assert wait_until(lambda: island._pill_rect().height() >= theme.PILL_EXPANDED[1] - 0.5, 2.0)
    assert island.height() > theme.PILL_COMPACT[1] + theme.SHADOW_MARGIN[1]


def test_partial_caption_keeps_the_newest_words(island):
    island.handle_event(Caption(text=LONG_CKB, role="user", final=False))
    shown = island.displayed_caption()
    assert "…" in shown
    assert shown.endswith("هێڵانە")               # live speech: the tail is what matters
    assert island.caption is not None and island.caption.tone == "user"


def test_short_caption_is_not_elided_and_clearing_collapses(island):
    island.show()
    island.handle_event(Transcript(role="user", text="ترەیدینگ ڤیو بکەرەوە", source="live"))
    assert island.displayed_caption() == "ترەیدینگ ڤیو بکەرەوە"
    assert wait_until(lambda: island._expansion > 0.99, 2.0)
    island.clear_caption()
    assert wait_until(lambda: island._expansion < 0.01, 2.0)
    assert round(island._pill_rect().height()) == theme.PILL_COMPACT[1]


def test_tool_events_show_sorani_labels(island):
    island.handle_event(ToolStarted(call_id="1", name="draw_on_chart", args={}))
    assert island.caption.text == "کێشان لەسەر چارت…"
    island.handle_event(ToolFinished(call_id="1", name="draw_on_chart", ok=True, summary="Drew 2 lines"))
    assert island.caption.tone == "success"
    # English tool summaries are not shown to a Sorani user: the island says it in Sorani.
    island.handle_event(ToolFinished(call_id="2", name="open_app", ok=False, summary="Not installed."))
    assert island.caption.text == tr("island.tool_failed", label="کردنەوەی بەرنامە")
    assert island.caption.text == "کردنەوەی بەرنامە سەرکەوتوو نەبوو" and island.caption.tone == "danger"
    # A Sorani summary is already fit for the user and is shown as it is.
    island.handle_event(ToolFinished(call_id="3", name="open_app", ok=False, summary="ترەیدینگ ڤیو دانەمەزراوە."))
    assert island.caption.text == "ترەیدینگ ڤیو دانەمەزراوە."


def test_worker_progress_line_and_level_meter(island):
    island.handle_event(WorkerProgress(task_id="t", step=2, max_steps=8, text_ckb="هەنگاوی دووەم"))
    assert island._progress == {"step": 2, "max": 8, "done": False, "ok": None, "done_at": None, "task_id": "t"}
    assert island.animating
    island.handle_event(VoiceState(state="listening"))
    island.handle_event(LevelMeter(source="mic", level=0.9))
    assert wait_until(lambda: island._level > 0.3, 1.0)


def test_progress_line_only_animates_while_it_moves(island):
    island.handle_event(WorkerProgress(task_id="w1", step=3, max_steps=6, text_ckb="هەنگاوی سێیەم"))
    assert island.animating                                    # gliding to 50 %
    assert wait_until(lambda: not island.animating, 3.0)       # static line: no 60 fps while a step runs
    assert island._progress_shown == pytest.approx(0.5)
    island.handle_event(WorkerProgress(task_id="w1", step=4, max_steps=6, text_ckb=""))
    assert island.animating
    island.handle_event(WorkerProgress(task_id="w1", step=6, max_steps=6, text_ckb="تەواو بوو", done=True, ok=True))
    assert wait_until(lambda: island._progress is None, 4.0)   # finished line fades out ...
    assert wait_until(lambda: not island.animating, 2.0)       # ... and the clock stops


def test_tool_finished_closes_its_progress_line(island):
    island.handle_event(WorkerProgress(task_id="call-7", step=1, max_steps=3, text_ckb="شیکردنەوە…"))
    island.handle_event(ToolFinished(call_id="other", name="get_price", ok=True, summary=""))
    assert island._progress is not None and not island._progress["done"]
    island.handle_event(ToolFinished(call_id="call-7", name="analyze_market", ok=False, summary="timeout"))
    assert island._progress["done"] and island._progress["ok"] is False
    assert wait_until(lambda: island._progress is None, 4.0)


def test_stale_progress_is_dropped(island):
    island._stale.setInterval(150)            # PROGRESS_STALE_MS in production
    island.handle_event(WorkerProgress(task_id="lost", step=0, max_steps=0, text_ckb="…"))
    assert island.animating                   # indeterminate sweep
    assert wait_until(lambda: island._progress is None, 3.0)
    assert wait_until(lambda: not island.animating, 2.0)


def test_background_layer_is_cached_at_the_painter_scale(island):
    from PySide6.QtGui import QPixmap

    island.show()
    pix = QPixmap(island.width() * 2, island.height() * 2)
    pix.setDevicePixelRatio(2.0)
    pix.fill(Qt.GlobalColor.transparent)
    island.render(pix)
    assert island._bg is not None and island._bg.devicePixelRatio() == 2.0   # crisp text, not a 1x upscale
    assert island._bg.width() == island.width() * 2


def test_idle_island_stops_its_animation_timer(island):
    island.handle_event(VoiceState(state="listening"))
    assert island.animating
    island.handle_event(VoiceState(state="idle"))
    assert wait_until(lambda: not island.animating, 2.0)       # ~0% CPU when idle
    for state in ("sleeping", "muted", "error"):
        island.handle_event(VoiceState(state=state))
        assert not island.animating


def test_confirmation_card_shows_and_hides(island):
    island.show()
    island.handle_event(ConfirmRequest(confirm_id="c1", question_ckb="فایلەکە بسڕمەوە؟", detail="x.txt",
                                       tool_name="files", expires_at=time.time() + 20))
    assert island.card.isVisible()
    assert island.card.question.text() == "فایلەکە بسڕمەوە؟"
    assert island.card.yes_button.text() == "بەڵێ" and island.card.no_button.text() == "نەخێر"
    assert island.height() > theme.PILL_COMPACT[1] + 100
    island.handle_event(ConfirmResult(confirm_id="c1", approved=False, via="voice"))
    assert not island.card.isVisible()


def test_card_click_emits_answer(island):
    answers: list[tuple[str, bool, str]] = []
    island.confirmAnswered.connect(lambda cid, ok, via: answers.append((cid, ok, via)))
    island.show()
    island.handle_event(ConfirmRequest(confirm_id="c2", question_ckb="بنێرم؟", expires_at=time.time() + 20))
    QTest.mouseClick(island.card.yes_button, Qt.MouseButton.LeftButton)
    assert answers == [("c2", True, "click")]
    assert not island.card.isVisible()


def test_card_expires_locally_as_no(island):
    answers: list[tuple[str, bool, str]] = []
    island.confirmAnswered.connect(lambda cid, ok, via: answers.append((cid, ok, via)))
    island.handle_event(ConfirmRequest(confirm_id="c3", question_ckb="؟", expires_at=time.time() + 0.3))
    assert wait_until(lambda: bool(answers), 3.0)
    assert answers == [("c3", False, "timeout")]


def test_card_countdown_uses_sorani_digits(island):
    island.handle_event(ConfirmRequest(confirm_id="c4", question_ckb="؟", expires_at=time.time() + 17.5))
    assert island.card.count_label.text().startswith("١٨")


def test_right_click_menu_has_the_sorani_actions(island):
    menu = island.build_menu()
    texts = [a.text() for a in menu.actions() if not a.isSeparator()]
    for key in ("menu.open_panel", "menu.mute", "menu.settings", "menu.quit"):
        assert tr(key) in texts
    muted: list[bool] = []
    island.muteRequested.connect(muted.append)
    next(a for a in menu.actions() if a.text() == tr("menu.mute")).trigger()
    assert muted == [True] and island.muted


def _click(island, double=False):
    centre = island._pill_rect().center().toPoint()
    if double:
        QTest.mouseDClick(island, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, centre)
    else:
        QTest.mouseClick(island, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, centre)


def test_click_toggles_listening_through_the_core(controller, ui_app):
    ui_app.voice = FakeVoice()
    controller.island.show()
    _click(controller.island)
    assert wait_until(lambda: ("toggle", None) in ui_app.voice.calls, 3.0)


def test_click_without_voice_explains_in_sorani(controller, ui_app):
    controller.island.show()
    _click(controller.island)
    assert wait_until(lambda: controller.island.caption is not None, 2.0)
    assert controller.island.caption.text == tr("island.voice_missing")


def test_double_click_opens_the_panel_without_toggling(controller, ui_app):
    ui_app.voice = FakeVoice()
    controller.island.show()
    _click(controller.island, double=True)
    assert wait_until(lambda: controller.panel is not None and controller.panel.isVisible(), 3.0)
    pump(400)
    assert ("toggle", None) not in ui_app.voice.calls


def test_drag_moves_the_island_and_remembers_the_position(qapp, make_app):
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QMouseEvent

    app = make_app()
    island = Island(app.config)
    island.show()
    pump(20)
    start = island._pill_rect().center()
    anchor = island.anchor()

    def send(kind, local: QPointF, buttons):
        glob = QPointF(island.mapToGlobal(local.toPoint()))
        event = QMouseEvent(kind, local, glob, Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier)
        qapp.sendEvent(island, event)

    send(QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton)
    send(QEvent.Type.MouseMove, start + QPointF(-120, 40), Qt.MouseButton.LeftButton)
    send(QEvent.Type.MouseButtonRelease, start + QPointF(-120, 40), Qt.MouseButton.NoButton)
    moved = island.anchor()
    assert moved.x() < anchor.x() - 50 and moved.y() > anchor.y() + 10
    saved = app.config.get("ui.island_pos")
    assert saved == [round(moved.x()), round(moved.y())]
    again = Island(app.config)
    assert abs(again.anchor().x() - moved.x()) < 1 and abs(again.anchor().y() - moved.y()) < 1
    again.reset_position()
    assert app.config.get("ui.island_pos") is None
    for w in (island, again):
        w.hide()
        w.deleteLater()


def test_saved_position_off_screen_falls_back_to_default(qapp, make_app):
    app = make_app()
    app.config.set("ui.island_pos", [-50000, -50000])
    island = Island(app.config)
    default = island.default_anchor()
    assert island.anchor() == default
    island.deleteLater()


def test_no_activate_style_is_skipped_offscreen(island):
    from sam.ui.island import apply_no_activate

    island.show()
    assert apply_no_activate(island) is False      # offscreen platform: nothing to do, no crash


def test_frame_rate_is_60_only_while_a_voice_level_moves(island):
    from sam.ui.island import FRAME_MS, SLOW_FRAME_MS

    assert 1000 / FRAME_MS <= 60 and 1000 / SLOW_FRAME_MS <= 31
    island.handle_event(VoiceState(state="thinking"))
    assert island.animating and island._timer.interval() == SLOW_FRAME_MS    # comet: slow motion
    island.handle_event(VoiceState(state="listening"))
    island.handle_event(LevelMeter(source="mic", level=0.8))
    assert island._timer.interval() == FRAME_MS                              # a voice moves the orb
    assert wait_until(lambda: island._timer.interval() == SLOW_FRAME_MS, 3.0)  # silence: back to 30 fps
    assert island.animating                                                  # the open mic still breathes
