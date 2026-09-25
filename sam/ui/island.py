"""The island: SAM's always-on-top pill at the top centre of the screen.

Window: frameless, translucent, always on top, a tool window (no taskbar
button) that never takes focus (Qt ``WindowDoesNotAcceptFocus`` +
``WA_ShowWithoutActivating`` + Win32 ``WS_EX_NOACTIVATE``), so clicking it does
not steal the keyboard from the app the user is working in.

Layout (right-to-left, Sorani first)::

    ╭──────────────────────────────────────────────╮
    │ ▁▃▅▃▁                            SAM   (orb) │   58 px compact pill
    │                              گوێ دەگرم       │
    │ ‥‥‥‥‥‥‥‥‥‥‥ caption, one line, elided ‥‥‥‥‥ │   grows to 94 px (animated)
    │ ━━━━━━━━━━ worker progress ━━━━━━━━━━━━━━━━━ │
    ╰──────────────────────────────────────────────╯
            [ confirmation card: بەڵێ / نەخێر ]

Cost: one QTimer runs only while something moves (listening, thinking,
speaking, working, a level decaying, a progress bar): 60 fps while a voice
level moves the orb, 30 fps for slow motion (see FRAME_MS). Idle / sleeping /
muted / error states stop it, so the idle island uses ~0% CPU (measured 0.00 %
on screen by acceptance/ui_onscreen.py).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import (Property, QEasingCurve, QElapsedTimer, QPoint, QPointF, QPropertyAnimation, QRect,
                            QRectF, Qt, QTimer, Signal)
from PySide6.QtGui import (QBrush, QColor, QFontMetricsF, QGuiApplication, QLinearGradient, QPainter,
                           QPainterPath, QPen, QPixmap, QRegion, QTextOption)
from PySide6.QtWidgets import QApplication, QMenu, QWidget

from ..events import (Alert, Caption, ConfirmRequest, ConfirmResult, Error, LevelMeter, SettingsChanged,
                      ToolFinished, ToolStarted, Transcript, VoiceState, WorkerProgress)
from ..textnorm import is_arabic_script
from . import theme
from .confirm_card import ConfirmCard
from .island_hints import IslandHints, is_notice
from .island_paint import device_scale, paint_glass, paint_meter, paint_progress
from .orb import paint_orb
from .strings import state_word, tool_label, tr
from .widgets import A_RIGHT, elide, is_rtl
from .win32 import apply_no_activate

# Frame clock. Measured on this PC's desktop (175 % scaling, 2026-09-24, before
# the orb caches): a 16 ms timer ran at 62.5 fps and cost 27-31 % of one core
# while listening, 17 ms ran at 58 fps for 24 %, 33 ms at 30 fps for 12.5 %
# (paint ~1.8 ms/frame in all three; the rest is the layered-window flush). So
# 60 fps is kept only while a voice level moves the orb and the meter; slow
# motion (thinking/working comet, a silent open mic, the progress glide) runs at
# 30. acceptance/ui_perf.py re-measures it (last run: 52 fps / 26 % listening,
# 30 fps / 10-12 % thinking or silent, 0 % idle).
FRAME_MS = 17                 # <= 60 fps
SLOW_FRAME_MS = 33            # ~30 fps
LIVELY_LEVEL = 0.06           # voice level above which the orb gets full frame rate
CAPTION_HOLD_FINAL_MS = 7000  # collapse this long after the last final caption
CAPTION_HOLD_PARTIAL_MS = 15000
PROGRESS_DONE_HOLD_S = 1.6    # a finished progress line stays this long, then fades out
# A progress line with no update for this long is dropped: a task that dies
# without its final event must not leave a line (and a running timer) behind.
# Worker steps are LLM calls with <= 60 s timeouts, so 4 minutes is generous.
PROGRESS_STALE_MS = 240_000
TONE_COLORS = {
    "assistant": "#EEF1F7", "user": "#BFEFEA", "system": "#A9B3C7", "success": theme.SUCCESS,
    "danger": theme.DANGER, "alert": theme.WARNING,
}
TONE_ICONS = {"user": "mic", "system": "bolt", "success": "check", "danger": "warning", "alert": "info"}


@dataclass
class CaptionLine:
    text: str
    tone: str = "assistant"      # assistant|user|system|success|danger|alert
    final: bool = True


class Island(QWidget):
    toggleListeningRequested = Signal()
    openPanelRequested = Signal()
    settingsRequested = Signal()
    quitRequested = Signal()
    muteRequested = Signal(bool)
    stopAllRequested = Signal()
    confirmAnswered = Signal(str, bool, str)      # confirm_id, approved, via

    def __init__(self, config: Any = None, parent: QWidget | None = None) -> None:
        flags = (Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
                 | Qt.WindowType.WindowDoesNotAcceptFocus | Qt.WindowType.NoDropShadowWindowHint)
        super().__init__(parent, flags)
        self.config = config
        self.setObjectName("SamIsland")
        self.setWindowTitle("SAM")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tr("island.click_hint"))

        self.state = "idle"
        self.engine = ""
        self.muted = False
        self.caption: CaptionLine | None = None
        self._hints = IslandHints()      # quota / listening notices (island_hints.py)
        self._families = theme.ui_families(self._cfg("ui.font_family"))
        self._fonts()
        self._level = 0.0
        self._targets = {"mic": 0.0, "speaker": 0.0}
        self._phase = 0.0
        self._progress: dict[str, Any] | None = None
        self._progress_shown = 0.0
        self._hover = False
        self._expansion = 0.0
        self._press: QPointF | None = None
        self._press_anchor: QPointF | None = None
        self._dragging = False
        self._win_styled = False
        self.frames = 0              # painted frames (on-screen fps check)
        self.paint_ms = 0.0          # total time spent in paintEvent (on-screen CPU check)
        self._painted_at = -1000     # clock ms of the last paint (60 fps cap across all repaint sources)
        self._bg: QPixmap | None = None
        self._bg_key: tuple[Any, ...] | None = None
        self._font_version = 0

        self._clock = QElapsedTimer()
        self._clock.start()
        self._last_ms = 0
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(FRAME_MS)
        self._timer.timeout.connect(self._tick)
        self._collapse = QTimer(self)
        self._collapse.setSingleShot(True)
        self._collapse.timeout.connect(self.clear_caption)
        self._click = QTimer(self)
        self._click.setSingleShot(True)
        self._click.setInterval(min(QApplication.doubleClickInterval() if QApplication.instance() else 300, 300))
        self._click.timeout.connect(self._clicked)
        self._anim = QPropertyAnimation(self, b"expansion", self)
        self._anim.setDuration(260)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._stale = QTimer(self)
        self._stale.setSingleShot(True)
        self._stale.setInterval(PROGRESS_STALE_MS)
        self._stale.timeout.connect(self.clear_progress)

        self.card = ConfirmCard(self._families, self)
        self.card.setFixedWidth(theme.CARD_WIDTH)
        self.card.answered.connect(self._on_card_answer)
        self._anchor = self._restore_anchor()
        self._relayout()

    # -- configuration helpers ------------------------------------------------------------------
    def _cfg(self, key: str, default: Any = None) -> Any:
        if self.config is None:
            return default
        try:
            return self.config.get(key, default)
        except Exception:  # noqa: BLE001 - settings are optional for the island
            return default

    def _fonts(self) -> None:
        self.f_name = theme.latin_font(15, 650)
        self.f_name.setLetterSpacing(self.f_name.SpacingType.AbsoluteSpacing, 1.4)
        self.f_status = theme.ui_font(12.5, 500, self._cfg("ui.font_family"))
        self.f_caption = theme.ui_font(13.5, 400, self._cfg("ui.font_family"))
        self.f_badge = theme.latin_font(8.5, 700)
        self.f_icon = theme.icon_font(12)

    # -- public state API (also used by tests) -------------------------------------------------------
    def status_text(self) -> str:
        override = self._hints.status_override(self.state)
        if override:
            return override
        return state_word("muted" if self.muted and self.state in ("idle", "sleeping") else self.state)

    def state_color(self) -> str:
        return theme.state_color("muted" if self.muted and self.state in ("idle", "sleeping") else self.state)

    def set_state(self, state: str, engine: str | None = None) -> None:
        state = state if state in theme.STATE_COLORS else "idle"
        if state == "muted":
            self.muted = True
        elif state == "listening":
            self.muted = False
        self.state = state
        if engine is not None:
            self.engine = engine
        self._ensure_timer()
        self.update()

    def set_caption(self, text: str, tone: str = "assistant", final: bool = True) -> None:
        text = " ".join((text or "").split())
        if not text:
            self.clear_caption()
            return
        self.caption = CaptionLine(text, tone if tone in TONE_COLORS else "assistant", final)
        self._animate_to(1.0)
        self._collapse.start(CAPTION_HOLD_FINAL_MS if final else CAPTION_HOLD_PARTIAL_MS)
        self.update()

    def clear_caption(self) -> None:
        self.caption = None
        self._collapse.stop()
        self._animate_to(0.0)
        self.update()

    def displayed_caption(self) -> str:
        """The caption exactly as painted (elided to the expanded width)."""
        if self.caption is None:
            return ""
        rect = self._caption_rect(self._pill_rect(theme.PILL_EXPANDED[0], theme.PILL_EXPANDED[1]))
        return elide(self.caption.text, QFontMetricsF(self.f_caption), rect.width(),
                     keep_tail=not self.caption.final)

    def set_progress(self, step: int, max_steps: int, done: bool = False, ok: bool | None = None,
                     task_id: str = "") -> None:
        now = time.monotonic()
        self._progress = {"step": step, "max": max_steps, "done": done, "ok": ok,
                          "done_at": now if done else None, "task_id": task_id}
        self._stale.start()
        self._ensure_timer()
        self.update()

    def clear_progress(self) -> None:
        self._progress = None
        self._progress_shown = 0.0
        self._stale.stop()
        self._ensure_timer()
        self.update()

    def _finish_progress(self, task_id: str, ok: bool) -> None:
        """A long tool ended (``ToolFinished``): close its progress line, which
        ``ToolContext.progress`` publishes with ``task_id == call_id``."""
        prog = self._progress
        if prog is not None and not prog["done"] and task_id and prog.get("task_id") == task_id:
            self.set_progress(prog["step"], prog["max"], True, ok, task_id)

    def set_level(self, source: str, level: float) -> None:
        self._targets[source if source in self._targets else "mic"] = max(0.0, min(1.0, float(level)))
        self._ensure_timer()

    @property
    def animating(self) -> bool:
        return self._timer.isActive()

    @property
    def expanded_target(self) -> bool:
        return (self._anim.endValue() or 0.0) > 0.5 if self._anim.state() else self._expansion > 0.5

    def _clicked(self) -> None:
        """A single click: listen (or, right after «کلیک بکە», re-open the owner's turn)."""
        self._hints.clear_hint()
        self.update()
        self.toggleListeningRequested.emit()

    # -- events from the core (GUI thread, via QtBridge) ----------------------------------------------
    def handle_event(self, ev: Any) -> None:
        if isinstance(ev, VoiceState):
            self.set_state(ev.state, ev.engine)
        elif isinstance(ev, LevelMeter):
            self.set_level(ev.source, ev.level)
        elif isinstance(ev, Caption):
            if ev.role == "user":
                self._hints.on_user_words()
            self.set_caption(ev.text, "user" if ev.role == "user" else
                             "system" if ev.role == "system" else "assistant", ev.final)
        elif is_notice(ev):
            shown = self._hints.on_notice(ev)
            if shown:
                self.set_caption(shown[0], shown[1], True)
        elif isinstance(ev, Transcript):
            self._hints.on_answer(ev.role, ev.text)
            if ev.role in ("user", "assistant") and ev.text:
                self.set_caption(ev.text, "user" if ev.role == "user" else "assistant", True)
        elif isinstance(ev, ToolStarted):
            self.set_caption(tool_label(ev.name) + "…", "system", False)
        elif isinstance(ev, ToolFinished):
            self._finish_progress(ev.call_id, ev.ok)
            if ev.ok:
                self.set_caption(tool_label(ev.name), "success", True)
            else:
                # Tool summaries are often English (the model rephrases them);
                # the island only shows Sorani-script summaries as they are.
                summary = ev.summary or ""
                text = summary if is_arabic_script(summary) else tr("island.tool_failed",
                                                                     label=tool_label(ev.name))
                self.set_caption(text, "danger", True)
        elif isinstance(ev, WorkerProgress):
            self.set_progress(ev.step, ev.max_steps, ev.done, ev.ok, ev.task_id)
            if ev.text_ckb:
                self.set_caption(ev.text_ckb, "success" if ev.done and ev.ok else
                                 "danger" if ev.done and ev.ok is False else "system", ev.done)
        elif isinstance(ev, Error):
            self.set_caption(ev.message_ckb, "danger", True)
        elif isinstance(ev, Alert):
            self.set_caption(ev.text_ckb, "alert", True)
        elif isinstance(ev, ConfirmRequest):
            self.card.push(ev.confirm_id, ev.question_ckb, ev.detail, ev.tool_name, ev.expires_at)
            self._relayout()
            self._ensure_timer()
        elif isinstance(ev, ConfirmResult):
            self.card.remove(ev.confirm_id)
            self._relayout()
        elif isinstance(ev, SettingsChanged) and ev.key == "ui.font_family":
            self._families = theme.ui_families(ev.value)
            self._fonts()
            self._font_version += 1
            self.update()

    def _on_card_answer(self, confirm_id: str, approved: bool, via: str) -> None:
        self._relayout()
        self.confirmAnswered.emit(confirm_id, approved, via)

    # -- geometry --------------------------------------------------------------------------------------
    def _pill_size(self) -> tuple[float, float]:
        (cw, ch), (ew, eh) = theme.PILL_COMPACT, theme.PILL_EXPANDED
        t = max(0.0, min(1.0, self._expansion))
        return cw + (ew - cw) * t, ch + (eh - ch) * t

    def _card_height(self) -> int:
        if not self.card.isVisibleTo(self):
            return 0
        lay = self.card.layout()
        return max(lay.totalHeightForWidth(theme.CARD_WIDTH), lay.totalMinimumSize().height())

    def _pill_rect(self, w: float | None = None, h: float | None = None) -> QRectF:
        if w is None or h is None:
            w, h = self._pill_size()
        ml, mt, _mr, _mb = theme.SHADOW_MARGIN
        content_w = self.width() - ml - _mr
        return QRectF(ml + (content_w - w) / 2.0, mt, w, h)

    def _window_size(self) -> tuple[int, int]:
        w, h = self._pill_size()
        ml, mt, mr, mb = theme.SHADOW_MARGIN
        card_h = self._card_height()
        # The window hugs the pill (it grows with it): transparent margins
        # would otherwise swallow clicks meant for the app underneath.
        content_w = max(w, theme.CARD_WIDTH if card_h else 0)
        height = mt + h + (theme.CARD_GAP + card_h if card_h else 0) + mb
        return int(math.ceil(content_w + ml + mr)), int(math.ceil(height))

    def _relayout(self) -> None:
        """Resize/move the window around the anchor (pill top-centre)."""
        width, height = self._window_size()
        cx, top = self._anchor.x(), self._anchor.y()
        ml, mt, mr, _mb = theme.SHADOW_MARGIN
        rect = QRect(round(cx - width / 2.0), round(top - mt), width, height)
        if self.geometry() != rect:
            self.setGeometry(rect)
        if self.card.isVisibleTo(self):
            _w, h = self._pill_size()
            card_h = self._card_height()
            self.card.setGeometry(round((width - theme.CARD_WIDTH) / 2.0), round(mt + h + theme.CARD_GAP),
                                  theme.CARD_WIDTH, card_h)
        self.update()

    def _screen_rect(self, point: QPointF | None = None) -> QRect:
        screen = None
        if point is not None:
            screen = QGuiApplication.screenAt(point.toPoint())
        screen = screen or QGuiApplication.primaryScreen()
        return screen.availableGeometry() if screen else QRect(0, 0, 1440, 900)

    def default_anchor(self) -> QPointF:
        area = self._screen_rect()
        return QPointF(area.center().x() + 0.5, area.top() + 10)

    def _clamp(self, anchor: QPointF) -> QPointF:
        area = self._screen_rect(anchor)
        half = theme.PILL_EXPANDED[0] / 2.0 + 4
        x = min(max(anchor.x(), area.left() + half), area.right() - half)
        y = min(max(anchor.y(), area.top() + 2), area.bottom() - theme.PILL_EXPANDED[1] - 8)
        return QPointF(x, y)

    def _restore_anchor(self) -> QPointF:
        pos = self._cfg("ui.island_pos")
        if isinstance(pos, (list, tuple)) and len(pos) == 2 and all(isinstance(v, (int, float)) for v in pos):
            point = QPointF(float(pos[0]), float(pos[1]))
            if QGuiApplication.screenAt(point.toPoint() + QPoint(0, 10)) is not None:
                return self._clamp(point)
        return self.default_anchor()

    def anchor(self) -> QPointF:
        return QPointF(self._anchor)

    def move_anchor(self, anchor: QPointF, save: bool = False) -> None:
        self._anchor = self._clamp(anchor)
        self._relayout()
        if save and self.config is not None:
            try:
                # [centre_x, top_y] of the pill in logical desktop px (setting ui.island_pos).
                self.config.set("ui.island_pos", [round(self._anchor.x()), round(self._anchor.y())])
            except Exception:  # noqa: BLE001
                pass

    def reset_position(self) -> None:
        self._anchor = self.default_anchor()
        self._relayout()
        if self.config is not None:
            try:
                self.config.set("ui.island_pos", None)
            except Exception:  # noqa: BLE001
                pass

    # -- expansion animation --------------------------------------------------------------------------------
    def _get_expansion(self) -> float:
        return self._expansion

    def _set_expansion(self, value: float) -> None:
        self._expansion = float(value)
        self._relayout()

    expansion = Property(float, _get_expansion, _set_expansion)

    def _animate_to(self, target: float) -> None:
        current_end = self._anim.endValue() if self._anim.state() == QPropertyAnimation.State.Running else None
        if current_end == target or (current_end is None and abs(self._expansion - target) < 1e-3):
            return
        self._anim.stop()
        self._anim.setStartValue(self._expansion)
        self._anim.setEndValue(float(target))
        self._anim.start()

    # -- animation clock ----------------------------------------------------------------------------------------
    def _needs_frames(self) -> bool:
        if self.state in theme.ANIMATED_STATES:
            return True
        if self._level > 0.01 or max(self._targets.values()) > 0.01:
            return True
        prog = self._progress
        if prog is not None:
            if prog["done"] or not prog["max"]:
                return True             # fading out / indeterminate sweep
            # A determinate line only needs frames while it glides to its new
            # value; a worker waiting on a slow step must not cost 60 fps.
            return abs(prog["step"] / prog["max"] - self._progress_shown) > 0.002
        return False

    def frame_interval(self) -> int:
        """60 fps while a voice level is audible, 30 fps for slow motion."""
        if self._level > LIVELY_LEVEL or max(self._targets.values()) > LIVELY_LEVEL:
            return FRAME_MS
        return SLOW_FRAME_MS

    def _ensure_timer(self) -> None:
        if self._needs_frames():
            interval = self.frame_interval()
            if self._timer.interval() != interval:
                self._timer.setInterval(interval)
            if not self._timer.isActive():
                self._last_ms = self._clock.elapsed()
                self._timer.start()
        elif self._timer.isActive():
            self._timer.stop()

    def _tick(self) -> None:
        now_ms = self._clock.elapsed()
        dt = max(0.001, min(0.1, (now_ms - self._last_ms) / 1000.0))
        self._last_ms = now_ms
        self._phase += dt
        # Levels: events arrive at <= 25 Hz; decay targets with a 150 ms half-life so
        # the orb settles when the meter stops, and smooth with fast attack/slow release.
        for key in self._targets:
            self._targets[key] *= 0.5 ** (dt / 0.15)
        source = "speaker" if self.state == "speaking" else "mic" if self.state == "listening" else None
        target = self._targets[source] if source else max(self._targets.values()) * 0.6
        rate = 22.0 if target > self._level else 7.0
        self._level += (target - self._level) * min(1.0, dt * rate)
        if self._progress is not None:
            self._tick_progress(dt)
        # 60 fps cap for every repaint source together: the caption's expansion
        # animation and geometry changes repaint on their own timers, and adding
        # this frame on top measured 67 paints/s on screen (acceptance/ui_onscreen.py).
        if now_ms - self._painted_at >= FRAME_MS - 1:
            self.update(self._dynamic_region())
        self._ensure_timer()

    def _tick_progress(self, dt: float) -> None:
        prog = self._progress
        assert prog is not None
        goal = 1.0 if prog["done"] else (prog["step"] / prog["max"] if prog["max"] else 0.0)
        self._progress_shown += (goal - self._progress_shown) * min(1.0, dt * 8.0)
        if abs(goal - self._progress_shown) < 0.002:
            self._progress_shown = goal
        if prog["done"] and prog["done_at"] is not None and time.monotonic() - prog["done_at"] > PROGRESS_DONE_HOLD_S:
            self.clear_progress()      # the line leaves the dynamic rect: repaint everything once

    # -- painting ---------------------------------------------------------------------------------------------
    def paintEvent(self, event: Any) -> None:  # noqa: N802
        # Two layers. The static one (shadow, glass, texts, caption, card glass)
        # is cached in a pixmap and re-rendered only when what it shows changes;
        # each animation frame only blits it and redraws the orb, the meter and
        # the progress line inside the dirty rect. Measured on this PC (175 %
        # scaling, 60 fps listening): painting everything every frame cost ~80 %
        # of a core; see acceptance/ui_onscreen.py for the current figure.
        began = time.perf_counter()
        self.frames += 1
        self._painted_at = self._clock.elapsed()
        p = QPainter(self)
        p.setClipRegion(event.region())
        p.drawPixmap(0, 0, self._background(device_scale(p, self)))
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pill = self._pill_rect()
        # The orb's glow and ripples stay inside the glass: a halo spilling past
        # the rounded edge looked like a smudge on bright wallpapers (render check).
        clip = QPainterPath()
        clip.addRoundedRect(pill, theme.PILL_RADIUS, theme.PILL_RADIUS)
        p.setClipPath(clip, Qt.ClipOperation.IntersectClip)
        paint_orb(p, self._orb_center(pill), 17.0, theme.qcolor(self.state_color()),
                  state="muted" if self.muted and self.state in ("idle", "sleeping") else self.state,
                  level=self._level, phase=self._phase)
        paint_meter(p, pill, self.state_color(), state=self.state, muted=self._meter_muted(), level=self._level,
                    phase=self._phase, icon_font=self.f_icon)
        if self._progress is not None:
            paint_progress(p, pill, self._progress, self._progress_shown, self._phase)
        p.end()
        self.paint_ms += (time.perf_counter() - began) * 1000.0

    def _meter_muted(self) -> bool:
        return self.muted and self.state in ("idle", "sleeping", "muted")

    @staticmethod
    def _orb_center(pill: QRectF) -> QPointF:
        return QPointF(pill.right() - 29.0, pill.top() + 29.0)

    def _dynamic_region(self) -> QRegion:
        """Areas that change between animation frames: orb + ripples, meter,
        progress line. Kept as separate rectangles (not their bounding box,
        which spanned the whole pill) so each frame repaints ~40 % of the
        pixels: the paint clip follows the region."""
        pill = self._pill_rect()
        c = self._orb_center(pill)
        region = QRegion(QRectF(c.x() - 40, c.y() - 40, 80, 80).toAlignedRect().adjusted(-2, -2, 2, 2))
        region += QRegion(QRectF(pill.left() + 14, pill.top() + 10, 48, 38).toAlignedRect().adjusted(-2, -2, 2, 2))
        if self._progress is not None:
            region += QRegion(QRectF(pill.left() + 24, pill.bottom() - 8, pill.width() - 48, 8)
                              .toAlignedRect().adjusted(-2, -2, 2, 2))
        return region

    def _background_key(self, dpr: float) -> tuple[Any, ...]:
        line = self.caption
        return (self.width(), self.height(), dpr, round(self._expansion, 4),
                self.state_color(), self.status_text(), self.engine, self._hover,
                (line.text, line.tone, line.final) if line else None,
                self.card.geometry().getRect() if self.card.isVisible() else None, self._font_version)

    def _background(self, dpr: float | None = None) -> QPixmap:
        """The static layer, cached at the painter's device scale (a monitor
        move or a 2x ``render()`` rebuilds it instead of upscaling a 1x copy)."""
        dpr = dpr or self.devicePixelRatioF()
        key = self._background_key(dpr)
        if key != self._bg_key or self._bg is None:
            pix = QPixmap(max(1, round(self.width() * dpr)), max(1, round(self.height() * dpr)))
            pix.setDevicePixelRatio(dpr)
            pix.fill(Qt.GlobalColor.transparent)
            p = QPainter(pix)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            pill = self._pill_rect()
            paint_glass(p, pill, radius=theme.PILL_RADIUS, tint_color=self.state_color(), hover=self._hover)
            self._paint_identity(p, pill, theme.qcolor(self.state_color()))
            reveal = max(0.0, min(1.0, (pill.height() - theme.PILL_COMPACT[1]) /
                                       (theme.PILL_EXPANDED[1] - theme.PILL_COMPACT[1])))
            if reveal > 0.02:
                self._paint_caption(p, pill, reveal)
            if self.card.isVisible():
                paint_glass(p, QRectF(self.card.geometry()), radius=18.0)
            p.end()
            self._bg, self._bg_key = pix, key
        return self._bg

    def _paint_identity(self, p: QPainter, pill: QRectF, color: QColor) -> None:
        right = pill.right() - 58.0
        p.setFont(self.f_name)
        name_rect = QRectF(pill.left() + 90, pill.top() + 8, right - pill.left() - 90, 22)
        p.setPen(QColor(theme.TEXT))
        p.drawText(name_rect, int(A_RIGHT), "SAM")
        if self.engine == "live":
            name_w = QFontMetricsF(self.f_name).horizontalAdvance("SAM")
            p.setFont(self.f_badge)
            badge_w = QFontMetricsF(self.f_badge).horizontalAdvance("LIVE") + 10
            badge = QRectF(right - name_w - 8 - badge_w, pill.top() + 12.5, badge_w, 14)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(theme.qcolor(self.state_color(), 0.18))
            p.drawRoundedRect(badge, 7, 7)
            p.setPen(theme.qcolor(self.state_color()))
            p.drawText(badge, int(Qt.AlignmentFlag.AlignCenter), "LIVE")
        p.setFont(self.f_status)
        status_rect = QRectF(pill.left() + 90, pill.top() + 30, right - pill.left() - 90, 20)
        option = QTextOption(A_RIGHT)
        option.setTextDirection(Qt.LayoutDirection.RightToLeft)
        p.setPen(color.lighter(118))
        p.drawText(status_rect, self.status_text(), option)

    def _caption_rect(self, pill: QRectF) -> QRectF:
        return QRectF(pill.left() + 24.0, pill.top() + 58.0, pill.width() - 48.0 - 22.0, 24.0)

    def _paint_caption(self, p: QPainter, pill: QRectF, reveal: float) -> None:
        p.save()
        clip = QPainterPath()
        clip.addRoundedRect(pill, theme.PILL_RADIUS, theme.PILL_RADIUS)
        p.setClipPath(clip)
        p.setOpacity(reveal)
        divider = QLinearGradient(pill.left() + 20, 0, pill.right() - 20, 0)
        divider.setColorAt(0.0, QColor(255, 255, 255, 0))
        divider.setColorAt(0.5, QColor(255, 255, 255, 20))
        divider.setColorAt(1.0, QColor(255, 255, 255, 0))
        p.setPen(QPen(QBrush(divider), 1.0))
        p.drawLine(QPointF(pill.left() + 20, pill.top() + 57.5), QPointF(pill.right() - 20, pill.top() + 57.5))
        line = self.caption
        if line is not None:
            rect = self._caption_rect(pill)
            tone_color = theme.qcolor(TONE_COLORS.get(line.tone, TONE_COLORS["assistant"]))
            icon = TONE_ICONS.get(line.tone)
            if icon:
                p.setFont(self.f_icon)
                p.setPen(tone_color)
                p.drawText(QRectF(rect.right() + 4, rect.top(), 18, rect.height()),
                           int(Qt.AlignmentFlag.AlignCenter), theme.ICONS[icon])
            p.setFont(self.f_caption)
            text = elide(line.text, QFontMetricsF(self.f_caption), rect.width(), keep_tail=not line.final)
            rtl = is_rtl(text)
            option = QTextOption(A_RIGHT)
            option.setTextDirection(Qt.LayoutDirection.RightToLeft if rtl else Qt.LayoutDirection.LeftToRight)
            option.setWrapMode(QTextOption.WrapMode.NoWrap)
            p.setPen(tone_color)
            p.drawText(rect, text, option)
        p.restore()

    # -- mouse ----------------------------------------------------------------------------------------------------
    def _in_pill(self, pos: QPointF) -> bool:
        return self._pill_rect().contains(pos)

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._in_pill(event.position()):
            self._press = event.globalPosition()
            self._press_anchor = QPointF(self._anchor)
            self._dragging = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802
        if self._press is not None and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.globalPosition() - self._press
            if self._dragging or delta.manhattanLength() > QApplication.startDragDistance():
                self._dragging = True
                self._click.stop()
                assert self._press_anchor is not None
                self.move_anchor(self._press_anchor + delta)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._press is not None:
            if self._dragging:
                self.move_anchor(self._anchor, save=True)
            elif self._in_pill(event.position()):
                # Wait briefly for a possible double-click (<= 300 ms keeps the click responsive).
                self._click.start()
            self._press = None
            self._dragging = False
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: Any) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._in_pill(event.position()):
            self._click.stop()
            self.openPanelRequested.emit()

    def contextMenuEvent(self, event: Any) -> None:  # noqa: N802
        self._click.stop()
        menu = self.build_menu()
        menu.exec(event.globalPos())

    def build_menu(self) -> QMenu:
        menu = QMenu(self)
        menu.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        menu.setStyleSheet(theme.menu_stylesheet(self._families))
        menu.addAction(tr("menu.open_panel"), self.openPanelRequested.emit)
        mute = menu.addAction(tr("menu.mute"))
        mute.setCheckable(True)
        mute.setChecked(self.muted)
        mute.triggered.connect(lambda checked: self._request_mute(bool(checked)))
        menu.addAction(tr("menu.stop_all"), self.stopAllRequested.emit)
        menu.addSeparator()
        menu.addAction(tr("menu.reset_position"), self.reset_position)
        menu.addAction(tr("menu.settings"), self.settingsRequested.emit)
        menu.addSeparator()
        menu.addAction(tr("menu.quit"), self.quitRequested.emit)
        return menu

    def _request_mute(self, muted: bool) -> None:
        self.muted = muted        # optimistic; the voice engine's VoiceState confirms
        self.update()
        self.muteRequested.emit(muted)

    def enterEvent(self, event: Any) -> None:  # noqa: N802
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event: Any) -> None:  # noqa: N802
        self._hover = False
        self.update()
        super().leaveEvent(event)

    # -- window styles -------------------------------------------------------------------------------------------------
    def showEvent(self, event: Any) -> None:  # noqa: N802
        super().showEvent(event)
        if not self._win_styled:
            self._win_styled = True
            apply_no_activate(self)


__all__ = ["Island", "CaptionLine", "apply_no_activate"]
