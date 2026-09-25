"""گفتوگۆ — chat bubbles with streaming text and a text input.

Typing works exactly like speaking: text goes to the open Live session when
there is one (``app.voice.send_text``, duck-typed, optional) and otherwise to
``app.submit_text`` (-> ``conversation.handle_text``), per docs/CONTRACTS.md.

Bubbles come from events, not from the reply value, so voice and typed turns
look the same: ``Caption`` (partial/final) opens or updates the live bubble of
that role, ``Transcript`` finalises it (or adds one); typed user text is shown
at once and de-duplicated when its ``Transcript`` arrives.
"""

from __future__ import annotations

import html
import time
from typing import Any

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QFontMetricsF, QPainter, QTextOption
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QSizePolicy,
                               QVBoxLayout, QWidget)

from ...events import Caption, ToolFinished, ToolStarted, Transcript, VoiceState
from ...textnorm import normalize_ckb
from .. import theme
from ..orb import orb_pixmap
from ..strings import ms_text, tool_label, tr
from ..widgets import A_LEFT, A_RIGHT, IconLabel, clock_text, is_rtl
from . import SCROLL_GUTTER, Page, scroll_area

LIVE_STATES = ("listening", "thinking", "speaking", "working")


def live_session_open(voice: Any) -> bool:
    """Same test as the brain's ``Conversation._live_forwarder``."""
    if voice is None or not callable(getattr(voice, "send_text", None)):
        return False
    flag = getattr(voice, "live_session_open", None)
    if flag is not None:
        return bool(flag)
    return getattr(voice, "engine_name", "") == "live" and getattr(voice, "state", "") in LIVE_STATES


class Bubble(QFrame):
    """One message. ``streaming`` shows typing dots until text arrives."""

    def __init__(self, role: str, text: str = "", meta: str = "", streaming: bool = False,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.role = role
        self.streaming = streaming
        self._text = ""
        self._max_width = 520
        self._text_font = theme.ui_font(14)
        self._meta_font = theme.ui_font(12)
        user = role == "user"
        bg = theme.ACCENT_SOFT if user else theme.SURFACE_2
        border = "#343C78" if user else theme.BORDER_SOFT
        # RTL: the user's bubble sits on the right with its "tail" corner at the top right.
        tail = "border-top-right-radius: 6px;" if user else "border-top-left-radius: 6px;"
        self.setObjectName("Bubble")
        self.setStyleSheet(f"#Bubble {{ background: {bg}; border: 1px solid {border}; border-radius: 16px; {tail} }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 8)
        lay.setSpacing(4)
        self.label = QLabel()
        self.label.setWordWrap(True)
        self.label.setTextFormat(Qt.TextFormat.RichText)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.dots = TypingDots()
        self.meta = QLabel(meta)
        self.meta.setObjectName("Faint")
        self.meta.setVisible(bool(meta))      # a live (partial) bubble has no meta line yet
        lay.addWidget(self.label)
        lay.addWidget(self.dots)
        lay.addWidget(self.meta)
        self.set_text(text)

    def text(self) -> str:
        return self._text

    def set_text(self, text: str) -> None:
        self._text = text or ""
        rtl = is_rtl(self._text)
        # Qt rich text honours dir= per paragraph; line-height makes Sorani easier to read.
        body = html.escape(self._text).replace("\n", "<br>")
        self.label.setText(f'<div dir="{"rtl" if rtl else "ltr"}" style="line-height:145%; font-size:14px;">'
                           f"{body}</div>")
        self.label.setAlignment(A_RIGHT if rtl else A_LEFT)
        self.label.setVisible(bool(self._text))
        self.dots.set_running(self.streaming and not self._text)
        self.fit()

    def fit(self, max_width: int | None = None) -> None:
        """Shrink-wrap: as wide as the longest line (a word-wrapped QLabel alone
        would wrap far too early), never wider than ``max_width``."""
        if max_width is not None:
            self._max_width = max_width
        fm = QFontMetricsF(self._text_font)
        natural = max((fm.horizontalAdvance(line) for line in self._text.split("\n")), default=0.0)
        meta = QFontMetricsF(self._meta_font).horizontalAdvance(self.meta.text()) if hasattr(self, "meta") else 0.0
        content = max(natural * 1.03 + 6, meta + 6, 56.0 if self.streaming else 40.0)
        width = int(min(self._max_width, content + 30))
        self.setFixedWidth(width)
        # A word-wrapped QLabel reports its size hint wrapped at a narrow default
        # width (measured: a 111 px hint for a 56 px two-line bubble), which
        # inflated the list's minimum height: a scroll bar and a blank gap
        # appeared after only three short exchanges. Pin the exact height for
        # the width the label really gets (bubble minus its 14 + 14 px margins).
        # The old fixed height is released first: QLabel.heightForWidth()
        # returns at least the label's minimum height, so a height pinned while
        # the panel was narrower would otherwise stick (a 2-line bubble kept a
        # 3-line box, seen in the render check).
        if self._text:
            self.label.setMinimumHeight(0)
            self.label.setMaximumHeight(16777215)          # QWIDGETSIZE_MAX
            height = self.label.heightForWidth(max(10, width - 28))
            if height > 0:
                self.label.setFixedHeight(height)

    def set_streaming(self, streaming: bool) -> None:
        self.streaming = streaming
        self.dots.set_running(streaming and not self._text)

    def set_meta(self, meta: str) -> None:
        self.meta.setText(meta)
        self.meta.setVisible(bool(meta))
        self.fit()


class TypingDots(QWidget):
    """Three pulsing dots; the timer runs only while visible and streaming."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(46, 18)
        self._t = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(90)
        self._timer.timeout.connect(self._step)
        self.hide()

    def set_running(self, running: bool) -> None:
        self.setVisible(running)
        if running:
            self._timer.start()
        else:
            self._timer.stop()

    def _step(self) -> None:
        self._t += 0.09
        self.update()

    def paintEvent(self, _event: Any) -> None:  # noqa: N802
        import math

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(3):
            a = 0.35 + 0.65 * max(0.0, math.sin(self._t * 6.0 - i * 0.8))
            p.setBrush(theme.qcolor(theme.TEXT_MUTED, a))
            p.drawEllipse(QPointF(self.width() - 8 - i * 12, 9), 3.2, 3.2)
        p.end()


class ToolChip(QWidget):
    """Inline tool activity row: icon + Sorani label + result."""

    def __init__(self, name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.icon = IconLabel("bolt", 12, theme.WARNING)
        self.text = QLabel(tool_label(name) + "…")
        self.text.setObjectName("Faint")
        lay.addStretch(1)
        lay.addWidget(self.icon)
        lay.addWidget(self.text)
        lay.addStretch(1)
        self.name = name

    def finish(self, ok: bool, duration_ms: float, summary: str) -> None:
        self.icon.set_icon("check" if ok else "warning", theme.SUCCESS if ok else theme.DANGER)
        self.text.setText(f"{tool_label(self.name)} · {ms_text(duration_ms)}")
        self.setToolTip(summary)


class ChatInput(QPlainTextEdit):
    """Enter sends, Shift+Enter adds a line; grows from 1 to 5 lines.

    Transparent: it sits inside the rounded ``InputShell``. The placeholder is
    painted here because QPlainTextEdit draws its own placeholder left-aligned
    even in a right-to-left editor (seen in the offscreen render check)."""

    submitted = Signal()
    focusChanged = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._placeholder = tr("chat.placeholder")
        self.setToolTip(tr("chat.input_hint"))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setTabChangesFocus(True)
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.viewport().setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        option = self.document().defaultTextOption()
        option.setTextDirection(Qt.LayoutDirection.RightToLeft)
        option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.document().setDefaultTextOption(option)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.document().contentsChanged.connect(self._fit)
        self.setStyleSheet("QPlainTextEdit { background: transparent; border: none; padding: 0 4px; font-size: 14px; }")
        self._fit()

    def placeholder(self) -> str:
        return self._placeholder

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        super().paintEvent(event)
        if self.document().isEmpty():
            p = QPainter(self.viewport())
            p.setPen(theme.qcolor(theme.TEXT_FAINT))
            p.setFont(self.font())
            margin = self.document().documentMargin()
            rect = QRectF(self.viewport().rect()).adjusted(margin, margin - 1, -margin - 2, -margin)
            p.drawText(rect, int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignAbsolute
                                 | Qt.AlignmentFlag.AlignTop), self._placeholder)
            p.end()

    def focusInEvent(self, event: Any) -> None:  # noqa: N802
        super().focusInEvent(event)
        self.focusChanged.emit(True)

    def focusOutEvent(self, event: Any) -> None:  # noqa: N802
        super().focusOutEvent(event)
        self.focusChanged.emit(False)

    def keyPressEvent(self, event: Any) -> None:  # noqa: N802
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.submitted.emit()
            return
        super().keyPressEvent(event)

    def _fit(self) -> None:
        lines = max(1, int(self.document().size().height()))
        line_h = self.fontMetrics().lineSpacing()
        self.setFixedHeight(int(min(5, lines) * line_h + 2 * self.document().documentMargin() + 6))
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded if lines > 5
                                        else Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(400, 36)


class SendButton(QPushButton):
    """Round accent button; the paper-plane glyph is mirrored for RTL."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("SendButton")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tr("chat.send"))
        self._font = theme.icon_font(15)

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        super().paintEvent(event)   # background from the style sheet
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setFont(self._font)
        p.setPen(theme.qcolor("#0B0E14" if self.isEnabled() else "#8C93B8"))
        p.translate(self.width(), 0)
        p.scale(-1, 1)
        p.drawText(QRectF(0, 0, self.width(), self.height()), int(Qt.AlignmentFlag.AlignCenter), theme.ICONS["send"])
        p.end()


class ChatPage(Page):
    key = "chat"

    def __init__(self, app: Any, bridge: Any, parent: QWidget | None = None) -> None:
        super().__init__(app, bridge, parent)
        self.add_header()
        self._live: dict[str, Bubble | None] = {"user": None, "assistant": None}
        self._pending_user: list[str] = []
        self._awaiting_reply = False
        self._chips: dict[str, ToolChip] = {}
        self._voice_state = "idle"
        self._history_loaded = False
        self.bubbles: list[Bubble] = []

        self.list_widget = QWidget()
        self.list_lay = QVBoxLayout(self.list_widget)
        # Qt layout margins are physical (not mirrored in RTL) and the RTL scroll
        # bar sits on the LEFT, so the gutter goes on the left.
        self.list_lay.setContentsMargins(SCROLL_GUTTER, 4, 4, 4)
        self.list_lay.setSpacing(10)
        self.list_lay.addStretch(1)
        self.scroll = scroll_area(self.list_widget)
        self.scroll.verticalScrollBar().rangeChanged.connect(self._stick_to_bottom)
        self._stick = True
        self.scroll.verticalScrollBar().valueChanged.connect(self._on_scroll)

        self.empty = self._build_empty()
        self.root.addWidget(self.empty, 1)
        self.root.addWidget(self.scroll, 1)
        self.scroll.hide()

        self.shell = QFrame()
        self.shell.setObjectName("InputShell")
        bar = QHBoxLayout(self.shell)
        bar.setContentsMargins(14, 5, 6, 5)
        bar.setSpacing(6)
        self.input = ChatInput()
        self.input.submitted.connect(self.send)
        self.input.focusChanged.connect(self._shell_focus)
        self._shell_focus(False)
        self.mic = QPushButton(theme.ICONS["mic"])
        self.mic.setObjectName("IconButton")
        self.mic.setToolTip(tr("chat.mic"))
        self.mic.setCursor(Qt.CursorShape.PointingHandCursor)
        self.mic.clicked.connect(self._toggle_mic)
        self.send_button = SendButton()
        self.send_button.clicked.connect(lambda: self.send())
        bar.addWidget(self.input, 1)
        bar.addWidget(self.mic, 0, Qt.AlignmentFlag.AlignBottom)
        bar.addWidget(self.send_button, 0, Qt.AlignmentFlag.AlignBottom)
        self.root.addWidget(self.shell)

    def _shell_focus(self, focused: bool) -> None:
        border = theme.ACCENT if focused else theme.BORDER
        self.shell.setStyleSheet(f"#InputShell {{ background: {theme.SURFACE_2}; border: 1px solid {border};"
                                 " border-radius: 24px; }")

    # -- layout pieces ----------------------------------------------------------------------------------
    def _build_empty(self) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.addStretch(2)
        orb = QLabel()
        orb.setPixmap(orb_pixmap(72, state="idle", glow=True, dpr=2.0))
        orb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = QLabel(tr("chat.empty_title"))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        sub = QLabel(tr("chat.empty_sub"))
        sub.setObjectName("Muted")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(orb)
        lay.addSpacing(6)
        lay.addWidget(title)
        lay.addWidget(sub)
        lay.addSpacing(14)
        chips = QHBoxLayout()
        chips.setSpacing(8)
        chips.addStretch(1)
        for i in range(1, 5):
            text = tr(f"chat.suggest.{i}")
            chip = QPushButton(text)
            chip.setObjectName("Chip")
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            chip.clicked.connect(lambda _=False, t=text: self.send(t))
            chips.addWidget(chip)
        chips.addStretch(1)
        lay.addLayout(chips)
        lay.addStretch(3)
        return box

    def _show_list(self) -> None:
        if self.scroll.isHidden():
            self.empty.hide()
            self.scroll.show()

    def _row(self, widget: QWidget, role: str) -> QWidget:
        holder = QWidget()
        lay = QHBoxLayout(holder)
        lay.setContentsMargins(0, 0, 0, 0)
        # RTL layout: the first item is on the right -> user bubbles right, SAM's left.
        if role == "user":
            lay.addWidget(widget)
            lay.addStretch(1)
        else:
            lay.addStretch(1)
            lay.addWidget(widget)
        return holder

    def _append(self, widget: QWidget, role: str = "center", at_top: bool = False) -> None:
        self._show_list()
        row = widget if role == "center" else self._row(widget, role)
        if at_top:
            self.list_lay.insertWidget(1, row)
        else:
            self.list_lay.insertWidget(self.list_lay.count(), row)
        self._fit_widths()

    def _new_bubble(self, role: str, text: str = "", meta: str = "", streaming: bool = False,
                    at_top: bool = False) -> Bubble:
        bubble = Bubble(role, text, meta, streaming)
        if at_top:
            self.bubbles.insert(0, bubble)
        else:
            self.bubbles.append(bubble)
        self._append(bubble, "user" if role == "user" else "assistant", at_top)
        return bubble

    def _fit_widths(self) -> None:
        width = max(260, int(self.scroll.viewport().width() * 0.74))
        for bubble in self.bubbles:
            bubble.fit(width)

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._fit_widths()

    def _on_scroll(self, value: int) -> None:
        bar = self.scroll.verticalScrollBar()
        self._stick = value >= bar.maximum() - 40

    def _stick_to_bottom(self, _min: int, maximum: int) -> None:
        if self._stick:
            self.scroll.verticalScrollBar().setValue(maximum)

    @staticmethod
    def _meta(role: str, source: str, at: float | None = None) -> str:
        who = tr("chat.you") if role == "user" else tr("chat.worker") if source == "worker" else tr("chat.sam")
        how = tr("chat.typed") if source == "text" else tr("chat.voice") if source in ("live", "cascade") else ""
        parts = [who, clock_text(at or time.time())] + ([how] if how else [])
        return " · ".join(parts)

    # -- sending -------------------------------------------------------------------------------------------
    def send(self, text: str | None = None) -> None:
        value = (text if text is not None else self.input.toPlainText()).strip()
        if not value:
            return
        if text is None:
            self.input.clear()
        self._pending_user.append(normalize_ckb(value))
        self._new_bubble("user", value, self._meta("user", "text"))
        if getattr(self.app, "conversation", None) is None:
            # No brain loaded: an open Live session can still take the text directly.
            voice = getattr(self.app, "voice", None)
            if live_session_open(voice):
                self.bridge.call(voice.send_text(value), on_err=self._on_error)
            else:
                self._system_note(tr("chat.unavailable"))
            return
        # conversation.handle_text forwards to an open Live session itself
        # (and treats "بەڵێ"/"نەخێر" as the answer to a pending confirmation).
        self._awaiting_reply = True
        self._live["assistant"] = self._live["assistant"] or self._new_bubble("assistant", streaming=True)
        self.bridge.call(self.app.submit_text(value), on_ok=self._on_reply, on_err=self._on_error)

    def _on_reply(self, reply: Any) -> None:
        # Normally the Transcript event already rendered the answer; this is the fallback.
        if self._awaiting_reply:
            self._awaiting_reply = False
            text = str(reply or "").strip()
            live = self._live["assistant"]
            if text:
                if live is None:
                    live = self._new_bubble("assistant")
                live.set_text(text)
                live.set_streaming(False)
                live.set_meta(self._meta("assistant", "text"))
            elif live is not None and not live.text():
                self._drop(live)
            self._live["assistant"] = None

    def _on_error(self, _error: BaseException) -> None:
        self._awaiting_reply = False
        live = self._live["assistant"]
        if live is not None and not live.text():
            self._drop(live)
        self._live["assistant"] = None
        self._system_note(tr("chat.failed"))

    def _drop(self, bubble: Bubble) -> None:
        if bubble in self.bubbles:
            self.bubbles.remove(bubble)
        holder = bubble.parentWidget()
        if holder is not None and holder is not self.list_widget:
            holder.setParent(None)
            holder.deleteLater()

    def _system_note(self, text: str) -> None:
        note = QLabel(text)
        note.setObjectName("Faint")
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._append(note)

    def _toggle_mic(self) -> None:
        if getattr(self.app, "voice", None) is None:
            self._system_note(tr("island.voice_missing"))
            return
        self.bridge.call(self.app.toggle_listening())

    # -- events ------------------------------------------------------------------------------------------------
    def handle_event(self, ev: Any) -> None:
        if isinstance(ev, Caption) and ev.role in ("user", "assistant") and ev.text.strip():
            if (ev.role == "user" and self._live["user"] is None
                    and normalize_ckb(ev.text.strip()) in self._pending_user):
                return          # the typed text is already on screen
            bubble = self._live[ev.role]
            if bubble is None:
                bubble = self._new_bubble(ev.role, streaming=True)
                self._live[ev.role] = bubble
            bubble.set_text(ev.text.strip())
            bubble.set_streaming(not ev.final)
        elif isinstance(ev, Transcript) and ev.role in ("user", "assistant") and ev.text.strip():
            self._on_transcript(ev)
        elif isinstance(ev, ToolStarted) and ev.source != "ui":
            chip = ToolChip(ev.name)
            self._chips[ev.call_id] = chip
            self._append(chip)
        elif isinstance(ev, ToolFinished):
            chip = self._chips.pop(ev.call_id, None)
            if chip is not None:
                chip.finish(ev.ok, ev.duration_ms, ev.summary)
        elif isinstance(ev, VoiceState):
            self._voice_state = ev.state
            listening = ev.state == "listening"
            self.mic.setText(theme.ICONS["mic"])
            self.mic.setStyleSheet(f"QPushButton#IconButton {{ background: {theme.state_color('listening')}; "
                                   "color: #0B0E14; border: none; }" if listening else "")

    def _on_transcript(self, ev: Transcript) -> None:
        text = ev.text.strip()
        role = ev.role
        meta = self._meta(role, ev.source, ev.at)
        live = self._live[role]
        if role == "user":
            key = normalize_ckb(text)
            if live is None and key in self._pending_user:
                self._pending_user.remove(key)       # typed: already shown
                return
        if live is not None:
            live.set_text(text)
            live.set_streaming(False)
            live.set_meta(meta)
            self._live[role] = None
        else:
            self._new_bubble(role, text, meta)
        if role == "assistant":
            self._awaiting_reply = False

    # -- history -----------------------------------------------------------------------------------------------
    def on_shown(self) -> None:
        if self._history_loaded:
            return
        self._history_loaded = True
        memory = getattr(self.app, "memory", None)
        if memory is None or not hasattr(memory, "recent_turns"):
            return
        self.bridge.on_core(memory.recent_turns, limit=40, on_ok=self._load_history,
                            on_err=lambda _e: None)

    def _load_history(self, turns: Any) -> None:
        rows = [t for t in (turns or []) if isinstance(t, dict) and t.get("role") in ("user", "assistant")
                and str(t.get("text") or "").strip()]
        for turn in reversed(rows):          # insert newest-first at the top -> chronological order
            role = str(turn["role"])
            self._new_bubble(role, str(turn["text"]).strip(),
                             self._meta(role, str(turn.get("source") or ""), turn.get("at")), at_top=True)


__all__ = ["ChatPage", "Bubble", "ChatInput", "ToolChip"]
