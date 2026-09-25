"""Confirmation card shown under the island pill (``ConfirmRequest``).

The island paints the card's glass background; this widget holds the text,
the بەڵێ / نەخێر buttons and the countdown. A click emits ``answered``; the
controller forwards it to ``app.confirm.resolve(id, approved, "click")``
(thread-safe by contract). The broker's own 20 s timeout is authoritative; the
card also expires locally so a stalled core never leaves a stale card on screen
(default NO, like the broker).

Every card asks about a risky action (deletion, sending, a shell command), so
the SAFE answer carries the emphasis: «نەخێر» is the filled button and «بەڵێ»
an amber outline (repair review 2026-09-24: a filled «بەڵێ» next to a muted
«نەخێر» invites a non-expert to press yes).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from . import theme
from .strings import countdown_text, tr
from .widgets import IconLabel, direction_of, paint_bar


@dataclass
class PendingCard:
    confirm_id: str
    question: str
    detail: str
    tool_name: str
    expires_at: float
    created_at: float


class ConfirmCard(QWidget):
    answered = Signal(str, bool, str)      # confirm_id, approved, via ("click" | "timeout")

    def __init__(self, font_families: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self._queue: list[PendingCard] = []
        self.current: PendingCard | None = None
        fam = ", ".join(f'"{f}"' for f in font_families)
        self.setStyleSheet(f"""
            QWidget {{ font-family: {fam}; color: {theme.TEXT}; }}
            QLabel#CardQuestion {{ font-size: 14px; font-weight: 600; color: {theme.TEXT}; padding: 0 3px; }}
            QLabel#CardDetail {{ font-size: 12px; color: {theme.TEXT_MUTED}; padding: 0 3px; }}
            QLabel#CardTitle {{ font-size: 12px; color: {theme.WARNING}; font-weight: 600; }}
            QLabel#CardCount {{ font-size: 12px; color: {theme.TEXT_MUTED}; }}
            QPushButton {{ border-radius: 10px; padding: 7px 0; font-size: 13px; font-weight: 600; }}
            QPushButton#Yes {{ background: rgba(255,255,255,0.06); color: {theme.TEXT};
                               border: 1px solid {theme.WARNING}; }}
            QPushButton#Yes:hover {{ background: rgba(255,255,255,0.11); }}
            QPushButton#No {{ background: {theme.ACCENT}; color: #0B0E14; border: none; }}
            QPushButton#No:hover {{ background: {theme.ACCENT_HOVER}; }}
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 16)
        lay.setSpacing(8)

        head = QHBoxLayout()
        head.setSpacing(8)
        head.addWidget(IconLabel("shield", 14, theme.WARNING))
        title = QLabel(tr("confirm.title"))
        title.setObjectName("CardTitle")
        head.addWidget(title)
        head.addStretch(1)
        self.count_label = QLabel("")
        self.count_label.setObjectName("CardCount")
        head.addWidget(self.count_label)
        lay.addLayout(head)

        self.question = QLabel("")
        self.question.setObjectName("CardQuestion")
        self.question.setWordWrap(True)
        lay.addWidget(self.question)
        self.detail = QLabel("")
        self.detail.setObjectName("CardDetail")
        self.detail.setWordWrap(True)
        lay.addWidget(self.detail)

        buttons = QHBoxLayout()
        buttons.setSpacing(10)
        self.yes_button = QPushButton(tr("confirm.yes"))
        self.yes_button.setObjectName("Yes")
        self.no_button = QPushButton(tr("confirm.no"))
        self.no_button.setObjectName("No")
        for button in (self.yes_button, self.no_button):
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            buttons.addWidget(button, 1)
        lay.addSpacing(4)
        lay.addLayout(buttons)
        self.yes_button.clicked.connect(lambda: self._answer(True, "click"))
        self.no_button.clicked.connect(lambda: self._answer(False, "click"))

        self._tick = QTimer(self)
        self._tick.setInterval(50)
        self._tick.timeout.connect(self._on_tick)
        self.hide()

    # -- queue ----------------------------------------------------------------------------
    def push(self, confirm_id: str, question: str, detail: str = "", tool_name: str = "",
             expires_at: float = 0.0) -> None:
        """Show a request (the newest one is on top; older ones wait)."""
        now = time.time()
        item = PendingCard(confirm_id, question, detail, tool_name, expires_at or now + 20.0, now)
        self._queue = [q for q in self._queue if q.confirm_id != confirm_id] + [item]
        self._show(item)

    def remove(self, confirm_id: str) -> bool:
        """Drop a request (answered elsewhere, e.g. by voice or timeout)."""
        before = len(self._queue)
        self._queue = [q for q in self._queue if q.confirm_id != confirm_id]
        if self.current is not None and self.current.confirm_id == confirm_id:
            self.current = None
            if self._queue:
                self._show(self._queue[-1])
            else:
                self._hide_card()
        return len(self._queue) != before

    def clear(self) -> None:
        self._queue.clear()
        self.current = None
        self._hide_card()

    @property
    def pending_ids(self) -> list[str]:
        return [q.confirm_id for q in self._queue]

    def seconds_left(self) -> float:
        return max(0.0, self.current.expires_at - time.time()) if self.current else 0.0

    # -- internals ------------------------------------------------------------------------------
    def _show(self, item: PendingCard) -> None:
        self.current = item
        self.question.setText(item.question)
        self.question.setLayoutDirection(direction_of(item.question))
        self.question.setAlignment(Qt.AlignmentFlag.AlignLeading | Qt.AlignmentFlag.AlignVCenter)
        detail = " ".join(item.detail.split())
        if len(detail) > 180:
            detail = detail[:177] + "…"
        self.detail.setText(detail)
        self.detail.setVisible(bool(detail))
        self.detail.setLayoutDirection(direction_of(detail))
        self.detail.setAlignment(Qt.AlignmentFlag.AlignLeading | Qt.AlignmentFlag.AlignVCenter)
        self._on_tick()
        self._tick.start()
        self.show()
        self.updateGeometry()

    def _hide_card(self) -> None:
        self._tick.stop()
        self.hide()

    def _answer(self, approved: bool, via: str) -> None:
        item = self.current
        if item is None:
            return
        self.remove(item.confirm_id)
        self.answered.emit(item.confirm_id, approved, via)

    def _on_tick(self) -> None:
        if self.current is None:
            self._tick.stop()
            return
        left = self.seconds_left()
        self.count_label.setText(f"{countdown_text(left)} {tr('set.voice.seconds')}")
        if left <= 0:
            self._answer(False, "timeout")
            return
        self.update()

    def paintEvent(self, _event: Any) -> None:  # noqa: N802
        if self.current is None:
            return
        total = max(0.5, self.current.expires_at - self.current.created_at)
        fraction = self.seconds_left() / total
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        bar = QRectF(18, self.height() - 6, self.width() - 36, 2.5)
        color = theme.WARNING if fraction > 0.25 else theme.DANGER
        # Anchored at the right (reading start) and shrinking towards it.
        paint_bar(p, bar, fraction, color, track_alpha=0.06, rtl=True)
        p.end()


__all__ = ["ConfirmCard", "PendingCard"]
