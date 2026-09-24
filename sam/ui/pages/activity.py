"""چالاکی — what SAM did (``activity`` table) and how fast (``timings`` table).

Speed is a feature (DESIGN §1): this page shows every tool call with its result
and duration, the per-stage timing averages of the last 24 hours (STT, first
LLM token, first audio, tools, CDP, MT5, ...) and the stage breakdown of the
latest turn. Reads run in a worker thread (``bridge.run``); a refresh is
debounced so a burst of tool events costs one query.
"""

from __future__ import annotations

import time
from typing import Any

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QFontMetricsF, QPainter
from PySide6.QtWidgets import (QAbstractItemView, QHeaderView, QLabel, QPushButton, QSplitter, QTableWidget,
                               QTableWidgetItem, QWidget)

from ...events import ConfirmResult, ToolFinished, ToolStarted
from .. import theme
from ..strings import ms_text, stage_label, tool_label, tr, tr_or
from ..widgets import A_LEFT, A_RIGHT, Card, EmptyState, bidi_text, clock_text, paint_bar
from . import Page, scroll_area

ACTIVITY_SQL = ("SELECT id, at, kind, name, ok, summary, duration_ms, source FROM activity "
                "ORDER BY id DESC LIMIT 150")
TIMINGS_SQL = ("SELECT stage, COUNT(*) AS n, AVG(ms) AS avg_ms, MAX(ms) AS max_ms FROM timings "
               "WHERE at > ? AND kind != 'startup' GROUP BY stage ORDER BY avg_ms DESC LIMIT 40")
LAST_TURN_SQL = ("SELECT stage, ms FROM timings WHERE turn_id = (SELECT turn_id FROM timings WHERE "
                 "turn_id IS NOT NULL AND turn_id != '' ORDER BY id DESC LIMIT 1) ORDER BY id")
COLUMNS = ("time", "what", "result", "duration", "summary")


def _load(db: Any) -> dict[str, Any]:
    return {"activity": db.query(ACTIVITY_SQL), "timings": db.query(TIMINGS_SQL, (time.time() - 86400,)),
            "last_turn": db.query(LAST_TURN_SQL)}


class TimingsView(QWidget):
    """Stage | avg bar | avg | max | count, painted for a clean, dense look.

    Stage names are shown in Sorani (``stage_label``); the technical name is in
    the tooltip of the row under the mouse."""

    ROW_H = 30
    HEAD_H = 26

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.rows: list[dict[str, Any]] = []
        self._font = theme.ui_font(13)
        self._head = theme.ui_font(12)
        self._mono = theme.latin_font(12, 500)
        self.setMouseTracking(True)

    def set_rows(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.setMinimumHeight(max(60, self.HEAD_H + len(rows) * self.ROW_H + 8))
        self.update()

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802
        index = int((event.position().y() - self.HEAD_H - 4) // self.ROW_H)
        self.setToolTip(str(self.rows[index].get("stage") or "") if 0 <= index < len(self.rows) else "")
        super().mouseMoveEvent(event)

    def paintEvent(self, _event: Any) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not self.rows:
            p.setFont(self._font)
            p.setPen(theme.qcolor(theme.TEXT_FAINT))
            p.drawText(self.rect(), int(Qt.AlignmentFlag.AlignCenter), tr("act.empty"))
            p.end()
            return
        top_avg = max(float(r.get("avg_ms") or 0) for r in self.rows) or 1.0
        w = self.width()
        name_w = 180.0
        num_w = 76.0
        p.translate(14, 0)          # room for the RTL scroll bar on the left
        w -= 14
        # Header: RTL, so the stage column is on the right and the numbers on the left.
        p.setFont(self._head)
        p.setPen(theme.qcolor(theme.TEXT_FAINT))
        p.drawText(QRectF(w - name_w, 0, name_w, self.HEAD_H), int(A_RIGHT), tr("act.stage"))
        for col, key in enumerate(("act.count", "act.max", "act.avg")):
            p.drawText(QRectF(col * num_w, 0, num_w, self.HEAD_H), int(A_LEFT), tr(key))
        p.fillRect(QRectF(0, self.HEAD_H - 1, w, 1), theme.qcolor(theme.BORDER_SOFT))
        for i, row in enumerate(self.rows):
            y = self.HEAD_H + 4 + i * self.ROW_H
            avg = float(row.get("avg_ms") or 0)
            # RTL: stage name on the right, numbers on the left, bar in between.
            p.setFont(self._font)
            p.setPen(theme.qcolor(theme.TEXT))
            fm = QFontMetricsF(self._font)
            stage = fm.elidedText(stage_label(str(row.get("stage"))), Qt.TextElideMode.ElideRight, name_w - 8)
            p.drawText(QRectF(w - name_w, y, name_w, self.ROW_H), int(A_RIGHT), stage)
            p.setFont(self._mono)
            p.setPen(theme.qcolor(theme.TEXT_MUTED))
            p.drawText(QRectF(0, y, num_w, self.ROW_H), int(A_LEFT),
                       f"×{int(row.get('n') or 0)}")
            p.drawText(QRectF(num_w, y, num_w, self.ROW_H), int(A_LEFT),
                       ms_text(float(row.get("max_ms") or 0)))
            p.setPen(theme.qcolor(theme.TEXT))
            p.drawText(QRectF(num_w * 2, y, num_w, self.ROW_H),
                       int(A_LEFT), ms_text(avg))
            bar_left = num_w * 3 + 6
            bar = QRectF(bar_left, y + self.ROW_H / 2 - 3, max(10.0, w - name_w - bar_left - 12), 6)
            color = theme.SUCCESS if avg < 1500 else theme.WARNING if avg < 4500 else theme.DANGER
            paint_bar(p, bar, avg / top_avg, color, track_alpha=0.05)
        p.end()


class ActivityPage(Page):
    key = "activity"

    def __init__(self, app: Any, bridge: Any, parent: QWidget | None = None) -> None:
        super().__init__(app, bridge, parent)
        refresh = QPushButton(tr("act.refresh"))
        refresh.clicked.connect(self.refresh)
        self.add_header([refresh])
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(400)
        self._debounce.timeout.connect(self.refresh)
        self.rows: list[dict[str, Any]] = []

        split = QSplitter(Qt.Orientation.Vertical)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(14)
        recent = Card(tr("act.recent"))
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.horizontalHeader().setVisible(False)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.table.setWordWrap(False)
        self.table.verticalHeader().setDefaultSectionSize(34)
        header = self.table.horizontalHeader()
        for i, mode in enumerate((QHeaderView.ResizeMode.ResizeToContents, QHeaderView.ResizeMode.ResizeToContents,
                                  QHeaderView.ResizeMode.ResizeToContents, QHeaderView.ResizeMode.ResizeToContents,
                                  QHeaderView.ResizeMode.Stretch)):
            header.setSectionResizeMode(i, mode)
        self.table_empty = EmptyState(tr("act.empty"), "activity")
        recent.body.addWidget(self.table)
        recent.body.addWidget(self.table_empty)

        timings = Card(tr("act.timings"))
        self.last_turn = QLabel("")
        self.last_turn.setObjectName("Muted")
        self.last_turn.setWordWrap(True)
        self.timings = TimingsView()
        timings.body.addWidget(self.last_turn)
        timings.body.addWidget(scroll_area(self.timings), 1)
        split.addWidget(recent)
        split.addWidget(timings)
        split.setSizes([400, 330])
        self.root.addWidget(split, 1)
        self._render_activity([])

    def on_shown(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        db = getattr(self.app, "db", None)
        if db is None:
            return
        self.bridge.run(_load, db, on_ok=self._loaded, on_err=lambda _e: None)

    def _loaded(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        self._render_activity(list(data.get("activity") or []))
        self.timings.set_rows(list(data.get("timings") or []))
        stages = [r for r in data.get("last_turn") or [] if r.get("stage") != "total"]
        total = next((r for r in data.get("last_turn") or [] if r.get("stage") == "total"), None)
        if stages or total:
            parts = [f"{stage_label(str(r['stage']))} {ms_text(float(r['ms']))}" for r in stages[:10]]
            if total:
                parts.append(f"{stage_label('total')} {ms_text(float(total['ms']))}")
            self.last_turn.setText(f"{tr('act.last_turn')}:  " + "  ·  ".join(parts))
        else:
            self.last_turn.setText("")

    def _render_activity(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            self._fill_row(i, row)
        self.table.setVisible(bool(rows))
        self.table_empty.setVisible(not rows)

    def _fill_row(self, i: int, row: dict[str, Any]) -> None:
        kind = str(row.get("kind") or "")
        name = str(row.get("name") or "")
        ok = row.get("ok")
        kind_text = tr_or(f"act.kind.{kind}", kind)
        what = tool_label(name) if kind == "tool" else f"{kind_text} · {tool_label(name)}" if name else kind_text
        cells = [
            clock_text(row.get("at")),
            what,
            tr("act.running") if ok is None and row.get("running") else ("✓" if ok else "✗" if ok is not None else ""),
            ms_text(row.get("duration_ms")) if row.get("duration_ms") is not None else "",
            # Summaries are often English: without an LTR embedding the RTL table
            # shows ".The user did not approve the action" (render check).
            bidi_text(str(row.get("summary") or "")),
        ]
        for col, text in enumerate(cells):
            item = QTableWidgetItem(text)
            item.setToolTip(str(row.get("summary") or "") if col == 4 else name)
            if col == 2 and ok is not None:
                item.setForeground(theme.qcolor(theme.SUCCESS if ok else theme.DANGER))
            if col in (0, 3):
                item.setForeground(theme.qcolor(theme.TEXT_MUTED))
            item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeading)
            self.table.setItem(i, col, item)

    def handle_event(self, ev: Any) -> None:
        if isinstance(ev, ToolStarted):
            if self.isVisible():
                self.rows.insert(0, {"at": ev.at, "kind": "tool", "name": ev.name, "ok": None, "running": True,
                                     "summary": "", "source": ev.source})
                self._render_activity(self.rows[:150])
        elif isinstance(ev, (ToolFinished, ConfirmResult)):
            if self.isVisible():
                self._debounce.start()


__all__ = ["ActivityPage", "TimingsView"]
