"""چاودێری — active alerts (cancel) and the alert history.

Rows are read from the core ``alerts`` table (``app.db.query`` is thread-safe;
it runs in a worker thread so the GUI never waits on SQLite). Cancelling uses
``app.trading.monitor.cancel`` when the monitor is loaded, else the
``cancel_alert`` tool. Prices keep Latin digits on purpose: they must match
what the user reads on TradingView and MT5; times use Sorani digits.
"""

from __future__ import annotations

import json
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ...events import Alert, ToolFinished
from .. import theme
from ..strings import ckb_digits, tr, tr_or
from ..widgets import A_RIGHT, Badge, EmptyState, IconLabel, is_rtl, when_text
from . import SCROLL_GUTTER, Page, scroll_area

KIND_COLORS = {"price_cross": theme.INFO, "zone_touch": theme.ACCENT, "candle_close": theme.CYAN,
               "volume_spike": theme.WARNING, "strategy_state": theme.SUCCESS}
STATUS_COLORS = {"active": theme.SUCCESS, "fired": theme.WARNING, "cancelled": theme.TEXT_FAINT,
                 "expired": theme.TEXT_FAINT}
ACTIVE_SQL = ("SELECT id, kind, symbol, timeframe, params, note, strategy_id, status, repeat, created_at, "
              "expires_at, fired_at, fire_count, last_value, last_text_ckb FROM alerts WHERE status='active' "
              "ORDER BY created_at DESC LIMIT 200")
HISTORY_SQL = ("SELECT id, kind, symbol, timeframe, params, note, strategy_id, status, repeat, created_at, "
               "expires_at, fired_at, fire_count, last_value, last_text_ckb FROM alerts WHERE status!='active' "
               "ORDER BY COALESCE(fired_at, created_at) DESC LIMIT 100")


def _params(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("params")
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _num(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:,.5f}".rstrip("0").rstrip(".")


def condition_text(row: dict[str, Any]) -> str:
    """Short human description of an alert's condition (Sorani words)."""
    params = _params(row)
    kind = row.get("kind")
    direction = params.get("direction")
    dir_text = tr(f"mon.dir.{direction}") if direction in ("up", "down", "any") else ""
    if kind in ("price_cross", "candle_close") and params.get("level") is not None:
        text = _num(params["level"])
    elif kind == "zone_touch" and params.get("low") is not None and params.get("high") is not None:
        text = f"{_num(params['low'])} – {_num(params['high'])}"
    elif kind == "volume_spike":
        text = tr("mon.volume", k=_num(params.get("k", 2)), n=params.get("n", 20))
    elif kind == "strategy_state":
        text = str(row.get("strategy_id") or params.get("strategy_id") or "")
    else:
        text = ""
    parts = [p for p in (text, dir_text, str(row.get("timeframe") or ""),
                         tr("mon.repeat") if row.get("repeat") else "") if p]
    return " · ".join(parts)


class AlertRow(QFrame):
    def __init__(self, row: dict[str, Any], on_cancel: Any = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.alert_id = row.get("id")
        self.setObjectName("AlertRow")
        self.setStyleSheet(f"#AlertRow {{ background: {theme.SURFACE}; border: 1px solid {theme.BORDER_SOFT};"
                           " border-radius: 12px; }")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(12)
        kind = str(row.get("kind") or "")
        lay.addWidget(IconLabel("monitor", 16, KIND_COLORS.get(kind, theme.ACCENT)))
        col = QVBoxLayout()
        col.setSpacing(2)
        top = QHBoxLayout()
        top.setSpacing(8)
        symbol = QLabel(str(row.get("symbol") or ""))
        symbol.setStyleSheet("font-size: 14px; font-weight: 650;")
        top.addWidget(symbol)
        top.addWidget(Badge(tr_or(f"mon.kind.{kind}", kind), KIND_COLORS.get(kind, theme.ACCENT)))
        status = str(row.get("status") or "active")
        if status != "active":
            top.addWidget(Badge(tr(f"mon.status.{status}"), STATUS_COLORS.get(status, theme.TEXT_FAINT)))
        top.addStretch(1)
        col.addLayout(top)
        # RLM makes the paragraph right-to-left even when it starts with a price,
        # so the parts read in Sorani order; A_RIGHT keeps every row aligned.
        cond = QLabel("\u200f" + condition_text(row))
        cond.setObjectName("Muted")
        cond.setAlignment(A_RIGHT)
        col.addWidget(cond)
        text = str(row.get("last_text_ckb") or row.get("note") or "")
        if text:
            said = QLabel(text)
            said.setWordWrap(True)
            said.setLayoutDirection(Qt.LayoutDirection.RightToLeft if is_rtl(text) else Qt.LayoutDirection.LeftToRight)
            said.setAlignment(Qt.AlignmentFlag.AlignLeading)
            col.addWidget(said)
        lay.addLayout(col, 1)
        when = QLabel(when_text(row.get("fired_at") or row.get("created_at")))
        when.setObjectName("Faint")
        lay.addWidget(when, 0, Qt.AlignmentFlag.AlignTop)
        if on_cancel is not None:
            cancel = QPushButton(theme.ICONS["close"])
            cancel.setObjectName("IconButton")
            cancel.setToolTip(tr("mon.cancel"))
            cancel.setCursor(Qt.CursorShape.PointingHandCursor)
            cancel.clicked.connect(lambda: on_cancel(self.alert_id))
            lay.addWidget(cancel, 0, Qt.AlignmentFlag.AlignVCenter)


class MonitorPage(Page):
    key = "monitor"

    def __init__(self, app: Any, bridge: Any, parent: QWidget | None = None) -> None:
        super().__init__(app, bridge, parent)
        self.cancel_all_button = QPushButton(tr("mon.cancel_all"))
        self.cancel_all_button.setObjectName("Danger")
        self.cancel_all_button.clicked.connect(self._cancel_all)
        self.add_header([self.cancel_all_button])
        self.active_rows: list[dict[str, Any]] = []
        self.history_rows: list[dict[str, Any]] = []

        inner = QWidget()
        self.lay = QVBoxLayout(inner)
        self.lay.setContentsMargins(SCROLL_GUTTER, 0, 0, 0)      # RTL scroll bar is on the left
        self.lay.setSpacing(8)
        self.active_title = _section(tr("mon.active"))
        self.active_box = QVBoxLayout()
        self.active_box.setSpacing(8)
        self.history_title = _section(tr("mon.history"))
        self.history_box = QVBoxLayout()
        self.history_box.setSpacing(8)
        self.lay.addWidget(self.active_title)
        self.lay.addLayout(self.active_box)
        self.lay.addSpacing(14)
        self.lay.addWidget(self.history_title)
        self.lay.addLayout(self.history_box)
        self.lay.addStretch(1)
        self.root.addWidget(scroll_area(inner), 1)
        self._render()

    def on_shown(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        db = getattr(self.app, "db", None)
        if db is None:
            return
        self.bridge.run(lambda: (db.query(ACTIVE_SQL), db.query(HISTORY_SQL)), on_ok=self._loaded,
                        on_err=lambda _e: None)

    def _loaded(self, result: Any) -> None:
        active, history = result
        self.active_rows, self.history_rows = list(active), list(history)
        self._render()

    def _render(self) -> None:
        for box in (self.active_box, self.history_box):
            while box.count():
                widget = box.takeAt(0).widget()
                if widget is not None:
                    widget.setParent(None)
                    widget.deleteLater()
        for row in self.active_rows:
            self.active_box.addWidget(AlertRow(row, self.cancel))
        if not self.active_rows:
            self.active_box.addWidget(EmptyState(tr("mon.empty"), "monitor"))
        for row in self.history_rows:
            self.history_box.addWidget(AlertRow(row))
        if not self.history_rows:
            empty = QLabel(tr("mon.history_empty"))
            empty.setObjectName("Faint")
            self.history_box.addWidget(empty)
        self.active_title.setText(f"{tr('mon.active')} ({ckb_digits(len(self.active_rows))})")
        self.cancel_all_button.setEnabled(bool(self.active_rows))

    def cancel(self, alert_id: Any) -> None:
        monitor = getattr(getattr(self.app, "trading", None), "monitor", None)
        if monitor is not None and hasattr(monitor, "cancel"):
            self.bridge.on_core(monitor.cancel, alert_id, on_ok=lambda _r: self.refresh(),
                                on_err=lambda _e: self.refresh())
        else:
            self.bridge.call(self.app.tools.dispatch("cancel_alert", {"alert_id": str(alert_id)}, source="ui"),
                             on_ok=lambda _r: self.refresh(), on_err=lambda _e: self.refresh())

    def _cancel_all(self) -> None:
        self.bridge.call(self.app.tools.dispatch("cancel_alert", {"alert_id": "all"}, source="ui"),
                         on_ok=lambda _r: self.refresh(), on_err=lambda _e: self.refresh())

    def handle_event(self, ev: Any) -> None:
        if isinstance(ev, Alert) or (isinstance(ev, ToolFinished) and ev.name in ("set_alert", "cancel_alert")):
            if self.isVisible():
                self.refresh()


def _section(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("CardTitle")
    return lab


__all__ = ["MonitorPage", "condition_text", "AlertRow"]
