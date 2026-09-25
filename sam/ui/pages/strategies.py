"""ستراتیژییەکان — the user's strategy cards (list, detail, add by paste).

Data: ``app.trading.strategies`` (StrategyStore: list/get/set_status/versions,
sync DB methods -> run on the core loop via ``bridge.on_core``). ``list()``
returns summary rows (``rules`` is a count, no card JSON), so selecting a card
loads the whole card with ``get(id)`` before the rules, timeframes, risk and
source text are shown. Adding goes through the ``strategy_save`` tool
(``app.tools.dispatch``) so the same extraction, validation, activity log and
events apply as for a spoken strategy.
"""

from __future__ import annotations

import json
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QButtonGroup, QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QPlainTextEdit, QPushButton, QSplitter, QVBoxLayout, QWidget)

from ...events import ToolFinished
from .. import theme
from ..strings import ckb_digits, tr, tr_or
from ..widgets import A_RIGHT, Badge, Card, EmptyState, HLine, is_rtl, when_text
from . import Page, scroll_area

STATUS_COLORS = {"active": theme.SUCCESS, "draft": theme.WARNING, "archived": theme.TEXT_FAINT}


def card_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Merge a strategy_cards row with its JSON ``card`` (either shape works)."""
    card = row.get("card")
    if isinstance(card, str):
        try:
            card = json.loads(card)
        except (json.JSONDecodeError, TypeError):
            card = None
    merged: dict[str, Any] = dict(card) if isinstance(card, dict) else {}
    for key, value in row.items():
        if key != "card" and value not in (None, ""):
            merged.setdefault(key, value)
            if key in ("status", "version", "title_ckb", "title_en", "summary_ckb"):
                merged[key] = value
    return merged


class StrategyRow(QWidget):
    def __init__(self, fields: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(3)
        top = QHBoxLayout()
        title = QLabel(str(fields.get("title_ckb") or fields.get("title_en") or fields.get("id") or ""))
        title.setStyleSheet("font-size: 14px; font-weight: 600;")
        status = str(fields.get("status") or "draft")
        top.addWidget(title, 1)
        top.addWidget(Badge(tr(f"strat.status.{status}"), STATUS_COLORS.get(status, theme.ACCENT)))
        lay.addLayout(top)
        rules = fields.get("rules")
        count = len(rules) if isinstance(rules, list) else rules if isinstance(rules, int) else None
        parts = (str(fields.get("title_en") or ""),
                 f"{tr('strat.version')} {ckb_digits(fields.get('version') or 1)}",
                 tr("strat.rules_count", n=ckb_digits(count)) if count else "")
        sub = QLabel(" · ".join(x for x in parts if x))
        sub.setObjectName("Faint")
        lay.addWidget(sub)


class StrategiesPage(Page):
    key = "strategies"

    def __init__(self, app: Any, bridge: Any, parent: QWidget | None = None) -> None:
        super().__init__(app, bridge, parent)
        self.add_button = QPushButton(tr("strat.add"))
        self.add_button.setObjectName("Primary")
        self.add_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.add_button.clicked.connect(self._toggle_add)
        self.add_header([self.add_button])
        self.rows: list[dict[str, Any]] = []
        self.selected: str | None = None
        self.filter = "all"

        # add-by-paste box (hidden until "New strategy")
        self.add_box = Card(tr("strat.add"), tr("strat.add_hint"))
        self.paste = QPlainTextEdit()
        self.paste.setPlaceholderText(tr("strat.add_hint"))
        self.paste.setMinimumHeight(120)
        self.paste.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.add_box.body.addWidget(self.paste)
        buttons = QHBoxLayout()
        self.save_button = QPushButton(tr("strat.save"))
        self.save_button.setObjectName("Primary")
        self.save_button.clicked.connect(self._save_new)
        cancel = QPushButton(tr("strat.cancel"))
        cancel.clicked.connect(self._toggle_add)
        self.add_status = QLabel("")
        self.add_status.setObjectName("Muted")
        self.add_status.setWordWrap(True)
        buttons.addWidget(self.save_button)
        buttons.addWidget(cancel)
        buttons.addWidget(self.add_status, 1)
        self.add_box.body.addLayout(buttons)
        self.add_box.hide()
        self.root.addWidget(self.add_box)

        # filter chips
        chips = QHBoxLayout()
        chips.setSpacing(8)
        self.chip_group = QButtonGroup(self)
        for key in ("all", "active", "draft", "archived"):
            chip = QPushButton(tr("strat.filter.all") if key == "all" else tr(f"strat.status.{key}"))
            chip.setObjectName("Chip")
            chip.setCheckable(True)
            chip.setChecked(key == "all")
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            chip.clicked.connect(lambda _=False, k=key: self._set_filter(k))
            self.chip_group.addButton(chip)
            chips.addWidget(chip)
        chips.addStretch(1)
        self.root.addLayout(chips)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(14)
        left = QFrame()
        left.setObjectName("Card")
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(8, 8, 8, 8)
        self.list = QListWidget()
        self.list.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.list.currentItemChanged.connect(self._on_select)
        self.list_empty = EmptyState(tr("strat.empty"), "strategies")
        left_lay.addWidget(self.list)
        left_lay.addWidget(self.list_empty)
        self.detail_inner = QWidget()
        self.detail_lay = QVBoxLayout(self.detail_inner)
        self.detail_lay.setContentsMargins(22, 20, 22, 20)
        self.detail_lay.setSpacing(12)
        detail_frame = QFrame()
        detail_frame.setObjectName("Card")
        frame_lay = QVBoxLayout(detail_frame)
        frame_lay.setContentsMargins(0, 0, 0, 0)
        frame_lay.addWidget(scroll_area(self.detail_inner))
        split.addWidget(left)
        split.addWidget(detail_frame)
        split.setSizes([340, 560])
        self.root.addWidget(split, 1)
        self._render_detail(None)

    # -- data -----------------------------------------------------------------------------------------
    @property
    def store(self) -> Any:
        trading = getattr(self.app, "trading", None)
        return getattr(trading, "strategies", None) if trading is not None else None

    def on_shown(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        store = self.store
        if store is None:
            self.rows = []
            self._render_list(unavailable=True)
            return
        self.bridge.on_core(store.list, None, on_ok=self._loaded, on_err=lambda _e: self._render_list())

    def _loaded(self, rows: Any) -> None:
        self.rows = [card_fields(r) for r in (rows or []) if isinstance(r, dict)]
        self._render_list()

    def _set_filter(self, key: str) -> None:
        self.filter = key
        self._render_list()

    def _render_list(self, unavailable: bool = False) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        shown = [r for r in self.rows if self.filter == "all" or r.get("status") == self.filter]
        order = {"active": 0, "draft": 1, "archived": 2}
        shown.sort(key=lambda r: (order.get(str(r.get("status")), 3), -float(r.get("updated_at") or 0)))
        for fields in shown:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, fields.get("id"))
            widget = StrategyRow(fields)
            item.setSizeHint(widget.sizeHint())
            self.list.addItem(item)
            self.list.setItemWidget(item, widget)
        self.list.blockSignals(False)
        self.list.setVisible(bool(shown))
        self.list_empty.setVisible(not shown)
        self.list_empty.set_text(tr("strat.unavailable") if unavailable else tr("strat.empty"))
        if self.selected and any(r.get("id") == self.selected for r in shown):
            for i in range(self.list.count()):
                if self.list.item(i).data(Qt.ItemDataRole.UserRole) == self.selected:
                    self.list.setCurrentRow(i)
        else:
            self._render_detail(None)

    def _on_select(self, current: QListWidgetItem | None, _prev: Any = None) -> None:
        if current is None:
            return
        self.selected = current.data(Qt.ItemDataRole.UserRole)
        fields = next((r for r in self.rows if r.get("id") == self.selected), None)
        self._render_detail(fields)
        store = self.store
        if store is None or fields is None:
            return
        selected = self.selected
        if hasattr(store, "get"):
            self.bridge.on_core(store.get, selected, on_ok=lambda card: self._show_full(selected, fields, card),
                                on_err=lambda _e: self._load_versions(selected))
        else:
            self._load_versions(selected)

    def _show_full(self, strategy_id: str, row: dict[str, Any], card: Any) -> None:
        """Re-render the detail with the whole card (rules, timeframes, risk, source)."""
        if strategy_id != self.selected:
            return                      # the user moved on meanwhile
        if isinstance(card, dict):
            self._render_detail(card_fields({**row, "card": card}))
        self._load_versions(strategy_id)

    def _load_versions(self, strategy_id: str) -> None:
        store = self.store
        if store is not None and hasattr(store, "versions"):
            self.bridge.on_core(store.versions, strategy_id, on_ok=self._show_versions, on_err=lambda _e: None)

    # -- detail -------------------------------------------------------------------------------------------------
    def _clear_detail(self) -> None:
        while self.detail_lay.count():
            item = self.detail_lay.takeAt(0)
            widget, child = item.widget(), item.layout()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
            elif child is not None:
                _delete_layout(child)

    def _render_detail(self, fields: dict[str, Any] | None) -> None:
        self._clear_detail()
        if fields is None:
            self.detail_lay.addWidget(EmptyState(tr("strat.select"), "info"), 1)
            return
        status = str(fields.get("status") or "draft")
        head = QHBoxLayout()
        title = QLabel(str(fields.get("title_ckb") or fields.get("id")))
        title.setStyleSheet("font-size: 19px; font-weight: 650;")
        title.setWordWrap(True)
        head.addWidget(title, 1)
        head.addWidget(Badge(tr(f"strat.status.{status}"), STATUS_COLORS.get(status, theme.ACCENT)))
        self.detail_lay.addLayout(head)
        if fields.get("title_en"):
            self.detail_lay.addWidget(_muted(str(fields["title_en"])))
        summary = str(fields.get("summary_ckb") or "")
        if summary:
            self.detail_lay.addWidget(_text(summary, 14))

        actions = QHBoxLayout()
        if status != "active":
            activate = QPushButton(tr("strat.activate"))
            activate.setObjectName("Primary")
            activate.clicked.connect(lambda: self._set_status("active"))
            actions.addWidget(activate)
        if status != "archived":
            archive = QPushButton(tr("strat.archive"))
            archive.clicked.connect(lambda: self._set_status("archived"))
            actions.addWidget(archive)
        else:
            draft = QPushButton(tr("strat.draft"))
            draft.clicked.connect(lambda: self._set_status("draft"))
            actions.addWidget(draft)
        actions.addStretch(1)
        self.detail_lay.addLayout(actions)
        self.detail_lay.addWidget(HLine())

        tfs = fields.get("timeframes") if isinstance(fields.get("timeframes"), dict) else {}
        facts = []
        if tfs:
            facts.append((tr("strat.timeframes"), " · ".join(
                f"{tr('strat.tf.' + k)}: {v}" for k, v in tfs.items() if v and k in ("bias", "setup", "entry"))))
        for key, label_key in (("markets", "strat.markets"), ("sessions", "strat.sessions")):
            value = fields.get(key)
            if value:
                facts.append((tr(label_key), " · ".join(map(str, value)) if isinstance(value, list) else str(value)))
        risk = fields.get("risk")
        if isinstance(risk, dict) and risk:
            facts.append((tr("strat.risk"), " · ".join(_risk_text(k, v) for k, v in risk.items() if v is not None)))
        for name, value in facts:
            line = QHBoxLayout()
            key_label = QLabel(name)
            key_label.setObjectName("Faint")
            key_label.setFixedWidth(96)
            line.addWidget(key_label)
            value_label = _text(value, 13)
            value_label.setAlignment(A_RIGHT)       # facts line up next to their names
            line.addWidget(value_label, 1)
            self.detail_lay.addLayout(line)

        rules = fields.get("rules") if isinstance(fields.get("rules"), list) else []
        if rules:
            head = QLabel(tr("strat.rules"))
            head.setObjectName("CardTitle")
            self.detail_lay.addWidget(head)
            for i, rule in enumerate(rules, 1):
                if isinstance(rule, dict):
                    self.detail_lay.addWidget(_rule_widget(i, rule))
        source = str(fields.get("source_text") or "")
        if source:
            head = QLabel(tr("strat.source"))
            head.setObjectName("CardTitle")
            self.detail_lay.addWidget(head)
            src = _text(source, 12.5)
            src.setStyleSheet(f"color: {theme.TEXT_MUTED}; background: {theme.SURFACE_2}; border-radius: 10px;"
                              " padding: 10px 12px;")
            self.detail_lay.addWidget(src)
        self.versions_label = QLabel("")
        self.versions_label.setObjectName("Faint")
        self.versions_label.setWordWrap(True)
        self.detail_lay.addWidget(self.versions_label)
        self.detail_lay.addStretch(1)

    def _show_versions(self, versions: Any) -> None:
        items = [v for v in (versions or []) if isinstance(v, dict)]
        if not items or not hasattr(self, "versions_label"):
            return
        parts = [f"{tr('strat.version')} {ckb_digits(v.get('version'))} — {when_text(v.get('created_at'))}"
                 + (f" ({v.get('reason')})" if v.get("reason") else "") for v in items[:8]]
        try:
            self.versions_label.setText(f"{tr('strat.versions')}: " + " · ".join(parts))
        except RuntimeError:
            pass     # detail was rebuilt meanwhile

    def _set_status(self, status: str) -> None:
        store = self.store
        if store is None or not self.selected:
            return
        self.bridge.on_core(store.set_status, self.selected, status, on_ok=lambda _r: self.refresh(),
                            on_err=lambda _e: self.refresh())

    # -- add by paste -----------------------------------------------------------------------------------------
    def _toggle_add(self) -> None:
        self.add_box.setVisible(not self.add_box.isVisible())
        if self.add_box.isVisible():
            self.paste.setFocus()

    def _save_new(self) -> None:
        text = self.paste.toPlainText().strip()
        if not text:
            return
        self.save_button.setEnabled(False)
        self.add_status.setText(tr("strat.saving"))
        self.bridge.call(self.app.tools.dispatch("strategy_save", {"text": text}, source="ui"),
                         on_ok=self._saved, on_err=lambda e: self._saved({"ok": False, "summary": tr("common.error")}))

    def _saved(self, result: Any) -> None:
        self.save_button.setEnabled(True)
        result = result if isinstance(result, dict) else {}
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        if result.get("ok"):
            # strategy_save returns the Sorani read-back as its summary.
            self.add_status.setText(str(data.get("readback_ckb") or result.get("summary") or tr("strat.saving")))
            self.add_status.setToolTip("")
        else:
            # Failure summaries are English diagnostics (already redacted by the
            # registry): Sorani on screen, the detail in the tooltip.
            self.add_status.setText(tr("strat.save_failed"))
            self.add_status.setToolTip(str(result.get("summary") or ""))
        if result.get("ok"):
            self.paste.clear()
            new_id = data.get("id") or data.get("strategy_id")
            if new_id:
                self.selected = str(new_id)
        self.refresh()

    def handle_event(self, ev: Any) -> None:
        if isinstance(ev, ToolFinished) and ev.ok and ev.name == "strategy_save" and self.isVisible():
            self.refresh()


def _text(value: str, size: float = 13) -> QLabel:
    lab = QLabel(value)
    lab.setWordWrap(True)
    lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    lab.setStyleSheet(f"font-size: {size}px;")
    lab.setLayoutDirection(Qt.LayoutDirection.RightToLeft if is_rtl(value) else Qt.LayoutDirection.LeftToRight)
    lab.setAlignment(Qt.AlignmentFlag.AlignLeading | Qt.AlignmentFlag.AlignTop)
    return lab


def _risk_text(key: str, value: object) -> str:
    """"زۆرترین مەترسی بۆ هەر مامەڵەیەک: ١٪" for known keys, "key: value" otherwise."""
    if key == "max_risk_pct":
        return f"{tr('strat.risk.max_risk_pct')}: {ckb_digits(value)}٪"
    return f"{tr_or(f'strat.risk.{key}', key)}: {ckb_digits(value)}"


def _muted(value: str) -> QLabel:
    lab = _text(value, 12.5)
    lab.setObjectName("Muted")
    return lab


def _rule_widget(index: int, rule: dict[str, Any]) -> QWidget:
    box = QFrame()
    box.setStyleSheet(f"QFrame {{ background: {theme.SURFACE_2}; border-radius: 10px; }}")
    lay = QHBoxLayout(box)
    lay.setContentsMargins(12, 9, 12, 9)
    lay.setSpacing(10)
    num = QLabel(ckb_digits(index))
    num.setFixedWidth(22)
    num.setAlignment(Qt.AlignmentFlag.AlignCenter)
    num.setStyleSheet(f"color: {theme.ACCENT}; font-weight: 700;")
    lay.addWidget(num, 0, Qt.AlignmentFlag.AlignTop)
    col = QVBoxLayout()
    col.setSpacing(3)
    text = str(rule.get("text_ckb") or rule.get("text_en") or "")
    col.addWidget(_text(text, 13))
    check = rule.get("check")
    meta = []
    if rule.get("kind"):
        meta.append(tr_or(f"strat.kind.{rule['kind']}", str(rule["kind"])))
    if isinstance(check, dict) and check.get("predicate"):
        meta.append(f"{tr('strat.predicate')}: {check['predicate']}")
    else:
        meta.append(tr("strat.judged"))
    hint = QLabel(" · ".join(meta))
    hint.setObjectName("Faint")
    col.addWidget(hint)
    lay.addLayout(col, 1)
    return box


def _delete_layout(layout: Any) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget, child = item.widget(), item.layout()
        if widget is not None:
            widget.setParent(None)
            widget.deleteLater()
        elif child is not None:
            _delete_layout(child)


__all__ = ["StrategiesPage", "card_fields"]
