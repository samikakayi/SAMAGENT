"""Panel pages. Every page: ``Page(app, bridge)`` with ``handle_event(ev)``
(GUI thread) and ``on_shown()`` (refresh when the user opens it). Pages never
call core objects directly except the thread-safe ones allowed by the
contract; everything else goes through ``bridge.call / on_core / run``."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from ..strings import tr

# Space between scrolled content and the scroll bar. Qt layout margins are
# physical (a right margin stays on the right in RTL; checked offscreen with
# PySide6 6.11) while an RTL QScrollArea puts its bar on the LEFT, so pages put
# this gutter in the left margin.
SCROLL_GUTTER = 12


class Page(QWidget):
    key = ""

    def __init__(self, app: Any, bridge: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.app = app
        self.bridge = bridge
        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(0, 0, 0, 0)
        self.root.setSpacing(16)
        self._loaded_once = False

    def add_header(self, actions: list[QWidget] | None = None) -> QHBoxLayout:
        head = QHBoxLayout()
        head.setSpacing(12)
        texts = QVBoxLayout()
        texts.setSpacing(2)
        title = QLabel(tr(f"tab.{self.key}"))
        title.setObjectName("PageTitle")
        sub = QLabel(tr(f"tab.{self.key}.sub"))
        sub.setObjectName("PageSub")
        texts.addWidget(title)
        texts.addWidget(sub)
        head.addLayout(texts)
        head.addStretch(1)
        for widget in actions or []:
            head.addWidget(widget, 0, Qt.AlignmentFlag.AlignBottom)
        self.root.addLayout(head)
        return head

    def handle_event(self, ev: Any) -> None:  # override
        pass

    def on_shown(self) -> None:  # override
        pass

    def cfg(self, key: str, default: Any = None) -> Any:
        try:
            return self.app.config.get(key, default)
        except Exception:  # noqa: BLE001
            return default


def scroll_area(inner: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    area.setWidget(inner)
    return area


__all__ = ["Page", "scroll_area", "SCROLL_GUTTER"]
