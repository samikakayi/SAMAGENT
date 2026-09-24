"""The panel: SAM's main window (dark, right-to-left, Sorani first).

Sidebar (right side in RTL) with the brand orb, the five sections and live
component status dots; the body is a stack of pages:
گفتوگۆ (chat) · ستراتیژییەکان (strategies) · چاودێری (alerts) · چالاکی
(activity + timings) · ڕێکخستنەکان (settings).

Closing the window only hides it: SAM keeps running in the island and tray.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (QAbstractButton, QButtonGroup, QFrame, QHBoxLayout, QLabel, QStackedWidget,
                               QVBoxLayout, QWidget)

from ..events import ComponentStatus, VoiceState
from . import theme
from .orb import orb_icon, orb_pixmap
from .pages.activity import ActivityPage
from .pages.chat import ChatPage
from .pages.monitor import MonitorPage
from .pages.settings import SettingsPage
from .pages.strategies import StrategiesPage
from .strings import en, state_word, tr, tr_or
from .widgets import A_RIGHT, StatusDot
from .win32 import bring_to_front, dark_title_bar

PAGES = ("chat", "strategies", "monitor", "activity", "settings")
NAV_ICONS = {"chat": "chat", "strategies": "strategies", "monitor": "monitor", "activity": "activity",
             "settings": "settings"}
# ComponentStatus.component -> sidebar row
COMPONENT_ROWS = {"voice": "voice", "live": "voice", "cascade": "voice", "omniroute": "omniroute",
                  "tradingview": "tradingview", "mt5": "mt5"}


class NavButton(QAbstractButton):
    """Sidebar entry: icon glyph + Sorani label, accent bar when selected."""

    def __init__(self, key: str, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.key = key
        self.setText(text)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setObjectName("NavButton")
        self.setToolTip(en(f"tab.{key}"))
        self._hover = False
        self._label_font = theme.ui_font(14, 500)
        self._label_font_bold = theme.ui_font(14, 650)
        self._icon_font = theme.icon_font(16)
        self._icon = theme.ICONS[NAV_ICONS[key]]

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(200, 42)

    def enterEvent(self, event: Any) -> None:  # noqa: N802
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event: Any) -> None:  # noqa: N802
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, _event: Any) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        checked = self.isChecked()
        if checked or self._hover:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(theme.qcolor(theme.ACCENT_SOFT if checked else theme.SURFACE_2))
            p.drawRoundedRect(rect, 10, 10)
        if checked:
            p.setBrush(theme.qcolor(theme.ACCENT))
            p.drawRoundedRect(QRectF(rect.right() - 3, rect.top() + 11, 3, rect.height() - 22), 1.5, 1.5)
        icon_rect = QRectF(rect.right() - 44, rect.top(), 36, rect.height())
        p.setFont(self._icon_font)
        p.setPen(theme.qcolor(theme.ACCENT if checked else theme.TEXT_MUTED))
        p.drawText(icon_rect, int(Qt.AlignmentFlag.AlignCenter), self._icon)
        p.setFont(self._label_font_bold if checked else self._label_font)
        p.setPen(theme.qcolor(theme.TEXT if checked or self._hover else theme.TEXT_MUTED))
        text_rect = QRectF(rect.left() + 10, rect.top(), rect.width() - 58, rect.height())
        p.drawText(text_rect, int(A_RIGHT), self.text())
        p.end()


class BrandHeader(QWidget):
    """Orb + "SAM" + live status word at the top of the sidebar."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = "idle"
        self.setFixedHeight(64)
        self._name_font = theme.latin_font(18, 650)
        self._name_font.setLetterSpacing(self._name_font.SpacingType.AbsoluteSpacing, 1.6)
        self._status_font = theme.ui_font(12.5, 500)
        self._pix = orb_pixmap(40, state="idle", dpr=2.0)

    def set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state if state in theme.STATE_COLORS else "idle"
            self._pix = orb_pixmap(40, state=self.state, dpr=2.0)
            self.update()

    def paintEvent(self, _event: Any) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        p.drawPixmap(w - 8 - 40, 12, self._pix)
        p.setFont(self._name_font)
        p.setPen(theme.qcolor(theme.TEXT))
        right = w - 8 - 40 - 12
        p.drawText(QRectF(0, 12, right, 22), int(A_RIGHT), "SAM")
        p.setFont(self._status_font)
        p.setPen(theme.qcolor(theme.state_color(self.state)))
        p.drawText(QRectF(0, 34, right, 20), int(A_RIGHT),
                   state_word(self.state))
        p.end()


class ComponentRow(QWidget):
    def __init__(self, key: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 2, 6, 2)
        lay.setSpacing(8)
        self.dot = StatusDot("unknown", 8)
        self.name = QLabel(tr(f"comp.{key}"))
        self.name.setObjectName("Faint")
        self.state_label = QLabel(tr("status.unknown"))
        self.state_label.setObjectName("Faint")
        lay.addWidget(self.dot)
        lay.addWidget(self.name)
        lay.addStretch(1)
        lay.addWidget(self.state_label)

    def set_state(self, state: str, detail: str = "") -> None:
        self.dot.set_state(state)
        self.state_label.setText(tr_or(f"status.{state}", state))
        self.setToolTip(detail)


class Panel(QWidget):
    hidden = Signal()

    def __init__(self, app: Any, bridge: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.app = app
        self.bridge = bridge
        self.setObjectName("Panel")
        self.setWindowTitle("SAM")
        self.setWindowIcon(orb_icon())
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.families = theme.ui_families(self._cfg("ui.font_family"))
        self.setStyleSheet(theme.panel_stylesheet(self.families))
        self.resize(1100, 740)
        self.setMinimumSize(900, 600)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(236)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(14, 16, 14, 16)
        side.setSpacing(4)
        self.brand = BrandHeader()
        side.addWidget(self.brand)
        side.addSpacing(14)
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav: dict[str, NavButton] = {}
        for key in PAGES:
            button = NavButton(key, tr(f"tab.{key}"))
            button.setFixedHeight(42)
            # toggled (not clicked): UI Automation's Toggle on a nav item checks it without a click.
            button.toggled.connect(lambda on, k=key: on and k != self.current and self.show_page(k))
            self.nav_group.addButton(button)
            self.nav[key] = button
            side.addWidget(button)
        side.addStretch(1)
        self.components: dict[str, ComponentRow] = {}
        for key in ("voice", "omniroute", "tradingview", "mt5"):
            comp = ComponentRow(key)
            self.components[key] = comp
            side.addWidget(comp)

        body = QWidget()
        body.setObjectName("PanelBody")
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(28, 22, 28, 22)
        self.stack = QStackedWidget()
        body_lay.addWidget(self.stack)
        self.pages: dict[str, Any] = {
            "chat": ChatPage(app, bridge),
            "strategies": StrategiesPage(app, bridge),
            "monitor": MonitorPage(app, bridge),
            "activity": ActivityPage(app, bridge),
            "settings": SettingsPage(app, bridge),
        }
        for key in PAGES:
            self.stack.addWidget(self.pages[key])

        root.addWidget(sidebar)      # RTL: first widget sits on the right
        root.addWidget(body, 1)
        self.current = ""
        self.in_front = False            # the last show_and_raise made the panel the foreground window
        self.show_page("chat", refresh=False)
        bridge.subscribe(None, self.handle_event)

    def _cfg(self, key: str, default: Any = None) -> Any:
        try:
            return self.app.config.get(key, default)
        except Exception:  # noqa: BLE001
            return default

    # -- navigation --------------------------------------------------------------------------------
    def show_page(self, key: str, refresh: bool = True) -> None:
        if key not in self.pages:
            return
        self.current = key
        self.nav[key].setChecked(True)
        self.stack.setCurrentWidget(self.pages[key])
        if refresh and self.isVisible():
            self._page_shown(key)

    def _page_shown(self, key: str) -> None:
        page = self.pages[key]
        if hasattr(page, "on_shown"):
            page.on_shown()

    def show_and_raise(self, page: str | None = None) -> None:
        if page:
            self.show_page(page, refresh=False)
        if self.isMinimized():
            self.showNormal()
        self.show()
        self.raise_()
        self.activateWindow()
        # activateWindow alone can leave the panel behind the app in front
        # (Windows foreground lock): the user asked for it, so bring it forward.
        self.in_front = bring_to_front(self)
        self._page_shown(self.current)

    # -- events ----------------------------------------------------------------------------------------
    def handle_event(self, ev: Any) -> None:
        if isinstance(ev, VoiceState):
            self.brand.set_state(ev.state)
        elif isinstance(ev, ComponentStatus):
            row = COMPONENT_ROWS.get(ev.component)
            if row in self.components:
                self.components[row].set_state(ev.state, ev.detail)
        for page in self.pages.values():
            handler = getattr(page, "handle_event", None)
            if handler is not None:
                handler(ev)

    def closeEvent(self, event: Any) -> None:  # noqa: N802
        event.ignore()          # SAM keeps running: hide to the island / tray
        self.hide()
        self.hidden.emit()

    def showEvent(self, event: Any) -> None:  # noqa: N802
        super().showEvent(event)
        dark_title_bar(self)


__all__ = ["Panel", "PAGES", "NavButton", "dark_title_bar"]
