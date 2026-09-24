"""Small shared widgets and text helpers (RTL-aware)."""

from __future__ import annotations

import time
import unicodedata
from typing import Any, Callable

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFontMetricsF, QPainter, QPen
from PySide6.QtWidgets import (QAbstractButton, QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget)

from . import theme
from .strings import ckb_digits


# Physical alignments for custom painting. Without AlignAbsolute Qt swaps
# Left/Right when the painter's layout direction is right-to-left
# (QGuiApplicationPrivate::visualAlignment), which would put Sorani text on the
# wrong side inside the RTL panel.
A_RIGHT = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignAbsolute | Qt.AlignmentFlag.AlignVCenter
A_LEFT = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignAbsolute | Qt.AlignmentFlag.AlignVCenter
A_CENTER = Qt.AlignmentFlag.AlignCenter


# -- text direction ----------------------------------------------------------------------
def is_rtl(text: str) -> bool:
    """Base direction by the first strong character (like HTML ``dir=auto``).

    Sorani/Arabic -> RTL; Latin -> LTR; no strong character -> RTL (the UI is
    Sorani-first). Needed so English tool summaries keep their punctuation on
    the right side while Sorani keeps it on the left.
    """
    for ch in text or "":
        bidi = unicodedata.bidirectional(ch)
        if bidi in ("R", "AL"):
            return True
        if bidi == "L":
            return False
    return True


def direction_of(text: str) -> Qt.LayoutDirection:
    return Qt.LayoutDirection.RightToLeft if is_rtl(text) else Qt.LayoutDirection.LeftToRight


LRE, PDF = "‪", "‬"     # left-to-right embedding / pop directional formatting


def bidi_text(text: str) -> str:
    """Text for a right-to-left widget that has no per-item direction (table
    cells): Latin-first text is wrapped in an LTR embedding so its punctuation
    stays at its end; Sorani text is returned unchanged."""
    if not text or is_rtl(text):
        return text
    return f"{LRE}{text}{PDF}"


def elide(text: str, metrics: QFontMetricsF, width: float, *, keep_tail: bool = False) -> str:
    """One-line elision. ``keep_tail`` keeps the newest words of a live caption
    (Qt ElideLeft removes the *logical* start, which RTL shows on the right)."""
    single = " ".join((text or "").split())
    mode = Qt.TextElideMode.ElideLeft if keep_tail else Qt.TextElideMode.ElideRight
    return metrics.elidedText(single, mode, max(0.0, width))


def clock_text(ts: float | None) -> str:
    """"14:05" local time with Eastern Arabic digits."""
    if not ts:
        return ""
    return ckb_digits(time.strftime("%H:%M", time.localtime(ts)))


def when_text(ts: float | None, now: float | None = None) -> str:
    """Local time for lists: "١٤:٠٥" today, "٢٤/٩ ١٤:٠٥" on other days.

    Day/month are joined with "/" on purpose: "/" joins Arabic-Indic digits
    into one number run, while "-" does not, so "٠٩-٢٤" was shown reordered as
    "٢٤-٠٩" inside the RTL panel (render check)."""
    if not ts:
        return ""
    moment = time.localtime(ts)
    today = time.localtime(now if now is not None else time.time())
    clock = time.strftime("%H:%M", moment)
    if (moment.tm_year, moment.tm_yday) == (today.tm_year, today.tm_yday):
        return ckb_digits(clock)
    return ckb_digits(f"{moment.tm_mday}/{moment.tm_mon} {clock}")


# -- widgets -------------------------------------------------------------------------------------
class StatusDot(QWidget):
    """A small coloured dot with a soft halo (component / key status)."""

    def __init__(self, state: str = "unknown", size: int = 10, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._color = theme.STATUS_COLORS.get(state, theme.STATUS_COLORS["unknown"])
        self.setFixedSize(size + 6, size + 6)

    @property
    def state(self) -> str:
        return self._state

    def set_state(self, state: str, color: str | None = None) -> None:
        self._state = state
        self._color = color or theme.STATUS_COLORS.get(state, theme.STATUS_COLORS["unknown"])
        self.update()

    def paintEvent(self, _event: Any) -> None:  # noqa: N802 - Qt API
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QPointF(self.width() / 2, self.height() / 2)
        r = (self.width() - 6) / 2
        halo = theme.qcolor(self._color, 0.22)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(halo)
        p.drawEllipse(c, r + 2.5, r + 2.5)
        p.setBrush(theme.qcolor(self._color))
        p.drawEllipse(c, r, r)
        p.end()


class ToggleSwitch(QAbstractButton):
    """iOS-style switch (checkable)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(44, 24)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(44, 24)

    def paintEvent(self, _event: Any) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        on = self.isChecked()
        track = theme.qcolor(theme.ACCENT if on else theme.SURFACE_3)
        if not self.isEnabled():
            track.setAlphaF(0.4)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(QRectF(0, 0, 44, 24), 12, 12)
        # In RTL the "on" knob sits on the left (mirrored like the rest of the UI).
        rtl = self.layoutDirection() == Qt.LayoutDirection.RightToLeft
        right_side = on != rtl
        x = 44 - 12 if right_side else 12
        p.setBrush(QColor("#0B0E14") if on else QColor(theme.TEXT_MUTED))
        p.drawEllipse(QPointF(x, 12), 8.5, 8.5)
        p.end()


class Card(QFrame):
    """Rounded surface with an optional title row."""

    def __init__(self, title: str = "", subtitle: str = "", parent: QWidget | None = None,
                 margins: tuple[int, int, int, int] = (20, 18, 20, 18), spacing: int = 12) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(*margins)
        self.body.setSpacing(spacing)
        if title:
            head = QLabel(title)
            head.setObjectName("CardTitle")
            self.body.addWidget(head)
            self.title_label = head
        if subtitle:
            sub = QLabel(subtitle)
            sub.setObjectName("Muted")
            sub.setWordWrap(True)
            self.body.addWidget(sub)


class Badge(QLabel):
    """Rounded status chip (strategy status, alert kind)."""

    def __init__(self, text: str = "", color: str = theme.ACCENT, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.set_color(color)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)

    def set_color(self, color: str) -> None:
        bg = theme.qcolor(color, 0.16)
        self.setStyleSheet(
            f"QLabel {{ color: {color}; background: rgba({bg.red()},{bg.green()},{bg.blue()},{bg.alpha()});"
            f" border-radius: 9px; padding: 2px 9px; font-size: 12px; font-weight: 600; }}")


class IconLabel(QLabel):
    """A glyph from the system icon font.

    The font is set in the label's *own* style sheet: an ancestor's
    ``QWidget { font-family: ... }`` rule would otherwise replace the icon font
    (a widget's own sheet always wins over inherited ones)."""

    def __init__(self, icon: str, size: int = 16, color: str | None = None, parent: QWidget | None = None) -> None:
        super().__init__(theme.ICONS.get(icon, icon), parent)
        self.setObjectName("Icon")
        self._size = size
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.set_color(color)

    def set_color(self, color: str | None) -> None:
        css = f'font-family: "{theme.icon_family()}"; font-size: {self._size}px; background: transparent;'
        self.setStyleSheet(css + (f" color: {color};" if color else ""))

    def set_icon(self, icon: str, color: str | None = None) -> None:
        self.setText(theme.ICONS.get(icon, icon))
        if color is not None:
            self.set_color(color)


class EmptyState(QWidget):
    """Centered hint shown when a list has nothing yet."""

    def __init__(self, text: str, icon: str = "info", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 28, 24, 28)
        lay.setSpacing(10)
        glyph = IconLabel(icon, 26, theme.TEXT_FAINT)
        self.label = QLabel(text)
        self.label.setObjectName("Muted")
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addStretch(1)
        lay.addWidget(glyph)
        lay.addWidget(self.label)
        lay.addStretch(1)

    def set_text(self, text: str) -> None:
        self.label.setText(text)


class HLine(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(1)
        self.setStyleSheet(f"background: {theme.BORDER_SOFT};")


def row(*widgets: QWidget | None, spacing: int = 10, stretch_at: int | None = None) -> QHBoxLayout:
    """HBox with the given widgets (None = stretch)."""
    lay = QHBoxLayout()
    lay.setSpacing(spacing)
    lay.setContentsMargins(0, 0, 0, 0)
    for i, w in enumerate(widgets):
        if w is None:
            lay.addStretch(1)
        else:
            lay.addWidget(w)
        if stretch_at is not None and i == stretch_at:
            lay.setStretch(i, 1)
    return lay


def label(text: str = "", name: str = "", *, wrap: bool = False, selectable: bool = False,
          align_auto: bool = False) -> QLabel:
    lab = QLabel(text)
    if name:
        lab.setObjectName(name)
    lab.setWordWrap(wrap)
    if selectable:
        lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    if align_auto:
        set_auto_direction(lab, text)
    return lab


def set_auto_direction(lab: QLabel, text: str) -> None:
    """Direction + alignment from the text itself (Sorani right, English left)."""
    lab.setLayoutDirection(direction_of(text))
    lab.setAlignment(Qt.AlignmentFlag.AlignLeading | Qt.AlignmentFlag.AlignVCenter)


class Clickable(QFrame):
    """A frame that emits ``clicked`` (list rows, chips)."""

    clicked = Signal()

    def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


def paint_bar(p: QPainter, rect: QRectF, fraction: float, color: str, track_alpha: float = 0.08,
              rtl: bool = True) -> None:
    """Thin rounded progress bar (activity timings, confirm countdown). In the
    RTL UI the fill grows from the right edge (the reading start)."""
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(255, 255, 255, round(255 * track_alpha)))
    radius = rect.height() / 2
    p.drawRoundedRect(rect, radius, radius)
    if fraction > 0:
        width = max(rect.height(), rect.width() * min(1.0, fraction))
        left = rect.right() - width if rtl else rect.left()
        p.setBrush(theme.qcolor(color))
        p.drawRoundedRect(QRectF(left, rect.top(), width, rect.height()), radius, radius)


def pen(color: str, width: float = 1.0, alpha: float | None = None) -> QPen:
    return QPen(theme.qcolor(color, alpha), width)


def call_safely(fn: Callable[..., Any] | None, *args: Any) -> None:
    if fn is not None:
        fn(*args)


def accessible(widget: QWidget, key: str, *, object_name: str = "", **fmt: Any) -> QWidget:
    """Give ``widget`` its UI Automation name (Sorani, ``strings.tr``) and
    description (English). ``object_name`` makes Qt's AutomationId unique
    (Qt builds it from the objectName chain) -- only for widgets whose
    objectName is not a style-sheet selector (#Primary, #Chip ...)."""
    from .strings import en, tr

    widget.setAccessibleName(tr(key, **fmt))
    widget.setAccessibleDescription(en(key, **fmt))
    if object_name:
        widget.setObjectName(object_name)
    return widget


__all__ = ["A_RIGHT", "A_LEFT", "A_CENTER", "is_rtl", "direction_of", "bidi_text", "elide", "clock_text", "when_text",
           "StatusDot", "ToggleSwitch", "Card",
           "Badge", "IconLabel", "EmptyState", "HLine", "row", "label", "set_auto_direction", "Clickable",
           "paint_bar", "pen", "call_safely", "accessible"]
