"""Pure painting helpers for the island (no state of their own).

Split out of ``island.py`` to keep that module about behaviour (state, events,
geometry, mouse); everything here only draws what it is given.
"""

from __future__ import annotations

import math
from typing import Any

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen, QRadialGradient
from PySide6.QtWidgets import QWidget

from . import theme

METER_WEIGHTS = (0.55, 0.85, 1.0, 0.8, 0.5)   # five bars, tallest in the middle


def device_scale(p: QPainter, widget: QWidget) -> float:
    """Device pixels per logical pixel for this paint. ``QPainter.device()`` is
    the widget even when ``QWidget.render`` redirects into a 2x pixmap; only the
    device transform shows the real scale there (measured: dpr 1.0, m11 2.0)."""
    return max(widget.devicePixelRatioF(), abs(p.deviceTransform().m11()), 1.0)


def paint_glass(p: QPainter, rect: QRectF, *, radius: float, tint_color: str | None = None,
                hover: bool = False) -> None:
    """Dark glass: layered soft shadow, vertical body gradient, an optional
    state-coloured glow behind the orb (right end) and a 1 px top highlight."""
    # Soft layered drop shadow (cheap: ~10 rounded rects, no blur effect).
    p.setPen(Qt.PenStyle.NoPen)
    for i in range(10, 0, -1):
        spread = i * 1.5
        alpha = int(20 * (1.0 - i / 11.0) ** 2) + 2
        p.setBrush(QColor(0, 0, 0, alpha))
        p.drawRoundedRect(rect.adjusted(-spread, -spread + 3, spread, spread + 4), radius + spread,
                          radius + spread)
    path = QPainterPath()
    path.addRoundedRect(rect, radius, radius)
    body = QLinearGradient(rect.topLeft(), rect.bottomLeft())
    body.setColorAt(0.0, QColor(*theme.ISLAND_TOP))
    body.setColorAt(1.0, QColor(*theme.ISLAND_BOTTOM))
    p.fillPath(path, QBrush(body))
    if tint_color:
        glow = QRadialGradient(QPointF(rect.right() - 30, rect.top() + 29), 150)
        glow.setColorAt(0.0, theme.qcolor(tint_color, 0.18))
        glow.setColorAt(1.0, theme.qcolor(tint_color, 0.0))
        p.fillPath(path, QBrush(glow))
    edge = QLinearGradient(rect.topLeft(), rect.bottomLeft())
    edge.setColorAt(0.0, QColor(255, 255, 255, 46 if hover and tint_color else 34))
    edge.setColorAt(0.5, QColor(255, 255, 255, 14))
    edge.setColorAt(1.0, QColor(255, 255, 255, 8))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(QBrush(edge), 1.0))
    p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), radius - 0.5, radius - 0.5)


def paint_meter(p: QPainter, pill: QRectF, color: str, *, state: str, muted: bool, level: float,
                phase: float, icon_font: QFont) -> None:
    """The level meter at the left end of the pill (the reading end in RTL).

    listening/speaking: bars follow the smoothed voice level; thinking/working:
    a slow wave; muted/error: an icon; otherwise flat dots."""
    cy = pill.top() + 29.0
    x0 = pill.left() + 24.0
    if muted or state == "error":
        p.setFont(icon_font)
        p.setPen(theme.qcolor(color, 0.9))
        glyph = theme.ICONS["mic_off" if muted else "warning"]
        p.drawText(QRectF(x0 - 4, cy - 10, 24, 20), int(Qt.AlignmentFlag.AlignCenter), glyph)
        return
    p.setPen(Qt.PenStyle.NoPen)
    for i, weight in enumerate(METER_WEIGHTS):
        if state in ("listening", "speaking"):
            wobble = 0.72 + 0.28 * math.sin(phase * 9.0 + i * 1.7)
            h = 3.0 + 17.0 * level * weight * wobble + 2.0 * weight
            alpha = 0.95
        elif state in ("thinking", "working"):
            h = 3.0 + 6.0 * (0.5 + 0.5 * math.sin(phase * 5.0 - i * 0.9))
            alpha = 0.85
        else:
            h, alpha = 3.0, 0.38
        p.setBrush(theme.qcolor(color, alpha))
        p.drawRoundedRect(QRectF(x0 + i * 6.0, cy - h / 2.0, 3.0, h), 1.5, 1.5)


def paint_progress(p: QPainter, pill: QRectF, prog: dict[str, Any], shown: float, phase: float) -> None:
    """Thin worker/tool progress line along the pill's bottom edge. RTL: it
    grows from the right; ``max == 0`` means indeterminate (a sweeping segment)."""
    track = QRectF(pill.left() + 30.0, pill.bottom() - 4.0, pill.width() - 60.0, 2.0)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(255, 255, 255, 18))
    p.drawRoundedRect(track, 1.0, 1.0)
    color = theme.DANGER if prog["done"] and prog["ok"] is False else \
        theme.SUCCESS if prog["done"] else theme.STATE_COLORS["working"]
    if prog["max"] or prog["done"]:
        width = max(4.0, track.width() * max(0.02, min(1.0, shown)))
        seg = QRectF(track.right() - width, track.top(), width, track.height())
    else:
        seg_w = track.width() * 0.28
        travel = (phase * 0.8) % 1.0
        seg = QRectF(track.right() - seg_w - travel * (track.width() - seg_w), track.top(), seg_w, track.height())
    grad = QLinearGradient(seg.topLeft(), seg.topRight())
    grad.setColorAt(0.0, theme.qcolor(color, 0.55))
    grad.setColorAt(1.0, theme.qcolor(color, 1.0))
    p.setBrush(QBrush(grad))
    p.drawRoundedRect(seg, 1.0, 1.0)


__all__ = ["device_scale", "paint_glass", "paint_meter", "paint_progress", "METER_WEIGHTS"]
