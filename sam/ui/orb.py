"""SAM's avatar: a glowing, softly moving orb drawn with QPainter.

No images, no characters from other products: gradients only, plus an "S"
monogram. The same painter draws the island orb (animated), the panel header
orb and the tray icon (static pixmaps).

Motion model (all driven by ``phase`` = seconds and ``level`` 0..1):
- idle / sleeping / muted / error: static body, no ripples (the island stops its
  timer in these states, so the orb costs nothing when SAM is idle);
- listening / speaking: two ripples expand from the rim; glow and blob spread
  follow the smoothed mic / speaker level from ``LevelMeter``;
- thinking / working: a comet arc orbits the rim.

Cost: everything is cached as pixmaps per (colour, size, device pixel ratio):
the halo (per 1/12 level step), the base disc, the three colour blobs, the
ripple rings (24 radii), the gloss/rim/monogram overlay and the circle mask. A
frame only blits them (blobs at new positions, rings with a fading opacity).
Measured at 175 % scaling (300 frames each, 2026-09-24): 2.4 ms per orb with
everything painted live (monogram text 1.4 ms, halo 1.1 ms); ~1.0 ms with live
blob gradients (cProfile: the three gradient fills were 0.59 ms); on screen the
two stroked ripples cost ~0.23 ms each. With every layer cached:
0.25 ms listening / 0.36 ms thinking / 0.13 ms idle, and the whole listening
island ~20-26 % of one core at 52-58 fps, down from 27-40 %
(acceptance/ui_perf.py re-measures both).
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QFont, QIcon, QImage, QLinearGradient, QPainter,
                           QPen, QPixmap, QRadialGradient)

from . import theme

BLOBS = ((34.0, 0.85, 0.42, 0.78), (-38.0, -0.62, 0.46, 0.72), (0.0, 1.25, 0.24, 0.55))
LEVEL_STEPS = 12
RIPPLE_STEPS = 24         # ripple radii cached per orb colour (smooth at 60 fps: < 1 px per step)
_CACHE: "OrderedDict[tuple[Any, ...], Any]" = OrderedDict()
_CACHE_MAX = 256          # per colour: 13 halo levels + disc + 3 blobs + 24 rings; + overlays


def _cached(key: tuple[Any, ...], build: Any) -> Any:
    item = _CACHE.get(key)
    if item is None:
        item = build()
        _CACHE[key] = item
        if len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
    else:
        _CACHE.move_to_end(key)
    return item


def _shift_hue(color: QColor, degrees: float, sat: float = 1.0, light: float = 1.0) -> QColor:
    h, s, l, a = color.getHslF()
    h = (h if h >= 0 else 0.0) + degrees / 360.0
    return QColor.fromHslF(h % 1.0, max(0.0, min(1.0, s * sat)), max(0.0, min(1.0, l * light)), a)


def _canvas(size: float, dpr: float) -> tuple[QPixmap, QPainter]:
    pix = QPixmap(max(1, math.ceil(size * dpr)), max(1, math.ceil(size * dpr)))
    pix.setDevicePixelRatio(dpr)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    p.setPen(Qt.PenStyle.NoPen)
    return pix, p


def _halo(base: QColor, r: float, strength: float, level_q: int, dpr: float) -> QPixmap:
    level = level_q / LEVEL_STEPS
    outer = r * (1.62 + 0.55 * level)
    pix, p = _canvas(outer * 2, dpr)
    c = QPointF(outer, outer)
    grad = QRadialGradient(c, outer)
    halo, edge = QColor(base), QColor(base)
    halo.setAlphaF(min(1.0, strength + 0.36 * level))
    edge.setAlphaF(0.0)
    grad.setColorAt(max(0.0, r / outer * 0.82), halo)
    grad.setColorAt(1.0, edge)
    p.setBrush(QBrush(grad))
    p.drawEllipse(c, outer, outer)
    p.end()
    return pix


def _base_disc(base: QColor, r: float, dpr: float) -> QPixmap:
    pix, p = _canvas(r * 2, dpr)
    c = QPointF(r, r)
    grad = QRadialGradient(QPointF(r - 0.28 * r, r - 0.34 * r), r * 1.45)
    grad.setColorAt(0.0, base.lighter(135))
    grad.setColorAt(0.45, base.darker(135))
    grad.setColorAt(1.0, base.darker(330))
    p.setBrush(QBrush(grad))
    p.drawEllipse(c, r, r)
    p.end()
    return pix


def _blob(tint: QColor, radius: float, dpr: float) -> QPixmap:
    """One soft colour blob (radial fade to transparent). Cached: only its
    position changes between frames, and three gradient fills per frame were
    the orb's largest cost (0.59 of ~1.0 ms, cProfile over 300 frames)."""
    pix, p = _canvas(radius * 2, dpr)
    c = QPointF(radius, radius)
    clear = QColor(tint)
    clear.setAlphaF(0.0)
    grad = QRadialGradient(c, radius)
    grad.setColorAt(0.0, tint)
    grad.setColorAt(1.0, clear)
    p.setBrush(QBrush(grad))
    p.drawEllipse(c, radius, radius)
    p.end()
    return pix


def _ring(base: QColor, r: float, step: int, dpr: float) -> QPixmap:
    """A 1.3 px ripple ring at step ``step`` of its expansion (opaque colour;
    the caller fades it with ``setOpacity``)."""
    radius = r * (1.04 + 0.62 * step / (RIPPLE_STEPS - 1))
    half = radius + 1.5
    pix, p = _canvas(half * 2, dpr)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(QColor(base.red(), base.green(), base.blue()), 1.3))
    p.drawEllipse(QPointF(half, half), radius, radius)
    p.end()
    return pix


def _mask(r: float, dpr: float) -> QPixmap:
    pix, p = _canvas(r * 2, dpr)
    p.setBrush(QColor(0, 0, 0, 255))
    p.drawEllipse(QPointF(r, r), r, r)
    p.end()
    return pix


def _overlay(r: float, dpr: float, monogram: str) -> QPixmap:
    """Shade, gloss, rim and monogram: colour independent, fully static."""
    pix, p = _canvas(r * 2, dpr)
    c = QPointF(r, r)
    shade = QLinearGradient(r, 0, r, 2 * r)
    shade.setColorAt(0.0, QColor(0, 0, 0, 0))
    shade.setColorAt(1.0, QColor(0, 0, 0, 90))
    p.setBrush(QBrush(shade))
    p.drawEllipse(c, r, r)
    gloss = QRadialGradient(QPointF(r - 0.34 * r, r - 0.46 * r), r * 0.72)
    gloss.setColorAt(0.0, QColor(255, 255, 255, 105))
    gloss.setColorAt(1.0, QColor(255, 255, 255, 0))
    p.setBrush(QBrush(gloss))
    p.drawEllipse(c, r, r)
    rim = QLinearGradient(r, 0, r, 2 * r)
    rim.setColorAt(0.0, QColor(255, 255, 255, 80))
    rim.setColorAt(1.0, QColor(255, 255, 255, 12))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(QBrush(rim), max(0.8, r / 22.0)))
    p.drawEllipse(c, r - 0.4, r - 0.4)
    if monogram:
        font = theme.latin_font(r, 600)
        font.setPixelSize(max(6, round(r * 0.98)))
        p.setFont(font)
        rect = QRectF(0, 0, 2 * r, 2 * r)
        p.setPen(QColor(0, 0, 0, 70))
        p.drawText(rect.translated(0, max(0.6, r / 30.0)), int(Qt.AlignmentFlag.AlignCenter), monogram)
        p.setPen(QColor(255, 255, 255, 236))
        p.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), monogram)
    p.end()
    return pix


def _blob_tint(base: QColor, hue: float, muted: bool) -> QColor:
    tint = _shift_hue(base, hue, sat=1.1, light=1.08)
    tint.setAlphaF(0.78 if not muted else 0.35)
    return tint


def _dpr_of(p: QPainter) -> float:
    """Device scale of this paint: the device's ratio, or the device transform's
    scale when ``QWidget.render`` redirects into a hi-DPI pixmap (the device then
    still reports 1.0). Rounded to 1/4 so cache keys stay few."""
    device = p.device()
    try:
        ratio = float(device.devicePixelRatioF()) if device is not None else 1.0
    except AttributeError:
        ratio = 1.0
    ratio = max(ratio, abs(p.deviceTransform().m11()), 1.0)
    return round(ratio * 4) / 4


def paint_orb(p: QPainter, center: QPointF, r: float, color: QColor, *, state: str = "idle",
              level: float = 0.0, phase: float = 0.0, glow: bool = True, monogram: str = "S",
              monogram_font: QFont | None = None) -> None:
    """Draw the orb centred on ``center`` with body radius ``r``.

    ``monogram_font`` is accepted for compatibility; the monogram uses the
    Latin display font (it is cached with the overlay)."""
    level = max(0.0, min(1.0, level))
    muted = state in ("muted", "sleeping")
    base = _shift_hue(QColor(color), 0, sat=0.45) if muted else QColor(color)
    rgba = base.rgba()
    dpr = _dpr_of(p)
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    p.setPen(Qt.PenStyle.NoPen)

    # 1) soft outer glow, stronger with the voice level (cached per 1/12 level)
    if glow:
        strength = 0.20 if state in ("idle", "sleeping", "muted", "error") else 0.34
        level_q = round(level * LEVEL_STEPS)
        halo = _cached(("halo", rgba, round(r, 2), strength, level_q, dpr),
                       lambda: _halo(base, r, strength, level_q, dpr))
        size = halo.width() / dpr
        p.drawPixmap(QPointF(center.x() - size / 2, center.y() - size / 2), halo)

    # 2) ripples while listening / speaking: cached rings (RIPPLE_STEPS radii)
    # blitted with opacity; two stroked ellipses cost ~0.23 ms each on screen.
    if state in ("listening", "speaking"):
        opacity = p.opacity()
        for k in range(2):
            t = (phase * (0.55 + 0.5 * level) + k * 0.5) % 1.0
            step = min(RIPPLE_STEPS - 1, int(t * RIPPLE_STEPS))
            ring = _cached(("ring", rgba, round(r, 2), step, dpr), lambda step=step: _ring(base, r, step, dpr))
            size = ring.width() / dpr
            p.setOpacity(opacity * (1.0 - t) ** 1.6 * (0.30 + 0.45 * level))
            p.drawPixmap(QPointF(center.x() - size / 2, center.y() - size / 2), ring)
        p.setOpacity(opacity)

    # 3) body: base disc + moving colour blobs on a small scratch image, masked to the circle
    side = math.ceil(2 * r * dpr)
    scratch = QImage(side, side, QImage.Format.Format_ARGB32_Premultiplied)
    scratch.setDevicePixelRatio(dpr)
    scratch.fill(Qt.GlobalColor.transparent)
    sp = QPainter(scratch)
    sp.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    sp.setPen(Qt.PenStyle.NoPen)
    sp.drawPixmap(0, 0, _cached(("disc", rgba, round(r, 2), dpr), lambda: _base_disc(base, r, dpr)))
    sp.setCompositionMode(QPainter.CompositionMode.CompositionMode_Screen)
    spread = 1.0 + 0.35 * level
    for i, (hue, speed, dist, size) in enumerate(BLOBS):
        angle = phase * speed + i * 2.1
        radius = r * size
        blob = _cached(("blob", rgba, i, round(radius, 2), muted, dpr),
                       lambda hue=hue, radius=radius: _blob(_blob_tint(base, hue, muted), radius, dpr))
        sp.drawPixmap(QPointF(r + math.cos(angle) * r * dist * spread - radius,
                              r + math.sin(angle) * r * dist * spread - radius), blob)
    sp.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
    sp.drawPixmap(0, 0, _cached(("mask", round(r, 2), dpr), lambda: _mask(r, dpr)))
    sp.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
    sp.drawPixmap(0, 0, _cached(("over", round(r, 2), dpr, monogram), lambda: _overlay(r, dpr, monogram)))
    sp.end()
    p.drawImage(QPointF(center.x() - r, center.y() - r), scratch)

    # 4) orbiting comet while thinking / working
    if state in ("thinking", "working"):
        ring_r = r + max(3.0, r * 0.2)
        cone = QConicalGradient(center, -phase * 260.0)
        head = QColor(base).lighter(125)
        tail = QColor(base)
        tail.setAlphaF(0.0)
        cone.setColorAt(0.0, head)
        cone.setColorAt(0.72, tail)
        cone.setColorAt(1.0, tail)
        pen = QPen(QBrush(cone), max(1.6, r / 11.0))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(pen)
        p.drawEllipse(center, ring_r, ring_r)
    p.restore()


def orb_pixmap(size: int, color: str | None = None, *, state: str = "idle", monogram: str = "S",
               glow: bool = False, dpr: float = 1.0) -> QPixmap:
    """A static orb (tray icon, panel header, window icon)."""
    pix = QPixmap(round(size * dpr), round(size * dpr))
    pix.setDevicePixelRatio(dpr)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    r = size / 2.0 * (0.66 if glow else 0.94)
    paint_orb(p, QPointF(size / 2.0, size / 2.0), r, theme.qcolor(color or theme.state_color(state)),
              state=state, phase=0.9, glow=glow, monogram=monogram)
    p.end()
    return pix


def orb_icon(color: str | None = None, *, state: str = "idle") -> QIcon:
    """Multi-size window/tray icon."""
    icon = QIcon()
    for size in (16, 20, 24, 32, 40, 48, 64, 128, 256):
        icon.addPixmap(orb_pixmap(size, color, state=state, monogram="S" if size >= 20 else ""))
    return icon


__all__ = ["paint_orb", "orb_pixmap", "orb_icon"]
