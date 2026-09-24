"""Design tokens (colours, radii, fonts) and the panel style sheet.

One dark theme. Colours are plain hex strings so non-Qt code (tests) can use
them; ``qcolor()`` converts. Contrast: body text #E7EBF3 on #0D1017 is ~15:1,
muted #98A2B6 on the same is ~7:1 (both above WCAG AA 4.5:1).

Fonts: nothing proprietary is bundled. The Sorani UI font is the first
installed family of setting ``ui.font_family`` ("Vazirmatn" by default),
"Vazirmatn", "Noto Naskh Arabic", "Noto Sans Arabic", "Segoe UI", "Tahoma".
Measured on this PC (2026-09-24, Pillow glyph check on the font files):
Segoe UI and Tahoma contain every Sorani letter (ە ێ ۆ ڕ ڵ ی ک گ چ پ ژ);
"Segoe UI Variable" does NOT (Latin only), so it is never used for Sorani text
(nor for Latin display text: see ``LATIN_DISPLAY_FONTS``).
Icons come from the system "Segoe Fluent Icons" font (Windows 11), falling back
to "Segoe MDL2 Assets" (same code points).
"""

from __future__ import annotations

from typing import Any, Iterable

# --- palette -------------------------------------------------------------------
BG = "#0B0E14"            # window background
SIDEBAR = "#0F131B"
SURFACE = "#131822"       # cards
SURFACE_2 = "#1A202C"     # inputs, hover
SURFACE_3 = "#222938"     # selected, pressed
BORDER = "#252D3C"
BORDER_SOFT = "#1C2330"
TEXT = "#E7EBF3"
TEXT_MUTED = "#98A2B6"
TEXT_FAINT = "#667085"
ACCENT = "#7C8CFF"        # SAM indigo
ACCENT_HOVER = "#909EFF"
ACCENT_SOFT = "#262C57"   # user bubbles, selected nav
CYAN = "#22D3EE"
SUCCESS = "#34D399"
WARNING = "#FBBF24"
DANGER = "#F87171"
INFO = "#60A5FA"

# Island glass (drawn with QPainter, alpha 0..255). Opaque on purpose: at
# alpha 246 the real on-screen grab (acceptance/ui_onscreen.py) showed the text
# of the window underneath ghosting through the caption; a layered window gets
# no DWM blur, so any translucency reads as a smudge, not as frosted glass.
ISLAND_TOP = (30, 34, 45, 255)
ISLAND_BOTTOM = (13, 15, 21, 255)
ISLAND_BORDER = (255, 255, 255, 26)
ISLAND_HIGHLIGHT = (255, 255, 255, 14)

# --- voice states ------------------------------------------------------------------
STATE_COLORS: dict[str, str] = {
    "idle": "#8B95FF",       # calm indigo: ready
    "sleeping": "#7A86A6",   # dimmer: conversation window closed
    "listening": "#2DD4BF",  # teal: mic open
    "thinking": "#B18CFF",   # violet
    "speaking": "#4FB8FF",   # sky blue
    "working": "#FBBF24",    # amber: tools / worker
    "error": "#F87171",      # red
    "muted": "#8A94A8",      # grey
}
# States that animate continuously (everything else is static -> ~0% idle CPU).
ANIMATED_STATES = frozenset({"listening", "thinking", "speaking", "working"})

STATUS_COLORS: dict[str, str] = {
    "ok": SUCCESS, "degraded": WARNING, "down": DANGER, "unconfigured": "#5B6477", "unknown": "#5B6477",
}

# --- geometry (logical px) ------------------------------------------------------------
PILL_COMPACT = (380, 58)
PILL_EXPANDED = (440, 94)
PILL_RADIUS = 29.0
SHADOW_MARGIN = (18, 8, 18, 24)      # left, top, right, bottom around the pill
CARD_WIDTH = 392
CARD_GAP = 8
RADIUS_CARD = 14
RADIUS_INPUT = 10

# --- fonts -----------------------------------------------------------------------------------
UI_FONT_FALLBACKS = ("Vazirmatn", "Noto Naskh Arabic", "Noto Sans Arabic", "Segoe UI", "Tahoma")
# Not "Segoe UI Variable Display": its first draw costs 310-350 ms (variable-font
# instancing) against 1-8 ms for "Segoe UI" (first draw in a fresh process, 3 runs,
# 2026-09-24), and that first draw sits on the island's start-up path.
LATIN_DISPLAY_FONTS = ("Segoe UI", "Arial")
ICON_FONTS = ("Segoe Fluent Icons", "Segoe MDL2 Assets")

# Segoe Fluent Icons / MDL2 code points (identical in both fonts).
ICONS: dict[str, str] = {
    "chat": "\ue8bd", "strategies": "\ue8f1", "monitor": "\uea8f", "activity": "\ue9d9",
    "settings": "\ue713", "send": "\ue724", "mic": "\ue720", "mic_off": "\uec54", "add": "\ue710",
    "close": "\ue711", "check": "\ue73e", "refresh": "\ue72c", "link": "\ue71b", "key": "\ue8d7",
    "shield": "\uea18", "chart": "\ue9d2", "gear": "\ue713", "warning": "\ue7ba", "delete": "\ue74d",
    "info": "\ue946", "voice": "\ue767", "stop": "\ue71a", "bolt": "\ue945",
}


def qcolor(value: str | tuple[int, ...], alpha: float | None = None) -> Any:
    """``QColor`` from a hex string or an (r, g, b[, a]) tuple; ``alpha`` 0..1."""
    from PySide6.QtGui import QColor

    color = QColor(*value) if isinstance(value, tuple) else QColor(value)
    if alpha is not None:
        color.setAlphaF(max(0.0, min(1.0, alpha)))
    return color


def state_color(state: str) -> str:
    return STATE_COLORS.get(state, STATE_COLORS["idle"])


_FAMILIES: set[str] | None = None


def available_families() -> set[str]:
    """Installed font families, read once per process (the first
    ``QFontDatabase.families()`` call enumerates ~400 fonts on this PC)."""
    global _FAMILIES
    if _FAMILIES is None:
        from PySide6.QtGui import QFontDatabase

        _FAMILIES = set(QFontDatabase.families())
    return _FAMILIES


def pick_families(preferred: Iterable[str | None], installed: set[str] | None = None) -> list[str]:
    """Installed families in preference order (duplicates removed)."""
    installed = available_families() if installed is None else installed
    out: list[str] = []
    for name in preferred:
        if name and name in installed and name not in out:
            out.append(name)
    return out


def ui_families(config_font: str | None = None) -> list[str]:
    """Font fallback list for Sorani UI text (``QFont.setFamilies``)."""
    families = pick_families([config_font, *UI_FONT_FALLBACKS])
    return families or ["Segoe UI"]


def ui_font(point_px: float = 13.0, weight: int = 400, config_font: str | None = None) -> Any:
    from PySide6.QtGui import QFont

    font = QFont()
    font.setFamilies(ui_families(config_font))
    font.setPixelSize(max(1, round(point_px)))
    font.setWeight(QFont.Weight(weight))
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    return font


def latin_font(point_px: float = 14.0, weight: int = 600) -> Any:
    from PySide6.QtGui import QFont

    font = QFont()
    font.setFamilies(pick_families(LATIN_DISPLAY_FONTS) or ["Segoe UI"])
    font.setPixelSize(max(1, round(point_px)))
    font.setWeight(QFont.Weight(weight))
    return font


def icon_font(point_px: float = 16.0) -> Any:
    from PySide6.QtGui import QFont

    font = QFont()
    font.setFamilies(pick_families(ICON_FONTS) or ["Segoe MDL2 Assets"])
    font.setPixelSize(max(1, round(point_px)))
    return font


def icon_family() -> str:
    families = pick_families(ICON_FONTS)
    return families[0] if families else "Segoe MDL2 Assets"


def panel_stylesheet(font_families: list[str]) -> str:
    """Qt style sheet for the panel (dark, rounded, generous spacing)."""
    fam = ", ".join(f'"{f}"' for f in font_families)
    icons = f'"{icon_family()}"'
    return f"""
    QWidget {{ color: {TEXT}; font-family: {fam}; font-size: 13px; }}
    #Panel, #PanelBody {{ background: {BG}; }}
    #Sidebar {{ background: {SIDEBAR}; border-left: 1px solid {BORDER_SOFT}; }}
    #NavButton {{ background: transparent; border: none; border-radius: 10px; padding: 0 12px;
                  text-align: right; color: {TEXT_MUTED}; font-size: 14px; min-height: 40px; }}
    #NavButton:hover {{ background: {SURFACE_2}; color: {TEXT}; }}
    #NavButton:checked {{ background: {ACCENT_SOFT}; color: {TEXT}; font-weight: 600; }}
    #PageTitle {{ font-size: 22px; font-weight: 600; color: {TEXT}; }}
    #PageSub {{ font-size: 13px; color: {TEXT_MUTED}; }}
    #Card {{ background: {SURFACE}; border: 1px solid {BORDER_SOFT}; border-radius: {RADIUS_CARD}px; }}
    #CardTitle {{ font-size: 15px; font-weight: 600; }}
    #Muted {{ color: {TEXT_MUTED}; }}
    #Faint {{ color: {TEXT_FAINT}; font-size: 12px; }}
    #Icon {{ font-family: {icons}; }}
    QLabel {{ background: transparent; }}
    QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 4px 2px 4px 2px; }}
    QScrollBar::handle:vertical {{ background: rgba(255, 255, 255, 0.09); border-radius: 3px; min-height: 36px; }}
    QScrollBar::handle:vertical:hover {{ background: rgba(255, 255, 255, 0.20); }}
    QScrollBar::handle:vertical:pressed {{ background: rgba(124, 140, 255, 0.55); }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    QScrollBar:horizontal {{ height: 0; }}
    QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QComboBox {{
        background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: {RADIUS_INPUT}px;
        padding: 7px 10px; selection-background-color: {ACCENT}; selection-color: #0B0E14; }}
    QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QSpinBox:focus, QComboBox:focus {{
        border: 1px solid {ACCENT}; }}
    QLineEdit:disabled, QComboBox:disabled {{ color: {TEXT_FAINT}; }}
    QComboBox::drop-down {{ border: none; width: 24px; }}
    QComboBox QAbstractItemView {{ background: {SURFACE_2}; border: 1px solid {BORDER};
        selection-background-color: {ACCENT_SOFT}; outline: none; padding: 4px; }}
    QSpinBox::up-button, QSpinBox::down-button {{ width: 0; border: none; }}
    QPushButton {{ background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: {RADIUS_INPUT}px;
        padding: 7px 14px; color: {TEXT}; }}
    QPushButton:hover {{ background: {SURFACE_3}; border-color: #34405A; }}
    QPushButton:pressed {{ background: #283044; }}
    QPushButton:disabled {{ color: {TEXT_FAINT}; background: {SURFACE}; border-color: {BORDER_SOFT}; }}
    QPushButton#Primary {{ background: {ACCENT}; border: none; color: #0B0E14; font-weight: 600; }}
    QPushButton#Primary:hover {{ background: {ACCENT_HOVER}; }}
    QPushButton#Primary:disabled {{ background: #3A4270; color: #8C93B8; }}
    QPushButton#Danger {{ background: transparent; border: 1px solid #5A2A33; color: {DANGER}; }}
    QPushButton#Danger:hover {{ background: #2A1519; }}
    QPushButton#Ghost {{ background: transparent; border: none; color: {TEXT_MUTED}; padding: 4px 8px; }}
    QPushButton#Ghost:hover {{ color: {TEXT}; background: {SURFACE_2}; }}
    QPushButton#Chip {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 16px;
        padding: 7px 14px; color: {TEXT_MUTED}; }}
    QPushButton#Chip:hover {{ color: {TEXT}; border-color: {ACCENT}; }}
    QPushButton#Chip:checked {{ background: {ACCENT_SOFT}; color: {TEXT}; border-color: {ACCENT}; }}
    QPushButton#IconButton {{ font-family: {icons}; font-size: 15px; padding: 0; min-width: 38px;
        min-height: 38px; max-width: 38px; max-height: 38px; border-radius: 19px; }}
    QPushButton#SendButton {{ font-family: {icons}; font-size: 15px; padding: 0; min-width: 40px;
        min-height: 40px; max-width: 40px; max-height: 40px; border-radius: 20px; background: {ACCENT};
        color: #0B0E14; border: none; }}
    QPushButton#SendButton:hover {{ background: {ACCENT_HOVER}; }}
    QPushButton#SendButton:disabled {{ background: #3A4270; color: #8C93B8; }}
    QListWidget, QTreeWidget, QTableWidget {{ background: transparent; border: none; outline: none; }}
    QListWidget::item {{ border-radius: 10px; padding: 0; margin: 2px 0; }}
    QListWidget::item:selected {{ background: {ACCENT_SOFT}; }}
    QListWidget::item:hover:!selected {{ background: {SURFACE_2}; }}
    QHeaderView::section {{ background: transparent; color: {TEXT_FAINT}; border: none;
        border-bottom: 1px solid {BORDER_SOFT}; padding: 6px 8px; font-size: 12px; }}
    QTableWidget {{ gridline-color: transparent; }}
    QTableWidget::item {{ padding: 4px 8px; border-bottom: 1px solid {BORDER_SOFT}; }}
    QTableWidget::item:selected {{ background: {SURFACE_2}; color: {TEXT}; }}
    QToolTip {{ background: {SURFACE_3}; color: {TEXT}; border: 1px solid {BORDER}; padding: 6px 8px;
        border-radius: 6px; }}
    QMenu {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; padding: 6px; }}
    QMenu::item {{ padding: 8px 18px; border-radius: 6px; }}
    QMenu::item:selected {{ background: {SURFACE_3}; }}
    QMenu::separator {{ height: 1px; background: {BORDER_SOFT}; margin: 5px 8px; }}
    QMenu::indicator {{ width: 14px; height: 14px; }}
    """


def menu_stylesheet(font_families: list[str]) -> str:
    """Stand-alone style for popup menus (island / tray menus have no panel parent)."""
    fam = ", ".join(f'"{f}"' for f in font_families)
    return f"""
    QMenu {{ background: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER}; border-radius: 10px;
             padding: 6px; font-family: {fam}; font-size: 13px; }}
    QMenu::item {{ padding: 8px 22px 8px 18px; border-radius: 6px; }}
    QMenu::item:selected {{ background: {SURFACE_3}; }}
    QMenu::item:disabled {{ color: {TEXT_FAINT}; }}
    QMenu::separator {{ height: 1px; background: {BORDER_SOFT}; margin: 5px 8px; }}
    """


__all__ = [n for n in dir() if n.isupper()] + [
    "qcolor", "state_color", "available_families", "pick_families", "ui_families", "ui_font", "latin_font",
    "icon_font", "icon_family", "panel_stylesheet", "menu_stylesheet"]
