"""Pure helpers for the TradingView bridge: spoken timeframes/symbols -> chart
values, colours, and validation of drawing requests from the model.

Builds on ``sam.trading.common`` (foundation-owned) and only adds what the chart
needs on top of it (measured gaps of ``normalize_timeframe`` on 2026-09-24):
indefinite Sorani forms (کاتژمێرێک، خولەکێک، ڕۆژێک، هەفتەیەک), half/quarter hours
(نیو کاتژمێر، چارەکێک), English number words, TradingView-only intervals
(2h/3h/45m), Sorani suffixes on symbols (زێڕی، زێڕەکە) and ``xau/usd``.

Everything here is deterministic and never touches the network or the chart.
"""

from __future__ import annotations

import math
import re
import time
from typing import Any

from ..textnorm import normalize_ckb
from .common import (COLORS, DRAWING_KINDS, DRAWING_POINTS, SYMBOL_ALIASES, canonical_symbol, normalize_timeframe,
                     to_tv_resolution)
from .symbols import instrument, resolve_instrument

# --- timeframes -------------------------------------------------------------------------------------------------
# Intervals TradingView offers without custom-interval permissions (minutes).
ALLOWED_MINUTES: tuple[int, ...] = (1, 3, 5, 15, 30, 45, 60, 120, 180, 240)
_UNITS: dict[str, str] = {}
for _unit, _forms in {
    "m": ("m", "min", "mins", "minute", "minutes", "خولەک", "دەقە", "دەقیقە"),
    "h": ("h", "hr", "hrs", "hour", "hours", "hourly", "کاتژمێر", "سەعات"),
    "d": ("d", "day", "days", "daily", "ڕۆژ", "ڕۆژانە"),
    "w": ("w", "week", "weeks", "weekly", "هەفتە", "هەفتانە"),
    "mn": ("month", "months", "monthly", "مانگ", "مانگانە"),
}.items():
    for _form in _forms:
        _UNITS[normalize_ckb(_form)] = _unit
_NUMBER_WORDS: dict[str, float] = {
    "one": 1, "a": 1, "an": 1, "two": 2, "three": 3, "four": 4, "five": 5, "ten": 10, "fifteen": 15,
    "thirty": 30, "fortyfive": 45, "sixty": 60, "half": 0.5, "quarter": 0.25,
    "یەک": 1, "دوو": 2, "سێ": 3, "چوار": 4, "پێنج": 5, "دە": 10, "پازدە": 15, "پانزە": 15, "سی": 30,
    "چلوپێنج": 45, "شەست": 60, "نیو": 0.5, "نیوە": 0.5, "چارەک": 0.25,
}
_SORANI_SUFFIXES = ("یەک", "ێک", "ەکە", "ەکان", "انە", "ی", "ە")
_RESOLUTION_LABELS_CKB: dict[str, str] = {
    "1": "یەک خولەک", "3": "سێ خولەک", "5": "پێنج خولەک", "15": "پازدە خولەک", "30": "نیو کاتژمێر",
    "45": "چل و پێنج خولەک", "60": "یەک کاتژمێر", "120": "دوو کاتژمێر", "180": "سێ کاتژمێر", "240": "چوار کاتژمێر",
    "1D": "ڕۆژانە", "D": "ڕۆژانە", "1W": "هەفتانە", "W": "هەفتانە", "1M": "مانگانە", "M": "مانگانە",
}


def _unit_of(token: str) -> tuple[str | None, float | None]:
    """'کاتژمێرێک' -> ('h', 1): the indefinite suffix means 'one'."""
    if token in _UNITS:
        return _UNITS[token], None
    for suffix in _SORANI_SUFFIXES:
        if token.endswith(suffix) and len(token) > len(suffix) + 1:
            base = token[: -len(suffix)]
            implied = 1.0 if suffix in ("یەک", "ێک") else None
            if base in _UNITS:
                return _UNITS[base], implied
            if base == "چارەک":          # چارەکێک = a quarter (of an hour)
                return "h", 0.25
    return None, None


def parse_tv_resolution(value: str | int | None) -> str | None:
    """Any spoken/typed timeframe -> TradingView resolution ('1', '15', '60', '240', '1D', '1W', '1M').

    '١٥ خولەک' -> '15'; 'کاتژمێرێک' / '1h' / 'H1' -> '60'; '٤ کاتژمێر' -> '240';
    'ڕۆژانە' / 'D' -> '1D'; 'هەفتانە' -> '1W'; 'دوو کاتژمێر' -> '120'. None when unknown.
    """
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    upper = raw.upper()
    if upper in ("D", "1D"):
        return "1D"
    if upper in ("W", "1W"):
        return "1W"
    if raw in ("M", "1M"):
        return "1M"
    canonical = normalize_timeframe(raw)
    if canonical:
        return to_tv_resolution(canonical)
    # TradingView-only intervals in compact form ('M3', '3m', 'H2', '45min'): the
    # fast path sends «٣ خولەکی» as 'M3' (live session 2026-09-25: "3 minutes").
    compact = re.fullmatch(r"(?:(m|h)(\d{1,3})|(\d{1,3})(m|min|mins|h|hr))", raw.lower().replace(" ", ""))
    if compact:
        unit = (compact.group(1) or compact.group(4) or "m")[0]
        count = int(compact.group(2) or compact.group(3))
        minutes = count * (60 if unit == "h" else 1)
        return str(minutes) if minutes in ALLOWED_MINUTES else None
    text = normalize_ckb(raw, strip_punct=True)
    text = text.replace("چل و پێنج", "چلوپێنج").replace("forty five", "fortyfive").replace("forty-five", "fortyfive")
    number: float | None = None
    unit: str | None = None
    for token in text.split():
        if token.isdigit():
            number = float(token)
            continue
        if token in _NUMBER_WORDS:
            number = _NUMBER_WORDS[token] if number is None else number
            continue
        found, implied = _unit_of(token)
        if found:
            unit = found
            if implied is not None and number is None:
                number = implied
    if unit in ("d", "w", "mn"):
        return {"d": "1D", "w": "1W", "mn": "1M"}[unit] if number in (None, 1) else None
    if number is None:
        return None
    minutes = number * (60 if unit == "h" else 1)
    if not float(minutes).is_integer() or int(minutes) not in ALLOWED_MINUTES:
        return None
    return str(int(minutes))


def same_resolution(a: str | None, b: str | None) -> bool:
    """'1D' == 'D', '60' == '60' (TradingView reports daily as '1D' or 'D')."""
    if a is None or b is None:
        return False
    return parse_tv_resolution(a) == parse_tv_resolution(b) or str(a).upper() == str(b).upper()


def resolution_label_ckb(resolution: str | None) -> str:
    """Sorani words for a resolution (spoken naturally by TTS)."""
    if not resolution:
        return "نەزانراو"
    key = str(resolution).strip()
    if key in _RESOLUTION_LABELS_CKB:
        return _RESOLUTION_LABELS_CKB[key]
    return f"{key} خولەک" if key.isdigit() else key


# --- symbols ---------------------------------------------------------------------------------------------------
# Default TradingView symbol per canonical instrument (setting trading.symbol_map {"XAUUSD": {"tv": ...}} wins).
TV_DEFAULT_SYMBOLS: dict[str, str] = {
    "XAUUSD": "OANDA:XAUUSD", "XAGUSD": "OANDA:XAGUSD", "BTCUSD": "BINANCE:BTCUSDT", "ETHUSD": "BINANCE:ETHUSDT",
    "EURUSD": "OANDA:EURUSD", "GBPUSD": "OANDA:GBPUSD", "USDJPY": "OANDA:USDJPY", "USOIL": "TVC:USOIL",
    "NAS100": "OANDA:NAS100USD", "US30": "OANDA:US30USD", "USDX": "TVC:DXY",
}
_EXTRA_ALIASES: dict[str, str] = {
    normalize_ckb(k): v for k, v in {
        "ئیتریۆم": "ETHUSD", "ئیسریۆم": "ETHUSD", "ئێسریۆم": "ETHUSD", "ئەسریۆم": "ETHUSD",
        "زێڕ و دۆلار": "XAUUSD", "gold spot": "XAUUSD", "spot gold": "XAUUSD", "xau usd": "XAUUSD",
        "یۆرۆ دۆلار": "EURUSD", "eur usd": "EURUSD", "داو جۆنز": "US30", "dow": "US30", "dow jones": "US30",
        "us30": "US30",
    }.items()
}
_TICKER = re.compile(r"^[A-Z0-9.!]{1,20}$")
_EXPLICIT = re.compile(r"^([A-Za-z0-9_]{1,24}):([A-Za-z0-9_.!/-]{1,32})$")


def canonical_request(text: str | None) -> str | None:
    """Spoken/typed instrument -> canonical ('زێڕی' -> 'XAUUSD', 'xau/usd' -> 'XAUUSD', 'AAPL' -> 'AAPL').
    None for unknown non-ticker words (never guess a symbol from free Sorani text) and for tickers
    shorter than 3 characters: the model sent tv_set_chart(symbol='z') for a timeframe-only request
    and TradingView resolved it to BATS:Z, a Nasdaq stock (review 2026-09-24)."""
    raw = (text or "").strip()
    if not raw:
        return None
    known = resolve_instrument(raw)
    if known:
        return known
    norm = normalize_ckb(raw)
    stripped = normalize_ckb(raw, strip_punct=True)
    for candidate in (norm, stripped, stripped.replace(" ", "")):
        if candidate in _EXTRA_ALIASES:
            return _EXTRA_ALIASES[candidate]
        if candidate in SYMBOL_ALIASES:
            return SYMBOL_ALIASES[candidate]
    tokens = stripped.split()
    hits: set[str] = set()
    for token in tokens:
        for form in [token] + [token[: -len(s)] for s in _SORANI_SUFFIXES if token.endswith(s) and len(token) > len(s) + 1]:
            target = SYMBOL_ALIASES.get(form) or _EXTRA_ALIASES.get(form)
            if target:
                hits.add(target)
                break
    if len(hits) == 1:
        return hits.pop()
    compact = re.sub(r"[\s/_-]", "", raw).upper()
    if _TICKER.match(compact) and len(compact) >= 3:
        return canonical_symbol(compact)
    return None


def instrument_key(symbol: str) -> str:
    """Same-instrument key: 'TVC:GOLD' ~ 'OANDA:XAUUSD' ~ 'XAUUSD'; 'BINANCE:BTCUSDT' ~ 'BTCUSD'."""
    return instrument(symbol)


def tv_symbol_for(request: str, *, current: str | None = None, overrides: dict[str, Any] | None = None,
                  learned: dict[str, Any] | None = None) -> tuple[str | None, str]:
    """Pick the TradingView symbol for ``request``.

    Returns (symbol, reason) with reason 'same' (the chart already shows this
    instrument: keep the user's own feed, e.g. TVC:GOLD for 'گۆڵد'), 'explicit'
    ('OANDA:XAUUSD' given), 'mapped' (settings or defaults), 'passthrough'
    (a plain ticker TradingView resolves itself) or 'unknown' (symbol None).
    'learned': the feed the user's own chart used last for this instrument (his gold
    chart is PEPPERSTONE:XAUUSD and his drawings live there, so 'گۆڵد' from bitcoin
    must not open OANDA:XAUUSD -- review 2026-09-24).
    """
    raw = (request or "").strip()
    explicit = _EXPLICIT.match(raw)
    if explicit:
        symbol = f"{explicit.group(1).upper()}:{explicit.group(2).upper()}"
        if current and symbol == current.upper():
            return current, "same"
        return symbol, "explicit"
    canonical = canonical_request(raw)
    if not canonical:
        return None, "unknown"
    if current and instrument_key(current) == instrument_key(canonical):
        return current, "same"
    entry = (overrides or {}).get(canonical)
    if isinstance(entry, dict) and isinstance(entry.get("tv"), str) and entry["tv"].strip():
        return entry["tv"].strip(), "mapped"
    seen = (learned or {}).get(canonical)
    if isinstance(seen, str) and _EXPLICIT.match(seen.strip()):
        return seen.strip(), "learned"
    if canonical in TV_DEFAULT_SYMBOLS:
        return TV_DEFAULT_SYMBOLS[canonical], "mapped"
    return canonical, "passthrough"


# --- colours ---------------------------------------------------------------------------------------------------
_NAMED_COLORS: dict[str, str] = {normalize_ckb(k): v for k, v in {
    "red": "#f23645", "green": "#089981", "blue": "#2962ff", "orange": "#ff9800", "yellow": "#fdd835",
    "purple": "#ab47bc", "white": "#ffffff", "black": "#131722", "gray": "#9598a1", "grey": "#9598a1",
    "سوور": "#f23645", "سور": "#f23645", "سەوز": "#089981", "شین": "#2962ff", "پرتەقاڵی": "#ff9800",
    "نارنجی": "#ff9800", "زەرد": "#fdd835", "مۆر": "#ab47bc", "وەنەوشەیی": "#ab47bc", "سپی": "#ffffff",
    "ڕەش": "#131722", "خۆڵەمێشی": "#9598a1",
}.items()}
_HEX = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")
_SEMANTIC_WORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("stop", ("stop", "sl", "stop loss", "ستۆپ", "ستۆپ لۆس")),
    ("target", ("target", "tp", "take profit", "ئامانج", "تارگێت", "تارگت")),
    ("entry", ("entry", "چوونەژوورەوە", "چوونە ژوورەوە", "ئینتری", "ئێنتری")),
    ("support", ("support", "demand", "پشتگیری", "ساپۆرت", "سەپۆرت", "داواکاری")),
    ("resistance", ("resistance", "supply", "بەرگری", "ڕێزیستانس", "ڕەزستەنس", "ڕێزستەنس")),
    ("liquidity", ("liquidity", "لیکویدیتی", "لیکوییدیتی")),
)
_KIND_DEFAULT_COLOR = {"horizontal_line": "info", "horizontal_ray": "info", "trend_line": "entry",
                       "rectangle": "zone", "fib_retracement": "info", "text": "info", "arrow_up": "support",
                       "arrow_down": "resistance", "long_position": "entry", "short_position": "entry"}


def parse_color(value: Any) -> str | None:
    """'#26a69a' / '26a69a' / '#fa0' / 'support' / 'red' / 'سەوز' -> '#rrggbb' (None if unknown)."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = normalize_ckb(value.strip())
    if text in COLORS:
        return COLORS[text]
    if text in _NAMED_COLORS:
        return _NAMED_COLORS[text]
    match = _HEX.match(value.strip())
    if not match:
        return None
    digits = match.group(1).lower()
    if len(digits) == 3:
        digits = "".join(c * 2 for c in digits)
    return "#" + digits[:6]           # TradingView wants #rrggbb; alpha is applied per kind


def rgba(color: str, alpha: float) -> str:
    digits = color.lstrip("#")
    r, g, b = (int(digits[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha:g})"


def semantic_role(text: str) -> str | None:
    """'هێڵی پشتگیری' -> 'support' (chooses the default colour of a labelled drawing)."""
    norm = normalize_ckb(text or "", strip_punct=True)
    if not norm:
        return None
    tokens = set(norm.split())
    for role, keywords in _SEMANTIC_WORDS:
        for keyword in keywords:
            key = normalize_ckb(keyword)
            if key in tokens or (len(key) >= 5 and key in norm):
                return role
    return None


# --- drawing requests --------------------------------------------------------------------------------------------
KIND_ALIASES: dict[str, str] = {
    "hline": "horizontal_line", "horizontal": "horizontal_line", "level": "horizontal_line",
    "line": "horizontal_line", "ray": "horizontal_ray", "hray": "horizontal_ray", "trendline": "trend_line",
    "trend": "trend_line", "box": "rectangle", "zone": "rectangle", "rect": "rectangle", "fib": "fib_retracement",
    "fibonacci": "fib_retracement", "label": "text", "note": "text", "long": "long_position",
    "short": "short_position", "up_arrow": "arrow_up", "down_arrow": "arrow_down",
}
KIND_LABELS_CKB: dict[str, str] = {
    "horizontal_line": "هێڵی ئاسۆیی", "horizontal_ray": "تیشکی ئاسۆیی", "trend_line": "هێڵی ترێند",
    "rectangle": "لاکێشە", "fib_retracement": "فیبۆناچی", "text": "نووسین", "arrow_up": "تیری سەرەوە",
    "arrow_down": "تیری خوارەوە", "long_position": "پۆزیشنی لۆنگ", "short_position": "پۆزیشنی شۆرت",
}
MAX_ITEMS = 30
MAX_TEXT = 60
DEFAULT_SPAN_BARS = 50          # timeless 2-point drawings span the last 50 bars
_LINE_STYLES = {"solid": 0, "dotted": 1, "dashed": 2, "large_dashed": 3}
_CONTROL = re.compile(r"[\x00-\x1f\x7f‪-‮⁦-⁩]")
_TAG = re.compile(r"[^\w:.\-]")
_TIME_MIN = 946_684_800          # 2000-01-01


def clean_text(value: Any) -> str:
    """Label shown on the user's chart: no control/bidi-override characters, <= 60 chars."""
    if value is None:
        return ""
    text = _CONTROL.sub(" ", str(value))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_TEXT]


def clean_tag(value: Any, default: str = "user-request") -> str:
    tag = _TAG.sub("", str(value or "").strip())[:40]
    return tag or default


def normalize_kind(value: Any) -> str | None:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    text = KIND_ALIASES.get(text, text)
    return text if text in DRAWING_KINDS else None


def _point(raw: Any, index: int, last_price: float | None, now: float) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        raw = {"price": raw}
    if not isinstance(raw, dict):
        return None, f"point {index + 1} must be an object with a price"
    try:
        price = float(raw.get("price"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None, f"point {index + 1} needs a numeric price"
    if not math.isfinite(price):
        return None, f"point {index + 1} has an invalid price"
    if last_price and last_price > 0 and not (last_price / 10 <= abs(price) <= last_price * 10):
        return None, f"price {price:g} is far from the chart price {last_price:g}"
    point: dict[str, Any] = {"price": price}
    if raw.get("time") not in (None, ""):
        try:
            stamp = int(float(raw["time"]))
        except (TypeError, ValueError):
            return None, f"point {index + 1} has an invalid time"
        if stamp > 100_000_000_000:          # milliseconds -> seconds
            stamp //= 1000
        if not (_TIME_MIN <= stamp <= now + 5 * 365 * 86400):
            return None, f"point {index + 1} time is out of range"
        point["time"] = stamp
    elif raw.get("bars_ago") not in (None, ""):
        try:
            ago = int(float(raw["bars_ago"]))
        except (TypeError, ValueError):
            return None, f"point {index + 1} has an invalid bars_ago"
        if not 0 <= ago <= 100_000:
            return None, f"point {index + 1} bars_ago is out of range"
        point["bars_ago"] = ago
    return point, None


def _style(style: Any) -> dict[str, Any]:
    style = style if isinstance(style, dict) else {}
    out: dict[str, Any] = {}
    try:
        out["width"] = min(4, max(1, int(style.get("width", 0)))) if style.get("width") is not None else None
    except (TypeError, ValueError):
        out["width"] = None
    out["line_style"] = _LINE_STYLES.get(str(style.get("line_style", "")).lower())
    out["lock"] = bool(style.get("lock", False))
    out["extend_right"] = bool(style.get("extend_right", False))
    out["extend_left"] = bool(style.get("extend_left", False))
    valign = str(style.get("label_valign", "")).lower()
    out["label_valign"] = valign if valign in ("top", "middle", "bottom") else "top"
    try:
        out["font_size"] = min(40, max(8, int(style["font_size"]))) if style.get("font_size") is not None else None
    except (TypeError, ValueError):
        out["font_size"] = None
    return out


def build_overrides(kind: str, color: str, text: str, style: dict[str, Any]) -> dict[str, Any]:
    """TradingView property overrides per kind (property names read from the live
    tools' getProperties() on 3.4.1, 2026-09-24)."""
    width = style.get("width")
    line_style = style.get("line_style")
    font = style.get("font_size")
    if kind in ("horizontal_line", "horizontal_ray"):
        out = {"linecolor": color, "linewidth": width or 2, "showPrice": True, "textcolor": color,
               "horzLabelsAlign": "right", "vertLabelsAlign": "bottom", "fontsize": font or 12}
    elif kind == "trend_line":
        out = {"linecolor": color, "linewidth": width or 2, "textcolor": color, "extendLeft": style["extend_left"],
               "extendRight": style["extend_right"], "fontsize": font or 12}
    elif kind == "rectangle":
        out = {"color": color, "backgroundColor": rgba(color, 0.15), "fillBackground": True, "transparency": 85,
               "linewidth": width or 1, "textColor": color, "extendRight": style["extend_right"],
               "extendLeft": style["extend_left"], "fontSize": font or 12, "horzLabelsAlign": "left",
               "vertLabelsAlign": style.get("label_valign", "top")}
    elif kind == "fib_retracement":
        out = {"showCoeffs": True, "showPrices": True, "fillBackground": True, "transparency": 85,
               "extendLines": style["extend_right"]}
    elif kind == "text":
        out = {"color": color, "fontsize": font or 14, "bold": False}
    elif kind in ("arrow_up", "arrow_down"):
        out = {"arrowColor": color, "color": color, "fontsize": font or 12, "showLabel": bool(text)}
    else:  # long_position / short_position: stop/target ticks are computed in the page
        out = {"linewidth": width or 1}
    if line_style is not None and kind in ("horizontal_line", "horizontal_ray", "trend_line", "rectangle"):
        out["linestyle"] = line_style
    if text and kind in ("horizontal_line", "horizontal_ray", "trend_line", "rectangle"):
        out["text"] = text
    return out


def normalize_item(item: Any, *, last_price: float | None = None, now: float | None = None
                   ) -> tuple[dict[str, Any] | None, str | None]:
    """Validate one drawing request -> (page spec, None) or (None, English error).

    Page spec: {kind, points [{price, time?|bars_ago?}], text, lock, overrides,
    position? {stop, target}, color, role}. Missing times resolve in the page
    (time > bars_ago > last bar); a 2-point drawing without any time spans the
    last DEFAULT_SPAN_BARS bars instead of collapsing onto one bar.
    """
    now = time.time() if now is None else now
    if not isinstance(item, dict):
        return None, "each item must be an object"
    kind = normalize_kind(item.get("kind"))
    if kind is None:
        return None, f"unknown kind {str(item.get('kind'))[:30]!r}; use one of {', '.join(DRAWING_KINDS)}"
    raw_points = item.get("points")
    if isinstance(raw_points, dict) or isinstance(raw_points, (int, float)):
        raw_points = [raw_points]
    if not isinstance(raw_points, list) or not raw_points:
        return None, f"{kind} needs points"
    need = DRAWING_POINTS[kind]
    if len(raw_points) != need:
        what = "entry, stop and target" if need == 3 else f"{need} point{'s' if need > 1 else ''}"
        return None, f"{kind} needs exactly {what} (got {len(raw_points)})"
    points: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_points):
        point, error = _point(raw, index, last_price, now)
        if error:
            return None, f"{kind}: {error}"
        points.append(point)  # type: ignore[arg-type]
    text = clean_text(item.get("text"))
    style = _style(item.get("style"))
    spec: dict[str, Any] = {"kind": kind, "text": "", "lock": style["lock"]}
    if kind in ("long_position", "short_position"):
        entry, stop, target = (p["price"] for p in points)
        if kind == "long_position" and not stop < entry < target:
            return None, "long_position needs stop < entry < target"
        if kind == "short_position" and not target < entry < stop:
            return None, "short_position needs target < entry < stop"
        first = {k: v for k, v in points[0].items() if k in ("time", "bars_ago")}
        spec["points"] = [{"price": entry, **first}]
        spec["position"] = {"stop": stop, "target": target}
        text = ""
    else:
        if need == 2 and all("time" not in p and "bars_ago" not in p for p in points):
            points[0]["bars_ago"] = DEFAULT_SPAN_BARS
            points[1]["bars_ago"] = 0
        spec["points"] = points
    role = semantic_role(text) or ("support" if kind == "arrow_up" else "resistance" if kind == "arrow_down" else None)
    color = parse_color(item.get("color")) or COLORS[role or _KIND_DEFAULT_COLOR[kind]]
    spec.update({"text": text, "color": color, "role": role or "",
                 "overrides": build_overrides(kind, color, text, style)})
    return spec, None


def kinds_summary_ckb(kinds: list[str]) -> str:
    """['horizontal_line', 'horizontal_line', 'trend_line'] -> '2 هێڵی ئاسۆیی، 1 هێڵی ترێند'."""
    counts: dict[str, int] = {}
    for kind in kinds:
        counts[kind] = counts.get(kind, 0) + 1
    return "، ".join(f"{n} {KIND_LABELS_CKB.get(k, k)}" for k, n in counts.items())


__all__ = ["parse_tv_resolution", "same_resolution", "resolution_label_ckb", "TV_DEFAULT_SYMBOLS",
           "canonical_request", "instrument_key", "tv_symbol_for", "parse_color", "rgba", "semantic_role",
           "normalize_kind", "normalize_item", "build_overrides", "clean_text", "clean_tag", "kinds_summary_ckb",
           "KIND_LABELS_CKB", "MAX_ITEMS", "ALLOWED_MINUTES"]
