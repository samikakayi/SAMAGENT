"""Vocabulary shared by the chart bridge, the MT5 feed and the engine.

Owned by the foundation stage; builders import from here instead of defining
their own copies (report any needed addition). Times are always UTC unix
seconds: TradingView bars are UTC; MT5 bars/ticks are broker server time and
the MT5 feed subtracts the per-connection broker offset
(round((tick.time - time.time()) / 900) * 900, +10800 s measured 2026-09-24)
before handing bars out.
"""

from __future__ import annotations

import re
from typing import Literal, TypedDict

from ..textnorm import normalize_ckb


class Bar(TypedDict):
    time: int          # bar open time, UTC unix seconds
    open: float
    high: float
    low: float
    close: float
    volume: float      # tick volume for MT5 FX/CFD symbols; real volume when the feed has it


class ChartPoint(TypedDict, total=False):
    time: int          # UTC unix seconds; omitted = resolved by the bridge (e.g. last bar)
    price: float
    bars_ago: int      # alternative to time: N bars before the last bar


DrawingKind = Literal["horizontal_line", "horizontal_ray", "trend_line", "rectangle", "fib_retracement",
                      "text", "arrow_up", "arrow_down", "long_position", "short_position"]
DRAWING_KINDS: tuple[str, ...] = ("horizontal_line", "horizontal_ray", "trend_line", "rectangle",
                                  "fib_retracement", "text", "arrow_up", "arrow_down", "long_position",
                                  "short_position")
# How many points each kind needs (long/short position: entry, stop, target).
DRAWING_POINTS: dict[str, int] = {"horizontal_line": 1, "horizontal_ray": 1, "trend_line": 2, "rectangle": 2,
                                  "fib_retracement": 2, "text": 1, "arrow_up": 1, "arrow_down": 1,
                                  "long_position": 3, "short_position": 3}
# Semantic colours so every module draws the same meaning the same way.
COLORS: dict[str, str] = {"support": "#26a69a", "resistance": "#ef5350", "entry": "#2962ff", "stop": "#f23645",
                          "target": "#089981", "zone": "#f5a623", "info": "#9598a1", "liquidity": "#ab47bc"}

TIMEFRAMES: tuple[str, ...] = ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1")
TIMEFRAME_SECONDS: dict[str, int] = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600, "H4": 14400,
                                     "D1": 86400, "W1": 604800, "MN1": 2592000}
_TV_RESOLUTION: dict[str, str] = {"M1": "1", "M5": "5", "M15": "15", "M30": "30", "H1": "60", "H4": "240",
                                  "D1": "1D", "W1": "1W", "MN1": "1M"}
_SORANI_NUMBERS: dict[str, int] = {
    "یەک": 1, "دوو": 2, "سێ": 3, "چوار": 4, "پێنج": 5, "شەش": 6, "دە": 10, "پازدە": 15, "پانزە": 15,
    "بیست": 20, "سی": 30, "سیی": 30,
}
_UNIT_WORDS: dict[str, str] = {}
for _unit, _words in {
    "m": ("m", "min", "mins", "minute", "minutes", "خولەک", "خولەکی", "خولەکە", "دەقیقە", "دەقە"),
    "h": ("h", "hr", "hour", "hours", "hourly", "سەعات", "سەعاتی", "سەعاتە", "کاتژمێر", "کاتژمێری"),
    "d": ("d", "day", "days", "daily", "ڕۆژ", "ڕۆژی", "ڕۆژانە"),
    "w": ("w", "week", "weeks", "weekly", "هەفتە", "هەفتەی", "هەفتانە"),
    "mn": ("mn", "month", "months", "monthly", "مانگ", "مانگی", "مانگانە"),
}.items():
    for _word in _words:
        _UNIT_WORDS[normalize_ckb(_word)] = _unit
_UNIT_PREFIX = {"m": "M", "h": "H", "d": "D", "w": "W", "mn": "MN"}


def normalize_timeframe(value: str | int) -> str | None:
    """'15', '15m', 'M15', 'm15', '1h', '60', '240', '4H', 'D', '1D', 'W',
    '١٥ خولەک', 'چوار سەعات', 'ڕۆژانە' ... -> canonical 'M15', 'H4', 'D1'.
    A bare number is minutes (TradingView convention). Case matters only for
    TradingView's '1M'/'M' (= one month). Returns None when unrecognised."""
    raw = str(value).strip()
    if raw in ("M", "1M", "MN", "MN1"):
        return "MN1"
    text = normalize_ckb(raw, strip_punct=True)
    if not text:
        return None
    compact = text.replace(" ", "")
    if compact.upper() in TIMEFRAMES:
        return compact.upper()
    number: int | None = None
    unit: str | None = None
    match = re.fullmatch(r"(mn|[mhdw])(\d+)", compact) or None
    if match:
        unit, number = match.group(1), int(match.group(2))
    else:
        match = re.fullmatch(r"(\d+)([a-z]*)", compact)
        if match:
            number = int(match.group(1))
            unit = _UNIT_WORDS.get(match.group(2)) if match.group(2) else None
            if match.group(2) and unit is None:
                return None
        else:
            for token in text.split():
                if token.isdigit():
                    number = int(token)
                elif token in _SORANI_NUMBERS:
                    number = _SORANI_NUMBERS[token]
                elif token in _UNIT_WORDS:
                    unit = _UNIT_WORDS[token]
    if number is None and unit in ("d", "w", "mn"):
        number = 1
    if number is None:
        return None
    if unit is None:
        unit = "m"
        if number >= 60 and number % 60 == 0:
            number, unit = number // 60, "h"
            if number == 24:
                number, unit = 1, "d"
    if unit == "h" and number == 24:
        number, unit = 1, "d"
    candidate = f"{_UNIT_PREFIX[unit]}{number}"
    return candidate if candidate in TIMEFRAMES else None


def to_tv_resolution(timeframe: str) -> str | None:
    """Canonical timeframe -> TradingView ``setResolution`` string."""
    canonical = normalize_timeframe(timeframe)
    return _TV_RESOLUTION.get(canonical) if canonical else None


def from_tv_resolution(resolution: str) -> str | None:
    """TradingView ``resolution()`` ('15', '60', '1D', 'D', 'W', '1M') -> canonical."""
    value = str(resolution).strip().upper()
    mapping = {"D": "D1", "1D": "D1", "W": "W1", "1W": "W1", "M": "MN1", "1M": "MN1"}
    if value in mapping:
        return mapping[value]
    if value.isdigit():
        return normalize_timeframe(value)
    return normalize_timeframe(value)


# Spoken/typed names -> canonical symbol. Broker names (XAUUSD.m.e ...) are the
# MT5 feed's job; TradingView prefixes (OANDA:, TVC:) are stripped/mapped here.
SYMBOL_ALIASES: dict[str, str] = {
    "زێڕ": "XAUUSD", "زیڕ": "XAUUSD", "ئاڵتون": "XAUUSD", "ئاڵتوون": "XAUUSD", "ئالتون": "XAUUSD",
    "گۆڵد": "XAUUSD", "گۆلد": "XAUUSD", "گولد": "XAUUSD", "gold": "XAUUSD", "xau": "XAUUSD", "xauusd": "XAUUSD",
    # Latin transliterations of زێڕ that models send as a "symbol". Measured
    # 2026-09-24 (voice live check): Groq gpt-oss-20b called get_price("ZAR") for
    # "نرخی زێڕ", MT5 matched EURZAR and the turn cost an extra model round. A bare
    # "ZAR" is not a tradable instrument (the rand trades as USDZAR/EURZAR).
    "zar": "XAUUSD", "zer": "XAUUSD", "zeer": "XAUUSD", "zêr": "XAUUSD", "zir": "XAUUSD", "zhir": "XAUUSD",
    "زیو": "XAGUSD", "silver": "XAGUSD", "xagusd": "XAGUSD",
    "بیتکۆین": "BTCUSD", "بیتکوین": "BTCUSD", "bitcoin": "BTCUSD", "btc": "BTCUSD", "btcusd": "BTCUSD",
    "ئیسریوم": "ETHUSD", "ئیتریوم": "ETHUSD", "ethereum": "ETHUSD", "eth": "ETHUSD", "ethusd": "ETHUSD",
    "یۆرۆ": "EURUSD", "euro": "EURUSD", "eurusd": "EURUSD", "پاوەند": "GBPUSD", "gbpusd": "GBPUSD",
    "ین": "USDJPY", "usdjpy": "USDJPY", "نەوت": "USOIL", "oil": "USOIL", "wti": "USOIL",
    "ناسداک": "NAS100", "nasdaq": "NAS100", "nas100": "NAS100", "us100": "NAS100",
    "دۆلار ئیندێکس": "USDX", "dxy": "USDX", "usdx": "USDX", "dollar index": "USDX",
}
# Any other Latin spelling of زێڕ: the integration smoke (2026-09-24) saw Groq
# gpt-oss-20b call get_price("ZEUR") for "نرخی زێڕ چەندە؟" after "ZAR" was mapped.
_GOLD_LATIN = re.compile(r"^zh?[aeêiîuûy]{1,3}r{1,2}$")
TV_SYMBOL_EQUIVALENTS: dict[str, str] = {"GOLD": "XAUUSD", "SILVER": "XAGUSD", "DXY": "USDX", "USOIL": "USOIL",
                                         "NQ1!": "NAS100", "NDX": "NAS100", "US100": "NAS100",
                                         # the TradingView defaults SAM itself selects (review 2026-09-24:
                                         # BINANCE:BTCUSDT / OANDA:NAS100USD never matched BTCUSD / NAS100,
                                         # so analyze_market silently skipped drawing on its own chart)
                                         "BTCUSDT": "BTCUSD", "ETHUSDT": "ETHUSD", "NAS100USD": "NAS100",
                                         "US30USD": "US30", "SPX500USD": "SPX500"}


def canonical_symbol(value: str) -> str:
    """'زێڕ' / 'gold' / 'OANDA:XAUUSD' / 'TVC:GOLD' / 'xauusd' -> 'XAUUSD'.
    Unknown names are upper-cased with the exchange prefix removed."""
    text = normalize_ckb(value or "", strip_punct=False).strip()
    if not text:
        return ""
    if text in SYMBOL_ALIASES:
        return SYMBOL_ALIASES[text]
    words = normalize_ckb(text, strip_punct=True)
    if words in SYMBOL_ALIASES:
        return SYMBOL_ALIASES[words]
    if _GOLD_LATIN.match(words):
        return "XAUUSD"
    bare = text.split(":", 1)[-1].upper()
    if bare in TV_SYMBOL_EQUIVALENTS:
        return TV_SYMBOL_EQUIVALENTS[bare]
    for suffix in (".M.E", ".CRP", ".M", ".E", ".PRO", ".RAW"):
        if bare.endswith(suffix):
            bare = bare[: -len(suffix)]
            break
    return bare


__all__ = ["Bar", "ChartPoint", "DrawingKind", "DRAWING_KINDS", "DRAWING_POINTS", "COLORS", "TIMEFRAMES",
           "TIMEFRAME_SECONDS", "normalize_timeframe", "to_tv_resolution", "from_tv_resolution",
           "SYMBOL_ALIASES", "canonical_symbol"]
