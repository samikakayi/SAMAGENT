"""Sorani command normalisation.

The user speaks naturally, so a command arrives as ordinary Kurdish rather than
fixed keywords: "بچۆ بۆ پێنج خولەکی" and "بچۆ ٥ خولەکی" and "بچۆ 5m" all mean the
same thing. This turns those into the structured intents the agent already
routes, without inventing meaning that was not said.

Nothing here translates Sorani into English for the model to guess at; it maps
recognised phrases onto intents SAM already implements, and returns None when
the phrase is not one of them.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

# Kurdish presentation forms and Arabic-Indic digits need folding before matching.
ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
# Single-character folds only; multi-character sequences are handled in normalize().
LETTER_FOLD = str.maketrans({"\u064a": "\u06cc", "\u0643": "\u06a9", "\ufefb": "\u0644\u0627"})

SPELLED_NUMBERS: dict[str, int] = {
    "یەک": 1, "یه‌ک": 1, "دوو": 2, "سێ": 3, "چوار": 4, "پێنج": 5, "شەش": 6,
    "حەوت": 7, "هەشت": 8, "نۆ": 9, "دە": 10, "پازدە": 15, "بیست": 20,
    "سی": 30, "چل": 40, "پەنجا": 45, "شەست": 60,
}

MINUTE_WORDS = ("خولەک", "خوله‌ک", "خولەکی", "دەقە", "min", "m")
HOUR_WORDS = ("کاتژمێر", "کاتژمير", "سەعات", "hour", "h")

# Spoken forms as the recogniser returns them, not only the written spellings:
# a spoken "وەستە" comes back as "وەستا", and both mean stop.
STOP_WORDS = ("وەستە", "وەستا", "بوەستە", "بوەستا", "ڕابگرە", "رابگرە", "ڕاوەستە", "ڕاوەستا",
              "بەسە", "بەسه", "stop", "cancel")
WAKE_WORDS = ("سام", "sam")

# "go to" as spoken and as recognised. "چووە"/"چوو" are how KurdishTTS returns
# an imperative "بچۆ", so leaving them out makes spoken navigation fail.
GO_WORDS = ("بچۆ", "بچو", "چووە", "چوو", "بڕۆ", "برۆ", "گەڕێوە", "go to", "goto", "switch")


@dataclass(slots=True)
class SoraniIntent:
    action: str
    arguments: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    matched: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"action": self.action, "arguments": self.arguments,
                "confidence": self.confidence, "matched": self.matched}


def normalize(text: str) -> str:
    """Fold digits, letter variants, and spacing so matching is stable."""
    if not text:
        return ""
    folded = unicodedata.normalize("NFKC", text)
    folded = folded.translate(ARABIC_INDIC).translate(LETTER_FOLD)
    # Zero-width joiner forms of the final "he" are written as a separate letter.
    folded = folded.replace("\u0647\u200c", "\u06d5").replace("\u200c", " ")
    folded = folded.replace("\u200f", "").replace("\u200e", "")
    folded = re.sub(r"[،؟!.,:;]+", " ", folded)
    return re.sub(r"\s+", " ", folded).strip()


# Recognised speech is noisy in ways written text is not. Across runs the same
# spoken command comes back as "شیکاری", "چی کاری" (split in two) or with a
# single letter swapped. Listing every variant would be endless and would still
# miss the next one, so keywords are matched by similarity instead.
FUZZY_THRESHOLD = 0.82
# Below this length a near-match is more likely to be a different word entirely.
FUZZY_MIN_LENGTH = 4


def _candidates(lowered: str) -> list[str]:
    """Tokens, plus adjacent pairs joined, so a split word can still match."""
    tokens = lowered.split()
    pairs = ["".join(tokens[index:index + 2]) for index in range(len(tokens) - 1)]
    return tokens + pairs


def fuzzy_contains(text: str, word: str, threshold: float = FUZZY_THRESHOLD) -> bool:
    """Whether `word` was probably said, allowing for recognition noise."""
    lowered = normalize(text).lower()
    target = normalize(word).lower()
    if not target:
        return False
    if target in lowered:
        return True
    # A multi-word phrase is matched literally; loosening it invites nonsense.
    if " " in target or len(target) < FUZZY_MIN_LENGTH:
        return False
    for candidate in _candidates(lowered):
        if difflib.SequenceMatcher(None, candidate, target).ratio() >= threshold:
            return True
        # A stem keyword may lead a longer inflected word.
        if len(candidate) > len(target):
            head = candidate[:len(target)]
            if difflib.SequenceMatcher(None, head, target).ratio() >= threshold:
                return True
    return False


def has_any(text: str, words: tuple[str, ...] | list[str], threshold: float = FUZZY_THRESHOLD) -> bool:
    return any(fuzzy_contains(text, word, threshold) for word in words)


def _number_in(text: str) -> int | None:
    digits = re.search(r"\b(\d{1,3})\b", text)
    if digits:
        return int(digits.group(1))
    # Longest first: "پازدە" (15) contains "دە" (10), so a shorter word must not
    # win just because it appears earlier in the table.
    for word, value in sorted(SPELLED_NUMBERS.items(), key=lambda item: -len(item[0])):
        if word in text:
            return value
    return None


def parse_timeframe(text: str) -> str | None:
    """Recognise a timeframe however it was said."""
    lowered = normalize(text).lower()
    direct = re.search(r"\b(\d{1,3})\s*(m|min|h|hour)\b", lowered)
    if direct:
        value, unit = int(direct.group(1)), direct.group(2)
        return f"{'H' if unit.startswith('h') else 'M'}{value}"
    if any(word in lowered for word in HOUR_WORDS):
        value = _number_in(lowered) or 1
        return f"H{value}"
    if any(word in lowered for word in MINUTE_WORDS):
        value = _number_in(lowered)
        if value:
            return f"M{value}"
    if "ڕۆژانە" in lowered or "رۆژانە" in lowered or "daily" in lowered:
        return "D1"
    if "هەفتانە" in lowered or "weekly" in lowered:
        return "W1"
    return None


SYMBOL_ALIASES: dict[str, tuple[str, ...]] = {
    "XAUUSD": ("زێڕ", "زير", "زێر", "طلا", "gold", "xauusd", "xau"),
    "XAGUSD": ("زیو", "زيو", "silver", "xagusd", "xag"),
    "EURUSD": ("یۆرۆ", "یورو", "euro", "eurusd"),
    "GBPUSD": ("پاوەند", "ستەرلینگ", "pound", "gbpusd"),
    "BTCUSD": ("بیتکۆین", "بیت کۆین", "bitcoin", "btcusd", "btc"),
    "ETHUSD": ("ئیسیریۆم", "ethereum", "ethusd", "eth"),
}

THEORY_ALIASES: dict[str, tuple[str, ...]] = {
    "snr": ("snr", "ساپۆرت", "ڕێزیستانس", "ریزیستانس", "پاڵپشت", "بەرگری"),
    "wyckoff": ("wyckoff", "وایکۆف", "وایکوف"),
    "ict": ("ict", "ئای سی تی"),
    "smc": ("smc", "سمارت مۆنی", "smart money"),
    "order_blocks": ("order block", "ئۆردەر بلۆک"),
    "fibonacci": ("fib", "فیبۆ", "فیبوناچی"),
    "elliott": ("elliott", "ئێلیۆت"),
    "harmonic": ("harmonic", "هارمۆنیک"),
    "volume_profile": ("volume profile", "پرۆفایلی قەبارە"),
}


def parse_symbol(text: str) -> str | None:
    lowered = normalize(text).lower()
    explicit = re.search(r"\b([A-Z]{3,6}USD|[A-Z]{6})\b", normalize(text))
    if explicit:
        return explicit.group(1).upper()
    for symbol, aliases in SYMBOL_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            return symbol
    return None


def parse_theories(text: str) -> list[str]:
    lowered = normalize(text).lower()
    found = [theory for theory, aliases in THEORY_ALIASES.items()
             if has_any(lowered, aliases)]
    return found


def is_stop_command(text: str) -> bool:
    # Stopping on request matters more than most commands, but a false stop is
    # also disruptive, so this is stricter than ordinary keyword matching.
    return has_any(text, STOP_WORDS, threshold=0.86)


def has_wake_word(text: str, wake_word: str = "سام") -> bool:
    lowered = normalize(text).lower()
    candidates = set(WAKE_WORDS) | {normalize(wake_word).lower()}
    return any(word and word in lowered for word in candidates)


def parse(text: str) -> SoraniIntent | None:
    """Map a spoken Sorani command onto one of SAM's existing actions."""
    if not text or not text.strip():
        return None
    lowered = normalize(text).lower()

    if is_stop_command(lowered):
        return SoraniIntent("stop", matched=text.strip(), confidence=1.0)

    # Opening or focusing the chart application.
    if has_any(lowered, ("ترەیدینگ ڤیو", "تریدینگ ڤیو", "tradingview", "چارت")) and \
       has_any(lowered, ("بکەرەوە", "بکه‌ره‌وه‌", "open", "بێنە")):
        return SoraniIntent("open_tradingview", matched=text.strip())

    # Clearing SAM's own drawings.
    # "بسڕەوە" on its own means "erase it", and on a chart there is nothing else
    # to erase, so the object word is not required. SAM only ever deletes its
    # own drawings, so a loose match here cannot touch the user's work.
    if has_any(lowered, ("بسڕەوە", "بسره‌وه‌", "بسڕە", "پاک بکەرەوە", "clear")):
        return SoraniIntent("clear_drawings", matched=text.strip())

    # Drawing support and resistance.
    if has_any(lowered, ("بکێشە", "بکێشه‌", "draw")):
        if has_any(lowered, ("tp", "تارگێت", "target")):
            return SoraniIntent("draw_targets", matched=text.strip())
        return SoraniIntent("draw_levels", {"theories": parse_theories(lowered) or ["snr"]},
                            matched=text.strip())

    # Placing stop and targets.
    if has_any(lowered, ("تارگێت", "target", "tp")) and \
       has_any(lowered, ("دابنێ", "دابنی", "set")):
        return SoraniIntent("draw_targets", matched=text.strip())
    if has_any(lowered, ("ستۆپ", "stop loss", "sl")) and "?" not in text:
        if has_any(lowered, ("کوێ", "چەند", "where")):
            return SoraniIntent("report_stop", matched=text.strip())

    # Monitoring a setup.
    if has_any(lowered, ("چاودێری", "چاودیری", "monitor")):
        return SoraniIntent("monitor_setup", matched=text.strip())

    # Hunting an entry.
    if has_any(lowered, ("ئینتری", "entry", "چوونە ژوورەوە", "چوونه‌ ژووره‌وه‌")):
        arguments: dict[str, Any] = {}
        timeframe = parse_timeframe(lowered)
        if timeframe:
            arguments["timeframe"] = timeframe
        symbol = parse_symbol(text)
        if symbol:
            arguments["symbol"] = symbol
        return SoraniIntent("find_entry", arguments, matched=text.strip())

    # Comparing theories.
    if has_any(lowered, ("بەراورد", "به‌راورد", "compare")):
        return SoraniIntent("compare_theories",
                            {"theories": parse_theories(lowered) or ["snr", "wyckoff", "ict"]},
                            matched=text.strip())

    # Changing the timeframe. "بچۆ" means "go to".
    if has_any(lowered, GO_WORDS):
        symbol = parse_symbol(text)
        timeframe = parse_timeframe(lowered)
        if timeframe and not symbol:
            return SoraniIntent("set_timeframe", {"timeframe": timeframe}, matched=text.strip())
        if symbol:
            return SoraniIntent("set_symbol", {"symbol": symbol}, matched=text.strip())

    # Running an analysis.
    if has_any(lowered, ("شیکاری", "شیکار", "analyz", "analys")):
        arguments = {}
        symbol = parse_symbol(text)
        if symbol:
            arguments["symbol"] = symbol
        theories = parse_theories(lowered)
        if theories:
            arguments["theories"] = theories
        timeframes = _all_timeframes(lowered)
        if timeframes:
            arguments["timeframes"] = timeframes
        return SoraniIntent("analyze", arguments, matched=text.strip())

    # A bare timeframe or symbol, said on its own, still means "switch to it".
    timeframe = parse_timeframe(lowered)
    if timeframe and any(word in lowered for word in ("ببینە", "ببینه‌", "پیشان", "show", "بۆ")):
        return SoraniIntent("set_timeframe", {"timeframe": timeframe}, matched=text.strip())
    symbol = parse_symbol(text)
    if symbol and not timeframe and len(lowered.split()) <= 3:
        return SoraniIntent("set_symbol", {"symbol": symbol}, matched=text.strip())
    return None


def _all_timeframes(text: str) -> list[str]:
    """Every timeframe mentioned, in the order they were said."""
    found: list[str] = []
    for piece in re.split(r"\s+(?:و|and)\s+|[,،]", text):
        timeframe = parse_timeframe(piece)
        if timeframe and timeframe not in found:
            found.append(timeframe)
    return found
