"""Numbers as Sorani words for text-to-speech (the panel keeps the digits).

The design asks for "numbers spoken naturally". Prices reached TTS as "4268.76"
and whether KurdishTTS reads Western-digit decimals as natural Sorani was never
verified (repair review 2026-09-24). Whole numbers it reads itself (measured,
see ``verbalize_numbers``); decimals and percentages are turned into words here:
4270 -> «چوار هەزار و دووسەد و حەفتا», 4268.76 -> «... و هەشت پۆینت حەفتا و شەش»,
1.0854 -> «یەک پۆینت سفر هەشت پێنج چوار», 80% -> «لەسەدا هەشتا».

Left alone: digits glued to letters (M15, H1, XAUUSD, ٢٤ی), clock times
(14:35) and anything longer than 12 digits (ids, not quantities).
"""

from __future__ import annotations

import re

_ONES = ("سفر", "یەک", "دوو", "سێ", "چوار", "پێنج", "شەش", "حەوت", "هەشت", "نۆ")
_TEENS = ("دە", "یازدە", "دوازدە", "سێزدە", "چواردە", "پازدە", "شازدە", "حەڤدە", "هەژدە", "نۆزدە")
_TENS = ("", "", "بیست", "سی", "چل", "پەنجا", "شەست", "حەفتا", "هەشتا", "نەوەد")
_HUNDREDS = ("", "سەد", "دووسەد", "سێسەد", "چوارسەد", "پێنجسەد", "شەشسەد", "حەوتسەد", "هەشتسەد", "نۆسەد")
_SCALES = ((10 ** 9, "ملیار"), (10 ** 6, "ملیۆن"), (1000, "هەزار"))
_POINT = "پۆینت"
_EASTERN = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫", "01234567890123456789.")
_NUMBER = re.compile(r"(?<![\w:.,])(\d{1,12}(?:\.\d{1,6})?)(\s*%)?(?![\w:%]|\.\d)")


def _below_thousand(n: int) -> list[str]:
    parts: list[str] = []
    hundreds, rest = divmod(n, 100)
    if hundreds:
        parts.append(_HUNDREDS[hundreds])
    if 10 <= rest < 20:
        parts.append(_TEENS[rest - 10])
    else:
        tens, ones = divmod(rest, 10)
        if tens:
            parts.append(_TENS[tens])
        if ones:
            parts.append(_ONES[ones])
    return parts


def integer_words(n: int) -> str:
    """Sorani words for a whole number (0 <= n < 10**12)."""
    if n == 0:
        return _ONES[0]
    parts: list[str] = []
    for value, word in _SCALES:
        count, n = divmod(n, value)
        if count:
            parts.append(word if count == 1 and value == 1000 else f"{' و '.join(_below_thousand(count))} {word}")
    parts += _below_thousand(n)
    return " و ".join(parts)


def number_words(text: str) -> str:
    """'4268.76' -> Sorani words; short decimals are read as a number, others
    (leading zero, 3+ digits) digit by digit."""
    whole, _, frac = text.partition(".")
    words = integer_words(int(whole))
    if not frac:
        return words
    if len(frac) <= 2 and not frac.startswith("0"):
        tail = integer_words(int(frac))
    else:
        tail = " ".join(_ONES[int(d)] for d in frac)
    return f"{words} {_POINT} {tail}"


def verbalize_numbers(text: str, *, integers: bool = True) -> str:
    """Every stand-alone number in ``text`` as Sorani words (``integers=False``:
    only decimals and percentages -- KurdishTTS reads whole numbers itself:
    measured 2026-09-24, «زێڕ لەسەر 4270 ...» with digits and with words gave the
    same 4.0 s of audio and the same transcript, and digits cost 4 characters
    of the 20k-a-month budget instead of 26)."""
    if not text or not any(ch.isdigit() for ch in text):
        return text

    def say(match: re.Match[str]) -> str:
        if not integers and "." not in match.group(1) and not match.group(2):
            return match.group(0)
        words = number_words(match.group(1))
        return f"لەسەدا {words}" if match.group(2) else words

    return _NUMBER.sub(say, text.translate(_EASTERN))


__all__ = ["verbalize_numbers", "number_words", "integer_words"]
