"""Sorani / Arabic-script text normalisation shared by every module.

Speech transcripts, typed text and stored notes use several spellings of the
same letters (Arabic ي/ك vs Kurdish ی/ک, ZWNJ, tatweel, Eastern Arabic or
Persian digits). Matching (yes/no answers, app aliases, FTS queries) must see
one canonical form, so everything that compares Sorani text calls
``normalize_ckb`` first. The trading research measured FTS5 trigram retrieval
on normalised Sorani at 69% top-1 vs 56% for local embeddings
(reports/trading-intelligence.json).
"""

from __future__ import annotations

import re
import unicodedata

# Arabic -> Kurdish letter forms, and presentation variants seen in STT output.
_LETTER_MAP = str.maketrans({
    "ي": "ی",  # ARABIC YEH -> FARSI YEH (Kurdish ی)
    "ى": "ی",  # ALEF MAKSURA -> ی
    "ك": "ک",  # ARABIC KAF -> KEHEH (Kurdish ک)
    "ة": "ە",  # TEH MARBUTA -> ە (Arabic-influenced STT spelling)
    "ـ": None,      # TATWEEL
    "‌": None,      # ZWNJ
    "‍": None,      # ZWJ
    "‏": None,      # RLM
    "‎": None,      # LRM
    "؜": None,      # ARABIC LETTER MARK
})
# Eastern Arabic (U+0660..) and Persian (U+06F0..) digits -> ASCII.
_DIGITS = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}
_DIGITS.update({ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")})
# Arabic diacritics (harakat) carry no meaning in Sorani script.
_DIACRITICS = re.compile("[ً-ٰٟ]")
_PUNCT = re.compile(r"[،؛؟۔.,!?;:\"'()\[\]{}«»…\-_/\\]+")
_SPACES = re.compile(r"\s+")


def normalize_ckb(text: str, *, lower: bool = True, strip_punct: bool = False) -> str:
    """Return the canonical comparison form of Sorani/English text.

    Keeps the meaning; only unifies spelling variants. ``strip_punct`` also
    turns punctuation into spaces (for word matching).
    """
    if not text:
        return ""
    value = unicodedata.normalize("NFC", text)
    value = value.translate(_LETTER_MAP).translate(_DIGITS)
    value = _DIACRITICS.sub("", value)
    if strip_punct:
        value = _PUNCT.sub(" ", value)
    if lower:
        value = value.lower()
    return _SPACES.sub(" ", value).strip()


_PERSIAN_E = re.compile("ه‌")          # Persian writes Kurdish ە as heh + ZWNJ
_STRAY_ZWNJ = re.compile("(?<=[ەێۆ])‌")   # ZWNJ after a Kurdish vowel letter does nothing
_DISPLAY_LETTERS = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ة": "ە"})


def fix_letters(text: str) -> str:
    """Kurdish letter forms for text that is SHOWN or SPOKEN (no lowercasing,
    punctuation and digits kept). Models write Persian/Arabic forms although the
    persona forbids them: the review's Groq probe returned «یه‌کێ» and
    «فینانسیه‌کانی» (heh + ZWNJ for ە)."""
    if not text:
        return text or ""
    value = _PERSIAN_E.sub("ە", text)
    value = _STRAY_ZWNJ.sub("", value)
    return value.translate(_DISPLAY_LETTERS)


def words(text: str) -> list[str]:
    """Normalised words of ``text`` (punctuation removed)."""
    return normalize_ckb(text, strip_punct=True).split()


def is_arabic_script(text: str, threshold: float = 0.5) -> bool:
    """True when most letters of ``text`` are Arabic-script (Sorani replies)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    arabic = sum(1 for c in letters if "؀" <= c <= "ۿ" or "ݐ" <= c <= "ݿ")
    return arabic / len(letters) >= threshold


__all__ = ["normalize_ckb", "words", "is_arabic_script", "fix_letters"]
