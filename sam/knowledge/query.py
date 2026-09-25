"""Turn a question into search concepts.

A concept is one idea of the question with every spelling that counts as a
hit: the word itself, its stem (Sorani suffixes such as -ەکان/-ییەکە/-ەکەم
and English -s/-ing/-ed stripped; the trigram index matches substrings, so
a stem finds every inflected form), and -- for trading terms -- the whole
glossary group in both languages. Question words and filler («چی»، «دەڵێت»،
"what", "book") are dropped. Every spelling is in the folded search form.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache

from ..textnorm import normalize_ckb
from .glossary import GROUPS
from .textfix import fold, search_form

MAX_CONCEPTS = 8
MAX_VARIANTS = 8

# Sorani suffixes, longest first (definite/plural/possessive/ezafe endings).
_CKB_SUFFIXES = tuple(sorted({
    "ەکانمان", "ەکانتان", "ەکانیان", "ییەکانم", "ییەکانی", "ییەکان", "یەکانم", "یەکانی", "یەکان", "ەکانی",
    "ەکانم", "ەکانت", "ەکان", "ییەکەم", "ییەکەی", "ییەکە", "یەکەم", "یەکەی", "یەکە", "ەکەمان", "ەکەتان",
    "ەکەیان", "ەکەم", "ەکەت", "ەکەی", "ەکە", "ێکی", "ێک", "یەک", "ەوە", "یان", "مان", "تان", "ان", "یی",
    "ی", "ە", "دا", "یش", "ش", "م", "ت"}, key=len, reverse=True))
_EN_SUFFIXES = (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", ""))
# Endings that keep an English glossary word the same word ("supports", "trending").
_EN_ENDINGS = frozenset({"s", "es", "ed", "d", "ing", "ings"})

_STOP_CKB = {normalize_ckb(w) for w in (
    "و", "لە", "بۆ", "بە", "کە", "ئەو", "ئەم", "ئەوە", "ئەمە", "ئەوەی", "چی", "چییە", "چیە", "چۆن", "چۆنە",
    "چەند", "کام", "کامە", "کێ", "یان", "یا", "هەروەها", "دەربارەی", "دەربارە", "لەسەر", "سەر", "لەگەڵ",
    "لەناو", "ناو", "تا", "هەتا", "ئایا", "من", "تۆ", "ئێمە", "ئەوان", "نییە", "هەیە", "دەڵێت", "دەڵێ",
    "دەڵێن", "بڵێ", "پێم", "پێمان", "بڵێم", "بکە", "بدە", "باس", "باسی", "دەکات", "دەکەن", "کراوە", "بکەن",
    "کتێب", "کتێبەکە", "کتێبەکان", "کتێبەکەم", "کتێبەکانم", "کتێبەکانی", "کتێبی", "لەکتێبەکەم", "پەرتووک",
    "بەڵگەنامە", "بەڵگەنامەکان", "فایل", "فایلەکە", "فایلەکان", "تێیدا", "تیایدا", "چی", "شتێک", "هەر",
    "هەموو", "زۆر", "کەمێک", "ئێستا", "پێویستە", "دەبێت", "دەتوانم", "دەتوانیت", "بزانم", "بەپێی",
    "وەک", "وەکو", "لێرە", "ئەوێ", "نووسراوە", "نووسیوە", "دەڵێن", "بڵێت", "نیە", "کوێ", "لەکوێ", "کەی",
    "بۆچی", "چما", "ئایە")}
_STOP_EN = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are", "was", "were", "be", "been",
    "what", "how", "why", "when", "which", "who", "whom", "does", "do", "did", "say", "says", "said", "about",
    "my", "me", "i", "you", "your", "it", "its", "this", "that", "these", "those", "with", "from", "by", "as",
    "at", "book", "books", "document", "documents", "file", "files", "pdf", "tell", "explain", "according",
    "there", "their", "they", "can", "could", "should", "would", "will", "into", "than", "then", "also",
    "any", "some", "all", "use", "used", "using", "get", "give", "show", "find", "please"}


@dataclass(frozen=True)
class Concept:
    label: str                   # the question words it came from (normalised)
    variants: tuple[str, ...]    # folded search-form spellings, each >= 3 characters
    glossary: bool = False


@lru_cache(maxsize=1)
def _glossary_index() -> list[tuple[str, int]]:
    """(folded spelling, group index), longest spelling first."""
    pairs = []
    for index, group in enumerate(GROUPS):
        for spelling in group:
            form = search_form(spelling)
            if len(form.replace(" ", "")) >= 3:
                pairs.append((form, index))
    pairs.sort(key=lambda item: len(item[0]), reverse=True)
    return pairs


@lru_cache(maxsize=256)
def group_variants(index: int) -> tuple[str, ...]:
    forms = []
    for spelling in GROUPS[index]:
        form = search_form(spelling)
        if len(form) >= 3 and form not in forms:
            forms.append(form)
    return tuple(forms[:MAX_VARIANTS])


def stem(word: str) -> str:
    """One inflection stripped (two for stacked Sorani endings). ``word`` is
    normalize_ckb-ed (unfolded) so the Sorani endings are still visible."""
    if re.fullmatch(r"[a-z0-9]+", word):
        for suffix, replacement in _EN_SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                return word[: -len(suffix)] + replacement
        return word
    current = word
    for _ in range(2):
        for suffix in _CKB_SUFFIXES:
            if current.endswith(suffix) and len(current) - len(suffix) >= 3:
                current = current[: -len(suffix)]
                break
        else:
            break
    return current


def analyse(question: str) -> list[Concept]:
    """Concepts of ``question`` in order of appearance (at most MAX_CONCEPTS)."""
    plain = normalize_ckb(question or "", strip_punct=True)
    words = plain.split()
    folded_words = [fold(w) for w in words]
    used = [False] * len(words)
    concepts: list[Concept] = []
    seen_groups: set[int] = set()
    # 1) glossary phrases (longest first), matched at word starts so Sorani
    #    endings are allowed: «پشتگیرییەکان» contains «پشتگیری».
    for form, index in _glossary_index():
        parts = form.split()
        n = len(parts)
        for i in range(len(words) - n + 1):
            if any(used[i:i + n]):
                continue
            window = folded_words[i:i + n]
            if window[:-1] == parts[:-1] and window[-1].startswith(parts[-1]):
                rest = window[-1][len(parts[-1]):]
                if len(rest) > 7 or (rest and parts[-1].isascii() and rest not in _EN_ENDINGS):
                    # a different word that only starts the same ("golden" is not "gold")
                    continue
                for j in range(i, i + n):
                    used[j] = True
                if index not in seen_groups:
                    seen_groups.add(index)
                    concepts.append(Concept(" ".join(words[i:i + n]), group_variants(index), True))
    # 2) remaining content words
    for i, word in enumerate(words):
        if used[i] or word in _STOP_CKB or word in _STOP_EN:
            continue
        variants = []
        for form in (fold(word), fold(stem(word))):
            if len(form) >= 3 and form not in variants and not form.isdigit():
                variants.append(form)
        if len(word) >= 2 and word.isdigit():
            variants = [word] if len(word) >= 3 else []
        if variants and all(v not in c.variants for c in concepts for v in variants):
            concepts.append(Concept(word, tuple(variants)))
    return concepts[:MAX_CONCEPTS]


def match_expression(variants: tuple[str, ...] | list[str]) -> str | None:
    """Safe FTS5 MATCH for a trigram index: quoted phrases OR-ed."""
    terms = [v.replace('"', '""') for v in variants if len(v) >= 3]
    return " OR ".join(f'"{t}"' for t in terms) if terms else None


def idf(total: int, df: int) -> float:
    return math.log((total + 1.0) / (df + 0.5)) + 0.1


__all__ = ["Concept", "analyse", "group_variants", "idf", "match_expression", "stem", "search_form"]
