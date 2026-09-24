"""Text clean-up for the knowledge library: what a PDF text layer, a DOCX or
an OCR pass gives back, turned into (a) readable passages and (b) one search
form that Sorani, Arabic-keyboard and OCR spellings all share.

Measured on this PC (2026-09-24, pypdf 6.19.0, work/kb-exp):

- A Sorani page printed to PDF by Edge/Chrome (the same Skia path Word's
  "Save as PDF" output resembles) extracts in LOGICAL order but as Arabic
  presentation forms (U+FB50-FDFF, U+FE70-FEFF: 28 distinct forms on one
  page). NFKC folds them back to the base letters; ە ێ ڕ come out as base
  letters already.
- The same page drawn by Qt's PDF writer extracts in VISUAL order (every
  Arabic-script line reversed: «ستراتیژیی» came out as «ﯽﯾﮋﯿﺗاﺮﺘﺳ»), with
  tabs between words and control characters where a glyph had no Unicode
  mapping. ``fix_visual_order`` detects that from where the positional forms
  sit in each word (an INITIAL form at the end of a word is only possible in
  visual order) and, without forms, from Sorani spelling (ئ starts words,
  ە ends them), then reverses the line and re-reverses Latin/digit runs.
- Windows OCR has no Kurdish recogniser: ar-SA reads ڵ ڕ ۆ ێ ە ڤ گ چ پ ژ as
  their Arabic base letters (sam/hands/ocr.py). ``search_form`` folds both
  sides the same way, so a scanned page and a typed question still meet.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from ..textnorm import normalize_ckb

# Control characters (keep \t \n), zero-width marks, BOM and U+FFFD.
_CONTROL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\u200e\u200f\u061c\ufeff\ufffd]")
_PRESENTATION = re.compile("[\ufb00-\ufdff\ufe70-\ufefe]+")
_ARABIC_WORD = re.compile("[\u0600-\u06ff\u0750-\u077f\ufb50-\ufdff\ufe70-\ufefe]+")
_ARABIC_CHAR = re.compile("[\u0600-\u06ff\u0750-\u077f\ufb50-\ufdff\ufe70-\ufefe]")
_LATIN_CHAR = re.compile("[A-Za-z]")
# Left-to-right runs inside a reversed RTL line: words, numbers, prices, times.
_LTR_RUN = re.compile(r"[A-Za-z0-9\u00c0-\u024f]+(?:[.,:/%'\-][A-Za-z0-9\u00c0-\u024f]+)*")
_HYPHEN_BREAK = re.compile(r"([A-Za-z])-\n([a-z])")
_INLINE_SPACE = re.compile(r"[ \t\u00a0\u2000-\u200a\u202f\u205f\u3000]+")
_BLANK_LINES = re.compile(r"\n{3,}")

# The OCR confusions of the ar-SA recogniser (same table as sam/hands/ocr.py)
# plus Arabic hamza/alef variants; applied to documents AND questions.
_FOLD = str.maketrans({"ڵ": "ل", "ڕ": "ر", "ۆ": "و", "ێ": "ی", "ە": "ه", "ھ": "ه", "ڤ": "ف", "گ": "ک",
                       "چ": "ج", "پ": "ب", "ژ": "ز", "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ؤ": "و"})


@lru_cache(maxsize=2048)
def _form_of(ch: str) -> str:
    """'i' initial, 'm' medial, 'f' final, 's' isolated, '' not a positional form."""
    code = ord(ch)
    if not (0xFB50 <= code <= 0xFDFF or 0xFE70 <= code <= 0xFEFE):
        return ""
    name = unicodedata.name(ch, "")
    for marker, kind in (("INITIAL FORM", "i"), ("MEDIAL FORM", "m"), ("FINAL FORM", "f"), ("ISOLATED FORM", "s")):
        if marker in name:
            return kind
    return ""


def order_evidence(text: str) -> tuple[int, int]:
    """(logical, visual) votes for the Arabic-script words of ``text``.

    Positional presentation forms are decisive: in logical order a word's
    first joined letter is an INITIAL form and its last a FINAL form; read in
    visual order they swap ends. Without forms (base letters), Sorani
    spelling votes: ئ only starts a word and a word never starts with ە, so
    «ئەوە» reversed («ەوئ») votes visual twice."""
    logical = visual = 0
    for word in _ARABIC_WORD.findall(text):
        if len(word) < 2:
            continue
        first, last = _form_of(word[0]), _form_of(word[-1])
        logical += (first == "i") + (last == "f")
        visual += (last == "i") + (first == "f")
        base = _PRESENTATION.sub(lambda m: unicodedata.normalize("NFKC", m.group()), word)
        if len(base) < 2:
            continue
        logical += base.startswith("ئ") + base.endswith("ە") + base.startswith("ال")
        visual += base.endswith("ئ") + base.startswith("ە") + (len(base) > 3 and base.endswith("لا"))
    return logical, visual


def is_visual_order(text: str) -> bool:
    """True when the Arabic-script text of a page was extracted reversed."""
    logical, visual = order_evidence(text)
    return visual >= 2 and visual > 2 * logical


def reverse_line(line: str) -> str:
    """Visual -> logical for one line: reverse it, then put Latin words and
    numbers (which were already left-to-right) back the right way round."""
    if not _ARABIC_CHAR.search(line):
        return line
    flipped = line[::-1]
    return _LTR_RUN.sub(lambda m: m.group()[::-1], flipped)


def fix_visual_order(text: str) -> tuple[str, bool]:
    """Reverse every Arabic-script line when the page reads in visual order."""
    if not is_visual_order(text):
        return text, False
    return "\n".join(reverse_line(line) for line in text.split("\n")), True


def clean_text(raw: str) -> tuple[str, bool]:
    """Readable text from a PDF page / DOCX / OCR pass. Returns (text, reversed).

    Order matters: the visual-order vote needs the presentation forms, so it
    runs before NFKC folds them away."""
    if not raw:
        return "", False
    text = raw.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text = _CONTROL.sub("", text)
    text, flipped = fix_visual_order(text)
    text = _PRESENTATION.sub(lambda m: unicodedata.normalize("NFKC", m.group()), text)
    text = unicodedata.normalize("NFC", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    lines = [_INLINE_SPACE.sub(" ", line).strip() for line in text.split("\n")]
    text = _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()
    return text, flipped


def search_form(text: str) -> str:
    """The one form documents are indexed in and questions are matched in:
    ``normalize_ckb`` (Kurdish letters, digits, lower case, no punctuation)
    plus the OCR fold (ە→ه, ێ→ی, ڕ→ر ... see module doc)."""
    if not text:
        return ""
    value = _PRESENTATION.sub(lambda m: unicodedata.normalize("NFKC", m.group()), text)
    return normalize_ckb(value, strip_punct=True).translate(_FOLD)


def fold(text: str) -> str:
    """Apply only the OCR fold to text that is already ``normalize_ckb``-ed."""
    return text.translate(_FOLD)


def script_of(text: str) -> str:
    """'ckb' (Arabic script), 'en' (Latin) or 'mixed' for a document sample."""
    arabic = len(_ARABIC_CHAR.findall(text[:20000]))
    latin = len(_LATIN_CHAR.findall(text[:20000]))
    if arabic == latin == 0:
        return ""
    if arabic >= 4 * max(latin, 1) or (arabic and not latin):
        return "ckb"
    if latin >= 4 * max(arabic, 1) or (latin and not arabic):
        return "en"
    return "mixed"


__all__ = ["clean_text", "fix_visual_order", "fold", "is_visual_order", "order_evidence", "reverse_line",
           "script_of", "search_form"]
