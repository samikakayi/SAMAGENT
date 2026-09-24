"""Text that is spoken: Markdown clean-up, streaming sentence splitting, CER.

Why a streaming splitter: the cascade speaks the FIRST sentence while the
model is still writing the rest (reports/realtime-voice.json: "the first
complete sentence goes at once to TTS ... the rest of the reply goes in one
more request, about 2 TTS requests per turn, to save RPD"). v1 waited for the
whole reply and then read it in 180-character chunks, one of the reasons its
time to first sound was 13-20 s (reports/audit-latency.json).

Sorani punctuation: sentences end with ``. ! ? ؟ …`` or a line break; ``، ؛ :
,`` are clause breaks (used only to cut a long FIRST piece early). A dot
between digits is a decimal point ("2650.5"), never a sentence end.
"""

from __future__ import annotations

import re
import unicodedata

from ..textnorm import fix_letters, is_arabic_script, normalize_ckb

STRONG_ENDS = frozenset(".!?؟…\n")
SOFT_ENDS = frozenset("،؛:,;")
_CLOSERS = frozenset("\"'»”’)]")

# Markdown / chat artefacts that must not be read aloud. v1 read "**", list
# dashes and backticks aloud (sam_backend/voice.py speakable_text).
_CODE_BLOCK = re.compile(r"```[\s\S]*?(?:```|$)")
_INLINE_CODE = re.compile(r"`([^`]*)`")
_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")
_URL = re.compile(r"https?://\S+")
_LIST_MARKER = re.compile(r"(?m)^\s*(?:[-*+•]|\d+[.)])\s+")
_HEADING = re.compile(r"(?m)^\s*#{1,6}\s*")
_MARKUP = re.compile(r"[*_#>|~]")
_SPACES = re.compile(r"[ \t]+")

# Characters Sorani has and Arabic/Persian do not: their presence separates a
# Sorani reply from a Persian or Arabic one (both Arabic script).
SORANI_LETTERS = "ەێۆڕڵ"


def _drop_symbols(text: str) -> str:
    """Remove emoji and other pictographs (Unicode category So/Sk + joiners)."""
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat in ("So", "Sk", "Cs") or ch in "️︎⃣":
            continue
        out.append(ch)
    return "".join(out)


def speakable_text(text: str) -> str:
    """The reply as it should sound: no Markdown, code, URLs or emoji, Kurdish
    letter forms (models write Persian ه‌ and Arabic ي/ك, review 2026-09-24)."""
    plain = _CODE_BLOCK.sub(" ", fix_letters(text or ""))
    plain = _LINK.sub(r"\1", plain)
    plain = _URL.sub(" ", plain)
    plain = _INLINE_CODE.sub(r"\1", plain)
    plain = _HEADING.sub("", plain)
    plain = _LIST_MARKER.sub("\n", plain)
    plain = _MARKUP.sub(" ", plain)
    plain = _drop_symbols(plain)
    lines = [_SPACES.sub(" ", line).strip() for line in plain.splitlines()]
    return "\n".join(line for line in lines if line)


def _has_words(text: str) -> bool:
    return any(ch.isalnum() for ch in text)


def _is_decimal_dot(buf: str, i: int) -> bool:
    return (buf[i] == "." and 0 < i < len(buf) - 1
            and buf[i - 1].isdigit() and buf[i + 1].isdigit())


def _extend_closers(buf: str, i: int) -> int:
    """Index just after a boundary char plus any closing quotes/brackets."""
    j = i + 1
    while j < len(buf) and buf[j] in _CLOSERS:
        j += 1
    return j


class SentenceSplitter:
    """Incremental splitter for streamed model text.

    ``feed(delta)`` returns the pieces that are ready to be spoken;
    ``flush()`` returns the rest when the stream ends.

    - First piece: as soon as a sentence ends (even a short "باشە."), or at a
      clause break once ``first_soft_chars`` are buffered, or cut at a space
      after ``first_hard_chars`` -- this piece decides time to first audio.
    - Later pieces: whole sentences packed up to at least ``next_min_chars``
      (fewer TTS requests; the free tiers count requests per day) and never
      more than ``max_chars`` (KurdishTTS refuses > 500 characters per
      request -- v1's reply of 552 characters went silent on 2026-09-24).
    """

    def __init__(self, *, first_soft_chars: int = 40, first_hard_chars: int = 120,
                 next_min_chars: int = 150, max_chars: int = 480, eager_first: bool = True) -> None:
        self.eager_first = eager_first
        self.first_soft_chars = first_soft_chars
        self.first_hard_chars = first_hard_chars
        self.next_min_chars = next_min_chars
        self.max_chars = max_chars
        self._buf = ""
        self._emitted = 0

    @property
    def pending(self) -> str:
        return self._buf

    def _boundaries(self, *, final: bool) -> list[int]:
        """End indexes (exclusive) of complete sentences in the buffer."""
        ends: list[int] = []
        buf = self._buf
        for i, ch in enumerate(buf):
            if ch not in STRONG_ENDS:
                continue
            if ch == "." and not final and i == len(buf) - 1:
                continue  # could still become a decimal point ("2650." + "5")
            if _is_decimal_dot(buf, i):
                continue
            end = _extend_closers(buf, i)
            if not ends or end > ends[-1]:
                ends.append(end)
        return ends

    def _soft_boundary(self) -> int | None:
        buf = self._buf
        for i, ch in enumerate(buf):
            if ch not in SOFT_ENDS or i + 1 < self.first_soft_chars:
                continue
            if ch in ",:" and i > 0 and buf[i - 1].isdigit() and (i == len(buf) - 1 or buf[i + 1].isdigit()):
                continue  # "2,650" or "10:30" is one token (or may still become one)
            return _extend_closers(buf, i)
        return None

    def _cut_at_space(self, limit: int) -> int:
        cut = self._buf.rfind(" ", 0, limit)
        return cut + 1 if cut >= limit // 4 else limit

    def _take(self, end: int) -> str | None:
        piece, self._buf = self._buf[:end], self._buf[end:]
        clean = speakable_text(piece).replace("\n", " ").strip()
        if not clean or not _has_words(clean):
            return None
        self._emitted += 1
        return clean

    def feed(self, delta: str) -> list[str]:
        if not delta:
            return []
        self._buf += delta
        out: list[str] = []
        while True:
            piece = self._next_piece(final=False)
            if piece is None:
                break
            if piece:
                out.append(piece)
        return out

    def _next_piece(self, *, final: bool) -> str | None:
        """'' = consumed junk (loop again), None = nothing ready."""
        if not self._buf.strip():
            return None
        ends = self._boundaries(final=final)
        if self._emitted == 0 and self.eager_first:
            if ends and ends[0] <= self.max_chars:
                return self._take(ends[0]) or ""
            soft = self._soft_boundary()
            if soft is not None and soft <= self.max_chars:
                return self._take(soft) or ""
            if len(self._buf) >= self.first_hard_chars:
                return self._take(self._cut_at_space(self.first_hard_chars)) or ""
            return None
        fitting = [e for e in ends if e <= self.max_chars]
        if fitting and (fitting[-1] >= self.next_min_chars or final):
            return self._take(fitting[-1]) or ""
        if len(self._buf) > self.max_chars:
            return self._take(fitting[-1] if fitting else self._cut_at_space(self.max_chars)) or ""
        return None

    def take_ready(self) -> list[str]:
        """Every complete sentence buffered now, ignoring ``next_min_chars``.

        The cascade calls this when the speaker is about to run dry, so a
        slow model stream never leaves a silent gap after the first piece."""
        ends = [e for e in self._boundaries(final=False) if e <= self.max_chars]
        if not ends:
            return []
        piece = self._take(ends[-1])
        return [piece] if piece else []

    def flush(self) -> list[str]:
        out: list[str] = []
        while True:
            piece = self._next_piece(final=True)
            if piece is None:
                break
            if piece:
                out.append(piece)
        if self._buf.strip():
            while self._buf:
                end = len(self._buf) if len(self._buf) <= self.max_chars else self._cut_at_space(self.max_chars)
                piece = self._take(end)
                if piece:
                    out.append(piece)
        self._buf = ""
        return out


def split_for_tts(text: str, max_chars: int = 480) -> list[str]:
    """Split a whole text into speakable requests of at most ``max_chars``."""
    splitter = SentenceSplitter(next_min_chars=max_chars, max_chars=max_chars, eager_first=False)
    pieces = splitter.feed(text)
    pieces += splitter.flush()
    return pieces


# -- self-test metrics -----------------------------------------------------------------

def _cer_norm(text: str) -> str:
    return re.sub(r"\s+", "", normalize_ckb(text or "", strip_punct=True))


def levenshtein(a: str, b: str) -> int:
    try:
        from rapidfuzz.distance import Levenshtein  # C implementation, already a dependency
        return int(Levenshtein.distance(a, b))
    except ImportError:  # pragma: no cover - rapidfuzz is in requirements
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate after Sorani normalisation (ي/ك unified, digits
    ASCII, punctuation and spaces removed). 0.0 = identical."""
    ref, hyp = _cer_norm(reference), _cer_norm(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def sorani_script_ok(text: str) -> bool:
    """Arabic script AND at least one Sorani-only letter (ە ێ ۆ ڕ ڵ).

    Kurmanji is written in Latin letters and Persian/Arabic lack these
    letters, so this separates the reply the user needs from the two most
    likely wrong ones (the Live language table lists only "Kurdish (ku)")."""
    return bool(text) and is_arabic_script(text) and any(ch in text for ch in SORANI_LETTERS)


__all__ = ["SentenceSplitter", "speakable_text", "split_for_tts", "cer", "levenshtein", "sorani_script_ok",
           "SORANI_LETTERS", "STRONG_ENDS", "SOFT_ENDS"]
