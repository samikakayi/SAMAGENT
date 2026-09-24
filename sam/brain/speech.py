"""Streaming text -> sentence-sized chunks for speech (and captions).

The cascade voice path sends every chunk ``Conversation.respond_stream``
yields straight to TTS, so the first chunk decides the time to first audio:
a short acknowledgement ("باشە.") is emitted the moment its sentence ends
instead of waiting for the whole reply (v1 waited for the full reply and then
read it; audit-latency.json measured 13-20 s to first sound).

Rules (ported from v1 ``voice.speakable_text/speakable_chunks``, made
incremental):
- a sentence ends at . ! ? ؟ ۔ … or a newline, but ``.`` only counts when
  whitespace follows, so prices such as ``2650.5`` are never cut;
- no chunk is longer than ``max_chars`` (180: KurdishTTS's free plan refuses
  requests over 500 characters, and v1's browser used 180-char chunks); a long
  sentence is cut at the last comma/semicolon, else the last space;
- Markdown is stripped for speech (models answer in Markdown even when told
  not to; v1's voice read "**" and list dashes aloud); code blocks are dropped.
"""

from __future__ import annotations

import re

from ..textnorm import fix_letters

MAX_CHUNK_CHARS = 180
_HARD_END = "!?؟۔…"
_SOFT_BREAKS = "،,؛;:"
_CODE_BLOCK = re.compile(r"```[\s\S]*?(```|$)")
_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")
_URL = re.compile(r"https?://\S+")
_LIST_MARKER = re.compile(r"(?m)^\s*(?:[-*+•]|\d+[.)])\s+")
_HEADING = re.compile(r"(?m)^\s*#{1,6}\s*")
_MARKUP = re.compile(r"[*_`>|]+")
_SPACES = re.compile(r"[ \t]+")
_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿️]")


def speakable(text: str) -> str:
    """Text as it should be heard: no Markdown, links reduced to their label,
    bare URLs and code dropped, emojis removed, whitespace collapsed, Kurdish
    letter forms (``fix_letters``: models still write Persian ه‌ and Arabic ي/ك)."""
    plain = _CODE_BLOCK.sub(" ", fix_letters(text or ""))
    plain = _LINK.sub(r"\1", plain)
    plain = _URL.sub(" ", plain)
    plain = _HEADING.sub("", plain)
    plain = _LIST_MARKER.sub("\n", plain)
    plain = _MARKUP.sub("", plain)
    plain = _EMOJI.sub("", plain)
    lines = [_SPACES.sub(" ", line).strip() for line in plain.splitlines()]
    return " ".join(line for line in lines if line).strip()


def _boundary(buffer: str) -> int:
    """Index just after the first complete sentence in ``buffer``, or -1.

    A boundary needs the following character to be known (whitespace), so a
    trailing ``.`` at the end of a streamed delta waits for the next delta.
    """
    for index, char in enumerate(buffer):
        if char == "\n":
            return index + 1
        if char in _HARD_END or char == ".":
            nxt = index + 1
            # swallow runs like "?!" or "...", closing quotes and Markdown emphasis
            while nxt < len(buffer) and (buffer[nxt] in _HARD_END or buffer[nxt] in ".\"'»”’)*_`"):
                nxt += 1
            if nxt < len(buffer) and buffer[nxt].isspace():
                return nxt
    return -1


def _cut_long(buffer: str, max_chars: int) -> int:
    """Where to cut a sentence that is too long: after the last soft break
    (comma/semicolon) in the window, else at the last space, else hard."""
    window = buffer[:max_chars]
    for pos in range(len(window) - 1, max_chars // 3, -1):
        if window[pos] in _SOFT_BREAKS:
            return pos + 1
    space = window.rfind(" ")
    return space if space > max_chars // 4 else max_chars


class SentenceChunker:
    """Incremental splitter: ``feed(delta)`` returns the chunks completed by
    that delta; ``flush()`` returns what is left at the end of the stream.

    ``speech=True`` strips Markdown from every chunk (cascade TTS); text
    chunks for the panel keep the model's characters unchanged.
    """

    def __init__(self, *, max_chars: int = MAX_CHUNK_CHARS, speech: bool = False) -> None:
        self.max_chars = max(20, int(max_chars))
        self.speech = speech
        self._buffer = ""
        self._in_code = False

    def feed(self, delta: str) -> list[str]:
        if not delta:
            return []
        self._buffer += delta
        out: list[str] = []
        while True:
            end = _boundary(self._buffer)
            if end < 0 and len(self._buffer) > self.max_chars:
                end = _cut_long(self._buffer, self.max_chars)
            if end < 0:
                break
            if end > self.max_chars:
                end = _cut_long(self._buffer, self.max_chars)
            piece, self._buffer = self._buffer[:end], self._buffer[end:]
            out.extend(self._emit(piece))
        return out

    def flush(self) -> list[str]:
        piece, self._buffer = self._buffer, ""
        return self._emit(piece)

    def _emit(self, piece: str) -> list[str]:
        if self.speech:
            # Code fences may span chunks: drop everything inside them.
            fences = piece.count("```")
            if self._in_code or fences:
                parts = piece.split("```")
                kept = [p for i, p in enumerate(parts) if (i % 2 == 0) != self._in_code]
                if fences % 2 == 1:
                    self._in_code = not self._in_code
                piece = " ".join(kept)
            piece = speakable(piece)
        else:
            piece = fix_letters(piece.strip())
        return [piece] if piece and any(c.isalnum() for c in piece) else []


def split_sentences(text: str, *, max_chars: int = MAX_CHUNK_CHARS, speech: bool = False) -> list[str]:
    """Split a finished text the same way the streaming chunker does."""
    chunker = SentenceChunker(max_chars=max_chars, speech=speech)
    return chunker.feed(text) + chunker.flush()


__all__ = ["SentenceChunker", "split_sentences", "speakable", "MAX_CHUNK_CHARS"]
