"""Split extracted ``Block``s into overlapping passages for the index.

A passage is ~``target`` characters of whole paragraphs/sentences. In paged
documents (PDF, DOCX with page marks) a passage never spans two pages and the
overlap never crosses a page break: measured on the synthetic corpus, joining
a short page to the next one and carrying the previous page's last sentence
over made a checklist line on page 2 come back cited as "p. 1" and "p. 3"
(tests/test_knowledge_retrieval.py). Within a page -- and in page-less TXT/MD
-- each passage after the first starts with the last sentence(s) of the one
before (``overlap`` characters, marked with "…"), so a rule that runs across
a boundary is still found. A section change starts a new passage once the
current one has ``min_chars``.

Sizes: ~900 characters is about 150 English or 170 Sorani words -- enough
for one rule with its conditions, small enough that five passages fit in a
tool result (the registry caps result data at 6000 characters).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .extract import Block

_SENTENCE_END = re.compile(r"(?<=[.!?؟۔:;])\s+|\n+")


@dataclass(frozen=True)
class Chunk:
    ord: int
    page_start: int | None
    page_end: int | None
    section: str
    source: str
    text: str            # includes the overlap prefix


def _units(text: str, max_chars: int) -> list[str]:
    """Paragraphs, then sentences for long paragraphs, then word-boundary
    slices for sentences longer than ``max_chars``."""
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            units.append(paragraph)
            continue
        for sentence in _SENTENCE_END.split(paragraph):
            sentence = sentence.strip()
            while len(sentence) > max_chars:
                cut = sentence.rfind(" ", 0, max_chars)
                cut = cut if cut > max_chars // 2 else max_chars
                units.append(sentence[:cut].strip())
                sentence = sentence[cut:].strip()
            if sentence:
                units.append(sentence)
    return units


def _tail(text: str, overlap: int) -> str:
    """The last sentence(s) of ``text`` within ``overlap`` characters (at a
    word boundary when no sentence fits)."""
    if overlap <= 0 or not text:
        return ""
    sentences = [s for s in _SENTENCE_END.split(text) if s.strip()]
    picked: list[str] = []
    size = 0
    for sentence in reversed(sentences):
        if size + len(sentence) > overlap:
            break
        picked.insert(0, sentence.strip())
        size += len(sentence) + 1
    if picked:
        return " ".join(picked)
    piece = text[-overlap:]
    space = piece.find(" ")
    return piece[space + 1:].strip() if 0 <= space < len(piece) - 1 else piece.strip()


def chunk_blocks(blocks: list[Block], *, target: int = 900, overlap: int = 150, max_chars: int = 1400,
                 min_chars: int = 200) -> list[Chunk]:
    chunks: list[Chunk] = []
    parts: list[str] = []
    size = 0
    page_start: int | None = None
    page_end: int | None = None
    section = ""
    source = "text"
    prefix = ""                 # overlap carried into the next passage
    prefix_page: int | None = None

    def emit() -> None:
        nonlocal parts, size, prefix, prefix_page, page_start, page_end
        if not parts:
            return
        body = "\n".join(parts)
        text = (f"… {prefix}\n{body}" if prefix else body).strip()
        chunks.append(Chunk(len(chunks), page_start, page_end, section, source, text))
        prefix, prefix_page = _tail(body, overlap), page_end
        parts, size = [], 0
        page_start = page_end = None

    for block in blocks:
        if parts and block.page != page_end:
            emit()
        elif parts and block.section != section and size >= min_chars:
            emit()
        if prefix and block.page != prefix_page:
            prefix = ""                   # a new page starts clean (see module doc)
        for unit in _units(block.text, max_chars):
            if parts and size + len(unit) > target and size >= min_chars:
                emit()
            if not parts:
                page_start = block.page
                section = block.section
                source = block.source
            parts.append(unit)
            size += len(unit) + 1
            page_end = block.page
            if block.source == "ocr":
                source = "ocr"
    emit()
    return chunks


__all__ = ["Chunk", "chunk_blocks"]
