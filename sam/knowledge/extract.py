"""Read PDF, DOCX, TXT and Markdown files into ``Block``s (page, section, text).

- PDF: the text layer through pypdf (pure Python, py3 wheel; 6.19.0 here).
  A page whose text layer is (nearly) empty is a scan: its embedded images
  go to the OCR callable (Windows OCR in SAM, a fake in tests), up to
  ``ocr_page_cap`` pages per document, so a 400-page scanned book cannot
  hold the OCR worker for many minutes. The outline (bookmarks) names the
  section of each page, so citations can say the chapter.
- DOCX: parsed with the standard library (zip + XML; no python-docx/lxml).
  Headings come from paragraph styles (styles.xml names / outline levels);
  page numbers from Word's own ``lastRenderedPageBreak`` marks and explicit
  page breaks (what Word last laid out), or none when the file has neither.
- TXT/MD: UTF-8 (BOM aware), UTF-16, else Windows-1256 (legacy Arabic-script
  text); form feeds split TXT pages; ``#`` headings name Markdown sections.

Everything here is synchronous and runs in a worker thread; ``cancel`` (a
``threading.Event``) is checked between pages.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:  # zipfile + ElementTree cost ~15 ms to import: loaded on first DOCX
    import zipfile
    from xml.etree import ElementTree as ET

from .textfix import clean_text, script_of

log = logging.getLogger("sam.knowledge.extract")

SUPPORTED = {".pdf": "pdf", ".docx": "docx", ".txt": "txt", ".md": "md", ".markdown": "md"}
OcrFn = Callable[[Any], str]            # PIL image -> text
ProgressFn = Callable[[int, int, str], None]   # (page, pages, stage)

W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_HEADING_STYLE = re.compile(r"^(heading|title|subtitle|berschrift|titre|titolo|encabezado|kop)\s*\d*$", re.I)
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_WORD_TITLE_PREFIX = re.compile(r"^(microsoft word|microsoft powerpoint)\s*-\s*", re.I)
_BAD_TITLES = {"untitled", "document", "doc", "title", "none", "unknown", "microsoft word"}


class ExtractError(Exception):
    """A file that cannot be read (encrypted, damaged, unsupported)."""


@dataclass(frozen=True)
class Block:
    page: int | None        # 1-based page, None when the format has no pages
    section: str            # nearest heading / outline entry ('' when none)
    text: str
    source: str = "text"    # text | ocr


@dataclass
class Extracted:
    title: str
    kind: str
    pages: int = 0
    blocks: list[Block] = field(default_factory=list)
    ocr_pages: int = 0          # pages read by OCR
    skipped_pages: int = 0      # scanned pages beyond the OCR cap or unreadable
    reversed_pages: int = 0     # pages whose Arabic-script text was in visual order
    language: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def chars(self) -> int:
        return sum(len(b.text) for b in self.blocks)


def kind_of(path: Path) -> str | None:
    return SUPPORTED.get(path.suffix.lower())


def _title_from(meta_title: Any, path: Path) -> str:
    title = str(meta_title or "").strip()
    title = _WORD_TITLE_PREFIX.sub("", title)
    title = re.sub(r"\.(docx?|pdf|txt|md|html?)$", "", title, flags=re.I).strip()
    if len(title) < 3 or title.lower() in _BAD_TITLES:
        return path.stem.replace("_", " ").strip() or path.name
    return title[:200]


def extract(path: Path, *, ocr: OcrFn | None = None, ocr_page_cap: int = 40, ocr_min_chars: int = 25,
            progress: ProgressFn | None = None, cancel: threading.Event | None = None) -> Extracted:
    kind = kind_of(path)
    if kind is None:
        raise ExtractError(f"unsupported file type: {path.suffix or path.name}")
    if kind == "pdf":
        result = extract_pdf(path, ocr=ocr, ocr_page_cap=ocr_page_cap, ocr_min_chars=ocr_min_chars,
                             progress=progress, cancel=cancel)
    elif kind == "docx":
        result = extract_docx(path)
    else:
        result = extract_text_file(path, markdown=(kind == "md"))
    result.language = script_of(" ".join(b.text for b in result.blocks[:200]))
    return result


# -- PDF ----------------------------------------------------------------------------------------------------------
def _outline_sections(reader: Any) -> list[tuple[int, str]]:
    """[(page_index, title)] sorted by page, from the PDF's bookmarks."""
    found: list[tuple[int, str]] = []

    def walk(items: Any, depth: int) -> None:
        for item in items:
            if isinstance(item, list):
                if depth < 3:
                    walk(item, depth + 1)
                continue
            try:
                index = reader.get_destination_page_number(item)
                title = str(getattr(item, "title", "") or "").strip()
            except Exception:  # noqa: BLE001 - broken bookmarks are common
                continue
            if index is not None and index >= 0 and title:
                found.append((int(index), clean_text(title)[0][:120]))

    try:
        walk(reader.outline, 0)
    except Exception:  # noqa: BLE001
        return []
    found.sort(key=lambda item: item[0])
    return found


def _section_for(sections: list[tuple[int, str]], index: int) -> str:
    current = ""
    for page_index, title in sections:
        if page_index > index:
            break
        current = title
    return current


def _page_images(page: Any) -> list[Any]:
    """Large embedded images of a page (a scan is usually one image, some
    scanners write strips), as PIL images in content order."""
    images = []
    try:
        for item in page.images:
            image = getattr(item, "image", None)
            if image is None:
                continue
            width, height = image.size
            if width * height >= 250_000 and min(width, height) >= 300:
                images.append(image)
    except Exception as exc:  # noqa: BLE001 - JBIG2 and friends need external decoders
        log.debug("page images unreadable: %s", exc)
    return images


def _prepare_for_ocr(image: Any, max_side: int = 3000) -> Any:
    """RGB, at most ``max_side`` px (Windows OCR allows 10000 px, but a
    300-dpi A4 scan is 2480x3508 and larger only costs time)."""
    image = image.convert("RGB") if image.mode not in ("RGB", "L") else image
    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
    return image


def extract_pdf(path: Path, *, ocr: OcrFn | None = None, ocr_page_cap: int = 40, ocr_min_chars: int = 25,
                progress: ProgressFn | None = None, cancel: threading.Event | None = None) -> Extracted:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - listed in requirements
        raise ExtractError("the PDF reader (pypdf) is not installed") from exc
    # pypdf warns once per font it cannot fully decode (measured: 5 lines for
    # one application manual whose text still extracted fine) -- not for the log.
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    try:
        reader = PdfReader(str(path), strict=False)
        if reader.is_encrypted:
            # Many books carry only an owner password (no password to open).
            if not reader.decrypt(""):
                raise ExtractError("the PDF is password-protected")
        pages = reader.pages
        count = len(pages)
    except ExtractError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ExtractError(f"the PDF could not be opened: {type(exc).__name__}") from exc
    try:
        meta_title = reader.metadata.title if reader.metadata is not None else None
    except Exception:  # noqa: BLE001
        meta_title = None
    result = Extracted(title=_title_from(meta_title, path), kind="pdf", pages=count)
    sections = _outline_sections(reader)
    ocr_left = max(0, int(ocr_page_cap))
    failed_pages = 0
    for index in range(count):
        if cancel is not None and cancel.is_set():
            result.warnings.append("cancelled")
            break
        if progress is not None:
            progress(index + 1, count, "text")
        section = _section_for(sections, index)
        try:
            raw = pages[index].extract_text() or ""
        except Exception as exc:  # noqa: BLE001 - one bad page must not lose the book
            log.debug("page %d of %s: %s", index + 1, path.name, exc)
            raw = ""
            failed_pages += 1
        text, flipped = clean_text(raw)
        result.reversed_pages += int(flipped)
        if len(re.sub(r"\s+", "", text)) >= ocr_min_chars:
            result.blocks.append(Block(index + 1, section, text, "text"))
            continue
        # A scan (or an empty page): OCR its images while the budget lasts.
        images = _page_images(pages[index])
        if not images:
            if text:
                result.blocks.append(Block(index + 1, section, text, "text"))
            continue
        if ocr is None or ocr_left <= 0:
            # counted, so the library can say "N scanned pages were not read"
            if text:
                result.blocks.append(Block(index + 1, section, text, "text"))
            result.skipped_pages += 1
            continue
        ocr_left -= 1
        if progress is not None:
            progress(index + 1, count, "ocr")
        parts = []
        for image in images:
            try:
                parts.append(ocr(_prepare_for_ocr(image)) or "")
            except Exception as exc:  # noqa: BLE001 - OCR unavailable or failed on this image
                result.warnings.append(f"OCR failed on page {index + 1}: {type(exc).__name__}")
                break
        ocr_text, _ = clean_text("\n".join(p for p in parts if p))
        if ocr_text:
            result.ocr_pages += 1
            result.blocks.append(Block(index + 1, section, ocr_text, "ocr"))
        else:
            result.skipped_pages += 1
    if failed_pages:
        result.warnings.append(f"{failed_pages} page(s) had unreadable text")
    return result


# -- DOCX ---------------------------------------------------------------------------------------------------------
def _docx_heading_styles(archive: zipfile.ZipFile) -> set[str]:
    """Style ids that are headings: by name (Heading 1, Title...) or outline level."""
    from xml.etree import ElementTree as ET

    headings: set[str] = set()
    try:
        root = ET.fromstring(archive.read("word/styles.xml"))
    except (KeyError, ET.ParseError):
        return headings
    for style in root.iter(f"{W_NS}style"):
        style_id = style.get(f"{W_NS}styleId") or ""
        name_el = style.find(f"{W_NS}name")
        name = (name_el.get(f"{W_NS}val") if name_el is not None else "") or ""
        outline = style.find(f"{W_NS}pPr/{W_NS}outlineLvl")
        if _HEADING_STYLE.match(name.strip()) or _HEADING_STYLE.match(style_id) or (
                outline is not None and (outline.get(f"{W_NS}val") or "9").isdigit()
                and int(outline.get(f"{W_NS}val") or "9") < 9):
            headings.add(style_id)
    return headings


def _docx_core_title(archive: zipfile.ZipFile) -> str:
    from xml.etree import ElementTree as ET

    try:
        root = ET.fromstring(archive.read("docProps/core.xml"))
    except (KeyError, ET.ParseError):
        return ""
    for element in root.iter():
        if element.tag.endswith("}title") and element.text:
            return element.text.strip()
    return ""


def extract_docx(path: Path) -> Extracted:
    import zipfile
    from xml.etree import ElementTree as ET

    try:
        with zipfile.ZipFile(path) as archive:
            document = ET.fromstring(archive.read("word/document.xml"))
            headings = _docx_heading_styles(archive)
            core_title = _docx_core_title(archive)
    except (zipfile.BadZipFile, KeyError, ET.ParseError, OSError) as exc:
        raise ExtractError(f"the Word file could not be read: {type(exc).__name__}") from exc
    body = document.find(f"{W_NS}body")
    if body is None:
        raise ExtractError("the Word file has no body")
    page = 1
    saw_breaks = False
    section = ""
    first_heading = ""
    raw_blocks: list[tuple[int, str, str]] = []

    def paragraph(p: ET.Element) -> tuple[str, int, bool]:
        """(text, page breaks before/inside, is heading)."""
        nonlocal saw_breaks
        breaks = 0
        style = p.find(f"{W_NS}pPr/{W_NS}pStyle")
        style_id = style.get(f"{W_NS}val") if style is not None else ""
        outline = p.find(f"{W_NS}pPr/{W_NS}outlineLvl")
        is_heading = bool(style_id and style_id in headings) or outline is not None
        parts: list[str] = []
        for node in p.iter():
            tag = node.tag
            if tag == f"{W_NS}t" and node.text:
                parts.append(node.text)
            elif tag == f"{W_NS}tab":
                parts.append(" ")
            elif tag == f"{W_NS}br":
                if node.get(f"{W_NS}type") == "page":
                    breaks += 1
                    saw_breaks = True
                else:
                    parts.append("\n")
            elif tag == f"{W_NS}lastRenderedPageBreak":
                breaks += 1
                saw_breaks = True
        return "".join(parts).strip(), breaks, is_heading

    for element in body:
        if element.tag == f"{W_NS}p":
            items = [element]
        elif element.tag == f"{W_NS}tbl":
            # One line per table row, cells joined with " | ".
            rows = []
            for row in element.iter(f"{W_NS}tr"):
                cells = []
                for cell in row.iter(f"{W_NS}tc"):
                    cells.append(" ".join(paragraph(p)[0] for p in cell.iter(f"{W_NS}p")).strip())
                rows.append(" | ".join(c for c in cells if c))
            text = "\n".join(r for r in rows if r)
            if text:
                raw_blocks.append((page, section, text))
            continue
        else:
            continue
        for p in items:
            text, breaks, is_heading = paragraph(p)
            page += breaks
            if not text:
                continue
            if is_heading:
                section = clean_text(text)[0][:120]
                first_heading = first_heading or section
            raw_blocks.append((page, section, text))
    title = _title_from(core_title or first_heading, path)
    result = Extracted(title=title, kind="docx", pages=page if saw_breaks else 0)
    for block_page, block_section, text in raw_blocks:
        cleaned, flipped = clean_text(text)
        result.reversed_pages += int(flipped)
        if cleaned:
            result.blocks.append(Block(block_page if saw_breaks else None, block_section, cleaned))
    return result


# -- TXT / Markdown --------------------------------------------------------------------------------------------------
def read_text(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("cp1256")      # legacy Windows Arabic-script text
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")


def extract_text_file(path: Path, *, markdown: bool) -> Extracted:
    try:
        raw = read_text(path)
    except OSError as exc:
        raise ExtractError(f"the file could not be read: {type(exc).__name__}") from exc
    pages = raw.split("\f") if "\f" in raw else [raw]
    has_pages = len(pages) > 1
    title = ""
    section = ""
    blocks: list[Block] = []
    for number, page_text in enumerate(pages, start=1):
        paragraph: list[str] = []

        def flush() -> None:
            if paragraph:
                cleaned, _ = clean_text("\n".join(paragraph))
                if cleaned:
                    blocks.append(Block(number if has_pages else None, section, cleaned))
                paragraph.clear()

        for line in page_text.splitlines():
            heading = _MD_HEADING.match(line) if markdown else None
            if heading:
                flush()
                section = clean_text(heading.group(2))[0][:120]
                title = title or section
                continue
            if not line.strip():
                flush()
                continue
            paragraph.append(line)
        flush()
    result = Extracted(title=_title_from(title if markdown else "", path), kind="md" if markdown else "txt",
                       pages=len(pages) if has_pages else 0, blocks=blocks)
    return result


__all__ = ["Block", "ExtractError", "Extracted", "SUPPORTED", "extract", "extract_docx", "extract_pdf",
           "extract_text_file", "kind_of", "read_text"]
