"""Knowledge library: reading PDF (text layer, outline, scanned pages + OCR
cap), DOCX, TXT and Markdown."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from knowledge_helpers import (ENGLISH_TITLE, FILES, FIXTURES, NOTES_TITLE, SCANNED_PAGES, SORANI_PAGES,
                               SORANI_TITLE, FakeOcr, make_docx)
from sam.knowledge.extract import ExtractError, extract, kind_of, read_text


def test_sorani_pdf_text_layer_title_pages_and_sections() -> None:
    result = extract(FIXTURES / FILES["sorani"])
    assert result.kind == "pdf" and result.pages == 3
    assert result.title == SORANI_TITLE
    assert result.language == "ckb"
    assert [b.page for b in result.blocks] == [1, 2, 3]
    # the PDF bookmarks name each page's section
    assert [b.section for b in result.blocks] == [p[0] for p in SORANI_PAGES]
    assert "ستۆپ لۆس هەمیشە لە ژێر نزمترین خاڵی ڕاماڵینەکە دابنێ" in result.blocks[2].text
    assert result.ocr_pages == 0 and result.skipped_pages == 0


def test_english_pdf_reads_every_page() -> None:
    result = extract(FIXTURES / FILES["english"])
    assert result.title == ENGLISH_TITLE and result.pages == 4 and result.language == "en"
    assert "golden cross" in result.blocks[2].text


def test_scanned_pages_go_to_ocr_one_image_per_page() -> None:
    ocr = FakeOcr()
    progress: list[tuple[int, int, str]] = []
    result = extract(FIXTURES / FILES["notes"], ocr=ocr, progress=lambda *a: progress.append(a))
    assert result.title == NOTES_TITLE and result.pages == 3
    assert result.ocr_pages == 2 and result.skipped_pages == 0
    assert [b.source for b in result.blocks] == ["text", "ocr", "ocr"]
    assert result.blocks[1].text.splitlines()[0] == SCANNED_PAGES[0][0]
    assert len(ocr.calls) == 2 and all(max(size) <= 3000 for size in ocr.calls)
    assert ("ocr" in {stage for _, _, stage in progress}) and progress[-1][:2] == (3, 3)


def test_ocr_cap_and_missing_ocr_skip_scanned_pages() -> None:
    capped = extract(FIXTURES / FILES["notes"], ocr=FakeOcr(), ocr_page_cap=1)
    assert capped.ocr_pages == 1 and capped.skipped_pages == 1
    no_ocr = extract(FIXTURES / FILES["notes"], ocr=None)
    assert [b.page for b in no_ocr.blocks] == [1] and no_ocr.skipped_pages == 2


def test_ocr_failure_is_a_warning_not_a_crash() -> None:
    def broken(_image: object) -> str:
        raise RuntimeError("Windows has no installed OCR language pack.")

    result = extract(FIXTURES / FILES["notes"], ocr=broken)
    assert result.ocr_pages == 0 and result.skipped_pages == 2
    assert any("OCR failed on page 2" in w for w in result.warnings)


def test_damaged_and_unsupported_files_raise_extract_error(tmp_path: Path) -> None:
    damaged = tmp_path / "broken.pdf"
    damaged.write_bytes(b"%PDF-1.7 not really a pdf")
    with pytest.raises(ExtractError):
        extract(damaged)
    with pytest.raises(ExtractError):
        extract(tmp_path / "picture.png")
    assert kind_of(Path("a.MD")) == "md" and kind_of(Path("a.docx")) == "docx" and kind_of(Path("a.exe")) is None


def test_encrypted_pdf_with_a_password_is_refused(tmp_path: Path) -> None:
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    for page in PdfReader(FIXTURES / FILES["english"]).pages[:1]:
        writer.add_page(page)
    writer.encrypt(user_password="secret-pass", owner_password="owner")
    locked = tmp_path / "locked.pdf"
    with locked.open("wb") as handle:
        writer.write(handle)
    with pytest.raises(ExtractError, match="password"):
        extract(locked)


def test_docx_headings_title_and_page_breaks(tmp_path: Path) -> None:
    path = make_docx(tmp_path / "rules.docx", [
        ("Title", "یاساکانی بازرگانی"),
        ("Heading1", "بەشی یەکەم"),
        ("", "هەرگیز بێ ستۆپ لۆس مامەڵە مەکە."),
        ("PAGE", ""),
        ("Heading1", "Risk"),
        ("", "Risk one percent per trade."),
    ], title="Trading Rules")
    result = extract(path)
    assert result.kind == "docx" and result.title == "Trading Rules"
    assert result.pages == 2
    body = [b for b in result.blocks if b.text in ("هەرگیز بێ ستۆپ لۆس مامەڵە مەکە.", "Risk one percent per trade.")]
    assert [(b.page, b.section) for b in body] == [(1, "بەشی یەکەم"), (2, "Risk")]


def test_docx_without_page_marks_has_no_page_numbers(tmp_path: Path) -> None:
    path = make_docx(tmp_path / "notes.docx", [("Heading1", "Notes"), ("", "Only text here.")])
    result = extract(path)
    assert result.pages == 0 and all(b.page is None for b in result.blocks)
    assert result.title == "Notes"


def test_text_files_form_feeds_are_pages_and_legacy_encodings_decode(tmp_path: Path) -> None:
    paged = tmp_path / "book.txt"
    paged.write_text("page one text about gold\fpage two text about risk", encoding="utf-8")
    result = extract(paged)
    assert result.pages == 2 and [b.page for b in result.blocks] == [1, 2]
    legacy = tmp_path / "old.txt"
    legacy.write_bytes("نرخ و بازار".encode("cp1256"))
    assert read_text(legacy) == "نرخ و بازار"
    utf16 = tmp_path / "wide.txt"
    utf16.write_text("زێڕ", encoding="utf-16")
    assert read_text(utf16) == "زێڕ"


def test_markdown_headings_become_sections_and_title(tmp_path: Path) -> None:
    path = tmp_path / "strategy.md"
    path.write_text("# My Gold Strategy\n\nIntro line.\n\n## Entry\n\nEnter after the sweep.\n", encoding="utf-8")
    result = extract(path)
    assert result.title == "My Gold Strategy" and result.pages == 0
    assert [(b.section, b.text) for b in result.blocks] == [("My Gold Strategy", "Intro line."),
                                                           ("Entry", "Enter after the sweep.")]


def test_cancel_stops_between_pages(tmp_path: Path) -> None:
    import threading

    cancel = threading.Event()
    cancel.set()
    copy = tmp_path / "copy.pdf"
    shutil.copy(FIXTURES / FILES["english"], copy)
    result = extract(copy, cancel=cancel)
    assert result.blocks == [] and "cancelled" in result.warnings
