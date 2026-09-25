"""Build the synthetic knowledge-library corpus (run by hand; the PDFs are
committed so tests never need a browser):

    .venv\\Scripts\\python.exe tests\\fixtures\\knowledge\\build_corpus.py

- sorani_strategy.pdf, price_action_essentials.pdf: HTML printed to PDF by
  headless Microsoft Edge (Chromium/Skia, the same PDF path as Chrome's
  "Save as PDF"; with --generate-pdf-document-outline the headings become
  bookmarks). Measured: Sorani text extracts in logical order as Arabic
  presentation forms (textfix.clean_text folds them).
- trading_notes.pdf: page 1 text; pages 2-3 are SCANS -- images rendered by
  Qt (full Arabic-script shaping) with no text layer, so only OCR can read them.

Edge runs with its own throw-away profile under work/ (the user's browser
profile is never touched).
"""

from __future__ import annotations

import base64
import html
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

from knowledge_helpers import (ENGLISH_PAGES, ENGLISH_TITLE, FILES, NOTES_TEXT_PAGE, NOTES_TITLE,  # noqa: E402
                               SCANNED_PAGES, SORANI_PAGES, SORANI_TITLE)

EDGE = Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe"
WORK = ROOT / "work" / "kb-corpus"
CSS = ("@page { size: A4; margin: 18mm; } body { font-family: 'Segoe UI', sans-serif; font-size: 13pt; "
       "line-height: 1.6; } section { page-break-after: always; } section:last-child { page-break-after: auto; }")


def _doc(title: str, pages: list[tuple[str, str]], *, rtl: bool) -> str:
    direction = ' dir="rtl" lang="ckb"' if rtl else ' lang="en"'
    parts = [f"<!doctype html><html{direction}><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
             f"<style>{CSS}</style></head><body>"]
    for number, (heading, body) in enumerate(pages):
        parts.append("<section>")
        if number == 0:
            parts.append(f"<h1>{html.escape(title)}</h1>")
        parts.append(f"<h2>{html.escape(heading)}</h2>")
        parts.extend(f"<p>{html.escape(p)}</p>" for p in body.split("\n"))
        parts.append("</section>")
    parts.append("</body></html>")
    return "".join(parts)


def _scan_png(lines: list[str]) -> bytes:
    """A 150-dpi A4 'scan': shaped text on an off-white page (Qt offscreen)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QRectF, Qt
    from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter, QTextOption

    _app = QGuiApplication.instance() or QGuiApplication(["corpus"])
    image = QImage(1240, 1754, QImage.Format.Format_Grayscale8)
    image.fill(QColor("#f3f1ea"))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    painter.setPen(QColor("#1b1b1b"))
    y = 150.0
    for number, line in enumerate(lines):
        font = QFont("Segoe UI", 1)
        font.setPixelSize(48 if number == 0 else 36)
        font.setBold(number == 0)
        painter.setFont(font)
        option = QTextOption()
        rtl = any("\u0600" <= ch <= "\u06ff" for ch in line)
        option.setTextDirection(Qt.LayoutDirection.RightToLeft if rtl else Qt.LayoutDirection.LeftToRight)
        option.setAlignment(Qt.AlignmentFlag.AlignRight if rtl else Qt.AlignmentFlag.AlignLeft)
        option.setWrapMode(QTextOption.WrapMode.WordWrap)
        painter.drawText(QRectF(110, y, 1020, 160), line, option)
        y += 110 if number == 0 else 95
    painter.end()
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    return bytes(data.data())


def _notes_html() -> str:
    heading, body = NOTES_TEXT_PAGE
    pages = [f"<section class='text'><h1>{html.escape(NOTES_TITLE)}</h1><h2>{html.escape(heading)}</h2>"
             f"<p dir='rtl' lang='ckb'>{html.escape(body)}</p></section>"]
    for lines in SCANNED_PAGES:
        png = base64.b64encode(_scan_png(lines)).decode("ascii")
        pages.append(f"<section class='scan'><img src='data:image/png;base64,{png}'></section>")
    css = ("@page { size: A4; margin: 0; } body { margin: 0; font-family: 'Segoe UI'; font-size: 13pt; } "
           "section { page-break-after: always; height: 297mm; } section:last-child { page-break-after: auto; } "
           ".text { padding: 18mm; box-sizing: border-box; } .scan img { width: 210mm; height: 297mm; display: block; }")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{html.escape(NOTES_TITLE)}</title>"
            f"<style>{css}</style></head><body>{''.join(pages)}</body></html>")


def print_pdf(source: Path, target: Path) -> None:
    if target.exists():
        target.unlink()
    profile = WORK / "edge-profile"
    command = [str(EDGE), "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
               f"--user-data-dir={profile}", "--no-pdf-header-footer", "--generate-pdf-document-outline",
               f"--print-to-pdf={target}", source.as_uri()]
    subprocess.run(command, timeout=120, check=False, capture_output=True)
    deadline = time.time() + 30
    while time.time() < deadline and not (target.exists() and target.stat().st_size > 0):
        time.sleep(0.25)
    if not target.exists():
        raise SystemExit(f"Edge did not write {target.name}")


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    documents = {
        "sorani": _doc(SORANI_TITLE, SORANI_PAGES, rtl=True),
        "english": _doc(ENGLISH_TITLE, ENGLISH_PAGES, rtl=False),
        "notes": _notes_html(),
    }
    for key, text in documents.items():
        source = WORK / f"{key}.html"
        source.write_text(text, encoding="utf-8")
        target = HERE / FILES[key]
        print_pdf(source, target)
        print(f"{target.name}: {target.stat().st_size} bytes")


if __name__ == "__main__":
    main()
