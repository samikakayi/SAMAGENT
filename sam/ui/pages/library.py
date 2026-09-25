"""کتێبخانە — the user's books and documents (the knowledge library).

Add files (file dialog, drag-and-drop anywhere on the page) or a whole folder,
see each document with its pages, passages, OCR'd pages and status, read one
again, open it, take it out of the library, and ask the books a question to
check what SAM will find (passages with «title», page N).

Threading: the list is read with ``app.knowledge.documents()`` in a worker
thread (``bridge.run``); adding and searching run on the core loop
(``bridge.call``: the library spawns its job there). Progress arrives as
``WorkerProgress`` events of the library's jobs and ``LibraryChanged``.

Strings and the icon are registered here (``STRINGS.setdefault`` /
``ICONS.setdefault``) so the page is self-contained; the panel only lists it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout,
                               QWidget)

from ...events import WorkerProgress
from .. import theme
from ..strings import STRINGS, ckb_digits, tr, tr_or
from ..widgets import A_RIGHT, Badge, EmptyState, IconLabel, accessible, is_rtl
from . import SCROLL_GUTTER, Page, scroll_area

theme.ICONS.setdefault("library", "\ue82d")          # Segoe "Dictionary" (a book)

LIBRARY_STRINGS: dict[str, tuple[str, str]] = {
    "tab.library": ("کتێبخانە", "Library"),
    "tab.library.sub": ("کتێب و بەڵگەنامەکانت؛ سام بە سەرچاوە و ژمارەی لاپەڕەوە وەڵام دەداتەوە.",
                        "Your books and documents; SAM answers with sources and page numbers."),
    "lib.add_files": ("زیادکردنی فایل", "Add files"),
    "lib.add_folder": ("زیادکردنی بوخچە", "Add folder"),
    "lib.refresh": ("نوێکردنەوە", "Refresh"),
    "lib.empty": ("هێشتا هیچ کتێبێک زیاد نەکراوە. فایلی PDF، Word، TXT یان Markdown ڕابکێشە ئێرە، یان کرتە لە "
                  "«زیادکردنی فایل» بکە.",
                  "No books yet. Drag PDF, Word, TXT or Markdown files here, or click “Add files”."),
    "lib.drop_hint": ("دەتوانیت فایل یان بوخچە ڕابکێشیتە ئێرە.", "You can drag files or folders here."),
    "lib.search_placeholder": ("پرسیارێک لە کتێبەکانت بکە…", "Ask your books a question…"),
    "lib.search": ("گەڕان", "Search"),
    "lib.no_results": ("لە کتێبەکانتدا شتێکی پەیوەندیدار نەدۆزرایەوە.", "Nothing relevant was found in your books."),
    "lib.results_title": ("ئەوەی سام دەیدۆزێتەوە", "What SAM finds"),
    "lib.docs_title": ("کتێب و بەڵگەنامەکان", "Books and documents"),
    "lib.summary": ("{docs} کتێب · {pages} لاپەڕە", "{docs} documents · {pages} pages"),
    "lib.pages": ("{n} لاپەڕە", "{n} pages"),
    "lib.passages": ("{n} بڕگە", "{n} passages"),
    "lib.ocr_pages": ("{n} لاپەڕەی سکانکراو خوێندرایەوە", "{n} scanned pages read"),
    "lib.skipped": ("{n} لاپەڕەی سکانکراو نەخوێندرایەوە", "{n} scanned pages not read"),
    "lib.status.ready": ("ئامادە", "Ready"),
    "lib.status.partial": ("بەشێکی خوێندرایەوە", "Partly read"),
    "lib.status.indexing": ("دەخوێندرێتەوە…", "Reading…"),
    "lib.status.queued": ("لە ڕیزدایە", "Queued"),
    "lib.status.failed": ("نەخوێندرایەوە", "Not read"),
    "lib.status.missing": ("فایلەکە نەماوە", "File missing"),
    "lib.remove": ("لابردن لە کتێبخانە (فایلەکە ناسڕدرێتەوە)", "Remove from the library (the file stays)"),
    "lib.reindex": ("دووبارە خوێندنەوە", "Read again"),
    "lib.open": ("کردنەوەی فایل", "Open the file"),
    "lib.dialog.files": ("هەڵبژاردنی کتێب و بەڵگەنامە", "Choose books and documents"),
    "lib.dialog.folder": ("هەڵبژاردنی بوخچە", "Choose a folder"),
    "lib.dialog.filter": ("بەڵگەنامەکان (*.pdf *.docx *.txt *.md *.markdown)",
                          "Documents (*.pdf *.docx *.txt *.md *.markdown)"),
    "lib.adding": ("زیادکردن…", "Adding…"),
    "lib.unavailable": ("بەشی کتێبخانە ئامادە نییە.", "The library is not available."),
    "lib.failed": ("کارەکە سەرکەوتوو نەبوو.", "That did not work."),
    "a11y.lib.add_files": ("زیادکردنی فایل بۆ کتێبخانە", "Add files to the library"),
    "a11y.lib.add_folder": ("زیادکردنی بوخچە بۆ کتێبخانە", "Add a folder to the library"),
    "a11y.lib.refresh": ("نوێکردنەوەی لیستی کتێبخانە", "Refresh the library list"),
    "a11y.lib.search_field": ("پرسیار لە کتێبەکان", "Question for your books"),
    "a11y.lib.search": ("گەڕان لە کتێبەکان", "Search your books"),
    "a11y.lib.remove": ("لابردنی «{title}» لە کتێبخانە", "Remove “{title}” from the library"),
    "a11y.lib.reindex": ("دووبارە خوێندنەوەی «{title}»", "Read “{title}” again"),
    "a11y.lib.open": ("کردنەوەی «{title}»", "Open “{title}”"),
    "a11y.lib.row": ("کتێبی «{title}»", "Document “{title}”"),
}
for _key, _value in LIBRARY_STRINGS.items():
    STRINGS.setdefault(_key, _value)

SUPPORTED = (".pdf", ".docx", ".txt", ".md", ".markdown")
STATUS_COLORS = {"ready": theme.SUCCESS, "partial": theme.WARNING, "indexing": theme.INFO, "queued": theme.INFO,
                 "failed": theme.DANGER, "missing": theme.TEXT_FAINT}


def _droppable(path: str) -> bool:
    return os.path.isdir(path) or path.lower().endswith(SUPPORTED)


class DocRow(QFrame):
    """One document: title, kind, status, counts, path, and actions."""

    def __init__(self, doc: dict[str, Any], page: "LibraryPage", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.doc_id = int(doc["id"])
        self.path = str(doc.get("path") or "")
        title = str(doc.get("title") or Path(self.path).stem)
        # A per-document objectName: Qt builds the UI Automation id from the
        # objectName chain, so each row's buttons get their own id.
        self.setObjectName(f"DocRow_{self.doc_id}")
        self.setStyleSheet(f"#DocRow_{self.doc_id} {{ background: {theme.SURFACE}; "
                           f"border: 1px solid {theme.BORDER_SOFT}; border-radius: 12px; }}")
        accessible(self, "a11y.lib.row", title=title)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(12)
        status = str(doc.get("status") or "queued")
        lay.addWidget(IconLabel("library", 18, STATUS_COLORS.get(status, theme.ACCENT)))
        col = QVBoxLayout()
        col.setSpacing(3)
        top = QHBoxLayout()
        top.setSpacing(8)
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-size: 14px; font-weight: 650;")
        self.title_label.setLayoutDirection(Qt.LayoutDirection.RightToLeft if is_rtl(title)
                                            else Qt.LayoutDirection.LeftToRight)
        top.addWidget(self.title_label)
        top.addWidget(Badge(str(doc.get("kind") or "").upper(), theme.CYAN))
        self.status_badge = Badge(tr_or(f"lib.status.{status}", status), STATUS_COLORS.get(status, theme.ACCENT))
        top.addWidget(self.status_badge)
        top.addStretch(1)
        col.addLayout(top)
        facts = []
        if doc.get("pages"):
            facts.append(tr("lib.pages", n=ckb_digits(doc["pages"])))
        if doc.get("chunks"):
            facts.append(tr("lib.passages", n=ckb_digits(doc["chunks"])))
        if doc.get("ocr_pages"):
            facts.append(tr("lib.ocr_pages", n=ckb_digits(doc["ocr_pages"])))
        if doc.get("skipped_pages"):
            facts.append(tr("lib.skipped", n=ckb_digits(doc["skipped_pages"])))
        self.facts_label = QLabel("\u200f" + " · ".join(facts))
        self.facts_label.setObjectName("Muted")
        self.facts_label.setAlignment(A_RIGHT)
        col.addWidget(self.facts_label)
        if doc.get("error") and status in ("failed", "partial"):
            error = QLabel(str(doc["error"]))
            error.setObjectName("Faint")
            error.setWordWrap(True)
            col.addWidget(error)
        path_label = QLabel(self.path)
        path_label.setObjectName("Faint")
        path_label.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        col.addWidget(path_label)
        lay.addLayout(col, 1)
        self.open_button = self._icon_button("link", "lib.open", "a11y.lib.open", title,
                                             lambda: page.open_document(self.path))
        self.reindex_button = self._icon_button("refresh", "lib.reindex", "a11y.lib.reindex", title,
                                                lambda: page.reindex(self.path))
        self.remove_button = self._icon_button("delete", "lib.remove", "a11y.lib.remove", title,
                                               lambda: page.remove(self.doc_id))
        for button in (self.open_button, self.reindex_button, self.remove_button):
            lay.addWidget(button, 0, Qt.AlignmentFlag.AlignVCenter)

    @staticmethod
    def _icon_button(icon: str, tip: str, a11y: str, title: str, slot: Any) -> QPushButton:
        button = QPushButton(theme.ICONS[icon])
        button.setObjectName("IconButton")          # style selector
        button.setToolTip(tr(tip))
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(slot)
        accessible(button, a11y, title=title)     # Sorani UIA name + English description
        return button


class ResultRow(QFrame):
    def __init__(self, passage: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ResultRow")
        self.setStyleSheet(f"#ResultRow {{ background: {theme.SURFACE_2}; border-radius: 10px; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(4)
        # RLM: a Sorani citation that starts with an English title is still a
        # right-to-left line («Title»، لاپەڕە ٤), not an LTR one.
        cite = QLabel("\u200f" + str(passage.get("citation_ckb") or passage.get("citation") or ""))
        cite.setObjectName("CardTitle")
        cite.setAlignment(A_RIGHT)
        lay.addWidget(cite)
        text = str(passage.get("text") or "")
        body = QLabel(text)
        body.setWordWrap(True)
        body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        body.setLayoutDirection(Qt.LayoutDirection.RightToLeft if is_rtl(text) else Qt.LayoutDirection.LeftToRight)
        body.setAlignment(Qt.AlignmentFlag.AlignLeading)
        lay.addWidget(body)
        self.text = text


class LibraryPage(Page):
    key = "library"

    def __init__(self, app: Any, bridge: Any, parent: QWidget | None = None) -> None:
        super().__init__(app, bridge, parent)
        self.setAcceptDrops(True)
        self.docs: list[dict[str, Any]] = []
        self.results: list[dict[str, Any]] = []
        self.progress_text = ""
        self.add_files_button = accessible(QPushButton(tr("lib.add_files")), "a11y.lib.add_files")
        self.add_files_button.setObjectName("Primary")
        self.add_files_button.clicked.connect(self.choose_files)
        self.add_folder_button = accessible(QPushButton(tr("lib.add_folder")), "a11y.lib.add_folder")
        self.add_folder_button.setObjectName("Ghost")
        self.add_folder_button.clicked.connect(self.choose_folder)
        self.refresh_button = accessible(QPushButton(theme.ICONS["refresh"]), "a11y.lib.refresh")
        self.refresh_button.setObjectName("IconButton")
        self.refresh_button.setToolTip(tr("lib.refresh"))
        self.refresh_button.clicked.connect(self.refresh_files)
        self.add_header([self.refresh_button, self.add_folder_button, self.add_files_button])

        self.summary_label = QLabel("")
        self.summary_label.setObjectName("Faint")
        self.progress_label = QLabel("")
        self.progress_label.setObjectName("Muted")
        self.progress_label.setWordWrap(True)
        self.progress_label.hide()
        status_row = QHBoxLayout()
        status_row.addWidget(self.summary_label)
        status_row.addStretch(1)
        status_row.addWidget(self.progress_label)
        self.root.addLayout(status_row)

        search_row = QHBoxLayout()
        search_row.setSpacing(8)
        self.search_field = accessible(QLineEdit(), "a11y.lib.search_field", object_name="LibrarySearchField")
        self.search_field.setPlaceholderText(tr("lib.search_placeholder"))
        self.search_field.returnPressed.connect(self.search)
        self.search_button = accessible(QPushButton(tr("lib.search")), "a11y.lib.search")
        self.search_button.clicked.connect(self.search)
        search_row.addWidget(self.search_field, 1)
        search_row.addWidget(self.search_button)
        self.root.addLayout(search_row)

        self.results_box = QFrame()
        self.results_box.setObjectName("Card")
        results_lay = QVBoxLayout(self.results_box)
        results_lay.setContentsMargins(14, 12, 14, 12)
        results_lay.setSpacing(8)
        title = QLabel(tr("lib.results_title"))
        title.setObjectName("CardTitle")
        results_lay.addWidget(title)
        self.results_list = QVBoxLayout()
        self.results_list.setSpacing(6)
        results_lay.addLayout(self.results_list)
        self.results_box.hide()
        self.root.addWidget(self.results_box)

        inner = QWidget()
        self.list_lay = QVBoxLayout(inner)
        self.list_lay.setContentsMargins(SCROLL_GUTTER, 0, 0, 0)      # RTL scroll bar is on the left
        self.list_lay.setSpacing(8)
        self.docs_title = QLabel(tr("lib.docs_title"))
        self.docs_title.setObjectName("CardTitle")
        self.list_lay.addWidget(self.docs_title)
        self.rows_box = QVBoxLayout()
        self.rows_box.setSpacing(8)
        self.list_lay.addLayout(self.rows_box)
        self.list_lay.addStretch(1)
        self.root.addWidget(scroll_area(inner), 1)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(400)
        self._refresh_timer.timeout.connect(self.refresh)
        self._render()

    # -- data ----------------------------------------------------------------------------------------------------
    @property
    def library(self) -> Any:
        return getattr(self.app, "knowledge", None)

    def on_shown(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        library = self.library
        if library is None:
            self._render()
            return
        self.bridge.run(library.documents, on_ok=self._loaded, on_err=lambda _e: None)

    def _loaded(self, docs: Any) -> None:
        self.docs = list(docs or [])
        self._render()

    def _render(self) -> None:
        while self.rows_box.count():
            widget = self.rows_box.takeAt(0).widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        if self.library is None:
            self.rows_box.addWidget(EmptyState(tr("lib.unavailable"), "library"))
        elif not self.docs:
            self.rows_box.addWidget(EmptyState(tr("lib.empty"), "library"))
        for doc in self.docs:
            self.rows_box.addWidget(DocRow(doc, self))
        pages = sum(int(d.get("pages") or 0) for d in self.docs)
        self.summary_label.setText(tr("lib.summary", docs=ckb_digits(len(self.docs)), pages=ckb_digits(pages))
                                   if self.docs else tr("lib.drop_hint"))
        enabled = self.library is not None
        for widget in (self.add_files_button, self.add_folder_button, self.search_button, self.search_field):
            widget.setEnabled(enabled)

    # -- actions -------------------------------------------------------------------------------------------------
    def add_paths(self, paths: list[str], *, force: bool = False) -> None:
        library = self.library
        paths = [p for p in paths if p]
        if library is None or not paths:
            return
        self._show_progress(tr("lib.adding"))
        self.bridge.call(library.add(paths, source="ui", wait_s=0, force=force), on_ok=self._added,
                         on_err=lambda _e: self._show_progress(tr("lib.failed")))

    def _added(self, report: Any) -> None:
        if isinstance(report, dict) and not report.get("ok") and report.get("say_ckb"):
            self._show_progress(str(report["say_ckb"]))
        self._refresh_timer.start()

    def choose_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, tr("lib.dialog.files"), str(Path.home()),
                                                tr("lib.dialog.filter"))
        self.add_paths(list(files))

    def choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, tr("lib.dialog.folder"), str(Path.home()))
        if folder:
            self.add_paths([folder])

    def refresh_files(self) -> None:
        library = self.library
        if library is not None:
            self.bridge.call(library.refresh(), on_ok=lambda _r: self._refresh_timer.start(),
                             on_err=lambda _e: self._refresh_timer.start())

    def reindex(self, path: str) -> None:
        self.add_paths([path], force=True)

    def remove(self, doc_id: int) -> None:
        library = self.library
        if library is not None:
            self.bridge.run(library.remove, str(doc_id), on_ok=lambda _r: self.refresh(),
                            on_err=lambda _e: self.refresh())

    def open_document(self, path: str) -> None:
        if path and os.path.exists(path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def search(self) -> None:
        library = self.library
        question = self.search_field.text().strip()
        if library is None or not question:
            return
        self.search_button.setEnabled(False)
        self.bridge.call(library.asearch(question, 5, max_chars=600), on_ok=self._show_results,
                         on_err=lambda _e: self._show_results({"passages": []}))

    def _show_results(self, result: Any) -> None:
        self.search_button.setEnabled(self.library is not None)
        self.results = list((result or {}).get("passages") or [])
        while self.results_list.count():
            widget = self.results_list.takeAt(0).widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        if not self.results:
            empty = QLabel(tr("lib.no_results"))
            empty.setObjectName("Muted")
            self.results_list.addWidget(empty)
        for passage in self.results:
            self.results_list.addWidget(ResultRow(passage))
        self.results_box.show()

    def _show_progress(self, text: str) -> None:
        self.progress_text = text
        self.progress_label.setText(text)
        self.progress_label.setVisible(bool(text))

    # -- events ------------------------------------------------------------------------------------------------------
    def handle_event(self, ev: Any) -> None:
        from ...knowledge.events import LibraryChanged

        if isinstance(ev, LibraryChanged):
            if self.isVisible():
                self._refresh_timer.start()
        elif isinstance(ev, WorkerProgress):
            library = self.library
            jobs = getattr(library, "jobs", {}) if library is not None else {}
            if ev.task_id in jobs:
                text = ev.text_ckb
                self._show_progress(text)
                if ev.done:
                    self._refresh_timer.start()
                    # the final sentence stays a few seconds, then the line clears
                    QTimer.singleShot(8000, lambda: self.progress_text == text and self._show_progress(""))

    # -- drag and drop -----------------------------------------------------------------------------------------------
    def dragEnterEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        mime = event.mimeData()
        if mime.hasUrls() and any(u.isLocalFile() and _droppable(u.toLocalFile()) for u in mime.urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: Any) -> None:  # noqa: N802
        self.dragEnterEvent(event)

    def dropEvent(self, event: Any) -> None:  # noqa: N802
        mime = event.mimeData()
        paths = [u.toLocalFile() for u in mime.urls() if u.isLocalFile() and _droppable(u.toLocalFile())]
        if paths:
            event.acceptProposedAction()
            self.add_paths(paths)
        else:
            event.ignore()


__all__ = ["LibraryPage", "DocRow", "ResultRow", "LIBRARY_STRINGS"]
