"""The «کتێبخانە» panel page: list, add (dialog + drag-and-drop), remove,
search, progress, accessibility (Qt offscreen, real Library on a temp home)."""

from __future__ import annotations

from typing import Any

import pytest
from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QFileDialog, QLabel

from knowledge_helpers import FILES, FIXTURES, FakeOcr
from sam.events import WorkerProgress
from sam.ui.pages.library import DocRow, LibraryPage, ResultRow
from sam.ui.strings import tr
from ui_helpers import all_texts, controller, core, pump, qapp, ui_app, wait_until  # noqa: F401


@pytest.fixture
def library_app(ui_app: Any) -> Any:
    ui_app.load_packages(["sam.knowledge"])
    ui_app.knowledge._ocr_override = FakeOcr()
    return ui_app


@pytest.fixture
def page(library_app: Any, controller: Any) -> LibraryPage:
    panel = controller.ensure_panel()
    panel.show()
    panel.show_page("library")
    pump(30)
    return panel.pages["library"]


def rows(page: LibraryPage) -> list[DocRow]:
    return page.findChildren(DocRow)


def test_page_is_in_the_nav_right_after_strategies(page: LibraryPage, controller: Any) -> None:
    panel = controller.ensure_panel()
    keys = list(panel.pages)
    assert keys.index("library") == keys.index("strategies") + 1
    assert panel.nav["library"].text() == "کتێبخانە"
    assert page.layoutDirection() == Qt.LayoutDirection.RightToLeft
    assert tr("lib.empty") in all_texts(page)


def test_controls_have_sorani_accessible_names(page: LibraryPage) -> None:
    assert page.add_files_button.accessibleName() == "زیادکردنی فایل بۆ کتێبخانە"
    assert page.add_folder_button.accessibleName() == "زیادکردنی بوخچە بۆ کتێبخانە"
    assert page.search_field.accessibleName() == "پرسیار لە کتێبەکان"
    assert page.search_button.accessibleName() == "گەڕان لە کتێبەکان"
    assert page.refresh_button.accessibleDescription() == "Refresh the library list"


def test_add_files_from_the_dialog_lists_books_with_pages(page: LibraryPage, monkeypatch: Any) -> None:
    chosen = [str(FIXTURES / FILES["sorani"]), str(FIXTURES / FILES["notes"])]
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", staticmethod(lambda *a, **k: (chosen, "")))
    page.add_files_button.click()
    assert wait_until(lambda: len(rows(page)) == 2, 15.0)
    texts = all_texts(page)
    assert "ستراتیژیی ڕاماڵینی شلەمەنی لە زێڕدا" in texts
    assert any("٣ لاپەڕە" in t for t in texts)
    assert any("٢ لاپەڕەی سکانکراو خوێندرایەوە" in t for t in texts)
    assert any(t == tr("lib.status.ready") for t in texts)
    notes = next(r for r in rows(page) if "Trading notes" in r.title_label.text())
    assert notes.remove_button.accessibleName() == "لابردنی «Trading notes week one» لە کتێبخانە"
    assert page.summary_label.text() == "٢ کتێب · ٦ لاپەڕە"
    # unique objectNames give each row's buttons their own UI Automation id
    assert len({r.objectName() for r in rows(page)}) == 2


def test_drag_and_drop_accepts_documents_and_folders_only(page: LibraryPage, tmp_path: Any) -> None:
    def mime(*paths: str) -> QMimeData:
        data = QMimeData()
        data.setUrls([QUrl.fromLocalFile(p) for p in paths])
        return data

    exe = tmp_path / "setup.exe"
    exe.write_bytes(b"MZ")
    wrong = mime(str(exe))              # events do not own their QMimeData: keep a reference
    refused = QDragEnterEvent(QPointF(10, 10).toPoint(), Qt.DropAction.CopyAction, wrong,
                              Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    page.dragEnterEvent(refused)
    assert not refused.isAccepted()
    data = mime(str(FIXTURES / FILES["english"]))
    welcome = QDragEnterEvent(QPointF(10, 10).toPoint(), Qt.DropAction.CopyAction, data,
                              Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    page.dragEnterEvent(welcome)
    assert welcome.isAccepted()
    drop = QDropEvent(QPointF(10, 10), Qt.DropAction.CopyAction, data, Qt.MouseButton.LeftButton,
                      Qt.KeyboardModifier.NoModifier)
    page.dropEvent(drop)
    assert wait_until(lambda: any("Price Action Essentials" in r.title_label.text() for r in rows(page)), 15.0)


def test_search_shows_passages_with_sorani_citations(page: LibraryPage) -> None:
    page.add_paths([str(FIXTURES / FILES["english"])])
    assert wait_until(lambda: len(rows(page)) == 1, 15.0)
    page.search_field.setText("پشتگیری چییە؟")
    page.search_button.click()
    assert wait_until(lambda: page.results_box.isVisible() and page.findChildren(ResultRow), 10.0)
    first = page.findChildren(ResultRow)[0]
    labels = [lab.text() for lab in first.findChildren(QLabel)]
    assert any(t.endswith("«Price Action Essentials»، لاپەڕە ١") for t in labels)
    assert "Support is a price zone" in first.text
    page.search_field.setText("banana smoothie")
    page.search()
    assert wait_until(lambda: not page.findChildren(ResultRow) or page.results == [], 10.0)


def test_remove_takes_a_book_out_and_keeps_the_file(page: LibraryPage) -> None:
    source = FIXTURES / FILES["english"]
    page.add_paths([str(source)])
    assert wait_until(lambda: len(rows(page)) == 1, 15.0)
    rows(page)[0].remove_button.click()
    assert wait_until(lambda: not rows(page), 10.0)
    assert source.exists()
    assert tr("lib.empty") in all_texts(page)


def test_progress_of_a_library_job_shows_on_the_page(page: LibraryPage, library_app: Any) -> None:
    from sam.knowledge.library import Job

    job = Job("kb-job-1", [], "ui")
    library_app.knowledge.jobs[job.id] = job
    page.handle_event(WorkerProgress(task_id="kb-job-1", step=1, max_steps=3,
                                     text_ckb="خوێندنەوەی «book.pdf» · لاپەڕە ١/٣"))
    assert page.progress_label.isVisibleTo(page) and "لاپەڕە ١/٣" in page.progress_label.text()
    page.handle_event(WorkerProgress(task_id="someone-else", step=1, max_steps=2, text_ckb="other work"))
    assert "other work" not in page.progress_label.text()


def test_page_without_the_library_says_so(ui_app: Any, controller: Any) -> None:
    panel = controller.ensure_panel()
    panel.show_page("library")
    pump(20)
    page = panel.pages["library"]
    page.refresh()
    pump(50)
    assert tr("lib.unavailable") in all_texts(page)
    assert not page.add_files_button.isEnabled()
