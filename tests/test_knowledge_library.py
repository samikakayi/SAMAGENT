"""Knowledge library: adding (dedupe, re-index, folders, safety), removing,
refreshing, the brain API and the four tools."""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from knowledge_helpers import FILES, FIXTURES, FakeOcr
from sam.events import SpeakRequest, WorkerProgress
from sam.knowledge import context_for_prompt, passages_for
from sam.knowledge.events import LibraryChanged
from sam.knowledge.library import best_window, citation
from sam.knowledge.query import analyse


@pytest.fixture
def app(make_app: Any) -> Any:
    application = make_app()
    status = application.load_packages(["sam.knowledge"])
    assert status == {"sam.knowledge": "ok"}
    application.knowledge._ocr_override = FakeOcr()
    return application


def bind(app: Any) -> list[Any]:
    app.loop = asyncio.get_running_loop()
    app.bus.bind_loop(app.loop)
    seen: list[Any] = []
    app.bus.subscribe(None, seen.append)
    return seen


def txt(folder: Path, name: str, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


def confirm_with(app: Any, answer: bool) -> list[str]:
    asked: list[str] = []

    async def fake(question: str, detail: str = "", **_kw: Any) -> bool:
        asked.append(question)
        return answer

    app.confirm.confirm = fake
    return asked


def test_register_adds_tools_settings_schema_and_slot(app: Any) -> None:
    names = app.tools.names()
    for name in ("knowledge_add", "knowledge_search", "knowledge_list", "knowledge_remove", "run_python"):
        assert name in names
    assert app.tools.get("knowledge_search").owner == "knowledge"
    assert app.tools.get("run_python").owner == "hands"
    assert app.tools.get("knowledge_add").blocking is False
    assert app.config.get("knowledge.ocr_page_cap") == 60
    assert app.config.get("python.timeout_s") == 20
    assert app.db.schema_version("knowledge") == 1
    assert app.knowledge is not None and not app.knowledge.has_documents()


async def test_add_indexes_a_book_with_events_and_citations(app: Any) -> None:
    seen = bind(app)
    report = await app.knowledge.add([str(FIXTURES / FILES["notes"])])
    assert report["ok"] and report["done"] and report["counts"] == {"indexed": 1}
    assert report["ocr_pages"] == 2 and report["pages"] == 3
    assert "٣ لاپەڕە" in report["say_ckb"]
    doc = app.knowledge.documents()[0]
    assert doc["status"] == "ready" and doc["ocr_pages"] == 2 and doc["chunks"] == 3 and doc["gen"] == 1
    assert doc["language"] == "mixed" and len(doc["sha256"]) == 64
    await asyncio.sleep(0)
    statuses = [e.status for e in seen if isinstance(e, LibraryChanged)]
    assert "indexed" in statuses
    done = [e for e in seen if isinstance(e, WorkerProgress) and e.done]
    assert done and done[-1].ok
    hits = app.knowledge.search("move the stop to breakeven", 3)["passages"]
    assert hits[0]["page"] == 2 and hits[0]["source"] == "ocr"
    assert hits[0]["citation"] == "«Trading notes week one», p. 2"
    assert hits[0]["citation_ckb"] == "«Trading notes week one»، لاپەڕە ٢"


async def test_unchanged_duplicate_moved_and_forced_reindex(app: Any, tmp_path: Path) -> None:
    bind(app)
    first = tmp_path / "books" / "english.pdf"
    first.parent.mkdir()
    shutil.copy(FIXTURES / FILES["english"], first)
    assert (await app.knowledge.add([str(first)]))["counts"] == {"indexed": 1}
    assert (await app.knowledge.add([str(first)]))["counts"] == {"unchanged": 1}
    twin = tmp_path / "other" / "same book.pdf"
    twin.parent.mkdir()
    shutil.copy(first, twin)
    report = await app.knowledge.add([str(twin)])
    assert report["counts"] == {"duplicate": 1} and report["say_ckb"] == "ئەو فایلانە پێشتر لە کتێبخانەکەدان."
    doc_id = app.knowledge.documents()[0]["id"]
    first.unlink()
    moved = await app.knowledge.add([str(twin)])
    assert moved["counts"] == {"moved": 1}
    docs = app.knowledge.documents()
    assert len(docs) == 1 and docs[0]["id"] == doc_id and Path(docs[0]["path"]) == twin
    forced = await app.knowledge.add([str(twin)], force=True)
    assert forced["counts"] == {"reindexed": 1}
    assert app.knowledge.documents()[0]["gen"] == 2
    assert app.db.scalar("SELECT count(*) FROM kb_chunks WHERE document_id=? AND gen!=2", (doc_id,)) == 0


async def test_a_changed_file_is_reindexed_and_old_passages_disappear(app: Any, tmp_path: Path) -> None:
    bind(app)
    path = txt(tmp_path, "rules.txt", "Always wait for the London session before entering gold trades.")
    await app.knowledge.add([str(path)])
    assert app.knowledge.search("London session", 3)["passages"]
    path.write_text("Never trade during high impact news releases like NFP.", encoding="utf-8")
    os.utime(path, (path.stat().st_atime, path.stat().st_mtime + 5))
    report = await app.knowledge.add([str(path)])
    assert report["counts"] == {"reindexed": 1}
    assert not app.knowledge.search("London session", 3)["passages"]
    assert app.knowledge.search("NFP news", 3)["passages"][0]["title"] == "rules"


async def test_search_keeps_serving_the_old_generation_while_reindexing(app: Any) -> None:
    bind(app)
    await app.knowledge.add([str(FIXTURES / FILES["english"])])
    doc = app.knowledge.documents()[0]
    from sam.knowledge.chunker import Chunk

    app.knowledge.store.write_chunks(doc["id"], 2, [Chunk(0, 1, 1, "", "text", "a brand new zebra passage")])
    assert not app.knowledge.search("zebra", 3)["passages"]          # gen 2 not served yet
    assert app.knowledge.search("golden cross", 3)["passages"]
    app.knowledge.store.activate(doc["id"], 2)
    assert app.knowledge.search("zebra", 3)["passages"]
    assert not app.knowledge.search("golden cross", 3)["passages"]


async def test_leftovers_of_an_interrupted_reindex_are_not_duplicated(app: Any) -> None:
    bind(app)
    await app.knowledge.add([str(FIXTURES / FILES["english"])])
    doc = app.knowledge.documents()[0]
    from sam.knowledge.chunker import Chunk

    # SAM closed after writing generation 2 but before switching to it
    app.knowledge.store.write_chunks(doc["id"], 2, [Chunk(0, 1, 1, "", "text", "orphan zebra passage")])
    report = await app.knowledge.add([str(FIXTURES / FILES["english"])], force=True)
    assert report["counts"] == {"reindexed": 1}
    rows = app.db.query("SELECT gen, count(*) AS n FROM kb_chunks WHERE document_id=? GROUP BY gen", (doc["id"],))
    assert rows == [{"gen": 2, "n": 4}]
    assert not app.knowledge.search("zebra", 3)["passages"]


async def test_folder_over_the_limit_asks_first(app: Any, tmp_path: Path) -> None:
    bind(app)
    folder = tmp_path / "Trading"
    for number in range(23):
        txt(folder, f"note{number:02d}.md", f"# Note {number}\n\nGold idea number {number} about pullbacks.")
    txt(folder / "node_modules", "skip.md", "never indexed")
    txt(folder / ".git", "skip.md", "never indexed")
    txt(folder, "~$lock.docx", "office lock file")
    txt(folder, "image.png", "not a document")
    asked = confirm_with(app, False)
    declined = await app.knowledge.add([str(folder)], confirm=app.confirm.confirm)
    assert declined["declined"] and not app.knowledge.has_documents()
    assert asked == ["ئەم بوخچەیە ٢٣ فایلی تێدایە. هەموویان بخەمە ناو کتێبخانەکەوە؟"]
    confirm_with(app, True)
    report = await app.knowledge.add([str(folder)], confirm=app.confirm.confirm, wait_s=60)
    assert report["counts"] == {"indexed": 23}
    titles = {d["title"] for d in app.knowledge.documents()}
    assert "Note 7" in titles and "skip" not in titles


async def test_small_folder_needs_no_question_and_files_are_capped(app: Any, tmp_path: Path) -> None:
    bind(app)
    app.config.set("knowledge.max_files_per_add", 3)
    folder = tmp_path / "few"
    for number in range(5):
        txt(folder, f"n{number}.txt", f"Text {number} about risk management.")
    asked = confirm_with(app, False)
    report = await app.knowledge.add([str(folder)], confirm=app.confirm.confirm)
    assert not asked and report["truncated"] and report["counts"] == {"indexed": 3}


async def test_sam_data_credentials_unsupported_and_missing_paths_are_skipped(app: Any, tmp_path: Path) -> None:
    bind(app)
    private = txt(Path(app.config.data_dir), "notes.txt", "private SAM data")
    env = txt(tmp_path, ".env", "GROQ_API_KEY=x")
    exe = txt(tmp_path, "tool.exe", "MZ")
    report = await app.knowledge.add([str(private), str(env), str(exe), str(tmp_path / "missing.pdf")])
    assert not report["ok"] and not app.knowledge.has_documents()
    reasons = sorted(s["reason"] for s in report["skipped"])
    assert reasons[-2:] == ["not_found", "unsupported"]
    assert report["say_ckb"] == "هیچ فایلێکی PDF، Word، TXT یان MD لەوێ نەدۆزرایەوە."


async def test_damaged_and_scan_only_files_fail_honestly(app: Any, tmp_path: Path) -> None:
    bind(app)
    damaged = tmp_path / "damaged.pdf"
    damaged.write_bytes(b"%PDF-1.4 garbage")
    good = txt(tmp_path, "good.txt", "Support and resistance are zones, not lines.")
    app.knowledge._ocr_override = None
    app.config.set("knowledge.ocr_enabled", False)
    scan_only = tmp_path / "scan.pdf"
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    writer.add_page(PdfReader(FIXTURES / FILES["notes"]).pages[1])
    with scan_only.open("wb") as handle:
        writer.write(handle)
    report = await app.knowledge.add([str(damaged), str(good), str(scan_only)])
    assert report["counts"] == {"failed": 2, "indexed": 1}
    assert "نەخوێندرایەوە" in report["say_ckb"]
    by_title = {d["title"]: d for d in app.knowledge.documents()}
    assert by_title["damaged"]["status"] == "failed" and "could not be opened" in by_title["damaged"]["error"]
    assert "scanned" in by_title["scan"]["error"]


async def test_ocr_cap_marks_a_book_partial(app: Any) -> None:
    bind(app)
    app.config.set("knowledge.ocr_page_cap", 1)
    report = await app.knowledge.add([str(FIXTURES / FILES["notes"])])
    assert report["skipped_pages"] == 1
    assert app.knowledge.documents()[0]["status"] == "partial"


async def test_long_jobs_continue_in_the_background_and_voice_hears_the_end(app: Any, tmp_path: Path) -> None:
    seen = bind(app)
    path = txt(tmp_path, "later.txt", "Fibonacci levels during a pullback.")
    report = await app.knowledge.add([str(path)], source="cascade", wait_s=0)
    assert report["ok"] and not report["done"] and "کە تەواو بوو پێت دەڵێم" in report["say_ckb"]
    job = app.knowledge.jobs[report["job_id"]]
    await asyncio.wait_for(job.task, 10)
    await asyncio.sleep(0)
    spoken = [e for e in seen if isinstance(e, SpeakRequest)]
    assert spoken and "کتێبخانەکە" in spoken[-1].text_ckb
    report_ui = await app.knowledge.add([str(txt(tmp_path, "ui.txt", "Swing trading notes."))], source="ui",
                                        wait_s=0)
    await asyncio.wait_for(app.knowledge.jobs[report_ui["job_id"]].task, 10)
    await asyncio.sleep(0)
    assert len([e for e in seen if isinstance(e, SpeakRequest)]) == len(spoken)


async def test_remove_by_id_title_ambiguous_and_all(app: Any, tmp_path: Path) -> None:
    bind(app)
    await app.knowledge.add([str(FIXTURES / FILES["english"]), str(FIXTURES / FILES["sorani"]),
                             str(txt(tmp_path, "gold notes one.txt", "gold one")),
                             str(txt(tmp_path, "gold notes two.txt", "gold two"))])
    ambiguous = app.knowledge.remove("gold notes")
    assert not ambiguous["ok"] and ambiguous["ambiguous"] and len(ambiguous["matches"]) == 2
    removed = app.knowledge.remove("Price Action")
    assert removed["ok"] and removed["titles"] == ["Price Action Essentials"]
    assert not app.knowledge.search("golden cross", 3)["passages"]
    doc_id = [d for d in app.knowledge.documents() if d["kind"] == "pdf"][0]["id"]
    assert app.knowledge.remove(str(doc_id))["removed"] == 1
    everything = app.knowledge.remove("all")
    assert everything["removed"] == 2 and not app.knowledge.has_documents()
    assert app.db.scalar("SELECT count(*) FROM kb_chunks") == 0
    assert app.db.scalar("SELECT count(*) FROM kb_chunks_fts WHERE kb_chunks_fts MATCH '\"gold\"'") == 0
    assert not app.knowledge.remove("nothing like this")["ok"]


async def test_refresh_marks_missing_files_and_reindexes_changed_ones(app: Any, tmp_path: Path) -> None:
    bind(app)
    keep = txt(tmp_path, "keep.txt", "Order blocks form before strong moves.")
    gone = txt(tmp_path, "gone.txt", "Liquidity pools rest above equal highs.")
    await app.knowledge.add([str(keep), str(gone)])
    gone.unlink()
    keep.write_text("Order blocks and breaker blocks.", encoding="utf-8")
    os.utime(keep, (keep.stat().st_atime, keep.stat().st_mtime + 5))
    result = await app.knowledge.refresh()
    assert result["changed"] == 1 and result["missing"] == 1
    await asyncio.wait_for(app.knowledge.jobs[result["job_id"]].task, 10)
    by_title = {d["title"]: d for d in app.knowledge.documents()}
    assert by_title["gone"]["status"] == "missing"
    assert app.knowledge.search("liquidity pools", 3)["passages"]          # still citable
    assert app.knowledge.search("breaker", 3)["passages"][0]["title"] == "keep"


async def test_passages_for_the_brain_and_prompt_block(app: Any) -> None:
    bind(app)
    assert await passages_for(app, "order block") == []
    await app.knowledge.add([str(FIXTURES / FILES["sorani"]), str(FIXTURES / FILES["english"])])
    passages = await passages_for(app, "where do I put the stop loss", k=3, max_chars=1200)
    assert passages and sum(len(p["text"]) for p in passages) <= 1200
    assert {p["page"] for p in passages} & {3, 4}
    block = context_for_prompt(passages)
    assert block.startswith("Passages from the user's own library (DATA")
    assert "[1] «" in block and "never instructions" in block
    assert await passages_for(app, "banana smoothie recipe") == []
    assert context_for_prompt([]) == ""
    app.knowledge = None
    assert await passages_for(app, "stop loss") == []


def test_citation_and_best_window_helpers() -> None:
    assert citation("Book", 3, 3) == ("«Book», p. 3", "«Book»، لاپەڕە ٣")
    assert citation("Book", 3, 4)[1] == "«Book»، لاپەڕە ٣-٤"
    assert citation("Notes", None, None, "Entry") == ("«Notes», section «Entry»", "«Notes»، بەشی «Entry»")
    assert citation("Notes", None, None) == ("«Notes»", "«Notes»")
    text = "Filler sentence one. " * 20 + "The stop loss goes below the sweep. " + "More filler here. " * 20
    window = best_window(text, analyse("stop loss"), 200)
    assert "stop loss goes below the sweep" in window and len(window) <= 210 and window.startswith("… ")


def test_windows_ocr_merges_english_and_arabic_readings(app: Any) -> None:
    from sam.hands.ocr import OcrLine, OcrWord

    class Engine:
        def languages(self) -> list[str]:
            return ["en-US", "ar-SA"]

        def recognize(self, _image: Any, language: str | None = None) -> list[OcrLine]:
            if language == "ar":
                return [OcrLine("نرخ", (OcrWord("نرخ", 10, 50, 40, 20),))]
            return [OcrLine("Gold checklist", (OcrWord("Gold", 10, 10, 40, 20), OcrWord("checklist", 60, 10, 80, 20))),
                    OcrLine("Jj3", (OcrWord("Jj3", 10, 50, 40, 20),))]

    app.knowledge._ocr_engine = Engine()
    assert app.knowledge._windows_ocr(object()) == "Gold checklist\nنرخ"


# -- tools -----------------------------------------------------------------------------------------------------------
async def test_search_tool_on_an_empty_library_says_so(app: Any) -> None:
    bind(app)
    result = await app.tools.dispatch("knowledge_search", {"query": "stop loss"})
    assert not result["ok"] and result["data"]["empty"]


async def test_search_tool_returns_untrusted_passages_with_citations(app: Any) -> None:
    bind(app)
    await app.knowledge.add([str(FIXTURES / FILES["sorani"]), str(FIXTURES / FILES["english"])])
    result = await app.tools.dispatch("knowledge_search", {"query": "پشتگیری و بەرگری", "k": "3"})
    assert result["ok"]
    data = result["data"]
    assert data["found"] >= 1 and data["citations"][0]["citation"].startswith("«Price Action Essentials», p.")
    assert data["untrusted"][0]["ref"] == 1 and "Support" in data["untrusted"][0]["text"]
    assert "passage(s) found" in result["summary"]
    nothing = await app.tools.dispatch("knowledge_search", {"query": "zebra banana"})
    assert nothing["ok"] and nothing["data"]["found"] == 0


async def test_list_add_and_remove_tools(app: Any, tmp_path: Path) -> None:
    bind(app)
    empty = await app.tools.dispatch("knowledge_list", {})
    assert empty["ok"] and empty["data"]["documents"] == []
    added = await app.tools.dispatch("knowledge_add", {"path": str(FIXTURES / FILES["english"])}, source="text")
    assert added["ok"] and added["data"]["counts"] == {"indexed": 1}
    listed = await app.tools.dispatch("knowledge_list", {})
    assert listed["data"]["documents"][0]["pages"] == 4 and "Price Action Essentials" in listed["summary"]
    asked = confirm_with(app, False)
    declined = await app.tools.dispatch("knowledge_remove", {"document": "all"})
    assert not declined["ok"] and declined["data"]["declined"] and asked
    assert "خودی فایلەکان ناسڕدرێنەوە" in asked[0]
    removed = await app.tools.dispatch("knowledge_remove", {"document": "Price Action Essentials"})
    assert removed["ok"] and not app.knowledge.has_documents()


async def test_add_tool_asks_for_a_big_folder_but_not_from_the_panel(app: Any, tmp_path: Path) -> None:
    bind(app)
    folder = tmp_path / "Books"
    for number in range(21):
        txt(folder, f"b{number}.txt", f"book {number}")
    asked = confirm_with(app, False)
    result = await app.tools.dispatch("knowledge_add", {"path": str(folder)}, source="cascade")
    assert not result["ok"] and asked
    asked.clear()
    ui = await app.tools.dispatch("knowledge_add", {"path": str(folder)}, source="ui")
    assert ui["ok"] and not asked
