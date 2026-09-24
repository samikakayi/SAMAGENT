"""Knowledge library: retrieval quality on the synthetic corpus (3 PDFs: a
Sorani strategy, an English price-action book, notes with two scanned pages).

Measured 2026-09-24 (acceptance/knowledge_retrieval.py, 28 questions:
Sorani, English, cross-language, scanned pages, Arabic-keyboard spellings):
top-1 96%, top-3 100% -- the same with the real Windows OCR reading the
scans; without the Sorani<->English glossary 86% / 89%, without the OCR
fold 86% / 89%, without stems 93% / 96% (real OCR). The thresholds below
leave room for one miss each.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from knowledge_helpers import FILES, FIXTURES, QUESTIONS, FakeOcr, hit_rate


@pytest.fixture
async def corpus(make_app: Any) -> tuple[Any, dict[int, str]]:
    app = make_app()
    app.load_packages(["sam.knowledge"])
    app.loop = asyncio.get_running_loop()
    app.bus.bind_loop(app.loop)
    app.knowledge._ocr_override = FakeOcr()
    report = await app.knowledge.add([str(FIXTURES / name) for name in FILES.values()], wait_s=120)
    assert report["counts"] == {"indexed": 3}
    keys = {int(r["document_id"]): key for r in report["results"] for key, name in FILES.items()
            if Path(r["path"]).name == name}
    return app, keys


async def test_top3_hit_rate_on_the_corpus(corpus: tuple[Any, dict[int, str]]) -> None:
    app, keys = corpus
    result = hit_rate(app.knowledge, keys, k=3)
    misses = [(r["question"], r["found"]) for r in result["rows"] if not r["topk"]]
    assert result["n"] == len(QUESTIONS) == 28
    assert result["topk"] >= 0.96, misses
    assert result["top1"] >= 0.89


async def test_cross_language_questions_find_the_other_language(corpus: tuple[Any, dict[int, str]]) -> None:
    app, keys = corpus
    sorani_question = app.knowledge.search("پشتگیری چییە؟", 3)["passages"]
    assert (keys[sorani_question[0]["document_id"]], sorani_question[0]["page"]) == ("english", 1)
    english_question = app.knowledge.search("where should I put my stop loss", 3)["passages"]
    assert {(keys[p["document_id"]], p["page"]) for p in english_question} >= {("sorani", 3)}


async def test_scanned_page_is_found_and_cited_by_its_page(corpus: tuple[Any, dict[int, str]]) -> None:
    app, keys = corpus
    hit = app.knowledge.search("move the stop to breakeven", 1)["passages"][0]
    assert (keys[hit["document_id"]], hit["page"], hit["source"]) == ("notes", 2, "ocr")
    assert hit["citation"] == "«Trading notes week one», p. 2"


async def test_keyboard_and_ocr_spellings_match_kurdish_letters(corpus: tuple[Any, dict[int, str]]) -> None:
    app, keys = corpus
    hit = app.knowledge.search("شلهمهنی له کوی کودهبیتهوه", 1)["passages"][0]
    assert (keys[hit["document_id"]], hit["page"]) == ("sorani", 2)


async def test_unrelated_questions_score_low(corpus: tuple[Any, dict[int, str]]) -> None:
    app, _ = corpus
    passages = app.knowledge.search("banana smoothie recipe", 3)["passages"]
    assert all(p["score"] < 0.35 for p in passages)
