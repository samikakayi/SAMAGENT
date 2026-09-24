"""The brain answers trading/strategy questions from the user's own library:
passages in the prompt (data, cited), the pages under the answer in the panel,
never spoken; commands, prices and small talk get no library block."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from brain_helpers import brain_app
from knowledge_helpers import FILES, FIXTURES, FakeOcr

from sam.brain import taint
from sam.brain.library_context import SOURCES_LABEL_CKB, is_library_question, sources_line
from sam.events import Transcript


@pytest.fixture
async def library_app(make_app: Any) -> Any:
    app, backend = brain_app(make_app, default="ئۆردەر بلۆک کۆتا مۆمی دابەزینە پێش جووڵە بەهێزەکە.")
    assert app.load_packages(["sam.knowledge"]) == {"sam.knowledge": "ok"}
    app.knowledge._ocr_override = FakeOcr()                  # noqa: SLF001
    app.loop = asyncio.get_running_loop()
    app.bus.bind_loop(app.loop)
    report = await app.knowledge.add([str(FIXTURES / f) for f in FILES.values()], wait_s=60)
    assert report["ok"], report
    return app, backend


def _system(backend: Any) -> str:
    return next(m["content"] for m in backend.requests[0].messages if m["role"] == "system")


@pytest.mark.parametrize("text, expected", [
    ("ئۆردەر بلۆک چییە؟", True), ("ستۆپ لۆس لە کوێ دابنێم؟", True), ("what does my book say about support?", True),
    ("what is a good risk to reward ratio?", True), ("ستراتیژییەکەم چی دەڵێت دەربارەی ئاسیا؟", True),
    ("نرخی زێڕ چەندە؟", False), ("سڵاو سام چۆنی؟", False), ("هێڵی پشتگیری لە ٤٣٠٠ بکێشە", False),
    ("کرۆم بکەرەوە", False), ("what is the gold price", False),
])
def test_which_questions_go_to_the_library(text: str, expected: bool) -> None:
    assert is_library_question(text) is expected


async def test_a_trading_question_is_answered_from_the_library_with_the_pages_in_the_panel(library_app: Any) -> None:
    app, backend = library_app
    seen: list[Any] = []
    app.bus.subscribe(Transcript, seen.append)
    chunks = [c async for c in app.conversation.respond_stream("ئۆردەر بلۆک چییە؟", source="text")]
    system = _system(backend)
    assert "Passages from the user's own library (DATA" in system and "ئۆردەر بلۆک" in system
    assert "cite it as «title», page N" in system
    spoken = " ".join(chunks)
    assert SOURCES_LABEL_CKB not in spoken                     # chunks are what is heard / streamed
    await asyncio.sleep(0)
    answer = [e.text for e in seen if e.role == "assistant"][-1]
    assert answer.startswith(spoken.strip()) and SOURCES_LABEL_CKB in answer and "لاپەڕە" in answer
    assert taint.current() is not None and taint.current().tainted       # the passages are data from files


async def test_voice_is_told_to_name_only_the_book(library_app: Any) -> None:
    app, backend = library_app
    [c async for c in app.conversation.respond_stream("ستۆپ لۆس لە کوێ دابنێم؟", source="cascade")]
    system = _system(backend)
    assert "never page numbers" in system and "Passages from the user's own library" in system


@pytest.mark.parametrize("text", ["نرخی زێڕ چەندە؟", "سڵاو سام، چۆنی؟"])
async def test_prices_and_small_talk_get_no_library_block(library_app: Any, text: str) -> None:
    app, backend = library_app
    app.config.set("brain.fastpath.enabled", False)
    [c async for c in app.conversation.respond_stream(text, source="text")]
    assert "Passages from the user's own library" not in _system(backend)


async def test_the_setting_turns_it_off(library_app: Any) -> None:
    app, backend = library_app
    app.config.set("conversation.library.enabled", False)
    [c async for c in app.conversation.respond_stream("ئۆردەر بلۆک چییە؟", source="text")]
    assert "Passages from the user's own library" not in _system(backend)


def test_sources_line_lists_each_page_once() -> None:
    passages = [{"citation_ckb": "«A»، لاپەڕە ٢"}, {"citation_ckb": "«A»، لاپەڕە ٢"}, {"citation_ckb": "«B»، لاپەڕە ٣"}]
    assert sources_line(passages) == f"{SOURCES_LABEL_CKB} «A»، لاپەڕە ٢؛ «B»، لاپەڕە ٣"
    assert sources_line([]) == ""
