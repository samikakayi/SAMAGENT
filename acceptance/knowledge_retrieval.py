"""Knowledge library: retrieval quality and OCR on the synthetic corpus.

Run by hand on this PC (local only: no network, no model, nothing leaves the
machine; a throw-away SAM home under work/ is used and deleted):

    .venv\\Scripts\\python.exe acceptance\\knowledge_retrieval.py            # real Windows OCR
    .venv\\Scripts\\python.exe acceptance\\knowledge_retrieval.py --fake-ocr

Reports top-1 / top-3 hit rates on tests/knowledge_helpers.QUESTIONS (28
questions: Sorani, English, cross-language, scanned pages, Arabic-keyboard spellings)
for the full search and three ablations (no glossary, no stems, no OCR
fold), the OCR text and time per scanned page, and indexing times.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from knowledge_helpers import FILES, FIXTURES, FakeOcr, hit_rate  # noqa: E402

WORK = ROOT / "work" / "knowledge-acceptance"


async def build(home: Path, *, fake_ocr: bool) -> tuple[Any, dict[int, str], float]:
    from sam.app import App

    shutil.rmtree(home, ignore_errors=True)
    (home / "data").mkdir(parents=True)
    app = App(home, environ={}, llm_backends={})
    app.load_packages(["sam.knowledge"])
    app.loop = asyncio.get_running_loop()
    app.bus.bind_loop(app.loop)
    library = app.knowledge
    if fake_ocr:
        library._ocr_override = FakeOcr()
    started = time.perf_counter()
    report = await library.add([str(FIXTURES / name) for name in FILES.values()], wait_s=600)
    seconds = time.perf_counter() - started
    keys = {}
    for result in report["results"]:
        for key, name in FILES.items():
            if Path(result["path"]).name == name:
                keys[int(result["document_id"])] = key
    return app, keys, seconds


async def measure(label: str, *, fake_ocr: bool, patch: Any = None) -> dict[str, Any]:
    home = WORK / label.replace(" ", "_")
    undo = patch() if patch else None
    try:
        app, keys, seconds = await build(home, fake_ocr=fake_ocr)
        result = hit_rate(app.knowledge, keys, k=3)
        result["index_s"] = seconds
        result["app"] = app
        return result
    finally:
        if undo:
            undo()


def no_glossary() -> Any:
    from sam.knowledge import query

    original = query._glossary_index
    query._glossary_index = lambda: []                       # type: ignore[assignment]
    query.group_variants.cache_clear()
    return lambda: setattr(query, "_glossary_index", original)


def no_stems() -> Any:
    from sam.knowledge import query

    original = query.stem
    query.stem = lambda word: word                           # type: ignore[assignment]
    return lambda: setattr(query, "stem", original)


def no_fold() -> Any:
    from sam.knowledge import query, textfix

    original = textfix._FOLD
    textfix._FOLD = {}                                       # type: ignore[assignment]
    query._glossary_index.cache_clear()                      # cached spellings were folded
    query.group_variants.cache_clear()

    def undo() -> None:
        textfix._FOLD = original
        query._glossary_index.cache_clear()
        query.group_variants.cache_clear()
    return undo


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake-ocr", action="store_true")
    args = parser.parse_args()
    fake = args.fake_ocr
    full = await measure("full", fake_ocr=fake)
    app = full.pop("app")
    print(f"indexing 3 PDFs (10 pages, 2 scanned) took {full['index_s']:.2f} s "
          f"({'fake' if fake else 'Windows'} OCR)")
    for doc in app.knowledge.documents():
        print(f"  #{doc['id']} {doc['title']!r}: {doc['pages']} pages, {doc['chunks']} passages, "
              f"{doc['ocr_pages']} OCR, status {doc['status']}")
        for chunk in app.knowledge.store.chunks_of(doc["id"]):
            if chunk["source"] == "ocr":
                print(f"    OCR p.{chunk['page_start']}: {chunk['text'][:300]!r}")
    print(f"\nfull search: top-1 {full['top1']:.0%}, top-3 {full['topk']:.0%} (n={full['n']})")
    for row in full["rows"]:
        if not row["topk"]:
            print(f"  MISS {row['question']!r}: {row['found']}")
    app.close()
    for label, patch in (("no glossary", no_glossary), ("no stems", no_stems), ("no fold", no_fold)):
        result = await measure(label, fake_ocr=fake, patch=patch)
        result.pop("app").close()
        print(f"{label:12s}: top-1 {result['top1']:.0%}, top-3 {result['topk']:.0%}")
    shutil.rmtree(WORK, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
