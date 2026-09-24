"""The knowledge-library tools: knowledge_add, knowledge_search,
knowledge_list, knowledge_remove (registered by ``sam.knowledge.register``).

Passages come back under ``untrusted`` (text read from files is data, never
instructions -- CONTRACTS 0) with their citations beside them, so the model
can quote «title», page N honestly.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..brain.tools import ToolContext, fail, ok, tool
from .library import ckb_number


def _library(ctx: ToolContext) -> Any:
    library = getattr(ctx.app, "knowledge", None)
    if library is None:
        raise RuntimeError("the knowledge library is not loaded")
    return library


def _from(report: dict[str, Any]) -> dict[str, Any]:
    body = dict(report)
    good = bool(body.pop("ok", False))
    summary = str(body.pop("summary", "") or ("Done." if good else "Failed."))
    return ok(summary, **body) if good else fail(summary, **body)


@tool("knowledge_add",
      description="Add a PDF, Word (.docx), text or Markdown file -- or every such file in a folder -- to the "
                  "user's local knowledge library (their trading books and notes), so knowledge_search can answer "
                  "from them with page citations. Scanned pages are read with Windows OCR. Re-adding a changed "
                  "file re-indexes it; identical copies are skipped. Big books continue in the background.",
      description_ckb="زیادکردنی کتێب بۆ کتێبخانە",
      params={"type": "object", "properties": {
          "path": {"type": "string", "description": "file or folder as the user said it, e.g. "
                                                    "'Desktop\\\\books\\\\ICT.pdf' or 'Documents\\\\Trading'"},
          "recursive": {"type": "boolean", "description": "include sub-folders (default true)"}},
          "required": ["path"]},
      risk="safe", blocking=False, timeout_s=90,
      examples_ckb=("ئەم کتێبە بخە ناو کتێبخانەکەت", "هەموو PDFەکانی بوخچەی Books بخوێنەوە"))
async def knowledge_add(ctx: ToolContext, path: str, recursive: bool = True, **_ignored: Any) -> dict[str, Any]:
    confirm = None if ctx.source == "ui" else ctx.confirm
    report = await _library(ctx).add([path], source=ctx.source, recursive=recursive, confirm=confirm)
    return _from(report)


@tool("knowledge_search",
      description="Search the user's own books and documents (the local knowledge library) and return the most "
                  "relevant passages with citations (title + page). Use it when the user asks what their books or "
                  "documents say, and for trading-theory questions when the library has books. Answer from the "
                  "passages and cite them as «title», page N; say so when nothing relevant was found.",
      description_ckb="گەڕان لە کتێبەکانی تۆ",
      params={"type": "object", "properties": {
          "query": {"type": "string", "description": "the question or key words, Sorani or English"},
          "k": {"type": "integer", "description": "how many passages (1-10, default 5)"}},
          "required": ["query"]},
      risk="safe", blocking=True, timeout_s=20,
      examples_ckb=("کتێبەکانم چی دەڵێن دەربارەی ئۆردەر بلۆک؟", "لە کتێبەکەمدا ستۆپ لۆس لە کوێ دادەنرێت؟"))
async def knowledge_search(ctx: ToolContext, query: str, k: int = 5, **_ignored: Any) -> dict[str, Any]:
    library = _library(ctx)
    if not library.has_documents():
        return fail("The knowledge library is empty: the user has not added any books yet (knowledge_add, or the "
                    "panel's «کتێبخانە» page).", empty=True, say_ckb="هێشتا هیچ کتێبێک لە کتێبخانەکەدا نییە.")
    k = max(1, min(int(k or 5), 10))
    # Keep the whole result under the registry's 6000-character data cap.
    per = max(300, min(900, 3600 // k))
    result = await library.asearch(query, k, max_chars=per)
    passages = result["passages"]
    if not passages:
        return ok("No passage in the library matches this question.", found=0, concepts=result["concepts"],
                  say_ckb="لە کتێبەکانتدا شتێکم لەسەر ئەمە نەدۆزییەوە.")
    cites = [p["citation"] for p in passages]
    weak = passages[0]["score"] < 0.35
    summary = (f"{len(passages)} passage(s) found{' (weak matches)' if weak else ''}: " + "; ".join(cites[:5]))
    return ok(summary, found=len(passages), weak=weak, concepts=result["concepts"],
              citations=[{"ref": n, "citation": p["citation"], "citation_ckb": p["citation_ckb"],
                          "document_id": p["document_id"], "page": p["page"], "score": p["score"]}
                         for n, p in enumerate(passages, start=1)],
              untrusted=[{"ref": n, "text": p["text"]} for n, p in enumerate(passages, start=1)])


@tool("knowledge_list",
      description="List the books and documents in the user's knowledge library with page counts and status.",
      description_ckb="لیستی کتێبەکانی کتێبخانە",
      params={"type": "object", "properties": {}},
      risk="safe", blocking=True, timeout_s=10,
      examples_ckb=("چ کتێبێکت لە کتێبخانەدا هەیە؟",))
async def knowledge_list(ctx: ToolContext, **_ignored: Any) -> dict[str, Any]:
    library = _library(ctx)
    docs = library.documents()
    if not docs:
        return ok("The knowledge library is empty.", documents=[], say_ckb="کتێبخانەکە بەتاڵە.")
    rows = [{"id": d["id"], "title": d["title"], "kind": d["kind"], "pages": d["pages"], "passages": d["chunks"],
             "status": d["status"], "ocr_pages": d["ocr_pages"]} for d in docs[:60]]
    pages = sum(int(d["pages"] or 0) for d in docs)
    summary = f"{len(docs)} document(s), {pages} pages: " + "; ".join(
        f"{d['title']} ({d['pages']} p., {d['status']})" for d in docs[:8])
    return ok(summary, documents=rows, jobs=library.running(),
              say_ckb=f"{ckb_number(len(docs))} کتێب و بەڵگەنامە لە کتێبخانەکەدان.")


def _remove_risk(args: dict[str, Any]) -> tuple[str, str | None]:
    ref = str(args.get("document") or "").strip().lower()
    if ref in ("all", "هەموو", "هەمووی", "*"):
        return "confirm", "هەموو کتێبەکان لە کتێبخانەکە لابەرم؟ (خودی فایلەکان ناسڕدرێنەوە)"
    return "safe", None


@tool("knowledge_remove",
      description="Take a document out of the knowledge library (by id, title or path; 'all' for everything). "
                  "The file itself is not deleted.",
      description_ckb="لابردنی کتێب لە کتێبخانە",
      params={"type": "object", "properties": {
          "document": {"type": "string", "description": "document id, title or path, or 'all'"}},
          "required": ["document"]},
      classify=_remove_risk, blocking=True, timeout_s=30,
      examples_ckb=("ئەو کتێبە لە کتێبخانەکە لابە",))
async def knowledge_remove(ctx: ToolContext, document: str, **_ignored: Any) -> dict[str, Any]:
    return _from(await asyncio.to_thread(_library(ctx).remove, document))


TOOLS = (knowledge_add, knowledge_search, knowledge_list, knowledge_remove)

__all__ = ["TOOLS", "knowledge_add", "knowledge_list", "knowledge_remove", "knowledge_search"]
