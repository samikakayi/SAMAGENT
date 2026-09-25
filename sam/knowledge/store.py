"""SQLite side of the knowledge library: documents, passages and search.

All methods are synchronous and short; the library calls the heavy ones
(writing a book's passages, searching) from a worker thread. Writes go in
batches of ``batch`` rows per transaction so the shared connection's lock is
never held for long (the core loop uses the same connection).

Search = candidates from the FTS5 trigram index (one query per concept, so a
very common word cannot crowd out the rare one) -> rerank in Python by
idf-weighted concept coverage (how much of the question a passage covers),
a bonus when every concept is present and when the book's title names a
concept, then at most two passages per page. Measured on the synthetic
corpus in tests/fixtures/knowledge (see tests/test_knowledge_retrieval.py).
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from .chunker import Chunk
from .query import Concept, idf, match_expression
from .textfix import search_form

VISIBLE_STATUSES = ("ready", "partial", "missing", "indexing")
_DOC_COLUMNS = ("id, path, title, kind, sha256, size, mtime, pages, chunks, chars, ocr_pages, skipped_pages, "
                "language, status, error, gen, added_at, indexed_at, meta")


def path_key(path: Path | str) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


class Store:
    def __init__(self, db: Any) -> None:
        self.db = db

    # -- documents ------------------------------------------------------------------------------------------------
    def document(self, doc_id: int) -> dict[str, Any] | None:
        return self.db.query_one(f"SELECT {_DOC_COLUMNS} FROM kb_documents WHERE id=?", (int(doc_id),))

    def by_path(self, path: Path | str) -> dict[str, Any] | None:
        return self.db.query_one(f"SELECT {_DOC_COLUMNS} FROM kb_documents WHERE path_key=?", (path_key(path),))

    def by_sha(self, sha: str, *, exclude_id: int | None = None) -> list[dict[str, Any]]:
        rows = self.db.query(f"SELECT {_DOC_COLUMNS} FROM kb_documents WHERE sha256=? AND gen > 0 "
                             "ORDER BY id", (sha,))
        return [r for r in rows if r["id"] != exclude_id]

    def documents(self) -> list[dict[str, Any]]:
        return self.db.query(f"SELECT {_DOC_COLUMNS} FROM kb_documents ORDER BY added_at DESC, id DESC")

    def create(self, path: Path, *, title: str, kind: str, size: int, mtime: float) -> int:
        return self.db.insert("kb_documents", {
            "path": str(path), "path_key": path_key(path), "title": title, "kind": kind, "size": int(size),
            "mtime": float(mtime), "status": "queued", "added_at": time.time()})

    def update(self, doc_id: int, **fields: Any) -> None:
        if not fields:
            return
        cols = []
        values = []
        for key, value in fields.items():
            if not key.replace("_", "").isalnum():
                raise ValueError("bad column")
            cols.append(f"{key}=?")
            values.append(json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value)
        self.db.execute(f"UPDATE kb_documents SET {', '.join(cols)} WHERE id=?", (*values, int(doc_id)))

    def move(self, doc_id: int, path: Path) -> None:
        self.update(doc_id, path=str(path), path_key=path_key(path))

    # -- passages ----------------------------------------------------------------------------------------------------
    def write_chunks(self, doc_id: int, gen: int, chunks: Sequence[Chunk], *, batch: int = 150,
                     cancel: threading.Event | None = None) -> int:
        written = 0
        for start in range(0, len(chunks), batch):
            if cancel is not None and cancel.is_set():
                break
            rows = [(doc_id, gen, c.ord, c.page_start, c.page_end, c.section, c.source, c.text,
                     search_form(f"{c.section}\n{c.text}" if c.section else c.text))
                    for c in chunks[start:start + batch]]
            with self.db.transaction() as conn:
                conn.executemany("INSERT INTO kb_chunks(document_id, gen, ord, page_start, page_end, section, "
                                 "source, text, text_norm) VALUES (?,?,?,?,?,?,?,?,?)", rows)
            written += len(rows)
        return written

    def activate(self, doc_id: int, gen: int, **fields: Any) -> None:
        """Serve generation ``gen`` (one small update), then drop the others."""
        self.update(doc_id, gen=gen, **fields)
        self.purge(doc_id, keep_gen=gen)

    def purge(self, doc_id: int, *, keep_gen: int | None = None, batch: int = 400) -> int:
        removed = 0
        while True:
            if keep_gen is None:
                ids = self.db.query("SELECT id FROM kb_chunks WHERE document_id=? LIMIT ?", (doc_id, batch))
            else:
                ids = self.db.query("SELECT id FROM kb_chunks WHERE document_id=? AND gen!=? LIMIT ?",
                                    (doc_id, keep_gen, batch))
            if not ids:
                return removed
            with self.db.transaction() as conn:
                conn.executemany("DELETE FROM kb_chunks WHERE id=?", [(r["id"],) for r in ids])
            removed += len(ids)

    def purge_generation(self, doc_id: int, gen: int, *, batch: int = 400) -> int:
        """Delete the rows of one generation (unserved leftovers)."""
        removed = 0
        while True:
            ids = self.db.query("SELECT id FROM kb_chunks WHERE document_id=? AND gen=? LIMIT ?", (doc_id, gen, batch))
            if not ids:
                return removed
            with self.db.transaction() as conn:
                conn.executemany("DELETE FROM kb_chunks WHERE id=?", [(r["id"],) for r in ids])
            removed += len(ids)

    def remove(self, doc_id: int) -> bool:
        if self.document(doc_id) is None:
            return False
        self.purge(doc_id)
        self.db.execute("DELETE FROM kb_documents WHERE id=?", (int(doc_id),))
        return True

    def total_chunks(self) -> int:
        return int(self.db.scalar(
            "SELECT count(*) FROM kb_chunks c JOIN kb_documents d ON d.id=c.document_id AND c.gen=d.gen "
            f"WHERE d.status IN ({','.join('?' * len(VISIBLE_STATUSES))})", VISIBLE_STATUSES) or 0)

    def chunks_of(self, doc_id: int) -> list[dict[str, Any]]:
        return self.db.query("SELECT c.id, c.ord, c.page_start, c.page_end, c.section, c.source, c.text "
                             "FROM kb_chunks c JOIN kb_documents d ON d.id=c.document_id AND c.gen=d.gen "
                             "WHERE c.document_id=? ORDER BY c.ord", (int(doc_id),))

    # -- search --------------------------------------------------------------------------------------------------------
    def search(self, concepts: Sequence[Concept], k: int = 5, *, per_concept: int = 40,
               document_ids: Iterable[int] | None = None) -> list[dict[str, Any]]:
        total = self.total_chunks()
        if not total or not concepts:
            return []
        weights: dict[int, float] = {}
        candidates: set[int] = set()
        for index, concept in enumerate(concepts):
            expr = match_expression(concept.variants)
            if expr is None:
                continue
            rows = self.db.query("SELECT rowid AS id FROM kb_chunks_fts WHERE kb_chunks_fts MATCH ? "
                                 "ORDER BY bm25(kb_chunks_fts) LIMIT ?", (expr, per_concept))
            if not rows:
                continue
            df = int(self.db.scalar("SELECT count(*) FROM (SELECT 1 FROM kb_chunks_fts WHERE kb_chunks_fts "
                                    "MATCH ? LIMIT 20000)", (expr,)) or 0)
            weights[index] = idf(total, df)
            candidates.update(int(r["id"]) for r in rows)
        if not candidates:
            return []
        rows = self._fetch(sorted(candidates), document_ids)
        weight_sum = sum(weights.values()) or 1.0
        scored = []
        for row in rows:
            norm = row["text_norm"]
            title_norm = search_form(row["title"])
            raw = 0.0
            matched: list[str] = []
            for index, weight in weights.items():
                concept = concepts[index]
                tf = sum(norm.count(v) for v in concept.variants)
                if tf:
                    raw += weight * (1.0 + 0.3 * math.log(tf))
                    matched.append(concept.label)
            if not matched:
                continue
            score = raw / weight_sum
            if len(matched) == len(weights) and len(weights) > 1:
                score += 0.15
            score += 0.05 * sum(1 for index in weights if any(v in title_norm for v in concepts[index].variants))
            scored.append((score, -len(norm), row, matched))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        picked: list[dict[str, Any]] = []
        per_page: dict[tuple[int, Any], int] = {}
        for score, _, row, matched in scored:
            key = (row["document_id"], row["page_start"])
            if per_page.get(key, 0) >= 2:
                continue
            per_page[key] = per_page.get(key, 0) + 1
            item = {k: row[k] for k in ("id", "document_id", "title", "path", "kind", "page_start", "page_end",
                                        "section", "source", "text")}
            item["score"] = round(score, 3)
            item["matched"] = matched
            picked.append(item)
            if len(picked) >= k:
                break
        return picked

    def _fetch(self, ids: list[int], document_ids: Iterable[int] | None) -> list[dict[str, Any]]:
        allowed = {int(d) for d in document_ids} if document_ids is not None else None
        out: list[dict[str, Any]] = []
        for start in range(0, len(ids), 400):
            part = ids[start:start + 400]
            out.extend(self.db.query(
                "SELECT c.id, c.document_id, c.page_start, c.page_end, c.section, c.source, c.text, c.text_norm, "
                "d.title, d.path, d.kind FROM kb_chunks c JOIN kb_documents d ON d.id=c.document_id AND c.gen=d.gen "
                f"WHERE c.id IN ({','.join('?' * len(part))}) "
                f"AND d.status IN ({','.join('?' * len(VISIBLE_STATUSES))})", (*part, *VISIBLE_STATUSES)))
        if allowed is not None:
            out = [r for r in out if r["document_id"] in allowed]
        return out


__all__ = ["Store", "VISIBLE_STATUSES", "path_key"]
