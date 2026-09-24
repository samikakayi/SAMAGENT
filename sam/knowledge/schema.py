"""Tables of the knowledge library (namespace ``knowledge`` in sam2.sqlite3).

- ``kb_documents``: one row per file (path, title, hash, pages, status).
  ``gen`` is the generation of chunks currently served: a re-index writes
  generation gen+1 in small batches while searches keep reading the old one,
  then flips ``gen`` in one tiny transaction (a 3000-passage book never holds
  the shared connection's lock for long; the core loop must not wait > 50 ms).
- ``kb_chunks``: passages with page_start/page_end/section for citations;
  ``text`` is readable, ``text_norm`` the folded search form (textfix).
- ``kb_chunks_fts``: FTS5 trigram index over ``text_norm`` (substring
  matching suits Sorani suffixes: «ستراتیژی» finds «ستراتیژییەکان»; the
  trading research measured trigram FTS at 69% top-1 on Sorani vs 56% for
  local embeddings, reports/trading-intelligence.json).
"""

from __future__ import annotations

KNOWLEDGE_V1 = """
CREATE TABLE IF NOT EXISTS kb_documents (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL,                   -- absolute path as added
    path_key TEXT NOT NULL UNIQUE,        -- os.path.normcase(path)
    title TEXT NOT NULL,
    kind TEXT NOT NULL,                   -- pdf|docx|txt|md
    sha256 TEXT NOT NULL DEFAULT '',
    size INTEGER NOT NULL DEFAULT 0,
    mtime REAL NOT NULL DEFAULT 0,
    pages INTEGER NOT NULL DEFAULT 0,     -- 0 = the format has no pages (txt/md, docx without page marks)
    chunks INTEGER NOT NULL DEFAULT 0,
    chars INTEGER NOT NULL DEFAULT 0,
    ocr_pages INTEGER NOT NULL DEFAULT 0,
    skipped_pages INTEGER NOT NULL DEFAULT 0,   -- scanned pages over the OCR cap / unreadable
    language TEXT NOT NULL DEFAULT '',    -- ckb|en|mixed
    status TEXT NOT NULL DEFAULT 'queued',  -- queued|indexing|ready|partial|failed|missing
    error TEXT NOT NULL DEFAULT '',
    gen INTEGER NOT NULL DEFAULT 0,       -- chunk generation being served
    added_at REAL NOT NULL,
    indexed_at REAL,
    meta TEXT                             -- JSON: warnings, reversed_pages, source
);
CREATE INDEX IF NOT EXISTS kb_documents_sha ON kb_documents(sha256);

CREATE TABLE IF NOT EXISTS kb_chunks (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES kb_documents(id) ON DELETE CASCADE,
    gen INTEGER NOT NULL,
    ord INTEGER NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    section TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'text',  -- text|ocr
    text TEXT NOT NULL,
    text_norm TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS kb_chunks_doc ON kb_chunks(document_id, gen, ord);

CREATE VIRTUAL TABLE IF NOT EXISTS kb_chunks_fts USING fts5(
    text_norm, content='kb_chunks', content_rowid='id', tokenize='trigram');
CREATE TRIGGER IF NOT EXISTS kb_chunks_ai AFTER INSERT ON kb_chunks BEGIN
    INSERT INTO kb_chunks_fts(rowid, text_norm) VALUES (new.id, new.text_norm);
END;
CREATE TRIGGER IF NOT EXISTS kb_chunks_ad AFTER DELETE ON kb_chunks BEGIN
    INSERT INTO kb_chunks_fts(kb_chunks_fts, rowid, text_norm) VALUES ('delete', old.id, old.text_norm);
END;
CREATE TRIGGER IF NOT EXISTS kb_chunks_au AFTER UPDATE OF text_norm ON kb_chunks BEGIN
    INSERT INTO kb_chunks_fts(kb_chunks_fts, rowid, text_norm) VALUES ('delete', old.id, old.text_norm);
    INSERT INTO kb_chunks_fts(rowid, text_norm) VALUES (new.id, new.text_norm);
END;
"""

MIGRATIONS = [(1, KNOWLEDGE_V1)]

__all__ = ["MIGRATIONS"]
