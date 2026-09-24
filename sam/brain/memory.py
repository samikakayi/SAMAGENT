"""Long-term memory: user facts, notes and the conversation log.

Retrieval is SQLite FTS5 with the trigram tokenizer (core schema). The trading
research measured FTS5 trigram at 69% top-1 on Sorani queries versus 56% for
local embeddings (reports/trading-intelligence.json), and it needs no model,
so it is the only index. Three layers make Sorani search robust:

1. ``search_key`` -- a loose spelling key. On top of ``normalize_ckb``
   (ي/ك -> ی/ک, ZWNJ, digits, diacritics) it folds the variants people type
   on Arabic keyboards and that STT emits: ه‌ / ه / ھ / ە -> ە, ڕ -> ر,
   ڵ -> ل, ێ -> ی, ۆ -> و, أ/إ/آ -> ا. Facts store this key in ``text_norm``
   (dedup + FTS), so "به‌ڵێ", "بەڵێ" and "بهلی" meet.
2. Query terms of >= 3 characters (a trigram needs 3) plus light Sorani
   suffix stripping (ستراتیژییەکەم -> ستراتیژی), OR-ed, ranked by bm25.
3. LIKE for short words, then a rapidfuzz fallback for typos/STT slips.

Facts are the user's own statements (typed, spoken, or extracted from their
turns by one cheap LLM call after a conversation ends). Key-shaped text is
redacted before anything is stored.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Iterable

from .tools import ToolContext, fail, ok, tool
from ..textnorm import normalize_ckb

log = logging.getLogger("sam.memory")

FACT_KINDS = ("fact", "preference", "person", "project", "trading")
NOTE_KINDS = ("note", "theory", "journal", "doc")

# --- Sorani search normalisation ---------------------------------------------------

_PRE = [("ه‌", "ە"), ("ھ‌", "ە"), ("ۀ", "ە")]
_FOLD = str.maketrans({"ه": "ە", "ھ": "ە", "ڕ": "ر", "ڵ": "ل", "ێ": "ی", "ۆ": "و",
                       "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ؤ": "و"})
# Longest first. Applied to query terms only (stored text keeps full words;
# trigram matching finds the stem inside them).
_SUFFIXES = tuple(sorted({
    "ەکانمان", "ەکانتان", "ەکانیان", "ەکانم", "ەکانت", "ەکانی", "ەکان",
    "ەکەمان", "ەکەتان", "ەکەیان", "ەکەم", "ەکەت", "ەکەی", "ەکە",
    "یەکان", "یەکە", "یەک", "ەیەک", "ێک", "یەکی",
    "مان", "تان", "یان", "ان", "ەی", "ەم", "ەت", "م", "ت", "ی", "ە",
}, key=len, reverse=True))


_FOLDED_SUFFIXES = tuple(sorted({s.translate(_FOLD) for s in _SUFFIXES}, key=len, reverse=True))


def search_key(text: str, *, strip_punct: bool = False) -> str:
    """Loose comparison key for Sorani/English text (see module docstring).
    ZWNJ-marked heh is resolved before ``normalize_ckb`` drops the ZWNJ."""
    if not text:
        return ""
    value = text
    for old, new in _PRE:
        value = value.replace(old, new)
    return normalize_ckb(value, strip_punct=strip_punct).translate(_FOLD)


def _stems(word: str) -> list[str]:
    """The word plus up to two suffix-stripped stems (ستراتیژییەکەم ->
    ستراتیژیی -> ستراتیژی), each >= 3 characters for the trigram index."""
    out = [word]
    current = word
    for _ in range(2):
        if len(current) < 5:
            break
        for suffix in _FOLDED_SUFFIXES:
            if current.endswith(suffix) and len(current) - len(suffix) >= 3:
                current = current[: -len(suffix)]
                out.append(current)
                break
        else:
            break
    return out


# Question words and fillers that would match almost every fact.
_STOP_WORDS = frozenset(search_key(w) for w in (
    "چی", "چییە", "چیە", "چۆن", "کێ", "کەی", "کوێ", "بۆ", "بۆچی", "ئەم", "ئەو", "ئەوە", "ئەمە", "ئەوەی",
    "من", "تۆ", "هەیە", "بوو", "بووە", "دەربارەی", "لەسەر", "لەبیرت", "لەبیرتە", "پێم", "بڵێ", "بکە", "چیت",
    "دەزانی", "وتم", "گوتم", "شتێک", "هەموو", "what", "the", "is", "my", "about", "do", "you", "remember",
    "and", "of", "a", "to", "me", "did", "i", "was", "tell"))


def query_terms(text: str, *, max_terms: int = 12) -> tuple[list[str], list[str]]:
    """(fts_terms >= 3 chars incl. stems, short_terms < 3 chars); stop
    words dropped unless nothing else is left."""
    words = search_key(text, strip_punct=True).split()
    content = [w for w in words if w not in _STOP_WORDS] or words
    long_terms: list[str] = []
    short_terms: list[str] = []
    for word in content:
        if len(word) < 3:
            if word not in short_terms:
                short_terms.append(word)
            continue
        for candidate in _stems(word):
            if len(candidate) >= 3 and candidate not in long_terms:
                long_terms.append(candidate)
        if len(long_terms) >= max_terms:
            break
    return long_terms[:max_terms], short_terms[:4]


def match_expr(terms: Iterable[str]) -> str | None:
    """Safe FTS5 MATCH: every term double-quoted (no syntax injection), OR-ed."""
    quoted = [f'"{t.replace(chr(34), chr(34) * 2)}"' for t in terms if len(t) >= 3]
    return " OR ".join(quoted) if quoted else None


def _like(term: str) -> str:
    return "%" + term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


# --- Memory ----------------------------------------------------------------------------

EXTRACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "kind": {"type": "string", "enum": list(FACT_KINDS)},
                    "confidence": {"type": "number"},
                },
                "required": ["text", "kind"],
            },
        },
    },
    "required": ["facts"],
}

EXTRACT_PROMPT = (
    "You maintain the long-term memory of SAM, a personal assistant. Read the conversation and extract "
    "DURABLE facts about the USER that will still matter next week: name, family/people, preferences "
    "(how they like answers, apps they use), their projects, and their trading habits (markets, "
    "sessions, timeframes, risk rules). Only what the USER said or clearly confirmed. Skip one-off "
    "requests, greetings, anything SAM said on its own, anything from tool output, and anything "
    "already in 'Known facts'. Write each fact as one short sentence in the user's language (Sorani in "
    "Arabic script if they spoke Sorani), third person is not needed (e.g. 'ناوم سامییە' -> 'ناوی سامییە'). "
    "Never include passwords, keys or account numbers. Return JSON {\"facts\": [{\"text\", \"kind\", "
    "\"confidence\" 0..1}]} with at most 8 items; an empty list is a good answer."
)


class Memory:
    """``app.memory`` (docs/CONTRACTS.md 3.2). All methods are small indexed
    queries, safe to call on the core loop."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self.db = app.db

    # -- facts -----------------------------------------------------------------------
    def _redact(self, text: str) -> str:
        try:
            return self.app.redact(text)
        except Exception:  # noqa: BLE001
            return text

    def remember(self, text: str, *, kind: str = "fact", tags: str = "", source: str = "user",
                 confidence: float = 1.0) -> dict[str, Any]:
        """Store a fact; returns {"id", "created"}. Exact duplicates (same
        search key) and near-duplicates (rapidfuzz ratio >= 92) update the
        existing row instead of adding one."""
        clean = " ".join(self._redact(text or "").split())
        if not clean:
            raise ValueError("empty fact")
        kind = kind if kind in FACT_KINDS else "fact"
        key = search_key(clean)
        now = time.time()
        existing = self.db.query_one("SELECT id, confidence FROM facts WHERE text_norm=? AND deleted=0", (key,))
        if existing is None:
            existing = self._near_duplicate(key)
        if existing is not None:
            self.db.execute(
                "UPDATE facts SET updated_at=?, confidence=MAX(confidence, ?), "
                "tags=CASE WHEN ?='' THEN tags ELSE ? END WHERE id=?",
                (now, float(confidence), tags, tags, existing["id"]))
            return {"id": int(existing["id"]), "created": False}
        fact_id = self.db.insert("facts", {
            "text": clean, "text_norm": key, "kind": kind, "tags": tags, "source": source,
            "confidence": float(confidence), "created_at": now, "updated_at": now, "use_count": 0, "deleted": 0})
        return {"id": fact_id, "created": True}

    def _near_duplicate(self, key: str) -> dict[str, Any] | None:
        from rapidfuzz import fuzz  # ~20 ms import, only when a new fact arrives

        best: tuple[float, dict[str, Any] | None] = (0.0, None)
        for row in self.db.query("SELECT id, text_norm, confidence FROM facts WHERE deleted=0"):
            score = fuzz.ratio(key, row["text_norm"])
            if score > best[0]:
                best = (score, row)
        return best[1] if best[0] >= 92 else None

    def forget(self, fact_id: int) -> bool:
        """Soft delete (the row stays for audit; it no longer matches)."""
        cursor = self.db.execute("UPDATE facts SET deleted=1, updated_at=? WHERE id=? AND deleted=0",
                                 (time.time(), int(fact_id)))
        return bool(cursor.rowcount)

    def get_fact(self, fact_id: int) -> dict[str, Any] | None:
        return self.db.query_one("SELECT * FROM facts WHERE id=?", (int(fact_id),))

    def recall(self, query: str, *, limit: int = 5, kinds: Iterable[str] | None = None,
               touch: bool = True) -> list[dict[str, Any]]:
        """Best-matching facts (FTS5 trigram + LIKE + fuzzy fallback)."""
        limit = max(1, min(int(limit or 5), 50))
        kinds_list = [k for k in (kinds or []) if k in FACT_KINDS]
        kind_sql = f" AND f.kind IN ({','.join('?' for _ in kinds_list)})" if kinds_list else ""
        long_terms, short_terms = query_terms(query)
        found: dict[int, dict[str, Any]] = {}
        expr = match_expr(long_terms)
        if expr:
            rows = self.db.query(
                "SELECT f.id, f.text, f.kind, f.tags, f.source, f.confidence, f.updated_at, "
                "bm25(facts_fts) AS rank FROM facts_fts JOIN facts f ON f.id = facts_fts.rowid "
                f"WHERE facts_fts MATCH ? AND f.deleted=0{kind_sql} ORDER BY rank LIMIT ?",
                (expr, *kinds_list, limit))
            for row in rows:
                found[row["id"]] = row
        if len(found) < limit and short_terms:
            for term in short_terms:
                for row in self.db.query(
                        "SELECT f.id, f.text, f.kind, f.tags, f.source, f.confidence, f.updated_at, 0.0 AS rank "
                        f"FROM facts f WHERE f.deleted=0 AND f.text_norm LIKE ? ESCAPE '\\'{kind_sql} "
                        "ORDER BY f.updated_at DESC LIMIT ?", (_like(term), *kinds_list, limit)):
                    found.setdefault(row["id"], row)
        if not found:
            found = {r["id"]: r for r in self._fuzzy(query, limit, kinds_list)}
        results = list(found.values())[:limit]
        if touch and results:
            now = time.time()
            self.db.executemany("UPDATE facts SET last_used_at=?, use_count=use_count+1 WHERE id=?",
                                [(now, r["id"]) for r in results])
        return results

    def _fuzzy(self, query: str, limit: int, kinds: list[str]) -> list[dict[str, Any]]:
        from rapidfuzz import fuzz

        key = search_key(query)
        if len(key) < 3:
            return []
        rows = self.db.query("SELECT id, text, text_norm, kind, tags, source, confidence, updated_at FROM facts "
                             "WHERE deleted=0")
        scored = []
        for row in rows:
            if kinds and row["kind"] not in kinds:
                continue
            score = fuzz.partial_ratio(key, row["text_norm"])
            if score >= 75:
                row = {k: v for k, v in row.items() if k != "text_norm"}
                row["rank"] = -score / 100.0
                scored.append(row)
        scored.sort(key=lambda r: r["rank"])
        return scored[:limit]

    def list_facts(self, *, limit: int = 100, include_deleted: bool = False) -> list[dict[str, Any]]:
        where = "" if include_deleted else "WHERE deleted=0"
        return self.db.query(f"SELECT * FROM facts {where} ORDER BY updated_at DESC LIMIT ?", (int(limit),))

    def facts_for_prompt(self, limit: int = 12, max_chars: int = 1200) -> str:
        """Short bullet list for the system prompt: user-stated facts first,
        then the most used and most recent."""
        rows = self.db.query(
            "SELECT text FROM facts WHERE deleted=0 ORDER BY (source='user') DESC, use_count DESC, "
            "updated_at DESC LIMIT ?", (int(limit),))
        lines: list[str] = []
        used = 0
        for row in rows:
            line = "- " + " ".join(str(row["text"]).split())[:200]
            if used + len(line) + 1 > max_chars:
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)

    # -- notes ------------------------------------------------------------------------
    def add_note(self, body: str, *, title: str = "", kind: str = "note", tags: str = "",
                 source: str = "user") -> int:
        body = self._redact(body or "").strip()
        if not body:
            raise ValueError("empty note")
        now = time.time()
        return self.db.insert("notes", {
            "title": self._redact(title or "")[:200], "body": body, "body_norm": search_key(body),
            "kind": kind if kind in NOTE_KINDS else "note", "tags": tags, "source": source,
            "created_at": now, "updated_at": now})

    def search_notes(self, query: str, *, limit: int = 5, kinds: Iterable[str] | None = None) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 5), 50))
        kinds_list = [k for k in (kinds or []) if k in NOTE_KINDS]
        kind_sql = f" AND n.kind IN ({','.join('?' for _ in kinds_list)})" if kinds_list else ""
        long_terms, short_terms = query_terms(query)
        expr = match_expr(long_terms)
        rows: list[dict[str, Any]] = []
        if expr:
            rows = self.db.query(
                "SELECT n.id, n.title, n.body, n.kind, n.tags, n.source, n.updated_at, bm25(notes_fts) AS rank "
                f"FROM notes_fts JOIN notes n ON n.id = notes_fts.rowid WHERE notes_fts MATCH ?{kind_sql} "
                "ORDER BY rank LIMIT ?", (expr, *kinds_list, limit))
        if not rows and short_terms:
            rows = self.db.query(
                "SELECT n.id, n.title, n.body, n.kind, n.tags, n.source, n.updated_at, 0.0 AS rank FROM notes n "
                f"WHERE n.body_norm LIKE ? ESCAPE '\\'{kind_sql} ORDER BY n.updated_at DESC LIMIT ?",
                (_like(short_terms[0]), *kinds_list, limit))
        return rows

    # -- conversation log ----------------------------------------------------------------
    def start_conversation(self, source: str = "voice") -> int:
        return self.db.insert("conversations", {"started_at": time.time(), "source": source})

    def end_conversation(self, conversation_id: int) -> None:
        self.db.execute("UPDATE conversations SET ended_at=? WHERE id=? AND ended_at IS NULL",
                        (time.time(), int(conversation_id)))

    def get_conversation(self, conversation_id: int) -> dict[str, Any] | None:
        return self.db.query_one("SELECT * FROM conversations WHERE id=?", (int(conversation_id),))

    def latest_conversation(self) -> dict[str, Any] | None:
        return self.db.query_one("SELECT * FROM conversations ORDER BY id DESC LIMIT 1")

    def add_turn(self, conversation_id: int, role: str, text: str, *, source: str,
                 meta: dict[str, Any] | None = None) -> int:
        """Append a turn. Only ``Conversation`` calls this (single writer)."""
        return self.db.insert("turns", {
            "conversation_id": int(conversation_id), "at": time.time(), "role": role,
            "text": self._redact(text or ""), "source": source,
            "meta": None if not meta else json.dumps(self.app.redact_obj(meta), ensure_ascii=False, default=str)})

    def recent_turns(self, conversation_id: int | None = None, limit: int = 12, *,
                     roles: Iterable[str] | None = None, after_id: int | None = None) -> list[dict[str, Any]]:
        """Newest ``limit`` turns, returned oldest -> newest."""
        clauses: list[str] = []
        params: list[Any] = []
        if conversation_id is not None:
            clauses.append("conversation_id=?")
            params.append(int(conversation_id))
        role_list = list(roles or [])
        if role_list:
            clauses.append(f"role IN ({','.join('?' for _ in role_list)})")
            params.extend(role_list)
        if after_id is not None:
            clauses.append("id>?")
            params.append(int(after_id))
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self.db.query(f"SELECT * FROM turns {where} ORDER BY id DESC LIMIT ?", (*params, int(limit)))
        for row in rows:
            if row.get("meta"):
                try:
                    row["meta"] = json.loads(row["meta"])
                except (TypeError, json.JSONDecodeError):
                    pass
        return list(reversed(rows))

    def count_turns(self, conversation_id: int, *, role: str | None = None) -> int:
        if role:
            return int(self.db.scalar("SELECT COUNT(*) FROM turns WHERE conversation_id=? AND role=?",
                                      (int(conversation_id), role)) or 0)
        return int(self.db.scalar("SELECT COUNT(*) FROM turns WHERE conversation_id=?", (int(conversation_id),)) or 0)

    # -- extraction -------------------------------------------------------------------------
    def _ladder_available(self, ladder: str) -> bool:
        llm = getattr(self.app, "llm", None)
        if llm is None:
            return False
        try:
            refs = llm.ladder(ladder)
            backends = llm.backends
        except Exception:  # noqa: BLE001
            return False
        for ref in refs:
            provider = ref.split(":", 1)[0]
            backend = backends.get(provider)
            if backend is not None and backend.configured():
                return True
        return False

    async def extract_facts(self, conversation_id: int) -> list[dict[str, Any]]:
        """ONE cheap LLM call (ladder ``memory.extract_ladder``, JSON schema)
        that turns a finished conversation into durable user facts. Skipped
        (returns []) with fewer than 2 user turns, when already done, or when
        no model is configured."""
        conv = self.get_conversation(conversation_id)
        if conv is None or conv.get("facts_extracted"):
            return []
        turns = self.recent_turns(conversation_id, limit=40, roles=("user", "assistant"))
        if sum(1 for t in turns if t["role"] == "user") < 2:
            return []
        ladder = str(self.app.config.get("memory.extract_ladder", "extract") or "extract")
        if not self._ladder_available(ladder):
            log.info("fact extraction skipped: no model configured for ladder %s", ladder)
            return []
        lines: list[str] = []
        budget = 6000
        for turn in reversed(turns):  # newest first until the budget is used
            limit = 400 if turn["role"] == "user" else 160
            line = f"{'USER' if turn['role'] == 'user' else 'SAM'}: {' '.join(str(turn['text']).split())[:limit]}"
            if budget - len(line) < 0:
                break
            budget -= len(line)
            lines.append(line)
        known = self.facts_for_prompt(limit=30, max_chars=1500) or "(none)"
        messages = [
            {"role": "system", "content": EXTRACT_PROMPT},
            {"role": "user", "content": f"Known facts:\n{known}\n\nConversation:\n" + "\n".join(reversed(lines))},
        ]
        from .llm import LLMError  # local: keeps import order simple

        try:
            response = await self.app.llm.chat(messages, ladder=ladder, json_schema=EXTRACT_SCHEMA,
                                               reasoning="low", timeout_s=45)
            payload = response.json()
        except LLMError as err:
            log.info("fact extraction failed: %s", err.kind)
            return []
        except (ValueError, TypeError):
            log.info("fact extraction returned no JSON")
            payload = {"facts": []}
        items = payload.get("facts") if isinstance(payload, dict) else payload if isinstance(payload, list) else []
        saved: list[dict[str, Any]] = []
        for item in (items or [])[:8]:
            if not isinstance(item, dict) or not str(item.get("text", "")).strip():
                continue
            text = str(item["text"]).strip()[:300]
            try:
                confidence = max(0.0, min(1.0, float(item.get("confidence", 0.7))))
            except (TypeError, ValueError):
                confidence = 0.7
            if confidence < 0.5:
                continue
            result = self.remember(text, kind=str(item.get("kind") or "fact"), source="extracted",
                                   confidence=confidence)
            if result["created"]:
                saved.append({"id": result["id"], "text": text, "kind": item.get("kind")})
        self.db.execute("UPDATE conversations SET facts_extracted=1 WHERE id=?", (int(conversation_id),))
        if saved:
            self.db.log_activity("system", "extract_facts", ok=True, summary=f"{len(saved)} new facts",
                                 source="memory")
        return saved


# --- tools --------------------------------------------------------------------------------

def _memory(ctx: ToolContext) -> Memory | None:
    return getattr(ctx.app, "memory", None)


@tool("remember",
      description="Save a lasting fact or preference about the user (name, people, projects, how they like "
                  "things, trading habits). Use when the user says 'remember...' or states something that "
                  "will matter later. Pass the fact in the user's own words.",
      description_ckb="لەبیرکردنی زانیارییەک دەربارەی بەکارهێنەر",
      params={"type": "object", "properties": {
          "text": {"type": "string", "description": "The fact, in the user's words (Sorani or English)."},
          "kind": {"type": "string", "enum": list(FACT_KINDS)}},
          "required": ["text"]},
      risk="safe", blocking=True, timeout_s=10,
      examples_ckb=("لەبیرت بێت من تەنها لە کاتی لەندەن ترەید دەکەم", "ناوی کوڕەکەم ئارانە"))
async def remember_tool(ctx: ToolContext, text: str, kind: str = "fact") -> dict[str, Any]:
    memory = _memory(ctx)
    if memory is None:
        return fail("Memory is not available.")
    result = memory.remember(text, kind=kind, source="user")
    return ok("Saved to memory." if result["created"] else "Already in memory (updated).", **result)


@tool("recall",
      description="Search SAM's memory (user facts and saved notes) for something the user told SAM before.",
      description_ckb="گەڕان لە بیرەوەری",
      params={"type": "object", "properties": {
          "query": {"type": "string", "description": "What to look for (Sorani or English keywords)."},
          "limit": {"type": "integer", "minimum": 1, "maximum": 10}},
          "required": ["query"]},
      risk="safe", blocking=True, timeout_s=10,
      examples_ckb=("چیت لەبیرە دەربارەی ستراتیژییەکەم؟", "ناوی کوڕەکەم چی بوو؟"))
async def recall_tool(ctx: ToolContext, query: str, limit: int = 5) -> dict[str, Any]:
    memory = _memory(ctx)
    if memory is None:
        return fail("Memory is not available.")
    limit = max(1, min(int(limit or 5), 10))
    facts = [{"id": f["id"], "text": f["text"], "kind": f["kind"]} for f in memory.recall(query, limit=limit)]
    notes = [{"id": n["id"], "title": n["title"], "kind": n["kind"], "body": str(n["body"])[:600]}
             for n in memory.search_notes(query, limit=3)]
    if not facts and not notes:
        return ok("Nothing found in memory for that.", facts=[], found=0)
    # Note bodies may be imported documents: data, never instructions.
    return ok(f"Found {len(facts)} facts and {len(notes)} notes.", facts=facts, found=len(facts) + len(notes),
              **({"untrusted": notes} if notes else {}))


@tool("forget",
      description="Forget (delete) a fact from SAM's memory when the user asks to forget something. Give "
                  "fact_id when known from recall, else a query; if several facts match, nothing is deleted "
                  "and the candidates are returned so you can ask which one.",
      description_ckb="سڕینەوەی زانیارییەک لە بیرەوەری",
      params={"type": "object", "properties": {
          "query": {"type": "string", "description": "What to forget, in the user's words."},
          "fact_id": {"type": "integer"}}},
      risk="safe", blocking=True, timeout_s=10,
      examples_ckb=("ئەوەی دەربارەی کاتی لەندەن وتم لەبیری بکە",))
async def forget_tool(ctx: ToolContext, query: str = "", fact_id: int | None = None) -> dict[str, Any]:
    memory = _memory(ctx)
    if memory is None:
        return fail("Memory is not available.")
    if fact_id is not None:
        row = memory.get_fact(fact_id)
        if row is None or row.get("deleted"):
            return fail("No such fact in memory.")
        memory.forget(fact_id)
        return ok("Forgotten.", forgotten=[{"id": row["id"], "text": row["text"]}])
    if not query.strip():
        return fail("Say what to forget (query) or give fact_id.")
    matches = memory.recall(query, limit=4, touch=False)
    if not matches:
        return ok("Nothing in memory matches that, so nothing was deleted.", forgotten=[])
    best = matches[0]
    ambiguous = len(matches) > 1 and float(matches[1].get("rank", 0)) <= float(best.get("rank", 0)) * 0.8
    if ambiguous:
        return fail("Several facts match; nothing deleted. Ask the user which one.",
                    candidates=[{"id": m["id"], "text": m["text"]} for m in matches])
    memory.forget(best["id"])
    return ok("Forgotten.", forgotten=[{"id": best["id"], "text": best["text"]}])


def register(app: Any) -> None:
    app.config.register_defaults({
        "memory.extract_ladder": "extract",
        "memory.facts_in_prompt": 12,
    })
    app.memory = Memory(app)
    for fn in (remember_tool, recall_tool, forget_tool):
        app.tools.add(fn, owner="brain.memory")


__all__ = ["Memory", "register", "search_key", "query_terms", "match_expr", "FACT_KINDS", "NOTE_KINDS",
           "EXTRACT_SCHEMA"]
