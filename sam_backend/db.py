from __future__ import annotations

import json
import hashlib
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class Database:
    """Small, thread-safe SQLite repository with explicit, inspectable tables."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.RLock()
        self.migrate()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=15, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=15000")
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        """Serialised write transaction, committed on clean exit.

        Modules that own their own tables (the task store, for example) use
        this rather than reaching for the private write lock, so every writer
        still serialises through one place.
        """
        with self._write_lock, self.connect() as connection:
            yield connection
            connection.commit()

    def migrate(self) -> None:
        schema = """
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            provider TEXT NOT NULL DEFAULT 'ollama',
            model TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            content TEXT NOT NULL DEFAULT '',
            tool_name TEXT,
            tool_call_id TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            tags_json TEXT NOT NULL DEFAULT '[]',
            domain TEXT NOT NULL DEFAULT 'user',
            importance REAL NOT NULL DEFAULT 0.5,
            source_conversation_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(id UNINDEXED, content, tags);
        CREATE TABLE IF NOT EXISTS approvals (
            id TEXT PRIMARY KEY,
            conversation_id TEXT,
            task_id TEXT,
            tool_name TEXT NOT NULL,
            tool_call_id TEXT NOT NULL DEFAULT '',
            risk_level TEXT NOT NULL,
            reason TEXT NOT NULL,
            arguments_json TEXT NOT NULL,
            request_hash TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'pending',
            requested_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            decided_at TEXT,
            decision_note TEXT,
            result_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_approvals_status ON approvals(status, requested_at DESC);
        CREATE TABLE IF NOT EXISTS audit_log (
            id TEXT PRIMARY KEY,
            event_type TEXT NOT NULL,
            actor TEXT NOT NULL,
            conversation_id TEXT,
            tool_name TEXT,
            risk_level TEXT,
            status TEXT NOT NULL,
            summary TEXT NOT NULL,
            details_json TEXT NOT NULL DEFAULT '{}',
            previous_hash TEXT NOT NULL DEFAULT '',
            entry_hash TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_log(created_at DESC);
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS model_usage (
            id TEXT PRIMARY KEY,
            task_id TEXT,
            conversation_id TEXT,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            route_mode TEXT NOT NULL,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cost_usd REAL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_model_usage_created ON model_usage(created_at DESC);
        CREATE TABLE IF NOT EXISTS custom_theories (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            version INTEGER NOT NULL,
            definition_json TEXT NOT NULL,
            archived INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(name, version)
        );
        CREATE TABLE IF NOT EXISTS trading_setups (
            id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            feed TEXT,
            theory TEXT NOT NULL,
            state TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            monitor_enabled INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_trading_setups_state ON trading_setups(state, updated_at DESC);
        CREATE TABLE IF NOT EXISTS setup_events (
            id TEXT PRIMARY KEY,
            setup_id TEXT NOT NULL REFERENCES trading_setups(id) ON DELETE CASCADE,
            previous_state TEXT,
            state TEXT NOT NULL,
            reason TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS trading_journal (
            id TEXT PRIMARY KEY,
            setup_id TEXT,
            symbol TEXT NOT NULL,
            theory TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS drawing_ownership (
            id TEXT PRIMARY KEY,
            setup_id TEXT,
            symbol TEXT NOT NULL,
            timeframe TEXT,
            layer TEXT NOT NULL,
            drawing_type TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            visible INTEGER NOT NULL DEFAULT 1,
            verified INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS trading_context (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chart_calibration (
            id TEXT PRIMARY KEY,
            window_handle INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            geometry_hash TEXT NOT NULL,
            slope REAL NOT NULL,
            intercept REAL NOT NULL,
            method TEXT NOT NULL,
            anchors_json TEXT NOT NULL DEFAULT '[]',
            verified INTEGER NOT NULL DEFAULT 0,
            max_error REAL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(window_handle, symbol, timeframe, geometry_hash)
        );
        """
        with self._write_lock, self.connect() as connection:
            connection.executescript(schema)
            approval_columns = {row[1] for row in connection.execute("PRAGMA table_info(approvals)")}
            # An approval belongs to whichever runtime raised it. Without this
            # the chat loop and the autonomous orchestrator cannot tell each
            # other's requests apart, and resolving one through the wrong
            # resolver fails.
            if "task_id" not in approval_columns:
                connection.execute("ALTER TABLE approvals ADD COLUMN task_id TEXT")
            if "tool_call_id" not in approval_columns:
                connection.execute("ALTER TABLE approvals ADD COLUMN tool_call_id TEXT NOT NULL DEFAULT ''")
            if "request_hash" not in approval_columns:
                connection.execute("ALTER TABLE approvals ADD COLUMN request_hash TEXT NOT NULL DEFAULT ''")
            audit_columns = {row[1] for row in connection.execute("PRAGMA table_info(audit_log)")}
            if "previous_hash" not in audit_columns:
                connection.execute("ALTER TABLE audit_log ADD COLUMN previous_hash TEXT NOT NULL DEFAULT ''")
            if "entry_hash" not in audit_columns:
                connection.execute("ALTER TABLE audit_log ADD COLUMN entry_hash TEXT NOT NULL DEFAULT ''")
            memory_columns = {row[1] for row in connection.execute("PRAGMA table_info(memories)")}
            if "domain" not in memory_columns:
                connection.execute("ALTER TABLE memories ADD COLUMN domain TEXT NOT NULL DEFAULT 'user'")
            # Ownership is queried by theory/strategy/label, so those stay real
            # columns instead of hiding inside the payload blob.
            calibration_columns = {row[1] for row in connection.execute("PRAGMA table_info(chart_calibration)")}
            for column, definition in (
                ("axis_x", "REAL"),
                ("precision", "REAL"),
                ("minutes_per_pixel", "REAL"),
                ("time_intercept", "REAL"),
                ("time_axis_y", "REAL"),
                ("minutes_span", "TEXT"),
            ):
                if column not in calibration_columns:
                    connection.execute(f"ALTER TABLE chart_calibration ADD COLUMN {column} {definition}")
            drawing_columns = {row[1] for row in connection.execute("PRAGMA table_info(drawing_ownership)")}
            for column, definition in (
                ("theory", "TEXT NOT NULL DEFAULT ''"),
                ("strategy", "TEXT NOT NULL DEFAULT ''"),
                ("label", "TEXT NOT NULL DEFAULT ''"),
                ("price", "REAL"),
                ("price_secondary", "REAL"),
                ("anchor_time", "TEXT"),
                ("external_id", "TEXT"),
                ("verified_at", "TEXT"),
            ):
                if column not in drawing_columns:
                    connection.execute(f"ALTER TABLE drawing_ownership ADD COLUMN {column} {definition}")
            connection.commit()

    @staticmethod
    def _id(prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex}"

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row is not None else None

    def create_conversation(self, title: str, provider: str, model: str) -> dict[str, Any]:
        conversation_id = self._id("conv")
        now = utc_now()
        with self._write_lock, self.connect() as connection:
            connection.execute(
                "INSERT INTO conversations(id,title,provider,model,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                (conversation_id, title.strip()[:160] or "New conversation", provider, model, now, now),
            )
            connection.commit()
        return self.get_conversation(conversation_id) or {}

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            return self._row(connection.execute("SELECT * FROM conversations WHERE id=?", (conversation_id,)).fetchone())

    def update_conversation_title(self, conversation_id: str, title: str) -> dict[str, Any] | None:
        normalized = title.strip()
        if not normalized or len(normalized) > 160:
            raise ValueError("Conversation title must contain between 1 and 160 characters")
        with self._write_lock, self.connect() as connection:
            cursor = connection.execute(
                "UPDATE conversations SET title=?,updated_at=? WHERE id=?",
                (normalized, utc_now(), conversation_id),
            )
            connection.commit()
        return self.get_conversation(conversation_id) if cursor.rowcount else None

    def list_conversations(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.conversation_id=c.id) AS message_count "
                "FROM conversations c ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._write_lock, self.connect() as connection:
            cursor = connection.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
            connection.commit()
            return cursor.rowcount > 0

    def update_conversation_model(self, conversation_id: str, provider: str, model: str) -> dict[str, Any] | None:
        with self._write_lock, self.connect() as connection:
            connection.execute(
                "UPDATE conversations SET provider=?, model=?, updated_at=? WHERE id=?",
                (provider, model, utc_now(), conversation_id),
            )
            connection.commit()
        return self.get_conversation(conversation_id)

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        message_id = self._id("msg")
        now = utc_now()
        with self._write_lock, self.connect() as connection:
            connection.execute(
                "INSERT INTO messages(id,conversation_id,role,content,tool_name,tool_call_id,metadata_json,created_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (message_id, conversation_id, role, content, tool_name, tool_call_id, json.dumps(metadata or {}), now),
            )
            connection.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, conversation_id))
            connection.commit()
        return self.get_message(message_id) or {}

    def get_message(self, message_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM messages WHERE id=?", (message_id,)).fetchone()
        return self._decode_message(self._row(row))

    @staticmethod
    def _decode_message(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is not None:
            row["metadata"] = json.loads(row.pop("metadata_json", "{}"))
        return row

    def list_messages(self, conversation_id: str, limit: int = 200) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM (SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at DESC LIMIT ?) "
                "ORDER BY created_at", (conversation_id, limit)
            ).fetchall()
        return [self._decode_message(dict(row)) or {} for row in rows]

    def add_memory(
        self,
        content: str,
        tags: list[str] | None = None,
        importance: float = 0.5,
        source_conversation_id: str | None = None,
        domain: str = "user",
    ) -> dict[str, Any]:
        memory_id = self._id("mem")
        now = utc_now()
        tags = [str(tag).strip() for tag in (tags or []) if str(tag).strip()]
        with self._write_lock, self.connect() as connection:
            connection.execute(
                "INSERT INTO memories(id,content,tags_json,domain,importance,source_conversation_id,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                (memory_id, content.strip(), json.dumps(tags), domain, max(0.0, min(1.0, importance)), source_conversation_id, now, now),
            )
            connection.execute("INSERT INTO memory_fts(id,content,tags) VALUES(?,?,?)", (memory_id, content.strip(), " ".join(tags)))
            connection.commit()
        return self.get_memory(memory_id) or {}

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
        return self._decode_memory(self._row(row))

    @staticmethod
    def _decode_memory(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is not None:
            row["tags"] = json.loads(row.pop("tags_json", "[]"))
        return row

    def list_memories(self, query: str = "", limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as connection:
            if query.strip():
                terms = re.findall(r"[^\W_]+", query, flags=re.UNICODE)[:16]
                if terms:
                    fts_query = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms)
                    try:
                        rows = connection.execute(
                            "SELECT m.* FROM memory_fts f JOIN memories m ON m.id=f.id WHERE memory_fts MATCH ? "
                            "ORDER BY bm25(memory_fts), m.importance DESC LIMIT ?", (fts_query, limit)
                        ).fetchall()
                    except sqlite3.OperationalError:
                        rows = connection.execute(
                            "SELECT * FROM memories ORDER BY importance DESC, updated_at DESC LIMIT ?", (limit,)
                        ).fetchall()
                else:
                    rows = connection.execute(
                        "SELECT * FROM memories ORDER BY importance DESC, updated_at DESC LIMIT ?", (limit,)
                    ).fetchall()
            else:
                rows = connection.execute("SELECT * FROM memories ORDER BY importance DESC, updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._decode_memory(dict(row)) or {} for row in rows]

    def delete_memory(self, memory_id: str) -> bool:
        with self._write_lock, self.connect() as connection:
            connection.execute("DELETE FROM memory_fts WHERE id=?", (memory_id,))
            cursor = connection.execute("DELETE FROM memories WHERE id=?", (memory_id,))
            connection.commit()
            return cursor.rowcount > 0

    def create_approval(
        self,
        *,
        conversation_id: str | None,
        tool_name: str,
        tool_call_id: str,
        risk_level: str,
        reason: str,
        arguments: dict[str, Any],
        ttl_minutes: int,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        approval_id = self._id("apr")
        now_dt = datetime.now(UTC)
        request_hash = self.approval_hash(conversation_id, tool_name, arguments, tool_call_id)
        with self._write_lock, self.connect() as connection:
            connection.execute(
                "INSERT INTO approvals(id,conversation_id,task_id,tool_name,tool_call_id,risk_level,reason,arguments_json,request_hash,status,requested_at,expires_at) "
                "VALUES(?,?,?,?,?,?,?,?,?, 'pending',?,?)",
                (approval_id, conversation_id, task_id, tool_name, tool_call_id, risk_level, reason, json.dumps(arguments), request_hash, now_dt.isoformat(), (now_dt + timedelta(minutes=ttl_minutes)).isoformat()),
            )
            connection.commit()
        return self.get_approval(approval_id) or {}

    @staticmethod
    def approval_hash(conversation_id: str | None, tool_name: str, arguments: dict[str, Any], tool_call_id: str = "") -> str:
        canonical = json.dumps(
            {"conversation_id": conversation_id, "tool_name": tool_name, "tool_call_id": tool_call_id, "arguments": arguments},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _decode_approval(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is not None:
            row["arguments"] = json.loads(row.pop("arguments_json", "{}"))
            if row.get("result_json"):
                row["result"] = json.loads(row.pop("result_json"))
            else:
                row.pop("result_json", None)
                row["result"] = None
        return row

    def get_approval(self, approval_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM approvals WHERE id=?", (approval_id,)).fetchone()
        return self._decode_approval(self._row(row))

    def list_approvals(self, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as connection:
            if status:
                rows = connection.execute("SELECT * FROM approvals WHERE status=? ORDER BY requested_at DESC LIMIT ?", (status, limit)).fetchall()
            else:
                rows = connection.execute("SELECT * FROM approvals ORDER BY requested_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._decode_approval(dict(row)) or {} for row in rows]

    def decide_approval(self, approval_id: str, decision: str, note: str = "") -> dict[str, Any] | None:
        now = utc_now()
        with self._write_lock, self.connect() as connection:
            connection.execute(
                "UPDATE approvals SET status=?, decided_at=?, decision_note=? WHERE id=? AND status='pending'",
                (decision, now, note[:1000], approval_id),
            )
            connection.commit()
        return self.get_approval(approval_id)

    def authorize_approval(self, approval_id: str, decision: str, note: str = "") -> dict[str, Any] | None:
        """Atomically deny or claim a single-use approval for execution."""
        now = utc_now()
        with self._write_lock, self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM approvals WHERE id=?", (approval_id,)).fetchone()
            if row is None:
                connection.rollback()
                return None
            record = dict(row)
            if record["status"] != "pending":
                connection.rollback()
                return self._decode_approval(record)
            if record["expires_at"] <= now:
                connection.execute("UPDATE approvals SET status='expired', decided_at=? WHERE id=?", (now, approval_id))
                connection.commit()
                return self.get_approval(approval_id)
            new_status = "executing" if decision == "approved" else "denied"
            connection.execute(
                "UPDATE approvals SET status=?, decided_at=?, decision_note=? WHERE id=? AND status='pending'",
                (new_status, now, note[:1000], approval_id),
            )
            connection.commit()
        return self.get_approval(approval_id)

    def set_approval_result(self, approval_id: str, result: dict[str, Any], status: str = "executed") -> None:
        with self._write_lock, self.connect() as connection:
            connection.execute("UPDATE approvals SET status=?, result_json=? WHERE id=?", (status, json.dumps(result), approval_id))
            connection.commit()

    def add_audit(
        self,
        event_type: str,
        status: str,
        summary: str,
        *,
        actor: str = "sam",
        conversation_id: str | None = None,
        tool_name: str | None = None,
        risk_level: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        audit_id = self._id("audit")
        now = utc_now()
        details_json = json.dumps(details or {}, ensure_ascii=False, sort_keys=True, default=str)
        with self._write_lock, self.connect() as connection:
            previous = connection.execute("SELECT entry_hash FROM audit_log ORDER BY rowid DESC LIMIT 1").fetchone()
            previous_hash = str(previous["entry_hash"]) if previous else ""
            canonical = json.dumps(
                {
                    "id": audit_id, "event_type": event_type, "actor": actor, "conversation_id": conversation_id,
                    "tool_name": tool_name, "risk_level": risk_level, "status": status, "summary": summary[:1000],
                    "details_json": details_json, "created_at": now,
                }, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            )
            entry_hash = hashlib.sha256((previous_hash + canonical).encode("utf-8")).hexdigest()
            connection.execute(
                "INSERT INTO audit_log(id,event_type,actor,conversation_id,tool_name,risk_level,status,summary,details_json,previous_hash,entry_hash,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (audit_id, event_type, actor, conversation_id, tool_name, risk_level, status, summary[:1000], details_json, previous_hash, entry_hash, now),
            )
            connection.commit()
        return self.get_audit(audit_id) or {}

    def get_audit(self, audit_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM audit_log WHERE id=?", (audit_id,)).fetchone()
        return self._decode_audit(self._row(row))

    @staticmethod
    def _decode_audit(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is not None:
            row["details"] = json.loads(row.pop("details_json", "{}"))
        return row

    def list_audit(self, limit: int = 200, event_type: str | None = None) -> list[dict[str, Any]]:
        with self.connect() as connection:
            if event_type:
                rows = connection.execute("SELECT * FROM audit_log WHERE event_type=? ORDER BY created_at DESC LIMIT ?", (event_type, limit)).fetchall()
            else:
                rows = connection.execute("SELECT * FROM audit_log ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._decode_audit(dict(row)) or {} for row in rows]

    def verify_audit_chain(self) -> bool:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM audit_log ORDER BY rowid").fetchall()
        previous_hash = ""
        for raw in rows:
            row = dict(raw)
            if row.get("previous_hash", "") != previous_hash:
                return False
            canonical = json.dumps(
                {
                    "id": row["id"], "event_type": row["event_type"], "actor": row["actor"],
                    "conversation_id": row["conversation_id"], "tool_name": row["tool_name"],
                    "risk_level": row["risk_level"], "status": row["status"], "summary": row["summary"],
                    "details_json": row["details_json"], "created_at": row["created_at"],
                }, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            )
            expected = hashlib.sha256((previous_hash + canonical).encode("utf-8")).hexdigest()
            if row.get("entry_hash") != expected:
                return False
            previous_hash = expected
        return True

    def get_settings(self) -> dict[str, Any]:
        with self.connect() as connection:
            rows = connection.execute("SELECT key,value_json FROM settings").fetchall()
        return {row["key"]: json.loads(row["value_json"]) for row in rows}

    def update_settings(self, values: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        with self._write_lock, self.connect() as connection:
            for key, value in values.items():
                connection.execute(
                    "INSERT INTO settings(key,value_json,updated_at) VALUES(?,?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at",
                    (key, json.dumps(value), now),
                )
            connection.commit()
        return self.get_settings()

    def add_model_usage(
        self,
        *,
        provider: str,
        model: str,
        route_mode: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost_usd: float | None = None,
        task_id: str | None = None,
        conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        usage_id = self._id("usage")
        with self._write_lock, self.connect() as connection:
            connection.execute(
                "INSERT INTO model_usage(id,task_id,conversation_id,provider,model,route_mode,input_tokens,output_tokens,cost_usd,metadata_json,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    usage_id, task_id, conversation_id, provider, model, route_mode,
                    input_tokens, output_tokens, cost_usd,
                    json.dumps(metadata or {}, ensure_ascii=False, default=str), utc_now(),
                ),
            )
            connection.commit()
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM model_usage WHERE id=?", (usage_id,)).fetchone()
        result = dict(row) if row else {}
        if result:
            result["metadata"] = json.loads(result.pop("metadata_json", "{}"))
        return result

    def list_model_usage(self, limit: int = 500) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM model_usage ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        result = []
        for raw in rows:
            row = dict(raw)
            row["metadata"] = json.loads(row.pop("metadata_json", "{}"))
            result.append(row)
        return result

    def model_cost_summary(self) -> dict[str, Any]:
        now = datetime.now(UTC)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        rows = self.list_model_usage(100_000)
        day_rows = [row for row in rows if datetime.fromisoformat(row["created_at"]) >= day_start]
        month_rows = [row for row in rows if datetime.fromisoformat(row["created_at"]) >= month_start]

        def summary(items: list[dict[str, Any]]) -> dict[str, Any]:
            known = [float(item["cost_usd"]) for item in items if item.get("cost_usd") is not None]
            return {
                "requests": len(items),
                "known_cost_requests": len(known),
                "cost_usd": round(sum(known), 8),
                "input_tokens": sum(int(item.get("input_tokens") or 0) for item in items),
                "output_tokens": sum(int(item.get("output_tokens") or 0) for item in items),
            }

        return {"today": summary(day_rows), "month": summary(month_rows), "recent": rows[:50]}

    def save_custom_theory(self, name: str, definition: dict[str, Any]) -> dict[str, Any]:
        clean_name = name.strip()[:120]
        if not clean_name:
            raise ValueError("Theory name cannot be empty")
        now = utc_now()
        theory_id = self._id("theory")
        with self._write_lock, self.connect() as connection:
            row = connection.execute("SELECT MAX(version) AS version FROM custom_theories WHERE lower(name)=lower(?)", (clean_name,)).fetchone()
            version = int(row["version"] or 0) + 1
            connection.execute(
                "INSERT INTO custom_theories(id,name,version,definition_json,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                (theory_id, clean_name, version, json.dumps(definition, ensure_ascii=False, default=str), now, now),
            )
            connection.commit()
        return self.get_custom_theory(theory_id=theory_id) or {}

    def get_custom_theory(
        self,
        *,
        theory_id: str | None = None,
        name: str | None = None,
        version: int | None = None,
    ) -> dict[str, Any] | None:
        if not theory_id and not name:
            raise ValueError("theory_id or name is required")
        with self.connect() as connection:
            if theory_id:
                row = connection.execute("SELECT * FROM custom_theories WHERE id=?", (theory_id,)).fetchone()
            elif version is not None:
                row = connection.execute("SELECT * FROM custom_theories WHERE lower(name)=lower(?) AND version=?", (name, version)).fetchone()
            else:
                row = connection.execute("SELECT * FROM custom_theories WHERE lower(name)=lower(?) ORDER BY version DESC LIMIT 1", (name,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["definition"] = json.loads(result.pop("definition_json", "{}"))
        result["archived"] = bool(result["archived"])
        return result

    def list_custom_theories(self, include_archived: bool = False) -> list[dict[str, Any]]:
        query = "SELECT * FROM custom_theories" + ("" if include_archived else " WHERE archived=0") + " ORDER BY lower(name), version DESC"
        with self.connect() as connection:
            rows = connection.execute(query).fetchall()
        result = []
        for raw in rows:
            row = dict(raw)
            row["definition"] = json.loads(row.pop("definition_json", "{}"))
            row["archived"] = bool(row["archived"])
            result.append(row)
        return result

    def archive_custom_theory(self, theory_id: str) -> bool:
        with self._write_lock, self.connect() as connection:
            cursor = connection.execute("UPDATE custom_theories SET archived=1,updated_at=? WHERE id=?", (utc_now(), theory_id))
            connection.commit()
            return cursor.rowcount > 0

    def set_trading_context(self, values: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        with self._write_lock, self.connect() as connection:
            for key, value in values.items():
                connection.execute(
                    "INSERT INTO trading_context(key,value_json,updated_at) VALUES(?,?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,updated_at=excluded.updated_at",
                    (key, json.dumps(value, ensure_ascii=False, default=str), now),
                )
            connection.commit()
        return self.get_trading_context()

    def get_trading_context(self) -> dict[str, Any]:
        with self.connect() as connection:
            rows = connection.execute("SELECT key,value_json,updated_at FROM trading_context").fetchall()
        return {row["key"]: json.loads(row["value_json"]) for row in rows}

    def create_trading_setup(self, symbol: str, feed: str | None, theory: str, state: str, payload: dict[str, Any]) -> dict[str, Any]:
        setup_id = self._id("setup")
        now = utc_now()
        with self._write_lock, self.connect() as connection:
            connection.execute(
                "INSERT INTO trading_setups(id,symbol,feed,theory,state,payload_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                (setup_id, symbol, feed, theory, state, json.dumps(payload, ensure_ascii=False, default=str), now, now),
            )
            connection.execute(
                "INSERT INTO setup_events(id,setup_id,previous_state,state,reason,payload_json,created_at) VALUES(?,?,?,?,?,?,?)",
                (self._id("event"), setup_id, None, state, "Setup created", "{}", now),
            )
            connection.commit()
        return self.get_trading_setup(setup_id) or {}

    def get_trading_setup(self, setup_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM trading_setups WHERE id=?", (setup_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["payload"] = json.loads(result.pop("payload_json", "{}"))
        result["monitor_enabled"] = bool(result["monitor_enabled"])
        return result

    def list_trading_setups(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM trading_setups ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        result = []
        for raw in rows:
            row = dict(raw)
            row["payload"] = json.loads(row.pop("payload_json", "{}"))
            row["monitor_enabled"] = bool(row["monitor_enabled"])
            result.append(row)
        return result

    def transition_trading_setup(self, setup_id: str, state: str, reason: str, payload: dict[str, Any] | None = None) -> dict[str, Any] | None:
        now = utc_now()
        with self._write_lock, self.connect() as connection:
            row = connection.execute("SELECT state,payload_json FROM trading_setups WHERE id=?", (setup_id,)).fetchone()
            if row is None:
                return None
            previous = row["state"]
            merged = json.loads(row["payload_json"] or "{}")
            merged.update(payload or {})
            connection.execute(
                "UPDATE trading_setups SET state=?,payload_json=?,updated_at=? WHERE id=?",
                (state, json.dumps(merged, ensure_ascii=False, default=str), now, setup_id),
            )
            connection.execute(
                "INSERT INTO setup_events(id,setup_id,previous_state,state,reason,payload_json,created_at) VALUES(?,?,?,?,?,?,?)",
                (self._id("event"), setup_id, previous, state, reason[:1000], json.dumps(payload or {}, ensure_ascii=False, default=str), now),
            )
            connection.commit()
        return self.get_trading_setup(setup_id)

    def set_setup_monitoring(self, setup_id: str, enabled: bool) -> dict[str, Any] | None:
        with self._write_lock, self.connect() as connection:
            cursor = connection.execute("UPDATE trading_setups SET monitor_enabled=?,updated_at=? WHERE id=?", (int(enabled), utc_now(), setup_id))
            connection.commit()
        return self.get_trading_setup(setup_id) if cursor.rowcount else None

    def list_setup_events(self, setup_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM setup_events WHERE setup_id=? ORDER BY created_at", (setup_id,)).fetchall()
        result = []
        for raw in rows:
            row = dict(raw)
            row["payload"] = json.loads(row.pop("payload_json", "{}"))
            result.append(row)
        return result

    # --- Drawing ownership -------------------------------------------------
    # SAM may only hide, re-verify, or delete annotations it recorded here.
    # Anything the user drew by hand is never represented in this table and is
    # therefore unreachable by every clear/delete path below.

    @staticmethod
    def _drawing_row(raw: sqlite3.Row) -> dict[str, Any]:
        row = dict(raw)
        row["payload"] = json.loads(row.pop("payload_json", "{}") or "{}")
        row["visible"] = bool(row["visible"])
        row["verified"] = bool(row["verified"])
        return row

    def record_drawing(
        self,
        *,
        symbol: str,
        layer: str,
        drawing_type: str,
        label: str = "",
        theory: str = "",
        strategy: str = "",
        timeframe: str | None = None,
        setup_id: str | None = None,
        price: float | None = None,
        price_secondary: float | None = None,
        anchor_time: str | None = None,
        external_id: str | None = None,
        verified: bool = False,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        drawing_id = self._id("draw")
        now = utc_now()
        with self._write_lock, self.connect() as connection:
            connection.execute(
                "INSERT INTO drawing_ownership("
                "id,setup_id,symbol,timeframe,layer,drawing_type,payload_json,visible,verified,"
                "created_at,updated_at,theory,strategy,label,price,price_secondary,anchor_time,external_id,verified_at"
                ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    drawing_id, setup_id, symbol.upper(), timeframe, layer, drawing_type,
                    json.dumps(payload or {}, ensure_ascii=False, default=str), 1, int(verified),
                    now, now, theory, strategy, label, price, price_secondary, anchor_time, external_id,
                    now if verified else None,
                ),
            )
            connection.commit()
        return self.get_drawing(drawing_id) or {}

    def get_drawing(self, drawing_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM drawing_ownership WHERE id=?", (drawing_id,)).fetchone()
        return self._drawing_row(row) if row is not None else None

    def list_drawings(
        self,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        layer: str | None = None,
        theory: str | None = None,
        setup_id: str | None = None,
        visible_only: bool = False,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        for column, value in (
            ("symbol", symbol.upper() if symbol else None),
            ("timeframe", timeframe),
            ("layer", layer),
            ("theory", theory),
            ("setup_id", setup_id),
        ):
            if value:
                clauses.append(f"{column}=?")
                parameters.append(value)
        if visible_only:
            clauses.append("visible=1")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(limit)
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM drawing_ownership{where} ORDER BY created_at DESC LIMIT ?", tuple(parameters)
            ).fetchall()
        return [self._drawing_row(row) for row in rows]

    def set_drawing_visibility(
        self,
        *,
        visible: bool,
        drawing_id: str | None = None,
        symbol: str | None = None,
        layer: str | None = None,
        theory: str | None = None,
        setup_id: str | None = None,
    ) -> int:
        clauses: list[str] = []
        parameters: list[Any] = [int(visible), utc_now()]
        for column, value in (
            ("id", drawing_id),
            ("symbol", symbol.upper() if symbol else None),
            ("layer", layer),
            ("theory", theory),
            ("setup_id", setup_id),
        ):
            if value:
                clauses.append(f"{column}=?")
                parameters.append(value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._write_lock, self.connect() as connection:
            cursor = connection.execute(f"UPDATE drawing_ownership SET visible=?,updated_at=?{where}", tuple(parameters))
            connection.commit()
        return cursor.rowcount

    def mark_drawing_verified(self, drawing_id: str, verified: bool, payload: dict[str, Any] | None = None) -> dict[str, Any] | None:
        now = utc_now()
        with self._write_lock, self.connect() as connection:
            row = connection.execute("SELECT payload_json FROM drawing_ownership WHERE id=?", (drawing_id,)).fetchone()
            if row is None:
                return None
            merged = json.loads(row["payload_json"] or "{}")
            merged.update(payload or {})
            connection.execute(
                "UPDATE drawing_ownership SET verified=?,verified_at=?,payload_json=?,updated_at=? WHERE id=?",
                (int(verified), now if verified else None, json.dumps(merged, ensure_ascii=False, default=str), now, drawing_id),
            )
            connection.commit()
        return self.get_drawing(drawing_id)

    def delete_drawings(
        self,
        *,
        drawing_id: str | None = None,
        symbol: str | None = None,
        layer: str | None = None,
        theory: str | None = None,
        setup_id: str | None = None,
        all_owned: bool = False,
    ) -> list[dict[str, Any]]:
        """Delete only SAM-owned rows. `all_owned` is required for an unfiltered clear."""
        selectors = {
            "id": drawing_id,
            "symbol": symbol.upper() if symbol else None,
            "layer": layer,
            "theory": theory,
            "setup_id": setup_id,
        }
        clauses = [f"{column}=?" for column, value in selectors.items() if value]
        parameters = [value for value in selectors.values() if value]
        if not clauses and not all_owned:
            raise ValueError("Refusing an unfiltered drawing delete without an explicit all_owned request.")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._write_lock, self.connect() as connection:
            rows = connection.execute(f"SELECT * FROM drawing_ownership{where}", tuple(parameters)).fetchall()
            removed = [self._drawing_row(row) for row in rows]
            connection.execute(f"DELETE FROM drawing_ownership{where}", tuple(parameters))
            connection.commit()
        return removed

    # --- Chart calibration -------------------------------------------------

    def save_chart_calibration(
        self,
        *,
        window_handle: int,
        symbol: str,
        timeframe: str,
        geometry_hash: str,
        slope: float,
        intercept: float,
        method: str,
        anchors: list[dict[str, Any]] | None = None,
        verified: bool = False,
        max_error: float | None = None,
        axis_x: float | None = None,
        precision: float | None = None,
        minutes_per_pixel: float | None = None,
        time_intercept: float | None = None,
        time_axis_y: float | None = None,
        minutes_span: list[float] | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        with self._write_lock, self.connect() as connection:
            connection.execute(
                "INSERT INTO chart_calibration("
                "id,window_handle,symbol,timeframe,geometry_hash,slope,intercept,method,anchors_json,verified,max_error,created_at,updated_at,"
                "axis_x,precision,minutes_per_pixel,time_intercept,time_axis_y,minutes_span"
                ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(window_handle,symbol,timeframe,geometry_hash) DO UPDATE SET "
                "slope=excluded.slope,intercept=excluded.intercept,method=excluded.method,"
                "anchors_json=excluded.anchors_json,verified=excluded.verified,max_error=excluded.max_error,"
                "updated_at=excluded.updated_at,axis_x=excluded.axis_x,precision=excluded.precision,"
                "minutes_per_pixel=excluded.minutes_per_pixel,time_intercept=excluded.time_intercept,"
                "time_axis_y=excluded.time_axis_y,minutes_span=excluded.minutes_span",
                (
                    self._id("cal"), int(window_handle), symbol.upper(), timeframe, geometry_hash,
                    float(slope), float(intercept), method,
                    json.dumps(anchors or [], ensure_ascii=False, default=str),
                    int(verified), max_error, now, now,
                    axis_x, precision, minutes_per_pixel, time_intercept, time_axis_y,
                    json.dumps(minutes_span) if minutes_span else None,
                ),
            )
            connection.commit()
        return self.get_chart_calibration(
            window_handle=window_handle, symbol=symbol, timeframe=timeframe, geometry_hash=geometry_hash
        ) or {}

    def get_chart_calibration(
        self, *, window_handle: int, symbol: str, timeframe: str, geometry_hash: str
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM chart_calibration WHERE window_handle=? AND symbol=? AND timeframe=? AND geometry_hash=?",
                (int(window_handle), symbol.upper(), timeframe, geometry_hash),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["anchors"] = json.loads(result.pop("anchors_json", "[]") or "[]")
        span = result.get("minutes_span")
        result["minutes_span"] = json.loads(span) if isinstance(span, str) and span else None
        result["verified"] = bool(result["verified"])
        return result

    def invalidate_chart_calibration(self, *, window_handle: int, symbol: str | None = None, timeframe: str | None = None) -> int:
        clauses = ["window_handle=?"]
        parameters: list[Any] = [int(window_handle)]
        if symbol:
            clauses.append("symbol=?")
            parameters.append(symbol.upper())
        if timeframe:
            clauses.append("timeframe=?")
            parameters.append(timeframe)
        with self._write_lock, self.connect() as connection:
            cursor = connection.execute(f"DELETE FROM chart_calibration WHERE {' AND '.join(clauses)}", tuple(parameters))
            connection.commit()
        return cursor.rowcount

    def add_trading_journal(self, symbol: str, theory: str, payload: dict[str, Any], setup_id: str | None = None) -> dict[str, Any]:
        journal_id = self._id("journal")
        now = utc_now()
        with self._write_lock, self.connect() as connection:
            connection.execute(
                "INSERT INTO trading_journal(id,setup_id,symbol,theory,payload_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                (journal_id, setup_id, symbol, theory, json.dumps(payload, ensure_ascii=False, default=str), now, now),
            )
            connection.commit()
        return {"id": journal_id, "setup_id": setup_id, "symbol": symbol, "theory": theory, "payload": payload, "created_at": now, "updated_at": now}

    def list_trading_journal(self, query: str = "", limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as connection:
            if query:
                rows = connection.execute(
                    "SELECT * FROM trading_journal WHERE symbol LIKE ? OR theory LIKE ? OR payload_json LIKE ? ORDER BY created_at DESC LIMIT ?",
                    tuple([f"%{query}%"] * 3 + [limit]),
                ).fetchall()
            else:
                rows = connection.execute("SELECT * FROM trading_journal ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        result = []
        for raw in rows:
            row = dict(raw)
            row["payload"] = json.loads(row.pop("payload_json", "{}"))
            result.append(row)
        return result
