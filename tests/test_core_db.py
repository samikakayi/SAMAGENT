from __future__ import annotations

import os

import pytest

from sam.config import Config, parse_env_file
from sam.db import Database, fts_match_expr
from sam.textnorm import normalize_ckb

CORE_TABLES = {"settings", "facts", "facts_fts", "conversations", "turns", "notes", "notes_fts", "strategy_cards",
               "strategy_card_versions", "strategy_fts", "alerts", "drawings", "v1_archive", "usage_counters",
               "timings", "activity", "schema_versions"}


def test_fresh_database_has_every_core_table_in_wal(tmp_path):
    db = Database(tmp_path / "sam2.sqlite3")
    names = {r["name"] for r in db.query("SELECT name FROM sqlite_master WHERE type IN ('table')")}
    assert CORE_TABLES <= names
    assert db.schema_version("core") == 1
    assert db.scalar("PRAGMA journal_mode").lower() == "wal"
    db.close()
    again = Database(tmp_path / "sam2.sqlite3")  # re-open: migrations are idempotent
    assert again.schema_version("core") == 1
    again.close()


def test_namespaced_migrations_apply_in_order_once(tmp_path):
    db = Database(tmp_path / "x.sqlite3")
    calls = []

    def v2(conn):
        calls.append(2)
        conn.execute("ALTER TABLE t_demo ADD COLUMN extra TEXT")

    migrations = [(2, v2), (1, "CREATE TABLE t_demo (id INTEGER PRIMARY KEY, name TEXT); -- comment 'quoted'")]
    assert db.ensure_schema("demo", migrations) == 2
    assert db.ensure_schema("demo", migrations) == 2
    assert calls == [2]
    db.insert("t_demo", {"name": "a", "extra": "b"})
    assert db.query_one("SELECT name, extra FROM t_demo") == {"name": "a", "extra": "b"}


def test_failed_migration_rolls_back(tmp_path):
    db = Database(tmp_path / "x.sqlite3")
    with pytest.raises(Exception):
        db.ensure_schema("broken", [(1, "CREATE TABLE ok_table (id INTEGER); CREATE TABLE ok_table (id INTEGER);")])
    assert db.schema_version("broken") == 0
    assert db.scalar("SELECT count(*) FROM sqlite_master WHERE name='ok_table'") == 0


def test_fts_trigram_finds_sorani_with_spelling_variants(tmp_path):
    db = Database(tmp_path / "x.sqlite3")
    text = "من ستراتیژی لیکویدیتی سویپ لەسەر زێڕ بەکاردەهێنم"
    db.insert("facts", {"text": text, "text_norm": normalize_ckb(text), "kind": "trading", "created_at": 1.0,
                        "updated_at": 1.0})
    # Arabic yeh/kaf variants in the query still match after normalisation.
    query = fts_match_expr("لیكويديتي")
    rows = db.query("SELECT f.text FROM facts_fts JOIN facts f ON f.id = facts_fts.rowid WHERE facts_fts MATCH ?",
                    (query,))
    assert rows and rows[0]["text"] == text
    db.execute("UPDATE facts SET text_norm=? WHERE id=1", (normalize_ckb("شتێکی تر"),))
    assert db.query("SELECT rowid FROM facts_fts WHERE facts_fts MATCH ?", (query,)) == []


def test_fts_match_expr_is_injection_safe():
    assert fts_match_expr('a" OR x NEAR(') == '"near"'
    assert fts_match_expr("ab") is None
    assert fts_match_expr('say "hi" there') == '"say" OR "there"'


def test_strategy_fts_and_versions_tables(tmp_path):
    db = Database(tmp_path / "x.sqlite3")
    db.insert("strategy_cards", {"id": "asia-sweep", "title_ckb": "سویپی ئاسیا", "card": {"rules": []},
                                 "search_text": normalize_ckb("asia sweep سویپی ئاسیا fvg"), "created_at": 1.0,
                                 "updated_at": 1.0})
    db.insert("strategy_card_versions", {"card_id": "asia-sweep", "version": 1, "card": {"rules": []},
                                         "created_at": 1.0})
    hits = db.query("SELECT c.id FROM strategy_fts JOIN strategy_cards c ON c.rowid = strategy_fts.rowid "
                    "WHERE strategy_fts MATCH ?", (fts_match_expr("sweep"),))
    assert hits == [{"id": "asia-sweep"}]


def test_usage_counters_accumulate_per_quota_day(tmp_path):
    db = Database(tmp_path / "x.sqlite3")
    db.bump_usage("groq", "openai/gpt-oss-20b", tokens_in=10, tokens_out=5)
    db.bump_usage("groq", "openai/gpt-oss-20b", errors=1, rate_limited=1)
    usage = db.usage_for("groq", "openai/gpt-oss-20b")
    assert usage["requests"] == 2 and usage["errors"] == 1 and usage["rate_limited"] == 1 and usage["tokens_in"] == 10
    # Gemini days roll over at midnight Pacific; 2026-09-24 05:00 UTC is still the 23rd there.
    assert Database.quota_day("gemini", 1790226000.0) == "2026-09-23"
    assert Database.quota_day("groq", 1790226000.0) == "2026-09-24"


def test_activity_and_transaction_helpers(tmp_path):
    db = Database(tmp_path / "x.sqlite3")
    db.log_activity("tool", "open_app", ok=True, summary="done", detail={"args": {"name": "Chrome"}}, duration_ms=3.2)
    row = db.query_one("SELECT kind, name, ok, summary FROM activity")
    assert row == {"kind": "tool", "name": "open_app", "ok": 1, "summary": "done"}
    with pytest.raises(RuntimeError):
        with db.transaction():
            db.insert("notes", {"body": "x", "body_norm": "x", "created_at": 1, "updated_at": 1})
            raise RuntimeError("boom")
    assert db.scalar("SELECT count(*) FROM notes") == 0


# --- config -----------------------------------------------------------------------

def test_env_file_parser(tmp_path):
    path = tmp_path / ".env"
    path.write_text("# comment\nexport A=1\nB = 'two words'\nC=\"x#y\"\nD=http://h:1/v1 # note\nbad line\n",
                    encoding="utf-8")
    assert parse_env_file(path) == {"A": "1", "B": "two words", "C": "x#y", "D": "http://h:1/v1"}


def test_settings_defaults_set_reset_and_change_callback(tmp_path):
    (tmp_path / ".env").write_text("LITELLM_FAST_MODEL=my-fast\nSAM_DATA_DIR=store\n", encoding="utf-8")
    config = Config(tmp_path, environ={})
    assert config.data_dir == (tmp_path / "store").resolve()
    assert config.get("llm.ladder.chat")[1] == "omniroute:my-fast"
    db = Database(config.db_path)
    changes = []
    config.attach_db(db, on_change=lambda k, v: changes.append((k, v)))
    assert config.get("voice.conversation_timeout_s") == 45
    config.set("voice.conversation_timeout_s", 60)
    assert config.get("voice.conversation_timeout_s") == 60
    assert Config(tmp_path, environ={}).get("unknown.key", "d") == "d"
    config.reset("voice.conversation_timeout_s")
    assert config.get("voice.conversation_timeout_s") == 45
    assert changes[0] == ("voice.conversation_timeout_s", 60)
    config.register_defaults({"hands.new_key": 1, "voice.conversation_timeout_s": 999})
    assert config.get("hands.new_key") == 1 and config.get("voice.conversation_timeout_s") == 45
    # get() returns copies: mutating a list does not change the setting.
    ladder = config.get("llm.ladder.chat")
    ladder.clear()
    assert config.get("llm.ladder.chat")


def test_env_values_are_not_exported(tmp_path):
    (tmp_path / ".env").write_text("SAM2_TEST_ONLY_VAR=zzz\n", encoding="utf-8")
    config = Config(tmp_path, environ={})
    assert config.env_value("SAM2_TEST_ONLY_VAR") == "zzz"
    assert "SAM2_TEST_ONLY_VAR" not in os.environ
    assert config.env_names() == ["SAM2_TEST_ONLY_VAR"]
