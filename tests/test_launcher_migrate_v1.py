"""sam.migrate_v1: v1 data comes over read-only, archived, keys never copied."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from sam import migrate_v1
from tests.conftest import FAKE_GEMINI, FAKE_GROQ
from tests.launcher_helpers import make_v1_db


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _side_files(path: Path) -> set[str]:
    return {p.name for p in path.parent.iterdir() if p.name.startswith(path.name + "-")}


def _all_text(db) -> str:
    """Every text value in every SAM 2 table (for 'no key anywhere' checks)."""
    chunks = []
    for (table,) in db._conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
        if table.endswith(("_data", "_idx", "_docsize", "_config", "_content")):
            continue
        for row in db._conn.execute(f'SELECT * FROM "{table}"').fetchall():
            chunks.extend(str(v) for v in tuple(row) if v is not None)
    return "\n".join(chunks)


@pytest.fixture
def v1_db(home: Path) -> Path:
    return make_v1_db(home / "data" / "sam.sqlite3", secret=FAKE_GROQ)


@pytest.fixture
def app(make_app, v1_db):
    app = make_app()
    app.load_packages(["sam.migrate_v1"])
    return app


def test_theories_become_archived_cards_with_version_history(app, v1_db):
    report = migrate_v1.run_migration(app)

    assert report["ok"] and report["strategy_cards"] == 2 and report["card_versions"] == 4
    cards = {c["id"]: c for c in app.db.query("SELECT * FROM strategy_cards")}
    assert set(cards) == {"v1-daily-workflow-strategy", "v1-ui-acceptance-strategy"}
    daily = cards["v1-daily-workflow-strategy"]
    assert daily["status"] == "archived" and daily["version"] == 3
    assert daily["note"] == "imported from SAM v1 (created by automated tests)"
    card = json.loads(daily["card"])
    assert card["status"] == "archived" and card["origin"]["v1_id"] == "theory_daily3"
    assert card["timeframes"] == {"bias": "H1", "setup": "M15", "entry": "M5"}
    assert json.loads(card["source_text"])["name"] == "Daily Workflow Strategy"   # verbatim v1 definition
    kinds = [r["kind"] for r in card["rules"]]
    assert {"bias", "setup", "trigger", "entry", "stop", "target", "risk", "filter"} <= set(kinds)
    checks = {r["check"]["predicate"] for r in card["rules"] if r["check"]}
    assert checks == {"trend_is", "swept", "mss_or_bos", "rr_at_least"}
    assert all(r["text_ckb"] and r["text_en"] for r in card["rules"])
    assert all("v1_condition" in r for r in card["rules"] if r.get("check") and r["check"]["predicate"] != "rr_at_least")
    versions = app.db.query("SELECT version, card FROM strategy_card_versions WHERE card_id=? ORDER BY version",
                            ("v1-daily-workflow-strategy",))
    assert [v["version"] for v in versions] == [1, 2, 3]
    assert [json.loads(v["card"])["origin"]["v1_id"] for v in versions] == ["theory_daily1", "theory_daily2", "theory_daily3"]
    # Sorani summary in Arabic script with Kurdish letters, findable through the strategy FTS.
    assert "ئەرشیف" in daily["summary_ckb"] and "ي" not in daily["summary_ckb"] and "ك" not in daily["summary_ckb"]
    assert app.db.scalar("SELECT count(*) FROM strategy_fts WHERE strategy_fts MATCH ?", ('"workflow"',)) == 1


def test_no_imported_card_is_active_or_listed_as_draft(app):
    migrate_v1.run_migration(app)

    assert app.db.scalar("SELECT count(*) FROM strategy_cards WHERE status <> 'archived'") == 0


def test_trading_rows_go_to_the_archive_as_json(app):
    report = migrate_v1.run_migration(app)

    counts = {r["source_table"]: r["n"] for r in app.db.query(
        "SELECT source_table, count(*) AS n FROM v1_archive GROUP BY source_table")}
    for table, n in {"trading_setups": 1, "setup_events": 1, "trading_journal": 1, "trading_context": 3,
                     "chart_calibration": 1, "custom_theories": 4, "conversations": 1}.items():
        assert counts[table] == n, table
    assert "audit_log" not in counts and "audit_log" in report["not_imported"]
    setup = json.loads(app.db.scalar("SELECT payload FROM v1_archive WHERE source_table='trading_setups'"))
    assert setup["payload"] == {"direction": None, "entry": None}   # *_json columns stored as JSON
    journal = json.loads(app.db.scalar("SELECT payload FROM v1_archive WHERE source_table='trading_journal'"))
    assert journal["payload"]["note"] == "زێڕ لە ٢٧٠٠"


def test_messages_with_anything_key_shaped_are_never_imported(app):
    report = migrate_v1.run_migration(app)

    messages = app.db.query("SELECT source_id FROM v1_archive WHERE source_table='messages'")
    assert {m["source_id"] for m in messages} == {"msg_1", "msg_2"}
    assert report["skipped"]["messages_key_shaped"] == 1
    assert FAKE_GROQ not in _all_text(app.db)


def test_settings_archive_skips_credentials_and_masks_secrets(app):
    report = migrate_v1.run_migration(app)

    keys = {r["source_id"] for r in app.db.query("SELECT source_id FROM v1_archive WHERE source_table='settings'")}
    assert "credential_metadata" not in keys and report["skipped"]["settings_credential"] == 1
    assert {"voice_language", "default_model", "custom_endpoint"} <= keys
    endpoint = app.db.scalar("SELECT payload FROM v1_archive WHERE source_id='custom_endpoint'")
    assert "[REDACTED]" in endpoint and FAKE_GROQ not in endpoint
    assert report["skipped"]["rows_redacted"] >= 1


def test_sorani_voice_language_changes_nothing_and_paid_models_are_not_carried_over(app):
    report = migrate_v1.run_migration(app)

    assert "already SAM 2's default" in report["settings"]["voice_language"]
    assert app.db.scalar("SELECT count(*) FROM facts") == 0
    assert "anthropic" not in json.dumps(app.config.all())
    assert app.config.get("trading.default_symbol") == "XAUUSD"


def test_an_english_v1_user_gets_a_preference_fact(make_app, home):
    path = make_v1_db(home / "data" / "sam.sqlite3")
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("UPDATE settings SET value_json=? WHERE key='voice_language'", (json.dumps("en-US"),))
    app = make_app()
    app.load_packages(["sam.migrate_v1"])
    migrate_v1.run_migration(app)
    migrate_v1.run_migration(app, force=True)

    facts = app.db.query("SELECT text, kind, source FROM facts")
    assert facts == [{"text": "The user prefers to talk with SAM in English.", "kind": "preference", "source": "import"}]


def test_v1_memories_become_facts_but_key_shaped_ones_do_not(make_app, home):
    path = make_v1_db(home / "data" / "sam.sqlite3")
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("INSERT INTO memories VALUES ('m1', 'بەکارهێنەر زێڕ ترەید دەکات', '[]', 0.5, NULL, ?, ?, 'trading')",
                     ("2026-09-01T00:00:00+00:00", "2026-09-01T00:00:00+00:00"))
        conn.execute("INSERT INTO memories VALUES ('m2', ?, '[]', 0.5, NULL, ?, ?, 'user')",
                     (f"gemini key {FAKE_GEMINI}", "2026-09-01T00:00:00+00:00", "2026-09-01T00:00:00+00:00"))
    app = make_app()
    app.load_packages(["sam.migrate_v1"])
    report = migrate_v1.run_migration(app)

    assert report["facts"] == 1
    assert app.db.query("SELECT text, kind, source FROM facts") == [
        {"text": "بەکارهێنەر زێڕ ترەید دەکات", "kind": "trading", "source": "import"}]
    assert FAKE_GEMINI not in _all_text(app.db)


def test_v1_file_is_never_written_and_no_side_files_are_created(app, v1_db):
    before, sides_before = _digest(v1_db), _side_files(v1_db)
    assert sides_before == set()          # the helper closed its connection: WAL checkpointed away
    migrate_v1.run_migration(app)
    migrate_v1.run_migration(app, force=True)

    assert _digest(v1_db) == before
    assert _side_files(v1_db) == set(), "reading must not create -wal/-shm next to v1's file"
    conn = migrate_v1.open_v1_readonly(v1_db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE sam2_probe (x)")
    finally:
        conn.close()


def test_rows_still_in_v1s_wal_are_read_while_v1_holds_the_file(app, v1_db):
    writer = sqlite3.connect(v1_db)       # v1 mid-session: its commit lives in the WAL
    try:
        writer.execute("PRAGMA wal_autocheckpoint=0")
        with writer:
            writer.execute("INSERT INTO trading_setups VALUES ('setup_live', 'XAUUSD', NULL, 'snr', 'WATCH', '{}', 0, ?, ?)",
                           ("2026-09-24T07:00:00+00:00", "2026-09-24T07:00:00+00:00"))
        assert (v1_db.parent / (v1_db.name + "-wal")).exists()
        data = migrate_v1.read_v1(v1_db)
    finally:
        writer.close()
    assert {r["id"] for r in data["trading_setups"]} == {"setup_a", "setup_live"}


def test_running_twice_adds_nothing_and_keeps_user_changes(app):
    first = migrate_v1.run_migration(app)
    app.db.execute("UPDATE strategy_cards SET status='active', note='mine now' WHERE id='v1-ui-acceptance-strategy'")
    second = migrate_v1.run_migration(app)
    third = migrate_v1.run_migration(app, force=True)

    assert first["strategy_cards"] == 2
    assert second == {"ok": True, "skipped": "already imported", "source": second["source"]}
    assert third["strategy_cards"] == 0 and third["card_versions"] == 0 and sum(third["archived"].values()) == 0
    row = app.db.query_one("SELECT status, note FROM strategy_cards WHERE id='v1-ui-acceptance-strategy'")
    assert row == {"status": "active", "note": "mine now"}
    assert app.db.scalar("SELECT count(*) FROM v1_archive") == sum(first["archived"].values())
    assert app.config.get("migrate.v1_done") is True


def test_a_changed_v1_file_is_imported_again_incrementally(app, v1_db):
    import os

    migrate_v1.run_migration(app)
    assert not migrate_v1.needs_run(app)
    with closing(sqlite3.connect(v1_db)) as conn, conn:
        conn.execute("INSERT INTO trading_setups VALUES ('setup_b', 'XAUUSD', NULL, 'liquidity', 'WATCH', '{}', 0, ?, ?)",
                     ("2026-09-24T08:00:00+00:00", "2026-09-24T08:00:00+00:00"))
    stamp = migrate_v1.source_mtime(v1_db) + 10
    os.utime(v1_db, (stamp, stamp))

    assert migrate_v1.needs_run(app)
    report = migrate_v1.run_migration(app)
    assert report["archived"]["trading_setups"] == 1 and report["strategy_cards"] == 0


def test_report_and_activity_contain_no_key(app):
    report = migrate_v1.run_migration(app)

    stored = app.config.get("migrate.v1_report")
    assert stored["strategy_cards"] == report["strategy_cards"]
    activity = app.db.query_one("SELECT summary, detail, ok FROM activity WHERE name='migrate_v1'")
    assert activity["ok"] == 1 and "strategy cards" in activity["summary"]
    assert FAKE_GROQ not in json.dumps(stored) + json.dumps(activity)


async def test_start_runs_in_the_background_once(app):
    app.loop = asyncio.get_running_loop()
    await migrate_v1.start(app)
    tasks = [t for t in app._tasks if t.get_name() == "sam:migrate-v1"]
    assert len(tasks) == 1
    await asyncio.gather(*tasks)
    assert app.config.get("migrate.v1_done") is True
    await migrate_v1.start(app)
    assert not [t for t in app._tasks if t.get_name() == "sam:migrate-v1"]


async def test_without_a_v1_database_nothing_happens(make_app):
    app = make_app()
    app.load_packages(["sam.migrate_v1"])
    await migrate_v1.start(app)

    assert not app._tasks
    assert app.config.get("migrate.v1_done") is False
    assert migrate_v1.run_migration(app)["skipped"] == "no v1 database"


def test_a_broken_v1_file_is_logged_not_raised(make_app, home):
    (home / "data" / "sam.sqlite3").write_bytes(b"not a database at all" * 100)
    app = make_app()
    app.load_packages(["sam.migrate_v1"])
    result = migrate_v1._run_logged(app)

    assert result["ok"] is False
    assert app.db.scalar("SELECT count(*) FROM activity WHERE name='migrate_v1' AND ok=0") == 1
    assert app.config.get("migrate.v1_done") is False


def test_the_package_registers_through_the_app(make_app, v1_db):
    app = make_app()
    status = app.load_packages(["sam.migrate_v1"])

    assert status == {"sam.migrate_v1": "ok"}
    assert app.config.get("migrate.v1_report") is None
