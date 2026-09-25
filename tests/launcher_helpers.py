"""Helpers for the launcher/migration tests: load ``SAM.pyw`` as a module and
build a synthetic database with SAM v1's schema (DDL copied from v1's
``data/sam.sqlite3`` on 2026-09-24; rows are invented -- the real v1 file is
never copied into tests)."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sqlite3
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = ROOT / "SAM.pyw"

V1_SCHEMA = """
CREATE TABLE custom_theories (id TEXT PRIMARY KEY, name TEXT NOT NULL, version INTEGER NOT NULL,
    definition_json TEXT NOT NULL, archived INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL, UNIQUE(name, version));
CREATE TABLE trading_setups (id TEXT PRIMARY KEY, symbol TEXT NOT NULL, feed TEXT, theory TEXT NOT NULL,
    state TEXT NOT NULL, payload_json TEXT NOT NULL DEFAULT '{}', monitor_enabled INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE setup_events (id TEXT PRIMARY KEY, setup_id TEXT NOT NULL REFERENCES trading_setups(id) ON DELETE CASCADE,
    previous_state TEXT, state TEXT NOT NULL, reason TEXT NOT NULL, payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL);
CREATE TABLE trading_journal (id TEXT PRIMARY KEY, setup_id TEXT, symbol TEXT NOT NULL, theory TEXT NOT NULL,
    payload_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE trading_context (key TEXT PRIMARY KEY, value_json TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE chart_calibration (id TEXT PRIMARY KEY, window_handle INTEGER NOT NULL, symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL, geometry_hash TEXT NOT NULL, slope REAL NOT NULL, intercept REAL NOT NULL,
    method TEXT NOT NULL, anchors_json TEXT NOT NULL DEFAULT '[]', verified INTEGER NOT NULL DEFAULT 0,
    max_error REAL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    UNIQUE(window_handle, symbol, timeframe, geometry_hash));
CREATE TABLE drawing_ownership (id TEXT PRIMARY KEY, setup_id TEXT, symbol TEXT NOT NULL, timeframe TEXT,
    layer TEXT NOT NULL, drawing_type TEXT NOT NULL, payload_json TEXT NOT NULL DEFAULT '{}',
    visible INTEGER NOT NULL DEFAULT 1, verified INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL);
CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT NOT NULL, tags_json TEXT NOT NULL DEFAULT '[]',
    importance REAL NOT NULL DEFAULT 0.5, source_conversation_id TEXT, created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL, domain TEXT NOT NULL DEFAULT 'user');
CREATE TABLE conversations (id TEXT PRIMARY KEY, title TEXT NOT NULL, provider TEXT NOT NULL DEFAULT 'ollama',
    model TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE messages (id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL, content TEXT NOT NULL DEFAULT '', tool_name TEXT, tool_call_id TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL);
CREATE TABLE settings (key TEXT PRIMARY KEY, value_json TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE audit_log (id TEXT PRIMARY KEY, event_type TEXT NOT NULL, actor TEXT NOT NULL, conversation_id TEXT,
    tool_name TEXT, risk_level TEXT, status TEXT NOT NULL, summary TEXT NOT NULL, details_json TEXT NOT NULL DEFAULT '{}',
    previous_hash TEXT NOT NULL DEFAULT '', entry_hash TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL);
"""

TS = "2026-08-29T19:54:49.356255+00:00"


def definition(name: str, *, rr: float = 2.0, direction: str = "LONG") -> dict[str, Any]:
    """A v1 ``composed_strategy`` definition shaped like the ones on this PC."""
    return {
        "name": name, "description": "1H bullish structure + 15m Support RBS", "kind": "composed_strategy",
        "context": "1H bullish structure", "setup": "15m Support RBS",
        "confirmation": "5m liquidity sweep and bullish MSS", "entry_trigger": "bullish_sweep",
        "entry": "bullish_sweep", "invalidation": "Below the sweep low", "stop": "",
        "targets": ["TP1 liquidity", "TP2 resistance"], "timeframes": ["H1", "M15", "M5"],
        "minimum_rr": rr, "direction": direction,
        "conditions": [{"predicate": "trend_is", "value": "BULLISH", "timeframe": "H1"},
                       {"predicate": "has_liquidity_sweep", "timeframe": "M15"},
                       {"predicate": "has_mss", "timeframe": "M5"},
                       {"predicate": "rsi_below", "value": 70, "timeframe": "M15"}],
        "schema_version": 1,
    }


def make_v1_db(path: Path, *, secret: str = "", wal: bool = True, extra_messages: int = 0) -> Path:
    """Write a v1-shaped database. ``secret`` (a FAKE key) is planted in one
    message and one setting value so tests can prove it never reaches SAM 2."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    if wal:
        conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(V1_SCHEMA)
    rows: list[tuple[str, tuple[Any, ...]]] = []
    for version in (1, 2, 3):
        rows.append(("custom_theories", (f"theory_daily{version}", "Daily Workflow Strategy", version,
                                         json.dumps(definition("Daily Workflow Strategy", rr=1.5 + version)), 0,
                                         f"2026-08-29T2{version}:00:00+00:00", f"2026-08-29T2{version}:00:00+00:00")))
    rows.append(("custom_theories", ("theory_ui1", "UI Acceptance Strategy", 1,
                                     json.dumps(definition("UI Acceptance Strategy", direction="BOTH")), 0, TS, TS)))
    rows.append(("trading_setups", ("setup_a", "XAUUSD", "Demo feed", "snr", "WATCH",
                                    json.dumps({"direction": None, "entry": None}), 0, TS, TS)))
    rows.append(("setup_events", ("event_a", "setup_a", None, "WATCH", "created", "{}", TS)))
    rows.append(("trading_journal", ("journal_a", "setup_a", "XAUUSD", "snr", json.dumps({"note": "زێڕ لە ٢٧٠٠"}), TS, TS)))
    for key, value in (("symbol", "XAUUSD"), ("timeframes", ["H1", "M15"]), ("setup_state", "WATCH")):
        rows.append(("trading_context", (key, json.dumps(value), TS)))
    rows.append(("chart_calibration", ("cal_a", 1234, "XAUUSD", "UNKNOWN", "f4b892fbd94273c5", -0.03, 4496.2,
                                       "windows_ocr_price_scale", json.dumps([{"y": 240.5, "price": 4488.0}]), 0, None, TS, TS)))
    rows.append(("conversations", ("conv_a", "Hello", "openrouter", "free-model", TS, TS)))
    rows.append(("messages", ("msg_1", "conv_a", "user", "سڵاو سام", None, None, "{}", TS)))
    rows.append(("messages", ("msg_2", "conv_a", "assistant", "سڵاو، چۆن یارمەتیت بدەم؟", None, None, "{}", TS)))
    if secret:
        rows.append(("messages", ("msg_key", "conv_a", "user", f"my key is {secret} keep it", None, None, "{}", TS)))
        rows.append(("settings", ("custom_endpoint", json.dumps({"url": "http://x", "auth": secret}), TS)))
    for i in range(extra_messages):
        rows.append(("messages", (f"msg_x{i}", "conv_a", "user", f"message {i}", None, None, "{}", TS)))
    for key, value in (("voice_language", "ckb-IQ"), ("default_model", "anthropic/claude-sonnet-4.5"),
                       ("credential_metadata", {"n8n_api_key": {"scopes": ["workflow:read"]}}),
                       ("hands_free_continuation_seconds", 10.0)):
        rows.append(("settings", (key, json.dumps(value), TS)))
    rows.append(("audit_log", ("audit_a", "tool", "sam", None, None, None, "ok", "ran", "{}", "", "", TS)))
    with conn:
        for table, values in rows:
            conn.execute(f"INSERT INTO {table} VALUES ({', '.join('?' for _ in values)})", values)
    conn.close()
    return path


def load_launcher(name: str = "sam_launcher_under_test") -> ModuleType:
    """Import SAM.pyw (no .py suffix) as a fresh module."""
    loader = importlib.machinery.SourceFileLoader(name, str(LAUNCHER))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


__all__ = ["make_v1_db", "load_launcher", "definition", "V1_SCHEMA", "ROOT", "LAUNCHER"]
