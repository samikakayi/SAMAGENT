"""Import SAM v1's data into SAM 2 -- read-only on the v1 database.

What the user asked to keep: v1's features and "the trading strategies and
theories that were written". The 40-theory knowledge catalogue is ported as
code by the trading package; this module brings over v1's *data* from
``<data_dir>/sam.sqlite3`` (v1 schema, tag ``v1-final``):

- ``custom_theories`` -> ``strategy_cards`` with status ``archived`` and the
  note "imported from SAM v1 (created by automated tests)". The 13 rows on
  this PC (2026-09-24) are 3 names with 1, 5 and 7 versions ("Workflow
  Strategy", "UI Acceptance Strategy", "Daily Workflow Strategy"), all made
  by v1's test runs, so nothing test-made becomes active. Rows with one name
  become ONE card whose history is kept in ``strategy_card_versions`` (id
  ``v1-<slug of the name>``; each version keeps its v1 id under ``origin``).
- ``trading_setups``, ``setup_events``, ``trading_journal``,
  ``trading_context``, ``chart_calibration``, ``drawing_ownership``,
  ``memories``, ``conversations``, ``messages``, ``settings`` and the raw
  ``custom_theories`` rows -> ``v1_archive`` (table name + JSON row). Nothing
  from v1 is lost, and nothing there is ever read back into a prompt.
- v1 settings: only what still has a meaning in SAM 2 is mapped (see
  :func:`_map_settings`); v1's paid model choices (anthropic/claude-sonnet-4.5
  through OpenRouter) are deliberately NOT carried over: SAM 2 is free-only.
- ``memories`` -> ``facts`` (source ``import``) as well; v1 had 0 rows here.

Safety: the v1 file is opened with ``mode=ro`` and ``PRAGMA query_only``; a
message with anything key-shaped (``sam.secrets.redact`` finds a match) is
never imported, every other payload is stored redacted. Not imported: v1's
operational logs (``agent_tasks``, ``approvals``, ``audit_log``,
``model_usage``): the v1 file itself stays on disk.

Runs once (setting ``migrate.v1_done``). While v1 still runs next to SAM 2
during the switch-over, v1 keeps writing its file, so a later start repeats
the import when the v1 file changed after the recorded import (insert-only
for cards, refresh for archive rows) -- idempotent by source ids.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .secrets import redact_count
from .textnorm import normalize_ckb

log = logging.getLogger("sam.migrate_v1")

CARD_NOTE = "imported from SAM v1 (created by automated tests)"
ARCHIVE_NOTE = "imported from SAM v1"

# v1 table -> its primary-key column, in import order.
ARCHIVE_TABLES: dict[str, str] = {
    "custom_theories": "id",
    "trading_setups": "id",
    "setup_events": "id",
    "trading_journal": "id",
    "trading_context": "key",
    "chart_calibration": "id",
    "drawing_ownership": "id",
    "memories": "id",
    "conversations": "id",
    "messages": "id",
    "settings": "key",
}
NOT_IMPORTED: tuple[str, ...] = ("agent_tasks", "approvals", "audit_log", "model_usage")
_CREDENTIAL_NAME = re.compile(r"(?i)(credential|api[_\-]?key|secret|token|password|passwd|cookie)")

# v1 custom-theory predicates (v1 analyst._execute_custom_theory) -> SAM 2
# predicate names (design 2.4). Checks only matter if a card is re-activated;
# the original v1 condition is always kept next to the mapped check.
_TREND = {"BULLISH": "up", "BEARISH": "down", "RANGE": "range", "RANGING": "range"}
_DIRECTION_CKB = {"LONG": "کڕین", "SHORT": "فرۆشتن", "BOTH": "هەردوو ئاڕاستە"}
_KIND_CKB = {
    "bias": "ئاڕاستەی سەرەکی", "setup": "ستاپ", "trigger": "پشتڕاستکردنەوە", "entry": "چوونەژوورەوە",
    "stop": "ستۆپ", "target": "ئامانج", "risk": "ڕیسک", "filter": "مەرج",
}

# ----------------------------------------------------------------------------------------------
# App hooks


def register(app: Any) -> None:
    """No tools; one extra setting holding the last import report (no keys)."""
    app.config.register_defaults({"migrate.v1_report": None})


async def start(app: Any) -> None:
    """Run the import in the background (it must not delay the island)."""
    if not needs_run(app):
        return
    app.spawn(asyncio.to_thread(_run_logged, app), "migrate-v1")


def _run_logged(app: Any) -> dict[str, Any]:
    try:
        return run_migration(app)
    except Exception as exc:  # noqa: BLE001 - never take SAM down over an import
        message = app.redact(f"{type(exc).__name__}: {exc}")
        log.error("v1 import failed: %s", message)
        try:
            app.db.log_activity("system", "migrate_v1", ok=False, summary=message[:300], source="startup")
        except Exception:  # noqa: BLE001
            pass
        return {"ok": False, "error": message}


def source_mtime(v1_path: Path) -> float:
    """mtime of the v1 main file. v1 opens short-lived connections, so its
    commits reach the main file at each checkpoint-on-close; our own reads
    never touch it (measured: unchanged across 3 reads while v1 ran). The WAL
    is not used here because opening a WAL database read-only creates an
    empty ``-wal`` (measured), which would re-trigger the import forever."""
    try:
        return v1_path.stat().st_mtime
    except OSError:
        return 0.0


def needs_run(app: Any, v1_path: Path | None = None) -> bool:
    """First start with a v1 file, or v1 changed its file since the import."""
    path = Path(v1_path or app.config.v1_db_path)
    if not path.is_file():
        return False
    if not app.config.get("migrate.v1_done", False):
        return True
    report = app.config.get("migrate.v1_report") or {}
    try:
        return source_mtime(path) > float(report.get("source_mtime") or 0.0) + 1.0
    except (OSError, TypeError, ValueError):
        return False


# ----------------------------------------------------------------------------------------------
# Reading v1 (read-only)


def open_v1_readonly(path: Path | str) -> sqlite3.Connection:
    """Open the v1 database so that it cannot be written through this handle.

    v1's file is in WAL mode. Measured on this PC (2026-09-24): a plain
    ``mode=ro`` open while v1 had no connection open CREATED ``sam.sqlite3-wal``
    (0 bytes) and ``-shm`` next to it -- files in v1's folder that SAM 2 must
    not create. So:
    - a ``-wal`` exists (v1 is mid-session): ``mode=ro`` uses v1's own WAL and
      shared memory, sees every commit and creates nothing new;
    - no ``-wal``: every commit is in the main file, and ``immutable=1`` reads
      it without creating any file. v1 writes only into a WAL (the mode is
      stored in the file), so the main file changes only at a checkpoint; a
      checkpoint racing our ~50 ms read would at worst fail this run, which
      is logged and retried at the next start (the import is idempotent).
    """
    source = Path(path).resolve()
    wal = source.with_name(source.name + "-wal")
    uri = source.as_uri() + ("?mode=ro" if wal.exists() else "?mode=ro&immutable=1")
    conn = sqlite3.connect(uri, uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = 1")
    return conn


def read_v1(path: Path | str, tables: Iterable[str] = ARCHIVE_TABLES) -> dict[str, list[dict[str, Any]]]:
    """Every row of the wanted tables that exist, from ONE read snapshot."""
    conn = open_v1_readonly(path)
    try:
        conn.execute("BEGIN")  # one consistent snapshot while v1 may be writing
        present = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        data: dict[str, list[dict[str, Any]]] = {}
        for table in tables:
            if table in present:
                data[table] = [dict(r) for r in conn.execute(f'SELECT * FROM "{table}"')]
        conn.execute("COMMIT")
        return data
    finally:
        conn.close()


def _parse_json_columns(row: dict[str, Any]) -> dict[str, Any]:
    """v1 keeps JSON in ``*_json`` text columns: store it as JSON, not text."""
    out: dict[str, Any] = {}
    for key, value in row.items():
        if key.endswith("_json") and isinstance(value, str):
            try:
                out[key[:-5]] = json.loads(value)
                continue
            except json.JSONDecodeError:
                pass
        out[key] = value
    return out


def _epoch(value: Any, default: float | None = None) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return default if default is not None else time.time()


# ----------------------------------------------------------------------------------------------
# custom_theories -> strategy cards


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:48] or "strategy"


def _rule(index: int, kind: str, text_en: str, check: dict[str, Any] | None = None,
          v1_condition: dict[str, Any] | None = None, text_ckb: str | None = None) -> dict[str, Any]:
    rule: dict[str, Any] = {
        "id": f"r{index}", "kind": kind, "text_en": text_en,
        "text_ckb": text_ckb or f"{_KIND_CKB.get(kind, 'مەرج')}: {text_en}", "check": check,
    }
    if v1_condition is not None:
        rule["v1_condition"] = v1_condition
    return rule


def _condition_rule(index: int, condition: dict[str, Any]) -> dict[str, Any]:
    """One v1 condition -> a rule with the closest SAM 2 predicate."""
    from .trading.common import normalize_timeframe

    predicate = str(condition.get("predicate", ""))
    tf = normalize_timeframe(str(condition.get("timeframe", ""))) or str(condition.get("timeframe", ""))
    value = condition.get("value")
    upper = str(value).upper() if value is not None else ""
    direction = {"BULLISH": "up", "BEARISH": "down"}.get(upper)
    if predicate == "trend_is":
        trend = _TREND.get(upper, upper.lower() or "any")
        sentence = {"up": f"ترێندی {tf} بەرەو سەرەوەیە", "down": f"ترێندی {tf} بەرەو خوارەوەیە",
                    "range": f"ترێندی {tf} لە مەودادایە"}.get(trend, f"ترێندی {tf}: {trend}")
        return _rule(index, "bias", f"{tf} trend is {trend}",
                     {"predicate": "trend_is", "params": {"tf": tf, "direction": trend}}, condition, sentence)
    if predicate == "has_liquidity_sweep":
        return _rule(index, "setup", f"liquidity sweep on {tf}",
                     {"predicate": "swept", "params": {"tf": tf, "side": direction or "any"}},
                     condition, f"لیکویدیتی لە {tf} سویپ کراوە")
    if predicate in ("has_mss", "has_bos"):
        kind = "mss" if predicate == "has_mss" else "bos"
        return _rule(index, "trigger", f"{kind.upper()} on {tf}",
                     {"predicate": "mss_or_bos", "params": {"tf": tf, "dir": direction or "any", "kind": kind}},
                     condition, f"{kind.upper()} لە {tf} ڕوویداوە")
    if predicate == "has_active_fvg":
        return _rule(index, "entry", f"price in an active FVG on {tf}",
                     {"predicate": "in_fvg", "params": {"tf": tf, "dir": direction or "any"}},
                     condition, f"نرخ لەناو FVGـی چالاکی {tf}ـدایە")
    if predicate in ("rsi_above", "rsi_below"):
        side = "above" if predicate == "rsi_above" else "below"
        word = "سەرووی" if side == "above" else "خوارووی"
        return _rule(index, "filter", f"RSI(14) on {tf} {side} {value}", None, condition, f"RSI لە {tf} {word} {value}ە")
    return _rule(index, "filter", f"v1 condition {predicate}", None, condition)


def card_from_v1(card_id: str, row: dict[str, Any]) -> dict[str, Any]:
    """Build a SAM 2 strategy card (contract 3.5 format) from one v1 row."""
    from .trading.common import normalize_timeframe

    try:
        definition = json.loads(row.get("definition_json") or "{}")
    except json.JSONDecodeError:
        definition = {}
    name = str(row.get("name") or definition.get("name") or "v1 strategy")
    tfs = [normalize_timeframe(str(t)) or str(t) for t in (definition.get("timeframes") or [])]
    rules: list[dict[str, Any]] = []
    for kind, key in (("bias", "context"), ("setup", "setup"), ("trigger", "confirmation"), ("entry", "entry"),
                      ("stop", "invalidation"), ("stop", "stop")):
        text = str(definition.get(key) or "").strip()
        if text:
            rules.append(_rule(len(rules) + 1, kind, text))
    for target in definition.get("targets") or []:
        if str(target).strip():
            rules.append(_rule(len(rules) + 1, "target", str(target).strip()))
    min_rr = definition.get("minimum_rr")
    if isinstance(min_rr, (int, float)):
        rules.append(_rule(len(rules) + 1, "risk", f"risk:reward at least {min_rr}",
                           {"predicate": "rr_at_least", "params": {"x": float(min_rr)}},
                           text_ckb=f"ڕێژەی قازانج بۆ زیان لانیکەم {min_rr} بێت"))
    for condition in definition.get("conditions") or []:
        if isinstance(condition, dict):
            rules.append(_condition_rule(len(rules) + 1, condition))
    direction = _DIRECTION_CKB.get(str(definition.get("direction", "")).upper(), "دیاری نەکراو")
    tf_text = "، ".join(tfs) or "دیاری نەکراو"
    summary = (f"لە SAM v1 هێنراوە و تاقیکردنەوە ئۆتۆماتیکییەکان دروستیان کردووە، بۆیە ئەرشیف کراوە. "
               f"ئاڕاستە: {direction}؛ تایمفرەیمەکان: {tf_text}"
               + (f"؛ کەمترین ڕێژەی قازانج بۆ زیان: {min_rr}." if isinstance(min_rr, (int, float)) else "."))
    return {
        "id": card_id,
        "title_ckb": f"{name} — لە SAM v1",
        "title_en": name,
        "markets": [],
        "timeframes": {"bias": tfs[0] if tfs else None, "setup": tfs[1] if len(tfs) > 1 else None,
                       "entry": tfs[2] if len(tfs) > 2 else (tfs[-1] if tfs else None)},
        "sessions": [],
        "rules": rules,
        "risk": {"max_risk_pct": None, "max_losses_per_day": None},
        "management": "",
        "source_text": json.dumps(definition, ensure_ascii=False, indent=1),
        "version": int(row.get("version") or 1),
        "status": "archived",
        "summary_ckb": summary,
        "note": CARD_NOTE,
        "origin": {"system": "sam-v1", "table": "custom_theories", "v1_id": row.get("id"),
                   "v1_version": row.get("version"), "v1_archived": bool(row.get("archived")),
                   "v1_created_at": row.get("created_at")},
    }


def group_theories(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """[(card_id, rows oldest->newest)], one card per v1 name, stable ids."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("name") or "v1 strategy"), []).append(row)
    used: set[str] = set()
    result: list[tuple[str, list[dict[str, Any]]]] = []
    for name in sorted(groups):
        base = "v1-" + _slug(name)
        card_id, n = base, 2
        while card_id in used:
            card_id, n = f"{base}-{n}", n + 1
        used.add(card_id)
        result.append((card_id, sorted(groups[name], key=lambda r: (int(r.get("version") or 0), str(r.get("created_at"))))))
    return result


def _search_text(card: dict[str, Any]) -> str:
    parts = [card["title_ckb"], card["title_en"], card["summary_ckb"]]
    parts += [f"{r.get('text_ckb', '')} {r.get('text_en', '')}" for r in card["rules"]]
    return normalize_ckb(" ".join(parts))


def _import_cards(app: Any, rows: list[dict[str, Any]]) -> tuple[int, int]:
    """Insert-only: an existing card (maybe changed by the user) is never overwritten."""
    new_cards = new_versions = 0
    db = app.db
    with db.transaction():
        for card_id, versions in group_theories(rows):
            latest = card_from_v1(card_id, versions[-1])
            created = _epoch(versions[0].get("created_at"))
            updated = _epoch(versions[-1].get("updated_at"), created)
            card = app.redact_obj(latest)
            cursor = db.execute(
                "INSERT OR IGNORE INTO strategy_cards(id, title_ckb, title_en, status, version, card, summary_ckb, "
                "source_text, search_text, note, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (card_id, card["title_ckb"], card["title_en"], "archived", card["version"],
                 json.dumps(card, ensure_ascii=False), card["summary_ckb"], card["source_text"],
                 _search_text(card), CARD_NOTE, created, updated))
            new_cards += max(cursor.rowcount, 0)
            for row in versions:
                version_card = app.redact_obj(card_from_v1(card_id, row))
                cursor = db.execute(
                    "INSERT OR IGNORE INTO strategy_card_versions(card_id, version, card, reason, created_at) "
                    "VALUES (?,?,?,?,?)",
                    (card_id, version_card["version"], json.dumps(version_card, ensure_ascii=False),
                     f"imported from SAM v1 version {row.get('version')} ({row.get('id')})",
                     _epoch(row.get("created_at"))))
                new_versions += max(cursor.rowcount, 0)
    return new_cards, new_versions


# ----------------------------------------------------------------------------------------------
# Archive + settings + memories


def _secret_count(app: Any, payload: Any) -> int:
    return redact_count(json.dumps(payload, ensure_ascii=False, default=str), app.secrets.known_values())[1]


def _archive(app: Any, data: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, int], dict[str, int]]:
    """Upsert every row into v1_archive; returns (new rows per table, skipped per reason)."""
    added: dict[str, int] = {}
    skipped: dict[str, int] = {"messages_key_shaped": 0, "settings_credential": 0, "rows_redacted": 0}
    now = time.time()
    db = app.db
    with db.transaction():
        for table, pk in ARCHIVE_TABLES.items():
            count = 0
            for row in data.get(table, []):
                source_id = str(row.get(pk))
                if table == "settings" and _CREDENTIAL_NAME.search(source_id):
                    skipped["settings_credential"] += 1
                    continue
                payload = _parse_json_columns(row)
                # redact_obj runs sam.secrets.redact on every string (all key shapes,
                # the stored key values) and masks secret-named fields: any change
                # means the row held something key-shaped.
                redacted = app.redact_obj(payload)
                secret_like = redacted != payload
                if secret_like and table == "messages":
                    skipped["messages_key_shaped"] += 1
                    continue
                note = ARCHIVE_NOTE
                if secret_like:
                    payload = redacted
                    skipped["rows_redacted"] += 1
                    note += " (secrets masked)"
                encoded = json.dumps(payload, ensure_ascii=False, default=str)
                existed = db.scalar("SELECT 1 FROM v1_archive WHERE source_table=? AND source_id=?", (table, source_id))
                db.execute(
                    "INSERT INTO v1_archive(source_table, source_id, payload, note, imported_at) VALUES (?,?,?,?,?) "
                    "ON CONFLICT(source_table, source_id) DO UPDATE SET payload=excluded.payload, note=excluded.note "
                    "WHERE v1_archive.payload <> excluded.payload",
                    (table, source_id, encoded, note, now))
                count += 0 if existed else 1
            if table in data:
                added[table] = count
    return added, skipped


_LANGUAGE_FACTS = {
    "en": ("The user prefers to talk with SAM in English.", "preference"),
    "ar": ("بەکارهێنەر حەز دەکات بە عەرەبی قسە بکات.", "preference"),
}


def _user_set(app: Any, key: str) -> bool:
    return app.db.scalar("SELECT 1 FROM settings WHERE key=?", (key,)) is not None


def _add_fact(app: Any, text: str, kind: str) -> bool:
    """Through Memory when the brain package is loaded, else straight into ``facts``."""
    if getattr(app, "memory", None) is not None:
        try:
            return bool(app.memory.remember(text, kind=kind, source="import").get("created"))
        except Exception as exc:  # noqa: BLE001
            log.warning("memory.remember failed during import: %s", app.redact(str(exc)))
    now = time.time()
    cursor = app.db.execute(
        "INSERT OR IGNORE INTO facts(text, text_norm, kind, tags, source, confidence, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?)", (text, normalize_ckb(text), kind, "v1", "import", 1.0, now, now))
    return cursor.rowcount > 0


def _map_settings(app: Any, data: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    """v1 settings that still mean something in SAM 2. Everything else is only
    archived: model/provider/budget keys (v1 pointed at a paid Anthropic model;
    SAM 2 is free-only), the wake word (SAM 2 has none: hotkey/click), and
    permission/computer-control switches (SAM 2 always confirms risky actions)."""
    result: dict[str, str] = {}
    settings = {}
    for row in data.get("settings", []):
        try:
            settings[str(row["key"])] = json.loads(row.get("value_json") or "null")
        except (json.JSONDecodeError, KeyError):
            continue
    language = settings.get("voice_language")
    if isinstance(language, str) and language:
        code = language.split("-")[0].lower()
        if code == "ckb":
            result["voice_language"] = f"{language}: Sorani, already SAM 2's default"
        else:
            if "voice.language" in app.config.defaults and not _user_set(app, "voice.language"):
                app.config.set("voice.language", code)
                result["voice_language"] = f"{language} -> voice.language={code}"
            fact = _LANGUAGE_FACTS.get(code)
            if fact and _add_fact(app, *fact):
                result["voice_language_fact"] = "added"
    context = {}
    for row in data.get("trading_context", []):
        try:
            context[str(row["key"])] = json.loads(row.get("value_json") or "null")
        except (json.JSONDecodeError, KeyError):
            continue
    symbol = context.get("symbol")
    if isinstance(symbol, str) and symbol:
        from .trading.common import canonical_symbol

        canonical = canonical_symbol(symbol)
        current = app.config.get("trading.default_symbol")
        if canonical == current:
            result["trading_context.symbol"] = f"{canonical}: already SAM 2's default"
        elif not _user_set(app, "trading.default_symbol"):
            app.config.set("trading.default_symbol", canonical)
            result["trading_context.symbol"] = f"-> trading.default_symbol={canonical}"
    return result


def _import_memories(app: Any, rows: list[dict[str, Any]]) -> int:
    added = 0
    for row in rows:
        text = str(row.get("content") or "").strip()
        if not text or _secret_count(app, text):
            continue
        kind = "trading" if str(row.get("domain", "")).lower() == "trading" else "fact"
        added += int(_add_fact(app, text, kind))
    return added


# ----------------------------------------------------------------------------------------------
# Entry point


def run_migration(app: Any, v1_path: Path | str | None = None, *, force: bool = False) -> dict[str, Any]:
    """Import v1 data (read-only on v1). Returns a JSON-able report (no keys).

    Safe to repeat: cards are insert-only, archive rows are keyed by (table,
    source id). ``force`` ignores the done-marker check.
    """
    began = time.perf_counter()
    path = Path(v1_path or app.config.v1_db_path)
    if not path.is_file():
        return {"ok": False, "skipped": "no v1 database", "source": str(path)}
    if not force and not needs_run(app, path):
        return {"ok": True, "skipped": "already imported", "source": str(path)}
    mtime = source_mtime(path)
    data = read_v1(path)
    cards, versions = _import_cards(app, data.get("custom_theories", []))
    archived, skipped = _archive(app, data)
    mapped = _map_settings(app, data)
    facts = _import_memories(app, data.get("memories", []))
    report: dict[str, Any] = {
        "ok": True, "source": str(path), "source_mtime": mtime, "at": time.time(),
        "read": {t: len(rows) for t, rows in data.items()},
        "strategy_cards": cards, "card_versions": versions, "archived": archived, "skipped": skipped,
        "settings": mapped, "facts": facts, "not_imported": list(NOT_IMPORTED),
        "ms": round((time.perf_counter() - began) * 1000.0, 1),
    }
    report = app.redact_obj(report)
    app.config.set("migrate.v1_report", report)
    app.config.set("migrate.v1_done", True)
    summary = (f"v1 import: {cards} new strategy cards ({versions} versions), "
               f"{sum(archived.values())} new archive rows, {skipped['messages_key_shaped']} key-shaped messages skipped")
    app.db.log_activity("system", "migrate_v1", ok=True, summary=summary, detail=report, duration_ms=report["ms"],
                        source="startup")
    log.info("%s in %.0f ms", summary, report["ms"])
    return report


__all__ = ["register", "start", "run_migration", "read_v1", "open_v1_readonly", "needs_run", "card_from_v1",
           "group_theories", "CARD_NOTE", "ARCHIVE_TABLES"]
