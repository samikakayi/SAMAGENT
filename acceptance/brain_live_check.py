"""Brain live check (run by hand on this PC; not part of pytest).

    .venv\\Scripts\\python.exe acceptance\\brain_live_check.py --pace 30       # 3 turns on the default ladders
    .venv\\Scripts\\python.exe acceptance\\brain_live_check.py --probe groq:openai/gpt-oss-20b,omniroute:sam-fast
    .venv\\Scripts\\python.exe acceptance\\brain_live_check.py --measure-tokens

What it does:
- Builds a SAM 2 App on SAM_HOME (default: the v1 folder, so SAM 2's own
  Secrets/Config read the real key store and .env at runtime; no key is ever
  printed) and registers the brain modules.
- Replaces EVERY tool with a DRY-RUN copy (same name, description and JSON
  schema; real schemas from modules that already define them, else the
  docs/CONTRACTS.md section 2 catalogue). Nothing is executed: open_app,
  tv_open, remember... only return a canned result.
- Runs real turns through Conversation.respond_stream (source "cascade"):
  "سڵاو، چۆنی؟" must be answered briefly with no self-introduction and no tool;
  "ترەیدینگ ڤیو بکەرەوە" must produce a tv_open/open_app call.
- Prints JSON with latency to the first model token and first chunk.

Side effects on the real SAM 2 DB (data/sam2.sqlite3): real usage counters
(these calls really use free quota). The test conversation and its turns are
deleted at the end; dry-run tools write no activity rows; timings are kept in
memory only.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam.app import App  # noqa: E402
from sam.brain.tools import ToolContext, ToolSpec, ok  # noqa: E402
from sam.events import Error, ToolStarted  # noqa: E402
from sam.textnorm import is_arabic_script, normalize_ckb  # noqa: E402
from sam.timing import Timing  # noqa: E402

DEFAULT_HOME = r"C:\Users\samit\Desktop\SAM-Agent"
DRAWING_KINDS = ["horizontal_line", "horizontal_ray", "trend_line", "rectangle", "fib_retracement", "text",
                 "arrow_up", "arrow_down", "long_position", "short_position"]


def _obj(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


S = {"type": "string"}
CATALOGUE: dict[str, tuple[str, dict[str, Any], tuple[str, ...]]] = {
    "open_app": ("Open or focus a Windows app by its English or Sorani name.",
                 _obj({"name": {"type": "string", "description": "App name, English or Sorani (e.g. ترەیدینگ ڤیو)"},
                       "args": S}, ["name"]), ("کرۆم بکەرەوە", "ترەیدینگ ڤیو بکەرەوە")),
    "window_control": ("List, focus, minimize, maximize, restore, close or snap windows.",
                       _obj({"action": {"type": "string", "enum": ["list", "focus", "minimize", "maximize", "restore",
                                                                   "close", "snap_left", "snap_right"]},
                             "target": S}, ["action"]), ("تێلێگرام بچووک بکەرەوە",)),
    "type_text": ("Type text into the focused control or a numbered control from the last screen_look.",
                  _obj({"text": S, "target": S, "press_enter": {"type": "boolean"}}, ["text"]), ("بنووسە سڵاو",)),
    "press_keys": ("Press a keyboard shortcut or media key (ctrl+s, alt+tab, volume_up, play_pause).",
                   _obj({"keys": S, "repeat": {"type": "integer"}}, ["keys"]), ("دەنگەکە بەرز بکەرەوە",)),
    "click": ("Click a numbered control from the last screen_look, a control name or visible text.",
              _obj({"target": S, "button": {"type": "string", "enum": ["left", "right", "double"]}, "window": S},
                   ["target"]), ("کلیک لە دوگمەی سەیڤ بکە",)),
    "screen_look": ("Look at a window: numbered controls, its text, or a short description.",
                    _obj({"window": S, "mode": {"type": "string", "enum": ["controls", "text", "describe"]},
                          "query": S}), ("سەیری شاشەکە بکە",)),
    "screen_act": ("Reach a goal on screen step by step with vision when controls are not enough.",
                   _obj({"goal": S, "window": S, "max_steps": {"type": "integer"}}, ["goal"]), ()),
    "run_powershell": ("Run a PowerShell command (read-only runs at once, others ask the user).",
                       _obj({"command": S, "timeout_s": {"type": "integer"}}, ["command"]), ()),
    "files": ("List, read, write, copy, move, delete, open, search or reveal files and folders.",
              _obj({"action": {"type": "string", "enum": ["list", "read", "write", "append", "copy", "move", "delete",
                                                          "open", "search", "reveal"]},
                    "path": S, "content": S, "dest": S, "pattern": S}, ["action", "path"]), ()),
    "open_url": ("Open an http/https address in the default browser.", _obj({"url": S}, ["url"]), ()),
    "web_search": ("Search the web and return the top results.",
                   _obj({"query": S, "open_in_browser": {"type": "boolean"}}, ["query"]), ("نرخی زێڕ لە ئینتەرنێت بگەڕێ",)),
    "build_project": ("Build a small website or program in a new folder, open it in VS Code and preview it.",
                      _obj({"description": S, "name": S, "kind": {"type": "string", "enum": ["website", "python", "other"]}},
                           ["description"]), ("ماڵپەڕێک بۆ دوکانەکەم دروست بکە",)),
    "tv_open": ("Open TradingView Desktop (with SAM's local chart connection) or bring it to the front.",
                _obj({}), ("ترەیدینگ ڤیو بکەرەوە",)),
    "tv_set_chart": ("Change the TradingView chart's symbol and/or timeframe (any alias, Sorani words ok).",
                     _obj({"symbol": S, "timeframe": S}), ("گۆڵد لەسەر ١٥ خولەک پیشان بدە",)),
    "chart_state": ("Read the TradingView chart: symbol, timeframe, last bar, visible range, drawings.", _obj({}), ()),
    "draw_on_chart": ("Draw lines, zones, fibs, arrows or positions on the TradingView chart.",
                      _obj({"items": {"type": "array", "items": _obj({
                          "kind": {"type": "string", "enum": DRAWING_KINDS},
                          "points": {"type": "array", "items": _obj({"price": {"type": "number"},
                                                                     "time": {"type": "integer"},
                                                                     "bars_ago": {"type": "integer"}}, ["price"])},
                          "text": S, "color": S}, ["kind", "points"])}, "tag": S}, ["items"]),
                      ("هێڵی پشتگیری و بەرگری بکێشە",)),
    "clear_my_drawings": ("Remove SAM's own drawings from the chart (never the user's).", _obj({"tag": S}),
                          ("هێڵەکانت بسڕەوە",)),
    "get_price": ("Current bid/ask/spread of a symbol from MetaTrader 5.", _obj({"symbol": S}), ("نرخی زێڕ چەندە؟",)),
    "analyze_market": ("Analyse a market with the engine and the user's strategy, then draw the plan.",
                       _obj({"symbol": S, "timeframes": {"type": "array", "items": S}, "strategy_id": S,
                             "draw": {"type": "string", "enum": ["none", "levels", "full"]},
                             "vision": {"type": "boolean"}}), ("زێڕ شی بکەرەوە بە ستراتیژییەکەم",)),
    "set_alert": ("Create an alert SAM watches and speaks (price cross, zone, candle close, volume, strategy).",
                  _obj({"kind": {"type": "string", "enum": ["price_cross", "zone_touch", "candle_close", "volume_spike",
                                                            "strategy_state"]},
                        "symbol": S, "level": {"type": "number"}, "low": {"type": "number"},
                        "high": {"type": "number"}, "direction": {"type": "string", "enum": ["up", "down", "any"]},
                        "timeframe": S, "k": {"type": "number"}, "n": {"type": "integer"}, "strategy_id": S,
                        "repeat": {"type": "boolean"}, "note": S}, ["kind"]),
                  ("ئەگەر زێڕ گەیشتە ٢٧٠٠ ئاگادارم بکەرەوە",)),
    "list_alerts": ("List alerts.", _obj({"status": {"type": "string", "enum": ["active", "fired", "all"]}}), ()),
    "cancel_alert": ("Cancel an alert by id, or all.", _obj({"alert_id": S}, ["alert_id"]), ()),
    "strategy_save": ("Save or update a trading strategy card from the user's words.",
                      _obj({"text": S, "strategy_id": S, "status": {"type": "string", "enum": ["draft", "active"]}},
                           ["text"]), ()),
    "strategy_list": ("List the user's strategy cards.",
                      _obj({"status": {"type": "string", "enum": ["draft", "active", "archived", "all"]}}), ()),
    "strategy_get": ("Read one strategy card with all its rules.", _obj({"strategy_id": S}, ["strategy_id"]), ()),
}
REAL_TOOL_MODULES = ("sam.hands.tools", "sam.trading.chart_tools", "sam.trading.tools")

CANNED = {
    "open_app": ("TradingView is open and focused.", {}),
    "tv_open": ("TradingView is open and connected.", {}),
    "get_price": ("XAUUSD bid 2651.42, ask 2651.71, spread 29 points.",
                  {"symbol": "XAUUSD", "bid": 2651.42, "ask": 2651.71, "spread_points": 29}),
}
TURNS = ("سڵاو، چۆنی؟", "ترەیدینگ ڤیو بکەرەوە", "نرخی زێڕ ئێستا چەندە؟")


def install_dry_run_tools(app: App) -> tuple[list[str], list[str]]:
    """Every tool becomes a dry-run copy. Returns (real_schema_names, contract_names)."""
    specs: dict[str, ToolSpec] = {s.name: s for s in app.tools.specs()}
    real: list[str] = []
    for module_name in REAL_TOOL_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001 - other builders may not be done yet
            continue
        for item in vars(module).values():
            spec = getattr(item, "tool_spec", None)
            if isinstance(spec, ToolSpec) and spec.name not in specs:
                specs[spec.name] = spec
                real.append(spec.name)
    contract: list[str] = []
    for name, (description, params, examples) in CATALOGUE.items():
        if name not in specs:
            specs[name] = ToolSpec(name=name, description=description, handler=None, params=params,  # type: ignore[arg-type]
                                   examples_ckb=examples)
            contract.append(name)
    for name in list(app.tools.names()):
        app.tools.remove(name)
    for spec in specs.values():
        async def dry_run(ctx: ToolContext, _name: str = spec.name, **kwargs: Any) -> dict[str, Any]:
            summary, data = CANNED.get(_name, ("Done.", {}))
            return ok(summary, dry_run=True, **data)
        app.tools.add(ToolSpec(**{**spec.__dict__, "handler": dry_run, "risk": "safe", "classify": None}),
                      owner=spec.owner or "dry-run")
    # Dry runs leave no activity/timing rows in the real DB.
    app.tools.db = None
    app.tools.timing = None
    return real, contract


def sentences(text: str) -> int:
    return len([p for p in re.split(r"[.!?؟۔\n]+", text) if p.strip()])


async def run_turn(app: App, text: str, ladder: str | None) -> dict[str, Any]:
    if ladder:
        app.conversation._ladder = lambda mode: ladder  # type: ignore[method-assign]
    calls: list[dict[str, Any]] = []
    errors: list[str] = []
    unsubscribe = app.bus.subscribe(ToolStarted, lambda e: calls.append({"name": e.name, "args": e.args}))
    unsubscribe_errors = app.bus.subscribe(Error, lambda e: errors.append(e.detail))
    turn = Timing(None).turn("cascade")
    started = time.perf_counter()
    first_chunk = None
    chunks: list[str] = []
    try:
        async for piece in app.conversation.respond_stream(text, source="cascade", turn=turn):
            if first_chunk is None:
                first_chunk = (time.perf_counter() - started) * 1000
            chunks.append(piece)
    finally:
        unsubscribe()
        unsubscribe_errors()
    total = (time.perf_counter() - started) * 1000
    stages = [(name, round(ms), extra) for name, ms, extra in turn.stages]
    firsts = [s for s in stages if s[0] == "llm_first_token"]
    models = [s[2].get("model") for s in stages if s[0] == "llm_total" and s[2]]
    reply = " ".join(chunks)
    return {"user": text, "reply": reply, "chunks": chunks, "tool_calls": calls, "models": models,
            "llm_first_token_ms": [s[1] for s in firsts], "first_chunk_ms": round(first_chunk or 0),
            "total_ms": round(total), "sentences": sentences(reply), "arabic_script": is_arabic_script(reply),
            "errors": errors}


def check(result_a: dict[str, Any], result_b: dict[str, Any], result_c: dict[str, Any] | None = None) -> dict[str, bool]:
    intro = re.compile(r"من سام|ناوم سام|سامم|یاریدەدەری|I am SAM|I'm SAM", re.I)
    names = {c["name"] for c in result_b["tool_calls"]}
    open_ok = "tv_open" in names or any(
        c["name"] == "open_app" and re.search(r"trading|ترەیدینگ|تریدینگ", json.dumps(c["args"], ensure_ascii=False), re.I)
        for c in result_b["tool_calls"])
    return {
        "greeting_brief": 0 < result_a["sentences"] <= 3,
        "greeting_sorani": result_a["arabic_script"],
        "greeting_no_intro": not intro.search(result_a["reply"]),
        "greeting_no_tool": not result_a["tool_calls"],
        "tradingview_tool_called": open_ok,
        "tradingview_reply_sorani": result_b["arabic_script"],
    } | ({} if result_c is None else {
        "price_tool_called": any(c["name"] == "get_price" for c in result_c["tool_calls"]),
        "price_in_reply": bool(re.search(r"265[012]|شەشسەد و پەنجا|شەش سەد و پەنجا",
                                         normalize_ckb(result_c["reply"]))),
        "price_reply_sorani": result_c["arabic_script"],
    })


async def measure_tokens(app: App) -> dict[str, Any]:
    """Two non-streamed calls on Groq gpt-oss-20b: persona only, then persona
    + every tool schema, to calibrate persona.estimate_tokens."""
    from sam.brain.persona import estimate_tokens

    system = app.persona.system_instruction("voice")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": "سڵاو"}]
    bare = await app.llm.chat(messages, ladder="groq:openai/gpt-oss-20b", reasoning="low")
    tools = app.tools.openai_tools()
    full = await app.llm.chat(messages, ladder="groq:openai/gpt-oss-20b", tools=tools, reasoning="low")
    return {"persona_chars": len(system), "persona_estimate": estimate_tokens(system),
            "prompt_tokens_persona": bare.usage.get("tokens_in"), "prompt_tokens_with_tools": full.usage.get("tokens_in"),
            "tools": len(tools), "tools_json_chars": len(json.dumps(tools, ensure_ascii=False))}


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", default=DEFAULT_HOME)
    parser.add_argument("--probe", default="", help="comma-separated model refs to compare")
    parser.add_argument("--measure-tokens", action="store_true")
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--pace", type=float, default=0.0, help="seconds between turn rounds")
    args = parser.parse_args()

    app = App(args.home)
    for name in ("memory", "persona", "worker", "conversation"):
        importlib.import_module(f"sam.brain.{name}").register(app)
    real, contract = install_dry_run_tools(app)
    report: dict[str, Any] = {"tools": len(app.tools.names()), "real_schemas": real, "contract_schemas": contract}
    created: list[int] = []
    try:
        if args.measure_tokens:
            report["tokens"] = await measure_tokens(app)
        refs = [r.strip() for r in args.probe.split(",") if r.strip()] or [None]
        conversations: dict[Any, int] = {}
        results: dict[Any, list[dict[str, Any]]] = {ref: [] for ref in refs}
        # Turn-major order with a pause between rounds: Groq's free tier allows
        # ~8k tokens per minute per model and one request here is ~4k tokens.
        for index, text in enumerate(TURNS[: args.turns]):
            if index and args.pace:
                await asyncio.sleep(args.pace)
            for ref in refs:
                if ref not in conversations:
                    app.conversation.conversation_id = None
                    conversations[ref] = app.conversation.new_conversation("voice")
                    created.append(conversations[ref])
                app.conversation.conversation_id = conversations[ref]
                results[ref].append(await run_turn(app, text, ref))
        report["runs"] = [{"ladder": ref or app.conversation._ladder("voice"), "turns": turns,
                           "checks": check(*turns[:3]) if len(turns) >= 2 else {}}
                          for ref, turns in results.items()]
    finally:
        for cid in created:
            app.db.execute("DELETE FROM turns WHERE conversation_id=?", (cid,))
            app.db.execute("DELETE FROM brain_conversation_state WHERE conversation_id=?", (cid,))
            app.db.execute("DELETE FROM conversations WHERE id=?", (cid,))
        await app.llm.aclose()
        app.close()
    print(json.dumps(app.redact_obj(report), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
