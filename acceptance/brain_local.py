"""Local brain + no-AI fast path, live on this PC (run by hand; not pytest).

    .venv\\Scripts\\python.exe acceptance\\brain_local.py            # SAM starts its own ollama on 11436
    .venv\\Scripts\\python.exe acceptance\\brain_local.py --host 127.0.0.1:11434 --keep-server

What it proves, with SAM 2's own code (no cloud request at all -- the App runs
on a temp home without keys, so every cloud rung is unconfigured):

1. ``OllamaServer`` finds ollama.exe / the model folder (settings or SAM_HOME)
   and starts ``ollama serve`` hidden on the chosen port;
2. the conversation's warm-up loads the model and reads SAM's stable prompt
   (persona + compact core tools) into Ollama's cache;
3. real Sorani turns through ``Conversation.respond_stream`` are answered by
   the local brain (the LAST rung) -- tools are DRY-RUN: every dispatch only
   records the call and returns a canned Sorani result, nothing is opened;
4. the same commands through the fast path take milliseconds and no model;
5. ``LLMClient.aclose`` stops the server SAM started (the port closes).

Paths default to the v1 folder (``--sam-agent``) because that is where the
portable Ollama and the models are on this PC; nothing there is written.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COMMANDS = ["ترەیدینگ ڤیو بکەرەوە", "نرخی زێڕ چەندە", "شیکاری زێڕ بکە و ئاستەکان بکێشە", "نۆتپاد بکەرەوە",
            "گۆڵد لەسەر ١٥ خولەک پیشان بدە", "ئاگادارکردنەوەکانم پیشان بدە"]
CHAT = ["سڵاو سام، چۆنی؟", "باشترین کات بۆ ترەیدی زێڕ کەیە؟"]
CANNED = {
    "get_price": "زێڕ ئێستا لەسەر ٤٣١٢ مامەڵە دەکرێت.",
    "tv_open": "ترەیدینگ ڤیو ئامادەیە.",
    "open_app": "",
    "tv_set_chart": "چارتەکە گۆڕا بۆ زێڕ لەسەر پازدە خولەک.",
    "analyze_market": "زێڕ لە ڕەوتی سەرەوەدایە؛ نزیکترین پشتگیری ٤٢٩٠ و بەرگری ٤٣٣٠.",
    "list_alerts": "2 ئاگادارکردنەوە (active).",
}


def build_app(args: argparse.Namespace) -> Any:
    from sam.app import App

    home = ROOT / "work" / "localbrain-accept"
    shutil.rmtree(home, ignore_errors=True)
    (home / "data").mkdir(parents=True)
    app = App(home, environ={})
    app.load_packages(["sam.brain.memory", "sam.brain.persona", "sam.hands", "sam.trading.chart_tools",
                       "sam.trading.tools", "sam.brain.worker", "sam.brain.conversation"])
    tools_dir = Path(args.sam_agent) / "tools"
    exe = sorted(tools_dir.glob("ollama*/ollama.exe"), reverse=True)
    app.config.set("llm.local.host", args.host)
    if exe:
        app.config.set("llm.local.ollama_exe", str(exe[0]))
    app.config.set("llm.local.models_dir", str(Path(args.sam_agent) / "data" / "ollama-models"))
    if args.keep_server:
        app.config.set("llm.local.stop_on_quit", False)
    return app


def dry_run(app: Any, calls: list[tuple[str, dict[str, Any]]]) -> None:
    async def dispatch(name: str, args: Any, *, source: str = "text", call_id: Any = None) -> dict[str, Any]:
        calls.append((name, dict(args or {})))
        text = CANNED.get(name, "")
        if name == "list_alerts":
            return {"ok": True, "summary": text, "data": {"alerts": []}}
        return {"ok": True, "summary": text or f"{name} done (dry run).", "data": {"state": "started"}}

    app.tools.dispatch = dispatch  # type: ignore[method-assign]


async def turn(app: Any, text: str, source: str) -> dict[str, Any]:
    started = time.perf_counter()
    first: float | None = None
    parts: list[str] = []
    async for piece in app.conversation.respond_stream(text, source=source):
        if first is None:
            first = time.perf_counter() - started
        parts.append(str(piece))
    return {"input": text, "first_s": round(first or 0.0, 2), "total_s": round(time.perf_counter() - started, 2),
            "reply": " ".join(parts)[:240]}


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1:11436")
    parser.add_argument("--sam-agent", default=r"C:\Users\samit\Desktop\SAM-Agent")
    parser.add_argument("--keep-server", action="store_true")
    args = parser.parse_args()
    app = build_app(args)
    app.bus.bind_loop(asyncio.get_running_loop())
    calls: list[tuple[str, dict[str, Any]]] = []
    dry_run(app, calls)
    out: dict[str, Any] = {"host": args.host}
    local = app.llm.local_backend()
    out["server_before"] = local.server.status()
    ok = True
    try:
        began = time.perf_counter()
        task = app.conversation.prewarm_local_brain("voice")
        out["prewarm_started"] = task is not None
        if task is not None:
            out["prewarm_ok"] = await task
        out["prewarm_s"] = round(time.perf_counter() - began, 2)
        out["server_after_start"] = local.server.status()
        app.config.set("brain.fastpath.enabled", False)
        rows = []
        for text in COMMANDS + CHAT:
            before = len(calls)
            row = await turn(app, text, "cascade")
            row["tools"] = [c[0] for c in calls[before:]]
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        out["local_turns"] = rows
        out["brain_mode"] = app.llm.brain_mode
        app.config.set("brain.fastpath.enabled", True)
        fast = []
        for text in COMMANDS:
            before = len(calls)
            row = await turn(app, text, "text")
            row["tools"] = [c[0] for c in calls[before:]]
            fast.append(row)
        out["fastpath_turns"] = fast
        out["usage"] = app.db.query("SELECT provider, model, requests, errors FROM usage_counters")
    except Exception as exc:  # noqa: BLE001
        ok = False
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        await app.llm.aclose()
        await asyncio.sleep(1.0)
        out["server_after_quit"] = local.server.status()
        app.close()
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
