"""Final-stage live checks on this PC, with NO cloud request (run by hand).

    set PYTHONIOENCODING=utf-8
    .venv\\Scripts\\python.exe acceptance\\final_checks.py [--home C:\\Users\\samit\\Desktop\\SAM-Agent]

SAM 2 starts in THIS process the way ``sam.__main__`` does (island first, the
panel opened like a normal launch), then every cloud rung is forced off in
memory (Groq, OmniRoute, Gemini and OpenRouter report "not configured", so no
request can reach them; the usage counters prove it) and:

1. start-up: the island and the panel are on screen, no package failed;
2. fast path: 10 common commands through the typed path -> the right tool,
   zero model requests (provider ``fastpath`` counts them);
3. local brain: Sorani commands the fast path does not know are answered by
   the local model -- SAM's own ``ollama serve`` on a spare port (v1's server
   on 11434 is never used) -- with the right tool;
4. library: the synthetic PDFs (two scanned pages, real Windows OCR) are added
   through knowledge_add, searched with citations, a library question is
   answered by the local brain with the pages under the answer (panel only),
   and the documents are removed again;
5. run_python: statistics code runs sandboxed (Low integrity) without asking,
   system code asks (declined here) and a trading order is blocked.

Clean-up: the chart's symbol/timeframe restored, SAM's drawings made here
removed, a new Notepad closed, the check's conversations/analyses/run folders
removed, SAM quits (stopping the ollama server it started; the port closes).
Every confirmation is declined; MT5 is read only; nothing is spoken (null
speaker) and the room is never recorded (silent file microphone).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

from _common import ROOT, Acceptance

import integration_smoke as smoke
import launcher_startup as proc

DEFAULT_HOME = r"C:\Users\samit\Desktop\SAM-Agent"
WORK = ROOT / "work" / "final-checks"
LOCAL_HOST = "127.0.0.1:11436"
CLOUD = ("groq", "omniroute", "gemini", "openrouter")
FAST = [
    ("نرخی زێڕ چەندە", "get_price"),
    ("what's the gold price?", "get_price"),
    ("ترەیدینگ ڤیو بکەرەوە", "tv_open"),
    ("گۆڵد لەسەر ١٥ خولەک پیشان بدە", "tv_set_chart"),
    ("هێڵی پشتگیری و بەرگری بکێشە", "analyze_market"),
    ("هێڵەکانت بسڕەوە", "clear_my_drawings"),
    ("ئاگادارکردنەوەکانم پیشان بدە", "list_alerts"),
    ("cancel alert 99999", "cancel_alert"),
    ("نۆتپاد بکەرەوە", "open_app"),
    ("بوەستە", "stop_all"),
]
# Sorani commands the fast path does not know. The window count is a known weak case
# for qwen3:8b (A/B 2026-09-25: an empty answer with 0 and with 2 history messages).
LOCAL = [
    ("نرخی زێڕ و زیو پێکەوە پێم بڵێ", {"get_price"}),
    ("چ ئاگادارکردنەوەیەکم بۆ زێڕ داناوە؟", {"list_alerts"}),
    ("ئەو پەنجەرانەی ئێستا کراونەتەوە بژمێرە", {"window_control"}),
]
LIBRARY_QUESTION = "ئۆردەر بلۆک چییە و چۆن بەکاری بهێنم؟"
PYTHON = [
    ("safe", "import statistics\nround(statistics.mean(data['closes']), 2)", '{"closes": [4301.5, 4312.25, 4298.0]}'),
    ("confirm", "import os\nlen(os.listdir('.'))", None),
    ("blocked", "import MetaTrader5 as mt5\nmt5.order_send({})", None),
]


class Driver:
    def __init__(self, acc: Acceptance, app: Any, core: Any, since: float) -> None:
        self.acc, self.app, self.core, self.since = acc, app, core, since
        self.events: list[Any] = []
        self.report: dict[str, Any] = {}
        self.conversations: set[int] = set()
        self.documents: list[int] = []

    # -- plumbing ------------------------------------------------------------------------------------
    def run(self, coro: Any, timeout: float = 60) -> Any:
        return self.core.submit(coro).result(timeout)

    def on_event(self, event: Any) -> None:
        from sam.events import ConfirmRequest

        self.events.append(event)
        if isinstance(event, ConfirmRequest):          # this check never approves anything
            self.app.confirm.resolve(event.confirm_id, False, via="click")

    async def _subscribe(self) -> None:
        from sam.events import ConfirmRequest, ToolFinished, Transcript

        self.app.bus.subscribe((ToolFinished, ConfirmRequest, Transcript), self.on_event)

    def usage(self) -> dict[str, int]:
        rows = self.app.db.query("SELECT provider, SUM(requests) AS n FROM usage_counters GROUP BY provider")
        return {r["provider"]: int(r["n"] or 0) for r in rows}

    def say(self, text: str, timeout: float = 90) -> dict[str, Any]:
        from sam.events import ConfirmRequest, ToolFinished, Transcript

        mark, began = len(self.events), time.perf_counter()
        try:
            reply = self.run(self.app.submit_text(text), timeout)
            error = ""
        except Exception as exc:  # noqa: BLE001 - reported by the check
            reply, error = "", f"{type(exc).__name__}: {exc}"
        time.sleep(0.3)                             # the last events of the turn reach on_event
        new = self.events[mark:]
        if self.app.conversation is not None and self.app.conversation.conversation_id:
            self.conversations.add(int(self.app.conversation.conversation_id))
        panel = [e.text for e in new if isinstance(e, Transcript) and e.role == "assistant"]
        return {"text": text, "reply": reply, "panel": panel[-1] if panel else "",
                "ms": round((time.perf_counter() - began) * 1000),
                "tools": [{"name": e.name, "ok": e.ok, "summary": e.summary[:140]} for e in new
                          if isinstance(e, ToolFinished)],
                "confirms": sum(isinstance(e, ConfirmRequest) for e in new), "error": error}

    def chart(self) -> dict[str, Any] | None:
        tv = self.app.trading.tv
        if tv is None or not self.run(tv.connect(), 30):
            return None
        return self.run(tv.chart_state(), 30)

    def notepads(self) -> set[int]:
        windows = self.run(self.app.hands.windows.list(), 30)
        return {w.hwnd for w in windows if w.process.lower() == "notepad.exe"}

    # -- the scenario --------------------------------------------------------------------------------
    def scenario(self) -> None:
        app = self.app
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and not app.db.scalar(
                "SELECT COUNT(*) FROM timings WHERE stage='startup:core_ready' AND at >= ?", (self.since,)):
            time.sleep(0.2)
        self.run(self._subscribe())
        self.cloud_off()
        with self.acc.check("1 start-up: island + panel on screen, packages started") as c:
            time.sleep(1.0)
            windows = proc.visible_windows({os.getpid()})
            rows = app.db.query("SELECT stage, ms FROM timings WHERE stage LIKE 'startup:%' AND at >= ?",
                                (self.since,))
            c.data = {"windows": windows, "failed": list(app.failed),
                      "timings": {r["stage"]: round(r["ms"]) for r in rows}}
            big = [w for w in windows if w["size"][0] >= 600 and w["size"][1] >= 400]
            small = [w for w in windows if w["size"][1] < 200]
            c.detail = f"{len(windows)} windows (panel {len(big)}, island {len(small)}); island_visible " \
                       f"{c.data['timings'].get('startup:island_visible')} ms"
            assert big and small and not app.failed, c.data
        chart0 = self.chart()
        owned0 = len(self.run(app.trading.tv.my_drawings(), 30)) if chart0 else 0
        pads0 = self.notepads()
        self.report["chart_before"] = {k: (chart0 or {}).get(k) for k in ("symbol", "resolution", "user_drawings")}
        try:
            self.fast_path(owned0)
            self.local_brain()
            self.library()
            self.python()
        finally:
            self.cleanup(chart0, owned0, pads0)

    def cloud_off(self) -> None:
        """Every cloud rung reports 'not configured' (in memory only) and the
        local brain uses SAM's own server on a spare port."""
        llm = self.app.llm
        for name in CLOUD:
            backend = llm.backends.get(name)
            if backend is not None:
                backend.configured = lambda: False   # type: ignore[method-assign]
        voice = self.app.voice
        if voice is not None:
            async def silent(*_args: Any, **_kwargs: Any) -> None:
                return None                        # no TTS request (KurdishTTS quota) for confirmation questions

            voice.speak = silent                   # type: ignore[method-assign]
        config = self.app.config
        original = config.get
        overrides = {"llm.local.host": LOCAL_HOST, "memory.extract_on_sleep": False,
                     "voice.listen_on_confirm": False}

        def get(key: str, default: Any = None) -> Any:
            return overrides[key] if key in overrides else original(key, default)

        config.get = get                          # type: ignore[method-assign]

    def fast_path(self, owned0: int) -> None:
        usage0 = self.usage()
        results = []
        for text, tool in FAST:
            if tool == "clear_my_drawings" and owned0:
                continue                           # never clear SAM drawings that were there before
            result = self.say(text)
            result["expected"] = tool
            results.append(result)
            time.sleep(1.0)
        usage1 = self.usage()
        self.report["fast_path"] = results
        with self.acc.check(f"2 fast path: {len(results)} commands, right tool, no model") as c:
            wrong = [r for r in results if [t["name"] for t in r["tools"]] != [r["expected"]]]
            spent = {p: usage1.get(p, 0) - usage0.get(p, 0) for p in (*CLOUD, "ollama")}
            fast = usage1.get("fastpath", 0) - usage0.get("fastpath", 0)
            c.data = {"results": [{k: r[k] for k in ("text", "reply", "ms", "tools")} for r in results],
                      "model_requests": spent, "fastpath_counted": fast}
            c.detail = "; ".join(f"{r['text']} -> {r['reply'][:50]} ({r['ms']} ms)" for r in results)
            assert not wrong and not any(spent.values()) and fast == len(results), \
                {"wrong": [(r["text"], r["tools"]) for r in wrong], "spent": spent, "fast": fast}

    def local_brain(self) -> None:
        from sam.brain.intents import match

        conversation = self.app.conversation
        with self.acc.check("3a local brain warm-up (own ollama serve on a spare port)") as c:
            began = time.perf_counter()
            task = self.run(_prewarm(conversation), 30)
            ok = self.run(_wait(task), 420) if task is not None else None
            c.data = {"warm_ok": ok, "seconds": round(time.perf_counter() - began, 1),
                      "status": self.app.llm.local_status()}
            c.detail = f"warm {ok} in {c.data['seconds']} s on {LOCAL_HOST}"
            assert ok, c.data
        from sam.brain.responder import LOCAL_NOT_DONE_CKB, SORANI_NOT_UNDERSTOOD

        with self.acc.check(f"3b local brain: {len(LOCAL)} Sorani commands outside the fast path, "
                            "right tool (at least all but one), nothing said without a tool, no cloud") as c:
            rows = []
            for text, wanted in LOCAL:
                assert match(text) is None, f"«{text}» is in the fast path grammar"
                usage0 = self.usage()
                result = self.say(text, timeout=300)
                usage1 = self.usage()
                spent = {p: usage1.get(p, 0) - usage0.get(p, 0) for p in (*CLOUD, "ollama")}
                called = [t["name"] for t in result["tools"]]
                # honest = every statement has a tool result behind it, or SAM says it did not do it
                # (run 4: the window count called list_alerts -- a wrong tool, but no invented answer)
                honest = bool(called) or result["reply"].strip() in (LOCAL_NOT_DONE_CKB, SORANI_NOT_UNDERSTOOD)
                rows.append({"text": text, "wanted": sorted(wanted), "called": called, "reply": result["reply"],
                             "ms": result["ms"], "requests": spent, "right": bool(set(called) & wanted),
                             "honest": honest})
            c.data = {"rows": rows}
            c.detail = "; ".join(f"«{r['text']}» -> {r['called']} {r['reply'][:60]} ({r['ms']} ms)" for r in rows)
            right = sum(r["right"] for r in rows)
            assert right >= len(rows) - 1, rows
            assert all(r["honest"] for r in rows), [r for r in rows if not r["honest"]]
            assert all(r["requests"]["ollama"] >= 1 and not any(r["requests"][p] for p in CLOUD) for r in rows), rows

    def library(self) -> None:
        app = self.app
        fixtures = ROOT / "tests" / "fixtures" / "knowledge"
        with self.acc.check("4a knowledge_add: the synthetic PDFs (real Windows OCR on the scans)") as c:
            before = {d["id"] for d in app.knowledge.documents()}
            result = self.run(app.tools.dispatch("knowledge_add", {"path": str(fixtures)}, source="text"), 120)
            deadline = time.monotonic() + 90
            while app.knowledge.running() and time.monotonic() < deadline:
                time.sleep(0.5)
            docs = [d for d in app.knowledge.documents() if d["id"] not in before]
            self.documents = [d["id"] for d in docs]
            c.data = {"summary": result["summary"], "documents": [{k: d[k] for k in ("title", "pages", "chunks",
                                                                                       "ocr_pages", "status")}
                                                                   for d in docs]}
            c.detail = "; ".join(f"{d['title']}: {d['pages']} pages, {d['ocr_pages']} OCR" for d in docs)
            assert result["ok"] and len(docs) == 3 and sum(d["ocr_pages"] for d in docs) >= 2, c.data
        with self.acc.check("4b knowledge_search: cited passages (Sorani + English questions)") as c:
            found = {}
            for query in ("ستۆپ لۆس لە کوێ دابنێم", "order block", "پشتگیری چییە", "RSI divergence"):
                res = self.run(app.tools.dispatch("knowledge_search", {"query": query, "k": 3}, source="text"), 30)
                cites = (res.get("data") or {}).get("citations") or []
                found[query] = [p.get("citation") for p in cites][:2]
            c.data = found
            c.detail = json.dumps(found, ensure_ascii=False)[:300]
            assert all(found.values()), found
        with self.acc.check("4c library question answered by the local brain, pages under the answer") as c:
            result = self.say(LIBRARY_QUESTION, timeout=300)
            c.data = result
            c.detail = f"{result['reply'][:120]} | panel: {result['panel'][-120:]}"
            assert result["reply"] and "لاپەڕە" in result["panel"] and "لاپەڕە" not in result["reply"], c.data

    def python(self) -> None:
        with self.acc.check("5 run_python: safe runs sandboxed, system code asks (declined), orders blocked") as c:
            out = {}
            for kind, code, data in PYTHON:
                args = {"code": code} if data is None else {"code": code, "data": data}
                mark = len(self.events)
                res = self.run(self.app.tools.dispatch("run_python", args, source="text"), 150)
                from sam.events import ConfirmRequest

                asked = sum(isinstance(e, ConfirmRequest) for e in self.events[mark:])
                out[kind] = {"ok": res["ok"], "summary": res["summary"][:120], "asked": asked,
                             "sandbox": (res.get("data") or {}).get("sandbox"),
                             "declined": (res.get("data") or {}).get("declined"),
                             "blocked": (res.get("data") or {}).get("blocked")}
                run_dir = (res.get("data") or {}).get("run_dir")
                if run_dir:
                    self.report.setdefault("run_dirs", []).append(run_dir)
            c.data = out
            c.detail = json.dumps(out, ensure_ascii=False)[:300]
            assert out["safe"]["ok"] and out["safe"]["sandbox"] == "low_integrity" and not out["safe"]["asked"], out
            assert out["confirm"]["asked"] == 1 and out["confirm"]["declined"], out
            assert out["blocked"]["blocked"] and not out["blocked"]["asked"], out

    def cleanup(self, chart0: dict | None, owned0: int, pads0: set[int]) -> None:
        app = self.app
        with self.acc.check("cleanup: chart restored, SAM drawings gone, Notepad closed, test data removed") as c:
            tv = app.trading.tv
            if chart0 is not None:
                if owned0 == 0:
                    c.data["cleared"] = self.run(tv.clear(), 30)
                state = self.run(tv.chart_state(), 30)
                if state.get("symbol") != chart0.get("symbol"):
                    self.run(tv.set_symbol(chart0["symbol"]), 60)
                if state.get("resolution") != chart0.get("resolution"):
                    self.run(tv.set_timeframe(chart0.get("timeframe") or chart0["resolution"]), 60)
                after = self.run(tv.chart_state(), 30)
                c.data["chart_after"] = {k: after.get(k) for k in ("symbol", "resolution", "user_drawings",
                                                                   "my_drawings")}
                assert after.get("symbol") == chart0.get("symbol"), c.data
                assert after.get("resolution") == chart0.get("resolution"), c.data
                assert after.get("user_drawings") == chart0.get("user_drawings"), c.data
            for hwnd in self.notepads() - pads0:
                self.run(app.hands.windows.close(hwnd), 30)
            c.data["notepad_left"] = len(self.notepads() - pads0)
            for doc_id in self.documents:
                app.knowledge.remove(doc_id)
            c.data["documents_left"] = [d["id"] for d in app.knowledge.documents() if d["id"] in self.documents]
            for run_dir in self.report.get("run_dirs", []):
                shutil.rmtree(run_dir, ignore_errors=True)
            ids = sorted(self.conversations | {r["id"] for r in app.db.query(
                "SELECT id FROM conversations WHERE started_at >= ?", (self.since,))})
            with app.db.transaction():
                for cid in ids:
                    app.db.execute("DELETE FROM turns WHERE conversation_id=?", (cid,))
                    app.db.execute("DELETE FROM brain_conversation_state WHERE conversation_id=?", (cid,))
                    app.db.execute("DELETE FROM conversations WHERE id=?", (cid,))
                app.db.execute("DELETE FROM analyses WHERE at >= ?", (self.since,))
            if app.conversation is not None:
                app.conversation.conversation_id = None
            c.data["conversations_removed"] = ids
            assert c.data["notepad_left"] == 0 and not c.data["documents_left"], c.data


async def _prewarm(conversation: Any) -> Any:
    return conversation.prewarm_local_brain("text")


async def _wait(task: Any) -> Any:
    return await task


def run_sam(acc: Acceptance, home: Path) -> dict[str, Any]:
    import sam.__main__ as sam_main
    import sam.ui
    from sam import winapp

    os.environ["SAM_LOG_DIR"] = str(WORK / "logs")
    os.environ["SAM_SHOW_PANEL"] = "1"            # a normal launch opens the panel
    smoke.silence_devices()
    since = time.time()
    real_run = sam.ui.run
    holder: dict[str, Any] = {}

    def run_with_driver(app: Any, core: Any, *, started: float | None = None) -> int:
        driver = holder["driver"] = Driver(acc, app, core, since)

        def go() -> None:
            try:
                driver.scenario()
            except Exception as exc:  # noqa: BLE001
                with acc.check("scenario crashed"):
                    raise exc
            finally:
                if not winapp.signal_quit():
                    from PySide6.QtCore import QMetaObject, Qt
                    from PySide6.QtWidgets import QApplication
                    QMetaObject.invokeMethod(QApplication.instance(), "quit", Qt.ConnectionType.QueuedConnection)

        threading.Thread(target=go, name="final-checks", daemon=True).start()
        return real_run(app, core, started=started)

    sam.ui.run = run_with_driver
    try:
        code = sam_main.main(["--home", str(home)])
    finally:
        sam.ui.run = real_run
    from sam.brain.local_server import is_listening

    with acc.check("quit: clean shutdown, SAM's ollama server stopped") as c:
        time.sleep(1.0)
        c.data = {"exit_code": code, "instance_running": winapp.instance_running(),
                  "ollama_port_open": is_listening(LOCAL_HOST), "v1_ollama_up": is_listening("127.0.0.1:11434")}
        assert code == 0 and not c.data["instance_running"] and not c.data["ollama_port_open"], c.data
    return holder["driver"].report if "driver" in holder else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", default=os.environ.get("SAM_HOME") or DEFAULT_HOME)
    args = parser.parse_args()
    home = Path(args.home)
    WORK.mkdir(parents=True, exist_ok=True)
    acc = Acceptance("final_checks")
    from sam import winapp
    from sam.brain.local_server import is_listening

    if winapp.instance_running():
        acc.skip("SAM 2 is already running; quit it first (this check never touches it)")
        return acc.finish()
    if is_listening(LOCAL_HOST):
        acc.skip(f"{LOCAL_HOST} is in use")
        return acc.finish()
    report = run_sam(acc, home)
    with acc.check("no key material in the logs") as c:
        logs = smoke.read_log(WORK / "logs")
        c.data = {"key_shapes_in_logs": len(smoke.KEY_SHAPES.findall(logs))}
        assert not c.data["key_shapes_in_logs"], c.data
    (WORK / "report.json").write_text(json.dumps({"report": report, "result": acc.result()}, ensure_ascii=False,
                                                 indent=1, default=str), encoding="utf-8")
    return acc.finish()


if __name__ == "__main__":
    sys.exit(main())
