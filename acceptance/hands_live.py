"""Live acceptance for sam/hands on this PC (run by hand or by run_all.py; it
cleans up after itself).

    .venv\\Scripts\\python.exe acceptance\\hands_live.py [--home PATH] [--skip-notepad] [--skip-paint]
                                                  [--skip-tv] [--skip-web]

1. Start-menu index; the 7 apps v1 missed (Chrome, Edge, TradingView,
   Telegram, MetaTrader 5, WhatsApp, Excel) plus VS Code and Notepad are
   resolved by English AND Sorani names (design acceptance 6); more aliases
   are reported.
2. open_app("ترەیدینگ ڤیو بکەرەوە") goes through the real chart bridge and
   reports its CDP state (TradingView is never restarted here: any restart
   question is answered NO by this script).
3. A NEW Notepad window: open_app -> type_text (Sorani, clipboard paste) ->
   read back through UI Automation -> UIA snapshot + screen_look timing ->
   OCR read and find (Sorani + English) -> clear -> close without saving.
   (Cleared first: an unsaved Notepad tab could otherwise be kept by
   Notepad's session restore and shown to the user later.)
3b. A save prompt: a NEW Paint window, its blank canvas changed (Ctrl+A,
   Delete) -> window_control close -> the prompt's buttons come back
   numbered -> click "Don't save" by number -> the window is gone.
4. Read-only UIA snapshot timing of windows that are already open (counts
   and timings only: control names can be private and are never printed).
5. system_control info (read only) and one DuckDuckGo search (no LLM).

Safety: it types only into the Notepad window it opened (a new hwnd that is
in front); if Notepad opens a tab in the user's window instead, nothing is
typed and only that empty "Untitled" tab is closed. The user's clipboard is
compared by hash (never printed). Confirmations are approved only for the
steps that close OUR window. The window that was in front at the start is
put back in front at the end.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from _common import Acceptance, ROOT  # noqa: F401 - ROOT puts the repo on sys.path

from sam.app import App
from sam.events import ConfirmRequest
from sam.hands import _win

SORANI_TEXT = "سڵاو، ئەمە تاقیکردنەوەی سامە بۆ نووسینی کوردی"
SEVEN = [("Chrome", "کرۆم"), ("Edge", "ئێج"), ("TradingView", "ترەیدینگ ڤیو"), ("Telegram", "تێلێگرام"),
         ("MetaTrader 5", "مێتاترەیدەر"), ("WhatsApp", "واتساپ"), ("Excel", "ئێکسڵ")]
MORE = [("VS Code", "ڤی ئێس کۆد"), ("Notepad", "نۆتپاد"), ("TradingView", "تریدینگ ڤیو"),
        ("TradingView", "ترێدینگ"), ("Chrome", "کرۆم بکەرەوە"), ("Settings", "ڕێکخستنەکان"),
        ("Calculator", "ژمێرەر"), ("Word", "وۆرد"), ("PowerPoint", "پاوەرپۆینت"), ("Paint", "پەینت"),
        ("Snipping Tool", "سنیپینگ تووڵ"), ("Terminal", "تێرمیناڵ"), ("Task Manager", "تاسک مانەجەر"),
        ("File Explorer", "فایل ئێکسپلۆرەر"), ("Cursor", "کێرسەر"), ("Spotify", "سپۆتیفای"),
        ("Discord", "دیسکۆرد"), ("Firefox", "فایەرفۆکس"), ("YouTube", "یوتیوب"), ("CMD", "سی ئێم دی"),
        ("Control Panel", "کۆنترۆڵ پانێڵ")]
APPROVE_TOOLS = {"window_control", "click"}  # only used on OUR Notepad/Paint windows


def ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def clipboard_hash(hands: Any) -> str:
    try:
        saved = hands.input.clipboard.snapshot()
    except OSError:
        return "locked"
    digest = hashlib.sha256()
    for fmt, data in saved:
        digest.update(str(fmt).encode())
        digest.update(data)
    return digest.hexdigest()[:16]


async def check_index(acc: Acceptance, app: App) -> None:
    apps = app.hands.apps
    with acc.check("Start-menu app index is built") as c:
        started = time.perf_counter()
        count = await apps.refresh()
        c.data.update(apps=count, ms=ms(started))
        assert count > 50, f"only {count} apps indexed"
    with acc.check("the 7 apps v1 missed are found by English and Sorani names") as c:
        found: dict[str, Any] = {}
        for english, sorani in SEVEN:
            a, b = await apps.resolve(english), await apps.resolve(sorani)
            found[english] = {"english": a.name if a else None, "sorani": b.name if b else None,
                              "kind": a.kind if a else None, "id": (a.aumid or a.path) if a else None}
        c.data["apps"] = found
        missing = [k for k, v in found.items() if not (v["english"] and v["sorani"])]
        assert not missing, f"not found: {missing}"
    with acc.check("more spoken names (VS Code, Notepad, spelling variants, Windows tools)") as c:
        started = time.perf_counter()
        results = {}
        for english, spoken in MORE:
            entry = await apps.resolve(spoken)
            results[spoken] = entry.name if entry else None
        c.data.update(resolved=results, ms_per_name=round(ms(started) / len(MORE), 2))
        wrong = [s for s, n in results.items() if n is None and s in ("ڤی ئێس کۆد", "نۆتپاد", "تریدینگ ڤیو", "ترێدینگ")]
        assert not wrong, f"not resolved: {wrong}"


async def check_tradingview(acc: Acceptance, app: App) -> None:
    with acc.check("open_app TradingView goes through the chart bridge (CDP port)") as c:
        if app.trading.tv is None:
            c.skip("the chart bridge package is not loaded")
        processes = await asyncio.to_thread(_win.running_processes)
        if not any(name.lower() == "tradingview.exe" for name in processes.values()):
            c.skip("TradingView is not running; this check never starts it")
        started = time.perf_counter()
        result = await app.tools.dispatch("open_app", {"name": "ترەیدینگ ڤیو بکەرەوە"}, source="text")
        data = result.get("data") or {}
        c.data.update(ok=result["ok"], tv_state=data.get("tv_state"), ms=ms(started), summary=result["summary"])
        assert result["ok"] and data.get("tv_state") in ("connected", "started"), result["summary"]
        state = await app.trading.tv.chart_state()
        c.data["chart"] = {"symbol": state.get("symbol"), "timeframe": state.get("timeframe")}


async def check_notepad(acc: Acceptance, app: App) -> None:
    hands = app.hands
    windows = hands.windows
    before = {w.hwnd: w for w in await windows.list() if w.process.lower() == "notepad.exe"}
    mine = None
    with acc.check("open_app opens a new Notepad window (Sorani name)") as c:
        started = time.perf_counter()
        opened = await app.tools.dispatch("open_app", {"name": "نۆتپاد", "new_window": True}, source="text")
        c.data.update(ok=opened["ok"], summary=opened["summary"], ms=ms(started), notepad_windows_before=len(before))
        await asyncio.sleep(0.8)
        ours = [w for w in await windows.list() if w.process.lower() == "notepad.exe" and w.hwnd not in before]
        if ours:
            mine = ours[0]
        else:
            await close_stray_tab(app, before, c)
        assert opened["ok"], opened["summary"]
        assert mine is not None, "Notepad added a tab to an existing window instead of a new window"
    if mine is None:
        return
    try:
        await notepad_steps(acc, app, mine)
    finally:
        with acc.check("our Notepad window is closed without saving") as c:
            c.data.update(await close_ours(app, mine.hwnd))
            assert c.data.get("closed_finally"), c.data


async def close_stray_tab(app: App, before: dict[int, Any], c: Any) -> None:
    """Notepad put a new tab into the user's window: close that tab only if it
    is an empty 'Untitled' tab (never anything with the user's text)."""
    hands = app.hands
    fg = await hands.windows.foreground()
    if fg is None or fg.hwnd not in before:
        return
    _, text = await hands.uia.focused_text()
    if text == "" and fg.title.lower().startswith("untitled"):
        await hands.run_input("press_keys", "ctrl+w")
        c.data["closed_our_empty_tab"] = True
    if before[fg.hwnd].minimized:
        await hands.windows.act("minimize", fg)


async def notepad_steps(acc: Acceptance, app: App, mine: Any) -> None:
    hands = app.hands
    windows = hands.windows
    with acc.check("type_text types Sorani by paste and restores the clipboard") as c:
        assert await windows.focus(mine.hwnd), "could not focus our Notepad window"
        fg = await windows.foreground()
        assert fg is not None and fg.hwnd == mine.hwnd, "our Notepad window is not in front; nothing typed"
        clip_before = clipboard_hash(hands)
        started = time.perf_counter()
        typed = await app.tools.dispatch("type_text", {"text": SORANI_TEXT}, source="text")
        data = typed.get("data") or {}
        c.data.update(ok=typed["ok"], summary=typed["summary"], ms=ms(started), verified_by_tool=data.get("verified"),
                      method=data.get("method"))
        c.data["clipboard_restored"] = clip_before == clipboard_hash(hands)
        assert typed["ok"] and data.get("verified"), typed["summary"]
        assert c.data["clipboard_restored"], "the user's clipboard changed"
    with acc.check("UI Automation reads the Sorani text back") as c:
        started = time.perf_counter()
        controls = await hands.uia.snapshot(mine.hwnd, title=mine.title, window_rect=mine.rect)
        c.data.update(controls=len(controls), snapshot_ms=ms(started), sample=[x.line() for x in controls[:8]])
        document = next((x for x in controls if x.role in ("document", "edit")), None)
        text = await hands.uia.read(document.number) if document else None
        c.data["text_matches"] = bool(text and SORANI_TEXT in text)
        assert c.data["text_matches"], "the document text does not contain what was typed"
    with acc.check("screen_look lists numbered controls") as c:
        started = time.perf_counter()
        look = await app.tools.dispatch("screen_look", {"mode": "controls"}, source="text")
        c.data.update(ok=look["ok"], summary=look["summary"], ms=ms(started))
        assert look["ok"] and "controls" in look["summary"], look["summary"]
    with acc.check("OCR reads the Notepad window and finds Sorani and English text") as c:
        started = time.perf_counter()
        lines = await hands.ocr.read(mine.hwnd)
        c.data.update(lines=len(lines), read_ms=ms(started))
        started = time.perf_counter()
        hits = await hands.ocr.find_text("تاقیکردنەوەی سامە", mine.hwnd)
        c.data.update(sorani_found=bool(hits), sorani_ms=ms(started), sorani_best=hits[0]["text"] if hits else None)
        started = time.perf_counter()
        menu = await hands.ocr.find_text("File", mine.hwnd)
        c.data.update(english_found=bool(menu), english_ms=ms(started))
        assert lines and menu, "OCR found no text / no 'File' menu"
        assert hits, "OCR did not find the Sorani phrase"


async def close_ours(app: App, hwnd: int) -> dict[str, Any]:
    """Clear the text, close the window, answer "Don't save" if asked."""
    hands = app.hands
    windows = hands.windows
    report: dict[str, Any] = {}
    if await windows.find(hwnd) is None:
        report["closed_finally"] = True
        return report
    await windows.focus(hwnd)
    fg = await windows.foreground()
    if fg is not None and fg.hwnd == hwnd:
        await hands.run_input("press_keys", "ctrl+a, delete")
        await asyncio.sleep(0.3)
    closed = await app.tools.dispatch("window_control", {"action": "close", "target": str(hwnd)}, source="text")
    report["close"] = {"ok": closed["ok"], "summary": closed["summary"]}
    if not closed["ok"]:
        await asyncio.sleep(0.5)
        await hands.uia.snapshot(hwnd)
        button = hands.uia.find("Don't save")
        if button is not None:
            clicked = await app.tools.dispatch("click", {"target": str(button.number)}, source="text")
            report["dont_save"] = {"ok": clicked["ok"], "summary": clicked["summary"]}
    for _ in range(25):
        if await windows.find(hwnd) is None:
            break
        await asyncio.sleep(0.2)
    report["closed_finally"] = await windows.find(hwnd) is None
    return report


async def check_save_prompt(acc: Acceptance, app: App) -> None:
    """window_control close on a changed window -> numbered prompt -> click
    "Don't save" (the flow the model follows after asking the user)."""
    hands = app.hands
    windows = hands.windows
    before = {w.hwnd for w in await windows.list() if w.process.lower() == "mspaint.exe"}
    mine = None
    try:
        with acc.check("a save prompt is answered with Don't save (new Paint window)") as c:
            opened = await app.tools.dispatch("open_app", {"name": "پەینت", "new_window": True}, source="text")
            c.data["open"] = opened["summary"]
            for _ in range(30):
                ours = [w for w in await windows.list() if w.process.lower() == "mspaint.exe" and w.hwnd not in before]
                if ours:
                    mine = ours[0]
                    break
                await asyncio.sleep(0.3)
            if mine is None:
                c.skip("Paint did not open a new window")
            await asyncio.sleep(1.5)  # let the canvas load
            assert await windows.focus(mine.hwnd), "could not focus our Paint window"
            fg = await windows.foreground()
            assert fg is not None and fg.hwnd == mine.hwnd, "our Paint window is not in front; nothing pressed"
            await hands.run_input("press_keys", "ctrl+a, delete")  # change OUR blank canvas
            await asyncio.sleep(0.5)
            started = time.perf_counter()
            closed = await app.tools.dispatch("window_control", {"action": "close", "target": str(mine.hwnd)},
                                              source="text")
            data = closed.get("data") or {}
            buttons = (data.get("untrusted") or {}).get("dialog_buttons") or []
            c.data.update(close_ok=closed["ok"], close_summary=closed["summary"], buttons=buttons,
                          close_ms=round((time.perf_counter() - started) * 1000))
            if closed["ok"]:
                c.data["prompt"] = False  # Paint closed without asking: nothing to answer
                return
            number = next((line.split(".", 1)[0] for line in buttons
                           if "save" in line.lower() and ("don" in line.lower() or "not" in line.lower())), None)
            assert number, f"no Don't save button among {buttons}"
            clicked = await app.tools.dispatch("click", {"target": number}, source="text")
            c.data["dont_save"] = {"ok": clicked["ok"], "summary": clicked["summary"]}
            for _ in range(25):
                if await windows.find(mine.hwnd) is None:
                    break
                await asyncio.sleep(0.2)
            c.data["closed"] = await windows.find(mine.hwnd) is None
            assert clicked["ok"] and c.data["closed"], c.data
    finally:
        if mine is not None and await windows.find(mine.hwnd) is not None:
            # Last resort for OUR window only: close again and answer Don't save.
            await windows.act("close", mine)
            await hands.uia.snapshot(mine.hwnd)
            button = hands.uia.find("Don't save")
            if button is not None:
                await hands.uia.click(button.number)


async def check_uia_existing(acc: Acceptance, app: App) -> None:
    with acc.check("UIA snapshot timing of windows already open (read only)") as c:
        timings = []
        for window in (await app.hands.windows.list())[:6]:
            if window.minimized:
                continue
            started = time.perf_counter()
            try:
                controls = await app.hands.uia.snapshot(window.hwnd, title=window.title, window_rect=window.rect)
                timings.append({"app": window.process, "controls": len(controls), "ms": ms(started)})
            except Exception as exc:  # noqa: BLE001 - a hung app is a result
                timings.append({"app": window.process, "error": type(exc).__name__, "ms": ms(started)})
        c.data["windows"] = timings
        assert timings and all("error" not in t for t in timings), timings


async def check_system_and_web(acc: Acceptance, app: App, skip_web: bool) -> None:
    with acc.check("system_control info reads battery, disks, memory and volume") as c:
        result = await app.tools.dispatch("system_control", {"action": "info"}, source="text")
        data = result.get("data") or {}
        c.data.update(ok=result["ok"], battery=data.get("battery"), volume=data.get("volume"),
                      disks=len(data.get("disks") or []), time=data.get("time"))
        assert result["ok"] and "level" in (data.get("volume") or {}), data.get("volume")
    with acc.check("web_search (DuckDuckGo, no LLM) returns parsed results") as c:
        if skip_web:
            c.skip("--skip-web")
        started = time.perf_counter()
        result = await app.tools.dispatch("web_search", {"query": "XAUUSD gold price today"}, source="text")
        data = result.get("data") or {}
        c.data.update(ok=result["ok"], engine=data.get("engine"), results=len(data.get("untrusted") or []),
                      ms=ms(started), first_url=((data.get("untrusted") or [{}])[0] or {}).get("url"))
        assert result["ok"] and c.data["results"] >= 3, result["summary"]


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", default=None)
    parser.add_argument("--skip-notepad", action="store_true")
    parser.add_argument("--skip-paint", action="store_true")
    parser.add_argument("--skip-tv", action="store_true")
    parser.add_argument("--skip-web", action="store_true")
    args = parser.parse_args()
    # A throw-away home: no keys are needed here and nothing is written to SAM_HOME.
    home = Path(args.home) if args.home else Path(tempfile.mkdtemp(prefix="sam2-hands-live-",
                                                                   dir=str(ROOT / "work")))
    (home / "data").mkdir(parents=True, exist_ok=True)
    app = App(home, environ={}, llm_backends={})
    app.load_packages(["sam.hands"] + ([] if args.skip_tv else ["sam.trading.chart_tools"]))
    await app.start()

    def answer(event: ConfirmRequest) -> None:
        # This script plays the user: YES only for closing our own Notepad
        # window; NO to everything else (e.g. restarting TradingView).
        app.confirm.resolve(event.confirm_id, event.tool_name in APPROVE_TOOLS, via="click")
    app.bus.subscribe(ConfirmRequest, answer)
    acc = Acceptance("hands_live")
    front = await app.hands.windows.foreground()
    try:
        await check_index(acc, app)
        if not args.skip_tv:
            await check_tradingview(acc, app)
        if not args.skip_notepad:
            await check_notepad(acc, app)
        if not args.skip_paint:
            await check_save_prompt(acc, app)
        await check_uia_existing(acc, app)
        await check_system_and_web(acc, app, args.skip_web)
    finally:
        if front is not None and await app.hands.windows.find(front.hwnd) is not None:
            await app.hands.windows.focus(front.hwnd)  # give the user their window back
        await app.stop()
        app.close()
        if not args.home:
            shutil.rmtree(home, ignore_errors=True)  # the throw-away home this run created
    return acc.finish()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
