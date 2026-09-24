"""The hands tools (contract section 2), each with Sorani examples and a
risk decided by code. Handlers reach the package through ``ctx.app.hands``.

Two small additions to the catalogue (reported to the lead): ``fetch_page``
(read a web page's text as untrusted data) and ``system_control`` (battery/
time/disk info and exact volume); ``files`` also accepts ``rename``, and
``open_app`` accepts ``new_window``.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from ..brain.tools import ToolContext, fail, ok, tool
from ..textnorm import normalize_ckb
from . import guards
from .windows import MT5_PROCESSES, scrub_title

def _hands(ctx: ToolContext) -> Any:
    hands = getattr(ctx.app, "hands", None)
    if hands is None:
        raise RuntimeError("the hands package is not loaded")
    return hands


def _result(data: dict[str, Any]) -> dict[str, Any]:
    """Turn a module result dict into a registry result."""
    body = dict(data)
    good = bool(body.pop("ok", False))
    summary = str(body.pop("summary", "") or ("Done." if good else "Failed."))
    return ok(summary, **body) if good else fail(summary, **body)


# -- open_app / windows -------------------------------------------------------------
@tool("open_app",
      description="Open or bring to the front any installed app by its English or Sorani name (Chrome, "
                  "TradingView, MetaTrader 5, Telegram, VS Code, Excel, Settings...). Checks that its window appeared.",
      description_ckb="کردنەوەی بەرنامە",
      params={"type": "object", "properties": {
          "name": {"type": "string", "description": "as the user said it"},
          "args": {"type": "string", "description": "command-line arguments"},
          "new_window": {"type": "boolean", "description": "new window even if already open"}},
          "required": ["name"]},
      # 150 s like tv_open: a port-less TradingView needs the 20 s confirmation
      # plus a graceful close (10 s) and a cold start with chart load (~60 s).
      # Arguments need a yes; shell/interpreter arguments are classified like
      # run_powershell (guards.open_app_risk).
      classify=guards.open_app_risk, blocking=True, timeout_s=150,
      examples_ckb=("کرۆم بکەرەوە", "ترەیدینگ ڤیو بکەرەوە", "مێتاترەیدەر بکەرەوە"))
async def open_app(ctx: ToolContext, name: str, args: str = "", new_window: bool = False,
                   **_ignored: Any) -> dict[str, Any]:
    return _result(await _hands(ctx).apps.launch(name, args, new_window=new_window, confirm=ctx.confirm))


def _window_risk(args: dict[str, Any]) -> tuple[str, str | None]:
    if str(args.get("action", "")).lower() == "close":
        target = str(args.get("target") or "").strip()
        return "confirm", f"پەنجەرەی «{target}» دابخەم؟" if target else "ئەم پەنجەرەیە دابخەم؟"
    return "safe", None


@tool("window_control",
      description="List, focus, minimize, maximize, restore, close or snap (left/right half) an open window, "
                  "verified afterwards.",
      description_ckb="کۆنترۆڵی پەنجەرەکان",
      params={"type": "object", "properties": {
          "action": {"type": "string", "enum": ["list", "focus", "minimize", "maximize", "restore", "close",
                                                 "snap_left", "snap_right"]},
          "target": {"type": "string", "description": "title or app name; empty = current window"}},
          "required": ["action"]},
      classify=_window_risk, blocking=True, timeout_s=20,
      examples_ckb=("کرۆم بچووک بکەرەوە", "نۆتپاد دابخە", "ئەم پەنجەرەیە ببە بۆ لای چەپ"))
async def window_control(ctx: ToolContext, action: str, target: str = "", **_ignored: Any) -> dict[str, Any]:
    hands = _hands(ctx)
    if action == "list":
        windows = await hands.windows.list()
        return ok(f"{len(windows)} windows are open.", untrusted=[w.brief() for w in windows[:40]])
    window = await hands.windows.find(target)
    if window is None:
        names = [w.title[:60] for w in (await hands.windows.list())[:15]]
        return fail(f"No open window matches '{target}'.", open_windows=names)
    result = await hands.windows.act(action, window)
    result.setdefault("window", window.title[:120])
    if action == "close" and not result.get("ok"):
        await _describe_save_prompt(hands, window, result)
    return _result(result)


async def _describe_save_prompt(hands: Any, window: Any, result: dict[str, Any]) -> None:
    """A window that stays open after WM_CLOSE is usually asking to save.
    Number its buttons (a UIA snapshot, so ``click`` can answer by number).
    Measured on this PC: Paint's WinUI prompt lists Save / Don’t save /
    Cancel as the first three buttons of the window's tree (snapshot 216 ms)."""
    try:
        controls = await hands.uia.snapshot(window.hwnd, title=window.title, window_rect=window.rect)
    except Exception:  # noqa: BLE001 - the plain result is still honest
        return
    buttons = [c.line() for c in controls if c.role == "button" and c.enabled][:6]
    if buttons:
        result["untrusted"] = {"dialog_buttons": buttons}
        result["summary"] = (f"'{window.title[:80]}' is still open and shows a prompt (probably about saving). "
                             "Its buttons are numbered: ask the user what to do, then click the number.")


# -- keyboard / mouse -----------------------------------------------------------------
async def _focus_work_window(hands: Any, target: str = "") -> Any:
    """The window the action is meant for (never SAM's own panel)."""
    window = await hands.windows.find(target) if target else await hands.windows.foreground()
    if window is not None and not window.foreground:
        await hands.windows.focus(window.hwnd)
    return window


@tool("type_text",
      description="Type text (Sorani works) into the current window, or into a control by its number from the "
                  "last screen_look or its name. The user's clipboard is restored.",
      description_ckb="نووسینی دەق",
      params={"type": "object", "properties": {
          "text": {"type": "string"},
          "target": {"type": "string", "description": "control number or name"},
          "press_enter": {"type": "boolean"}},
          "required": ["text"]},
      classify=lambda args: guards.type_risk(args), blocking=True, timeout_s=30, private_args=("text",),
      examples_ckb=("بنووسە سڵاو چۆنی", "ئەمە بنووسە و ئینتەر دابگرە"))
async def type_text(ctx: ToolContext, text: str, target: str = "", press_enter: bool = False,
                    **_ignored: Any) -> dict[str, Any]:
    hands = _hands(ctx)
    window = await _focus_work_window(hands)
    if window is None:
        return fail("There is no window to type into.")
    if target:
        await hands.uia.ensure_fresh(window.hwnd)
        if hands.uia.find(target) is not None:
            return _result(await hands.uia.type(target, text, press_enter=press_enter))
        hits = await hands.ocr.find_text(target, window.hwnd)
        if not hits:
            return fail(f"Could not find '{target}' in '{window.title[:60]}'.")
        rect = hits[0]["rect"]
        await hands.run_input("click", (rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2)
    typed = await hands.run_input("type_text", text, press_enter=press_enter)
    await asyncio.sleep(0.15)
    name, content = await hands.uia.focused_text()
    verified = content is not None and normalize_ckb(text)[:40] in normalize_ckb(content)
    return ok(f"Typed {len(text)} characters into '{window.title[:60]}'" + (" (checked on screen)." if verified else "."),
              verified=verified, method=typed.get("method"), clipboard_restored=typed.get("restored"))


@tool("press_keys",
      description="Press shortcuts or keys in the current window ('ctrl+s', 'alt+tab', 'enter', 'f5'; several "
                  "separated by commas) or media keys: volume_up, volume_down, volume_mute, play_pause, next_track, "
                  "prev_track.",
      description_ckb="داگرتنی کلیلەکان",
      params={"type": "object", "properties": {
          "keys": {"type": "string"}, "repeat": {"type": "integer", "description": "1-50, default 1"}},
          "required": ["keys"]},
      classify=guards.keys_risk, blocking=True, timeout_s=20,
      examples_ckb=("دەنگەکە بەرز بکەرەوە", "کۆنترۆڵ ئێس دابگرە", "گۆرانییەکە ڕابگرە"))
async def press_keys(ctx: ToolContext, keys: str, repeat: int = 1, **_ignored: Any) -> dict[str, Any]:
    from .input import parse_keys

    hands = _hands(ctx)
    try:
        chords = parse_keys(keys)
    except ValueError as exc:
        return fail(f"Unknown key: {exc}.")
    media = all(len(c) == 1 and 0xAD <= c[0] <= 0xB3 for c in chords)
    window = None if media else await _focus_work_window(hands)
    sent = await hands.run_input("press_keys", keys, repeat)
    if media and any(c[0] in (0xAD, 0xAE, 0xAF) for c in chords):
        try:
            volume = await asyncio.to_thread(hands.system.get_volume)
            return ok(f"Pressed {', '.join(sent)}; volume is now {volume['level']}%"
                      + (" (muted)." if volume["muted"] else "."), volume=volume)
        except OSError:
            pass
    return ok(f"Pressed {', '.join(sent)}" + (f" in '{window.title[:60]}'." if window else "."), keys=sent)


SCROLL_CLICKS = 5  # wheel steps per scroll request (one step = 3 text lines in most apps)


@tool("click",
      description="Click a control number from the last screen_look, a control name or visible text; "
                  "scroll_up/scroll_down scrolls over it instead.",
      description_ckb="کلیک کردن",
      params={"type": "object", "properties": {
          "target": {"type": "string"},
          "button": {"type": "string", "enum": ["left", "right", "double", "scroll_up", "scroll_down"]},
          "window": {"type": "string", "description": "empty = current window"}},
          "required": ["target"]},
      classify=guards.click_risk, blocking=True, timeout_s=30,
      examples_ckb=("کلیک لە Save بکە", "دوگمەی ژمارە ٣ دابگرە", "لیستەکە بەرەو خوارەوە ببە"))
async def click(ctx: ToolContext, target: str, button: str = "left", window: str = "",
                **_ignored: Any) -> dict[str, Any]:
    hands = _hands(ctx)
    win = await _focus_work_window(hands, window)
    if win is None:
        return fail("There is no window to click in.")
    scroll = SCROLL_CLICKS if button == "scroll_up" else -SCROLL_CLICKS if button == "scroll_down" else 0
    double = button == "double"
    mouse_button = "right" if button == "right" else "left"

    async def on_control() -> dict[str, Any]:
        if scroll:
            return _result(await hands.uia.scroll(target, scroll))
        return _result(await hands.uia.click(target, button=mouse_button, double=double))

    if target.strip().isdigit():
        if not hands.uia.controls:
            return fail("Use screen_look first: there is no numbered list of controls yet.")
        return await on_control()
    await hands.uia.ensure_fresh(win.hwnd)
    if hands.uia.find(target) is not None:
        return await on_control()
    hits = await hands.ocr.find_text(target, win.hwnd)
    if not hits:
        return fail(f"'{target}' is not visible in '{win.title[:60]}'.")
    rect = hits[0]["rect"]
    x, y = (rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2
    if scroll:
        await hands.run_input("scroll", x, y, scroll)
        return ok(f"Scrolled {'up' if scroll > 0 else 'down'} over the text '{hits[0]['text']}'.", at=[x, y])
    await hands.run_input("click", x, y, button=mouse_button, double=double)
    return ok(f"Clicked the text '{hits[0]['text']}' ({mouse_button}{', double' if double else ''}).", at=[x, y])


# -- screen ----------------------------------------------------------------------------
@tool("screen_look",
      description="Look at a window: 'controls' (default) numbers its buttons/fields for click and type_text; "
                  "'text' reads the visible text; 'describe' asks a vision model the query (daily budget).",
      description_ckb="سەیرکردنی شاشە",
      params={"type": "object", "properties": {
          "window": {"type": "string", "description": "empty = current window"},
          "mode": {"type": "string", "enum": ["controls", "text", "describe"]},
          "query": {"type": "string", "description": "Question for mode describe"}}},
      risk="safe", blocking=True, timeout_s=60,
      examples_ckb=("چی لەسەر شاشەکەیە؟", "ئەم پەنجەرەیە چی تێدایە؟"))
async def screen_look(ctx: ToolContext, window: str = "", mode: str = "controls", query: str = "",
                      **_ignored: Any) -> dict[str, Any]:
    hands = _hands(ctx)
    win = await _focus_work_window(hands, window)
    if win is None:
        return fail(f"No window matches '{window}'." if window else "There is no window to look at.")
    title = scrub_title(win.process, win.title)[:120]
    private = win.process.lower() in MT5_PROCESSES
    if mode == "controls":
        controls = await hands.uia.snapshot(win.hwnd, title=win.title, window_rect=win.rect)
        if len(controls) >= 3:
            listing = hands.uia.describe()
            return ok(f"'{title}' has {len(controls)} controls (numbered).", window=title,
                      ms=hands.uia.last_ms, untrusted=_mt5_text(listing) if private else listing)
        mode = "text"  # canvas/Electron windows expose almost nothing to UIA
    if mode == "text":
        lines = await hands.ocr.read(win.hwnd)
        text = "\n".join(line["text"] for line in lines[:120])
        return ok(f"Read {len(lines)} lines of text in '{title}'.", window=title,
                  untrusted=_mt5_text(text) if private else text)
    shot = await hands.screen.capture_ex(win.hwnd)
    answer = await hands.vision.describe(shot, query)
    return ok(f"Looked at '{title}' with the vision model.", window=title, untrusted=answer)


_BALANCE_LINE = re.compile(r"(?i)balance|equity|margin|profit|credit|login|account|بالانس|ئیکویتی")


def _mt5_text(text: str) -> str:
    """MetaTrader 5 screen text for a model: no account-number digit runs and no
    balance/equity/margin lines (screenshots were already blanked; text was not)."""
    kept = [scrub_title("terminal64.exe", line) for line in str(text).splitlines() if not _BALANCE_LINE.search(line)]
    return "\n".join(kept)


@tool("screen_act",
      description="Do a multi-step task inside one window by looking at screenshots (slow, limited per day). "
                  "Only when click, type_text and press_keys cannot do it.",
      description_ckb="کارکردن لەسەر شاشە بە بینین",
      params={"type": "object", "properties": {
          "goal": {"type": "string", "description": "the complete goal"},
          "window": {"type": "string"}, "max_steps": {"type": "integer", "description": "1-12"}},
          "required": ["goal"]},
      risk="safe", blocking=False, timeout_s=420,
      examples_ckb=("لەسەر شاشەکە دوگمەی داگرتن بدۆزەرەوە و کلیکی لێبکە",))
async def screen_act(ctx: ToolContext, goal: str, window: str = "", max_steps: int = 12,
                     **_ignored: Any) -> dict[str, Any]:
    hands = _hands(ctx)
    win = await _focus_work_window(hands, window)
    if win is None:
        return fail("There is no window to work in.")
    result = await hands.vision.act(goal, win.hwnd, max_steps, confirm=ctx.confirm, progress=ctx.progress,
                                    cancel=ctx.cancel)
    return _result(result)


# -- shell / files ------------------------------------------------------------------------
def _powershell_risk(args: dict[str, Any]) -> tuple[str, str | None]:
    from .policy import classify_powershell

    risk, reason = classify_powershell(str(args.get("command", "")))
    if risk == "confirm":
        # The command is on the confirmation card; it is not read aloud (text
        # the model wrote could contain a "yes" that SAM's mic hears back).
        return "confirm", "ئەم فەرمانەی پاوەرشێڵ جێبەجێ بکەم؟ فەرمانەکە لەسەر شاشەیە."
    if risk == "blocked":
        return "blocked", reason
    return "safe", None


@tool("run_powershell",
      description="Run a PowerShell command and return its output. Read-only commands run at once; changes are "
                  "confirmed with the user; dangerous ones are blocked.",
      description_ckb="جێبەجێکردنی فەرمانی پاوەرشێڵ",
      params={"type": "object", "properties": {
          "command": {"type": "string"}, "timeout_s": {"type": "integer", "description": "1-300"}},
          "required": ["command"]},
      classify=_powershell_risk, blocking=True, timeout_s=320,
      examples_ckb=("ئای پی یەکەم چییە؟", "چەند بۆشایی لە دیسکەکەم ماوە؟"))
async def run_powershell(ctx: ToolContext, command: str, timeout_s: int | None = None,
                         **_ignored: Any) -> dict[str, Any]:
    from .shell import run_powershell as run

    limit = float(timeout_s or ctx.app.config.get("hands.powershell_timeout_s", 45) or 45)
    result = await run(command, timeout_s=max(1.0, min(limit, 300.0)), cancel=ctx.cancel)
    output = {"stdout": ctx.app.redact(result.stdout), "stderr": ctx.app.redact(result.stderr)}
    data = {"exit_code": result.exit_code, "ms": result.duration_ms, "untrusted": output,
            "truncated": result.truncated}
    if result.cancelled:
        return fail("Stopped by the user; the command was killed.", cancelled=True, **data)
    if result.timed_out:
        return fail(f"The command did not finish in {limit:.0f} s and was stopped.", timeout=True, **data)
    if result.exit_code != 0:
        return fail(f"The command failed (exit code {result.exit_code}).", **data)
    return ok("The command finished." if result.stdout else "The command finished with no output.", **data)


def _files_risk(args: dict[str, Any]) -> tuple[str, str | None]:
    hands = getattr(guards.APP.get("app"), "hands", None)
    if hands is None:
        return "confirm", None
    action = str(args.get("action", "")).lower()
    path = str(args.get("path", ""))
    risk, reason = hands.policy.classify_path(path, action, dest=args.get("dest") or None,
                                              content=args.get("content") or None)
    if risk == "blocked":
        return "blocked", reason
    if risk == "safe":
        return "safe", None
    name = path.replace("\\", "/").rstrip("/").split("/")[-1] or path
    question = {"delete": f"«{name}» بسڕمەوە؟ دەچێتە زبڵدانەوە.",
                "write": f"فایلی «{name}» بنووسم؟", "append": f"شت بۆ فایلی «{name}» زیاد بکەم؟",
                "move": f"«{name}» بگوازمەوە بۆ «{args.get('dest', '')}»؟",
                "copy": f"«{name}» کۆپی بکەم بۆ «{args.get('dest', '')}»؟",
                "rename": f"ناوی «{name}» بگۆڕم بۆ «{args.get('dest', '')}»؟",
                "open": f"«{name}» بکەمەوە؟ بەرنامەیەک جێبەجێ دەکات."}.get(action, f"ئەم کارە لەسەر «{name}» بکەم؟")
    return "confirm", question


@tool("files",
      description="Files and folders. Paths may start with Desktop/, Documents/, Downloads/, Pictures/, "
                  "Projects/ (Sorani names work). delete = Recycle Bin; search matches names (content = text "
                  "inside); reveal shows it in Explorer.",
      description_ckb="کارکردن لەگەڵ فایلەکان",
      params={"type": "object", "properties": {
          "action": {"type": "string", "enum": ["list", "read", "write", "append", "copy", "move", "rename",
                                                 "delete", "open", "search", "reveal"]},
          "path": {"type": "string"}, "content": {"type": "string", "description": "text to write or find"},
          "dest": {"type": "string", "description": "destination, or the new name"},
          "pattern": {"type": "string", "description": "name pattern, e.g. *.pdf or words"}},
          "required": ["action", "path"]},
      classify=_files_risk, blocking=True, timeout_s=30, private_args=("content",),
      examples_ckb=("فایلەکانی سەر دێسکتۆپ پیشان بدە", "فایلێکی نوێ لە دێسکتۆپ دروست بکە"))
async def files(ctx: ToolContext, action: str, path: str, content: str = "", dest: str = "",
                pattern: str = "", **_ignored: Any) -> dict[str, Any]:
    hands = _hands(ctx)
    result = await asyncio.to_thread(hands.files.run, action, path, content=content or None, dest=dest or None,
                                     pattern=pattern or None)
    return _result(result)


# -- web -------------------------------------------------------------------------------------
def _url_risk(args: dict[str, Any]) -> tuple[str, str | None]:
    from .policy import Policy

    risk, reason = Policy.classify_url(str(args.get("url", "")))
    return (risk, reason if risk == "blocked" else None)


@tool("open_url",
      description="Open a web address (http/https) in the user's default browser.",
      description_ckb="کردنەوەی ماڵپەڕ",
      params={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
      classify=_url_risk, blocking=True, timeout_s=20,
      examples_ckb=("گووگڵ بکەرەوە", "یوتیوب لە براوزەر بکەرەوە"))
async def open_url(ctx: ToolContext, url: str, **_ignored: Any) -> dict[str, Any]:
    return _result(await _hands(ctx).web.open_url(url))


@tool("web_search",
      description="Search the web and return the results; open_in_browser also shows the search.",
      description_ckb="گەڕان لە ئینتەرنێت",
      params={"type": "object", "properties": {
          "query": {"type": "string"}, "open_in_browser": {"type": "boolean"}}, "required": ["query"]},
      risk="safe", blocking=True, timeout_s=45,
      examples_ckb=("لە ئینتەرنێت بگەڕێ بۆ هەواڵی زێڕ", "هەواڵی ئەمڕۆ بگەڕێ"))
async def web_search(ctx: ToolContext, query: str, open_in_browser: bool = False,
                     **_ignored: Any) -> dict[str, Any]:
    hands = _hands(ctx)
    result = await hands.web.search(query)
    if open_in_browser:
        from .web import Web

        await hands.web.open_url(Web.search_url(query))
        result["opened_in_browser"] = True
    return _result(result)


@tool("fetch_page",
      description="Read the text of a public web page (http/https).",
      description_ckb="خوێندنەوەی پەڕەی ماڵپەڕ",
      params={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
      classify=_url_risk, blocking=True, timeout_s=30,
      examples_ckb=("ئەم پەڕەیە بخوێنەرەوە و کورتی بکەرەوە",))
async def fetch_page(ctx: ToolContext, url: str, **_ignored: Any) -> dict[str, Any]:
    return _result(await _hands(ctx).web.fetch_page(url))


# -- projects / system -------------------------------------------------------------------------
@tool("build_project",
      description="Build a new website (default) or Python program from a description in ~/SAM Projects, open "
                  "it in VS Code and preview it in the browser (takes a minute or two).",
      description_ckb="دروستکردنی پرۆژە",
      params={"type": "object", "properties": {
          "description": {"type": "string", "description": "everything the user wants, in their words"},
          "name": {"type": "string"}, "kind": {"type": "string", "enum": ["website", "python", "other"]}},
          "required": ["description"]},
      # 600 s: generation has its own 540 s budget (hands.build_timeout_s) so a
      # slow free model still ends with an honest partial result, not a timeout.
      risk="safe", blocking=False, timeout_s=600,
      examples_ckb=("ماڵپەڕێکی پۆرتفۆلیۆ بۆم دروست بکە", "ماڵپەڕێک بۆ فرۆشتنی جل دروست بکە"))
async def build_project(ctx: ToolContext, description: str, name: str = "", kind: str = "website",
                        **_ignored: Any) -> dict[str, Any]:
    hands = _hands(ctx)
    return _result(await hands.code.build(description, name=name or None, kind=kind, progress=ctx.progress,
                                          cancel=ctx.cancel, source=ctx.source))


@tool("system_control",
      description="Computer status (info: time, battery, disks, memory, volume) and exact volume: set_volume "
                  "(level 0-100), volume_up/volume_down (by 10), mute, unmute.",
      description_ckb="باری کۆمپیوتەر و دەنگ",
      params={"type": "object", "properties": {
          "action": {"type": "string", "enum": ["info", "set_volume", "volume_up", "volume_down", "mute", "unmute"]},
          "level": {"type": "integer", "description": "0-100 for set_volume"}}, "required": ["action"]},
      risk="safe", blocking=True, timeout_s=20,
      examples_ckb=("باتریەکەم چەندە؟", "دەنگەکە بکە بە پەنجا"))
async def system_control(ctx: ToolContext, action: str, level: int | None = None,
                         **_ignored: Any) -> dict[str, Any]:
    system = _hands(ctx).system
    if action == "info":
        info = await asyncio.to_thread(system.info, str(ctx.app.config.get("app.timezone", "Asia/Baghdad")))
        return ok("Computer status read.", **info)
    try:
        if action == "set_volume":
            if level is None:
                return fail("set_volume needs a level from 0 to 100.")
            state = await asyncio.to_thread(system.set_volume, max(0, min(100, int(level))), None)
        elif action in ("volume_up", "volume_down"):
            current = await asyncio.to_thread(system.get_volume)
            step = 10 if action == "volume_up" else -10
            state = await asyncio.to_thread(system.set_volume, max(0, min(100, current["level"] + step)), False)
        else:
            state = await asyncio.to_thread(system.set_volume, None, action == "mute")
    except OSError as exc:
        return fail(f"Volume control failed: {exc}.")
    return ok(f"Volume is {state['level']}%" + (" and muted." if state["muted"] else "."), **state)


TOOLS = (open_app, window_control, type_text, press_keys, click, screen_look, screen_act, run_powershell, files,
         open_url, web_search, fetch_page, build_project, system_control)
# Risk classifiers are plain functions of the arguments (registry contract);
# the few that need live state (foreground app, last snapshot) find the app in
# guards.APP.
_APP = guards.APP


def register_tools(app: Any) -> list[str]:
    guards.APP["app"] = app
    return [app.tools.add(fn, owner="hands").name for fn in TOOLS]


__all__ = ["TOOLS", "register_tools"]
