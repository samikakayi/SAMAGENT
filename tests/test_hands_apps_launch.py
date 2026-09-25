"""open_app launching: the right Windows mechanism per AppID shape, reuse of
an open window, verification by window, and TradingView delegation."""

from __future__ import annotations

from typing import Any

from sam.hands.apps import AppIndex
from tests.hands_helpers import START_ROWS, fake_windows, win


class Recorder:
    def __init__(self, api: Any = None, spawn: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.api = api
        self.spawn = spawn or {}

    def _maybe_spawn(self, key: str) -> None:
        window = self.spawn.get(key)
        if window is not None and self.api is not None:
            self.api.add(window)

    def activate(self, aumid: str, args: str) -> int:
        self.calls.append(("activate", aumid, args))
        self._maybe_spawn(aumid)
        return 4242

    def startfile(self, target: str) -> None:
        self.calls.append(("startfile", target))
        self._maybe_spawn(target)

    def popen(self, argv: list[str], **kwargs: Any) -> Any:
        self.calls.append(("popen", *argv))
        self.popen_env = kwargs.get("env")
        self._maybe_spawn(argv[0])

        class Proc:
            pid = 5151
        return Proc()


async def make(make_app, windows_list: list, spawn: dict[str, Any] | None = None, *, foreground: int | None = None,
               load: bool = True):
    app = make_app()
    if load:
        app.load_packages(["sam.hands"])
    windows, api = fake_windows(windows_list, foreground=foreground)
    rec = Recorder(api, spawn)
    index = AppIndex(app, windows=windows, enumerate_fn=lambda: [dict(r) for r in START_ROWS],
                     activate_fn=rec.activate, startfile_fn=rec.startfile, popen_fn=rec.popen)
    await index.refresh()
    app.config.set("hands.launch_wait_s", 1)
    return app, index, rec, api


async def test_packaged_app_is_activated_and_verified_by_its_window(make_app) -> None:
    notepad = win(50, "Untitled - Notepad", "Notepad.exe", pid=4242)
    app, index, rec, api = await make(make_app, [], {"Microsoft.WindowsNotepad_8wekyb3d8bbwe!App": notepad})
    result = await index.launch("نۆتپاد")
    assert result["ok"] and result["state"] == "started" and result["window"] == "Untitled - Notepad"
    assert rec.calls == [("activate", "Microsoft.WindowsNotepad_8wekyb3d8bbwe!App", "")]
    assert api.fg == 50


async def test_registered_desktop_app_uses_the_start_menu_id(make_app) -> None:
    chrome = win(60, "New Tab - Google Chrome", "chrome.exe", pid=77)
    app, index, rec, _ = await make(make_app, [], {"shell:AppsFolder\\Chrome": chrome})
    result = await index.launch("کرۆم بکەرەوە")
    assert result["ok"] and rec.calls == [("startfile", "shell:AppsFolder\\Chrome")]


async def test_arguments_start_the_target_exe_directly(make_app) -> None:
    exe = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    chrome = win(61, "Example - Google Chrome", "chrome.exe", pid=5151)
    app, index, rec, _ = await make(make_app, [], {exe: chrome})
    result = await index.launch("Chrome", "--incognito https://example.com")
    assert result["ok"]
    assert rec.calls[0] == ("popen", exe, "--incognito", "https://example.com")


async def test_started_apps_never_inherit_electron_host_variables(make_app, monkeypatch) -> None:
    import os

    monkeypatch.setenv("ELECTRON_RUN_AS_NODE", "1")
    monkeypatch.setenv("VSCODE_ESM_ENTRYPOINT", "vs/workbench/api/node/extensionHostProcess")
    exe = r"C:\Program Files\MetaTrader 5\terminal64.exe"
    app, index, rec, _ = await make(make_app, [], {exe: win(62, "MetaTrader 5", "terminal64.exe", pid=5151)})
    # Loading hands removed them from SAM's own environment: ShellExecute-based
    # launches (os.startfile, shell:AppsFolder) inherit that environment.
    assert "ELECTRON_RUN_AS_NODE" not in os.environ and "VSCODE_ESM_ENTRYPOINT" not in os.environ
    monkeypatch.setenv("ELECTRON_RUN_AS_NODE", "1")  # set again later: direct starts still drop it
    assert (await index.launch("MetaTrader 5"))["ok"]
    assert rec.popen_env is not None and "ELECTRON_RUN_AS_NODE" not in rec.popen_env and "PATH" in rec.popen_env


async def test_known_folder_exe_is_started_by_path(make_app) -> None:
    exe = r"C:\Program Files\MetaTrader 5\terminal64.exe"
    mt5 = win(62, "MetaTrader 5", "terminal64.exe", pid=5151)
    app, index, rec, _ = await make(make_app, [], {exe: mt5})
    result = await index.launch("مێتاترەیدەر")
    assert result["ok"] and rec.calls == [("popen", exe)]


async def test_an_open_window_is_focused_instead_of_launching_again(make_app) -> None:
    telegram = win(70, "Telegram", "Telegram.exe", pid=88, minimized=True)
    app, index, rec, api = await make(make_app, [telegram])
    result = await index.launch("تێلێگرام")
    assert result["ok"] and result["state"] == "focused"
    assert rec.calls == [] and api.fg == 70 and not api.state[70]["min"]
    result = await index.launch("تێلێگرام", new_window=True)
    assert rec.calls == [("startfile", "shell:AppsFolder\\Telegram.TelegramDesktop")]


async def test_no_window_and_no_process_is_reported_as_failure(make_app, monkeypatch) -> None:
    from sam.hands import _win

    monkeypatch.setattr(_win, "running_processes", lambda: {})
    monkeypatch.setattr(_win, "pid_alive", lambda pid: False)
    app, index, rec, _ = await make(make_app, [])
    result = await index.launch("Excel")
    assert not result["ok"] and result["state"] == "failed"


async def test_unknown_app_is_not_found_with_suggestions(make_app) -> None:
    app, index, rec, _ = await make(make_app, [])
    result = await index.launch("Photoshop")
    assert not result["ok"] and result["state"] == "not_found" and rec.calls == []


async def test_url_apps_open_in_the_browser(make_app) -> None:
    app, index, rec, _ = await make(make_app, [])
    result = await index.launch("یوتیوب")
    assert result["ok"] and rec.calls == [("startfile", "https://www.youtube.com")]


class FakeTv:
    """The chart bridge's ensure_running: asks ``confirm`` before a restart,
    exactly like sam.trading.tradingview.TradingViewBridge."""

    def __init__(self, state: str = "started", *, running_without_port: bool = False) -> None:
        self.state = state
        self.running_without_port = running_without_port
        self.calls: list[dict[str, Any]] = []

    async def ensure_running(self, *, allow_restart: bool = False, confirm: Any = None,
                             focus: bool = True) -> dict[str, Any]:
        self.calls.append({"allow_restart": allow_restart, "confirm": confirm, "focus": focus})
        if self.running_without_port:
            if not allow_restart:
                return {"ok": False, "state": "needs_restart", "detail": "no port"}
            if confirm is not None and not await confirm("TradingView دابخەم و دووبارە بیکەمەوە؟"):
                return {"ok": False, "state": "needs_restart", "declined": True, "detail": "declined"}
            return {"ok": True, "state": "restarted", "detail": ""}
        return {"ok": self.state != "failed", "state": self.state, "detail": f"TradingView {self.state}"}


async def test_tradingview_is_delegated_to_the_chart_bridge(make_app) -> None:
    tv_window = win(80, "GOLD / TradingView", "TradingView.exe", pid=90)
    app, index, rec, api = await make(make_app, [tv_window])
    app.trading.tv = FakeTv("connected")
    result = await index.launch("ترەیدینگ ڤیو")
    assert result["ok"] and result["state"] == "delegated" and result["tv_state"] == "connected"
    assert rec.calls == []                                   # never started by hands itself
    assert app.trading.tv.calls == [{"allow_restart": False, "confirm": None, "focus": True}]


async def test_tradingview_without_its_port_is_restarted_only_after_the_user_agrees(make_app) -> None:
    tv_window = win(80, "GOLD / TradingView", "TradingView.exe", pid=90)
    app, index, rec, api = await make(make_app, [tv_window])
    asked: list[str] = []

    async def say_no(question: str, detail: str = "") -> bool:
        asked.append(question)
        return False

    async def say_yes(question: str, detail: str = "") -> bool:
        asked.append(question)
        return True

    app.trading.tv = FakeTv(running_without_port=True)
    result = await index.launch("تریدینگ ڤیو", confirm=say_no)
    assert not result["ok"] and result["declined"] and result["tv_state"] == "needs_restart"
    assert app.trading.tv.calls[-1]["allow_restart"] is True and len(asked) == 1
    result = await index.launch("TradingView", confirm=say_yes)
    assert result["ok"] and result["tv_state"] == "restarted" and "Restarted" in result["summary"]
    # nobody to ask (no confirm callback): never restarted behind the user's back
    result = await index.launch("TradingView")
    assert not result["ok"] and app.trading.tv.calls[-1]["allow_restart"] is False and len(asked) == 2


async def test_open_app_tool_passes_the_users_confirmation_to_the_bridge(make_app) -> None:
    tv_window = win(80, "GOLD / TradingView", "TradingView.exe", pid=90)
    app, index, rec, api = await make(make_app, [tv_window])
    app.hands.apps = index
    app.trading.tv = FakeTv(running_without_port=True)
    import asyncio

    task = asyncio.ensure_future(app.tools.dispatch("open_app", {"name": "ترەیدینگ ڤیو بکەرەوە"}, source="text"))
    for _ in range(100):
        if app.confirm.has_pending:
            break
        await asyncio.sleep(0.01)
    assert app.confirm.has_pending                           # the question reached the user (card + voice)
    app.confirm.offer_transcript("بەڵێ")
    result = await task
    assert result["ok"] and result["data"]["tv_state"] == "restarted", result


async def test_tools_ignore_arguments_they_do_not_declare(make_app) -> None:
    tv_window = win(80, "GOLD / TradingView", "TradingView.exe", pid=90)
    app, index, rec, api = await make(make_app, [tv_window])
    app.hands.apps = index
    app.trading.tv = FakeTv("connected")
    # Gemini sometimes adds a 'reason'; Live dispatches it unfiltered.
    result = await app.tools.dispatch("open_app", {"name": "TradingView", "reason": "user asked"}, source="live")
    assert result["ok"], result


async def test_tradingview_without_the_bridge_still_gets_the_debug_port(make_app) -> None:
    tv_window = win(81, "TradingView", "TradingView.exe", pid=4242)
    aumid = "TradingView.Desktop_n534cwy3pjxzj!TradingView.Desktop"
    app, index, rec, _ = await make(make_app, [], {aumid: tv_window})
    assert app.trading.tv is None
    result = await index.launch("تریدینگ ڤیو")
    assert result["ok"]
    assert rec.calls == [("activate", aumid, "--remote-debugging-port=9222")]
