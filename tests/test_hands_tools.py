"""The hands tools as the model sees them: names, parameters, risk and
blocking mode from CONTRACTS.md section 2, risk decided by code, and
dispatch through the registry with fake Windows/input."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from sam.hands.input import Input
from tests.hands_helpers import FakeClipboard, RecorderBackend, fake_windows, win

CONTRACT = {  # name -> (required params, blocking)
    "open_app": ({"name"}, True), "window_control": ({"action"}, True), "type_text": ({"text"}, True),
    "press_keys": ({"keys"}, True), "click": ({"target"}, True), "screen_look": (set(), True),
    "screen_act": ({"goal"}, False), "run_powershell": ({"command"}, True), "files": ({"action", "path"}, True),
    "open_url": ({"url"}, True), "web_search": ({"query"}, True), "build_project": ({"description"}, False),
}


@pytest.fixture
def app(make_app):
    app = make_app()
    status = app.load_packages(["sam.hands"])
    assert status == {"sam.hands": "ok"}, app.failed
    return app


def test_every_contract_tool_is_registered_with_its_shape(app) -> None:
    for name, (required, blocking) in CONTRACT.items():
        spec = app.tools.get(name)
        assert spec is not None, name
        assert set(spec.params.get("required", [])) == required, name
        assert spec.blocking is blocking, name
        assert spec.examples_ckb and spec.description_ckb, name
        assert spec.owner == "hands"
    assert app.tools.get("window_control").params["properties"]["action"]["enum"] == [
        "list", "focus", "minimize", "maximize", "restore", "close", "snap_left", "snap_right"]
    assert app.tools.get("screen_look").params["properties"]["mode"]["enum"] == ["controls", "text", "describe"]
    assert set(app.tools.get("files").params["properties"]["action"]["enum"]) >= {
        "list", "read", "write", "append", "copy", "move", "delete", "open", "search", "reveal"}
    # Gemini + OpenAI schemas build from the same definitions
    names = {t["function"]["name"] for t in app.tools.openai_tools()}
    assert set(CONTRACT) <= names
    declarations = {d.name: d for d in app.tools.gemini_declarations(live=True)}
    assert str(declarations["screen_act"].behavior).endswith("NON_BLOCKING")


def install_fakes(app, windows_list: list, foreground: int | None = None):
    windows, api = fake_windows(windows_list, foreground=foreground)
    backend, clipboard = RecorderBackend(), FakeClipboard([(13, "user\x00".encode("utf-16-le"))])
    app.hands.windows = windows
    app.hands.input = Input(backend, clipboard, sleep=lambda s: None, paste_settle_s=0.0)

    async def focused_text() -> tuple[str, str | None]:
        return "editor", clipboard_text.get("typed")
    clipboard_text: dict[str, Any] = {}
    app.hands.uia.focused_text = focused_text  # type: ignore[method-assign]
    return api, backend, clipboard, clipboard_text


def test_risk_is_decided_by_code(app) -> None:
    install_fakes(app, [win(1, "Chat - Telegram", "Telegram.exe")], foreground=1)
    risk = app.tools.risk_of
    # "routine" = ordinary: no question while the user gives SAM full authority (2026-09-25), else asked
    assert risk("window_control", {"action": "close", "target": "کرۆم"}) == ("routine", "پەنجەرەی «کرۆم» دابخەم؟")
    assert risk("window_control", {"action": "minimize"})[0] == "safe"
    assert risk("press_keys", {"keys": "alt+f4"})[0] == "routine"
    assert risk("press_keys", {"keys": "ctrl+shift+delete"})[0] == "confirm"    # clears data for good
    assert risk("press_keys", {"keys": "shift+delete"})[0] == "confirm"         # deletes for good
    assert risk("press_keys", {"keys": "win+l"})[0] == "routine"
    assert risk("press_keys", {"keys": "ctrl+s"})[0] == "safe"
    assert risk("press_keys", {"keys": "enter"})[0] == "confirm"          # Telegram is in front: sends
    assert risk("type_text", {"text": "سڵاو", "press_enter": True})[0] == "confirm"
    assert risk("type_text", {"text": "سڵاو"})[0] == "safe"
    assert risk("click", {"target": "Send"})[0] == "confirm"
    assert risk("click", {"target": "بیسڕەوە"})[0] == "confirm"
    assert risk("click", {"target": "Save"})[0] == "safe"
    assert risk("run_powershell", {"command": "Get-Date"})[0] == "safe"
    assert risk("run_powershell", {"command": "Remove-Item a.txt"})[0] == "confirm"     # permanent deletion
    assert risk("run_powershell", {"command": "Copy-Item a.txt b.txt"})[0] == "routine"
    assert risk("run_powershell", {"command": "Get-Content .env"})[0] == "blocked"
    assert risk("open_url", {"url": "file:///C:/x"})[0] == "blocked"
    assert risk("open_url", {"url": "https://example.com"})[0] == "safe"
    assert risk("files", {"action": "read", "path": "Desktop/a.txt"})[0] == "safe"
    assert risk("files", {"action": "delete", "path": "Desktop/a.txt"})[0] == "routine"      # to the Recycle Bin
    assert risk("files", {"action": "delete", "path": "Desktop"})[0] == "blocked"


def test_numbered_click_uses_the_label_from_the_last_snapshot(app) -> None:
    from sam.hands.uia import Control

    install_fakes(app, [win(1, "Mail", "chrome.exe")], foreground=1)
    app.hands.uia.controls = [Control(1, "Send", "button", (0, 0, 10, 10), True)]
    assert app.tools.risk_of("click", {"target": "1"})[0] == "confirm"


async def test_press_keys_goes_to_the_work_window_not_sams_panel(app) -> None:
    from tests.hands_helpers import OWN_PID

    api, backend, _, _ = install_fakes(app, [win(1, "Doc - Notepad", "Notepad.exe"),
                                             win(2, "SAM", "python.exe", pid=OWN_PID)], foreground=2)
    result = await app.tools.dispatch("press_keys", {"keys": "ctrl+s"})
    assert result["ok"], result
    assert api.fg == 1  # focused Notepad before pressing
    assert backend.events == [("key", 0x11, "down"), ("key", 0x53, "down"), ("key", 0x53, "up"), ("key", 0x11, "up")]


async def test_type_text_pastes_restores_and_verifies(app) -> None:
    api, backend, clipboard, typed = install_fakes(app, [win(1, "Doc - Notepad", "Notepad.exe")], foreground=1)
    typed["typed"] = "سڵاو لە سام"
    result = await app.tools.dispatch("type_text", {"text": "سڵاو لە سام"})
    assert result["ok"] and result["data"]["verified"] and result["data"]["clipboard_restored"]
    assert clipboard.get_text() == "user"


async def test_window_control_list_and_unknown_target(app) -> None:
    install_fakes(app, [win(1, "Doc - Notepad", "Notepad.exe"), win(2, "News - Google Chrome", "chrome.exe")],
                  foreground=1)
    result = await app.tools.dispatch("window_control", {"action": "list"})
    assert result["ok"] and len(result["data"]["untrusted"]) == 2
    result = await app.tools.dispatch("window_control", {"action": "minimize", "target": "Photoshop"})
    assert not result["ok"] and "No open window" in result["summary"]
    result = await app.tools.dispatch("window_control", {"action": "minimize", "target": "کرۆم"})
    assert result["ok"] and "Minimized" in result["summary"]


async def test_close_waits_for_the_users_answer(app) -> None:
    api, *_ = install_fakes(app, [win(1, "Doc - Notepad", "Notepad.exe")], foreground=1)
    task = asyncio.ensure_future(app.tools.dispatch("window_control", {"action": "close", "target": "notepad"}))
    await asyncio.sleep(0.05)
    assert app.confirm.has_pending and api.state[1]["alive"]
    app.confirm.offer_transcript("نەخێر")
    result = await task
    assert not result["ok"] and result["data"]["declined"] and api.state[1]["alive"]


async def test_a_save_prompt_is_numbered_so_the_user_can_answer_it(app) -> None:
    from tests.test_hands_uia import FakeUiaBackend

    api, *_ = install_fakes(app, [win(1, "Untitled - Paint", "mspaint.exe")], foreground=1)
    api.close_behavior[1] = "prompt"            # WM_CLOSE shows "save changes?" inside the window
    backend = FakeUiaBackend([
        {"name": "Save", "type": 50000, "rect": (10, 10, 60, 30), "enabled": True},
        {"name": "Don’t save", "type": 50000, "rect": (70, 10, 120, 30), "enabled": True},
        {"name": "Cancel", "type": 50000, "rect": (130, 10, 180, 30), "enabled": True},
        {"name": "Brushes", "type": 50000, "rect": (0, 40, 30, 60), "enabled": True}])
    app.hands.uia._backend = backend
    task = asyncio.ensure_future(app.tools.dispatch("window_control", {"action": "close", "target": "Paint"}))
    await asyncio.sleep(0.05)
    app.confirm.offer_transcript("بەڵێ")
    result = await task
    assert not result["ok"] and "prompt" in result["summary"]
    assert result["data"]["untrusted"]["dialog_buttons"][:3] == ["1. button: Save", "2. button: Don’t save",
                                                               "3. button: Cancel"]
    clicked = await app.tools.dispatch("click", {"target": "2"})   # the user said "don't save"
    assert clicked["ok"] and backend.invoked == ["Don’t save"]


async def test_blocked_powershell_never_runs(app, monkeypatch) -> None:
    import sam.hands.shell as shell

    async def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("must not run")
    monkeypatch.setattr(shell, "run_powershell", boom)
    result = await app.tools.dispatch("run_powershell", {"command": "mimikatz"})
    assert not result["ok"] and result["data"]["blocked"]


async def test_run_powershell_output_is_redacted_untrusted_data(app, monkeypatch) -> None:
    import sam.hands.shell as shell
    from tests.conftest import FAKE_GROQ

    async def fake_run(command: str, **kwargs: Any) -> Any:
        return shell.ShellResult(exit_code=0, stdout=f"key is {FAKE_GROQ}\nسڵاو", stderr="")
    monkeypatch.setattr(shell, "run_powershell", fake_run)
    result = await app.tools.dispatch("run_powershell", {"command": "Get-Date"})
    assert result["ok"]
    stdout = result["data"]["untrusted"]["stdout"]
    assert FAKE_GROQ not in stdout and "سڵاو" in stdout


async def test_system_control_reports_the_read_back_volume(app, monkeypatch) -> None:
    from sam.hands import system

    state = {"level": 40, "muted": False}

    def fake_set(level: int | None = None, mute: bool | None = None) -> dict[str, Any]:
        if level is not None:
            state["level"] = level
        if mute is not None:
            state["muted"] = mute
        return dict(state)
    monkeypatch.setattr(system, "set_volume", fake_set)
    monkeypatch.setattr(system, "get_volume", lambda: dict(state))
    result = await app.tools.dispatch("system_control", {"action": "set_volume", "level": "50"})
    assert result["ok"] and result["data"]["level"] == 50
    result = await app.tools.dispatch("system_control", {"action": "volume_up"})
    assert result["data"]["level"] == 60
    result = await app.tools.dispatch("system_control", {"action": "mute"})
    assert result["data"]["muted"]


async def test_click_can_scroll_a_control_without_asking(app) -> None:
    from sam.hands.uia import Control

    api, backend, _, _ = install_fakes(app, [win(1, "Files - Explorer", "explorer.exe")], foreground=1)
    app.hands.uia.controls = [Control(1, "Delete", "button", (0, 0, 10, 10), True),
                              Control(2, "Items View", "list item", (100, 200, 900, 1000), True)]
    app.hands.uia.last_hwnd = 1
    assert app.tools.risk_of("click", {"target": "1", "button": "scroll_down"})[0] == "safe"
    assert app.tools.risk_of("click", {"target": "1"})[0] == "confirm"
    result = await app.tools.dispatch("click", {"target": "2", "button": "scroll_down"})
    assert result["ok"] and "down" in result["summary"], result
    assert ("wheel", -5 * 120) in backend.events and backend.pos == (500, 600)
