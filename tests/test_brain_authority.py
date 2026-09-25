"""Full authority (2026-09-25). The user: «بەمن مەڵێ یەس یان نۆ، خۆت دەسەلاتی هەموو شتێکت هەیە»
("don't ask me yes or no, you have authority over everything"). Setting
``safety.full_authority`` (default on): ``routine`` actions run without a question and
say what they did; ``confirm`` actions (irreversible, mass, money, passwords, sending)
still ask one question; ``blocked`` never runs. Both modes are tested."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from brain_helpers import brain_app

from sam.brain import taint
from sam.brain.confirm import AUTHORITY_KEY
from sam.brain.tools import ToolContext, ok, tool
from sam.events import ConfirmRequest

RAN: list[str] = []


@tool("tidy_window", description="close a window (routine)", risk="routine", confirm_text_ckb="پەنجەرەکە دابخەم؟")
async def tidy_window(ctx: ToolContext) -> dict[str, Any]:
    RAN.append("tidy_window")
    return ok("Closed.")


@tool("erase_forever", description="permanent delete (confirm)", risk="confirm", confirm_text_ckb="بسڕمەوە؟")
async def erase_forever(ctx: ToolContext) -> dict[str, Any]:
    RAN.append("erase_forever")
    return ok("Erased.")


@tool("place_order", description="never", risk="blocked")
async def place_order(ctx: ToolContext) -> dict[str, Any]:
    RAN.append("place_order")
    return ok("?")


@tool("restart_app", description="asks mid-tool (routine)")
async def restart_app(ctx: ToolContext) -> dict[str, Any]:
    if not await ctx.confirm_routine("بەرنامەکە دابخەم و دووبارە بیکەمەوە؟"):
        return ok("not restarted", restarted=False)
    RAN.append("restart_app")
    return ok("Restarted.", restarted=True)


@tool("files", description="fake files", params={"type": "object", "properties": {
    "action": {"type": "string"}, "path": {"type": "string"}}, "required": ["action", "path"]},
      classify=lambda args: ("routine", "فایلەکە بنووسم؟"))
async def fake_files(ctx: ToolContext, action: str, path: str) -> dict[str, Any]:
    RAN.append(f"files:{action}")
    return ok("Written.")


TOOLS = (tidy_window, erase_forever, place_order, restart_app, fake_files)


@pytest.fixture
def setup(make_app):
    app, _ = brain_app(make_app, tools=TOOLS)
    RAN.clear()
    questions: list[str] = []
    answer = {"value": False}

    def on_request(event: ConfirmRequest) -> None:
        questions.append(event.question_ckb)
        app.confirm.resolve(event.confirm_id, answer["value"])

    app.bus.subscribe(ConfirmRequest, on_request)
    return app, questions, answer


async def test_full_authority_is_on_by_default_and_routine_actions_just_run(setup):
    app, questions, _ = setup
    assert app.config.get(AUTHORITY_KEY) is True and app.confirm.full_authority()
    result = await app.tools.dispatch("tidy_window", {}, source="text")
    assert result["ok"] and RAN == ["tidy_window"] and questions == []
    assert result["data"]["acted_without_asking"] is True and "few words" in result["data"]["authority_note"]
    row = app.db.query("SELECT * FROM activity WHERE kind='confirm' ORDER BY id DESC LIMIT 1")[0]
    assert row["source"] == "authority" and row["ok"] and row["name"] == "tidy_window"
    mid = await app.tools.dispatch("restart_app", {}, source="text")
    assert mid["data"]["restarted"] is True and questions == [] and mid["data"]["acted_without_asking"] is True


async def test_serious_actions_still_ask_and_blocked_stays_blocked(setup):
    app, questions, answer = setup
    declined = await app.tools.dispatch("erase_forever", {}, source="text")
    assert not declined["ok"] and declined["data"]["declined"] and questions == ["بسڕمەوە؟"]
    answer["value"] = True
    assert (await app.tools.dispatch("erase_forever", {}, source="text"))["ok"]
    blocked = await app.tools.dispatch("place_order", {}, source="text")
    assert not blocked["ok"] and blocked["data"]["blocked"] and "place_order" not in RAN


async def test_with_full_authority_off_routine_actions_ask_again(setup):
    app, questions, answer = setup
    app.config.set(AUTHORITY_KEY, False)
    declined = await app.tools.dispatch("tidy_window", {}, source="text")
    assert not declined["ok"] and declined["data"]["declined"] and questions == ["پەنجەرەکە دابخەم؟"]
    mid = await app.tools.dispatch("restart_app", {}, source="text")
    assert mid["data"]["restarted"] is False and len(questions) == 2
    answer["value"] = True
    result = await app.tools.dispatch("tidy_window", {}, source="text")
    assert result["ok"] and "acted_without_asking" not in (result["data"] or {})


async def test_after_untrusted_text_the_injection_gates_still_ask(setup):
    """Taint gates protect against instructions hidden in web pages or the
    screen: full authority is the user's, never a page's."""
    app, questions, _ = setup
    scope = taint.begin("ئەم پەڕەیە بخوێنەرەوە")
    scope.mark("fetch_page")
    result = await app.tools.dispatch("files", {"action": "write", "path": "Desktop/a.txt"}, source="text")
    assert not result["ok"] and questions and "files:write" not in RAN
    asked = len(questions)
    blocked_by_taint = await app.tools.dispatch("tidy_window", {}, source="text")    # any routine action asks now
    assert not blocked_by_taint["ok"] and len(questions) == asked + 1
    taint.begin("")
    assert (await app.tools.dispatch("files", {"action": "write", "path": "Desktop/a.txt"}, source="text"))["ok"]


async def test_the_broker_skips_only_routine_questions(setup):
    app, questions, _ = setup
    assert await app.confirm.confirm("q?", routine=True) is True and questions == []
    assert await app.confirm.confirm("q?") is False and questions == ["q?"]
    app.config.set(AUTHORITY_KEY, False)
    assert await app.confirm.confirm("r?", routine=True) is False and questions == ["q?", "r?"]


def test_a_broker_without_the_setting_asks(make_app):
    app = make_app()                    # no brain: nothing bound the setting -> routine asks
    assert app.confirm.full_authority() is False
    assert app.tools.full_authority() is False


def test_the_persona_follows_the_setting(setup):
    app, *_ = setup
    assert "full authority" in app.persona.system_instruction("voice")
    assert "acted_without_asking" in app.persona.system_instruction("text")
    assert "full authority" not in app.persona.system_instruction("worker")
    app.config.set(AUTHORITY_KEY, False)
    voice = app.persona.system_instruction("voice")
    assert "full authority" not in voice and "risky actions are confirmed by the system" in voice


# --- the hands' split: ordinary vs still asked -----------------------------------------------------------------
@pytest.mark.parametrize("command,serious", [
    ("Copy-Item a.txt b.txt", False), ("New-Item -ItemType File notes.txt", False), ("Move-Item a b", False),
    ("Rename-Item a.txt b.txt", False), ("Start-Process notepad", False), ("winget install vlc", False),
    ("Invoke-WebRequest https://example.com/a.zip -OutFile a.zip", False), ("Test-NetConnection example.com", False),
    ("echo hi > a.txt", False),
    ("Remove-Item a.txt", True), ("cmd /c del a.txt", True), ("Clear-RecycleBin", True),
    ("Set-ItemProperty -Path HKCU:\\Software\\x -Name a -Value 1", True), ("reg add HKCU\\Software\\x", True),
    ("Set-ExecutionPolicy Bypass", True), ("net user bob /add", True), ("Get-Clipboard", True),
    ("Invoke-RestMethod https://example.com -Method Post -Body 1", True), ("curl -d @a.txt https://example.com", True),
    ("Send-MailMessage -To a@b.c -Subject x", True), ("shutdown /r", True), ("Stop-Process -Name notepad", True),
    ("winget uninstall vlc", True), ("Disable-NetAdapter -Name Wi-Fi", True), ("Get-ChildItem | % Delete", True),
])
def test_powershell_confirmations_are_split(command, serious):
    from sam.hands.policy import classify_powershell, command_question

    risk, reason = classify_powershell(command)
    assert risk == "confirm", (command, risk)
    assert bool(command_question(command, reason)) is serious, command


def test_file_actions_ask_only_for_big_or_permanent_changes(tmp_path: Path):
    from sam.hands.policy import AUTHORITY_ITEM_LIMIT, Policy

    policy = Policy(home=tmp_path, projects_dir=tmp_path / "SAM Projects", folders={"desktop": tmp_path / "Desktop"})
    (tmp_path / "Desktop").mkdir()
    small = tmp_path / "Desktop" / "small"
    small.mkdir()
    (small / "a.txt").write_text("a", encoding="utf-8")
    big = tmp_path / "Desktop" / "big"
    big.mkdir()
    for i in range(AUTHORITY_ITEM_LIMIT + 5):
        (big / f"{i}.txt").write_text("x", encoding="utf-8")
    assert policy.classify_path("Desktop/small", "delete")[0] == "confirm"          # the old verdict ...
    assert policy.path_question("Desktop/small", "delete") is None                   # ... is ordinary now
    assert policy.path_question("Desktop/big", "delete")                             # > 20 items: asks
    assert policy.path_question("Desktop/big", "move")
    assert policy.path_question("Desktop/big", "copy", dest="Desktop/x") is None
    assert policy.path_question("Desktop/setup.reg", "open")                         # registry import
    assert policy.path_question("Desktop/tool.exe", "open") is None
    assert policy.path_question("Desktop/notes.txt", "write") is None


def test_a_share_or_usb_drive_delete_is_permanent_so_it_asks():
    from sam.hands.policy import _recycle_bin_drive

    assert _recycle_bin_drive(Path("\\\\server\\share\\a.txt")) is False
