"""Acceptance fixes 2026-09-24 (hands): every ForEach-Object member-call form
the verify review used to delete files while "Read-only command.", and the
trading click guard judging the window the click goes to (not the one in
front), plus 'Algo Trading' / Ctrl+E."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from hands_helpers import fake_windows, win

from sam.hands import guards
from sam.hands.policy import classify_powershell, foreach_members


@pytest.mark.parametrize("command,risk", [
    # verify3/ps_quoted_proof.py: each of these deleted files as 'safe'
    (r"gci C:\scratch -Recurse -File | % 'Delete'", "blocked"),
    (r"gci C:\scratch -Recurse -File | % -Mem Delete", "blocked"),
    (r"gci C:\scratch -Recurse -File | % -Me Delete", "blocked"),
    (r"gci C:\scratch -Recurse -File | ForEach-Object -MemberName:Delete", "blocked"),
    (r"gci C:\scratch -Recurse -File | ForEach-Object -MemberName:'Delete'", "blocked"),
    (r"$m='Delete'; gci ~\Documents -Recurse -File | % $m", "blocked"),
    (r"gci ~\Documents -Recurse -File | % 'MoveTo' 'C:\Users\Public\x'", "blocked"),
    (r"Get-Process | % 'Kill'", "confirm"),
    # the same through abbreviations, expressions, splatting and nested shells
    (r"gci -Rec | % Delete", "blocked"),
    (r"gci -Depth 5 | % Delete", "blocked"),
    (r"gci -r | % ('Del'+'ete')", "blocked"),
    (r"gci -r | % $('Delete')", "blocked"),
    (r"$p=@{MemberName='Delete'}; gci -r | % @p", "blocked"),
    (r"gci -r | %{ $_.psobject.Methods['Delete'].Invoke() }", "blocked"),
    (r"(gci -r -file).ForEach('Delete')", "blocked"),
    (r"powershell -c 'gci -r | % Delete'", "blocked"),
    (r'pwsh -Command "Get-ChildItem ~ -Recurse | % Delete"', "blocked"),
    (r"Get-ChildItem C:\Users\x\Documents -File | % Delete", "blocked"),        # a whole main folder, no -Recurse
    (r"Remove-Item ~\Documents\*", "blocked"),
    (r"Remove-Item C:\Users\samit\Desktop\*.*", "blocked"),
    (r"[IO.Directory]::Delete('C:\Users\x\Documents', $true)", "blocked"),
    (r"(Get-Item C:\Users\x\Documents).Delete($true)", "blocked"),
    (r"foreach ($f in gci -r) { $f.Delete() }", "blocked"),
    # still asked, not blocked
    (r"Get-Item C:\Users\x\Desktop\important.docx | % Delete", "confirm"),
    (r"Remove-Item C:\Users\samit\Desktop\old\*", "confirm"),
    (r"powershell -c 'Get-Date'", "confirm"),
    # still read-only
    (r"gci -r | % -Process { $_ } -End { }", "safe"),
    (r"gci -r | % { $_.FullName }", "safe"),
    (r"gci -Recurse -Filter *.log | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) }", "safe"),
    (r"Get-Service | % Name", "safe"),
    (r"Get-Process | ConvertTo-Json -Depth 2", "safe"),
    (r"5 % 2", "safe"),
])
def test_foreach_member_calls_in_every_form(command, risk):
    assert classify_powershell(command)[0] == risk, classify_powershell(command)


def test_foreach_members_parses_the_forms():
    assert foreach_members("gci | % -membername:'delete'") == ["delete"]
    assert foreach_members("gci | foreach-object -begin {1} -process {2} -end {3}") == []
    assert foreach_members("foreach ($x in $y) { $x }") == []
    assert foreach_members("gci | % -erroraction silentlycontinue name") == ["name"]


# -- the trading click guard judges the click's own window --------------------------------------------------------
MT5_TITLE = "12345678 - InfinoxLimited-MT5Demo: Demo Account - Hedge - Infinox Limited"


@pytest.fixture
def desktop():
    """Chrome in front, MetaTrader 5 behind it (verify3/click_probe2.py)."""
    chrome = win(700001, "Gold news - Google Chrome", "chrome.exe", pid=11)
    mt5 = win(700002, MT5_TITLE, "terminal64.exe", pid=12)
    windows, _api = fake_windows([chrome, mt5], foreground=chrome.hwnd)
    windows.own_pid = 99
    uia = SimpleNamespace(label_of=lambda target: str(target), last_hwnd=None)
    previous = guards.APP.get("app")
    guards.APP["app"] = SimpleNamespace(hands=SimpleNamespace(windows=windows, uia=uia))
    yield SimpleNamespace(uia=uia, chrome=chrome, mt5=mt5)
    guards.APP["app"] = previous


@pytest.mark.parametrize("window", ["مێتاتڕەیدەر", "MetaTrader", "Infinox", "terminal64", "metatrader 5"])
def test_an_order_click_aimed_at_mt5_is_blocked_whatever_is_in_front(desktop, window):
    assert guards.click_risk({"target": "Close Position", "window": window})[0] == "blocked"


def test_a_numbered_click_is_judged_by_the_window_of_its_snapshot(desktop):
    desktop.uia.last_hwnd = desktop.mt5.hwnd                  # screen_look was taken on MT5
    desktop.uia.label_of = lambda target: "Close Position" if str(target) == "3" else str(target)
    assert guards.click_risk({"target": "3"})[0] == "blocked"
    desktop.uia.last_hwnd = desktop.chrome.hwnd
    assert guards.click_risk({"target": "3"})[0] != "blocked"


@pytest.mark.parametrize("label", ["Algo Trading", "AutoTrading", "Expert Advisors", "ئەلگۆ ترەیدینگ"])
def test_automated_trading_switches_are_blocked_in_mt5(desktop, label):
    assert guards.click_risk({"target": label, "window": "MetaTrader"})[0] == "blocked"


def test_ctrl_e_is_blocked_in_mt5_only(desktop):
    guards.APP["app"].hands.windows._api.set_foreground(desktop.mt5.hwnd)          # noqa: SLF001
    assert guards.keys_risk({"keys": "ctrl+e"})[0] == "blocked"
    guards.APP["app"].hands.windows._api.set_foreground(desktop.chrome.hwnd)       # noqa: SLF001
    assert guards.keys_risk({"keys": "ctrl+e"})[0] == "safe"


def test_ordinary_clicks_elsewhere_are_not_blocked(desktop):
    assert guards.click_risk({"target": "Close Position"})[0] != "blocked"          # a web page's text, Chrome
    assert guards.click_risk({"target": "Reload", "window": "Chrome"})[0] == "safe"


# -- grounded search leaves a resting Gemini alone ------------------------------------------------------------------
async def test_web_search_skips_gemini_while_it_rests(make_app):
    import time

    import httpx
    from test_hands_web import DDG_HTML, make_web

    from tests.conftest import FAKE_GEMINI

    class Refusing:
        def __init__(self) -> None:
            self.calls = 0

        async def generate_content(self, **kwargs):
            self.calls += 1
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

    models = Refusing()
    genai = SimpleNamespace(aio=SimpleNamespace(models=models))
    web, app = make_web(make_app, lambda request: httpx.Response(200, text=DDG_HTML), gemini=FAKE_GEMINI, genai=genai)
    first = await web.search("gold price")
    second = await web.search("silver price")
    assert first["engine"] == second["engine"] == "duckduckgo"
    assert models.calls == 1                                 # the failed model rests; the next search goes to DDG
    web._gemini_rest_until = 0.0                             # noqa: SLF001
    app.llm._cooldown["gemini:gemini-3.5-flash-lite"] = time.monotonic() + 60     # noqa: SLF001
    await web.search("oil price")
    assert models.calls == 1                                 # resting in the LLM client too: not asked
