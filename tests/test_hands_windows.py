"""Window operations against a fake window manager: finding by name
(English/Sorani/"this"), verified actions, protections (ported from v1
windows_control: verify after every action, never act on SAM itself)."""

from __future__ import annotations

from sam.hands.windows import SW_MAXIMIZE, SW_MINIMIZE
from tests.hands_helpers import OWN_PID, fake_windows, win

CHROME = win(1, "Gold price - Google Chrome", "chrome.exe", pid=10)
NOTEPAD = win(2, "*notes - Notepad", "Notepad.exe", pid=20, minimized=True)
TV = win(3, "GOLD ▼ 4,255 / TradingView", "TradingView.exe", pid=30, maximized=True)
SAM = win(4, "SAM", "python.exe", pid=OWN_PID)
DESKTOP = win(5, "Program Manager", "explorer.exe", pid=40, cls="Progman")


def setup(foreground: int = 1):
    return fake_windows([CHROME, NOTEPAD, TV, SAM, DESKTOP], foreground=foreground)


async def test_list_hides_sam_and_the_desktop() -> None:
    windows, _ = setup()
    titles = [w.title for w in await windows.list()]
    assert "SAM" not in titles and "Program Manager" not in titles
    assert len(titles) == 3


async def test_find_by_app_alias_title_and_this() -> None:
    windows, _ = setup(foreground=1)
    assert (await windows.find("کرۆم")).hwnd == 1
    assert (await windows.find("chrome")).hwnd == 1
    assert (await windows.find("نۆتپاد")).hwnd == 2
    assert (await windows.find("ترەیدینگ ڤیو")).hwnd == 3
    assert (await windows.find("notes")).hwnd == 2
    assert (await windows.find("ئەم پەنجەرەیە")).hwnd == 1
    assert (await windows.find("")).hwnd == 1
    assert await windows.find("Photoshop") is None
    assert (await windows.find(3)).hwnd == 3


async def test_foreground_skips_sams_own_panel() -> None:
    # The user typed the command into SAM's panel: act on the window below it.
    windows, _ = setup(foreground=4)
    fg = await windows.foreground()
    assert fg is not None and fg.pid != OWN_PID and not fg.minimized
    assert windows.foreground_sync().hwnd == fg.hwnd


async def test_minimize_maximize_restore_are_verified() -> None:
    windows, api = setup()
    result = await windows.act("minimize", await windows.find("chrome"))
    assert result["ok"] and api.state[1]["min"]
    assert ("show", 1, SW_MINIMIZE) in api.calls
    result = await windows.act("maximize", await windows.find("notes"))
    assert result["ok"] and api.state[2]["max"]
    assert ("show", 2, SW_MAXIMIZE) in api.calls
    assert api.fg == 2  # a maximized window is also brought to the front
    result = await windows.act("restore", await windows.find("tradingview"))
    assert result["ok"] and not api.state[3]["max"]


async def test_focus_restores_a_minimized_window_and_reports_refusal() -> None:
    windows, api = setup()
    assert await windows.focus(2)
    assert api.fg == 2 and not api.state[2]["min"]
    api.refuse_focus.add(1)
    result = await windows.act("focus", await windows.find("chrome"))
    assert not result["ok"] and "did not let" in result["summary"]


async def test_close_verified_and_save_prompt_reported() -> None:
    windows, api = setup()
    result = await windows.act("close", await windows.find("chrome"))
    assert result["ok"] and not api.state[1]["alive"]
    api.close_behavior[2] = "prompt"
    result = await windows.act("close", await windows.find("notes"))
    assert not result["ok"]
    assert result["pending_dialog"] == ["Notepad"]
    assert "save" in result["summary"]


async def test_sam_and_shell_windows_are_protected() -> None:
    windows, api = setup()
    sam = next(w for w in await windows.list(include_own=True) if w.pid == OWN_PID)
    result = await windows.act("close", sam)
    assert not result["ok"] and "SAM's own" in result["summary"]
    assert ("close", 4) not in api.calls
    result = await windows.act("minimize", DESKTOP)
    assert not result["ok"]


async def test_snap_fills_the_half_including_invisible_borders() -> None:
    windows, api = setup()
    result = await windows.act("snap_left", await windows.find("chrome"))
    assert result["ok"]
    move = next(c for c in api.calls if c[0] == "move")
    # work area 2880 wide -> left half 0..1440, grown by the 11 px borders
    assert move[2] == (-11, 0, 1451, 1739)
    assert result["rect"] == [0, 0, 1440, 1728]
    result = await windows.act("snap_right", await windows.find("tradingview"))
    assert result["ok"] and result["rect"] == [1440, 0, 2880, 1728]
    assert not api.state[3]["max"]  # a maximized window is restored first


async def test_snap_reports_a_window_that_refuses_the_size() -> None:
    windows, api = setup()
    api.min_width[1] = 2000
    result = await windows.act("snap_left", await windows.find("chrome"))
    assert not result["ok"] and "different size" in result["summary"]


async def test_contract_shorthands_find_the_window_by_name() -> None:
    windows, api = fake_windows([win(1, "Doc - Notepad", "Notepad.exe"), win(2, "News - Google Chrome", "chrome.exe")],
                                foreground=1)
    assert (await windows.minimize("کرۆم"))["ok"] and api.state[2]["min"]
    assert (await windows.restore(2))["ok"] and not api.state[2]["min"]
    assert (await windows.maximize("notepad"))["ok"] and api.state[1]["max"]
    assert (await windows.snap("chrome", side="right"))["ok"]
    assert not (await windows.minimize("Photoshop"))["ok"]
    assert (await windows.close("notepad"))["ok"] and not api.state[1]["alive"]
