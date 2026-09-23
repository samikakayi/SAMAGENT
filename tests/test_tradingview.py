"""The TradingView controller: what it observes, and what it refuses to do.

Everything here that touches the desktop goes through one seam --
`_modules()`, which hands back psutil and the four pywin32 modules -- so the
controller can be driven with a fake Windows behind it. That is what these
tests do: no real window, no OCR, no screen, but the controller's own
selection rules, fail-closed gates and state assembly run for real.

The contract these pin is "what does SAM believe about the chart, and when
does it refuse to act", because every drawing and calibration decision is
made from that belief.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sam_backend.trading.tradingview import TradingViewController, TradingViewState

TV_PROCESS = [{"pid": 100, "name": "TradingView.exe"}]


class FakeGui:
    """win32gui, with a desktop of whatever windows a test declares."""

    def __init__(self, windows, foreground=0, client=(1200, 800), fail_client=False):
        self.windows = windows
        self.foreground = foreground
        self.client = client
        self.fail_client = fail_client
        self.calls: list[tuple] = []
        self.iconic: set[int] = set()
        self.becomes_foreground_after = None

    def IsWindowVisible(self, hwnd):
        return next((w.get("visible", True) for w in self.windows if w["hwnd"] == hwnd), False)

    def GetWindowText(self, hwnd):
        return next((w["title"] for w in self.windows if w["hwnd"] == hwnd), "")

    def EnumWindows(self, callback, extra):
        for window in self.windows:
            callback(window["hwnd"], extra)

    def GetForegroundWindow(self):
        return self.foreground

    def GetWindowRect(self, hwnd):
        return (10, 20, 1210, 820)

    def GetClientRect(self, hwnd):
        if self.fail_client:
            raise OSError("invalid window handle")
        return (0, 0, *self.client)

    def ClientToScreen(self, hwnd, point):
        return (10, 20)

    def IsIconic(self, hwnd):
        return hwnd in self.iconic

    def ShowWindow(self, hwnd, flag):
        self.calls.append(("ShowWindow", hwnd))
        self.iconic.discard(hwnd)

    def BringWindowToTop(self, hwnd):
        self.calls.append(("BringWindowToTop", hwnd))

    def SetForegroundWindow(self, hwnd):
        self.calls.append(("SetForegroundWindow", hwnd))
        attempts = sum(1 for call in self.calls if call[0] == "SetForegroundWindow")
        if self.becomes_foreground_after is None or attempts >= self.becomes_foreground_after:
            self.foreground = hwnd


class FakeProcess:
    def __init__(self, pid_of):
        self.pid_of = pid_of

    def GetWindowThreadProcessId(self, hwnd):
        return (900, self.pid_of.get(hwnd, 0))


class FakeApi:
    def __init__(self, fail=False):
        self.fail = fail

    def MonitorFromWindow(self, hwnd, flag):
        if self.fail:
            raise OSError("no monitor")
        return 5

    def GetMonitorInfo(self, handle):
        return {"Device": r"\\.\DISPLAY1", "Monitor": (0, 0, 2560, 1440),
                "Work": (0, 0, 2560, 1400), "Flags": 1}


class FakeCon:
    SW_RESTORE = 9
    SW_SHOW = 5
    MONITOR_DEFAULTTONEAREST = 2
    VK_CONTROL = 0x11
    VK_RETURN = 0x0D
    KEYEVENTF_KEYUP = 0x0002
    CF_UNICODETEXT = 13


class FakePsutil:
    NoSuchProcess = type("NoSuchProcess", (Exception,), {})
    AccessDenied = type("AccessDenied", (Exception,), {})

    def __init__(self, processes):
        self.processes = processes

    def process_iter(self, fields):
        for entry in self.processes:
            if isinstance(entry, Exception):
                raise entry
            yield type("Proc", (), {"info": entry})()


def make_controller(tmp_path: Path, windows, processes=None, *, foreground=0,
                    api=None, gui=None, **permissions) -> TradingViewController:
    controller = TradingViewController(tmp_path, **permissions)
    fake_gui = gui or FakeGui(windows, foreground=foreground)
    fake_api = api or FakeApi()
    pid_of = {window["hwnd"]: window["pid"] for window in windows}
    controller._modules = staticmethod(lambda: (
        FakePsutil(TV_PROCESS if processes is None else processes),
        fake_api, FakeCon(), fake_gui, FakeProcess(pid_of),
    ))
    controller.gui = fake_gui
    return controller


# -- reading the window title --------------------------------------------------

@pytest.mark.parametrize("title, symbol, price", [
    ("XAUUSD 3412.50", "XAUUSD", 3412.5),
    ("BTCUSD 68,250.75", "BTCUSD", 68250.75),          # thousands separators
    ("EURUSD \u25bc 1.0842", "EURUSD", 1.0842),        # direction arrow
    ("xauusd 3412.50", "XAUUSD", 3412.5),              # lower case is normalised
    ("US30 38,500", "US30", 38500.0),
    ("XAUUSD 3412.50 · OANDA", "XAUUSD", 3412.5),   # symbol, price and feed
    ("Untitled", None, None),
    ("", None, None),
])
def test_the_window_title_yields_the_symbol_and_price_or_nothing(title, symbol, price):
    parsed = TradingViewController._parse_title(title)

    assert parsed["symbol"] == symbol
    assert parsed["price"] == price


def test_a_title_that_is_only_a_symbol_yields_nothing():
    """The pattern needs something after the symbol. Real chart titles always
    carry a price, so this is the shape of an unexpected title, and an
    unexpected title is answered with None rather than a guess."""
    assert TradingViewController._parse_title("XAUUSD")["symbol"] is None
    assert TradingViewController._parse_title("XAUUSD ")["symbol"] == "XAUUSD"


def test_a_feed_is_only_read_when_the_title_actually_names_one():
    """A feed is never inferred. The capture is deliberately loose, and the
    value is reported for display only -- nothing decides anything on it."""
    assert TradingViewController._parse_title("XAUUSD 3412.50")["feed"] is None
    assert TradingViewController._parse_title("XAUUSD 3412.50 \u00b7 OANDA")["feed"] == "OANDA"


# -- what SAM believes about the chart ------------------------------------------

def test_no_tradingview_at_all_is_reported_as_not_running(tmp_path):
    state = make_controller(tmp_path, [], processes=[]).observe()

    assert state.running is False and state.process_ids == []
    assert state.window_handle is None and state.interactive is False
    assert "no visible targetable chart window" in state.observations[0]


def test_tradingview_running_without_a_chart_window_is_distinguished(tmp_path):
    """Running but unusable is not the same as absent, and the state says which."""
    state = make_controller(tmp_path, []).observe()

    assert state.running is True and state.process_ids == [100]
    assert state.window_handle is None and state.interactive is False


def test_one_chart_window_produces_a_complete_observation(tmp_path):
    state = make_controller(tmp_path, [{"hwnd": 7, "pid": 100, "title": "XAUUSD 3412.50 \u00b7 OANDA"}],
                            foreground=7).observe()

    assert (state.window_handle, state.symbol, state.current_price) == (7, "XAUUSD", 3412.5)
    assert state.feed == "OANDA" and state.active is True and state.interactive is True
    assert state.window_geometry == {"left": 10, "top": 20, "right": 1210, "bottom": 820,
                                     "width": 1200, "height": 800}
    assert state.client_geometry["width"] == 1200
    assert state.monitor["device"] == r"\\.\DISPLAY1" and state.monitor["primary"] is True
    # The title never carries the interval; it is only ever verified by OCR.
    assert state.timeframe is None and state.timeframe_verified is False
    assert any("does not expose the selected timeframe" in note for note in state.observations)


def test_the_foreground_window_is_chosen_and_the_ambiguity_is_declared(tmp_path):
    controller = make_controller(tmp_path, [
        {"hwnd": 7, "pid": 100, "title": "AAAAAAAAAAAAAAAA 1.0"},
        {"hwnd": 9, "pid": 100, "title": "BBB 2.0"},
    ], foreground=9)

    state = controller.observe()

    assert state.window_handle == 9, "the window the user is looking at wins"
    assert any("2 TradingView windows are open" in note for note in state.observations)


def test_with_nothing_in_front_the_largest_title_is_taken_as_the_chart(tmp_path):
    state = make_controller(tmp_path, [
        {"hwnd": 7, "pid": 100, "title": "AAA 1.0"},
        {"hwnd": 9, "pid": 100, "title": "BBBBBBBBBBBBBB 2.0"},
    ], foreground=0).observe()

    assert state.window_handle == 9 and state.active is False


@pytest.mark.parametrize("window, why", [
    ({"hwnd": 7, "pid": 999, "title": "Notepad"}, "another process's window"),
    ({"hwnd": 7, "pid": 100, "title": "XAUUSD 1.0", "visible": False}, "a hidden window"),
    ({"hwnd": 7, "pid": 100, "title": "   "}, "a window with no title"),
])
def test_a_window_that_is_not_a_visible_tradingview_chart_is_never_selected(tmp_path, window, why):
    assert make_controller(tmp_path, [window]).observe().window_handle is None, why


def test_an_unreadable_client_area_does_not_lose_the_whole_observation(tmp_path):
    """Drawing prefers the client area and falls back to the frame, so the frame
    must survive even when the client rect cannot be read."""
    gui = FakeGui([{"hwnd": 7, "pid": 100, "title": "XAUUSD 1.0"}], foreground=7, fail_client=True)

    state = make_controller(tmp_path, gui.windows, gui=gui).observe()

    assert state.client_geometry is None
    assert state.window_geometry is not None and state.window_handle == 7


def test_an_unreadable_monitor_does_not_lose_the_observation(tmp_path):
    state = make_controller(tmp_path, [{"hwnd": 7, "pid": 100, "title": "XAUUSD 1.0"}],
                            api=FakeApi(fail=True)).observe()

    assert state.monitor is None and state.window_handle == 7


def test_a_process_that_vanishes_mid_scan_is_skipped_not_fatal(tmp_path):
    """psutil raises for a process that exits while it is being enumerated."""
    controller = make_controller(tmp_path, [{"hwnd": 7, "pid": 100, "title": "XAUUSD 1.0"}])
    original = controller._modules()

    class Vanishing(FakePsutil):
        def process_iter(self, fields):
            yield type("Proc", (), {"info": {"pid": 100, "name": "TradingView.exe"}})()
            raise self.NoSuchProcess("gone")

    controller._modules = staticmethod(lambda: (Vanishing([]), *original[1:]))

    with pytest.raises(FakePsutil.NoSuchProcess):
        controller._windows()


def test_an_unavailable_desktop_fails_closed_rather_than_guessing(tmp_path):
    """No Windows, no pywin32, no answer -- and the reason is carried out."""
    controller = TradingViewController(tmp_path)
    controller._modules = staticmethod(
        lambda: (_ for _ in ()).throw(RuntimeError("TradingView Desktop control is available only on Windows"))
    )

    state = controller.observe()

    assert state.running is False and state.interactive is False
    assert state.window_handle is None
    assert "only on Windows" in state.observations[0]


def test_an_observation_describes_exactly_one_window(tmp_path):
    """Geometry, title and monitor must all come from the handle that was picked;
    a mixed observation would calibrate one chart and draw on another."""
    seen: list[int] = []
    gui = FakeGui([{"hwnd": 7, "pid": 100, "title": "AAA 1.0"},
                   {"hwnd": 9, "pid": 100, "title": "BBBBBBBB 2.0"}], foreground=9)
    original_rect = gui.GetWindowRect
    gui.GetWindowRect = lambda hwnd: (seen.append(hwnd), original_rect(hwnd))[1]
    controller = make_controller(tmp_path, gui.windows, gui=gui)
    api_seen: list[int] = []
    controller._modules().__class__  # touch, keeps the tuple alive
    original_api = FakeApi.MonitorFromWindow
    FakeApi.MonitorFromWindow = lambda self, hwnd, flag: (api_seen.append(hwnd), 5)[1]
    try:
        state = controller.observe()
    finally:
        FakeApi.MonitorFromWindow = original_api

    assert state.window_handle == 9
    assert seen == [9] and api_seen == [9], "every read used the selected handle"


def test_the_state_serialises_for_the_api(tmp_path):
    payload = make_controller(tmp_path, [{"hwnd": 7, "pid": 100, "title": "XAUUSD 3412.50"}],
                              foreground=7).observe().as_dict()

    assert payload["symbol"] == "XAUUSD" and payload["window_handle"] == 7
    assert payload["timeframe_verified"] is False
    assert isinstance(payload["observations"], list)


# -- bringing the chart forward ---------------------------------------------------

def test_focus_is_refused_without_computer_control(tmp_path):
    controller = make_controller(tmp_path, [{"hwnd": 7, "pid": 100, "title": "XAUUSD 1.0"}],
                                 computer_control=False)

    result = controller.focus()

    assert result.error_code == "COMPUTER_CONTROL_DISABLED" and not result.executed
    assert controller.gui.calls == []


def test_focus_without_a_window_says_so(tmp_path):
    controller = make_controller(tmp_path, [], computer_control=True)

    assert controller.focus().error_code == "WINDOW_NOT_FOUND"


def test_a_chart_already_in_front_is_not_disturbed(tmp_path):
    controller = make_controller(tmp_path, [{"hwnd": 7, "pid": 100, "title": "XAUUSD 1.0"}],
                                 foreground=7, computer_control=True)

    result = controller.focus()

    assert result.verified and result.status.value == "SUCCESS"
    assert controller.gui.calls == [], "no window was raised; it was already there"


def test_a_background_chart_is_raised_and_the_result_is_confirmed(tmp_path):
    controller = make_controller(tmp_path, [{"hwnd": 7, "pid": 100, "title": "XAUUSD 1.0"}],
                                 foreground=0, computer_control=True)

    result = controller.focus()

    assert result.verified and result.data["window_handle"] == 7
    assert ("SetForegroundWindow", 7) in controller.gui.calls


def test_a_minimised_chart_is_restored_before_being_raised(tmp_path):
    gui = FakeGui([{"hwnd": 7, "pid": 100, "title": "XAUUSD 1.0"}], foreground=0)
    gui.iconic.add(7)
    controller = make_controller(tmp_path, gui.windows, gui=gui, computer_control=True)

    assert controller.focus().verified
    assert gui.calls[0] == ("ShowWindow", 7), "restore comes first"


def test_the_foreground_lock_is_retried_and_a_late_success_still_counts(tmp_path, monkeypatch):
    """Windows refuses the first requests while another process holds the lock."""
    monkeypatch.setattr("sam_backend.trading.tradingview.time.sleep", lambda seconds: None)
    gui = FakeGui([{"hwnd": 7, "pid": 100, "title": "XAUUSD 1.0"}], foreground=0)
    gui.becomes_foreground_after = 3
    controller = make_controller(tmp_path, gui.windows, gui=gui, computer_control=True)

    result = controller.focus()

    assert result.verified
    assert sum(1 for call in gui.calls if call[0] == "SetForegroundWindow") == 3


def test_a_window_that_never_comes_forward_is_reported_not_claimed(tmp_path, monkeypatch):
    monkeypatch.setattr("sam_backend.trading.tradingview.time.sleep", lambda seconds: None)
    gui = FakeGui([{"hwnd": 7, "pid": 100, "title": "XAUUSD 1.0"}], foreground=0)
    gui.becomes_foreground_after = 99
    controller = make_controller(tmp_path, gui.windows, gui=gui, computer_control=True)

    result = controller.focus()

    assert result.executed and not result.verified
    assert result.status.value == "PARTIAL"
    assert "did not confirm" in result.error
    assert sum(1 for call in gui.calls if call[0] == "SetForegroundWindow") == 3, "bounded retries"


def test_a_raise_that_raises_is_reported_as_a_focus_failure(tmp_path):
    gui = FakeGui([{"hwnd": 7, "pid": 100, "title": "XAUUSD 1.0"}], foreground=0)

    def explode(hwnd):
        raise OSError("window died")

    gui.ShowWindow = explode
    controller = make_controller(tmp_path, gui.windows, gui=gui, computer_control=True)

    result = controller.focus()

    assert result.error_code == "FOCUS_FAILED" and result.executed


# -- the toolbar, and the screen -------------------------------------------------

def test_the_toolbar_band_is_taken_from_the_client_area_when_there_is_one(tmp_path):
    controller = make_controller(tmp_path, [{"hwnd": 7, "pid": 100, "title": "XAUUSD 1.0"}])
    state = controller.observe()

    band = controller._toolbar_band(state)

    left, top, right, bottom = band
    assert left == state.client_geometry["left"]
    assert top == state.client_geometry["top"] + TradingViewController.TOOLBAR_TOP_INSET
    assert bottom - top == TradingViewController.TOOLBAR_HEIGHT
    assert right - left <= TradingViewController.TOOLBAR_WIDTH


def test_without_geometry_there_is_no_toolbar_to_read(tmp_path):
    controller = make_controller(tmp_path, [])
    blank = TradingViewState(False, [], None, None, None, None, None, False, None, None, None, False, False)

    assert controller._toolbar_band(blank) is None
    words, meta = controller._read_toolbar_words(blank)
    assert words == [] and meta["reason"] == "no chart geometry"


def test_the_interval_is_never_claimed_without_screen_access(tmp_path):
    controller = make_controller(tmp_path, [{"hwnd": 7, "pid": 100, "title": "XAUUSD 1.0"}],
                                 screen_access=False)

    picked, meta = controller._read_toolbar_timeframe(controller.observe())

    assert picked is None and "Screen Access is off" in meta["reason"]


def test_an_ocr_failure_leaves_the_interval_unknown_rather_than_wrong(tmp_path, monkeypatch):
    controller = make_controller(tmp_path, [{"hwnd": 7, "pid": 100, "title": "XAUUSD 1.0"}],
                                 screen_access=True)
    monkeypatch.setattr(controller, "_read_toolbar_words",
                        lambda state: ([], {"reason": "toolbar could not be read: screen locked"}))

    picked, meta = controller._read_toolbar_timeframe(controller.observe())

    assert picked is None and "could not be read" in meta["reason"]


def test_an_empty_toolbar_reading_picks_no_interval(tmp_path, monkeypatch):
    """OCR that returns nothing is not an error, but it is not an answer either."""
    controller = make_controller(tmp_path, [{"hwnd": 7, "pid": 100, "title": "XAUUSD 1.0"}],
                                 screen_access=True)
    monkeypatch.setattr(controller, "_read_toolbar_words", lambda state: ([], {"band": {}, "tokens": []}))

    picked, meta = controller._read_toolbar_timeframe(controller.observe())

    assert picked is None and meta["picked"] is None


def test_the_interval_is_read_from_the_toolbar_when_ocr_can_see_it(tmp_path, monkeypatch):
    controller = make_controller(tmp_path, [{"hwnd": 7, "pid": 100, "title": "XAUUSD 1.0"}],
                                 screen_access=True)
    words = [type("W", (), {"text": text, "center_x": x})() for text, x in
             (("XAUUSD", 40.0), ("15", 420.0), ("Indicators", 600.0))]
    monkeypatch.setattr(controller, "_read_toolbar_words", lambda state: (words, {}))

    picked, meta = controller._read_toolbar_timeframe(controller.observe())

    assert picked == "M15", "the raw toolbar token is normalised to SAM's vocabulary"
    assert meta["picked"] == "M15"


@pytest.mark.parametrize("method, code", [
    ("capture", "SCREEN_ACCESS_DISABLED"),
])
def test_capture_needs_screen_access(tmp_path, method, code):
    controller = make_controller(tmp_path, [{"hwnd": 7, "pid": 100, "title": "XAUUSD 1.0"}],
                                 screen_access=False)

    assert getattr(controller, method)().error_code == code


def test_capture_without_a_window_captures_nothing(tmp_path):
    controller = make_controller(tmp_path, [], screen_access=True)

    assert controller.capture().error_code == "WINDOW_NOT_FOUND"


def test_a_capture_is_written_once_and_identified_by_its_own_bytes(tmp_path, monkeypatch):
    from PIL import Image

    controller = make_controller(tmp_path, [{"hwnd": 7, "pid": 100, "title": "XAUUSD 1.0"}],
                                 screen_access=True)
    grabbed: list[tuple] = []

    def grab(bbox, all_screens=False):
        grabbed.append(bbox)
        return Image.new("RGB", (bbox[2] - bbox[0], bbox[3] - bbox[1]), "black")

    monkeypatch.setattr("PIL.ImageGrab.grab", grab)

    result = controller.capture()

    assert result.verified and grabbed == [(10, 20, 1210, 820)], "the observed window rectangle"
    written = Path(result.data["path"])
    assert written.exists() and written.parent.name == "screenshots"
    assert result.data["width"] == 1200 and result.data["height"] == 800
    import hashlib
    assert result.data["sha256"] == hashlib.sha256(written.read_bytes()).hexdigest()


def test_a_screen_that_cannot_be_grabbed_fails_closed(tmp_path, monkeypatch):
    controller = make_controller(tmp_path, [{"hwnd": 7, "pid": 100, "title": "XAUUSD 1.0"}],
                                 screen_access=True)

    def refuse(bbox, all_screens=False):
        raise OSError("the screen is locked")

    monkeypatch.setattr("PIL.ImageGrab.grab", refuse)

    result = controller.capture()

    assert result.error_code == "SCREEN_CAPTURE_FAILED" and result.executed
    assert not (tmp_path / "screenshots").exists() or not list((tmp_path / "screenshots").iterdir())


# -- navigation gates --------------------------------------------------------------

def test_an_interval_tradingview_has_no_shortcut_for_is_refused_before_acting(tmp_path):
    controller = make_controller(tmp_path, [{"hwnd": 7, "pid": 100, "title": "XAUUSD 1.0"}],
                                 foreground=7, computer_control=True)

    result = controller.set_timeframe("M7")

    assert result.error_code == "INVALID_TIMEFRAME" and not result.executed
    assert controller.gui.calls == []


def test_navigation_is_refused_when_the_chart_cannot_be_focused(tmp_path):
    controller = make_controller(tmp_path, [], computer_control=True)

    assert controller.set_timeframe("M15").error_code == "WINDOW_NOT_FOUND"
    assert controller.set_symbol("EURUSD").error_code in {"WINDOW_NOT_FOUND", "COMPUTER_CONTROL_DISABLED"}


def test_navigation_is_refused_without_computer_control(tmp_path):
    controller = make_controller(tmp_path, [{"hwnd": 7, "pid": 100, "title": "XAUUSD 1.0"}],
                                 computer_control=False)

    assert controller.set_timeframe("M15").error_code == "COMPUTER_CONTROL_DISABLED"
    assert controller.set_symbol("EURUSD").error_code == "COMPUTER_CONTROL_DISABLED"
    assert controller.launch().error_code == "COMPUTER_CONTROL_DISABLED"


def test_launching_when_a_chart_is_already_open_just_focuses_it(tmp_path):
    controller = make_controller(tmp_path, [{"hwnd": 7, "pid": 100, "title": "XAUUSD 1.0"}],
                                 foreground=0, computer_control=True)

    result = controller.launch()

    assert result.verified and result.data["window_handle"] == 7
    assert ("SetForegroundWindow", 7) in controller.gui.calls


def test_a_verified_interval_is_remembered_for_that_window_only(tmp_path):
    """The interval is per window handle: two charts do not share one answer."""
    controller = make_controller(tmp_path, [{"hwnd": 7, "pid": 100, "title": "XAUUSD 1.0"}],
                                 foreground=7)
    controller._last_verified_timeframe[7] = "M15"

    state = controller.observe()

    assert state.timeframe == "M15" and state.timeframe_verified is True
    assert any("verified from the TradingView toolbar" in note for note in state.observations)

    controller._last_verified_timeframe.clear()
    controller._last_verified_timeframe[9] = "H1"
    assert controller.observe().timeframe is None, "another window's interval is not this one's"


# -- confirming what the chart is showing -------------------------------------------

@pytest.mark.parametrize("shown, wanted, matches", [
    ("XAUUSD", "XAUUSD", True),
    ("XAUUSD", "xauusd", True),                # case is not a mismatch
    ("OANDA:XAUUSD", "XAUUSD", True),          # a feed-qualified symbol still matches
    ("XAUUSD", "EURUSD", False),
    (None, "XAUUSD", False),                   # nothing on screen is never a match
])
def test_the_chart_symbol_is_confirmed_or_the_mismatch_is_named(tmp_path, shown, wanted, matches):
    title = f"{shown} 1.0" if shown else "Untitled"
    controller = make_controller(tmp_path, [{"hwnd": 7, "pid": 100, "title": title}])

    result = controller.verify_symbol(wanted)

    assert result.verified is matches
    if not matches:
        assert result.error_code == "SYMBOL_MISMATCH"
        assert wanted.upper() in result.error


def test_an_interval_must_read_the_same_twice_before_it_is_believed(tmp_path, monkeypatch):
    """One OCR pass can catch a button mid-animation or under a tooltip."""
    monkeypatch.setattr("sam_backend.trading.tradingview.time.sleep", lambda seconds: None)
    controller = make_controller(tmp_path, [{"hwnd": 7, "pid": 100, "title": "XAUUSD 1.0"}],
                                 screen_access=True)
    readings = iter(["M15", "H1", "M15", "M15"])
    monkeypatch.setattr(controller, "_read_toolbar_timeframe",
                        lambda state: (next(readings, None), {"picked": "x"}))

    state, meta = controller._confirm_toolbar_timeframe("M15", timeout_seconds=5.0, poll_interval=0.0)

    assert state is not None and state.window_handle == 7
    # The single M15 was not enough; the run of two that followed was.


def test_an_interval_that_never_settles_is_not_confirmed(tmp_path, monkeypatch):
    monkeypatch.setattr("sam_backend.trading.tradingview.time.sleep", lambda seconds: None)
    controller = make_controller(tmp_path, [{"hwnd": 7, "pid": 100, "title": "XAUUSD 1.0"}],
                                 screen_access=True)
    flapping = iter(["M15", "H1"] * 20)
    monkeypatch.setattr(controller, "_read_toolbar_timeframe",
                        lambda state: (next(flapping, None), {}))

    state, _meta = controller._confirm_toolbar_timeframe("M15", timeout_seconds=0.05, poll_interval=0.0)

    assert state is None, "a flapping reading is never accepted as confirmation"


def test_an_interval_already_showing_is_accepted_without_typing_anything(tmp_path, monkeypatch):
    monkeypatch.setattr("sam_backend.trading.tradingview.time.sleep", lambda seconds: None)
    controller = make_controller(tmp_path, [{"hwnd": 7, "pid": 100, "title": "XAUUSD 1.0"}],
                                 foreground=7, computer_control=True, screen_access=True)
    monkeypatch.setattr(controller, "_read_toolbar_timeframe", lambda state: ("M15", {"tokens": ["15"]}))
    typed: list[str] = []
    monkeypatch.setattr(controller, "_send_text_and_enter", lambda text: typed.append(text))

    result = controller.set_timeframe("M15")

    assert result.verified and typed == [], "nothing was sent; the toolbar already agreed"
    assert result.data["timeframe"] == "M15" and result.data["timeframe_verified"] is True
    assert controller._last_verified_timeframe[7] == "M15"
    assert any("already shows M15" in note for note in result.observations)


# -- typing into the chart ------------------------------------------------------------

def test_text_is_typed_character_by_character_and_committed_with_enter(tmp_path, monkeypatch):
    monkeypatch.setattr("sam_backend.trading.tradingview.time.sleep", lambda seconds: None)
    controller = make_controller(tmp_path, [{"hwnd": 7, "pid": 100, "title": "XAUUSD 1.0"}])
    typed: list[str] = []
    monkeypatch.setattr(TradingViewController, "_send_unicode", staticmethod(typed.append))
    keys: list[tuple] = []
    api = controller._modules()[1]
    api.keybd_event = lambda code, scan, flags, extra: keys.append((code, flags))

    controller._send_text_and_enter("15")

    assert typed == ["1", "5"], "each character goes in independently of the keyboard layout"
    assert len(keys) == 2 and keys[0][0] == FakeCon.VK_RETURN, "then Enter, down and up"


def test_pasting_replaces_the_field_and_gives_the_clipboard_back(tmp_path, monkeypatch):
    """TradingView is Electron and ignores synthetic key events, so text goes
    via a real Ctrl+V -- which means borrowing the user's clipboard."""
    monkeypatch.setattr("sam_backend.trading.tradingview.time.sleep", lambda seconds: None)
    controller = make_controller(tmp_path, [{"hwnd": 7, "pid": 100, "title": "XAUUSD 1.0"}])
    clipboard = ["the user's own text"]
    monkeypatch.setattr(TradingViewController, "_clipboard_read", staticmethod(lambda: clipboard[-1]))
    written: list[str] = []

    def write(text):
        written.append(text)
        clipboard.append(text)
        return True

    monkeypatch.setattr(TradingViewController, "_clipboard_write", staticmethod(write))
    keys: list[int] = []
    controller._modules()[1].keybd_event = lambda code, scan, flags, extra: keys.append(code)

    assert controller._paste_text("EURUSD") is True

    assert written == ["EURUSD", "the user's own text"], "borrowed, then handed back"
    # Ctrl down, key down, key up, Ctrl up -- for A then V, in that order.
    assert keys == [FakeCon.VK_CONTROL, 0x41, 0x41, FakeCon.VK_CONTROL,
                    FakeCon.VK_CONTROL, 0x56, 0x56, FakeCon.VK_CONTROL]


def test_a_clipboard_that_cannot_be_written_stops_the_paste(tmp_path, monkeypatch):
    monkeypatch.setattr("sam_backend.trading.tradingview.time.sleep", lambda seconds: None)
    controller = make_controller(tmp_path, [{"hwnd": 7, "pid": 100, "title": "XAUUSD 1.0"}])
    monkeypatch.setattr(TradingViewController, "_clipboard_read", staticmethod(lambda: None))
    monkeypatch.setattr(TradingViewController, "_clipboard_write", staticmethod(lambda text: False))
    keys: list[int] = []
    controller._modules()[1].keybd_event = lambda code, scan, flags, extra: keys.append(code)

    assert controller._paste_text("EURUSD") is False
    assert keys == [], "nothing was sent to a field whose contents could not be set"


def test_a_clipboard_with_no_text_in_it_is_not_restored_over(tmp_path, monkeypatch):
    """If the user held an image, SAM must not replace it with its own string."""
    monkeypatch.setattr("sam_backend.trading.tradingview.time.sleep", lambda seconds: None)
    controller = make_controller(tmp_path, [{"hwnd": 7, "pid": 100, "title": "XAUUSD 1.0"}])
    monkeypatch.setattr(TradingViewController, "_clipboard_read", staticmethod(lambda: None))
    written: list[str] = []
    monkeypatch.setattr(TradingViewController, "_clipboard_write",
                        staticmethod(lambda text: written.append(text) or True))
    controller._modules()[1].keybd_event = lambda *a: None

    controller._paste_text("EURUSD")

    assert written == ["EURUSD"], "nothing was written back, because nothing text-like was taken"
