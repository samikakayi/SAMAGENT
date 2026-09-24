"""Global hotkey: chord parsing, and a REAL RegisterHotKey round trip on a
chord nobody uses (Ctrl+Shift+Alt+F24 -- F24 is not on physical keyboards).
The press is simulated by posting WM_HOTKEY to the hotkey thread, so no key
is pressed and no low-level keyboard hook is installed."""

from __future__ import annotations

import sys
import threading

import pytest

from sam.voice.hotkey import MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN, GlobalHotkey, parse_hotkey


def test_parse_hotkey_chords():
    assert parse_hotkey("ctrl+alt+space") == (MOD_CONTROL | MOD_ALT, 0x20)
    assert parse_hotkey(" Win + Alt + Space ") == (MOD_WIN | MOD_ALT, 0x20)
    assert parse_hotkey("ctrl+shift+alt+f24") == (MOD_CONTROL | MOD_SHIFT | MOD_ALT, 0x87)
    assert parse_hotkey("ctrl+k") == (MOD_CONTROL, ord("K"))
    assert parse_hotkey("alt+`") == (MOD_ALT, 0xC0)


@pytest.mark.parametrize("bad", ["", "space", "ctrl+alt", "ctrl+a+b", "ctrl+ژ", "ctrl+nosuchkey"])
def test_parse_hotkey_rejects_bad_chords(bad):
    with pytest.raises(ValueError):
        parse_hotkey(bad)


def test_bad_chord_is_reported_not_raised():
    hotkey = GlobalHotkey("ctrl+alt", lambda: None)
    assert hotkey.start() is False and hotkey.error and not hotkey.registered


@pytest.mark.skipif(sys.platform != "win32", reason="RegisterHotKey is Windows only")
def test_real_register_press_and_release():
    pressed = threading.Event()
    hotkey = GlobalHotkey("ctrl+shift+alt+f24", pressed.set)
    try:
        assert hotkey.start(), hotkey.error
        assert hotkey.registered
        # A second registration of the same chord is refused (what happens
        # when another program owns the chord, e.g. Ctrl+Alt+Space here).
        rival = GlobalHotkey("ctrl+shift+alt+f24", lambda: None)
        assert rival.start() is False and "another program" in (rival.error or "")
        rival.stop()
        assert hotkey.simulate_press()
        assert pressed.wait(2.0) and hotkey.presses == 1
    finally:
        hotkey.stop()
    assert not hotkey.registered
    again = GlobalHotkey("ctrl+shift+alt+f24", lambda: None)   # released: free again
    try:
        assert again.start()
    finally:
        again.stop()
