"""Keyboard shortcuts by virtual key (layout independent, v1
desktop_input lesson) and clipboard paste that restores the user's clipboard."""

from __future__ import annotations

import pytest

from sam.hands.input import Input, describe_chord, key_code, parse_keys
from tests.hands_helpers import FakeClipboard, RecorderBackend


def make(clipboard: FakeClipboard | None = None) -> tuple[Input, RecorderBackend, FakeClipboard]:
    backend = RecorderBackend()
    clip = clipboard or FakeClipboard([(13, "old text\x00".encode("utf-16-le")), (49161, b"\x01\x02rich")])
    return Input(backend, clip, sleep=lambda s: None, paste_settle_s=0.0), backend, clip


def test_letters_use_fixed_virtual_keys_on_any_layout() -> None:
    # VkKeyScan under a Kurdish/Arabic layout returns -1 for Latin letters;
    # SAM never asks the layout.
    assert key_code("s") == 0x53 and key_code("S") == 0x53
    assert key_code("7") == 0x37
    assert key_code("enter") == 0x0D and key_code("ئینتەر") == 0x0D
    assert key_code("F5") == 0x74 and key_code("volume_up") == 0xAF and key_code("play pause") == 0xB3
    assert key_code("plus") == 0xBB and key_code("-") == 0xBD
    with pytest.raises(ValueError):
        key_code("س")  # a Sorani letter is not a shortcut key


def test_parse_chords_and_sequences() -> None:
    assert parse_keys("ctrl+s") == [[0x11, 0x53]]
    assert parse_keys("ctrl+shift+esc") == [[0x11, 0x10, 0x1B]]
    assert parse_keys("ctrl+a, delete") == [[0x11, 0x41], [0x2E]]
    assert parse_keys("alt+tab alt+tab") == [[0x12, 0x09], [0x12, 0x09]]
    assert parse_keys("Win+D") == [[0x5B, 0x44]]
    with pytest.raises(ValueError):
        parse_keys("  ")
    assert describe_chord([0x11, 0x53]) == "ctrl+S"


def test_chord_presses_in_order_and_releases_in_reverse() -> None:
    inp, backend, _ = make()
    inp.press_keys("ctrl+shift+esc")
    assert backend.events == [("key", 0x11, "down"), ("key", 0x10, "down"), ("key", 0x1B, "down"),
                              ("key", 0x1B, "up"), ("key", 0x10, "up"), ("key", 0x11, "up")]


def test_repeat_is_capped() -> None:
    inp, backend, _ = make()
    inp.press_keys("down", repeat=500)
    assert len(backend.events) == 100  # 50 presses x (down, up)


def test_paste_restores_every_saved_clipboard_format() -> None:
    clip = FakeClipboard([(13, "old text\x00".encode("utf-16-le")), (49161, b"\x01\x02rich")])
    inp, backend, _ = make(clip)
    result = inp.type_text("سڵاو، چۆنی؟")
    assert result == {"method": "paste", "restored": True, "saved_formats": 2}
    # snapshot -> put SAM's text -> Ctrl+V -> restore, in that order
    assert clip.log == ["snapshot", "set:سڵاو، چۆنی؟", "restore"]
    assert clip.private == [True]  # kept out of Win+V history / cloud clipboard
    assert backend.events == [("key", 0x11, "down"), ("key", 0x56, "down"), ("key", 0x56, "up"), ("key", 0x11, "up")]
    assert clip.formats == [(13, "old text\x00".encode("utf-16-le")), (49161, b"\x01\x02rich")]


def test_clipboard_is_restored_even_if_the_paste_fails() -> None:
    class Broken(RecorderBackend):
        def key(self, vk: int, up: bool) -> None:
            if vk == 0x56 and not up:
                raise OSError("SendInput was blocked")
            super().key(vk, up)

    clip = FakeClipboard([(13, "keep me\x00".encode("utf-16-le"))])
    inp = Input(Broken(), clip, sleep=lambda s: None, paste_settle_s=0.0)
    with pytest.raises(OSError):
        inp.type_text("x")
    assert clip.log[-1] == "restore"
    assert clip.get_text() == "keep me"


def test_locked_clipboard_falls_back_to_unicode_typing() -> None:
    clip = FakeClipboard(locked=True)
    inp, backend, _ = make(clip)
    result = inp.type_text("ئا😀", press_enter=True)
    assert result["method"] == "unicode"
    units = [e[1] for e in backend.events if e[0] == "uni" and e[2] == "down"]
    assert units == [0x0626, 0x0627, 0xD83D, 0xDE00]  # surrogate pair for the emoji
    assert backend.events[-2:] == [("key", 0x0D, "down"), ("key", 0x0D, "up")]


def test_click_approaches_the_target_and_double_clicks() -> None:
    inp, backend, _ = make()
    inp.click(500, 400, double=True)
    moves = [e for e in backend.events if e[0] == "move"]
    assert moves == [("move", 500, 397), ("move", 500, 400)]
    buttons = [e for e in backend.events if e[0] == "button"]
    assert buttons == [("button", "left", "down"), ("button", "left", "up")] * 2
    inp.scroll(10, 10, -3)
    assert backend.events[-1] == ("wheel", -360)
