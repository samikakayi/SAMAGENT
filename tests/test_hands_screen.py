"""Screenshots: downscale to <= 1440 px, blank sensitive areas (password
fields, the MT5 account number/balances) before anything leaves the PC."""

from __future__ import annotations

import io
from typing import Any

from PIL import Image

from sam.hands.ocr import OcrLine, OcrWord
from sam.hands.screen import Screen, blank, clip, encode
from tests.hands_helpers import fake_windows, win


def test_encode_downscales_to_max_side() -> None:
    image = Image.new("RGB", (2880, 1800), (200, 200, 200))
    data, width, height = encode(image, max_side=1440)
    assert (width, height) == (1440, 900)
    assert Image.open(io.BytesIO(data)).format == "JPEG"
    data, width, height = encode(Image.new("RGB", (800, 600)), max_side=1440, fmt="png")
    assert (width, height) == (800, 600) and data.startswith(b"\x89PNG")


def test_clip_and_blank() -> None:
    assert clip((-10, -10, 50, 50), (0, 0, 100, 100)) == (0, 0, 50, 50)
    assert clip((200, 200, 300, 300), (0, 0, 100, 100)) is None
    image = Image.new("RGB", (100, 100), (255, 255, 255))
    assert blank(image, [(110, 110, 130, 130), (500, 500, 600, 600)], origin=(100, 100)) == 1
    assert image.getpixel((15, 15)) == (0, 0, 0) and image.getpixel((50, 50)) == (255, 255, 255)


class FakeOcr:
    async def read_image(self, image: Any, language: str | None = None) -> list[OcrLine]:
        return [OcrLine("Balance: 10 000.00 USD", (OcrWord("Balance:", 10, 500, 80, 14),)),
                OcrLine("Account 100130161", (OcrWord("Account", 10, 520, 60, 14), OcrWord("100130161", 80, 520, 90, 14))),
                OcrLine("XAUUSD H1", (OcrWord("XAUUSD", 10, 100, 60, 14),))]


async def test_window_capture_blanks_mt5_and_password_fields() -> None:
    mt5 = win(5, "100130161 - InfinoxLimited-MT5Demo: Demo Account", "terminal64.exe", rect=(0, 0, 1000, 800))
    windows, api = fake_windows([mt5], foreground=5)
    api.border = (0, 0, 0, 0)
    grabbed: list[Any] = []

    def grabber(rect: tuple[int, int, int, int]) -> Image.Image:
        grabbed.append(rect)
        return Image.new("RGB", (rect[2] - rect[0], rect[3] - rect[1]), (255, 255, 255))
    screen = Screen(windows, grabber=grabber)
    screen._screen_bounds_sync = lambda: (0, 0, 2880, 1800)  # type: ignore[method-assign]
    screen.ocr = FakeOcr()

    async def passwords(window: Any, rect: Any) -> list[tuple[int, int, int, int]]:
        return [(300, 300, 400, 330)]
    screen.redactors.append(passwords)
    shot = await screen.capture_ex(5, max_side=1440)
    assert grabbed == [(0, 0, 1000, 800)]
    assert (shot.width, shot.height, shot.out_width) == (1000, 800, 1000)
    blanked = set(shot.blanked)
    assert (300, 300, 400, 330) in blanked                       # password field
    assert (0, 0, 1000, 40) in blanked                           # title bar with the account number
    assert any(r[1] <= 500 <= r[3] for r in blanked)              # the balance line
    assert any(r[1] <= 520 <= r[3] for r in blanked)              # the account line
    assert not any(r[1] <= 100 <= r[3] and r[0] < 50 for r in blanked if r != (0, 0, 1000, 40))


async def test_minimized_window_cannot_be_captured() -> None:
    windows, _ = fake_windows([win(6, "x", "notepad.exe", minimized=True)])
    screen = Screen(windows, grabber=lambda rect: Image.new("RGB", (10, 10)))
    screen._screen_bounds_sync = lambda: (0, 0, 2880, 1800)  # type: ignore[method-assign]
    try:
        await screen.capture_ex(6)
    except LookupError as exc:
        assert "minimized" in str(exc)
    else:
        raise AssertionError("expected LookupError")


async def test_a_failing_redactor_blanks_everything() -> None:
    windows, api = fake_windows([win(7, "Bank", "chrome.exe", rect=(0, 0, 400, 300))], foreground=7)
    api.border = (0, 0, 0, 0)
    screen = Screen(windows, grabber=lambda rect: Image.new("RGB", (400, 300), (255, 255, 255)))
    screen._screen_bounds_sync = lambda: (0, 0, 2880, 1800)  # type: ignore[method-assign]

    async def broken(window: Any, rect: Any) -> list[Any]:
        raise RuntimeError("uia died")
    screen.redactors.append(broken)
    shot = await screen.capture_ex(7)
    assert shot.blanked == [(0, 0, 400, 300)]
    image = Image.open(io.BytesIO(shot.data))
    assert image.convert("L").getextrema()[1] < 20  # all black
