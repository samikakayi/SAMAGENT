"""Screenshots for OCR and vision models (mss + Pillow).

Measured on this PC (reports/computer-control.json): a full 2880x1800 grab
takes ~212 ms; JPEG q80 at 1440 px wide is ~114 KiB in ~194 ms, versus 494 KiB
PNG at full size. Vision models get a window crop scaled to <= 1440 px JPEG;
OCR gets the full-resolution crop (small UI text needs every pixel).

Privacy: before any image leaves the PC, sensitive areas are blanked:
password fields (UIA ``IsPassword``), the MetaTrader 5 title bar (it shows
the account number, e.g. "100130161 - InfinoxLimited-MT5Demo: Demo Account")
and any MT5 text that looks like an account number or a balance line
(found with local OCR). Extra rectangles can be passed by the caller.
"""

from __future__ import annotations

import asyncio
import io
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from . import _win

Rect = tuple[int, int, int, int]
Redactor = Callable[[Any, Rect], Awaitable[list[Rect]]]
MT5_PROCESSES = frozenset({"terminal64.exe", "terminal.exe", "metatrader.exe"})
_ACCOUNT_RE = re.compile(r"\b\d{6,12}\b")
_BALANCE_WORDS = ("balance", "equity", "margin", "free margin", "profit", "credit", "login", "account")


@dataclass
class Shot:
    """An encoded screenshot plus what is needed to map points back."""

    data: bytes
    fmt: str
    left: int            # screen origin of the captured area (physical px)
    top: int
    width: int           # captured size in physical px
    height: int
    out_width: int       # encoded image size
    out_height: int
    hwnd: int | None = None
    title: str = ""
    blanked: list[Rect] = field(default_factory=list)

    @property
    def scale(self) -> float:
        return self.out_width / self.width if self.width else 1.0

    @property
    def mime(self) -> str:
        return "image/jpeg" if self.fmt == "jpeg" else "image/png"


def clip(rect: Rect, bounds: Rect) -> Rect | None:
    left, top = max(rect[0], bounds[0]), max(rect[1], bounds[1])
    right, bottom = min(rect[2], bounds[2]), min(rect[3], bounds[3])
    if right - left < 2 or bottom - top < 2:
        return None
    return (left, top, right, bottom)


def blank(image: Any, rects: list[Rect], origin: tuple[int, int]) -> int:
    """Paint screen rectangles black on ``image`` (captured at ``origin``)."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    count = 0
    for rect in rects:
        box = (rect[0] - origin[0], rect[1] - origin[1], rect[2] - origin[0], rect[3] - origin[1])
        if box[2] <= 0 or box[3] <= 0 or box[0] >= image.width or box[1] >= image.height:
            continue
        draw.rectangle(box, fill=(0, 0, 0))
        count += 1
    return count


def draw_marks(image: Any, marks: list[tuple[int, Rect]], origin: tuple[int, int]) -> int:
    """Outline each candidate element in red with its number (set-of-marks).
    Sizes scale with the image so the number stays readable after the
    downscale to 1440 px (a 2880 px capture is halved)."""
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(image)
    size = max(14, image.width // 80)
    try:
        font = ImageFont.truetype("arialbd.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    width = max(2, image.width // 700)
    for number, rect in marks:
        box = (rect[0] - origin[0], rect[1] - origin[1], rect[2] - origin[0], rect[3] - origin[1])
        if box[2] <= 0 or box[3] <= 0 or box[0] >= image.width or box[1] >= image.height:
            continue
        draw.rectangle(box, outline=(230, 30, 30), width=width)
        label = str(number)
        text_w = draw.textlength(label, font=font)
        top = box[1] - size - 4 if box[1] - size - 4 >= 0 else box[1]
        draw.rectangle((box[0], top, box[0] + text_w + 6, top + size + 4), fill=(230, 30, 30))
        draw.text((box[0] + 3, top + 1), label, fill=(255, 255, 255), font=font)
    return len(marks)


def encode(image: Any, *, max_side: int = 1440, fmt: str = "jpeg", quality: int = 80) -> tuple[bytes, int, int]:
    from PIL import Image

    width, height = image.size
    factor = min(1.0, max_side / max(width, height)) if max_side else 1.0
    if factor < 1.0:
        image = image.resize((max(1, round(width * factor)), max(1, round(height * factor))), Image.BILINEAR)
    buffer = io.BytesIO()
    if fmt == "png":
        image.save(buffer, format="PNG")
    else:
        image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue(), image.size[0], image.size[1]


class Screen:
    """Captures in physical pixels on a DPI-aware worker thread."""

    def __init__(self, windows: Any, grabber: Callable[[Rect], Any] | None = None) -> None:
        self.windows = windows
        self._grabber = grabber
        self.redactors: list[Redactor] = []

    def _grab_sync(self, rect: Rect) -> Any:
        if self._grabber is not None:
            return self._grabber(rect)
        import mss
        from PIL import Image

        with _win.dpi_aware(), mss.mss() as sct:
            raw = sct.grab({"left": rect[0], "top": rect[1], "width": rect[2] - rect[0], "height": rect[3] - rect[1]})
            return Image.frombuffer("RGB", raw.size, raw.bgra, "raw", "BGRX")

    def _screen_bounds_sync(self) -> Rect:
        with _win.dpi_aware():
            u = _win.user32()
            left, top = u.GetSystemMetrics(76), u.GetSystemMetrics(77)
            return (left, top, left + u.GetSystemMetrics(78), top + u.GetSystemMetrics(79))

    async def area_of(self, hwnd: int | None = None, region: Rect | None = None) -> tuple[Rect, Any]:
        """(screen rect, WindowInfo | None) to capture. Default: the whole screen."""
        bounds = await asyncio.to_thread(self._screen_bounds_sync)
        window = None
        if region is not None:
            rect = clip(tuple(int(v) for v in region), bounds)  # type: ignore[arg-type]
        elif hwnd:
            window = await self.windows.find(int(hwnd))
            if window is None:
                raise LookupError("that window no longer exists")
            if window.minimized:
                raise LookupError(f"'{window.title[:60]}' is minimized")
            frame = await self.windows._run(self.windows.api.frame_rect, int(hwnd))
            rect = clip(frame, bounds)
        else:
            rect = bounds
        if rect is None:
            raise LookupError("the area to capture is off screen")
        return rect, window

    async def grab(self, hwnd: int | None = None, region: Rect | None = None) -> tuple[Any, tuple[int, int]]:
        """Full-resolution PIL image of a window/region and its screen origin."""
        rect, _ = await self.area_of(hwnd, region)
        image = await asyncio.to_thread(self._grab_sync, rect)
        return image, (rect[0], rect[1])

    async def capture_ex(self, hwnd: int | None = None, *, region: Rect | None = None, max_side: int = 1440,
                         fmt: str = "jpeg", quality: int = 80, blank_rects: tuple[Rect, ...] | list[Rect] = (),
                         redact: bool = True, marks: list[tuple[int, Rect]] | None = None) -> Shot:
        rect, window = await self.area_of(hwnd, region)
        image = await asyncio.to_thread(self._grab_sync, rect)
        rects = list(blank_rects)
        origin = (rect[0], rect[1])
        if redact:
            subject = window if window is not None else await self.windows.foreground()
            for redactor in self.redactors:
                try:
                    rects.extend(await redactor(subject, rect) or [])
                except Exception:  # noqa: BLE001 - a failing redactor must not leak: blank all
                    rects.append(rect)
            shown = [window] if window is not None else [
                w for w in await self.windows.list() if not w.minimized and clip(w.rect, rect)]
            for item in shown:
                if item.process.lower() in MT5_PROCESSES:
                    area = rect if item is window else clip(item.rect, rect)
                    if area is not None:
                        rects.extend(await self._mt5_rects(image, origin, area))
        if rects:
            await asyncio.to_thread(blank, image, rects, origin)
        if marks:
            await asyncio.to_thread(draw_marks, image, marks, origin)
        data, out_w, out_h = await asyncio.to_thread(encode, image, max_side=max_side, fmt=fmt, quality=quality)
        return Shot(data=data, fmt=fmt, left=rect[0], top=rect[1], width=rect[2] - rect[0], height=rect[3] - rect[1],
                    out_width=out_w, out_height=out_h, hwnd=getattr(window, "hwnd", None),
                    title=getattr(window, "title", ""), blanked=rects)

    async def capture(self, hwnd: int | None = None, *, max_side: int = 1440, fmt: str = "jpeg",
                      blank_rects: tuple[Rect, ...] | list[Rect] = ()) -> bytes:
        """Contract API: encoded bytes of a window crop (or the whole screen)."""
        return (await self.capture_ex(hwnd, max_side=max_side, fmt=fmt, blank_rects=blank_rects)).data

    async def _mt5_rects(self, image: Any, origin: tuple[int, int], rect: Rect) -> list[Rect]:
        """MetaTrader 5 window at screen ``rect``: blank its title bar (account
        number) and any OCR'd account-number/balance text. Local OCR only; if
        OCR fails, the lower third (Toolbox: balance/equity) is blanked too."""
        rects: list[Rect] = [(rect[0], rect[1], rect[2], rect[1] + max(40, (rect[3] - rect[1]) // 30))]
        ocr = getattr(self, "ocr", None)
        crop = image.crop((rect[0] - origin[0], rect[1] - origin[1], rect[2] - origin[0], rect[3] - origin[1]))
        try:
            lines = await ocr.read_image(crop, "en") if ocr is not None else None
        except Exception:  # noqa: BLE001
            lines = None
        if lines is None:
            height = rect[3] - rect[1]
            return rects + [(rect[0], rect[1] + (2 * height) // 3, rect[2], rect[3])]
        for line in lines:
            text = line.text.lower()
            if _ACCOUNT_RE.search(line.text) or any(word in text for word in _BALANCE_WORDS):
                left, top, right, bottom = line.rect
                rects.append((left + rect[0] - 4, top + rect[1] - 4, right + rect[0] + 4, bottom + rect[1] + 4))
        return rects


def password_redactor(uia: Any) -> Redactor:
    async def redactor(window: Any, rect: Rect) -> list[Rect]:
        if window is None:
            return []
        return await uia.password_rects(window.hwnd)
    return redactor


__all__ = ["MT5_PROCESSES", "Screen", "Shot", "blank", "clip", "draw_marks", "encode", "password_redactor"]
