"""Offline screen OCR built on the Windows built-in recognizer.

The engine ships with Windows, runs locally, and downloads nothing. When it is
unavailable the adapter reports UNCONFIGURED instead of guessing, so callers can
degrade to an explicit manual calibration rather than invent chart geometry.
"""

from __future__ import annotations

import asyncio
import io
import re
import threading
from dataclasses import dataclass
from typing import Any

from ..contracts import CapabilityState

_PRICE_PATTERN = re.compile(r"^[+-]?\d{1,3}(?:[ ,]\d{3})*(?:[.,]\d+)?$|^[+-]?\d+(?:[.,]\d+)?$")


@dataclass(slots=True, frozen=True)
class OcrWord:
    text: str
    left: float
    top: float
    width: float
    height: float

    @property
    def center_y(self) -> float:
        return self.top + self.height / 2

    @property
    def center_x(self) -> float:
        return self.left + self.width / 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
            "center_x": self.center_x,
            "center_y": self.center_y,
        }


class WindowsOcrEngine:
    """Thin synchronous wrapper around Windows.Media.Ocr."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._engine: Any = None
        self._probe_error: str | None = None
        self._probed = False

    @staticmethod
    def _imports() -> tuple[Any, Any, Any, Any]:
        from winsdk.windows.graphics.imaging import BitmapDecoder
        from winsdk.windows.media.ocr import OcrEngine
        from winsdk.windows.storage.streams import DataWriter, InMemoryRandomAccessStream

        return OcrEngine, BitmapDecoder, InMemoryRandomAccessStream, DataWriter

    def _ensure_engine(self) -> Any:
        with self._lock:
            if self._engine is not None:
                return self._engine
            if self._probed and self._probe_error:
                raise RuntimeError(self._probe_error)
            self._probed = True
            try:
                ocr_engine, *_ = self._imports()
            except ImportError as exc:
                self._probe_error = f"Windows OCR bindings are not installed: {exc}"
                raise RuntimeError(self._probe_error) from exc
            engine = ocr_engine.try_create_from_user_profile_languages()
            if engine is None:
                available = ocr_engine.available_recognizer_languages
                if available:
                    engine = ocr_engine.try_create_from_language(available[0])
            if engine is None:
                self._probe_error = "Windows has no installed OCR language pack."
                raise RuntimeError(self._probe_error)
            self._engine = engine
            return engine

    def capability(self) -> dict[str, Any]:
        try:
            engine = self._ensure_engine()
        except RuntimeError as exc:
            return {
                "name": "screen_ocr",
                "state": CapabilityState.UNCONFIGURED.value,
                "engine": "windows-media-ocr",
                "reason": str(exc),
            }
        return {
            "name": "screen_ocr",
            "state": CapabilityState.AVAILABLE.value,
            "engine": "windows-media-ocr",
            "language": engine.recognizer_language.display_name,
            "reason": None,
        }

    async def _recognize_async(self, png_bytes: bytes) -> list[OcrWord]:
        _, bitmap_decoder, memory_stream, data_writer = self._imports()
        engine = self._ensure_engine()
        stream = memory_stream()
        writer = data_writer(stream.get_output_stream_at(0))
        writer.write_bytes(png_bytes)
        await writer.store_async()
        await writer.flush_async()
        stream.seek(0)
        decoder = await bitmap_decoder.create_async(stream)
        bitmap = await decoder.get_software_bitmap_async()
        result = await engine.recognize_async(bitmap)
        words: list[OcrWord] = []
        for line in result.lines:
            for word in line.words:
                rect = word.bounding_rect
                words.append(OcrWord(word.text, rect.x, rect.y, rect.width, rect.height))
        return words

    def recognize(self, image: Any) -> list[OcrWord]:
        """Recognize a PIL image. Raises RuntimeError when OCR is unavailable."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        payload = buffer.getvalue()
        # winsdk's APIs are async and need a loop of their own. If the caller
        # already has one running, borrow a separate thread rather than failing.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._recognize_async(payload))
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(self._recognize_async(payload))).result()


def parse_price(text: str) -> float | None:
    """Parse a price-axis label, rejecting anything that is not purely numeric."""
    cleaned = text.strip().replace("−", "-").replace(" ", "")
    if not cleaned or not _PRICE_PATTERN.match(cleaned):
        return None
    cleaned = cleaned.replace(",", "") if cleaned.count(",") > 1 or "." in cleaned else cleaned.replace(",", ".")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value if value == value and abs(value) != float("inf") else None
