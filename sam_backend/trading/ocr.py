"""Offline screen OCR built on the Windows built-in recognizer.

The engine ships with Windows, runs locally, and downloads nothing. When it is
unavailable the adapter reports UNCONFIGURED instead of guessing, so callers can
degrade to an explicit manual calibration rather than invent chart geometry.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import ctypes
import io
import re
import threading
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from ..contracts import CapabilityState

_PRICE_PATTERN = re.compile(r"^[+-]?\d{1,3}(?:[ ,]\d{3})*(?:[.,]\d+)?$|^[+-]?\d+(?:[.,]\d+)?$")

_T = TypeVar("_T")


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


# Marks the OCR worker thread, so work already running there is not queued
# behind itself on a one-thread pool.
_on_ocr_worker = threading.local()


def _enter_multithreaded_apartment() -> None:
    """Start the OCR worker in the multithreaded apartment, before any WinRT call.

    winsdk 1.0.0b10 does not choose an apartment when it is imported. Its first
    activation joins whatever apartment the calling thread is already in, so on a
    thread that pywin32 has made single-threaded (STA) the engine is born in that
    STA. On a thread nothing has initialised, C++/WinRT takes the implicit MTA
    instead (measured with CoGetApartmentType).
    """
    _on_ocr_worker.active = True
    try:
        from winsdk import _winrt
    except ImportError:
        return  # capability() reports the missing bindings by itself.
    # An initializer that raises breaks the pool for the life of the process,
    # and every later capability() would report UNCONFIGURED. So a winsdk build
    # without init_apartment (AttributeError) is treated like a refused call.
    try:
        _winrt.init_apartment(_winrt.MTA)
    except (OSError, AttributeError):
        pass  # Still uninitialised, so first activation takes the implicit MTA.
    # Joining the MTA alone is not enough. At shutdown the pool's exit hook
    # joins this worker, and only then does module teardown release the engine,
    # on the main thread. The worker was the MTA's only member, so the MTA was
    # already gone by then. Measured: the process died at exit, with no Python
    # frame, in 10 of 11 runs. Holding one MTA usage for the life of the process
    # (C++/WinRT does the same for an uninitialised thread) keeps the MTA alive
    # for that last release. Measured: 25 of 25 runs then exited cleanly.
    try:
        ctypes.oledll.ole32.CoIncrementMTAUsage(ctypes.byref(ctypes.c_void_p()))
    except (OSError, AttributeError):
        pass


class WindowsOcrEngine:
    """Thin synchronous wrapper around Windows.Media.Ocr.

    Every winsdk call runs on one dedicated worker thread in the multithreaded
    apartment: the imports, creating the engine, reading its properties, and the
    whole recognition pipeline. Callers on other threads only ever get plain
    Python values back.

    SAM's request threads are not safe for this. The shared asyncio.to_thread
    pool also runs VoiceService.sapi_voices, and its win32com import leaves that
    pool thread in a COM single-threaded apartment. An engine created there took
    the whole process down with an access violation, twice in production, at
    capability()'s language read. Measured in isolation, with the engine created
    on such a thread, the old code crashed in 15 of 15 runs, each within its
    first 400 status reads. A recognition through that engine from another
    thread failed with "Operation aborted". Behind this worker, the same load
    ran cleanly in every run.
    """

    # One worker for the process, shared by every instance. tradingview.py builds
    # a throwaway engine for each toolbar read, and a thread per instance would
    # come and go with each read. The pool's own exit hook joins an idle worker
    # when the interpreter shuts down, so it never keeps SAM alive.
    _worker: concurrent.futures.ThreadPoolExecutor | None = None
    _worker_guard = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._engine: Any = None
        # Plain str copy of engine.recognizer_language.display_name. It is read
        # once, on the worker, when the engine is created.
        self._language: str | None = None
        self._probe_error: str | None = None
        self._probed = False

    @classmethod
    def _on_worker(cls, call: Callable[[], _T]) -> _T:
        if getattr(_on_ocr_worker, "active", False):
            return call()
        with cls._worker_guard:
            if cls._worker is None:
                cls._worker = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="sam-ocr",
                    initializer=_enter_multithreaded_apartment,
                )
            worker = cls._worker
        return worker.submit(call).result()

    @staticmethod
    def _imports() -> tuple[Any, Any, Any, Any]:
        from winsdk.windows.graphics.imaging import BitmapDecoder
        from winsdk.windows.media.ocr import OcrEngine
        from winsdk.windows.storage.streams import DataWriter, InMemoryRandomAccessStream

        return OcrEngine, BitmapDecoder, InMemoryRandomAccessStream, DataWriter

    def _ensure_engine(self) -> Any:
        """Create the engine once. Runs on the OCR worker only."""
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
            self._language = str(engine.recognizer_language.display_name)
            self._engine = engine
            return engine

    def _probe_language(self) -> str | None:
        self._ensure_engine()
        return self._language

    def capability(self) -> dict[str, Any]:
        # Once the engine exists, the cached name answers without visiting the
        # worker, so a status poll never waits behind a recognition in progress.
        language = self._language
        if language is None:
            try:
                language = self._on_worker(self._probe_language)
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
            "language": language,
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
        # winsdk's APIs are async and need a loop of their own. The worker never
        # has one running, so asyncio.run works there even when the caller's
        # thread has a loop running.
        return self._on_worker(lambda: asyncio.run(self._recognize_async(payload)))


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
