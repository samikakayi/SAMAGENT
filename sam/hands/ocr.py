"""Windows OCR (Windows.Media.Ocr) on screen captures: read text with
positions, find a phrase, click it. Local, free, 25-96 ms per frame
(measured, reports/computer-control.json); installed recognisers on this PC:
en-US and ar-SA (there is no Kurdish one; ar-SA reads Arabic-script text).

Threading (ported from v1 ``trading/ocr.py``, whose comments hold the
measurements): SAM v1 died twice with an access violation (0xC0000005) when
the engine was created on a shared thread that another library had put into
a COM single-threaded apartment. So every WinRT call -- imports, engine
creation, property reads, recognition -- runs on ONE dedicated worker thread
that joins the multithreaded apartment before any WinRT call, and holds one
MTA usage for the life of the process (without it the process crashed at
exit in 10 of 11 runs; with it 25 of 25 exited cleanly). Callers only get
plain Python values back.

SAM 2 uses the modular ``winrt-*`` 3.x projections (Python 3.13 wheels)
instead of v1's ``winsdk``; the bitmap is handed over as BMP (an in-memory
copy) instead of PNG (~200 ms of encoding on a 2880x1800 frame).
"""

from __future__ import annotations

import asyncio
import ctypes
import io
import re
import threading
from dataclasses import dataclass
from typing import Any, Callable

from ..textnorm import is_arabic_script, normalize_ckb

_on_ocr_worker = threading.local()


@dataclass(frozen=True)
class OcrWord:
    text: str
    left: float
    top: float
    width: float
    height: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.left + self.width / 2, self.top + self.height / 2)


@dataclass(frozen=True)
class OcrLine:
    text: str
    words: tuple[OcrWord, ...]

    @property
    def rect(self) -> tuple[int, int, int, int]:
        left = min(w.left for w in self.words)
        top = min(w.top for w in self.words)
        right = max(w.left + w.width for w in self.words)
        bottom = max(w.top + w.height for w in self.words)
        return (int(left), int(top), int(right), int(bottom))


def logical_line(text: str, words: list[OcrWord]) -> OcrLine:
    """Windows OCR lists words left to right (measured live on Sorani text in
    Notepad): for Arabic-script lines that is reverse reading order, so the
    words and the line text are put back into reading (right-to-left) order."""
    joined = " ".join(w.text for w in words)
    if is_arabic_script(joined):
        ordered = sorted(words, key=lambda w: -(w.left + w.width))
        return OcrLine(" ".join(w.text for w in ordered), tuple(ordered))
    return OcrLine(text or joined, tuple(words))


def _enter_multithreaded_apartment() -> None:
    """Worker initializer: join the MTA before any WinRT call (see module doc)."""
    _on_ocr_worker.active = True
    try:
        from winrt import _winrt  # type: ignore[attr-defined]
    except ImportError:
        return
    try:
        _winrt.init_apartment(_winrt.MTA)
    except (OSError, AttributeError, RuntimeError):
        pass  # still uninitialised: the first activation takes the implicit MTA
    try:
        ctypes.oledll.ole32.CoIncrementMTAUsage(ctypes.byref(ctypes.c_void_p()))
    except (OSError, AttributeError):
        pass


class WindowsOcrEngine:
    """Synchronous wrapper; every WinRT call runs on the shared OCR worker."""

    _worker: Any = None
    _worker_guard = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._engines: dict[str, Any] = {}
        self._languages: list[str] | None = None
        self._probe_error: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @classmethod
    def _on_worker(cls, call: Callable[[], Any]) -> Any:
        if getattr(_on_ocr_worker, "active", False):
            return call()
        import concurrent.futures

        with cls._worker_guard:
            if cls._worker is None:
                cls._worker = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="sam-ocr", initializer=_enter_multithreaded_apartment)
            worker = cls._worker
        return worker.submit(call).result()

    @staticmethod
    def _imports() -> tuple[Any, Any, Any, Any, Any]:
        from winrt.windows.globalization import Language
        from winrt.windows.graphics.imaging import BitmapDecoder
        from winrt.windows.media.ocr import OcrEngine
        from winrt.windows.storage.streams import DataWriter, InMemoryRandomAccessStream

        return OcrEngine, BitmapDecoder, InMemoryRandomAccessStream, DataWriter, Language

    def _engine(self, language: str | None) -> Any:
        """Engine for a BCP-47 prefix ("ar", "en") or the user's default. Worker only."""
        with self._lock:
            key = language or ""
            if key in self._engines:
                return self._engines[key]
            if self._probe_error:
                raise RuntimeError(self._probe_error)
            try:
                ocr_engine, _, _, _, language_cls = self._imports()
            except ImportError as exc:
                self._probe_error = f"Windows OCR bindings are not installed: {exc}"
                raise RuntimeError(self._probe_error) from exc
            available = list(ocr_engine.available_recognizer_languages)
            self._languages = [str(item.language_tag) for item in available]
            engine = None
            if language:
                for item in available:
                    if str(item.language_tag).lower().startswith(language.lower()):
                        engine = ocr_engine.try_create_from_language(language_cls(str(item.language_tag)))
                        break
            if engine is None:
                engine = ocr_engine.try_create_from_user_profile_languages()
            if engine is None and available:
                engine = ocr_engine.try_create_from_language(available[0])
            if engine is None:
                self._probe_error = "Windows has no installed OCR language pack."
                raise RuntimeError(self._probe_error)
            self._engines[key] = engine
            return engine

    def languages(self) -> list[str]:
        if self._languages is None:
            self._on_worker(lambda: self._engine(None))
        return list(self._languages or [])

    def capability(self) -> dict[str, Any]:
        try:
            languages = self.languages()
        except RuntimeError as exc:
            return {"name": "screen_ocr", "state": "unconfigured", "engine": "windows-media-ocr", "reason": str(exc)}
        return {"name": "screen_ocr", "state": "available", "engine": "windows-media-ocr",
                "languages": languages, "reason": None}

    async def _recognize_async(self, bmp: bytes, language: str | None) -> list[OcrLine]:
        _, bitmap_decoder, memory_stream, data_writer, _ = self._imports()
        engine = self._engine(language)
        stream = memory_stream()
        writer = data_writer(stream.get_output_stream_at(0))
        writer.write_bytes(bmp)
        await writer.store_async()
        await writer.flush_async()
        stream.seek(0)
        decoder = await bitmap_decoder.create_async(stream)
        bitmap = await decoder.get_software_bitmap_async()
        result = await engine.recognize_async(bitmap)
        lines: list[OcrLine] = []
        for line in result.lines:
            words = []
            for word in line.words:
                rect = word.bounding_rect
                words.append(OcrWord(str(word.text), float(rect.x), float(rect.y), float(rect.width), float(rect.height)))
            if words:
                lines.append(logical_line(str(line.text), words))
        return lines

    def _run_on_worker(self, bmp: bytes, language: str | None) -> list[OcrLine]:
        # WinRT async operations need an event loop; the worker keeps its own
        # (it never runs one otherwise), independent of SAM's core loop.
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(self._recognize_async(bmp, language))

    def recognize(self, image: Any, language: str | None = None) -> list[OcrLine]:
        """Recognise a PIL image. Raises RuntimeError when OCR is unavailable."""
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="BMP")
        payload = buffer.getvalue()
        return self._on_worker(lambda: self._run_on_worker(payload, language))


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    cx, cy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


def merge_lines(latin: list[OcrLine], arabic: list[OcrLine]) -> list[OcrLine]:
    """Combine the English and Arabic recognisers' readings of one image.

    Measured on this PC: the ar-SA recogniser turns English UI text into
    Latin garbage and the en-US one does the same to Arabic-script text, so
    each Arabic-script line from ar-SA replaces the en-US line at the same
    place; everything else comes from en-US."""
    kept_arabic = [line for line in arabic if is_arabic_script(line.text)]
    merged = [line for line in latin
              if not any(_overlaps(line.rect, a.rect) or _overlaps(a.rect, line.rect) for a in kept_arabic)]
    merged.extend(kept_arabic)
    merged.sort(key=lambda line: (line.rect[1] // 12, line.rect[0]))
    return merged


# Windows has no Kurdish recogniser; ar-SA reads Sorani-only letters as their
# Arabic base letters (ڵ->ل, ۆ->و, ێ->ی, ە->ه, and the Persian-style letters
# پ چ ژ گ ڤ as ب ج ز ک ف). Both sides are folded the same way before comparing.
_FOLD = str.maketrans({"ڵ": "ل", "ڕ": "ر", "ۆ": "و", "ێ": "ی", "ە": "ه", "ھ": "ه", "ڤ": "ف", "گ": "ک",
                       "چ": "ج", "پ": "ب", "ژ": "ز"})


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", normalize_ckb(text, strip_punct=True).translate(_FOLD)).strip()


def find_phrase(lines: list[OcrLine], phrase: str, *, threshold: float = 80.0) -> list[dict[str, Any]]:
    """Where ``phrase`` appears: exact word runs first, then fuzzy lines.

    Returns ``[{"text", "rect", "score"}]`` in image coordinates, best first.
    """
    from rapidfuzz import fuzz

    wanted = _clean(phrase).split()
    if not wanted:
        return []
    hits: list[dict[str, Any]] = []
    n = len(wanted)
    for line in lines:
        words = [_clean(w.text) for w in line.words]
        for i in range(len(words) - n + 1):
            window = " ".join(words[i:i + n])
            score = 100.0 if window == " ".join(wanted) else float(fuzz.ratio(window, " ".join(wanted)))
            if score >= threshold:
                chunk = line.words[i:i + n]
                left = min(w.left for w in chunk)
                top = min(w.top for w in chunk)
                right = max(w.left + w.width for w in chunk)
                bottom = max(w.top + w.height for w in chunk)
                hits.append({"text": " ".join(w.text for w in chunk), "score": round(score, 1),
                             "rect": (int(left), int(top), int(right), int(bottom))})
        if n > 1 or not hits:
            score = float(fuzz.partial_ratio(" ".join(wanted), _clean(line.text)))
            if score >= max(threshold, 90.0) and len(_clean(line.text)) <= 3 * len(" ".join(wanted)) + 10:
                hits.append({"text": line.text, "rect": line.rect, "score": round(score - 5, 1)})
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits


class Ocr:
    """Async facade over the engine + screen captures (screen coordinates)."""

    def __init__(self, screen: Any, engine: Any = None) -> None:
        self.screen = screen
        self.engine = engine or WindowsOcrEngine()

    async def read_image(self, image: Any, language: str | None = None) -> list[OcrLine]:
        return await asyncio.to_thread(self.engine.recognize, image, language)

    def _has(self, prefix: str) -> bool:
        try:
            return any(tag.lower().startswith(prefix) for tag in self.engine.languages())
        except Exception:  # noqa: BLE001
            return False

    async def read_merged(self, image: Any) -> list[OcrLine]:
        """English + (if installed) Arabic readings merged (see ``merge_lines``)."""
        latin = await self.read_image(image, "en")
        if not self._has("ar"):
            return latin
        arabic = await self.read_image(image, "ar")
        return merge_lines(latin, arabic)

    async def read(self, hwnd: int | None = None, region: tuple[int, int, int, int] | None = None, *,
                   language: str | None = None) -> list[dict[str, Any]]:
        """Lines of text with screen rectangles: ``[{"text", "rect"}]``."""
        image, origin = await self.screen.grab(hwnd=hwnd, region=region)
        lines = await (self.read_image(image, language) if language else self.read_merged(image))
        return [{"text": line.text, "rect": _offset(line.rect, origin)} for line in lines]

    async def find_text(self, text: str, hwnd: int | None = None, *,
                        region: tuple[int, int, int, int] | None = None) -> list[dict[str, Any]]:
        """Screen rectangles where ``text`` is visible, best match first. The
        recogniser matching the query's script is tried first."""
        image, origin = await self.screen.grab(hwnd=hwnd, region=region)
        languages: list[str | None] = ["ar", "en", None] if is_arabic_script(text) else ["en", None]
        for language in languages:
            try:
                lines = await self.read_image(image, language)
            except RuntimeError:
                continue
            hits = find_phrase(lines, text)
            if hits:
                return [{**h, "rect": _offset(h["rect"], origin)} for h in hits]
        return []


def _offset(rect: tuple[int, int, int, int], origin: tuple[int, int]) -> tuple[int, int, int, int]:
    return (rect[0] + origin[0], rect[1] + origin[1], rect[2] + origin[0], rect[3] + origin[1])


__all__ = ["Ocr", "OcrLine", "OcrWord", "WindowsOcrEngine", "find_phrase"]
