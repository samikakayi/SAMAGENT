"""Windows OCR must never share a thread with other COM users (ported from
v1 tests/test_ocr_apartment.py).

SAM v1 died twice in production with an access violation because the OCR
engine was created on a shared thread that SAPI had left in a COM STA. The
real crash only reproduces in a separate process, so the first test runs one
there (it skips when WinRT OCR or pywin32 is missing). The others pin the
threading contract with fake bindings, so they run anywhere."""

from __future__ import annotations

import asyncio
import importlib.util
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from sam.hands.ocr import OcrLine, OcrWord, WindowsOcrEngine, find_phrase, merge_lines

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATUS_READS = 400  # v1: the pre-fix engine crashed in 15/15 runs within 400 reads

STA_THEN_POOL = r'''
import concurrent.futures, faulthandler, gc, itertools, sys, threading
faulthandler.enable()
sys.path.insert(0, sys.argv[2])
import sam.hands.ocr as ocr_module
from sam.hands.ocr import WindowsOcrEngine
READS = int(sys.argv[1])
print(ocr_module.__file__, flush=True)
local = threading.local()
slots = itertools.count()

def enter_sta():
    import pythoncom, win32com.client
    if not getattr(local, "sta", False):
        pythoncom.CoInitialize()
        local.sta = True
    try:
        len(list(win32com.client.Dispatch("SAPI.SpVoice").GetVoices()))
    except pythoncom.com_error:
        pass

engine = WindowsOcrEngine()
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as first:
    def first_read():
        enter_sta()
        return engine.capability()["state"]
    state = first.submit(first_read).result()
if state != "available":
    print("OCR-UNAVAILABLE", flush=True)
    sys.exit(0)

def poll(index):
    if not hasattr(local, "slot"):
        local.slot = next(slots)
    if local.slot % 2 == 0 and (not getattr(local, "sta", False) or index % 50 == 0):
        enter_sta()
    return engine.capability()["state"]

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    for start in range(0, READS, 100):
        assert set(pool.map(poll, range(start, start + 100))) == {"available"}
        gc.collect()
print("survived", READS, flush=True)
'''


def _windows_com_stack_installed() -> bool:
    return sys.platform == "win32" and all(
        importlib.util.find_spec(name) is not None for name in ("winrt.windows.media.ocr", "pythoncom", "win32com"))


@pytest.mark.skipif(not _windows_com_stack_installed(), reason="needs Windows with winrt OCR and pywin32")
def test_status_reads_survive_an_engine_first_touched_from_an_sta_thread() -> None:
    finished = subprocess.run([sys.executable, "-c", STA_THEN_POOL, str(STATUS_READS), str(PROJECT_ROOT)],
                              cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", timeout=180)
    assert finished.returncode == 0, (f"OCR process died with exit code {finished.returncode & 0xFFFFFFFF:#010x}\n"
                                      f"{finished.stdout}\n{finished.stderr}")
    if "OCR-UNAVAILABLE" in finished.stdout:
        pytest.skip("no Windows OCR language pack on this machine")
    imported = Path(finished.stdout.splitlines()[0]).resolve()
    assert imported == PROJECT_ROOT / "sam" / "hands" / "ocr.py"
    assert f"survived {STATUS_READS}" in finished.stdout


# --- the threading contract with fake bindings ---------------------------------------
def _fake_bindings(calls: list[tuple[str, str]], *, language_pack: bool = True):
    def note(what: str) -> None:
        calls.append((what, threading.current_thread().name))

    class Lang:
        def __init__(self, tag: str) -> None:
            self.language_tag = tag

    class Engine:
        async def recognize_async(self, bitmap):
            note("recognize_async")
            rect = SimpleNamespace(x=10.0, y=20.0, width=30.0, height=8.0)
            word = SimpleNamespace(text="4485.25", bounding_rect=rect)
            return SimpleNamespace(lines=[SimpleNamespace(text="4485.25", words=[word])])

    class OcrEngine:
        available_recognizer_languages = [Lang("ar-SA"), Lang("en-US")] if language_pack else []

        @staticmethod
        def try_create_from_user_profile_languages():
            note("create")
            return Engine() if language_pack else None

        @staticmethod
        def try_create_from_language(lang):
            note("create:" + str(getattr(lang, "tag", lang)))
            return Engine() if language_pack else None

    class Stream:
        def get_output_stream_at(self, position):
            return self

        def seek(self, position):
            note("seek")

    class DataWriter:
        def __init__(self, stream):
            pass

        def write_bytes(self, data):
            note("write_bytes")

        async def store_async(self):
            return None

        async def flush_async(self):
            return None

    class Decoder:
        async def get_software_bitmap_async(self):
            note("bitmap")
            return object()

    class BitmapDecoder:
        @staticmethod
        async def create_async(stream):
            return Decoder()

    class Language:
        def __init__(self, tag):
            self.tag = tag

    def imports():
        note("imports")
        return OcrEngine, BitmapDecoder, Stream, DataWriter, Language

    return imports


class _Image:
    def convert(self, mode):
        return self

    def save(self, buffer, format):
        buffer.write(b"BM")


def test_every_winrt_call_runs_on_the_one_ocr_worker() -> None:
    calls: list[tuple[str, str]] = []
    engine = WindowsOcrEngine()
    engine._imports = _fake_bindings(calls)
    capability = engine.capability()

    async def from_a_running_loop():
        return engine.recognize(_Image())
    lines = asyncio.run(from_a_running_loop())
    again = WindowsOcrEngine()
    again._imports = _fake_bindings(calls)
    again.recognize(_Image(), "ar")

    assert capability["state"] == "available" and capability["languages"] == ["ar-SA", "en-US"]
    assert [(w.text, w.left, w.top) for w in lines[0].words] == [("4485.25", 10.0, 20.0)]
    threads = {name for _, name in calls}
    assert len(threads) == 1 and threads.pop().startswith("sam-ocr")
    assert ("create:ar-SA", ) == tuple(c[0] for c in calls if c[0].startswith("create:"))


def test_a_created_engine_is_reused() -> None:
    calls: list[tuple[str, str]] = []
    engine = WindowsOcrEngine()
    engine._imports = _fake_bindings(calls)
    engine.recognize(_Image())
    engine.recognize(_Image())
    assert sum(1 for c in calls if c[0] == "create") == 1


def test_missing_bindings_and_language_packs_report_unconfigured() -> None:
    engine = WindowsOcrEngine()

    def no_bindings():
        raise ImportError("No module named 'winrt'")
    engine._imports = no_bindings
    capability = engine.capability()
    assert capability["state"] == "unconfigured" and "not installed" in capability["reason"]

    calls: list[tuple[str, str]] = []
    bare = WindowsOcrEngine()
    bare._imports = _fake_bindings(calls, language_pack=False)
    assert bare.capability()["state"] == "unconfigured"
    with pytest.raises(RuntimeError, match="no installed OCR language pack"):
        bare.recognize(_Image())


# --- text matching -----------------------------------------------------------------------
def _line(text: str, left: float, top: float) -> OcrLine:
    words, x = [], left
    for part in text.split():
        words.append(OcrWord(part, x, top, 10.0 * len(part), 12.0))
        x += 10.0 * len(part) + 5
    return OcrLine(text, tuple(words))


def test_find_phrase_exact_words_first_then_fuzzy() -> None:
    lines = [_line("File Edit View", 0, 0), _line("Save as PDF", 0, 40), _line("Sav changes", 0, 80)]
    hits = find_phrase(lines, "save as")
    assert hits[0]["text"] == "Save as" and hits[0]["score"] == 100.0
    assert hits[0]["rect"] == (0, 40, 65, 52)
    assert find_phrase(lines, "Save changes")[0]["text"] == "Sav changes"
    assert find_phrase(lines, "Delete everything") == []


def test_sorani_phrases_match_after_normalisation() -> None:
    lines = [_line("سڵاو چۆنی", 100, 10)]
    assert find_phrase(lines, "سڵاو")[0]["text"] == "سڵاو"
    assert find_phrase(lines, "سلاو چونی")  # speech-to-text spelling without Kurdish letters


def test_merge_prefers_arabic_readings_of_arabic_lines() -> None:
    latin = [_line("File Edit", 0, 0), _line("IOJ o.Jlw", 0, 50)]      # en-US reading Sorani as garbage
    arabic = [_line("Fiie Ed1t", 0, 0), _line("سڵاو چۆنی", 0, 50)]      # ar-SA reading English as garbage
    merged = [line.text for line in merge_lines(latin, arabic)]
    assert merged == ["File Edit", "سڵاو چۆنی"]


def test_arabic_lines_are_put_in_reading_order() -> None:
    from sam.hands.ocr import logical_line

    # Windows OCR lists words by x (left to right); Sorani reads right to left.
    words = [OcrWord("سامە", 10, 0, 40, 12), OcrWord("تاقیکردنەوەی", 60, 0, 90, 12), OcrWord("ئەمە", 160, 0, 40, 12)]
    line = logical_line("ignored", words)
    assert line.text == "ئەمە تاقیکردنەوەی سامە"
    hit = find_phrase([line], "تاقیکردنەوەی سامە")[0]
    assert hit["score"] == 100.0 and hit["rect"] == (10, 0, 150, 12)
    latin = logical_line("File Edit", [OcrWord("File", 0, 0, 30, 12), OcrWord("Edit", 40, 0, 30, 12)])
    assert latin.text == "File Edit"
