"""Windows OCR must not share SAM's request threads with COM.

SAM died twice in production with a Windows access violation, and faulthandler
put it at ocr.py's capability(). The OCR engine had been created on a shared
asyncio.to_thread worker that VoiceService.sapi_voices had already left in a
COM single-threaded apartment (STA). The real crash can only be reproduced in a
separate process, so the first test runs one there. The others pin the
threading contract with fake bindings, so they run on any machine.
"""

from __future__ import annotations

import asyncio
import importlib.util
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from sam_backend.trading.ocr import WindowsOcrEngine

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Measured on this machine: the pre-fix engine crashed in 15 of 15 runs, and
# 5 of 5 already crashed with only 100 reads. Each crash came within 2.5 s.
# Four hundred reads leave margin and finish in about 1.5 s on the fixed engine.
STATUS_READS = 400

# The production sequence in miniature. A pool thread imports win32com, which
# makes it the process's first STA. It then serves the first trading status
# read, which creates the engine, and retires. After that, a mixed pool keeps
# polling. Before the fix this died at "ocr.py ... in capability", the frame
# production showed.
STA_THEN_POOL = r'''
import concurrent.futures
import faulthandler
import gc
import itertools
import sys
import threading

faulthandler.enable()

import sam_backend.trading.ocr as ocr_module
from sam_backend.trading.ocr import WindowsOcrEngine

READS = int(sys.argv[1])
print(ocr_module.__file__, flush=True)
local = threading.local()
slots = itertools.count()


def enter_sta():
    # What VoiceService.sapi_voices leaves behind on a pool thread.
    import pythoncom
    import win32com.client

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
if state != "AVAILABLE":
    print("OCR-UNAVAILABLE", flush=True)
    sys.exit(0)


def poll(index):
    if not hasattr(local, "slot"):
        local.slot = next(slots)
    # Half the pool are STAs that keep touching SAPI; half never initialise COM.
    if local.slot % 2 == 0 and (not getattr(local, "sta", False) or index % 50 == 0):
        enter_sta()
    return engine.capability()["state"]


with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    for start in range(0, READS, 100):
        assert set(pool.map(poll, range(start, start + 100))) == {"AVAILABLE"}
        gc.collect()
print("survived", READS, flush=True)
'''


def _windows_com_stack_installed() -> bool:
    return sys.platform == "win32" and all(
        importlib.util.find_spec(name) is not None for name in ("winsdk", "pythoncom", "win32com")
    )


@pytest.mark.skipif(not _windows_com_stack_installed(), reason="needs Windows with winsdk and pywin32")
def test_status_reads_survive_an_engine_first_touched_from_an_sta_thread():
    finished = subprocess.run(
        [sys.executable, "-c", STA_THEN_POOL, str(STATUS_READS)], cwd=PROJECT_ROOT,
        capture_output=True, text=True, encoding="utf-8", timeout=120,
    )

    assert finished.returncode == 0, (
        f"OCR process died with exit code {finished.returncode & 0xFFFFFFFF:#010x}\n"
        f"{finished.stdout}\n{finished.stderr}"
    )
    if "OCR-UNAVAILABLE" in finished.stdout:
        pytest.skip("Windows OCR language pack is not installed on this machine")
    imported = Path(finished.stdout.splitlines()[0]).resolve()
    assert imported == PROJECT_ROOT / "sam_backend" / "trading" / "ocr.py"
    assert f"survived {STATUS_READS}" in finished.stdout


# --- The threading contract, with fake bindings ------------------------------


def _fake_bindings(calls: list[tuple[str, str]], *, language_pack: bool = True):
    def note(what: str) -> None:
        calls.append((what, threading.current_thread().name))

    class Language:
        @property
        def display_name(self) -> str:
            note("display_name")
            return "English (United States)"

    class Engine:
        recognizer_language = Language()

        async def recognize_async(self, bitmap):
            note("recognize_async")
            rect = SimpleNamespace(x=10.0, y=20.0, width=30.0, height=8.0)
            return SimpleNamespace(lines=[SimpleNamespace(words=[SimpleNamespace(text="4485.25", bounding_rect=rect)])])

    class OcrEngine:
        available_recognizer_languages: list = []

        @staticmethod
        def try_create_from_user_profile_languages():
            note("create")
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

    def imports():
        note("imports")
        return OcrEngine, BitmapDecoder, Stream, DataWriter

    return imports


class _Image:
    def save(self, buffer, format):
        buffer.write(b"png")


def test_every_winsdk_call_runs_on_the_one_ocr_worker():
    calls: list[tuple[str, str]] = []
    engine = WindowsOcrEngine()
    engine._imports = _fake_bindings(calls)

    capability = engine.capability()

    async def from_a_running_loop():
        return engine.recognize(_Image())

    words = asyncio.run(from_a_running_loop())
    again = WindowsOcrEngine()
    again._imports = _fake_bindings(calls)
    again.recognize(_Image())

    assert capability == {
        "name": "screen_ocr", "state": "AVAILABLE", "engine": "windows-media-ocr",
        "language": "English (United States)", "reason": None,
    }
    assert type(capability["language"]) is str
    assert [(word.text, word.left, word.top) for word in words] == [("4485.25", 10.0, 20.0)]
    threads = {name for _, name in calls}
    # One long-lived worker for every instance, never the caller's own thread.
    assert len(threads) == 1 and threads.pop().startswith("sam-ocr")


def test_a_created_engine_answers_status_without_touching_winrt_again():
    calls: list[tuple[str, str]] = []
    engine = WindowsOcrEngine()
    engine._imports = _fake_bindings(calls)
    engine.capability()
    calls.clear()

    for _ in range(3):
        assert engine.capability()["language"] == "English (United States)"

    assert calls == []


def test_missing_bindings_still_report_unconfigured_with_the_same_reason():
    engine = WindowsOcrEngine()

    def no_bindings():
        raise ImportError("No module named 'winsdk'")

    engine._imports = no_bindings

    assert engine.capability() == {
        "name": "screen_ocr", "state": "UNCONFIGURED", "engine": "windows-media-ocr",
        "reason": "Windows OCR bindings are not installed: No module named 'winsdk'",
    }


def test_a_machine_without_a_language_pack_still_reports_unconfigured():
    calls: list[tuple[str, str]] = []
    engine = WindowsOcrEngine()
    engine._imports = _fake_bindings(calls, language_pack=False)

    capability = engine.capability()

    assert capability["state"] == "UNCONFIGURED"
    assert capability["reason"] == "Windows has no installed OCR language pack."
    with pytest.raises(RuntimeError, match="no installed OCR language pack"):
        engine.recognize(_Image())
