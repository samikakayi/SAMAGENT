"""The hands-free line must not claim to wait for "Hey SAM" when nothing can hear it.

Found live: the SAM started at 03:16:58 lost its wake listener to a numpy
import race seconds after starting it (tests/test_startup_import_race.py), and
its speech model never loaded. /api/voice/handsfree said so -- wake.running
false, wake.error and wake.detector.error set -- but the state stayed
WAKE_LISTENING, and the panel, which only looked at wake.ready, kept saying
"Waiting for Hey SAM…" until a restart.

The wake block comes from a real WakeWordService whose listener dies the way
that one did, so its keys are the production ones. The panel is the real
frontend/hands-free.js with the real frontend/i18n.js, run by
tests/render_hands_free.js.
"""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

from sam_backend.config import Settings
from sam_backend.voice_session import VoiceConversationController
from sam_backend.wake import LocalPhraseDetector, WakeWordService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# What the 03:16:58 process reported, verbatim.
IMPORT_RACE = ("cannot import name '__cpu_features__' from partially initialized module "
               "'numpy._core._multiarray_umath' (most likely due to a circular import)")
MODEL_FAILURE = "Could not load Whisper model 'small': cannot load module more than once per process"
WAITING = "Waiting for Hey SAM…"
STOPPED = "Not listening for Hey SAM: the listener stopped"
MODEL_FAILED = "Hey SAM may not be heard: the speech model reported an error"


class BrokenDictation:
    """The dictation model the detector shares, failing to load as it did."""

    def available(self) -> bool:
        return True

    def load(self):
        raise RuntimeError(MODEL_FAILURE)


class Voice:
    stt = None
    on_speaking = None


@pytest.fixture()
def dead_listener(tmp_path: Path) -> dict:
    """/api/voice/handsfree once the listener has died after starting."""
    settings = Settings(project_root=tmp_path, workspace_root=tmp_path / "workspace",
                        data_dir=tmp_path / "data", hands_free_enabled=True)
    opened = threading.Event()

    def microphone():
        # Dies after the loop has seen it start, as the real one did: the
        # import that failed came after the listener was already running.
        opened.wait(5)
        raise ImportError(IMPORT_RACE)

    detector = LocalPhraseDetector("small", shared=BrokenDictation())
    wake = WakeWordService(settings, detector=detector, stream_factory=microphone)
    session = VoiceConversationController(settings, Voice(), agent=None, wake=wake)
    session.start()
    opened.set()
    deadline = time.monotonic() + 5
    while (wake.running or not detector.ready) and time.monotonic() < deadline:
        time.sleep(0.01)
    payload = session.describe()
    session.stop()
    assert payload["wake"]["running"] is False
    assert payload["wake"]["error"] == f"ImportError: {IMPORT_RACE}"
    assert payload["wake"]["detector"]["error"] == f"RuntimeError: {MODEL_FAILURE}"
    # What that process kept reporting throughout.
    payload.update(state="WAKE_LISTENING", detail=WAITING)
    return payload


def healthy(payload: dict) -> dict:
    state = copy.deepcopy(payload)
    state["wake"].update(running=True, error="")
    state["wake"]["detector"]["error"] = ""
    return state


def render(tmp_path: Path, states: list[dict], locale: str = "en") -> list[dict]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    scenario = tmp_path / "hands-free-scenario.json"
    scenario.write_text(json.dumps({"locale": locale, "states": states}), encoding="utf-8")
    finished = subprocess.run(
        [node, str(Path(__file__).parent / "render_hands_free.js"), str(PROJECT_ROOT / "frontend" / "i18n.js"),
         str(PROJECT_ROOT / "frontend" / "hands-free.js"), str(scenario)],
        check=True, capture_output=True, text=True, encoding="utf-8",
    )
    return json.loads(finished.stdout)


def test_a_dead_listener_is_shown_as_stopped_with_its_errors_in_the_tooltip(tmp_path, dead_listener):
    [seen] = render(tmp_path, [dead_listener])
    assert seen["text"] == STOPPED
    assert seen["shown"]
    assert f"Listener: ImportError: {IMPORT_RACE}" in seen["tooltip"]
    assert f"Speech model: RuntimeError: {MODEL_FAILURE}" in seen["tooltip"]
    assert "restart SAM" in seen["tooltip"]
    # The microphone is closed, so the dot is not lit and the button starts it
    # again instead of stopping a listener that is already gone.
    assert seen["dot"] == "0.35"
    assert seen["button"] == "Start"


def test_a_healthy_listener_still_waits_for_the_phrase(tmp_path, dead_listener):
    [seen] = render(tmp_path, [healthy(dead_listener)])
    assert seen == {"text": WAITING, "tooltip": "", "shown": True, "button": "Stop", "dot": "1"}


def test_a_listener_that_stopped_without_saying_why_is_still_not_waiting(tmp_path, dead_listener):
    silent = healthy(dead_listener)
    silent["wake"]["running"] = False
    [seen] = render(tmp_path, [silent])
    assert seen["text"] == STOPPED
    assert seen["tooltip"] == "Press Start to listen again. If it stops again, restart SAM."


def test_a_speech_model_error_is_shown_while_the_microphone_stays_open(tmp_path, dead_listener):
    unheard = healthy(dead_listener)
    unheard["wake"]["detector"]["error"] = f"RuntimeError: {MODEL_FAILURE}"
    [seen] = render(tmp_path, [unheard])
    assert seen["text"] == MODEL_FAILED
    assert seen["tooltip"].startswith(f"Speech model: RuntimeError: {MODEL_FAILURE}")
    assert (seen["dot"], seen["button"]) == ("1", "Stop")


def test_the_fault_clears_once_the_listener_is_back(tmp_path, dead_listener):
    dead, back = render(tmp_path, [dead_listener, healthy(dead_listener)])
    assert dead["text"] == STOPPED
    assert (back["text"], back["tooltip"]) == (WAITING, "")


def test_switched_off_or_mid_turn_the_line_says_what_the_loop_is_doing(tmp_path, dead_listener):
    """Only the waiting line speaks for the listener.

    Stopped by the user, the listener is meant to be gone. During a turn the
    command is captured by the voice service itself, so "Thinking…" is true
    whatever happened to the listener; the turn ends back on the waiting line.
    """
    off = dict(dead_listener, state="OFF", detail="Hands-free voice is switched off.")
    thinking = dict(dead_listener, state="THINKING", detail="Thinking…")
    switched_off, mid_turn = render(tmp_path, [off, thinking])
    assert not switched_off["shown"]
    assert switched_off["tooltip"] == ""
    assert (mid_turn["text"], mid_turn["tooltip"]) == ("Thinking…", "")


def test_the_fault_is_said_in_sorani(tmp_path, dead_listener):
    [seen] = render(tmp_path, [dead_listener], locale="ckb-IQ")
    assert seen["text"] == "گوێ لە Hey SAM ناگیرێت: گوێگرەکە وەستاوە"
    assert f"گوێگر: ImportError: {IMPORT_RACE}" in seen["tooltip"]
    assert f"مۆدێلی قسە: RuntimeError: {MODEL_FAILURE}" in seen["tooltip"]
