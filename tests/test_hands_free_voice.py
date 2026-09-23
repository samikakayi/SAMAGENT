"""Hands-free voice: hearing "Hey SAM" without handing the room to anybody.

Every test here runs on synthetic audio and fake engines. None opens a
microphone, because a test that needs a room to be quiet is a test that fails
for reasons that have nothing to do with the code.

The questions that matter most are not "does it transcribe". They are: does it
stay off until asked, does it refuse to hear itself, and does a spoken request
get any authority a typed one would not.
"""

from __future__ import annotations

import threading
import time

import numpy
import pytest

from sam_backend.config import Settings
from sam_backend.voice_session import VoiceConversationController
from sam_backend.wake import (
    ECHO_COOLDOWN_SECONDS,
    SENSITIVITY_ENERGY,
    VoiceState,
    WakeWordService,
    phrase_heard,
)


FRAME = 16_000 * 30 // 1000  # what a real microphone hands over, 30 ms at a time


def frames_of(level: float, seconds: float) -> list:
    """A run of real-sized frames, as MicrophoneStream would deliver them."""
    count = max(1, int(seconds * 1000 / 30))
    return [numpy.full(FRAME, level, dtype="float32") for _ in range(count)]


def loud(seconds: float = 1.0) -> list:
    """Audio the energy gate will treat as speech."""
    return frames_of(0.2, seconds)


def quiet(seconds: float = 1.0) -> list:
    return frames_of(0.0, seconds)


class ScriptedStream:
    """A microphone that plays a fixed script and then waits to be stopped."""

    def __init__(self, frames, hold: bool = True):
        self._frames = frames
        self._hold = hold
        self.opened = 0
        self.closed = 0

    def __enter__(self):
        self.opened += 1
        return self

    def __exit__(self, *exc):
        self.closed += 1
        return False

    def frames(self, timeout: float = 0.4):
        for item in self._frames:
            yield from (item if isinstance(item, list) else [item])
        while self._hold:
            time.sleep(0.005)
            yield numpy.zeros(FRAME, dtype="float32")


class ScriptedDetector:
    """Says the phrase was heard whenever the transcript it is fed says so."""

    def __init__(self, transcripts, available: bool = True):
        self._transcripts = list(transcripts)
        self._available = available
        self.examined = 0

    available_error = ""

    @property
    def available(self) -> bool:
        return self._available

    def detect(self, audio, phrase) -> bool:
        self.examined += 1
        text = self._transcripts.pop(0) if self._transcripts else ""
        return phrase_heard(text, phrase)

    def describe(self):
        return {"engine": "scripted", "cloud": False, "credential_required": False}


class FakeVoice:
    """Stands in for VoiceService: capture and speech, no devices."""

    def __init__(self, utterances=None, speak_fails: bool = False):
        self._utterances = list(utterances or [])
        self.spoken: list[str] = []
        self.speak_fails = speak_fails
        self.listen_calls = 0
        self.speaking_while_suppressed: list[bool] = []
        self.wake: WakeWordService | None = None
        self.on_speaking = None

    def listen_once(self, **kwargs):
        self.listen_calls += 1
        if not self._utterances:
            return {"captured": False, "reason": "no speech"}
        item = self._utterances.pop(0)
        if isinstance(item, Exception):
            raise item
        if item is None:
            return {"captured": False, "reason": "no speech"}
        return {"captured": True, "text": item, "seconds": 1.0}

    def speak(self, text, language=None):
        # The real VoiceService announces its own speech so the wake listener
        # can go deaf, whoever asked it to talk. The double must too, or these
        # tests would prove something the product does not do.
        if self.on_speaking is not None:
            self.on_speaking(True)
        try:
            if self.wake is not None:
                self.speaking_while_suppressed.append(self.wake.suppressed)
            if self.speak_fails:
                return {"ok": False, "error": "audio device busy"}
            self.spoken.append(text)
            return {"ok": True, "seconds": 0.1}
        finally:
            if self.on_speaking is not None:
                self.on_speaking(False)


def controller(**overrides):
    settings = Settings.from_env()
    settings.hands_free_enabled = overrides.pop("enabled", True)
    settings.hands_free_continuation_seconds = overrides.pop("continuation", 0.0)
    settings.hands_free_auto_speak = overrides.pop("auto_speak", True)
    settings.voice_wake_word = overrides.pop("phrase", "Hey SAM")
    settings.hands_free_sensitivity = overrides.pop("sensitivity", "NORMAL")

    voice = overrides.pop("voice", FakeVoice(["what is gold doing"]))
    detector = overrides.pop("detector", ScriptedDetector(["hey sam"]))
    stream = overrides.pop("stream", None)
    wake = WakeWordService(settings, detector=detector,
                           stream_factory=(lambda: stream) if stream else None)
    voice.wake = wake
    replies = overrides.pop("replies", ["Gold is trading quietly."])

    def submit(text):
        if not replies:
            return {"reply": "", "error": None}
        answer = replies.pop(0)
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, dict):
            return answer
        return {"reply": answer, "error": None}

    return VoiceConversationController(settings, voice, agent=None, wake=wake,
                                       submit=overrides.pop("submit", submit))


# --- 1/2. off until asked --------------------------------------------------------


def test_hands_free_is_off_on_a_fresh_install():
    """A microphone that starts listening because software was installed is not a feature."""
    assert Settings.from_env().hands_free_enabled is False


def test_a_disabled_controller_does_not_open_the_microphone():
    stream = ScriptedStream([loud()])
    session = controller(enabled=False, stream=stream)

    state = session.start()

    assert state["state"] == VoiceState.OFF.value
    assert stream.opened == 0, "no device was touched"
    assert session.wake.running is False


def test_enabling_it_starts_the_wake_listener():
    stream = ScriptedStream([quiet(0.03)])
    session = controller(stream=stream)

    state = session.start()
    try:
        assert state["state"] == VoiceState.WAKE_LISTENING.value
        assert "Hey SAM" in state["detail"]
        assert session.wake.running is True
    finally:
        session.stop()
    assert stream.closed == 1, "the device is released on stop"


# --- 3/4/5. what wakes it, and what does not ------------------------------------


@pytest.mark.parametrize("heard, wakes", [
    ("Hey SAM", True),
    ("hey sam", True),
    ("Hey, SAM!", True),
    ("okay hey sam what is gold doing", True),
    ("hey some", False),
    ("say ham", False),
    ("the exam was hard", False),
    ("", False),
    ("samantha called", False),
])
def test_only_the_phrase_wakes_it(heard, wakes):
    assert phrase_heard(heard, "Hey SAM") is wakes


def test_one_wake_phrase_starts_one_session():
    stream = ScriptedStream([loud(0.6), loud(0.6)])
    detector = ScriptedDetector(["hey sam", "hey sam"])
    session = controller(stream=stream, detector=detector)
    started: list[int] = []
    session.wake._on_wake = lambda: started.append(1)

    session.wake.start(lambda: started.append(1))
    time.sleep(0.6)
    session.wake.stop()

    assert len(started) == 1, f"{len(started)} sessions began from one utterance"


def test_a_repeated_phrase_is_debounced():
    """Saying it twice quickly is one request, not two."""
    from sam_backend.wake import WAKE_DEBOUNCE_SECONDS

    assert WAKE_DEBOUNCE_SECONDS >= 2.0
    stream = ScriptedStream([loud(0.6)] * 6)
    detector = ScriptedDetector(["hey sam"] * 6)
    session = controller(stream=stream, detector=detector)
    fired: list[int] = []

    session.wake.start(lambda: fired.append(1))
    time.sleep(0.8)
    session.wake.stop()

    assert len(fired) == 1, f"{len(fired)} wakes inside the debounce window"


def test_silence_never_reaches_the_recogniser():
    """The energy gate is what keeps idle listening free."""
    stream = ScriptedStream([quiet(0.03)] * 30, hold=False)
    detector = ScriptedDetector([])
    session = controller(stream=stream, detector=detector)

    session.wake.start(lambda: None)
    time.sleep(0.4)
    session.wake.stop()

    assert detector.examined == 0, "a quiet room was sent to the recogniser"


@pytest.mark.parametrize("level", ["LOW", "NORMAL", "HIGH"])
def test_sensitivity_changes_the_gate_not_the_phrase(level):
    session = controller(sensitivity=level)

    assert session.wake.sensitivity == level
    assert session.wake.energy_gate == SENSITIVITY_ENERGY[level]
    assert session.wake.phrase == "Hey SAM"


# --- 6. Sorani after an English wake phrase -------------------------------------


def test_a_sorani_command_follows_an_english_wake_phrase():
    """The phrase may be English; the request need not be."""
    voice = FakeVoice(["ئێستا بازاڕ چۆنە"])
    session = controller(voice=voice, replies=["بازاڕ هێمنە."])

    session.run_session()

    assert session.transcripts == ["ئێستا بازاڕ چۆنە"]
    assert voice.spoken == ["بازاڕ هێمنە."]
    assert session.status.state is VoiceState.WAKE_LISTENING


def test_the_configured_language_is_passed_to_capture():
    captured: dict = {}
    voice = FakeVoice(["hello"])
    original = voice.listen_once

    def spy(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)

    voice.listen_once = spy
    session = controller(voice=voice)
    session.settings.voice_language = "ckb-IQ"

    session.run_session()

    assert captured["language"] == "ckb-IQ", "Sorani is not overridden by the English phrase"


# --- 7/8. end of speech and bounds ----------------------------------------------


def test_a_turn_is_bounded_in_time():
    from sam_backend.voice_session import MAX_COMMAND_SECONDS

    captured: dict = {}
    voice = FakeVoice(["hello"])
    original = voice.listen_once

    def spy(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)

    voice.listen_once = spy
    controller(voice=voice).run_session()

    assert captured["max_seconds"] == MAX_COMMAND_SECONDS
    assert 0 < MAX_COMMAND_SECONDS <= 60, "a microphone must not record without end"


def test_silence_after_the_wake_phrase_returns_to_waiting():
    voice = FakeVoice([None])
    session = controller(voice=voice)

    session.run_session()

    assert session.status.state is VoiceState.WAKE_LISTENING
    assert voice.spoken == []


# --- 9/10. the continuation window ----------------------------------------------


def test_a_follow_up_needs_no_wake_phrase():
    voice = FakeVoice(["open gold", "what timeframe is it"])
    session = controller(voice=voice, continuation=5.0,
                         replies=["Gold is open.", "Fifteen minutes."])

    session.run_session()

    assert session.transcripts == ["open gold", "what timeframe is it"]
    assert voice.spoken == ["Gold is open.", "Fifteen minutes."]


def test_the_window_closes_on_silence():
    voice = FakeVoice(["open gold", None])
    session = controller(voice=voice, continuation=5.0, replies=["Gold is open."])

    session.run_session()

    assert session.transcripts == ["open gold"]
    assert session.status.state is VoiceState.WAKE_LISTENING
    assert session.status.continuation_active is False


def test_a_zero_window_means_one_turn():
    voice = FakeVoice(["open gold", "and the timeframe"])
    session = controller(voice=voice, continuation=0.0, replies=["Gold is open.", "never asked"])

    session.run_session()

    assert session.transcripts == ["open gold"], "no follow-up was invited"


# --- 11. SAM must not hear itself ------------------------------------------------


def test_wake_detection_is_deaf_while_sam_speaks():
    """Otherwise: SAM speaks, SAM hears "Hey SAM", SAM speaks, forever."""
    voice = FakeVoice(["what is gold doing"])
    session = controller(voice=voice, replies=["Hey SAM is listening, gold is quiet."])

    session.run_session()

    assert voice.speaking_while_suppressed == [True], "the listener was awake during playback"
    assert session.wake.suppressed is False, "and listening again afterwards"


def test_frames_arriving_during_playback_are_discarded():
    session = controller()
    session.wake.suppress()
    session.wake._buffer.extend(loud(0.3))
    session.wake.suppress()

    assert len(session.wake._buffer) == 0, "SAM's own voice was kept for examination"


def test_there_is_a_cooldown_after_speaking():
    assert ECHO_COOLDOWN_SECONDS > 0, "the tail of SAM's own voice could wake it"


# --- 12/13/14/15. failures that must not cost the answer -------------------------


def test_a_tts_failure_still_keeps_the_answer():
    voice = FakeVoice(["what is gold doing"], speak_fails=True)
    session = controller(voice=voice, replies=["Gold is quiet."])

    session.run_session()

    assert session.status.last_reply == "Gold is quiet.", "the text survived"
    assert session.status.last_error == "audio device busy", "and the reason is kept"
    assert voice.spoken == [], "nothing was actually spoken"


def test_a_capture_failure_returns_to_waiting():
    voice = FakeVoice([RuntimeError("device disconnected")])
    session = controller(voice=voice)

    session.run_session()

    assert session.status.state is VoiceState.WAKE_LISTENING
    assert "device disconnected" in session.status.last_error


def test_an_agent_error_is_reported_and_does_not_speak():
    voice = FakeVoice(["do something"])
    session = controller(voice=voice, replies=[{"reply": "", "error": "the model is unavailable"}])

    session.run_session()

    assert session.status.state is VoiceState.WAKE_LISTENING
    assert "model is unavailable" in session.status.last_error
    assert voice.spoken == []


def test_a_missing_wake_engine_is_reported_not_crashed():
    session = controller(detector=ScriptedDetector([], available=False))

    state = session.start()

    assert state["state"] == VoiceState.ERROR.value
    assert "microphone button still works" in state["detail"]


def test_a_microphone_that_will_not_open_is_a_state():
    class Broken:
        def __enter__(self):
            raise OSError("audio device busy")

        def __exit__(self, *exc):
            return False

    session = controller(stream=Broken())
    session.start()
    time.sleep(0.3)

    assert session.wake.error, "the failure was recorded"
    session.stop()


# --- 16/17. a spoken request earns no extra authority ----------------------------


def test_a_spoken_request_goes_through_the_ordinary_agent_path(client, app):
    """Not a second brain: the same chat entry point the typed box uses."""
    import inspect

    source = inspect.getsource(VoiceConversationController._ask)
    assert "self.agent.chat" in source
    assert "policy" not in source and "approve" not in source, \
        "the voice path must not make its own permission decisions"


def test_voice_adds_no_route_that_bypasses_approval(app):
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    voice_paths = {path for path in paths if "/voice/" in path}

    assert "/api/voice/handsfree" in voice_paths
    for path in voice_paths:
        assert "approve" not in path and "execute" not in path, \
            f"{path} looks like a voice-only action path"


def test_starting_hands_free_while_disabled_is_refused(client):
    response = client.post("/api/voice/handsfree/start")

    assert response.status_code == 409
    assert "Settings" in response.json()["detail"]["error"]


def test_the_status_route_reports_state_without_audio(client):
    payload = client.get("/api/voice/handsfree").json()

    assert payload["state"] == VoiceState.OFF.value
    assert payload["enabled"] is False
    assert "No audio is written to disk" in payload["privacy"]
    for key in ("audio", "samples", "pcm", "wav", "recording"):
        assert key not in payload


def test_no_live_trading_appears_through_voice(app):
    """Voice must not create an execution path trading does not already have."""
    trading = app.state.trading

    assert getattr(trading, "live_trading_enabled", False) is False
    from sam_backend.voice_session import VoiceConversationController as Controller

    assert not any(name for name in dir(Controller) if "order" in name.lower() or "trade" in name.lower())


# --- 18. the preference persists -------------------------------------------------


def test_hands_free_settings_persist_across_a_restart(client, app, settings):
    saved = client.put("/api/settings", json={
        "hands_free_enabled": True, "hands_free_sensitivity": "HIGH",
        "hands_free_continuation_seconds": 12.0, "hands_free_auto_speak": False,
        "voice_wake_word": "Hey SAM"})

    assert saved.status_code == 200
    overrides = saved.json()["overrides"]
    assert overrides["hands_free_enabled"] is True
    assert overrides["hands_free_sensitivity"] == "HIGH"
    assert overrides["hands_free_continuation_seconds"] == 12.0
    assert overrides["hands_free_auto_speak"] is False

    # A brand new application over the same database -- the restart.
    import sam_backend.config as config
    from sam_backend.app import create_app

    config.is_elevated_windows_process = lambda: False
    restarted = create_app(settings)
    assert restarted.state.settings.hands_free_enabled is True
    assert restarted.state.settings.hands_free_sensitivity == "HIGH"
    assert restarted.state.settings.hands_free_continuation_seconds == 12.0
    assert restarted.state.settings.hands_free_auto_speak is False


def test_a_rejected_sensitivity_never_reaches_settings(client):
    response = client.put("/api/settings", json={"hands_free_sensitivity": "MAXIMUM"})

    assert response.status_code == 422


# --- what the real microphone taught us ----------------------------------------
#
# Everything below is a defect that only appeared once a real room, a real
# speaker and a real speech model were in the loop. Each one silently stopped
# SAM hearing anything at all, and none of them could fail in a fake.


def test_the_phrase_is_examined_after_it_ends_not_when_it_starts():
    """The window must hold the utterance, not the silence leading up to it.

    Examining the moment energy rises hands the recogniser the two seconds
    *before* the phrase. Measured on a real microphone that came back as ""
    and "Thank you." -- the standard hallucination on near-silence.
    """
    seen: list[float] = []

    class Recorder(ScriptedDetector):
        def detect(self, audio, phrase):
            seen.append(float(numpy.abs(audio).max()))
            return super().detect(audio, phrase)

    stream = ScriptedStream([quiet(1.0), loud(1.0), quiet(1.0)])
    session = controller(stream=stream, detector=Recorder(["hey sam"]))
    session.wake.start(lambda: None)
    time.sleep(1.2)
    session.wake.stop()

    assert seen, "the utterance was never examined"
    assert seen[0] > 0.1, f"examined silence (peak {seen[0]}): the phrase had not been said yet"


def test_the_window_reaches_back_to_the_first_syllable():
    """A whole sentence is one breath, and the phrase starts it.

    A fixed trailing window loses the phrase by the time a sentence ends, so
    the span examined has to be the utterance, not the last two seconds.
    """
    spans: list[float] = []

    class Recorder(ScriptedDetector):
        def detect(self, audio, phrase):
            spans.append(len(audio) / 16_000)
            return super().detect(audio, phrase)

    stream = ScriptedStream([quiet(0.3), loud(3.0), quiet(1.0)])
    session = controller(stream=stream, detector=Recorder(["hey sam"]))
    session.wake.start(lambda: None)
    time.sleep(2.0)
    session.wake.stop()

    assert spans, "the utterance was never examined"
    assert spans[0] >= 3.0, f"only {spans[0]:.1f}s examined; the start of the sentence was lost"


def test_a_quiet_moment_does_not_kill_the_listener():
    """An empty frame queue is a pause in the room, not a lost microphone.

    `frames()` returns when nothing arrives for a moment, which happens
    whenever transcription briefly outruns the audio callback. Treating that
    as the end of the stream let the listener exit after its first
    examination, and SAM never heard anything again.
    """

    class Exhausting:
        """Runs dry once, as a real queue does, then keeps delivering."""

        def __init__(self):
            self.generators = 0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def frames(self, timeout: float = 0.4):
            self.generators += 1
            if self.generators == 1:
                yield from loud(0.4)
                return  # the queue ran dry
            for _ in range(400):
                time.sleep(0.005)
                yield numpy.zeros(FRAME, dtype="float32")

    stream = Exhausting()
    session = controller(stream=stream, detector=ScriptedDetector([]))
    session.wake.start(lambda: None)
    time.sleep(0.5)
    alive = session.wake.running
    session.wake.stop()

    assert stream.generators > 1, "the listener gave up the first time the queue was empty"
    assert alive, "the wake listener died on a quiet moment"


def test_the_name_spelled_out_is_still_the_name():
    """A real microphone returned the name as an initialism, with full stops.

    The phrase was there and the match failed on punctuation alone.
    """
    assert phrase_heard("Hey, S.A.M. What is Gold doing right now?", "Hey SAM")
    assert phrase_heard("hey s.a.m.", "Hey SAM")
    # Still not a licence to fire on anything nearby.
    assert not phrase_heard("hey s a m", "Hey SAM")
    assert not phrase_heard("S.A.M.", "Hey SAM")


def test_the_bar_for_speech_rises_with_the_room():
    """A desk that sounded silent put 54% of frames above the fixed gate.

    Everything looked like speech, so the utterance never appeared to end and
    the detector was handed six seconds of fan noise with a phrase buried in
    it. The bar has to sit above whatever this room is already doing.
    """
    session = controller(stream=ScriptedStream([]), detector=ScriptedDetector([]))
    wake = session.wake
    quiet_bar = wake.threshold

    for _ in range(4000):  # a room humming well above the configured gate
        wake._track_noise(0.03, speech=True)
    noisy_bar = wake.threshold

    assert quiet_bar == wake.energy_gate, "a silent room should use the configured sensitivity"
    assert noisy_bar > 0.03, f"bar {noisy_bar} sits inside the noise, so noise reads as speech"
    assert noisy_bar < 0.2, "the bar climbed above ordinary speech, so nothing would be heard"

    for _ in range(400):  # and the room goes quiet again
        wake._track_noise(0.0, speech=False)
    assert wake.threshold == wake.energy_gate, "the bar never came back down"


def test_speaking_deafens_the_listener_whoever_asked():
    """Suppression belongs to the voice service, not to this loop.

    The manual voice endpoint speaks through the same service. If only the
    hands-free loop suppressed the listener, SAM would hear itself say the
    phrase through any other path and answer itself.
    """
    voice = FakeVoice()
    session = controller(voice=voice)
    voice.wake = session.wake

    # Nobody goes through the controller here: this is /api/voice/speak.
    voice.speak("Hey SAM. Hey SAM.")

    assert voice.speaking_while_suppressed == [True], "the listener was awake during playback"
    assert session.wake.suppressed is False, "and listening again afterwards"


def test_capturing_a_command_stands_the_listener_down():
    """Two listeners, one microphone.

    Left running, the wake listener transcribed the command as a wake
    candidate while the capture got nothing at all.
    """
    voice = FakeVoice(["what is gold doing"])
    session = controller(voice=voice)
    during: list[bool] = []
    original = voice.listen_once
    voice.listen_once = lambda **kw: (during.append(session.wake.suppressed), original(**kw))[1]

    session._capture()

    assert during == [True], "the wake listener was still holding the microphone"
    assert session.wake.suppressed is False, "and it never started listening again"


def test_silence_with_a_reason_says_the_reason():
    """Saying nothing is the wrong answer when the cause is known.

    A command language local speech cannot transcribe comes back as empty text
    plus an explanation. Swallowing it leaves the user with a microphone that
    appears deaf for no reason.
    """
    voice = FakeVoice()
    voice.listen_once = lambda **kw: {
        "captured": True,
        "text": "",
        "error": "No Sorani speech provider is configured.",
    }
    session = controller(voice=voice)

    session._turn()

    assert "Sorani" in session.status.detail, f"the user was told {session.status.detail!r}"
    assert "Sorani" in session.status.last_error


def test_nothing_is_examined_while_the_model_is_loading():
    """The first transcription of a cold model blocks for seconds.

    It runs on the thread draining the microphone, so every frame arriving
    during the load is dropped -- which is exactly the first wake phrase
    somebody says after switching hands-free on.
    """

    class Cold(ScriptedDetector):
        ready = False

    detector = Cold(["hey sam"])
    session = controller(stream=ScriptedStream([quiet(0.2), loud(0.5), quiet(0.8)]),
                         detector=detector)
    fired: list[int] = []
    session.wake.start(lambda: fired.append(1))
    time.sleep(1.0)
    session.wake.stop()

    assert detector.examined == 0, "a loading model was asked to transcribe on the audio thread"
    assert fired == [], "it woke on a model that had not loaded"
