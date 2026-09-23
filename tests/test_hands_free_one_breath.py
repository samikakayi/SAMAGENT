"""Hands-free, in one breath: "Hey SAM, open Gold" without waiting to be asked.

The wake listener used to examine an utterance, confirm the phrase, and only
then open a new capture for the command -- so whatever followed the phrase in
the same breath had already been spent on wake detection and was gone. These
tests pin the fix: the words after the phrase belong to the command, they are
carried forward on the stream that heard them, and they go to the configured
command recogniser -- never the English wake transcript, and never anything
from before the phrase.

Synthetic audio makes the boundary exact: silence before, the phrase at one
level, the command at another. Whatever reaches the command recogniser can be
checked sample by sample for what it should and should not contain.
"""

from __future__ import annotations

import time

import numpy
import pytest

from sam_backend.wake import (
    WakeDetection,
    phrase_end,
    phrase_heard,
)
from tests.test_hands_free_voice import (
    FakeVoice,
    ScriptedDetector,
    ScriptedStream,
    controller,
    frames_of,
    quiet,
)

PHRASE_LEVEL = 0.2   # "Hey SAM"
COMMAND_LEVEL = 0.3  # whatever follows it


def phrase(seconds: float = 0.6) -> list:
    return frames_of(PHRASE_LEVEL, seconds)


def command(seconds: float) -> list:
    return frames_of(COMMAND_LEVEL, seconds)


def wait_for(predicate, seconds: float = 5.0) -> bool:
    limit = time.monotonic() + seconds
    while time.monotonic() < limit:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class LocatingDetector(ScriptedDetector):
    """Finds the phrase the way the real detector does: by where it ends.

    The phrase is the first loud sound in the window and lasts
    `phrase_seconds`; the answer is that moment, in seconds into the window.
    """

    def __init__(self, phrase_seconds: float = 0.6, heard: bool = True):
        super().__init__([])
        self.phrase_seconds = phrase_seconds
        self.heard = heard
        self.windows: list[float] = []

    def locate(self, audio, wake_phrase):
        self.examined += 1
        self.windows.append(len(audio) / 16_000)
        if not self.heard:
            return None
        loud = numpy.flatnonzero(numpy.abs(audio) > 0.1)
        if not len(loud):
            return None
        return loud[0] / 16_000 + self.phrase_seconds

    def detect(self, audio, wake_phrase):
        return self.locate(audio, wake_phrase) is not None


class OneBreathVoice(FakeVoice):
    """FakeVoice that can also transcribe audio it did not capture itself."""

    def __init__(self, utterances=None, transcripts=None):
        super().__init__(utterances)
        self._transcripts = list(transcripts or ["open gold"])
        self.transcribed: list[dict] = []

    def transcribe_captured(self, audio, language=None):
        self.transcribed.append({"audio": numpy.asarray(audio), "language": language})
        item = self._transcripts.pop(0) if self._transcripts else ""
        if isinstance(item, dict):
            return {"captured": True, **item}
        return {"captured": True, "text": item, "seconds": len(audio) / 16_000}


def run(stream_frames, *, voice=None, detector=None, language="ckb-IQ", continuation=0.0,
        replies=None, stream=None, onset_seconds=1.0):
    voice = voice or OneBreathVoice()
    session = controller(stream=stream or ScriptedStream(stream_frames), voice=voice,
                         detector=detector or LocatingDetector(), continuation=continuation,
                         replies=replies or ["Gold is trading quietly."])
    session.settings.voice_language = language
    # Shorter than production so a wake-only test does not sit out five
    # seconds; the tests about waiting set it back.
    session.wake.onset_seconds = onset_seconds
    states: list[str] = []
    session.subscribe(lambda status: states.append(status.state.value))
    session.start()
    return session, voice, states


def finish(session, voice, *, spoken: int = 1, seconds: float = 6.0):
    try:
        assert wait_for(lambda: len(voice.spoken) >= spoken, seconds), (
            f"SAM never answered; states={session.status.state.value}, "
            f"error={session.status.last_error!r}")
        assert wait_for(lambda: not session._busy.is_set(), seconds)
    finally:
        session.stop()


def kept_of(audio) -> int:
    """How many command samples reached the recogniser."""
    return int(numpy.isclose(audio, COMMAND_LEVEL).sum())


def full(frames) -> int:
    """How many samples a run of frames holds -- all of it must arrive."""
    return sum(len(frame) for frame in frames)


def levels(audio) -> set[float]:
    return {round(float(x), 2) for x in numpy.unique(numpy.round(audio, 2))}


# --- 1 / 7. the phrase alone: the two-step flow still works ---------------------


def test_the_phrase_alone_still_waits_for_the_command():
    session, voice, states = run([quiet(0.3), phrase(), quiet(4.0)],
                                 voice=OneBreathVoice(["what is gold doing"]))
    finish(session, voice)

    assert voice.transcribed == [], "silence after the phrase was sent to be transcribed"
    assert voice.listen_calls == 1, "nothing followed the phrase, so SAM must ask for the command"
    assert session.transcripts == ["what is gold doing"]


def test_nothing_after_the_phrase_takes_the_listening_path():
    session, voice, states = run([quiet(0.3), phrase(), quiet(4.0)],
                                 voice=OneBreathVoice(["open gold"]))
    finish(session, voice)

    assert "WAKE_DETECTED" in states and "LISTENING" in states
    assert states.index("WAKE_DETECTED") < states.index("LISTENING")


# --- 2 / 3. the words after the phrase are the command --------------------------


def test_the_command_in_the_same_breath_is_kept():
    session, voice, states = run([quiet(0.3), phrase(), command(0.9), quiet(1.5)],
                                 voice=OneBreathVoice(transcripts=["open gold"]))
    finish(session, voice)

    assert voice.listen_calls == 0, "the command was already spoken; asking again loses it"
    assert len(voice.transcribed) == 1
    heard = voice.transcribed[0]["audio"]
    assert COMMAND_LEVEL in levels(heard), "the command itself never reached the recogniser"
    assert PHRASE_LEVEL not in levels(heard), "the wake phrase was sent as part of the command"
    assert sum(numpy.isclose(heard, COMMAND_LEVEL)) / 16_000 >= 0.89, "the command was cut short"
    assert session.transcripts == ["open gold"]
    assert "TRANSCRIBING" in states and "THINKING" in states and "SPEAKING" in states


def test_a_sorani_command_after_an_english_phrase_goes_to_the_command_recogniser():
    """English wake, Sorani command, one breath.

    The wake recogniser only ever hears English; what it made of the Sorani is
    irrelevant. The command audio goes to whatever the configured language
    uses -- for this user, the Sorani provider -- and its transcript is what
    SAM answers.
    """
    sorani = "ئێستا گۆڵد چۆنە؟"
    session, voice, _ = run([quiet(0.3), phrase(), command(1.2), quiet(1.5)],
                            voice=OneBreathVoice(transcripts=[sorani]), language="ckb-IQ")
    finish(session, voice)

    assert voice.transcribed[0]["language"] == "ckb-IQ"
    assert session.transcripts == [sorani]
    assert session.status.last_transcript == sorani


def test_captured_audio_is_transcribed_by_the_configured_language(monkeypatch):
    """At the voice service: Sorani goes to the Sorani path, English to Whisper."""
    from sam_backend.config import Settings
    from sam_backend.voice import VoiceService

    service = VoiceService(Settings.from_env())
    routes: list[str] = []
    monkeypatch.setattr(service, "_transcribe_sorani",
                        lambda audio, seconds: routes.append("sorani") or {"text": "سڵاو"})
    monkeypatch.setattr(service.stt, "transcribe",
                        lambda audio, language=None: routes.append(f"whisper:{language}") or {"text": "hello"})
    audio = numpy.full(16_000, 0.3, dtype="float32")

    assert service.transcribe_captured(audio, language="ckb-IQ")["text"] == "سڵاو"
    assert service.transcribe_captured(audio, language="en-US")["text"] == "hello"
    assert routes == ["sorani", "whisper:en-US"]


# --- 4 / 15. finding where the phrase ends --------------------------------------


@pytest.mark.parametrize("words, expected", [
    # at the start
    ([(" Hey,", 0.1, 0.6), (" SAM", 0.7, 0.9), (" open", 1.0, 1.2), (" Gold.", 1.2, 1.5)], 0.9),
    # an initialism split into pieces, as a real transcript returned it
    ([(" Hey", 0.1, 0.6), (" S", 0.6, 0.8), (".A", 0.8, 0.9), (".M", 0.9, 1.1), (" tell", 1.1, 1.3)], 1.1),
    ([(" Hey,", 0.1, 0.6), (" SA", 0.7, 0.9), ("-M", 0.9, 1.0), (" Open,", 1.0, 1.3)], 1.0),
    ([(" Hey,", 0.1, 0.6), (" S.A.M.,", 0.7, 1.0), (" tell", 1.1, 1.3)], 1.0),
    # exactly what the room returned for "Hey SAM what is gold doing right now"
    ([(" Hey,", 0.1, 0.4), (" S", 0.4, 0.58), ("-A", 0.58, 0.72), ("-M,", 0.72, 0.9),
      (" what", 0.9, 1.16)], 0.9),
    # in the middle
    ([(" Okay,", 0.1, 0.5), (" hey,", 0.6, 0.8), (" Sam", 0.9, 1.1), (" status.", 1.2, 1.6)], 1.1),
    # at the very end: nothing follows it
    ([(" So", 0.1, 0.3), (" hey", 0.4, 0.6), (" Sam.", 0.7, 0.9)], 0.9),
    # a following word glued on without a space must not be eaten by the cut
    ([(" Hey", 0.1, 0.5), (" Sam", 0.6, 0.9), (",what", 0.9, 1.2)], 0.9),
    ([("Hey", 0.1, 0.5), ("Sam", 0.6, 0.9)], 0.9),
    # broken times only ever move the cut later, never before earlier words
    ([(" okay", 0.1, 1.4), (" hey", 0.5, 0.8), (" Sam", 0.9, 1.1), (" gold", 1.4, 1.6)], 1.4),
    ([(" okay", 0.0, 0.0), (" hey", 0.0, 0.0), (" Sam", 0.0, 0.0)], 0.0),
    # not the phrase at all
    ([(" open", 0.1, 0.3), (" gold", 0.3, 0.6), (" hey", 0.7, 0.8), (" same", 0.8, 1.0)], None),
    ([(" they", 0.1, 0.3), (" sampled", 0.3, 0.8)], None),
    ([(" Hey,", 0.1, 0.6), (" Sammy", 0.7, 1.0)], None),
])
def test_the_end_of_the_phrase_is_found_in_the_words(words, expected):
    assert phrase_end(words, "Hey SAM") == expected


# --- 5 / 6. short and long commands ---------------------------------------------


def test_a_one_word_command_is_enough():
    """ "Hey SAM, gold." is a whole request; it must not be mistaken for silence."""
    session, voice, _ = run([quiet(0.3), phrase(), command(0.2), quiet(1.5)],
                            voice=OneBreathVoice(transcripts=["gold"]))
    finish(session, voice)

    assert voice.listen_calls == 0
    assert session.transcripts == ["gold"]


def test_a_long_command_is_not_cut_off_by_the_wake_window():
    """Longer than the wake buffer holds, and nothing may be lost from either end.

    The listener used to wait for the end of an utterance before examining
    it; a sentence longer than the buffer had lost its own beginning -- the
    phrase -- by then, and never woke SAM at all.
    """
    session, voice, _ = run([quiet(0.3), phrase(), command(6.4), quiet(1.5)],
                            voice=OneBreathVoice(transcripts=["open gold and analyse the chart"]))
    finish(session, voice, seconds=10.0)

    heard = voice.transcribed[0]["audio"]
    kept = sum(numpy.isclose(heard, COMMAND_LEVEL)) / 16_000
    assert kept >= 6.35, f"only {kept:.2f}s of a 6.4s command survived"
    assert PHRASE_LEVEL not in levels(heard)


# --- 8. when the command cannot be read -----------------------------------------


def test_a_command_that_cannot_be_transcribed_gets_a_second_chance():
    voice = OneBreathVoice(["open gold"], transcripts=[{"text": "", "error": "KurdishTTS: 503"}])
    session, voice, states = run([quiet(0.3), phrase(), command(0.8), quiet(1.5)], voice=voice)
    finish(session, voice)

    assert voice.listen_calls == 1, "the session was dropped instead of asking again"
    assert "KurdishTTS: 503" in session.status.last_error, "the provider's error was swallowed"
    assert session.transcripts[-1] == "open gold"


# --- 9. what must not wake it ---------------------------------------------------


@pytest.mark.parametrize("heard", [
    "hey same here", "hey sammy", "they sampled it", "an essay about gold", "I am here",
    "hey some", "say ham",
])
def test_near_misses_do_not_wake_it(heard):
    assert phrase_heard(heard, "Hey SAM") is False


@pytest.mark.parametrize("heard", ["Hey SAM", "Hey, SAM!", "Hey S.A.M.", "hey sam", "Hey SA-M, open",
                                   "Hey, S-A-M, what is school doing right now?"])
def test_the_phrase_in_its_usual_spellings_still_wakes_it(heard):
    assert phrase_heard(heard, "Hey SAM") is True


def test_speech_without_the_phrase_starts_nothing():
    detector = LocatingDetector(heard=False)
    session, voice, _ = run([quiet(0.3), command(1.0), quiet(1.5)], detector=detector)
    try:
        assert wait_for(lambda: detector.examined >= 1)
        time.sleep(0.2)
        assert voice.transcribed == [] and voice.listen_calls == 0
        assert session.status.state.value == "WAKE_LISTENING"
    finally:
        session.stop()


# --- 10. SAM's own voice --------------------------------------------------------


def test_a_suppressed_listener_hands_nothing_on():
    """What SAM says is dropped before examination, so it cannot become a command."""
    voice = OneBreathVoice()
    detector = LocatingDetector()
    session = controller(stream=ScriptedStream([quiet(0.3), phrase(), command(1.0), quiet(1.5)]),
                         voice=voice, detector=detector)
    session.wake.suppress()
    session.start()
    try:
        time.sleep(0.5)
        assert detector.examined == 0
        assert session.wake.take_detection() is None
        assert voice.transcribed == []
    finally:
        session.stop()


# --- 11. the follow-up window ---------------------------------------------------


def test_the_follow_up_window_still_needs_no_phrase():
    voice = OneBreathVoice(["and silver?"], transcripts=["open gold"])
    session, voice, _ = run([quiet(0.3), phrase(), command(0.8), quiet(1.5)], voice=voice,
                            continuation=5.0, replies=["Gold is quiet.", "Silver too."])
    finish(session, voice, spoken=2)

    assert session.transcripts[:2] == ["open gold", "and silver?"]
    assert voice.listen_calls >= 1, "the follow-up is captured the ordinary way"


# --- 12. privacy ----------------------------------------------------------------


def test_nothing_from_before_the_phrase_reaches_the_command_recogniser():
    """Only what follows the phrase may leave the machine, and only after it."""
    before = frames_of(0.05, 0.9)  # somebody talking before they addressed SAM
    session, voice, _ = run([before, quiet(0.7), phrase(), command(0.9), quiet(1.5)])
    finish(session, voice)

    heard = voice.transcribed[0]["audio"]
    assert 0.05 not in levels(heard), "speech from before the phrase was sent on"
    assert PHRASE_LEVEL not in levels(heard)


def test_a_detection_hands_over_only_what_followed():
    detection = WakeDetection(phrase_end=0.9)
    assert detection.speech is False and detection.audio is None
    assert detection.wait(0.01) is False, "an unfinished capture must not look finished"


# --- 13 / 14. one microphone, and a listener that keeps going -------------------


def test_one_breath_uses_the_stream_that_heard_it():
    stream = ScriptedStream([quiet(0.3), phrase(), command(0.9), quiet(1.5)])
    voice = OneBreathVoice()
    session = controller(stream=stream, voice=voice, detector=LocatingDetector())
    session.start()
    finish(session, voice)

    assert stream.opened == 1, "a second microphone stream was opened"
    assert voice.listen_calls == 0


def test_the_listener_survives_utterance_after_utterance():
    detector = LocatingDetector(heard=False)
    script = []
    for _ in range(4):
        script += [command(0.8), quiet(0.9)]
    session, voice, _ = run(script, detector=detector)
    try:
        assert wait_for(lambda: detector.examined >= 4)
        assert session.wake.running, "the listener died between utterances"
    finally:
        session.stop()


# --- the capture after an answer ------------------------------------------------


class PacedStream:
    """A microphone that delivers frames at a pace, so silence takes time."""

    def __init__(self, frames, *args, **kwargs):
        self._frames = [frame for group in frames for frame in group]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def frames(self, timeout: float = 0.4):
        for frame in self._frames:
            time.sleep(0.004)
            yield frame


def paced_listen(monkeypatch, script, **kwargs):
    import sam_backend.voice as voice_module
    from sam_backend.config import Settings

    monkeypatch.setattr(voice_module, "MicrophoneStream", lambda *a, **k: PacedStream(script))
    service = voice_module.VoiceService(Settings.from_env())
    kept: list[float] = []
    monkeypatch.setattr(service, "transcribe_captured",
                        lambda audio, **kw: kept.append(len(audio) / 16_000) or {"text": "x", "captured": True})
    service.listen_once(max_seconds=5, silence_ms=60, language="en-US", **kwargs)
    return kept


def test_a_click_is_not_a_question(monkeypatch):
    """With nobody pressing a button, one loud frame used to start a capture.

    In a room measured as quiet, isolated spikes over the fixed gate were
    enough to "capture" up to a second of nothing, which the Sorani provider
    -- no silence filter -- would transcribe into words. Hands-free now asks
    for a syllable of voice above the room's own threshold.
    """
    click = frames_of(0.2, 0.06)          # two frames: a key, a chair
    speech = frames_of(0.2, 0.6)
    # Paced at 4 ms a frame, so a 0.6 s gap outlasts the 60 ms silence window.
    kept = paced_listen(monkeypatch, [quiet(0.2), click, quiet(0.6), speech, quiet(0.6)],
                        threshold=0.010, min_speech_frames=5)

    assert len(kept) == 1
    assert 0.6 <= kept[0] < 0.6 + 0.6 + 0.05, f"{kept[0]:.2f}s kept: the click and the gap came with it"


def test_push_to_talk_is_unchanged(monkeypatch):
    """Without the hands-free arguments, capture starts on the first voiced frame, as before."""
    click = frames_of(0.2, 0.06)
    kept = paced_listen(monkeypatch, [quiet(0.2), click, quiet(0.4)])

    assert len(kept) == 1, "push-to-talk no longer captures what it used to"


def test_hands_free_asks_for_the_room_threshold_and_a_syllable():
    captured: list[dict] = []
    voice = OneBreathVoice(["open gold"])
    original = voice.listen_once
    voice.listen_once = lambda **kw: (captured.append(kw), original(**kw))[1]
    session = controller(voice=voice)

    session._capture()

    assert captured[0]["threshold"] == session.wake.command_threshold
    assert captured[0]["min_speech_frames"] >= 5


# --- what an independent review found -----------------------------------------


class TimedStream:
    """A microphone with a past and a present.

    `queued` arrives at once, as frames buffered during an examination do;
    `live` arrives at a pace, as the room does; after that it is quiet. With
    `report_queue`, it can say how much is still waiting, as MicrophoneStream
    does.
    """

    def __init__(self, queued, live=(), *, pace: float = 0.005, report_queue: bool = False,
                 queued_pace: float = 0.0):
        self._queued = [frame for group in queued for frame in group]
        self._live = [frame for group in live for frame in group]
        self._pace = pace
        self._queued_pace = queued_pace
        self._left = len(self._queued)
        if report_queue:
            self.pending = lambda: self._left
        self.opened = 0

    def __enter__(self):
        self.opened += 1
        return self

    def __exit__(self, *exc):
        return False

    def frames(self, timeout: float = 0.4):
        while self._queued:
            frame = self._queued.pop(0)
            self._left = len(self._queued)
            if self._queued_pace:
                time.sleep(self._queued_pace)
            yield frame
        while self._live:
            time.sleep(self._pace)
            yield self._live.pop(0)
        while True:
            time.sleep(self._pace)
            yield numpy.zeros(len(quiet(0.03)[0]), dtype="float32")


def test_a_command_started_after_the_cue_keeps_its_first_syllable():
    """The review's HIGH finding.

    After catching up with the room, the listener used to wait one second for
    the command and then hand over to a second capture -- right when people
    answer a "Listening..." cue, so "what is gold doing" arrived as "...is gold
    doing". It now waits on the same stream, and never counts while voice is
    arriving.
    """
    stream = TimedStream(queued=[quiet(0.3), phrase(), quiet(0.7)],
                         live=[quiet(1.2), command(0.8), quiet(1.2)], pace=0.005)
    session, voice, _ = run([], stream=stream, onset_seconds=5.0,
                            voice=OneBreathVoice(transcripts=["what is gold doing"]))
    finish(session, voice, seconds=10.0)

    assert voice.listen_calls == 0, "the command was handed to a second capture and clipped"
    heard = voice.transcribed[0]["audio"]
    assert kept_of(heard) == full(command(0.8)), "the start of the command was lost"


def test_clicks_after_the_phrase_do_not_add_up_to_a_command():
    """Review finding M2: five scattered loud frames were enough to finish.

    Three clicks, a gap, two more clicks used to be sent as a command, and the
    real command after them was never read. Each run of sound now has to be a
    syllable on its own.
    """
    click_level = 0.15
    session, voice, _ = run([quiet(0.3), phrase(), quiet(0.2),
                             frames_of(click_level, 0.09), quiet(0.9),
                             frames_of(click_level, 0.06), quiet(0.9),
                             command(0.8), quiet(1.5)],
                            voice=OneBreathVoice(transcripts=["open gold"]))
    finish(session, voice)

    heard = voice.transcribed[0]["audio"]
    assert click_level not in levels(heard), "the clicks were sent as part of the command"
    assert kept_of(heard) == full(command(0.8))


def test_a_command_already_queued_is_not_lost_to_a_slow_read():
    """Review finding M3: slow reads could look like having caught up.

    Two queued frames arriving a few milliseconds late started the wait for a
    command against audio that was still in the queue, and a command recorded
    during the examination was thrown away. The microphone now says how much
    it is holding.
    """
    stream = TimedStream(queued=[quiet(0.3), phrase(), quiet(0.7), quiet(1.5), command(0.8), quiet(1.2)],
                         report_queue=True, queued_pace=0.006)
    session, voice, _ = run([], stream=stream, onset_seconds=0.2,
                            voice=OneBreathVoice(transcripts=["open gold"]))
    finish(session, voice, seconds=10.0)

    assert voice.listen_calls == 0
    assert kept_of(voice.transcribed[0]["audio"]) == full(command(0.8))


def test_switching_off_mid_command_acts_on_nothing():
    """Review finding M4: half a command is not a smaller request.

    Stopping part-way used to transcribe and submit what had been heard so far
    -- "close all positions" without "except gold". Now nothing is sent, and
    the panel says hands-free is off.
    """
    stream = TimedStream(queued=[quiet(0.3), phrase()], live=[command(6.0), quiet(1.5)], pace=0.01)
    session, voice, states = run([], stream=stream)
    try:
        assert wait_for(lambda: "LISTENING" in states, 5.0)
        time.sleep(0.4)
    finally:
        session.stop()
    assert wait_for(lambda: not session._busy.is_set(), 5.0)

    assert voice.transcribed == [], "a cut-off command was sent to be transcribed"
    assert voice.spoken == [] and session.transcripts == []
    assert session.status.state.value == "OFF"


def test_a_silent_microphone_does_not_hold_the_session_hostage():
    """Review finding L1: a stalled device kept the loop waiting ~38 seconds.

    With no frames arriving at all, the capture could not notice time passing.
    It now does, and SAM asks for the command the ordinary way.
    """

    class Stalled:
        def __init__(self, script):
            self._script = [frame for group in script for frame in group]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def frames(self, timeout: float = 0.4):
            while self._script:
                yield self._script.pop(0)
            time.sleep(0.05)  # the queue stays empty: frames() gives up, as the real one does

    started = time.monotonic()
    session, voice, _ = run([], stream=Stalled([quiet(0.3), phrase(), quiet(0.7)]), onset_seconds=0.5,
                            voice=OneBreathVoice(["open gold"]))
    finish(session, voice, seconds=8.0)

    assert voice.listen_calls == 1
    assert time.monotonic() - started < 6.0


class FakeTimedModel:
    """A speech model that answers with the timed words it is given."""

    def __init__(self, words):
        self._words = words

    def transcribe(self, audio, **kwargs):
        from types import SimpleNamespace

        assert kwargs.get("word_timestamps") is True
        pieces = [SimpleNamespace(word=w, start=s, end=e) for w, s, e in self._words]
        return [SimpleNamespace(words=pieces)], None


class FakeShared:
    def __init__(self, words):
        self._model = FakeTimedModel(words)

    def available(self):
        return True

    def load(self):
        return self._model


@pytest.mark.parametrize("words, expected", [
    # words follow the phrase: cut right after it
    ([(" Hey,", 0.1, 0.5), (" Sam", 0.6, 0.9), (" open", 1.0, 1.2), (" gold", 1.2, 1.5)], 0.9),
    # nothing recognisable follows: the tail of "SAM" and the room are not a
    # command, so nothing in this window is carried forward (review M2)
    ([(" Hey,", 0.1, 0.5), (" Sam.", 0.6, 0.9)], 3.0),
    # times collapsed to zero: cutting at 0 would send what came before the
    # phrase to the cloud, so nothing is carried forward (review M1)
    ([(" okay", 0.0, 0.0), (" hey", 0.0, 0.0), (" Sam", 0.0, 0.0), (" gold", 0.0, 0.0)], 3.0),
    # not the phrase
    ([(" hey", 0.1, 0.5), (" same", 0.6, 0.9), (" here", 1.0, 1.2)], None),
])
def test_the_real_detector_only_carries_forward_what_it_heard(words, expected):
    from sam_backend.wake import LocalPhraseDetector

    detector = LocalPhraseDetector(shared=FakeShared(words))
    assert detector.locate(numpy.zeros(48_000, dtype="float32"), "Hey SAM") == expected
