"""Endpointer state machine (classifier decisions injected) and the classifier."""

from __future__ import annotations

from voice_helpers import quiet_frame, tone_frame

from sam.voice.vad import Endpointer, FrameClassifier


def run(ep: Endpointer, pattern: str, t0: float = 0.0):
    """'#' = voiced frame, '.' = silent frame; returns [(time, event)]."""
    events = []
    t = t0
    for ch in pattern:
        t += 0.03
        frame = tone_frame() if ch == "#" else quiet_frame()
        ev = ep.process(frame, ch == "#", t)
        if ev is not None:
            events.append((round(t, 2), ev))
    return events


def test_start_needs_sustained_speech_and_end_needs_silence():
    ep = Endpointer(silence_ms=600)
    events = run(ep, "..#.." + "#####" + "#" * 20 + "." * 25)
    kinds = [e.kind for _, e in events]
    assert kinds == ["start", "end"]
    start_t, end_ev = events[0][0], events[1][1]
    assert start_t == 0.27                      # the blip + 4 burst frames = 5 voiced in the 240 ms window
    assert not end_ev.too_short and 700 <= end_ev.speech_ms <= 800
    assert abs(end_ev.eos_at - 0.9) < 1e-6      # last voiced frame, not the endpoint decision
    # pre-roll + speech + <= 200 ms trailing silence are kept
    assert len(end_ev.pcm) // 960 >= 26 + 2


def test_blip_is_marked_too_short():
    ep = Endpointer(silence_ms=300, min_speech_ms=250)
    events = run(ep, "######" + "." * 12)
    assert [e.kind for _, e in events] == ["start", "end"]
    assert events[1][1].too_short


def test_pause_shorter_than_silence_keeps_one_utterance():
    ep = Endpointer(silence_ms=600)
    events = run(ep, "#" * 20 + "." * 15 + "#" * 20 + "." * 21)
    assert [e.kind for _, e in events] == ["start", "end"]


def test_max_utterance_forces_an_end():
    ep = Endpointer(silence_ms=600, max_utterance_s=1.5)
    events = run(ep, "#" * 80)
    ends = [e for _, e in events if e.kind == "end"]
    assert ends and ends[0].forced and not ends[0].too_short
    assert len(ends[0].pcm) // 960 == 50          # cut at 1.5 s, then a new utterance starts


def test_classifier_rejects_silence_and_quiet_noise():
    fc = FrameClassifier()
    assert not fc.is_speech(quiet_frame(), 0.0)
    assert not fc.is_speech(tone_frame(20), 0.001)   # below the energy floor, webrtcvad not even asked
