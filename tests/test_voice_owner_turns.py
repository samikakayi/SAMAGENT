"""The user's real session on 2026-09-25 (sam2.log + activity/timings/turns), replayed with fakes.

1. Enrolled (5 clips, consistency 0.686, -33.9 dBFS), then his OWN voice was
   rejected four times: scores 0.055 / 0.346 / 0.212 / 0.224 at -36.4 / -25.9 /
   -13.4 / -25.6 dBFS against the old threshold 0.40.
2. «وەڵاهی جارێ ترێیت ملیۆم لۆ بکەوە» + «بڕۆ سەر چار چی دەکەی؟» became two
   turns (end of speech 600 ms; the first turn was cancelled and a merged one
   re-ran -- an extra model call).
3. «کوڕە دەنگی بنەکەرە!» ("stop talking") went to the model; the voice engine
   must offer an immediate stop, and «بەسە / بوەستە / بێدەنگ بە» over SAM's voice
   must stop it at once.
4. Two overlapping turns both spoke: the older turn's queued audio and its
   confirmation question after the newer turn started.

Synthetic embeddings reproduce the real scores (no audio is played or
recorded, no network): the enrollment is e0; the owner's real-use voice shares a
"room/headset" direction e1 the enrollment lacks, which is why it scored so low.
"""

from __future__ import annotations

import array
import asyncio
import math
import time

import pytest
from test_voice_listening import EnergyClassifier, frame_at, notices, plain_store, say
from voice_helpers import FakeMic, FakeSpeaker, FakeStt, FakeTts, fake_llm, quiet_frame, settle

import sam.voice.engine as engine_mod
from sam.events import ConfirmRequest, Transcript, VoiceState
from sam.voice import strings
from sam.voice.cascade import CascadeVoice, is_stop_phrase
from sam.voice.engine import VoiceEngine
from sam.voice.gate import GateSettings, NearFieldGate
from sam.voice.hooks import RecordingHooks
from sam.voice.voiceprint import SENSITIVITY, SpeakerCheck, cosine, normalize, normalize_level
from sam.voice.vad import Endpointer

DIMS = 16
REAL_SCORES = (0.055, 0.346, 0.212, 0.224)          # the owner vs his enrollment, 2026-09-25
REAL_LEVELS = (-36.4, -25.9, -13.4, -25.6)          # dBFS of those utterances
ENROLL_LEVEL = -33.9


def unit(index: int) -> list[float]:
    vector = [0.0] * DIMS
    vector[index] = 1.0
    return vector


def owner_voice(score: float, jitter: int) -> list[float]:
    """The owner through the real room/headset: ``score`` against the enrollment (e0)."""
    room = normalize([0.95 * a + 0.3 * b for a, b in zip(unit(1), unit(jitter))])
    rest = math.sqrt(1 - score * score)
    return normalize([score * a + rest * b for a, b in zip(unit(0), room)])


def other_voice(jitter: int) -> list[float]:
    """Someone else (TV, family) heard through the same room."""
    return normalize([0.10 * a + 0.20 * b + 0.97 * c for a, b, c in zip(unit(0), unit(1), unit(jitter))])


ENROLLED = unit(0)
# pitch (the fake voice's square-wave period) -> embedding
TABLE = {3: owner_voice(REAL_SCORES[0], 2), 4: owner_voice(REAL_SCORES[1], 3), 5: owner_voice(REAL_SCORES[2], 4),
         6: owner_voice(REAL_SCORES[3], 5), 7: owner_voice(0.20, 6), 9: other_voice(10), 11: other_voice(11),
         8: owner_voice(0.25, 7)}


class TableEmbedder:
    """Speaker-embedding stand-in: the square wave's period picks a vector (loudness-proof by construction)."""

    name = "table"

    def __init__(self, path=None):
        self.calls = 0

    def load(self):
        pass

    def embed(self, pcm, rate=16000):
        self.calls += 1
        samples = array.array("h")
        samples.frombytes(pcm[: len(pcm) - len(pcm) % 2])
        flips = sum(1 for a, b in zip(samples, samples[1:]) if (a >= 0) != (b >= 0))
        period = max(1, round(len([s for s in samples if s]) / max(1, flips)))
        return list(TABLE.get(period, other_voice(12)))


@pytest.fixture
async def voice(make_app, monkeypatch):
    monkeypatch.setattr(engine_mod, "FrameClassifier", EnergyClassifier)
    engines = []

    async def build(*, stt=None, llm=None, llm_log=None, enrolled=True, tts=None):
        app = make_app()
        app.bus.bind_loop(asyncio.get_running_loop())
        events = []
        app.bus.subscribe(None, events.append)
        mics: list[FakeMic] = []
        speaker = FakeSpeaker()
        check = SpeakerCheck(app, embedder_factory=TableEmbedder, store=plain_store(app))
        if enrolled:
            check.store.save(ENROLLED, model="fake", level_db=ENROLL_LEVEL, clips=5, consistency=0.686)
            app.config.set("voice.gate_user_level_db", ENROLL_LEVEL)

        def mic_factory():
            mics.append(FakeMic())
            return mics[-1]

        eng = VoiceEngine(app, mic_factory=mic_factory, speaker=speaker, stt=stt or FakeStt(["نرخی زێڕ چەندە؟"]),
                          tts=tts or FakeTts(), llm_stream=llm or fake_llm(["باشە."], llm_log),
                          hotkey_factory=lambda keys, cb: type("H", (), {"start": lambda s: True, "stop": lambda s: None,
                                                                         "registered": True})(),
                          speaker_check=check)
        app.voice = eng
        await eng.start()
        engines.append(eng)
        return app, eng, mics, speaker, events

    yield build
    for eng in engines:
        await eng.stop()
        for task in (eng._watch_task, eng._selftest_task):  # noqa: SLF001
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)


def activity(app, name):
    return app.db.query("SELECT summary FROM activity WHERE name=? ORDER BY id", (name,))


async def click(eng) -> None:
    """The island click / the hotkey (listening closed first: a click while listening closes it)."""
    if eng.listening:
        await eng.stop_listening()
    assert await eng.toggle_listening() is True


# -- 1. only my voice: the owner's turn, low owner-calibrated thresholds, loudness, the hint --------------------

def test_the_synthetic_voices_reproduce_the_real_scores():
    scores = [round(cosine(TABLE[p], ENROLLED), 3) for p in (3, 4, 5, 6)]
    assert scores == list(REAL_SCORES)
    assert all(score < 0.40 for score in scores)                           # the old threshold rejected every one
    assert sum(score >= SENSITIVITY["normal"] for score in scores) == 3     # the new base keeps 3 of 4


async def test_the_owners_turn_after_a_click_is_never_rejected_and_adapts_the_voiceprint(voice):
    stt = FakeStt(["یەک", "دوو", "سێ", "چوار"])
    app, eng, mics, speaker, events = await voice(stt=stt)
    for index, (pitch, level) in enumerate(zip((3, 4, 5, 6), REAL_LEVELS)):
        await click(eng)
        say(mics[-1], db=level, pitch=pitch, frames=40)                     # 1.2 s, as loud as he really was
        assert await settle(lambda: len(stt.calls) == index + 1), (pitch, level)
        assert await settle(lambda: len((eng.speaker_check.profile() or {}).get("owners") or []) == index + 1)
    assert activity(app, "not_my_voice") == []                              # 0 rejections (was 4)
    assert not notices(events, "not_recognized")
    stored = eng.speaker_check.store.load()                                 # the same protected blob
    assert len(stored["owners"]) == 4 and stored["enroll"] == pytest.approx(ENROLLED)
    raw = app.db.query_one("SELECT blob FROM voice_profile WHERE id=1")["blob"]
    assert b"owners" not in bytes(raw)                                      # never stored in clear
    adapted = [row["summary"] for row in activity(app, "voiceprint_adapted")]
    assert len(adapted) == 4 and adapted[0].startswith("score=0.055")
    # The follow-up threshold follows how the owner really scores now (never below the base).
    assert SENSITIVITY["normal"] <= eng.speaker_check.threshold() <= 0.35


async def test_followups_pass_for_the_owner_and_reject_only_clearly_other_voices(voice):
    stt = FakeStt(["یەک", "دوو", "سێ", "چوار", "ئەی زیو؟"])
    log: list[str] = []
    app, eng, mics, speaker, events = await voice(stt=stt, llm_log=log)
    for index, (pitch, level) in enumerate(zip((3, 4, 5, 6), REAL_LEVELS)):
        await click(eng)
        say(mics[-1], db=level, pitch=pitch, frames=40)
        assert await settle(lambda: len((eng.speaker_check.profile() or {}).get("owners") or []) == index + 1)
    check = eng.speaker_check
    threshold = check.threshold()
    owner_follow_up, other = TABLE[7], TABLE[9]
    assert cosine(owner_follow_up, ENROLLED) == pytest.approx(0.20, abs=1e-3)   # the old policy (0.40): rejected
    profile = check.profile()["vector"]
    assert cosine(owner_follow_up, profile) >= threshold > cosine(other, profile)
    assert await settle(lambda: eng.listening_status()["followup_open"])    # SAM answered the 4th turn
    say(mics[-1], db=-25.0, pitch=9)                                        # the TV / family, close enough
    assert await settle(lambda: activity(app, "not_my_voice"))
    assert len(stt.calls) == 4
    say(mics[-1], db=-13.4, pitch=7)                                        # the owner, 20 dB over his enrollment
    assert await settle(lambda: len(stt.calls) == 5 and log[-1] == "ئەی زیو؟")
    assert eng.speaker_check.last.ok and eng.speaker_check.last.reason == "match"


async def test_a_rejected_followup_shows_the_hint_once_and_a_click_reopens_the_owners_turn(voice):
    stt = FakeStt(["نرخی زێڕ چەندە؟", "سڵاو"])
    app, eng, mics, speaker, events = await voice(stt=stt)
    await click(eng)
    say(mics[-1], pitch=8)
    assert await settle(lambda: eng.listening_status()["followup_open"])
    for _ in range(3):                                                      # repeated tries by another voice
        say(mics[-1], pitch=9)
    assert await settle(lambda: len(activity(app, "not_my_voice")) == 3)
    hints = notices(events, "not_recognized")
    assert len(hints) == 1 and hints[0].text_ckb == strings.VOICE_NOT_RECOGNIZED == "دەنگەکەت نەناسرایەوە — کلیک بکە"
    assert len(stt.calls) == 1
    assert await eng.toggle_listening() is True                             # «کلیک بکە» -> a click: "it is me"
    assert eng.listening and eng.listening_status()["owner_turn_pending"]
    say(mics[-1], pitch=9)                                                  # the owner's turn: never rejected
    assert await settle(lambda: len(stt.calls) == 2)
    assert len(activity(app, "not_my_voice")) == 3


def test_threshold_follows_the_owners_leave_one_out_scores(make_app):
    app = make_app()
    check = SpeakerCheck(app, embedder_factory=TableEmbedder, store=plain_store(app))
    check.store.save(ENROLLED, model="fake", level_db=ENROLL_LEVEL, clips=5, consistency=0.686)
    check.forget_cache()
    assert check.threshold() == 0.20 and check.threshold(speech_ms=600) == 0.15

    async def learn(vectors):
        for vector in vectors:
            assert await check.learn_owner(vector, speech_ms=1500)
    asyncio.run(learn([TABLE[3]]))                                          # 0.055: low -> stays at the base
    assert check.owner_scores() == [pytest.approx(0.055, abs=1e-3)] and check.threshold() == 0.20
    asyncio.run(learn([TABLE[4], TABLE[5], TABLE[6]]))
    assert check.threshold() == pytest.approx(0.25, abs=0.02)               # q25(loo) - 0.15, the real numbers
    same = [owner_voice(0.25, 2)] * 6                                       # the owner's voice, consistently
    asyncio.run(learn(same))
    assert check.threshold() == 0.35                                        # capped: other voices far below
    assert check.status()["owner_utterances"] == 10                         # bounded running average
    assert asyncio.run(check.learn_owner(TABLE[3], speech_ms=500)) is None  # too short to adapt


def test_loudness_is_normalised_before_embedding(make_app):
    class LoudnessEmbedder(TableEmbedder):
        def embed(self, pcm, rate=16000):
            samples = array.array("h")
            samples.frombytes(pcm)
            rms = math.sqrt(sum(s * s for s in samples) / max(1, len(samples))) / 32768.0
            bucket = max(0, min(DIMS - 1, int(-20 * math.log10(max(rms, 1e-6)) / 4)))
            return unit(bucket)                                             # a model that WOULD hear loudness

    app = make_app()
    check = SpeakerCheck(app, embedder_factory=LoudnessEmbedder, store=plain_store(app))
    quiet = frame_at(-36.4, pitch=5) * 40
    loud = frame_at(-13.4, pitch=5) * 40
    enrolled = frame_at(ENROLL_LEVEL, pitch=5) * 40
    vectors = [asyncio.run(check.embed(pcm)) for pcm in (quiet, loud, enrolled)]
    assert vectors[0] == vectors[1] == vectors[2]
    normalized = array.array("h")
    normalized.frombytes(normalize_level(frame_at(-2.0, pitch=5) * 10))
    assert max(abs(s) for s in normalized) <= 0.97 * 32768                  # peak-limited, never clipped further


def test_the_gate_accepts_the_owner_20_db_louder_than_enrollment():
    gate = NearFieldGate(GateSettings(user_level_db=ENROLL_LEVEL), clock=lambda: 100.0)
    for _ in range(60):
        gate.classify(10 ** (-70 / 20), False)                              # a quiet room
    assert gate.classify(10 ** (-13.4 / 20), True)                          # frustrated and loud
    assert gate.classify(10 ** (-36.4 / 20), True)                          # quieter than enrollment
    assert gate.learn_user_level(-13.4) is None                             # an outlier: not learned, not rejected
    assert gate.classify(10 ** (-13.4 / 20), True)


# -- 2. one utterance split by a pause is one turn ------------------------------------------------------------------

def test_end_of_speech_waits_900_ms_so_a_short_pause_is_one_utterance(make_app):
    app = make_app()
    eng = VoiceEngine(app, speaker=FakeSpeaker(), stt=FakeStt(), tts=FakeTts())
    assert app.config.get("voice.end_silence_ms") == 900 and eng._end_silence_ms() == 900  # noqa: SLF001
    endpointer = Endpointer(silence_ms=900)
    events = []
    t = 0.0
    for voiced in [True] * 30 + [False] * 23 + [True] * 30 + [False] * 31:  # a 0.7 s pause (the real split)
        t += 0.03
        event = endpointer.process(b"\x00\x00" * 480, voiced, t)
        if event is not None:
            events.append(event.kind)
    assert events == ["start", "end"]


async def test_a_continuation_right_after_an_utterance_is_one_request_and_one_model_call(voice):
    """«وەڵاهی جارێ ترێیت ملیۆم لۆ بکەوە» + «بڕۆ سەر چار چی دەکەی؟»: the second
    starts right after the first ended (a longer pause than 0.9 s) while the
    first is still being transcribed -- no cancelled turn, one model call."""
    first, second = "وەڵاهی جارێ ترێیت ملیۆم لۆ بکەوە", "بڕۆ سەر چار چی دەکەی؟"
    log: list[str] = []
    stt = FakeStt([first, second], delay=0.3)
    app, eng, mics, speaker, events = await voice(stt=stt, llm_log=log, enrolled=False)
    await click(eng)
    mics[-1].push(frame_at(-20.0), 30)
    mics[-1].push(quiet_frame(), 31)                                        # the end of the first utterance ...
    say(mics[-1], db=-21.0, frames=30)                                      # ... and at once the rest
    assert await settle(lambda: log, timeout=4.0)
    await asyncio.sleep(0.2)
    assert log == [f"{first} {second}"] and len(stt.calls) == 2
    users = [e.text for e in events if isinstance(e, Transcript) and e.role == "user"]
    assert users == [f"{first} {second}"]                                   # one user turn, not three
    totals = app.db.query("SELECT extra FROM timings WHERE kind='cascade' AND stage='total' ORDER BY id")
    outcomes = [row["extra"] for row in totals]
    assert any('"merged"' in extra for extra in outcomes) and not any('"cancelled"' in extra for extra in outcomes)
    assert eng.cascade.merged == 1


async def test_a_continuation_after_the_reply_started_silently_is_carried_into_one_answer(voice):
    first, second = "بڕۆ 100 چار 3 خولەکی یەکسەر", "لە گوڵت."
    log: list[str] = []
    release = asyncio.Event()
    app, eng, mics, speaker, events = await voice(stt=FakeStt([first, second]), enrolled=False,
                                                  llm=fake_llm([release, "باشە."], log))
    await click(eng)
    mics[-1].push(frame_at(-20.0), 30)
    mics[-1].push(quiet_frame(), 31)
    assert await settle(lambda: log == [first])                             # its reply started, silent
    say(mics[-1], db=-20.0, frames=20)
    assert await settle(lambda: len(log) == 2)
    assert log[1] == f"{first} {second}"
    release.set()
    assert await settle(lambda: "باشە." in eng.tts.texts)
    assert eng.tts.texts.count("باشە.") == 1                                # one answer is heard


async def test_quieter_speech_right_after_an_utterance_is_not_joined(voice):
    log: list[str] = []
    app, eng, mics, speaker, events = await voice(stt=FakeStt(["نرخی زێڕ چەندە؟"], delay=0.3), llm_log=log,
                                                  enrolled=False)
    await click(eng)
    mics[-1].push(frame_at(-20.0), 30)
    mics[-1].push(quiet_frame(), 31)
    say(mics[-1], db=-40.0, frames=30, pitch=3)                             # the TV across the room, right after
    assert await settle(lambda: log, timeout=3.0)
    await asyncio.sleep(0.2)
    assert log == ["نرخی زێڕ چەندە؟"] and len(eng.stt.calls) == 1


async def test_a_stop_phrase_right_after_an_utterance_cancels_it_instead_of_joining(voice):
    log: list[str] = []
    app, eng, mics, speaker, events = await voice(stt=FakeStt(["ترەیدینگ ڤیو بکەرەوە", "بەسە"], delay=0.3),
                                                  llm_log=log, enrolled=False)
    await click(eng)
    mics[-1].push(frame_at(-20.0), 30)
    mics[-1].push(quiet_frame(), 31)
    say(mics[-1], db=-20.0, frames=12)                                      # «بەسە» right after it
    assert await settle(lambda: log, timeout=4.0)
    await asyncio.sleep(0.2)
    assert log == ["بەسە"]                                                  # the held request was not answered
    users = [e.text for e in events if isinstance(e, Transcript) and e.role == "user"]
    assert users == ["ترەیدینگ ڤیو بکەرەوە", "بەسە"]                        # both kept in the history


async def test_a_cough_after_an_utterance_releases_the_held_request(voice):
    log: list[str] = []
    app, eng, mics, speaker, events = await voice(stt=FakeStt(["نرخی زێڕ چەندە؟"], delay=0.3), llm_log=log,
                                                  enrolled=False)
    await click(eng)
    mics[-1].push(frame_at(-20.0), 30)
    mics[-1].push(quiet_frame(), 31)
    mics[-1].push(frame_at(-20.0), 6)                                       # a cough: 180 ms
    mics[-1].push(quiet_frame(), 35)
    assert await settle(lambda: log == ["نرخی زێڕ چەندە؟"], timeout=3.0)
    assert len(eng.stt.calls) == 1 and eng.cascade.status()["held"] is False


# -- 3. stop talking ---------------------------------------------------------------------------------------------

def test_stop_phrases():
    for text in ("بەسە", "بوەستە!", "بێدەنگ بە", "سام بەسە", "کوڕە بەسە", "stop", "قسە مەکە", "ڕاوەستە", "بێدەنگبە"):
        assert is_stop_phrase(text), text
    for text in ("بوەستە لەسەر چارتەکە", "بەس نرخی زێڕ", "stop the alarm", "نرخی زێڕ چەندە؟", "", "بە"):
        assert not is_stop_phrase(text), text


@pytest.fixture
def cascade_env(make_app):
    def build(llm, *, stt=None):
        app = make_app()
        app.bus.bind_loop(asyncio.get_running_loop())
        events = []
        app.bus.subscribe(None, events.append)
        speaker, hooks = FakeSpeaker(), RecordingHooks()
        tts = FakeTts()
        cascade = CascadeVoice(app, speaker, stt or FakeStt(["نرخی زێڕ"]), tts, hooks, llm_stream=llm)
        return app, cascade, speaker, tts, events
    return build


async def test_a_spoken_stop_phrase_silences_sam_before_any_model_hears_it(cascade_env):
    gate = asyncio.Event()
    seen: list[tuple[str, int]] = []
    holder: dict = {}

    async def llm(text, turn):
        seen.append((text, holder["speaker"].flushes))
        if text == "بەسە":
            yield "باشە."
            return
        yield "یەکەم ڕستە تەواو بوو. "
        await gate.wait()
        yield "ئەمە هەرگیز نابیسترێت."

    app, cascade, speaker, tts, events = cascade_env(llm, stt=FakeStt(["نرخی زێڕ", "بەسە"]))
    holder["speaker"] = speaker
    cascade.submit_utterance(b"\x01\x00" * 8000, time.perf_counter())
    assert await settle(lambda: speaker.chunks)                             # SAM is talking
    before = speaker.flushes
    cascade.submit_utterance(b"\x01\x00" * 8000, time.perf_counter())       # «بەسە»
    assert await settle(lambda: len(seen) == 2)
    assert seen[1] == ("بەسە", before + 1)                                  # flushed BEFORE the brain was asked
    gate.set()
    await settle()
    assert "ئەمە هەرگیز نابیسترێت." not in " ".join(tts.texts)
    assert cascade._carry is None                                          # noqa: SLF001 - nothing carried over


async def test_a_stop_phrase_also_stops_an_alert_being_read(cascade_env):
    """Fixed speech (an alert, a worker summary) is not a turn: only the stop
    itself can end it before its last sentence."""
    app, cascade, speaker, tts, events = cascade_env(fake_llm(["باشە."]), stt=FakeStt(["بەسە"]))
    tts.delay = 0.05
    long_alert = "زێڕ گەیشتە ئاستی بەرگری. " + "ئەمە ڕستەیەکی درێژە بۆ ئەوەی بە چەند بەش بخوێنرێتەوە. " * 12
    speaking = asyncio.ensure_future(cascade.speak(long_alert, source="alert"))
    assert await settle(lambda: speaker.chunks)
    cascade.submit_utterance(b"\x01\x00" * 8000, time.perf_counter())       # «بەسە»
    await asyncio.wait_for(speaking, 3.0)
    pieces = len([t for t in tts.texts if t != "باشە."])
    assert pieces == 1                                                      # the rest of the alert was not read


async def test_the_brain_can_stop_older_speech_from_inside_its_own_turn(cascade_env):
    gate = asyncio.Event()
    holder: dict = {}

    async def llm(text, turn):
        if text == "کوڕە دەنگی بنەکەرە!":
            await holder["cascade"].stop_speaking()                          # the brain's "stop talking" intent
            yield "باشە."
            return
        yield "ڕستەیەکی درێژ. "
        await gate.wait()
        yield "کۆتایی."

    app, cascade, speaker, tts, events = cascade_env(llm, stt=FakeStt(["نرخی زێڕ", "کوڕە دەنگی بنەکەرە!"]))
    holder["cascade"] = cascade
    cascade.submit_utterance(b"\x01\x00" * 8000, time.perf_counter())
    assert await settle(lambda: speaker.chunks)
    cascade.submit_utterance(b"\x01\x00" * 8000, time.perf_counter())
    assert await settle(lambda: "باشە." in tts.texts)                       # its own short answer is said
    gate.set()
    await settle()
    assert "کۆتایی." not in tts.texts


async def test_a_short_stop_word_over_sams_voice_stops_it(voice):
    """«بەسە» is ~0.3 s: too short to cut SAM off (400 ms), yet it stops him."""
    app, eng, mics, speaker, events = await voice(stt=FakeStt(["بەسە"]), enrolled=False)
    app.config.set("voice.gate_user_level_db", -20.0)                       # the user's level is known
    await eng.start_listening(explicit=False)
    speaker.playing = True                                                  # SAM is reading an answer
    flushes = speaker.flushes
    mics[-1].push(frame_at(-20.0), 11)                                      # ~330 ms
    mics[-1].push(quiet_frame(), 35)
    assert await settle(lambda: speaker.flushes > flushes, timeout=3.0)
    assert await settle(lambda: getattr(speaker, "gain", 1.0) == 1.0)
    assert eng.stt.calls


async def test_a_short_other_word_over_sams_voice_is_not_a_request(voice):
    log: list[str] = []
    app, eng, mics, speaker, events = await voice(stt=FakeStt(["ئەها"]), enrolled=False, llm_log=log)
    app.config.set("voice.gate_user_level_db", -20.0)
    await eng.start_listening(explicit=False)
    speaker.playing = True
    flushes = speaker.flushes
    mics[-1].push(frame_at(-20.0), 11)
    mics[-1].push(quiet_frame(), 35)
    assert await settle(lambda: eng.stt.calls, timeout=3.0)
    await asyncio.sleep(0.2)
    assert log == [] and speaker.flushes == flushes                         # SAM goes on


async def test_engine_stop_speaking_is_immediate_for_sync_and_async_callers(voice):
    app, eng, mics, speaker, events = await voice(enrolled=False)
    flushes = speaker.flushes
    eng.stop_speaking_now()
    await eng.stop_speaking()
    assert speaker.flushes == flushes + 2 and not eng.listening             # the mic is not touched


# -- 4. only the newest turn is heard -----------------------------------------------------------------------------

async def test_an_older_turns_audio_is_dropped_when_a_newer_turn_starts(cascade_env):
    gate = asyncio.Event()

    async def llm(text, turn):
        if text == "دووەم":
            yield "وەڵامی نوێ."
            return
        yield "ئێستا دەیکەم. "                                              # the older turn's acknowledgement
        await gate.wait()                                                   # its tool runs ...
        yield "وەڵامی کۆن."

    app, cascade, speaker, tts, events = cascade_env(llm, stt=FakeStt(["یەکەم", "دووەم"]))
    running: list[dict] = []
    app.tools.running = lambda: running
    cascade.submit_utterance(b"\x01\x00" * 8000, time.perf_counter())
    assert await settle(lambda: speaker.chunks)
    running.append({"call_id": "t1", "name": "tv_open", "source": "cascade"})   # inside a tool: kept, muted
    flushes = speaker.flushes
    cascade.submit_utterance(b"\x01\x00" * 8000, time.perf_counter())
    assert await settle(lambda: "وەڵامی نوێ." in tts.texts)
    assert speaker.flushes == flushes + 1                                   # the older turn's queued audio dropped
    running.clear()
    gate.set()
    await settle()
    assert "وەڵامی کۆن." not in tts.texts                                   # the older turn never speaks again


async def test_an_older_turns_confirmation_question_is_not_read_out(voice):
    """Real use: the older turn's tool asked «TradingView دەبێت دابخرێت ...؟»
    after the newer turn had started, and it was read out."""
    release = asyncio.Event()
    answers: dict = {}
    holder: dict = {}

    async def llm(text, turn):
        app = holder["app"]
        if text == "یەکەم":
            await release.wait()                                            # (inside its tool)
            answers["old"] = await app.confirm.confirm("کۆن: TradingView دابخەم؟", tool_name="tv_open",
                                                       timeout_s=0.5)
            yield "کۆن."
        else:
            answers["new"] = await app.confirm.confirm("نوێ: TradingView دابخەم؟", tool_name="tv_open",
                                                       timeout_s=0.5)
            yield "نوێ."

    app, eng, mics, speaker, events = await voice(enrolled=False, llm=llm)
    holder["app"] = app
    running: list[dict] = []
    app.tools.running = lambda: running
    eng.cascade.submit_utterance(b"", time.perf_counter(), text="یەکەم")
    await settle()
    running.append({"call_id": "t1", "name": "tv_set_chart", "source": "cascade"})
    eng.cascade.submit_utterance(b"", time.perf_counter(), text="دووەم")
    assert await settle(lambda: "new" in answers, timeout=3.0)
    assert "نوێ: TradingView دابخەم؟" in eng.tts.texts                      # the newest turn's question is heard
    release.set()
    assert await settle(lambda: "old" in answers, timeout=3.0)
    assert await settle(lambda: any(isinstance(e, ConfirmRequest) and e.question_ckb.startswith("کۆن")
                                    for e in events))                      # the card still shows it
    await settle()
    assert "کۆن: TradingView دابخەم؟" not in eng.tts.texts                  # but it is not read out
    assert eng.cascade.stale_dropped >= 1
    running.clear()


async def test_status_reports_turns_and_the_listening_policy(voice):
    app, eng, mics, speaker, events = await voice()
    status = eng.status()
    assert status["end_silence_ms"] == 900 and set(status["turns"]) >= {"generation", "held", "merged"}
    assert status["voiceprint"]["base_threshold"] == 0.2 and "owner_scores" in status["voiceprint"]
    assert status["listening_window"]["owner_turn_pending"] is False
    await click(eng)
    assert eng.status()["listening_window"]["owner_turn_pending"] is True
    assert [e for e in events if isinstance(e, VoiceState)][-1].state == "listening"
