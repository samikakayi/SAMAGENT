"""Deliberate, noise-robust, quota-safe listening (2026-09-24 evening fixes).

Real use that night: the TV and a family conversation were transcribed as
commands in the always-open window (149 KurdishTTS STT calls, every free model
quota used up), and a Gemini TTS 429 + the SDK's own retry left the island on
«بیردەکەمەوە» for ~40 s. Covered here with fakes (no mic, no speakers, no
network): push-to-talk windows, the always-listening name check, the
near-field gate, Live sending only accepted utterances, "only my voice", and
the voice enrollment flow.
"""

from __future__ import annotations

import array
import asyncio
import math

import pytest
from conftest import FAKE_GEMINI
from voice_helpers import FakeLiveClient, FakeMic, FakeSpeaker, FakeStt, FakeTts, fake_llm, quiet_frame, settle

import sam.voice.engine as engine_mod
from sam.events import Transcript, VoiceState
from sam.voice import strings
from sam.voice.engine import VoiceEngine
from sam.voice.gate import GateSettings, NearFieldGate, level_db
from sam.voice.listening import asks_enrollment, starts_with_name
from sam.voice.live import LiveVoice
from sam.voice.notices import VoiceEnrollRequest, VoiceNotice
from sam.voice.voiceprint import SpeakerCheck, VoiceprintStore


def frame_at(db: float, *, pitch: int = 8) -> bytes:
    """A 30 ms square-wave frame at ``db`` dBFS RMS; ``pitch`` tags the 'voice'."""
    amp = int(32767 * 10 ** (db / 20))
    return array.array("h", [amp if (i // pitch) % 2 else -amp for i in range(480)]).tobytes()


class EnergyClassifier:
    """VAD stand-in: any frame above -60 dBFS is 'voiced' (the gate decides near/far)."""

    def __init__(self, **kwargs):
        pass

    def is_speech(self, frame, rms):
        return rms > 0.001


class PitchEmbedder:
    """Speaker-embedding stand-in: the 'voice' is the square wave's period."""

    name = "fake"

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
        vector = [0.0] * 32
        vector[min(31, period)] = 1.0
        vector[0] = 0.2
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector]


def plain_store(app):
    return VoiceprintStore(app.db, protect=lambda b: b[::-1], unprotect=lambda b: bytes(b)[::-1])


@pytest.fixture
async def voice(make_app, monkeypatch):
    monkeypatch.setattr(engine_mod, "FrameClassifier", EnergyClassifier)
    engines = []

    async def build(*, gemini=False, llm_log=None, stt=None, enrolled_pitch=None, client=None, llm=None):
        app = make_app(env_text=f"GEMINI_API_KEY={FAKE_GEMINI}\n" if gemini else "")
        if gemini:
            app.config.set("voice.auto_live", True)   # these tests drive Live through "auto" (off by default)
            app.config.set("voice.selftest", {"ok": True, "cer": 0.1, "at": 1.0})
        app.bus.bind_loop(asyncio.get_running_loop())
        events = []
        app.bus.subscribe(None, events.append)
        mics: list[FakeMic] = []
        speaker = FakeSpeaker()
        client = client or FakeLiveClient()
        check = SpeakerCheck(app, embedder_factory=PitchEmbedder, store=plain_store(app))
        if enrolled_pitch is not None:
            vector = PitchEmbedder().embed(frame_at(-20, pitch=enrolled_pitch) * 30)
            check.store.save(vector, model="fake", level_db=-20.0, clips=5, consistency=0.9)

        def mic_factory():
            mics.append(FakeMic())
            return mics[-1]

        eng = VoiceEngine(app, mic_factory=mic_factory, speaker=speaker, stt=stt or FakeStt(["نرخی زێڕ چەندە؟"]),
                          tts=FakeTts(), llm_stream=llm or fake_llm(["نرخی زێڕ ٢٦٥٠ دۆلارە."], llm_log),
                          hotkey_factory=lambda keys, cb: type("H", (), {"start": lambda s: True, "stop": lambda s: None,
                                                                         "registered": True})(),
                          live_factory=lambda: LiveVoice(app, speaker, eng, client_factory=lambda key: client,
                                                         reconnect_delays=(0.01,), ready_timeout_s=2.0),
                          speaker_check=check)
        app.voice = eng
        await eng.start()
        engines.append(eng)
        return app, eng, mics, speaker, client, events

    yield build
    for eng in engines:
        await eng.stop()
        for task in (eng._watch_task, eng._selftest_task):  # noqa: SLF001
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)


def say(mic: FakeMic, db: float = -20.0, frames: int = 30, pitch: int = 8, silence: int = 35) -> None:
    mic.push(frame_at(db, pitch=pitch), frames)
    mic.push(quiet_frame(), silence)


def notices(events, kind=None):
    return [e for e in events if isinstance(e, VoiceNotice) and (kind is None or e.kind == kind)]


# -- (a) push-to-talk windows ------------------------------------------------------------------------------

async def test_without_a_voiceprint_or_level_each_request_needs_a_click(voice):
    """No way to tell the user from the TV yet: no follow-up window at all
    (the review's closed loop: TV -> STT -> model -> answer -> window -> TV)."""
    log: list[str] = []
    app, eng, mics, speaker, _, events = await voice(llm_log=log)
    await eng.toggle_listening()
    say(mics[0])
    assert await settle(lambda: log == ["نرخی زێڕ چەندە؟"])
    assert await settle(lambda: any(isinstance(e, VoiceState) and e.detail == "turn_end" for e in events),
                        timeout=3.0)                                          # closed right after the answer
    assert not eng.listening
    last = [e for e in events if isinstance(e, VoiceState)][-1]
    assert (last.state, last.detail) == ("idle", "turn_end")
    assert notices(events, "closed")[-1].text_ckb == strings.LISTEN_CLOSED_ENROLL
    assert "sleeping" not in [e.state for e in events if isinstance(e, VoiceState)]  # same conversation
    say(mics[0])                                                              # the TV after the answer
    await asyncio.sleep(0.2)
    assert len(eng.stt.calls) == 1 and len(log) == 1


async def test_one_utterance_then_a_followup_window_then_closed(voice):
    log: list[str] = []
    app, eng, mics, speaker, _, events = await voice(llm_log=log)
    app.config.set("voice.gate_user_level_db", -20.0)                        # the user's level is known
    app.config.set("voice.followup_s", 1)
    await eng.toggle_listening()
    say(mics[0])
    assert await settle(lambda: log == ["نرخی زێڕ چەندە؟"])
    assert await settle(lambda: any(isinstance(e, VoiceState) and e.detail == "turn_end" for e in events),
                        timeout=4.0)                                          # follow-up window over
    assert not eng.listening
    last = [e for e in events if isinstance(e, VoiceState)][-1]
    assert (last.state, last.detail) == ("idle", "turn_end")
    assert notices(events, "closed")[-1].text_ckb == strings.LISTEN_CLOSED
    assert "sleeping" not in [e.state for e in events if isinstance(e, VoiceState)]  # same conversation


async def test_followup_utterance_is_answered_and_conversation_ends_later(voice):
    log: list[str] = []
    stt = FakeStt(["نرخی زێڕ چەندە؟", "ئەی زیو؟"])
    app, eng, mics, speaker, _, events = await voice(llm_log=log, stt=stt)
    app.config.set("voice.gate_user_level_db", -20.0)
    app.config.set("voice.followup_s", 2)
    app.config.set("voice.conversation_timeout_s", 1)
    await eng.start_listening()
    say(mics[0])
    assert await settle(lambda: len(log) == 1)
    await asyncio.sleep(0.4)                                               # inside the follow-up window
    say(mics[0])
    assert await settle(lambda: len(log) == 2 and log[1] == "ئەی زیو؟")
    assert await settle(lambda: not eng.listening, timeout=5.0)
    assert await settle(lambda: any(isinstance(e, VoiceState) and e.state == "sleeping"
                                    and e.detail == "conversation_end" for e in events), timeout=4.0)


async def test_speech_after_the_window_closed_is_never_heard(voice):
    log: list[str] = []
    app, eng, mics, *_ = await voice(llm_log=log)
    app.config.set("voice.start_timeout_s", 1)
    await eng.start_listening()
    assert await settle(lambda: not eng.listening, timeout=4.0)
    say(mics[0])                                                           # the TV after the window closed
    await asyncio.sleep(0.2)
    assert log == [] and eng.stt.calls == []


# -- (a) always listening needs «سام» ------------------------------------------------------------------------------

def test_name_and_enrollment_phrases():
    assert starts_with_name("سام، نرخی زێڕ چەندە؟") and starts_with_name("هێی سام نرخی زێڕ")
    assert starts_with_name("Sam open chrome") and starts_with_name("hey SAM, what time is it")
    assert not starts_with_name("کچێکی پێکەنین وەرگیراوە باوکە گیان عوسمان")
    assert not starts_with_name("ئەمڕۆ سام هات") and not starts_with_name("")
    assert starts_with_name("هەیسام نرخی زێڕ چەندە؟")                         # KurdishTTS, live clip
    assert asks_enrollment("دەنگم بناسە") and asks_enrollment("تکایە دەنگی من بناسەوە")
    assert asks_enrollment("سام دەنگەکەم بناسە") and not asks_enrollment("دەنگی تەلەفزیۆنەکە بەرز بکەرەوە")
    assert not asks_enrollment("دەنگم تۆمار بکە")                             # may mean a voice note


async def test_always_listening_ignores_speech_without_the_name(voice):
    log: list[str] = []
    stt = FakeStt(["کچێکی پێکەنین وەرگیراوە باوکە گیان عوسمان", "سام، نرخی زێڕ چەندە؟"])
    app, eng, mics, speaker, _, events = await voice(llm_log=log, stt=stt)
    app.config.set("voice.always_listening", True)
    await eng.start_listening()
    say(mics[0])
    assert await settle(lambda: len(stt.calls) == 1)
    await asyncio.sleep(0.1)
    assert log == []                                                        # no model call for the TV
    assert notices(events, "ignored")[-1].text_ckb == strings.IGNORED_NO_NAME
    say(mics[0])
    assert await settle(lambda: log == ["سام، نرخی زێڕ چەندە؟"])
    assert eng.listening and eng.choose_engine() == "cascade"


async def test_always_listening_accepts_a_reply_right_after_sam_answered(voice):
    log: list[str] = []
    stt = FakeStt(["سام، نرخی زێڕ چەندە؟", "ئەی زیو؟", "ئەی نەوت؟", "ئەی بیتکۆین؟"])
    app, eng, mics, *_ = await voice(llm_log=log, stt=stt)
    app.config.set("voice.always_listening", True)
    app.config.set("voice.gate_user_level_db", -20.0)
    app.config.set("voice.followup_turns", 2)
    await eng.start_listening()
    answered = lambda: eng.listening_status()["followup_open"]  # noqa: E731 - SAM finished answering
    say(mics[0])
    assert await settle(lambda: len(log) == 1) and await settle(answered)
    say(mics[0])                                                            # within the follow-up grace
    assert await settle(lambda: log == ["سام، نرخی زێڕ چەندە؟", "ئەی زیو؟"])
    assert await settle(answered)
    say(mics[0])
    assert await settle(lambda: len(log) == 3)
    await asyncio.sleep(0.2)
    say(mics[0])                                                            # the budget per «سام» is used
    assert await settle(lambda: len(eng.stt.calls) == 4)
    await asyncio.sleep(0.1)
    assert len(log) == 3


async def test_always_listening_needs_the_name_again_without_trust(voice):
    """No voiceprint and no known level: after SAM answers, speech without
    «سام» is not a follow-up (the TV right after an answer used to be)."""
    log: list[str] = []
    stt = FakeStt(["سام، نرخی زێڕ چەندە؟", "کچێکی پێکەنین وەرگیراوە باوکە گیان عوسمان"])
    app, eng, mics, speaker, _, events = await voice(llm_log=log, stt=stt)
    app.config.set("voice.always_listening", True)
    await eng.start_listening()
    say(mics[0])
    assert await settle(lambda: len(log) == 1 and eng.tts.texts)           # SAM answered
    await asyncio.sleep(1.3)                                                # (a same-level continuation within
    say(mics[0])                                                            #  voice.merge_window_s would join it)
    assert await settle(lambda: len(eng.stt.calls) == 2)
    await asyncio.sleep(0.1)
    assert log == ["سام، نرخی زێڕ چەندە؟"]
    assert notices(events, "ignored")[-1].text_ckb == strings.IGNORED_NO_NAME


async def test_tv_cannot_open_the_enrollment_in_always_listening(voice):
    app, eng, mics, speaker, _, events = await voice(stt=FakeStt(["دەنگم بناسە"]))
    app.config.set("voice.always_listening", True)
    await eng.start_listening()
    say(mics[0])
    assert await settle(lambda: len(eng.stt.calls) == 1)
    await asyncio.sleep(0.1)
    assert not any(isinstance(e, VoiceEnrollRequest) for e in events)


async def test_spoken_enrollment_request_opens_the_dialog_without_a_model_call(voice):
    log: list[str] = []
    app, eng, mics, speaker, _, events = await voice(llm_log=log, stt=FakeStt(["دەنگم بناسە"]))
    await eng.start_listening()
    say(mics[0])
    assert await settle(lambda: any(isinstance(e, VoiceEnrollRequest) for e in events))
    assert await settle(lambda: strings.ENROLL_SPOKEN in eng.tts.texts)
    assert log == []


# -- (b) near-field gate -------------------------------------------------------------------------------------------------

def test_gate_threshold_floor_user_level_and_background():
    gate = NearFieldGate(GateSettings(margin_db=14, abs_min_db=-50, ceiling_db=-30), clock=lambda: 100.0)
    for _ in range(60):                                                     # a room with a fan at -45 dBFS
        gate.classify(10 ** (-45 / 20), False)
    assert -46 < gate.floor_db < -44 and gate.threshold_db == pytest.approx(-31, abs=1)
    assert not gate.classify(10 ** (-38 / 20), True)                         # the TV over the fan
    assert gate.classify(10 ** (-22 / 20), True)                             # the user into the headset
    quiet = NearFieldGate(GateSettings(), clock=lambda: 100.0)
    for _ in range(60):
        quiet.classify(10 ** (-75 / 20), False)
    assert quiet.threshold_db == -50                                         # quiet room: absolute minimum
    quiet.settings.user_level_db = -26.0                                     # enrollment / learned turns
    assert quiet.threshold_db == pytest.approx(-34) and quiet.ceiling_db == pytest.approx(-30)
    assert not quiet.classify(10 ** (-40 / 20), True)                        # TV 14 dB under the user
    quiet.note_background(-31.0)                                             # a rejected talker at -31
    assert quiet.threshold_db == pytest.approx(-30)                          # ...never above user - 4 dB
    assert quiet.learn_user_level(-20.0) == pytest.approx(-24.2)
    assert level_db(0.0) == -96.0 and round(level_db(0.1)) == -20


async def test_quiet_far_speech_never_reaches_stt(voice):
    app, eng, mics, *_ = await voice()
    app.config.set("voice.gate_user_level_db", -20.0)                        # the user's level is known
    await eng.start_listening()
    mics[0].push(quiet_frame(), 20)
    say(mics[0], db=-38)                                                     # a TV across the room
    await asyncio.sleep(0.3)
    assert eng.stt.calls == []
    say(mics[0], db=-20)                                                     # the user
    assert await settle(lambda: len(eng.stt.calls) == 1)


async def test_first_utterances_after_clicks_teach_the_gate_the_user_level(voice):
    """Stored only after 3 agreeing near-field turns (one TV-mixed first
    utterance used to be stored at once and let the TV in for good)."""
    app, eng, mics, *_ = await voice()
    assert app.config.get("voice.gate_user_level_db") is None
    for click in range(3):
        await eng.start_listening()
        say(mics[click], db=-24)
        assert await settle(lambda: len(eng.stt.calls) == click + 1)
        if click < 2:
            assert await settle(lambda: not eng.listening, timeout=3.0)    # no trust yet: no follow-up
            assert app.config.get("voice.gate_user_level_db") is None
    assert await settle(lambda: app.config.get("voice.gate_user_level_db") is not None)
    assert app.config.get("voice.gate_user_level_db") == pytest.approx(-24, abs=0.6)
    assert await settle(lambda: eng.listening_status()["followup_open"])   # now trusted: a follow-up window


async def test_far_speech_after_a_click_is_never_learned_as_the_user(voice):
    app, eng, mics, *_ = await voice()
    for click in range(3):
        await eng.start_listening()
        mics[click].push(quiet_frame(), 20)
        say(mics[click], db=-41)                                             # the TV across the room
        assert await settle(lambda: len(eng.stt.calls) == click + 1)
        assert await settle(lambda: not eng.listening, timeout=3.0)
    assert app.config.get("voice.gate_user_level_db") is None


def test_learned_level_moves_slowly_and_ignores_outliers():
    gate = NearFieldGate(GateSettings(user_level_db=-26.0), clock=lambda: 100.0)
    for _ in range(40):
        gate.classify(10 ** (-70 / 20), False)                               # a quiet room
    assert gate.learn_user_level(-36.0) == pytest.approx(-28.0)              # at most 2 dB per turn
    assert gate.learn_user_level(-45.0) is None                              # far speech: not a candidate
    assert gate.learn_user_level(-12.0) is None                              # > 10 dB away: an outlier
    gate.note_background(-29.0)                                              # a rejected talker at -29
    assert gate.learn_user_level(-27.0) is None                              # too close to that talker
    fresh = NearFieldGate(GateSettings(), clock=lambda: 100.0)
    for _ in range(40):
        fresh.classify(10 ** (-70 / 20), False)
    assert [fresh.learn_user_level(x) for x in (-25.0, -33.0, -24.0, -26.0)] == [None] * 4  # 9 dB apart
    assert fresh.learn_user_level(-25.0) == pytest.approx(-25.0)             # the last 3 agree


# -- (c) Live: only accepted utterances are sent -----------------------------------------------------------------------

async def test_live_gets_no_audio_from_the_tv_and_the_user_utterance_with_stream_end(voice):
    from sam.voice.live_config import build_live_config
    app, eng, mics, speaker, client, events = await voice(gemini=True)
    app.config.set("voice.gate_user_level_db", -20.0)
    await eng.start_listening()
    assert await settle(lambda: eng.live is not None and eng.live.ready)
    session = client.sessions[0]
    mics[0].push(quiet_frame(), 10)
    say(mics[0], db=-40, frames=40)                                          # TV: nothing is sent
    await asyncio.sleep(0.3)
    assert session.audio == [] and not any("audio_stream_end" in r for r in session.realtime)
    say(mics[0], db=-20, frames=20)
    assert await settle(lambda: any(r.get("audio_stream_end") for r in session.realtime))
    assert len(session.audio) >= 20                                          # pre-roll + the utterance
    config = build_live_config(app, "gemini-3.8-live", with_tools=False)
    detection = config.realtime_input_config.automatic_activity_detection
    assert str(detection.start_of_speech_sensitivity).endswith("START_SENSITIVITY_LOW")


async def test_live_holds_audio_until_the_voiceprint_matches(voice):
    """The owner's turn after the click is sent without a voiceprint check;
    after it, someone else close by is held back (nothing sent) and the user's
    own voice goes through."""
    app, eng, mics, speaker, client, events = await voice(gemini=True, enrolled_pitch=8)
    await eng.start_listening()
    assert await settle(lambda: eng.live is not None and eng.live.ready)
    session = client.sessions[0]
    say(mics[0], db=-20, frames=20, pitch=8)                                 # the owner's turn after the click
    assert await settle(lambda: len(session.audio) >= 15)
    sent = len(session.audio)
    say(mics[0], db=-20, frames=50, pitch=3)                                 # someone else, close by
    await asyncio.sleep(0.4)
    assert len(session.audio) == sent
    assert notices(events, "not_recognized")[-1].text_ckb == strings.VOICE_NOT_RECOGNIZED
    say(mics[0], db=-20, frames=50, pitch=8)                                 # the user
    assert await settle(lambda: len(session.audio) >= sent + 40)


# -- (f) only my voice ------------------------------------------------------------------------------------------------------

async def test_other_voices_cost_no_stt_when_a_voiceprint_exists(voice):
    """After the owner's turn (never checked), a follow-up by another voice
    costs no STT and shows «دەنگەکەت نەناسرایەوە — کلیک بکە» once; the
    user's own follow-up is taken."""
    log: list[str] = []
    stt = FakeStt(["نرخی زێڕ چەندە؟", "ئەی زیو؟"])
    app, eng, mics, speaker, _, events = await voice(llm_log=log, enrolled_pitch=8, stt=stt)
    await eng.start_listening()
    say(mics[0], pitch=8)                                                    # the owner's turn after the click
    assert await settle(lambda: log == ["نرخی زێڕ چەندە؟"])
    assert await settle(lambda: eng.listening_status()["followup_open"])     # SAM answered: a follow-up window
    say(mics[0], pitch=3)                                                    # family member at the headset
    assert await settle(lambda: notices(events, "not_recognized"))
    say(mics[0], pitch=3)                                                    # ... again: no second notice
    assert await settle(lambda: len(app.db.query("SELECT 1 FROM activity WHERE name='not_my_voice'")) == 2)
    assert len(stt.calls) == 1 and len(log) == 1
    assert len(notices(events, "not_recognized")) == 1
    assert eng.gate.background_db is not None                               # that talker now stays below
    say(mics[0], pitch=8)                                                    # the user's follow-up
    assert await settle(lambda: log == ["نرخی زێڕ چەندە؟", "ئەی زیو؟"])
    last = eng.speaker_check.last
    assert last is not None and last.ok and last.reason == "match"


async def test_only_my_voice_off_or_not_enrolled_lets_the_gate_decide(voice):
    app, eng, mics, *_ = await voice(enrolled_pitch=8)
    app.config.set("voice.only_my_voice", False)
    await eng.start_listening()
    say(mics[0], pitch=3)
    assert await settle(lambda: len(eng.stt.calls) == 1)


async def test_speaker_check_thresholds_and_store(make_app):
    app = make_app()
    check = SpeakerCheck(app, embedder_factory=PitchEmbedder, store=plain_store(app))
    assert not check.enrolled and (await check.verify(b"")).reason == "not_enrolled"
    # Real use 2026-09-25: the owner scored 0.055-0.346 against his enrollment; 0.40 rejected him.
    assert check.threshold() == 0.2 and check.threshold(speech_ms=600) == 0.15
    app.config.set("voice.only_my_voice_sensitivity", "high")
    assert check.threshold() == 0.28
    app.config.set("voice.only_my_voice_sensitivity", "low")
    assert check.threshold() == 0.15 and check.threshold(speech_ms=600) == 0.10   # never below 0.10
    app.config.set("voice.only_my_voice_sensitivity", "normal")
    check.store.save(PitchEmbedder().embed(frame_at(-20) * 20), model="fake", level_db=-21.0, clips=5,
                     consistency=0.8)
    check.forget_cache()
    assert check.enrolled and check.status()["enrolled"] and check.status()["clips"] == 5
    assert (await check.verify(frame_at(-20) * 20)).ok
    other = await check.verify(frame_at(-20, pitch=3) * 20)
    assert not other.ok and other.reason == "other_voice" and other.score < other.threshold
    assert check.store.delete() and not check.store.exists()


def test_voiceprint_is_dpapi_protected_at_rest(make_app):
    import os
    if os.name != "nt":
        pytest.skip("DPAPI is Windows-only")
    app = make_app()
    store = VoiceprintStore(app.db)
    store.save([0.1, 0.2, 0.3], model="m", level_db=-20.0, clips=4, consistency=0.9)
    raw = app.db.query_one("SELECT blob FROM voice_profile WHERE id=1")["blob"]
    assert b"vector" not in bytes(raw) and b"0.1" not in bytes(raw)
    assert store.load()["vector"] == pytest.approx([0.1, 0.2, 0.3])


async def test_enrollment_records_sentences_and_stores_the_voiceprint(voice):
    app, eng, mics, speaker, _, events = await voice()
    begun = await eng.enroll_begin()
    assert begun["ok"] and begun["sentences"] == list(strings.ENROLL_SENTENCES)

    async def read_sentence(index, pitch=8, db=-22.0, frames=45):
        task = asyncio.ensure_future(eng.enroll_record(index, timeout_s=5.0))
        assert await settle(lambda: len(mics) > index and mics[index].started)
        mics[index].push(quiet_frame(), 15)
        mics[index].push(frame_at(db, pitch=pitch), frames)
        mics[index].push(quiet_frame(), 40)
        return await task

    short = await read_sentence(0, frames=20)                               # 0.6 s: too short
    assert not short["ok"] and short["reason"] == "too_short"
    results = [await read_sentence(i + 1) for i in range(5)]
    assert all(r["ok"] for r in results) and all(m.stopped for m in mics)
    done = await eng.enroll_finish()
    assert done["ok"] and done["clips"] == 5 and done["consistency"] >= 0.40
    assert app.config.get("voice.only_my_voice") is True
    assert app.config.get("voice.gate_user_level_db") == pytest.approx(-22, abs=0.6)
    assert eng.speaker_check.enrolled and eng.speaker_check.enabled
    deleted = await eng.voiceprint_delete()
    assert deleted["existed"] and not eng.speaker_check.enrolled


async def record_clip(eng, mics, index, pitch, *, db=-22.0, lead=15):
    task = asyncio.ensure_future(eng.enroll_record(index, timeout_s=5.0))
    assert await settle(lambda: len(mics) > 0 and mics[-1].started and not mics[-1].stopped)
    mics[-1].push(quiet_frame(), lead)
    mics[-1].push(frame_at(db, pitch=pitch), 45)
    mics[-1].push(quiet_frame(), 40)
    return await task


async def test_enrollment_refuses_a_mixed_voiceprint(voice):
    app, eng, mics, *_ = await voice()
    assert (await eng.enroll_begin())["ok"]
    for index, pitch in enumerate((8, 3, 8, 3, 5)):
        assert (await record_clip(eng, mics, index, pitch))["ok"]
    done = await eng.enroll_finish()
    assert not done["ok"] and done["reason"] == "inconsistent" and not eng.speaker_check.enrolled


async def test_one_other_voice_among_the_clips_is_found_and_read_again(voice):
    """4 clips of the user + 1 of another man used to be ACCEPTED (review:
    consistency 0.48 vs the mean of all clips); every pair must match now."""
    app, eng, mics, *_ = await voice()
    assert (await eng.enroll_begin())["ok"]
    for index, pitch in enumerate((8, 8, 3, 8, 8)):
        assert (await record_clip(eng, mics, index, pitch))["ok"]
    done = await eng.enroll_finish()
    assert not done["ok"] and done["retry_index"] == 2 and done["message_ckb"] == strings.ENROLL_REPEAT_ONE
    assert not eng.speaker_check.enrolled
    assert (await record_clip(eng, mics, 2, 8))["ok"]                      # that sentence again, by the user
    done = await eng.enroll_finish()
    assert done["ok"] and done["clips"] == 5 and eng.speaker_check.enrolled


async def test_enrollment_needs_four_clips_and_skips_speech_already_going_on(voice):
    app, eng, mics, *_ = await voice()
    assert (await eng.enroll_begin())["ok"]
    task = asyncio.ensure_future(eng.enroll_record(0, timeout_s=5.0))
    assert await settle(lambda: mics and mics[-1].started)
    mics[-1].push(frame_at(-22, pitch=3), 45)                               # the TV, already talking
    mics[-1].push(quiet_frame(), 40)
    mics[-1].push(frame_at(-22, pitch=8), 45)                               # then the user reads
    mics[-1].push(quiet_frame(), 40)
    assert (await task)["ok"]
    for index in (1, 2):
        assert (await record_clip(eng, mics, index, 8))["ok"]
    done = await eng.enroll_finish()                                        # 3 clips: too few now
    assert not done["ok"] and done["reason"] == "too_few"
    assert (await eng.enroll_begin())["ok"]
    for index in range(4):
        assert (await record_clip(eng, mics, index, 8))["ok"]
    assert (await eng.enroll_finish())["ok"]
    stored = eng.speaker_check.store.load()["vector"]
    assert max(range(len(stored)), key=lambda i: stored[i]) == PitchEmbedder().embed(frame_at(-22, pitch=8) * 30).index(
        max(PitchEmbedder().embed(frame_at(-22, pitch=8) * 30)))            # the user's voice, not the TV's


# -- (e) notices ---------------------------------------------------------------------------------------------------------

async def test_models_exhausted_notice_names_the_gemini_reset_time(voice):
    app, eng, mics, speaker, _, events = await voice()
    app.db.bump_usage("gemini", "gemini-3.5-flash-lite", kind="text", rate_limited=1)
    from sam.brain.responder import SORANI_NO_MODEL
    app.bus.publish(Transcript(role="assistant", text=SORANI_NO_MODEL, source="cascade"))
    await settle()
    notice = notices(events, "models")[-1]
    assert notice.until > 0 and "دوای کاتژمێر" in notice.text_ckb and "سنووری ئەمڕۆ پڕە" in notice.text_ckb


# -- speech while SAM works; confirmations on the voice path (review 2026-09-24) ------------------------------

async def test_other_speech_while_sam_thinks_is_not_taken_without_a_voiceprint(voice):
    """The TV speaking while SAM thinks used to cancel the user's request
    (barge-in) and become the next request (carry): now it costs nothing."""
    log: list[str] = []
    release = asyncio.Event()
    stt = FakeStt(["نرخی زێڕ چەندە؟", "کچێکی پێکەنین وەرگیراوە باوکە گیان عوسمان"])
    app, eng, mics, speaker, _, events = await voice(stt=stt, llm=fake_llm([release, "نرخی زێڕ ٢٦٥٠ دۆلارە."], log))
    await eng.start_listening()
    say(mics[0])
    assert await settle(lambda: log == ["نرخی زێڕ چەندە؟"])
    await asyncio.sleep(1.3)                                                # past voice.merge_window_s: not a continuation
    say(mics[0], pitch=3)                                                   # the TV while SAM thinks
    await asyncio.sleep(0.3)
    assert len(stt.calls) == 1 and log == ["نرخی زێڕ چەندە؟"]
    release.set()
    assert await settle(lambda: "نرخی زێڕ ٢٦٥٠ دۆلارە." in " ".join(eng.tts.texts))  # the answer is spoken


async def test_the_users_own_voice_may_correct_a_request_while_sam_thinks(voice):
    log: list[str] = []
    release = asyncio.Event()
    stt = FakeStt(["نرخی زێڕ چەندە؟", "نا، نرخی زیو"])
    app, eng, mics, *_ = await voice(stt=stt, enrolled_pitch=8, llm=fake_llm([release, "باشە."], log))
    await eng.start_listening()
    say(mics[0], frames=45)
    assert await settle(lambda: len(log) == 1)
    say(mics[0], frames=45)                                                 # verified: replaces the request
    assert await settle(lambda: len(log) == 2, timeout=3.0)
    assert log[1] == "نرخی زێڕ چەندە؟ نا، نرخی زیو"                          # carried over
    release.set()


def confirm_llm(app, log, outcome):
    async def stream(text, turn):
        log.append(text)
        if len(log) == 1:
            ok = await app.confirm.confirm("ئەم نامەیە بنێرم؟", tool_name="type_text", timeout_s=10.0)
            outcome["approved"] = ok
            yield "نامەکە نێردرا. " if ok else "نەمنارد. "
        else:
            yield "وەڵامێکی نوێ. "
    return stream


async def test_unclear_word_while_a_confirmation_waits_asks_again_and_the_answer_is_spoken(voice):
    from sam.brain.confirm import ASK_AGAIN_CKB
    log: list[str] = []
    outcome: dict = {}
    stt = FakeStt(["نامەیەک بنێرە بۆ ئەحمەد", "باشە", "بەڵێ"])
    app, eng, mics, speaker, _, events = await voice(stt=stt)
    eng.cascade._llm_stream = confirm_llm(app, log, outcome)  # noqa: SLF001
    await eng.start_listening()
    say(mics[0])
    assert await settle(lambda: app.confirm.has_pending)
    await asyncio.sleep(2.0)                                                # past the question's echo guard
    say(mics[0])                                                            # «باشە»: neither yes nor no
    assert await settle(lambda: ASK_AGAIN_CKB in eng.tts.texts)
    assert log == ["نامەیەک بنێرە بۆ ئەحمەد"]                                # no new model turn
    await asyncio.sleep(2.0)
    say(mics[0])                                                            # «بەڵێ»
    assert await settle(lambda: outcome.get("approved") is True)
    assert await settle(lambda: any("نامەکە نێردرا" in t for t in eng.tts.texts))  # the result is heard


async def test_always_listening_tv_while_a_confirmation_waits_reaches_no_model(voice):
    log: list[str] = []
    outcome: dict = {}
    stt = FakeStt(["سام، نامەیەک بنێرە بۆ ئەحمەد", "کچێکی پێکەنین وەرگیراوە باوکە گیان عوسمان", "نەخێر"])
    app, eng, mics, speaker, _, events = await voice(stt=stt)
    app.config.set("voice.always_listening", True)
    eng.cascade._llm_stream = confirm_llm(app, log, outcome)  # noqa: SLF001
    await eng.start_listening()
    say(mics[0])
    assert await settle(lambda: app.confirm.has_pending)
    await asyncio.sleep(2.0)
    say(mics[0])                                                            # the TV, no «سام»
    assert await settle(lambda: len(stt.calls) == 2)
    await asyncio.sleep(0.2)
    assert log == ["سام، نامەیەک بنێرە بۆ ئەحمەد"] and app.confirm.has_pending
    assert notices(events, "ignored")[-1].text_ckb == strings.IGNORED_NO_NAME
    say(mics[0])                                                            # «نەخێر» needs no name
    assert await settle(lambda: outcome.get("approved") is False)
    assert await settle(lambda: any("نەمنارد" in t for t in eng.tts.texts))


async def test_a_voiceprint_that_cannot_run_is_reported_not_trusted(voice):
    app, eng, mics, speaker, _, events = await voice(enrolled_pitch=8)

    def broken(pcm, rate=16000):
        raise RuntimeError("invalid model file")
    eng.speaker_check.embedder().embed = broken
    await eng.start_listening()
    say(mics[0], pitch=3)                                                   # lets the speech through ...
    assert await settle(lambda: len(eng.stt.calls) == 1)
    assert await settle(lambda: notices(events, "voiceprint"))
    assert notices(events, "voiceprint")[-1].text_ckb == strings.VOICEPRINT_UNAVAILABLE  # ... and says so
    assert not eng.speaker_check.usable and eng.speaker_check.status()["usable"] is False
