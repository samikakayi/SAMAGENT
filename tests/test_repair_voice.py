"""Repair review 2026-09-24 (voice lens): kept-alive shared HTTPS pool, the
confirmation echo window, backchannels, the KurdishTTS budget, the first
answer's audio timed apart from the acknowledgement, spoken numbers."""

from __future__ import annotations

import asyncio
import time

import pytest
from voice_helpers import FakeSpeaker, FakeStt, FakeTts, fake_llm, settle

from sam.voice import kurdish_http
from sam.voice.cascade import CascadeVoice, is_backchannel
from sam.voice.hooks import RecordingHooks
from sam.voice.numbers_ckb import verbalize_numbers


@pytest.fixture
def cascade_env(make_app):
    def build(pieces, stt_texts, *, tts=None):
        app = make_app()
        app.bus.bind_loop(asyncio.get_running_loop())
        log: list[str] = []
        tts = tts or FakeTts()
        cascade = CascadeVoice(app, FakeSpeaker(), FakeStt(stt_texts), tts, RecordingHooks(),
                               llm_stream=fake_llm(pieces, log))
        return app, cascade, tts, log
    return build


@pytest.mark.parametrize("text,spoken", [
    ("زێڕ لەسەر 4270", "زێڕ لەسەر چوار هەزار و دووسەد و حەفتا"),
    ("4268.76", "چوار هەزار و دووسەد و شەست و هەشت پۆینت حەفتا و شەش"),
    ("1.0854", "یەک پۆینت سفر هەشت پێنج چوار"),
    ("80%", "لەسەدا هەشتا"),
    ("M15 و 14:35", "M15 و 14:35"),
    ("٢٧٠٠", "دوو هەزار و حەوتسەد"),
])
def test_numbers_are_spoken_as_sorani_words(text, spoken):
    assert verbalize_numbers(text) == spoken


def test_by_default_only_decimals_become_words():
    """KurdishTTS reads whole numbers itself (measured); digits cost 4 budget characters, words 26."""
    assert verbalize_numbers("زێڕ لەسەر 4270 و 1.5", integers=False) == "زێڕ لەسەر 4270 و یەک پۆینت پێنج"


def test_speech_to_text_and_text_to_speech_share_one_kept_alive_pool(make_app):
    from sam.voice.stt import KurdishTtsStt
    from sam.voice.tts import KurdishTts

    app = make_app()
    assert KurdishTtsStt(app)._http() is KurdishTts(app)._http()                 # noqa: SLF001
    assert kurdish_http.limits().keepalive_expiry == 120


async def test_what_the_mic_hears_during_sams_question_is_not_an_answer(cascade_env):
    app, cascade, tts, log = cascade_env(["باشە."], ["بەڵێ"])
    asking = asyncio.ensure_future(app.confirm.confirm("ئەم نامەیە بنێرم؟", tool_name="type_text", timeout_s=0.6))
    await settle(lambda: app.confirm.has_pending)
    await cascade.speak("ئەم نامەیە بنێرم؟", source="confirm")
    assert cascade.confirm_quiet_until > time.perf_counter()
    cascade.submit_utterance(b"\x01\x00" * 8000, time.perf_counter())             # SAM's own voice, heard back
    assert await asyncio.wait_for(asking, 2) is False                              # not approved: it timed out
    assert log == []


async def test_a_backchannel_that_cut_a_reply_starts_no_new_turn(cascade_env):
    app, cascade, tts, log = cascade_env(["باشە."], ["ئەها"])
    assert is_backchannel("ئەها") and not is_backchannel("بوەستە")
    cascade.submit_utterance(b"\x01\x00" * 8000, time.perf_counter(), cut_reply=True)
    await settle(lambda: cascade._queue.empty() and not cascade._processing)      # noqa: SLF001
    await asyncio.sleep(0.05)
    assert log == []


async def test_a_low_kurdishtts_budget_speaks_only_the_first_sentence(cascade_env):
    class LowTts(FakeTts):
        def low_budget(self, share):
            return True

    app, cascade, tts, log = cascade_env(["یەکەم ڕستە. ", "دووەم ڕستە. ", "سێیەم ڕستە."], ["شیکاری بکە"],
                                         tts=LowTts())
    cascade.submit_utterance(b"\x01\x00" * 8000, time.perf_counter())
    await settle(lambda: tts.texts, timeout=2)
    await asyncio.sleep(0.2)
    assert len(tts.texts) == 1 and tts.texts[0].startswith("یەکەم")


async def test_the_first_answer_audio_is_timed_apart_from_the_acknowledgement(make_app):
    from brain_helpers import Reply, brain_app

    app, _ = brain_app(make_app, [Reply(calls=[("tv_open", {})]), Reply(text="ترەیدینگ ڤیو ئامادەیە.")])
    app.bus.bind_loop(asyncio.get_running_loop())
    cascade = CascadeVoice(app, FakeSpeaker(), FakeStt(["ترەیدینگ ڤیو بکەرەوە"]), FakeTts(), RecordingHooks())
    cascade.submit_utterance(b"\x01\x00" * 8000, time.perf_counter())
    await settle(lambda: app.db.scalar("SELECT COUNT(*) FROM timings WHERE stage='first_answer_audio'"), timeout=3)
    stages = {r["stage"] for r in app.db.query("SELECT stage FROM timings WHERE kind='cascade'")}
    assert {"first_audio", "first_answer_audio"} <= stages
