"""CascadeVoice with fake STT / brain stream / TTS / speaker: first-sentence
pipelining, barge-in, confirmations answered by voice, carry-over, errors."""

from __future__ import annotations

import asyncio
import time

import pytest
from conftest import FakeBackend
from voice_helpers import FakeSpeaker, FakeStt, FakeTts, fake_llm, settle

from sam.events import Caption, Error, Transcript
from sam.voice import strings
from sam.voice.cascade import CascadeVoice
from sam.voice.hooks import RecordingHooks
from sam.voice.stt import SttError
from sam.voice.tts import TtsError


@pytest.fixture
def env(make_app):
    created = []

    def build(pieces=None, *, stt=None, tts=None, app=None, llm_log=None, use_llm=True):
        app = app or make_app()
        app.bus.bind_loop(asyncio.get_running_loop())
        events = []
        app.bus.subscribe(None, events.append)
        speaker, hooks = FakeSpeaker(), RecordingHooks()
        stt = stt or FakeStt(["زێڕ شی بکەرەوە"])
        tts = tts or FakeTts()
        llm = fake_llm(pieces or ["باشە."], llm_log) if use_llm else None
        cascade = CascadeVoice(app, speaker, stt, tts, hooks, llm_stream=llm)
        created.append(cascade)
        return app, cascade, speaker, stt, tts, hooks, events
    yield build


def transcripts(events, role=None):
    return [e for e in events if isinstance(e, Transcript) and (role is None or e.role == role)]


async def test_first_sentence_is_spoken_while_the_model_is_still_writing(env):
    gate = asyncio.Event()
    app, cascade, speaker, stt, tts, hooks, events = env(
        ["باشە، ", "ئێستا شیکاری دەکەم. ", gate, "زێڕ لەسەر ٢٦٥٠ ـە ", "و ترێندەکە بەرەو سەرەوەیە."])
    cascade.submit_utterance(b"\x01\x00" * 16000, time.perf_counter() - 0.6)
    assert await settle(lambda: speaker.chunks)
    assert tts.texts == ["باشە، ئێستا شیکاری دەکەم."]       # first sentence already audible...
    assert not gate.is_set()                                # ...before the model finished
    assert ("speaking", "") in hooks.states
    gate.set()
    assert await settle(lambda: len(tts.texts) == 2)
    assert tts.texts[1] == "زێڕ لەسەر ٢٦٥٠ ـە و ترێندەکە بەرەو سەرەوەیە."
    assert await settle(lambda: len(transcripts(events)) == 2)
    user, assistant = transcripts(events)
    assert (user.role, user.text, user.source) == ("user", "زێڕ شی بکەرەوە", "cascade")
    assert assistant.role == "assistant" and assistant.text.startswith("باشە، ئێستا شیکاری دەکەم.")
    rows = app.db.query("SELECT stage, ms FROM timings WHERE kind='cascade'")
    stages = {r["stage"]: r["ms"] for r in rows}
    assert {"end_of_speech", "stt", "tts_first_audio", "first_audio", "total"} <= set(stages)
    assert stages["end_of_speech"] >= 500          # measured from the last voiced frame
    assert stages["first_audio"] >= stages["end_of_speech"]
    assert cascade.last_ttfa_ms is not None
    assert await settle(lambda: hooks.last_state == "listening")


async def test_barge_in_flushes_and_cancels_a_speaking_reply(env):
    gate = asyncio.Event()
    app, cascade, speaker, stt, tts, hooks, events = env(["یەکەم ڕستە تەواو بوو. ", gate, "ئەمە نابیسترێت."])
    cascade.submit_utterance(b"\x01\x00" * 8000, time.perf_counter())
    assert await settle(lambda: speaker.chunks)
    assert cascade.barge_in()
    assert speaker.flushes == 1 and hooks.last_state == "listening"
    assert await settle(lambda: cascade._current.done)  # noqa: SLF001
    gate.set()
    await settle()
    assert tts.texts == ["یەکەم ڕستە تەواو بوو."]
    said = transcripts(events, "assistant")
    assert said and said[-1].text.endswith("…")         # what was heard, marked as cut


async def test_barge_in_during_a_tool_keeps_the_reply_running_muted(env):
    gate = asyncio.Event()
    app, cascade, speaker, stt, tts, hooks, events = env(["باشە، دایدەخەم. ", gate, "داخرا."])
    cascade.submit_utterance(b"\x01\x00" * 8000, time.perf_counter())
    assert await settle(lambda: speaker.chunks)
    app.tools.running = lambda: [{"call_id": "x", "name": "window_control", "source": "cascade"}]
    cascade.barge_in()
    await settle()
    assert not cascade._current.done  # noqa: SLF001 - the tool must finish honestly
    gate.set()
    assert await settle(lambda: cascade._current.done)  # noqa: SLF001
    assert tts.texts == ["باشە، دایدەخەم."]              # the rest was not spoken over the user
    assert transcripts(events, "assistant")[-1].text.endswith("داخرا.")  # but it is in the history


async def test_a_spoken_yes_answers_a_pending_confirmation(env):
    log = []
    app, cascade, speaker, stt, tts, hooks, events = env(["نابێت بگات."], stt=FakeStt(["بەڵێ"]), llm_log=log)
    asking = asyncio.ensure_future(app.confirm.confirm("فایلەکە بسڕمەوە؟", tool_name="files"))
    await settle(lambda: app.confirm.has_pending)
    cascade.submit_utterance(b"\x01\x00" * 8000, time.perf_counter())
    assert await asyncio.wait_for(asking, 2) is True
    await settle()
    assert log == []                                      # consumed: not sent to the brain
    assert transcripts(events, "user")[0].text == "بەڵێ"  # still recorded in the conversation


async def test_silent_reply_interrupted_by_more_speech_is_merged(env):
    gate = asyncio.Event()
    log = []
    stt = FakeStt(["زێڕ پیشان بدە", "لەسەر ١٥ خولەک"])
    app, cascade, speaker, _, tts, hooks, events = env([gate, "باشە."], stt=stt, llm_log=log)
    cascade.submit_utterance(b"\x01\x00" * 8000, time.perf_counter())
    assert await settle(lambda: log == ["زێڕ پیشان بدە"])
    cascade.barge_in()                                    # the user keeps talking before any audio
    cascade.submit_utterance(b"\x02\x00" * 8000, time.perf_counter())
    assert await settle(lambda: len(log) == 2)
    assert log[1] == "زێڕ پیشان بدە لەسەر ١٥ خولەک"
    gate.set()
    assert await settle(lambda: tts.texts == ["باشە."])


async def test_blip_after_barge_in_resumes_the_carried_request(env):
    gate = asyncio.Event()
    log = []
    app, cascade, *_ = env([gate, "باشە."], llm_log=log)
    cascade.submit_utterance(b"\x01\x00" * 8000, time.perf_counter())
    assert await settle(lambda: len(log) == 1)
    cascade.barge_in()
    cascade.resume_carry()                                # the "speech" was a cough
    assert await settle(lambda: len(log) == 2) and log[1] == log[0]
    gate.set()


async def test_stt_failure_is_reported_and_apologised(env):
    app, cascade, speaker, stt, tts, hooks, events = env(stt=FakeStt([SttError("network", provider="kurdishtts")]))
    cascade.submit_utterance(b"\x01\x00" * 8000, time.perf_counter())
    assert await settle(lambda: tts.texts)
    assert tts.texts == [strings.STT_FAILED_SPOKEN]
    errors = [e for e in events if isinstance(e, Error)]
    assert errors and errors[0].message_ckb == strings.STT_FAILED


async def test_unconfigured_stt_warns_once(env):
    app, cascade, speaker, stt, tts, hooks, events = env(stt=FakeStt(configured=False))
    for _ in range(2):
        cascade.submit_utterance(b"\x01\x00" * 8000, time.perf_counter())
    await settle(lambda: not cascade.busy)
    errors = [e for e in events if isinstance(e, Error)]
    assert len(errors) == 1 and errors[0].message_ckb == strings.STT_UNCONFIGURED and tts.texts == []


async def test_speak_fixed_text_and_interrupt(env):
    app, cascade, speaker, stt, tts, hooks, events = env()
    await cascade.speak("زێڕ گەیشتە ٢٧٠٠ دۆلار.", source="alert")
    assert tts.texts == ["زێڕ گەیشتە ٢٧٠٠ دۆلار."] and speaker.chunks
    assert any(isinstance(e, Caption) and e.final and "٢٧٠٠" in e.text for e in events)
    await cascade.speak("بوەستە!", interrupt=True, source="alert")
    assert speaker.flushes == 1 and tts.texts[-1] == "بوەستە!"


async def test_tts_failure_is_reported_once_per_reply(env):
    app, cascade, speaker, stt, tts, hooks, events = env(
        ["یەک. ", "دوو دوو دوو دوو دوو دوو دوو دوو دوو دوو دوو. " * 5], tts=FakeTts(fail=TtsError("server", provider="gemini")))
    cascade.submit_utterance(b"\x01\x00" * 8000, time.perf_counter())
    assert await settle(lambda: transcripts(events, "assistant"))
    errors = [e for e in events if isinstance(e, Error) and e.where == "voice.tts"]
    assert len(errors) == 1 and errors[0].message_ckb == strings.TTS_FAILED


async def test_uses_the_brain_respond_stream_sentence_chunks(env, make_app):
    """The brain yields finished sentences (no trailing space) and stores the
    reply itself: each sentence is spoken without waiting for the next chunk,
    a sentence held back for packing is released while a tool runs, and the
    voice does not store the assistant turn a second time."""
    app = make_app()
    calls = []
    gate = asyncio.Event()

    class Conversation:
        async def respond_stream(self, text, *, source="cascade", turn=None):
            calls.append((text, source, turn is not None))
            yield "باشە، دەیکەمەوە."
            yield "ترەیدینگ ڤیو کرایەوە."
            await gate.wait()                       # e.g. a tool round
            yield "هیچی تر؟"

    app.conversation = Conversation()
    _, cascade, speaker, stt, tts, hooks, events = env(app=app, use_llm=False)
    cascade.submit_utterance(b"\x01\x00" * 8000, time.perf_counter())
    assert await settle(lambda: tts.texts == ["باشە، دەیکەمەوە.", "ترەیدینگ ڤیو کرایەوە."])
    assert not gate.is_set()
    gate.set()
    assert await settle(lambda: tts.texts[-1] == "هیچی تر؟")
    assert calls == [("زێڕ شی بکەرەوە", "cascade", True)]
    await settle(lambda: not cascade.busy)
    assert transcripts(events, "assistant") == []    # the brain is the single writer of its reply
    final = [e for e in events if isinstance(e, Caption) and e.role == "assistant" and e.final]
    assert final and final[-1].text == "باشە، دەیکەمەوە. ترەیدینگ ڤیو کرایەوە. هیچی تر؟"


async def test_falls_back_to_plain_llm_without_a_brain(env, make_app):
    app = make_app(backends={"groq": FakeBackend("groq", {"openai/gpt-oss-20b": ["سڵاو. چۆنی؟"]})})
    _, cascade, speaker, stt, tts, *_ = env(app=app, use_llm=False)
    cascade.submit_utterance(b"\x01\x00" * 8000, time.perf_counter())
    assert await settle(lambda: tts.texts and "چۆنی" in " ".join(tts.texts))
    assert tts.texts[0] == "سڵاو."
