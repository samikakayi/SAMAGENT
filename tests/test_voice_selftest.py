"""Live self-test with fake TTS and a scripted fake Live session: CER of the
input transcription, Sorani-script check of the reply, TTFA, storage."""

from __future__ import annotations

import asyncio

import pytest
from conftest import FAKE_GEMINI
from voice_helpers import ApiError, FakeLiveClient, FakeLiveSession, FakeTts, audio_msg, idle_msg, msg

from sam.voice import strings
from sam.voice.selftest import SELFTEST_INSTRUCTION, run_selftest


class ScriptedSession(FakeLiveSession):
    """Replies once per clip: when silence follows speech (the clip ended),
    push the scripted heard/said pair, some audio and an idle turn end."""

    def __init__(self, replies):
        super().__init__()
        self.replies = list(replies)
        self._was_speech = False

    async def send_realtime_input(self, **kwargs):
        await super().send_realtime_input(**kwargs)
        data = self.audio[-1]
        speech = any(data)
        if self._was_speech and not speech and self.replies:
            heard, said = self.replies.pop(0)
            self.push(msg({"server_content": {"input_transcription": {"text": heard, "finished": True}}}),
                      audio_msg(), msg({"server_content": {"output_transcription": {"text": said}}}), idle_msg())
        self._was_speech = speech


GOOD = [(s, "باشە، تێگەیشتم.") for s in strings.SELFTEST_SENTENCES]


@pytest.fixture
def gemini_app(make_app):
    return make_app(env_text=f"GEMINI_API_KEY={FAKE_GEMINI}\n")


async def test_without_a_key_the_result_says_so(make_app):
    app = make_app()
    result = await run_selftest(app)
    assert result["ok"] is False and result["error"] == "no_gemini_key"
    assert result["message_ckb"] == strings.SELFTEST_NO_KEY
    assert app.config.get("voice.selftest")["error"] == "no_gemini_key"


async def test_good_sorani_passes_and_is_stored(gemini_app):
    session = ScriptedSession(GOOD)
    client = FakeLiveClient([session])
    tts = FakeTts(chunks=4, chunk_bytes=4800)
    result = await run_selftest(gemini_app, tts=tts, client_factory=lambda key: client, pace=0)
    assert result["ok"] is True, result
    assert result["cer"] == 0.0 and result["script_ok"] is True and result["ttfa_ms"] is not None
    assert result["model"] == "gemini-3.8-live" and result["reply_sample"] == "باشە، تێگەیشتم."
    assert tts.texts == list(strings.SELFTEST_SENTENCES)
    model, config = client.connects[0]
    assert config.tools is None and config.system_instruction == SELFTEST_INSTRUCTION
    assert all(r["audio"].mime_type == "audio/pcm;rate=16000" for r in session.realtime if "audio" in r)
    stored = gemini_app.config.get("voice.selftest")
    assert stored["ok"] is True and len(stored["details"]) == 3
    assert gemini_app.db.query("SELECT * FROM activity WHERE name='selftest'")


async def test_kurmanji_or_persian_reply_fails_the_script_check(gemini_app):
    replies = [(s, "Baş e, min fêm kir.") for s in strings.SELFTEST_SENTENCES]
    client = FakeLiveClient([ScriptedSession(replies)])
    result = await run_selftest(gemini_app, tts=FakeTts(chunks=2, chunk_bytes=4800),
                                client_factory=lambda key: client, pace=0)
    assert result["script_ok"] is False and result["ok"] is False


async def test_poor_understanding_fails_on_cer(gemini_app):
    replies = [("سلام خوبی", "باشە.")] * 3
    client = FakeLiveClient([ScriptedSession(replies)])
    result = await run_selftest(gemini_app, tts=FakeTts(chunks=2, chunk_bytes=4800),
                                client_factory=lambda key: client, pace=0)
    assert result["cer"] > 0.5 and result["ok"] is False


async def test_primary_model_rejected_uses_the_fallback_model(gemini_app):
    client = FakeLiveClient([ApiError(404, "model not found"), ScriptedSession(GOOD)])
    result = await run_selftest(gemini_app, tts=FakeTts(chunks=2, chunk_bytes=4800),
                                client_factory=lambda key: client, pace=0)
    assert result["model"] == "gemini-3.1-flash-live-preview" and result["ok"] is True


async def test_silent_session_times_out_and_fails(gemini_app):
    client = FakeLiveClient([FakeLiveSession()])
    result = await asyncio.wait_for(run_selftest(gemini_app, tts=FakeTts(chunks=1, chunk_bytes=960),
                                                 client_factory=lambda key: client, pace=0,
                                                 turn_timeout_s=0.2), 5)
    assert result["ok"] is False and all(d["timed_out"] for d in result["details"])
