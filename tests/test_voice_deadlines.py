"""Never stall on a voice provider that accepts a request and never answers
(adversarial review 2026-09-24, stall_probe.py: KurdishTTS held a piece for the
client's 30 s read timeout, again for every piece; a Gemini TTS stream that
stalled after its first chunk held the voice line indefinitely).

KurdishTTS here is a local 127.0.0.1 server that accepts and never answers
(no internet); Gemini is a fake client. Deadlines are shortened by settings."""

from __future__ import annotations

import asyncio
import base64
import time
import types as pytypes

import httpx
import pytest
from conftest import FAKE_GEMINI

from sam.events import Error
from sam.voice import strings
from sam.voice.quota import rests
from sam.voice.stt import GeminiStt, KurdishTtsStt, SttError, SttRouter
from sam.voice.tts import GeminiTts, KurdishTts, TtsError, TtsRouter

KT_STT = "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c"
KT_TTS = "a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9"
ENV = f"KURDISHTTS_STT_API_KEY={KT_STT}\nKURDISHTTS_TTS_API_KEY={KT_TTS}\nGEMINI_API_KEY={FAKE_GEMINI}\n"


@pytest.fixture
async def silent_server():
    """A local HTTP 'server' that reads the request and never answers."""
    writers = []

    async def handle(reader, writer):
        writers.append(writer)
        try:
            await reader.read(65536)
            await asyncio.sleep(3600)
        except (asyncio.CancelledError, ConnectionError):
            pass

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}/api"
    for writer in writers:
        writer.close()
    server.close()


@pytest.fixture
def app(make_app, silent_server):
    app = make_app(env_text=ENV)
    app.config.set("voice.kurdishtts_base_url", silent_server)
    return app


def pcm(seconds: float) -> bytes:
    return (300).to_bytes(2, "little", signed=True) * int(16000 * seconds)


async def test_kurdishtts_tts_gives_up_after_its_first_audio_deadline_and_rests(app):
    app.config.set("voice.kurdishtts_tts_first_audio_s", 0.5)
    tts = KurdishTts(app, transport=httpx.AsyncHTTPTransport())
    began = time.perf_counter()
    with pytest.raises(TtsError) as info:
        async for _chunk in tts.stream("سڵاو، چۆنی؟"):
            pass
    assert time.perf_counter() - began < 3.0 and info.value.kind == "timeout"    # not the 30 s read timeout
    assert rests(app).resting("kurdishtts_tts") and not tts.configured()           # the next piece does not wait
    router = TtsRouter(app, {"kurdishtts": tts})
    assert router.resting_only()
    await tts.aclose()


async def test_kurdishtts_stt_deadline_then_the_other_provider_is_asked_first(app):
    app.config.set("voice.kurdishtts_stt_timeout_s", 0.3)
    reply = pytypes.SimpleNamespace(text="نرخی زێڕ", usage_metadata=None)

    class Models:
        async def generate_content(self, **kwargs):
            return reply

    gemini = GeminiStt(app, client_factory=lambda key: pytypes.SimpleNamespace(
        aio=pytypes.SimpleNamespace(models=Models())))
    kurdish = KurdishTtsStt(app, transport=httpx.AsyncHTTPTransport())
    router = SttRouter(app, {"kurdishtts": kurdish, "gemini": gemini})
    began = time.perf_counter()
    first = await router.transcribe(pcm(1.0))
    assert time.perf_counter() - began < 5.0                                      # 0.3 s + 1 s of audio
    assert first.attempts == ["kurdishtts:network", "gemini:ok"]
    second = await router.transcribe(pcm(1.0))
    assert second.attempts == ["gemini:ok"]                                       # resting KurdishTTS goes last
    assert router.configured()                                                   # a rest is not "no STT"
    await kurdish.aclose()


async def test_kurdishtts_stt_alone_fails_fast_instead_of_30_s(make_app, silent_server):
    app = make_app(env_text=f"KURDISHTTS_STT_API_KEY={KT_STT}\n")
    app.config.set("voice.kurdishtts_base_url", silent_server)
    app.config.set("voice.kurdishtts_stt_timeout_s", 0.3)
    kurdish = KurdishTtsStt(app, transport=httpx.AsyncHTTPTransport())
    began = time.perf_counter()
    with pytest.raises(SttError):
        await SttRouter(app, {"kurdishtts": kurdish}).transcribe(pcm(0.5))
    assert time.perf_counter() - began < 4.0
    await kurdish.aclose()


async def test_gemini_tts_stream_that_stalls_after_audio_ends_the_piece(app):
    app.config.set("voice.tts_chunk_gap_s", 0.5)
    chunk = pytypes.SimpleNamespace(event_type="step.delta",
                                    delta=pytypes.SimpleNamespace(type="audio",
                                                                  data=base64.b64encode(b"\x10\x00" * 4800)))

    class Interactions:
        async def create(self, **body):
            async def events():
                yield chunk
                await asyncio.sleep(3600)                                         # the stream stalls
                yield chunk
            return events()

    gemini = GeminiTts(app, client_factory=lambda key: pytypes.SimpleNamespace(
        aio=pytypes.SimpleNamespace(interactions=Interactions())))
    router = TtsRouter(app, {"gemini": gemini})
    got: list[bytes] = []
    began = time.perf_counter()
    async for audio in router.stream("سڵاو، چۆنی؟"):
        got.append(audio)
    assert got and time.perf_counter() - began < 3.0                              # the piece ended, audio kept
    assert rests(app).resting("gemini_tts")                                       # next piece -> KurdishTTS


async def test_cascade_says_once_that_the_voice_rests(app):
    from voice_helpers import FakeSpeaker

    from sam.voice.cascade import CascadeVoice
    from sam.voice.hooks import RecordingHooks

    app.bus.bind_loop(asyncio.get_running_loop())
    events: list = []
    app.bus.subscribe(Error, events.append)
    rests(app).on_transient("kurdishtts_tts", "timeout")
    rests(app).on_transient("gemini_tts", "timeout")
    router = TtsRouter(app, {"gemini": GeminiTts(app), "kurdishtts": KurdishTts(app)})
    cascade = CascadeVoice(app, FakeSpeaker(), None, router, RecordingHooks())
    await cascade.speak("یەکەم.")
    await cascade.speak("دووەم.")
    await asyncio.sleep(0.05)
    messages = [e.message_ckb for e in events]
    assert messages == [strings.TTS_RESTING]
