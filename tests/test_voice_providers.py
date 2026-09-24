"""STT and TTS providers over httpx.MockTransport / fake google-genai clients:
request shapes, error mapping, fallbacks, budgets, and no key in any error."""

from __future__ import annotations

import base64
import json
import types as pytypes

import httpx
import pytest
from conftest import FAKE_GEMINI

from sam.events import ComponentStatus
from sam.voice.audio import wav_to_pcm16
from sam.voice.stt import GeminiStt, KurdishTtsStt, SttError, SttRouter
from sam.voice.tts import GeminiTts, KurdishTts, TtsError, TtsRouter

KT_STT = "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c"
KT_TTS = "a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9"
ENV = f"KURDISHTTS_STT_API_KEY={KT_STT}\nKURDISHTTS_TTS_API_KEY={KT_TTS}\n"


@pytest.fixture
def app(make_app):
    return make_app(env_text=ENV)


@pytest.fixture
def gemini_app(make_app):
    app = make_app(env_text=ENV + f"GEMINI_API_KEY={FAKE_GEMINI}\n")
    # These tests exercise "Gemini first, KurdishTTS as fallback" (still a user
    # choice); the shipped default is the reverse (test_default_tts_order_*).
    app.config.set("voice.tts_provider", "gemini")
    return app


def pcm(seconds: float, value: int = 300) -> bytes:
    return (value.to_bytes(2, "little", signed=True)) * int(16000 * seconds)


# -- KurdishTTS STT ------------------------------------------------------------------------------------

async def test_kurdishtts_stt_request_shape_and_result(app):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-api-key")
        seen["body"] = request.read()
        return httpx.Response(200, json={"text": "نرخی زێڕ چەندە؟ ي ك", "detected_dialect": "sorani"})

    stt = KurdishTtsStt(app, transport=httpx.MockTransport(handler))
    result = await stt.transcribe(pcm(0.4))
    assert seen["url"] == "https://www.kurdishtts.com/api/stt-proxy" and seen["key"] == KT_STT
    body = seen["body"]
    assert b'name="dialect"' in body and b"sorani" in body and b'filename="audio.wav"' in body
    wav = body[body.index(b"RIFF"):]
    audio, rate = wav_to_pcm16(wav[: wav.index(b"\r\n--")])
    assert rate == 16000 and len(audio) >= 32000          # padded to >= 1 s
    assert result.text == "نرخی زێڕ چەندە؟ ی ک" and result.provider == "kurdishtts"
    usage = app.db.usage_for("kurdishtts", "stt-proxy", "stt")
    assert usage["requests"] == 1 and usage["units"] == pytest.approx(1.0)
    await stt.aclose()


@pytest.mark.parametrize("status,kind", [(401, "auth"), (403, "quota"), (429, "rate_limit"), (500, "server"),
                                         (400, "bad_request")])
async def test_kurdishtts_stt_errors_never_carry_the_key(app, status, kind):
    transport = httpx.MockTransport(lambda r: httpx.Response(status, text=f"bad key {KT_STT}"))
    stt = KurdishTtsStt(app, transport=transport)
    with pytest.raises(SttError) as info:
        await stt.transcribe(pcm(1.2))
    assert info.value.kind == kind and KT_STT not in str(info.value)
    if kind == "quota":
        assert not stt.configured()                     # the month's plan is used up


async def test_kurdishtts_stt_monthly_budget_blocks_before_network(app):
    app.config.set("voice.kurdishtts_monthly_stt_s", 2)
    app.db.bump_usage("kurdishtts", "stt-proxy", kind="stt", units=3.0)
    calls = []
    stt = KurdishTtsStt(app, transport=httpx.MockTransport(lambda r: calls.append(r) or httpx.Response(200, json={})))
    with pytest.raises(SttError) as info:
        await stt.transcribe(pcm(1))
    assert info.value.kind == "quota" and calls == []


class FakeModels:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def generate_content(self, *, model, contents, config):
        self.calls.append((model, contents, config))
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def fake_genai(models=None, interactions=None):
    aio = pytypes.SimpleNamespace(models=models, interactions=interactions)
    return pytypes.SimpleNamespace(aio=aio)


class CodeError(Exception):
    def __init__(self, code, text=""):
        super().__init__(f"{code} {text}")
        self.code = code


async def test_gemini_stt_prompt_audio_and_thinking_retry(gemini_app):
    reply = pytypes.SimpleNamespace(text="«تکایە ترەیدینگ ڤیو بکەرەوە»",
                                    usage_metadata=pytypes.SimpleNamespace(prompt_token_count=40, candidates_token_count=9))
    models = FakeModels([CodeError(400, "thinking level MINIMAL is not supported"), reply])
    stt = GeminiStt(gemini_app, client_factory=lambda key: fake_genai(models=models))
    result = await stt.transcribe(pcm(1.5))
    assert result.text == "تکایە ترەیدینگ ڤیو بکەرەوە" and result.model == "gemini-3.5-flash-lite"
    (model, contents, first_cfg), (_, _, second_cfg) = models.calls
    assert first_cfg.thinking_config is not None and second_cfg.thinking_config is None
    audio_part, text_part = contents[0].parts
    assert audio_part.inline_data.mime_type == "audio/wav" and audio_part.inline_data.data[:4] == b"RIFF"
    assert "verbatim" in first_cfg.system_instruction.lower() and "Sorani" in first_cfg.system_instruction
    assert first_cfg.temperature == 0.0
    usage = gemini_app.db.usage_for("gemini", "gemini-3.5-flash-lite", "stt")
    assert usage["requests"] == 1 and usage["tokens_in"] == 40


async def test_gemini_stt_rate_limit_cools_down_and_daily_cap(gemini_app):
    stt = GeminiStt(gemini_app, client_factory=lambda key: fake_genai(models=FakeModels([CodeError(429)])))
    with pytest.raises(SttError) as info:
        await stt.transcribe(pcm(1))
    assert info.value.kind == "rate_limit" and not stt.configured()
    capped = GeminiStt(gemini_app, client_factory=lambda key: fake_genai(models=FakeModels([])))
    gemini_app.config.set("llm.daily_caps", {"gemini:gemini-3.5-flash-lite": 1})
    with pytest.raises(SttError) as info:
        await capped.transcribe(pcm(1))
    assert info.value.kind == "quota"


async def test_stt_router_falls_back_on_failure_not_on_content(gemini_app):
    reply = pytypes.SimpleNamespace(text="سڵاو", usage_metadata=None)
    models = FakeModels([reply])
    router = SttRouter(gemini_app, {
        "kurdishtts": KurdishTtsStt(gemini_app, transport=httpx.MockTransport(lambda r: httpx.Response(503))),
        "gemini": GeminiStt(gemini_app, client_factory=lambda key: fake_genai(models=models))})
    result = await router.transcribe(pcm(1))
    assert result.provider == "gemini" and result.attempts == ["kurdishtts:server", "gemini:ok"]
    empty = SttRouter(gemini_app, {"kurdishtts": KurdishTtsStt(gemini_app, transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"text": ""})))})
    assert (await empty.transcribe(pcm(1))).text == ""   # empty is an answer, not a failure


async def test_stt_router_without_keys(make_app):
    app = make_app()
    router = SttRouter.default(app)
    assert not router.configured()
    with pytest.raises(SttError) as info:
        await router.transcribe(pcm(1))
    assert info.value.kind == "unconfigured"


# -- KurdishTTS TTS -------------------------------------------------------------------------------------------

async def test_kurdishtts_tts_streams_even_pcm_and_counts_characters(app):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["json"] = json.loads(request.read())
        seen["key"] = request.headers.get("x-api-key")
        return httpx.Response(200, headers={"content-type": "audio/pcm"},
                              stream=httpx.ByteStream(b"\x01\x02\x03" * 101))

    tts = KurdishTts(app, transport=httpx.MockTransport(handler))
    chunks = [c async for c in tts.stream("سڵاو، چۆنی؟")]
    assert all(len(c) % 2 == 0 for c in chunks) and sum(map(len, chunks)) == 302
    assert seen["key"] == KT_TTS
    assert seen["json"] == {"text": "سڵاو، چۆنی؟", "speaker_id": "sorani_1", "model_version": "v4",
                            "stream_format": "pcm"}
    assert app.db.usage_for("kurdishtts", "tts-stream", "tts")["units"] == len("سڵاو، چۆنی؟")


async def test_kurdishtts_tts_quota_json_and_budget_warning(app):
    app.bus.bind_loop(__import__("asyncio").get_running_loop())
    events = []
    app.bus.subscribe(ComponentStatus, events.append)
    quota = KurdishTts(app, transport=httpx.MockTransport(lambda r: httpx.Response(403, json={"error": "quota"})))
    with pytest.raises(TtsError) as info:
        [c async for c in quota.stream("سڵاو")]
    assert info.value.kind == "quota" and not quota.configured()
    html = KurdishTts(app, transport=httpx.MockTransport(
        lambda r: httpx.Response(200, headers={"content-type": "application/json"}, json={"x": 1})))
    with pytest.raises(TtsError):
        [c async for c in html.stream("سڵاو")]
    app.config.set("voice.kurdishtts_monthly_tts_chars", 10)
    warn = KurdishTts(app, transport=httpx.MockTransport(
        lambda r: httpx.Response(200, headers={"content-type": "audio/pcm"}, content=b"\x00\x00" * 10)))
    [c async for c in warn.stream("سڵاو چۆنی")]                 # 9 of 10 characters: >= 80 %
    await __import__("asyncio").sleep(0.01)
    assert any(e.component == "kurdishtts" and e.state == "degraded" for e in events)
    assert warn.month_used() == 9 and warn.configured()      # warned, still under the budget


# -- Gemini TTS -------------------------------------------------------------------------------------------------------

class FakeInteractions:
    def __init__(self, script):
        self.script = list(script)
        self.bodies = []

    async def create(self, **body):
        self.bodies.append(body)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item

        async def events():
            for event in item:
                yield event
        return events()


def delta(pcm_bytes: bytes):
    return pytypes.SimpleNamespace(event_type="step.delta",
                                   delta=pytypes.SimpleNamespace(type="audio", data=base64.b64encode(pcm_bytes).decode()))


async def test_gemini_tts_request_body_and_streamed_pcm(gemini_app):
    interactions = FakeInteractions([[pytypes.SimpleNamespace(event_type="interaction.start"),
                                      delta(b"\x01\x00\x02"), delta(b"\x00\x03\x00"),
                                      pytypes.SimpleNamespace(event_type="step.delta",
                                                              delta=pytypes.SimpleNamespace(type="text", data=None))]])
    tts = GeminiTts(gemini_app, client_factory=lambda key: fake_genai(interactions=interactions))
    audio = b"".join([c async for c in tts.stream("سڵاو، ئەمڕۆ باشیت؟")])
    assert audio == b"\x01\x00\x02\x00\x03\x00"
    body = interactions.bodies[0]
    assert body["model"] == "gemini-3.8-flash-lite-tts" and body["stream"] is True and body["store"] is False
    assert body["input"] == [{"type": "user_input", "content": [{"type": "text", "text": "سڵاو، ئەمڕۆ باشیت؟"}]}]
    assert body["response_format"] == {"type": "audio", "mime_type": "audio/l16", "sample_rate": 24000}
    assert body["generation_config"] == {"speech_config": [{"voice": "Kore"}]}
    assert gemini_app.db.usage_for("gemini", "gemini-3.8-flash-lite-tts", "tts")["units"] == len("سڵاو، ئەمڕۆ باشیت؟")


async def test_gemini_tts_retries_without_store_and_style_goes_to_metadata(gemini_app):
    gemini_app.config.set("voice.tts_style", "calm, friendly")
    interactions = FakeInteractions([CodeError(400, "Unknown field store"), [delta(b"\x00\x00")]])
    tts = GeminiTts(gemini_app, client_factory=lambda key: fake_genai(interactions=interactions))
    assert [c async for c in tts.stream("باشە")] == [b"\x00\x00"]
    assert "store" not in interactions.bodies[1]
    content = interactions.bodies[1]["input"][0]["content"][0]
    assert content["text"] == "باشە"                        # verbatim: no stage directions in the text
    assert content["annotations"] == [{"type": "speech_metadata", "style": "calm, friendly"}]


async def test_tts_router_falls_back_before_first_chunk_only(gemini_app):
    interactions = FakeInteractions([CodeError(429, "RESOURCE_EXHAUSTED")])
    gem = GeminiTts(gemini_app, client_factory=lambda key: fake_genai(interactions=interactions))
    kt = KurdishTts(gemini_app, transport=httpx.MockTransport(
        lambda r: httpx.Response(200, headers={"content-type": "audio/pcm"}, content=b"\x05\x00" * 4)))
    router = TtsRouter(gemini_app, {"gemini": gem, "kurdishtts": kt})
    audio = b"".join([c async for c in router.stream("سڵاو")])
    assert audio == b"\x05\x00" * 4 and router.last_provider == "kurdishtts"
    assert not gem.configured()                              # cooling down after 429


async def test_tts_router_does_not_repeat_words_after_audio_started(gemini_app):
    class HalfTts:
        provider = "gemini"

        def configured(self):
            return True

        async def stream(self, text):
            yield b"\x01\x00"
            raise TtsError("network", provider="gemini")

        async def aclose(self):
            pass

    used = []
    kt = KurdishTts(gemini_app, transport=httpx.MockTransport(lambda r: used.append(1) or httpx.Response(200)))
    router = TtsRouter(gemini_app, {"gemini": HalfTts(), "kurdishtts": kt})
    assert b"".join([c async for c in router.stream("سڵاو")]) == b"\x01\x00" and used == []


async def test_tts_router_splits_long_text_into_capped_requests(gemini_app):
    texts = []

    class Recorder:
        provider = "gemini"

        def configured(self):
            return True

        async def stream(self, text):
            texts.append(text)
            yield b"\x00\x00"

        async def aclose(self):
            pass

    router = TtsRouter(gemini_app, {"gemini": Recorder()})
    [c async for c in router.stream("ئەمە ڕستەیەکی درێژە بۆ تاقیکردنەوە. " * 40)]
    assert len(texts) >= 3 and all(len(t) <= 480 for t in texts)


async def test_tts_router_order_without_gemini_key(app):
    router = TtsRouter.default(app)
    assert [p.provider for p in router.order() if p.configured()] == ["kurdishtts"]


def test_default_tts_order_is_kurdishtts_then_gemini(make_app):
    """The user's A/B listening test (2026-09-24) chose KurdishTTS's voice:
    it speaks first by default, Gemini TTS only when KurdishTTS cannot."""
    app = make_app(env_text=ENV + f"GEMINI_API_KEY={FAKE_GEMINI}\n")
    assert app.config.get("voice.tts_provider") == "kurdishtts"
    router = TtsRouter.default(app)
    assert [p.provider for p in router.order()] == ["kurdishtts", "gemini"]
