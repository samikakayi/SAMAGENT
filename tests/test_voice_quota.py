"""TTS/STT never stall and never waste the free quotas (2026-09-24 evening).

Real use: Gemini TTS answered 429 and the google-genai SDK waited ~27 s
before asking again (Retry-After), so the island sat on «بیردەکەمەوە» for
~40 s before KurdishTTS spoke. Covered with fakes and an httpx mock transport
(no network, no speakers).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import time
from zoneinfo import ZoneInfo

import httpx
import pytest
from conftest import FAKE_GEMINI
from voice_helpers import ApiError

from sam.voice import strings
from sam.voice.genai_client import make_client
from sam.voice.notices import VoiceNotice
from sam.voice.quota import ProviderRests, classify_429, next_pacific_midnight, reset_time_ckb, rests
from sam.voice.stt import GeminiStt, KurdishTtsStt
from sam.voice.tts import GeminiTts, TtsError, TtsRouter

DAILY_429 = ("429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Quota exceeded for metric: "
             "generativelanguage.googleapis.com/generate_requests_per_model_per_day, limit: 10', 'details': "
             "[{'violations': [{'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}]}, "
             "{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '27s'}]}}")
MINUTE_429 = "429 Quota exceeded for quota metric 'Requests per minute' (PerMinute). Please retry in 42.5s."
IRAQ = ZoneInfo("Asia/Baghdad")


class KurdishFake:
    provider = "kurdishtts"

    def __init__(self):
        self.calls: list[str] = []

    def configured(self):
        return True

    def voice_id(self):
        return "sorani_1|v4"

    async def stream(self, text):
        self.calls.append(text)
        yield b"\x01\x00" * 2400

    async def aclose(self):
        pass


class FakeInteractions:
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        return await self.behaviour()


def fake_client(behaviour):
    interactions = FakeInteractions(behaviour)
    client = type("C", (), {})()
    client.aio = type("A", (), {"interactions": interactions})()
    return client, interactions


async def gather_audio(router, text):
    return b"".join([chunk async for chunk in router.stream(text)])


def gemini_app(make_app):
    app = make_app(env_text=f"GEMINI_API_KEY={FAKE_GEMINI}\n")
    app.bus.bind_loop(asyncio.get_running_loop())
    return app


# -- quota helpers -----------------------------------------------------------------------------------------

def test_classify_429_bodies():
    assert classify_429(DAILY_429) == ("daily", 27.0)
    assert classify_429(MINUTE_429) == ("minute", 42.5)
    assert classify_429("429 Too Many Requests") == ("unlabelled", None)


def test_gemini_reset_is_10_am_in_iraq_in_summer_and_11_in_winter():
    evening = dt.datetime(2026, 9, 24, 21, 0, tzinfo=IRAQ).timestamp()
    reset = dt.datetime.fromtimestamp(next_pacific_midnight(evening), IRAQ)
    assert (reset.date(), reset.hour, reset.minute) == (dt.date(2026, 9, 25), 10, 0)
    assert reset_time_ckb(reset.timestamp()) == "کاتژمێر ١٠ی بەیانی"
    winter = dt.datetime.fromtimestamp(next_pacific_midnight(dt.datetime(2026, 12, 10, 21, 0, tzinfo=IRAQ)
                                                             .timestamp()), IRAQ)
    assert winter.hour == 11
    assert reset_time_ckb(dt.datetime(2026, 9, 25, 22, 30, tzinfo=IRAQ).timestamp()) == "کاتژمێر ١٠:٣٠ی شەو"


async def test_rests_persist_and_escalate(make_app):
    app = gemini_app(make_app)
    holder = ProviderRests(app)
    until = holder.on_rate_limit("gemini_tts", MINUTE_429)
    assert 59 <= until - time.time() <= 61 and holder.reason("gemini_tts") == "minute"
    assert ProviderRests(app).resting("gemini_tts")                      # stored in voice.rests
    holder.rest("gemini_tts", seconds=0.0, reason="x")                   # the rest is over
    holder.on_rate_limit("gemini_tts", ApiError(429, "Too Many Requests"))
    assert holder.reason("gemini_tts") == "daily"                        # a second 429 soon after: the day is gone
    assert abs(holder.until("gemini_tts") - next_pacific_midnight()) < 2


# -- the SDK never retries by itself ---------------------------------------------------------------------------------

async def test_genai_client_makes_exactly_one_request_on_429():
    requests: list[str] = []

    def handler(request):
        requests.append(request.url.path)
        return httpx.Response(429, headers={"retry-after": "27"}, json={"error": {
            "code": 429, "message": "Quota exceeded ... per_day", "status": "RESOURCE_EXHAUSTED"}})

    client = make_client(FAKE_GEMINI, httpx_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    began = time.perf_counter()
    with pytest.raises(Exception) as caught:
        events = await client.aio.interactions.create(model="gemini-3.8-flash-lite-tts", input="x", stream=True)
        async for _ in events:
            pass
    assert (getattr(caught.value, "status_code", None) or getattr(caught.value, "code", None)) == 429
    assert len(requests) == 1 and time.perf_counter() - began < 5      # the default SDK made 4 (Retry-After)
    requests.clear()
    with pytest.raises(Exception):
        await client.aio.models.generate_content(model="gemini-3.5-flash-lite", contents="x")
    assert len(requests) == 1


# -- Gemini TTS -> KurdishTTS at once ------------------------------------------------------------------------------------

async def test_daily_429_switches_to_kurdishtts_now_and_for_later_sentences(make_app):
    app = gemini_app(make_app)
    events = []
    app.bus.subscribe(VoiceNotice, events.append)

    async def refuse():
        raise ApiError(429, DAILY_429)

    client, interactions = fake_client(refuse)
    kurdish = KurdishFake()
    router = TtsRouter(app, {"gemini": GeminiTts(app, client_factory=lambda key: client), "kurdishtts": kurdish})
    began = time.perf_counter()
    assert await gather_audio(router, "سڵاو، چۆنی؟")
    assert time.perf_counter() - began < 1.0 and router.last_provider == "kurdishtts"
    assert await gather_audio(router, "ڕستەی دووەم.")
    assert interactions.calls == 1 and kurdish.calls == ["سڵاو، چۆنی؟", "ڕستەی دووەم."]
    assert rests(app).reason("gemini_tts") == "daily" and not router.providers["gemini"].configured()
    await asyncio.sleep(0.05)
    assert events and "دوای کاتژمێر" in events[-1].text_ckb
    assert events[-1].text_ckb.startswith(strings.GEMINI_VOICE_DAILY.split("{")[0])
    usage = app.db.usage_for("gemini", "gemini-3.8-flash-lite-tts", kind="tts")
    assert usage["errors"] == 1


async def test_no_first_audio_within_the_deadline_goes_to_kurdishtts(make_app):
    app = gemini_app(make_app)
    app.config.set("voice.tts_first_audio_s", 0.5)

    async def slow_stream():
        async def events():
            await asyncio.sleep(30)
            yield None
        return events()

    client, interactions = fake_client(slow_stream)
    kurdish = KurdishFake()
    router = TtsRouter(app, {"gemini": GeminiTts(app, client_factory=lambda key: client), "kurdishtts": kurdish})
    began = time.perf_counter()
    assert await gather_audio(router, "سڵاو")
    assert time.perf_counter() - began < 1.5 and kurdish.calls == ["سڵاو"]
    assert rests(app).reason("gemini_tts") == "timeout"
    with pytest.raises(TtsError) as caught:
        await gather_audio(TtsRouter(app, {"gemini": GeminiTts(app, client_factory=lambda key: client)}), "x")
    assert caught.value.kind == "unconfigured"                           # resting: not asked again


async def test_gemini_stt_429_rests_it(make_app):
    app = gemini_app(make_app)

    class Models:
        calls = 0

        async def generate_content(self, **kwargs):
            Models.calls += 1
            raise ApiError(429, MINUTE_429)

    client = type("C", (), {})()
    client.aio = type("A", (), {"models": Models()})()
    stt = GeminiStt(app, client_factory=lambda key: client)
    from sam.voice.stt import SttError
    with pytest.raises(SttError):
        await stt.transcribe(b"\x00\x00" * 16000)
    assert Models.calls == 1 and not stt.configured() and rests(app).reason("gemini_stt") == "minute"


async def test_kurdishtts_stt_month_used_up_is_shown(make_app):
    app = make_app(env_text="KURDISHTTS_STT_API_KEY=" + "ab" * 20 + "\n")
    app.bus.bind_loop(asyncio.get_running_loop())
    events = []
    app.bus.subscribe(VoiceNotice, events.append)
    stt = KurdishTtsStt(app, transport=httpx.MockTransport(lambda r: httpx.Response(403, text="credit exceeded")))
    from sam.voice.stt import SttError
    with pytest.raises(SttError):
        await stt.transcribe(b"\x00\x00" * 16000)
    await asyncio.sleep(0.05)
    assert not stt.configured() and events and events[-1].text_ckb == strings.KURDISH_STT_MONTH
