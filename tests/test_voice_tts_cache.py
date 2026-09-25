"""TTS phrase cache: registered fixed phrases (the brain's acknowledgements)
are synthesized once, persisted and replayed without a provider request;
ordinary reply sentences, cut-off audio and another voice never are."""

from __future__ import annotations

import asyncio

from sam.voice.tts import TtsError, TtsRouter
from sam.voice.tts_cache import PhraseCache


class CountingTts:
    provider = "kurdishtts"

    def __init__(self, voice: str = "sorani_1|v4", fail: bool = False) -> None:
        self.calls: list[str] = []
        self.voice = voice
        self.fail = fail

    def configured(self) -> bool:
        return True

    def voice_id(self) -> str:
        return self.voice

    async def stream(self, text: str):
        self.calls.append(text)
        if self.fail:
            raise TtsError("quota", "403", provider=self.provider)
        for i in range(3):
            await asyncio.sleep(0)
            yield bytes([i + 1, 0]) * 2400          # 3 x 100 ms


async def collect(router: TtsRouter, text: str) -> bytes:
    return b"".join([chunk async for chunk in router.stream(text)])


async def test_short_phrase_is_synthesized_once_then_replayed(make_app):
    app = make_app()
    provider = CountingTts()
    router = TtsRouter(app, {"kurdishtts": provider}, cache=PhraseCache(app.db))
    assert router.register_phrases(["یەک چرکە."]) == 1
    first = await collect(router, "یەک چرکە.")
    assert provider.calls == ["یەک چرکە."] and router.last_cached is False
    second = await collect(router, "یەک چرکە.")
    assert second == first and provider.calls == ["یەک چرکە."] and router.last_cached is True
    # persisted: a new cache object (next SAM start) on the same database hits too
    fresh = TtsRouter(app, {"kurdishtts": provider}, cache=PhraseCache(app.db))
    fresh.register_phrases(["یەک چرکە."])
    assert await collect(fresh, "یەک چرکە.") == first and provider.calls == ["یەک چرکە."]
    row = app.db.query_one("SELECT provider, voice, text, uses FROM voice_tts_cache")
    assert row["provider"] == "kurdishtts" and row["text"] == "یەک چرکە." and row["uses"] == 2
    assert router.status()["cache"]["hits"] == 1


async def test_reply_sentences_other_voice_and_punctuation_are_not_shared(make_app):
    app = make_app()
    provider = CountingTts()
    router = TtsRouter(app, {"kurdishtts": provider}, cache=PhraseCache(app.db))
    router.register_phrases(["باشە.", "باشە؟", "x" * 130])
    reply = "ئێستا زێڕ چوار هەزار و دوو سەد و پەنجا و هەشتە."   # measured live: a one-off sentence
    await collect(router, reply)
    await collect(router, reply)
    assert provider.calls.count(reply) == 2                       # not registered: never cached
    assert not router.cache.has("kurdishtts", provider.voice, reply)
    assert router.cache.cacheable("x" * 130) is False              # too long to register
    await collect(router, "باشە.")
    await collect(router, "باشە؟")                                # other intonation, other entry
    provider.voice = "sorani_986|v4"                              # user picked another voice
    await collect(router, "باشە.")
    assert provider.calls[2:] == ["باشە.", "باشە؟", "باشە."]


async def test_audio_cut_by_barge_in_is_not_cached(make_app):
    app = make_app()
    provider = CountingTts()
    router = TtsRouter(app, {"kurdishtts": provider}, cache=PhraseCache(app.db))
    router.register_phrases(["با بزانم."])
    stream = router.stream("با بزانم.")
    async for _chunk in stream:
        break                                                     # the speaker was flushed
    await stream.aclose()
    assert app.db.query("SELECT key FROM voice_tts_cache") == []
    await collect(router, "با بزانم.")
    assert provider.calls == ["با بزانم.", "با بزانم."]


async def test_cache_is_pruned_to_the_most_recently_used(make_app):
    app = make_app()
    cache = PhraseCache(app.db, max_rows=3)
    cache.register([f"ڕستە {i}" for i in range(5)])
    for i in range(5):
        assert cache.put("kurdishtts", "v", f"ڕستە {i}", b"\x01\x00" * 10)
    rows = app.db.query("SELECT text FROM voice_tts_cache ORDER BY rowid")
    assert len(rows) == 3 and rows[-1]["text"] == "ڕستە 4"


async def test_prewarm_adds_only_missing_phrases_and_waits_for_idle(make_app):
    app = make_app()
    provider = CountingTts()
    router = TtsRouter(app, {"kurdishtts": provider}, cache=PhraseCache(app.db))
    router.register_phrases(["باشە."])
    await collect(router, "باشە.")
    busy = {"on": True}

    async def release():
        await asyncio.sleep(0.05)
        busy["on"] = False

    releaser = asyncio.ensure_future(release())
    added = await router.prewarm(["باشە.", "بەسەرچاو.", "x" * 130, "یەک چرکە."], idle=lambda: not busy["on"],
                                 pause_s=0.01)
    await releaser
    assert added == 2 and provider.calls == ["باشە.", "بەسەرچاو.", "یەک چرکە."]
    assert await router.prewarm(["باشە.", "بەسەرچاو."]) == 0


async def test_prewarm_stops_quietly_when_the_provider_refuses(make_app):
    app = make_app()
    router = TtsRouter(app, {"kurdishtts": CountingTts(fail=True)}, cache=PhraseCache(app.db))
    assert await router.prewarm(["باشە.", "یەک چرکە."]) == 0


async def test_prewarm_is_off_by_default_and_never_uses_gemini(make_app):
    """Real use 2026-09-24: the startup prewarm (plus the self-test) spent the
    free Gemini TTS quota before the user's first sentence. Now phrases are
    cached lazily when really spoken; an opted-in prewarm uses KurdishTTS only."""
    from voice_helpers import FakeMic, FakeSpeaker, FakeStt

    from sam.brain.conversation import ACKS_DO, ACKS_LOOK
    from sam.voice import strings
    from sam.voice.engine import VoiceEngine

    app = make_app()
    app.bus.bind_loop(asyncio.get_running_loop())
    gemini, kurdish = CountingTts(), CountingTts()
    gemini.provider = "gemini"
    router = TtsRouter(app, {"gemini": gemini, "kurdishtts": kurdish}, cache=PhraseCache(app.db))
    eng = VoiceEngine(app, mic_factory=FakeMic, speaker=FakeSpeaker(), stt=FakeStt([]), tts=router)
    await eng.start_listening()
    await asyncio.sleep(0.2)
    assert gemini.calls == [] and kurdish.calls == []             # default: no prewarm at all
    await eng.stop_listening()
    app.config.set("voice.tts_prewarm", True)
    eng._prewarm_started = False  # noqa: SLF001
    await eng.start_listening()
    expected = [*ACKS_DO, *ACKS_LOOK, strings.STT_FAILED_SPOKEN]
    for _ in range(200):
        if len(kurdish.calls) == len(expected):
            break
        await asyncio.sleep(0.01)
    assert kurdish.calls == expected and gemini.calls == []      # never the Gemini quota
    await eng.stop()
