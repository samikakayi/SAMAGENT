"""VoiceEngine: package registration, engine choice, listening/conversation
window, mute, spoken alerts and confirmations, Live -> Cascade fallback,
echo guard and the hotkey (all with fake devices; nothing is played or recorded)."""

from __future__ import annotations

import asyncio

import pytest
from conftest import FAKE_GEMINI
from voice_helpers import (msg, FakeLiveClient, FakeMic, FakeSpeaker, FakeStt, FakeTts, audio_msg, fake_llm, quiet_frame,
                           settle, tone_frame)

import sam.voice.engine as engine_mod
from sam.events import ConfirmRequest, Error, SpeakRequest, VoiceState
from sam.voice import strings
from sam.voice.engine import VoiceEngine
from sam.voice.live import LiveVoice


class FakeHotkey:
    instances: list["FakeHotkey"] = []

    def __init__(self, keys, callback, ok=True):
        self.keys, self.callback, self.ok = keys, callback, ok
        self.registered = False
        self.error = None if ok else "already in use by another program"
        FakeHotkey.instances.append(self)

    def start(self):
        self.registered = self.ok
        return self.ok

    def stop(self):
        self.registered = False

    def press(self):
        self.callback()


class EnergyClassifier:
    """VAD stand-in: 'speech' = loud frame (webrtcvad itself is tested separately)."""

    def __init__(self, **kwargs):
        pass

    def is_speech(self, frame, rms):
        return rms > 0.05


@pytest.fixture
async def voice(make_app, monkeypatch):
    monkeypatch.setattr(engine_mod, "FrameClassifier", EnergyClassifier)
    engines = []

    async def build(*, gemini=False, llm=None, stt=None, tts=None, client=None, hotkey_ok=True, selftest_ok=True):
        app = make_app(env_text=f"GEMINI_API_KEY={FAKE_GEMINI}\n" if gemini else "")
        if gemini and selftest_ok:
            # "Automatic" uses Live only after a passing self-test (design 2.1).
            app.config.set("voice.selftest", {"ok": True, "cer": 0.1, "at": 1.0})
        app.bus.bind_loop(asyncio.get_running_loop())
        events = []
        app.bus.subscribe(None, events.append)
        mics: list[FakeMic] = []
        speaker = FakeSpeaker()
        client = client or FakeLiveClient()

        def mic_factory():
            mics.append(FakeMic())
            return mics[-1]

        eng = VoiceEngine(app, mic_factory=mic_factory, speaker=speaker, stt=stt or FakeStt(["نرخی زێڕ چەندە؟"]),
                          tts=tts or FakeTts(), llm_stream=llm or fake_llm(["نرخی زێڕ ٢٦٥٠ دۆلارە."]),
                          hotkey_factory=lambda keys, cb: FakeHotkey(keys, cb, ok=hotkey_ok),
                          live_factory=lambda: LiveVoice(app, speaker, eng, client_factory=lambda key: client,
                                                         reconnect_delays=(0.01,), ready_timeout_s=2.0))
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


def states(events):
    return [e.state for e in events if isinstance(e, VoiceState)]


def speak_utterance(mic: FakeMic, speech_frames: int = 30, silence_frames: int = 25) -> None:
    mic.push(tone_frame(), speech_frames)
    mic.push(quiet_frame(), silence_frames)


async def test_register_through_the_app(make_app):
    app = make_app()
    status = app.load_packages(["sam.voice"])
    assert status == {"sam.voice": "ok"} and isinstance(app.voice, VoiceEngine)
    assert app.config.get("voice.kurdishtts_speaker") == "sorani_1"
    assert app.config.get("voice.hotkey") == "ctrl+alt+space"
    assert not app.voice.listening and app.voice.mic is None
    assert app.tools.names() == ["stop_all"]            # voice registers no tools


def test_register_imports_no_heavy_modules(tmp_path):
    """Startup budget: google.genai (~2 s), numpy (~2 s), sounddevice,
    webrtcvad and httpx (~41 ms: most of voice's register time before it was
    made lazy) must not be imported by register (checked in a fresh process)."""
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    (tmp_path / "data").mkdir()
    code = ("import sys, time; from sam.app import App; app = App(sys.argv[1], environ={}); t = time.perf_counter(); "
            "s = app.load_packages(['sam.voice']); ms = (time.perf_counter() - t) * 1000; "
            "heavy = [m for m in ('google.genai', 'numpy', 'sounddevice', 'webrtcvad', 'httpx') if m in sys.modules]; "
            "print(s['sam.voice'], heavy, round(ms)); app.close()")
    out = subprocess.run([sys.executable, "-c", code, str(tmp_path)], cwd=root, capture_output=True, text=True,
                         timeout=60)
    assert out.stdout.startswith("ok []"), out.stdout + out.stderr


async def test_engine_choice_rules(voice):
    app, eng, *_ = await voice()
    assert eng.choose_engine() == "cascade"                               # no Gemini key today
    app2, eng2, *_ = await voice(gemini=True, selftest_ok=False)
    # Live only after a PASSING self-test (design 2.1; the first real one on
    # 2026-09-24 measured CER 0.54): not run, failed or inconclusive -> cascade.
    assert eng2.choose_engine() == "cascade"                              # key, self-test never run
    app2.config.set("voice.selftest", {"ok": False, "cer": 0.6})
    assert eng2.choose_engine() == "cascade"                              # failed self-test
    app2.config.set("voice.selftest", {"ok": False, "error": "no_gemini_key"})
    assert eng2.choose_engine() == "cascade"                              # that run had no key
    app2.config.set("voice.selftest", {"ok": False, "error": "live_transient: ConnectionError"})
    assert eng2.choose_engine() == "cascade"                              # network hiccup: inconclusive
    app2.config.set("voice.selftest", {"ok": False, "error": "tts_failed: TtsError"})
    assert eng2.choose_engine() == "cascade"
    app2.config.set("voice.selftest", {"ok": True, "cer": 0.1})
    assert eng2.choose_engine() == "live" and eng2.live_text_trusted()   # passed
    app2.config.set("voice.selftest", {"ok": False, "error": "live_auth: 403"})
    assert eng2.choose_engine() == "cascade"                              # this key cannot open Live
    app2.config.set("voice.engine", "live")
    app2.config.set("voice.selftest", {"ok": False, "cer": 0.9})
    assert eng2.choose_engine() == "live"                                 # user forced Live
    app2.config.set("voice.engine", "cascade")
    assert eng2.choose_engine() == "cascade"
    app2.config.set("voice.engine", "auto")
    app2.config.set("voice.selftest", {"ok": True})
    eng2.live_degraded = True
    assert eng2.choose_engine() == "cascade"


async def test_automatic_selftest_runs_once_and_retries_only_inconclusive(voice):
    import time
    app, eng, *_ = await voice(gemini=True, selftest_ok=False)
    runs = []

    async def fake_selftest():
        runs.append(app.config.get("voice.selftest"))
        return {"ok": True}

    eng.run_selftest = fake_selftest
    await asyncio.wait_for(eng._auto_selftest(first_delay_s=0, every_s=0.01), 2)    # never ran -> runs  # noqa: SLF001
    assert len(runs) == 1
    app.config.set("voice.selftest", {"ok": False, "error": "live_quota: 429", "at": time.time()})
    with pytest.raises(asyncio.TimeoutError):                                        # too recent: waits
        await asyncio.wait_for(eng._auto_selftest(first_delay_s=0, every_s=0.01), 0.2)  # noqa: SLF001
    assert len(runs) == 1
    app.config.set("voice.selftest", {"ok": False, "error": "live_quota: 429", "at": time.time() - 7200})
    await asyncio.wait_for(eng._auto_selftest(first_delay_s=0, every_s=0.01), 2)  # noqa: SLF001
    assert len(runs) == 2
    app.config.set("voice.selftest", {"ok": False, "cer": 0.8, "at": time.time() - 7200})
    with pytest.raises(asyncio.TimeoutError):                                        # a measured fail stays
        await asyncio.wait_for(eng._auto_selftest(first_delay_s=0, every_s=0.01), 0.2)  # noqa: SLF001
    assert len(runs) == 2


async def test_cascade_turn_end_to_end_through_the_engine(voice):
    app, eng, mics, speaker, _, events = await voice()
    assert await eng.toggle_listening() is True
    assert eng.listening and eng.engine_name == "cascade" and mics[0].started
    speak_utterance(mics[0])
    assert await settle(lambda: eng.tts.texts == ["نرخی زێڕ ٢٦٥٠ دۆلارە."])
    assert speaker.chunks and "listening" in states(events) and "speaking" in states(events)
    assert any(e.source == "mic" for e in events if type(e).__name__ == "LevelMeter")
    assert await eng.toggle_listening() is False
    assert mics[0].stopped and states(events)[-1] == "sleeping"


async def test_conversation_window_sleeps_after_silence(voice):
    app, eng, mics, *_, events = await voice()
    app.config.set("voice.conversation_timeout_s", 1)
    await eng.start_listening()
    assert await settle(lambda: not eng.listening, timeout=4.0)
    await settle()
    assert states(events)[-1] == "sleeping" and mics[0].stopped, [(e.state, e.detail, e.at) for e in events if isinstance(e, VoiceState)]


async def test_always_listening_keeps_the_window_open(voice):
    app, eng, *_ = await voice()
    app.config.set("voice.conversation_timeout_s", 1)
    app.config.set("voice.always_listening", True)
    await eng.start_listening()
    await asyncio.sleep(2.2)
    assert eng.listening


async def test_speak_request_and_confirmation_question(voice):
    app, eng, mics, speaker, _, events = await voice()
    app.bus.publish(SpeakRequest(text_ckb="زێڕ گەیشتە ٢٧٠٠ دۆلار.", source="alert"))
    assert await settle(lambda: "زێڕ گەیشتە ٢٧٠٠ دۆلار." in eng.tts.texts)
    assert not eng.listening                      # alerts do not open the mic
    app.bus.publish(ConfirmRequest(confirm_id="c1", question_ckb="فایلەکە بسڕمەوە؟", tool_name="files"))
    assert await settle(lambda: "فایلەکە بسڕمەوە؟" in eng.tts.texts)
    assert eng.listening                          # SAM asks, then listens for بەڵێ / نەخێر


async def test_mute_closes_the_mic_but_alerts_are_still_spoken(voice):
    app, eng, mics, speaker, _, events = await voice()
    await eng.start_listening()
    await eng.set_muted(True)
    assert mics[0].stopped and eng.muted and states(events)[-1] == "muted"
    await eng.speak("ئاگاداری: نرخ گۆڕا.", source="alert")
    assert "ئاگاداری: نرخ گۆڕا." in eng.tts.texts
    await settle()
    assert states(events)[-1] == "muted"          # never "sleeping": the conversation goes on
    await eng.set_muted(False)
    assert eng.listening and len(mics) == 2


async def test_live_engine_streams_mic_audio_and_speaks_model_audio(voice):
    app, eng, mics, speaker, client, events = await voice(gemini=True)
    await eng.start_listening()
    assert eng.engine_name == "live"
    assert await settle(lambda: eng.live is not None and eng.live.ready)
    mics[0].push(quiet_frame(), 3)
    session = client.sessions[0]
    assert await settle(lambda: len(session.audio) >= 3)
    session.push(audio_msg(b"\x09\x00" * 480))
    assert await settle(lambda: speaker.chunks)
    assert eng.status()["live"]["connected"] is True


async def test_typed_text_goes_into_an_open_live_session(voice):
    from sam.events import Transcript
    app, eng, mics, speaker, client, events = await voice(gemini=True)
    assert eng.live_session_open is False and await eng.send_text("سڵاو") is False
    await eng.start_listening()
    assert await settle(lambda: eng.live_session_open)
    assert await eng.send_text("نرخی زێڕ چەندە؟") is True
    session = client.sessions[0]
    assert {"text": "نرخی زێڕ چەندە؟"} in session.realtime
    typed = [e for e in events if isinstance(e, Transcript) and e.role == "user"]
    assert typed and typed[-1].source == "text" and typed[-1].text == "نرخی زێڕ چەندە؟"


async def test_cascade_mode_does_not_take_typed_text(voice):
    app, eng, *_ = await voice()
    await eng.start_listening()
    assert eng.engine_name == "cascade" and not eng.live_session_open
    assert await eng.send_text("سڵاو") is False


async def test_live_stall_falls_back_to_cascade_for_the_window(voice):
    app, eng, mics, speaker, client, events = await voice(gemini=True)
    app.config.set("voice.watchdog_s", 0.2)
    await eng.start_listening()
    assert await settle(lambda: eng.live is not None and eng.live.ready)
    session = client.sessions[0]
    speak_utterance(mics[0])
    session.push(msg({"server_content": {"input_transcription": {"text": "نرخی زێڕ چەندە؟"}}}))
    # Live heard the words but never answers.
    assert await settle(lambda: eng.engine_name == "cascade", timeout=3.0)
    assert eng.live_degraded and eng.live is None
    assert await settle(lambda: eng.tts.texts == ["نرخی زێڕ ٢٦٥٠ دۆلارە."])
    assert not eng.stt.calls                      # passing self-test: Live's own transcript is trusted
    assert any(isinstance(e, Error) and e.message_ckb == strings.LIVE_DEGRADED for e in events)
    await eng.stop_listening()
    await eng.start_listening()                   # next window: Live gets another chance
    assert eng.engine_name == "live"


async def test_live_connect_failure_uses_cascade(voice):
    from voice_helpers import ApiError
    client = FakeLiveClient([ApiError(403, "API key not valid")])
    app, eng, mics, *_ , events = await voice(gemini=True, client=client)
    await eng.start_listening()
    assert await settle(lambda: eng.engine_name == "cascade")
    assert any(isinstance(e, Error) and e.message_ckb == strings.LIVE_FAILED for e in events)


async def test_echo_guard_silences_quiet_frames_while_speaking(voice):
    app, eng, mics, speaker, client, events = await voice(gemini=True)
    app.config.set("voice.echo_guard", "on")
    await eng.start_listening()
    assert await settle(lambda: eng.live is not None and eng.live.ready)
    speaker.playing = True
    quiet_but_not_silent = (b"\x20\x00" * 480)
    mics[0].push(quiet_but_not_silent)            # SAM's own voice leaking into the mic
    mics[0].push(tone_frame(20000))               # the user talking over SAM
    session = client.sessions[0]
    assert await settle(lambda: len(session.audio) >= 2)
    assert session.audio[0] == bytes(960) and session.audio[1] == tone_frame(20000)


async def test_hotkey_toggles_listening_and_failure_is_reported(voice):
    FakeHotkey.instances.clear()
    app, eng, mics, *_ = await voice()
    assert await settle(lambda: FakeHotkey.instances and FakeHotkey.instances[-1].registered)  # registered off-path
    hotkey = FakeHotkey.instances[-1]
    assert hotkey.keys == "ctrl+alt+space"
    hotkey.press()
    assert await settle(lambda: eng.listening)
    hotkey.press()
    assert await settle(lambda: not eng.listening)
    app2, eng2, *_, events = await voice(hotkey_ok=False)
    assert await settle(lambda: [e for e in events if isinstance(e, Error) and e.where == "voice.hotkey"])
    errors = [e for e in events if isinstance(e, Error) and e.where == "voice.hotkey"]
    assert errors and "ctrl+alt+space" in errors[0].message_ckb
    assert eng2.status()["hotkey"]["registered"] is False


async def test_hotkey_fallback_when_the_chord_is_taken_and_stop_releases_it(voice):
    """Measured 2026-09-24: another program owns Ctrl+Alt+Space on this PC.
    The first free fallback is used and stored; stop() unregisters it."""
    app, eng, *_, events = await voice()
    await eng.stop()                                   # drop the fixture's default registration
    eng._hotkey_factory = lambda keys, cb: FakeHotkey(keys, cb, ok=keys != "ctrl+alt+space")  # noqa: SLF001
    FakeHotkey.instances.clear()
    await eng.start()
    assert await settle(lambda: eng.status()["hotkey"]["registered"])
    assert eng.status()["hotkey"]["keys"] == "win+alt+space"
    assert app.config.get("voice.hotkey") == "win+alt+space"
    assert any(isinstance(e, Error) and e.where == "voice.hotkey" and "win+alt+space" in e.message_ckb
               for e in events)
    held = FakeHotkey.instances[-1]
    await eng.stop()
    assert held.registered is False


async def test_start_returns_before_a_slow_hotkey_registration(voice):
    """start() must not wait for RegisterHotKey (island visible < 3 s)."""
    import threading
    release = threading.Event()

    class SlowHotkey(FakeHotkey):
        def start(self):
            release.wait(2.0)
            return super().start()

    app, eng, *_ = await voice()
    await eng.stop()
    eng._hotkey_factory = lambda keys, cb: SlowHotkey(keys, cb)  # noqa: SLF001
    loop = asyncio.get_running_loop()
    began = loop.time()
    await eng.start()
    assert loop.time() - began < 0.5
    assert not eng.status()["hotkey"]["registered"]
    release.set()
    assert await settle(lambda: eng.status()["hotkey"]["registered"])


async def test_stop_all_flushes_speech(voice):
    app, eng, mics, speaker, *_ = await voice()
    await eng.speak("ڕستەیەکی درێژ.", source="system")
    await app.stop_all()
    assert speaker.flushes >= 1


async def test_test_key_for_kurdishtts_keys(make_app):
    import httpx

    from sam.voice.stt import KurdishTtsStt, SttRouter
    from sam.voice.tts import KurdishTts, TtsRouter
    app = make_app(env_text="KURDISHTTS_STT_API_KEY=" + "ab" * 20 + "\nKURDISHTTS_TTS_API_KEY=" + "cd" * 20 + "\n")
    tts_ok = KurdishTts(app, transport=httpx.MockTransport(
        lambda r: httpx.Response(200, headers={"content-type": "audio/pcm"}, content=b"\x00\x00" * 50)))
    stt_bad = KurdishTtsStt(app, transport=httpx.MockTransport(lambda r: httpx.Response(401, text="Invalid API key")))
    eng = VoiceEngine(app, speaker=FakeSpeaker(), stt=SttRouter(app, {"kurdishtts": stt_bad}),
                      tts=TtsRouter(app, {"kurdishtts": tts_ok}))
    ok = await eng.test_key("kurdishtts_tts_api_key")
    assert ok["ok"] is True and ok["status"] == "connected" and ok["latency_ms"] >= 0
    bad = await eng.test_key("kurdishtts_stt_api_key")
    assert bad["ok"] is False and bad["status"] == "auth_failed"
    assert (await eng.test_key("n8n_api_key"))["ok"] is False


async def test_status_has_no_key_material(voice):
    app, eng, *_ = await voice(gemini=True)
    status = eng.status()
    assert {"engine", "state", "live_degraded", "devices", "last_ttfa_ms", "hotkey", "stt", "tts"} <= set(status)
    assert FAKE_GEMINI not in repr(status)


async def test_live_noise_without_words_keeps_live(voice):
    """A sound Live heard no words in (cough, door, TV) is not a stall: Live
    stays and no STT is spent (review 2026-09-24)."""
    app, eng, mics, speaker, client, events = await voice(gemini=True)
    app.config.set("voice.watchdog_s", 0.2)
    await eng.start_listening()
    assert await settle(lambda: eng.live is not None and eng.live.ready)
    speak_utterance(mics[0])                      # no transcription, no answer
    await asyncio.sleep(0.6)
    assert eng.engine_name == "live" and not eng.live_degraded and not eng.stt.calls


async def test_an_untrusted_live_transcript_goes_through_stt(voice):
    """Without a passing self-test the cascade transcribes the audio itself."""
    app, eng, mics, speaker, client, events = await voice(gemini=True)
    app.config.set("voice.watchdog_s", 0.2)
    app.config.set("voice.engine", "live")        # user forced Live, never tested
    app.config.set("voice.selftest", None)
    await eng.start_listening()
    assert await settle(lambda: eng.live is not None and eng.live.ready)
    speak_utterance(mics[0])
    client.sessions[0].push(msg({"server_content": {"input_transcription": {"text": "Kashmir Chand"}}}))
    assert await settle(lambda: eng.engine_name == "cascade", timeout=3.0)
    assert await settle(lambda: eng.stt.calls, timeout=3.0)


async def test_a_short_sound_over_sams_voice_only_ducks_it(voice):
    """Barge-in needs ~400 ms of voiced audio while SAM speaks; a quick sound
    ducks the speaker and is dropped without STT."""
    app, eng, mics, speaker, *_ = await voice()
    await eng.start_listening()
    speaker.playing = True
    cut = []
    eng.cascade.barge_in = lambda: cut.append(1) or True
    mics[0].push(tone_frame(), 6)                 # ~180 ms of voice
    assert await settle(lambda: getattr(speaker, "gain", 1.0) < 1.0)
    mics[0].push(quiet_frame(), 25)
    assert await settle(lambda: getattr(speaker, "gain", 1.0) == 1.0)
    await asyncio.sleep(0.1)
    assert not cut and not eng.stt.calls
    speak_utterance(mics[0], speech_frames=30)    # ~900 ms: a real barge-in
    assert await settle(lambda: cut)
