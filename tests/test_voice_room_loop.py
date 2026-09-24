"""Closed-loop living-room scenes (voice_room_sim.py): the TV must not feed
itself through STT -> model -> answer -> follow-up window (adversarial review
2026-09-24: 61 STT and 61 model calls in 300 s from 3 clicks, before the fix).
Virtual time: a 150 s scene runs in about a second."""

from __future__ import annotations

import asyncio

import pytest
from voice_helpers import FakeMic
from voice_room_sim import CLOCK, USER_TEXT, Scene, VSpeaker, VStt, VTts, run_scene

import sam.voice.cascade as cascade_mod
import sam.voice.engine as engine_mod
import sam.voice.frames as frames_mod
import sam.voice.listening as listening_mod
from sam.voice.engine import VoiceEngine


class EnergyClassifier:
    def __init__(self, **kwargs):
        pass

    def is_speech(self, frame, rms):
        return rms > 0.001


@pytest.fixture
async def room(make_app, monkeypatch):
    for module in (listening_mod, frames_mod, engine_mod, cascade_mod):
        monkeypatch.setattr(module, "time", CLOCK)
    monkeypatch.setattr(engine_mod, "FrameClassifier", EnergyClassifier)
    engines = []

    async def build(scene: Scene, *, user_text: str = USER_TEXT, settings: dict | None = None):
        app = make_app()
        app.bus.bind_loop(asyncio.get_running_loop())
        app.config.set("voice.selftest_auto", False)
        for key, value in (settings or {}).items():
            app.config.set(key, value)
        hotkey = type("H", (), {"start": lambda s: True, "stop": lambda s: None, "registered": True})
        eng = VoiceEngine(app, mic_factory=FakeMic, speaker=VSpeaker(), stt=VStt(user_text), tts=VTts(),
                          llm_stream=scene.llm(), hotkey_factory=lambda keys, cb: hotkey())
        app.voice = eng
        await eng.start()
        engines.append(eng)
        return app, eng

    yield build
    for eng in engines:
        await eng.stop()


async def test_tv_room_push_to_talk_with_nothing_known_costs_one_stt_per_click(room):
    scene = Scene(seconds=150.0, clicks=[10.0, 60.0, 110.0])
    app, eng = await room(scene)
    result = await run_scene(eng, scene)
    assert result["stt"] == ["user", "user", "user"]
    assert result["llm"] == [USER_TEXT] * 3
    assert result["mic_open_s"] < 40                                        # ~8-10 s per click, not 150
    assert app.config.get("voice.gate_user_level_db") is None or \
        app.config.get("voice.gate_user_level_db") > -30                     # never learned as the TV


async def test_tv_room_with_the_users_level_known_keeps_the_tv_out_of_follow_ups(room):
    scene = Scene(seconds=150.0, clicks=[10.0, 60.0, 110.0])
    app, eng = await room(scene, settings={"voice.gate_user_level_db": -20.0})
    result = await run_scene(eng, scene)
    assert result["stt"] == ["user", "user", "user"] and len(result["llm"]) == 3


async def test_tv_room_always_listening_reaches_the_model_only_with_the_name(room):
    scene = Scene(seconds=150.0, clicks=[10.0, 60.0, 110.0])
    app, eng = await room(scene, user_text="سام، " + USER_TEXT, settings={"voice.always_listening": True})
    result = await run_scene(eng, scene)
    assert result["llm"] == ["سام، " + USER_TEXT] * 3                        # the TV never reaches a model
    assert result["stt"].count("user") == 3 and result["stt"].count("tv") <= 4  # rejected talker stays below
