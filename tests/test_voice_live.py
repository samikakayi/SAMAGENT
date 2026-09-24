"""LiveVoice against a fake Live session that replays real LiveServerMessage
objects: config, transcripts, barge-in, tool round trips (BLOCKING and
NON_BLOCKING), resumption on GoAway and on a dropped socket, the watchdog,
say(), and confirmations that survive a server-side cancellation."""

from __future__ import annotations

import asyncio
import time

import pytest
from conftest import FAKE_GEMINI
from voice_helpers import (ApiError, FakeLiveClient, FakeLiveSession, FakeSpeaker, audio_msg, idle_msg, msg,
                           settle)

from sam.brain.tools import ok, tool
from sam.events import Caption, ConfirmRequest, ToolStarted, Transcript
from sam.voice.hooks import RecordingHooks
from sam.voice.live import LiveVoice

CLOSED: list[str] = []


@tool("get_price", description="Current price of a symbol.", blocking=True,
      params={"type": "object", "properties": {"symbol": {"type": "string"}}})
async def get_price(ctx, symbol: str = "XAUUSD"):
    return ok("XAUUSD 2650.5", price=2650.5, symbol=symbol)


@tool("analyze_market", description="Slow analysis.", blocking=False)
async def analyze_market(ctx):
    await asyncio.sleep(0.02)
    return ok("WAIT: no setup")


@tool("close_window", description="Close a window.", risk="confirm", confirm_text_ckb="{target} دابخەم؟",
      params={"type": "object", "properties": {"target": {"type": "string"}}, "required": ["target"]})
async def close_window(ctx, target: str):
    CLOSED.append(target)
    return ok(f"closed {target}")


@pytest.fixture
async def live_env(make_app):
    app = make_app(env_text=f"GEMINI_API_KEY={FAKE_GEMINI}\n")
    app.bus.bind_loop(asyncio.get_running_loop())
    for fn in (get_price, analyze_market, close_window):
        app.tools.add(fn, owner="test")
    events = []
    app.bus.subscribe(None, events.append)
    client = FakeLiveClient()
    speaker = FakeSpeaker()
    hooks = RecordingHooks()
    live = LiveVoice(app, speaker, hooks, client_factory=lambda key: client, reconnect_delays=(0.01, 0.01),
                     ready_timeout_s=2.0, say_wait_s=0.3)
    CLOSED.clear()
    yield app, live, client, speaker, hooks, events
    await live.close()


def of(events, kind):
    return [e for e in events if isinstance(e, kind)]


async def test_connect_config_matches_design(live_env):
    app, live, client, *_ = live_env
    assert await live.open()
    model, config = client.connects[0]
    assert model == "gemini-3.8-live"
    assert [str(getattr(m, "value", m)) for m in config.response_modalities] == ["AUDIO"]
    assert config.input_audio_transcription is not None and config.output_audio_transcription is not None
    vad = config.realtime_input_config.automatic_activity_detection
    assert vad.silence_duration_ms == 600 and vad.prefix_padding_ms == 200
    assert config.context_window_compression.sliding_window is not None
    assert config.session_resumption is not None and config.session_resumption.handle is None
    assert config.speech_config.voice_config.prebuilt_voice_config.voice_name == "Kore"
    assert "RESPOND IN CENTRAL KURDISH (SORANI), ARABIC SCRIPT" in str(config.system_instruction)
    behaviors = {d.name: str(getattr(d.behavior, "value", d.behavior)) for d in config.tools[0].function_declarations}
    assert behaviors["get_price"] == "BLOCKING" and behaviors["analyze_market"] == "NON_BLOCKING"
    assert app.db.query("SELECT stage FROM timings WHERE stage='live_connect'")


async def test_pending_audio_is_sent_after_connect_in_order(live_env):
    _, live, client, *_ = live_env
    await live.send_audio(b"\x01\x00" * 480)
    await live.send_audio(b"\x02\x00" * 480)
    assert await live.open()
    await live.send_audio(b"\x03\x00" * 480)
    session = client.sessions[0]
    assert [a[:2] for a in session.audio] == [b"\x01\x00", b"\x02\x00", b"\x03\x00"]
    assert session.realtime[0]["audio"].mime_type == "audio/pcm;rate=16000"


async def test_transcripts_audio_and_ttfa(live_env):
    app, live, client, speaker, hooks, events = live_env
    assert await live.open()
    session = client.sessions[0]
    live.note_end_of_speech(b"\x00\x00" * 1600, time.perf_counter())
    session.push(msg({"server_content": {"input_transcription": {"text": "تکایە "}}}),
                 msg({"server_content": {"input_transcription": {"text": "ترەیدینگ ڤیو بکەرەوە", "finished": True}}}),
                 audio_msg(b"\x05\x00" * 480),
                 msg({"server_content": {"output_transcription": {"text": "باشە، ئێستا دەیکەمەوە."}}}),
                 audio_msg(b"\x06\x00" * 480), idle_msg())
    assert await settle(lambda: len(of(events, Transcript)) == 2)
    user, assistant = of(events, Transcript)
    assert (user.role, user.text, user.source) == ("user", "تکایە ترەیدینگ ڤیو بکەرەوە", "live")
    assert (assistant.role, assistant.text, assistant.source) == ("assistant", "باشە، ئێستا دەیکەمەوە.", "live")
    assert speaker.audio == b"\x05\x00" * 480 + b"\x06\x00" * 480
    assert ("speaking", "") in hooks.states and await settle(lambda: hooks.last_state == "listening")
    assert live.last_ttfa_ms is not None and live.last_ttfa_ms >= 0
    stages = {r["stage"] for r in app.db.query("SELECT stage FROM timings WHERE kind='live'")}
    assert {"end_of_speech", "first_audio", "total"} <= stages
    assert any(isinstance(e, Caption) and e.role == "user" and not e.final for e in events)


async def test_barge_in_flushes_playback_immediately(live_env):
    _, live, client, speaker, hooks, events = live_env
    assert await live.open()
    session = client.sessions[0]
    session.push(audio_msg(), msg({"server_content": {"output_transcription": {"text": "نرخی زێڕ ئێستا"}}}),
                 audio_msg())
    assert await settle(lambda: len(speaker.chunks) == 2)
    session.push(msg({"server_content": {"interrupted": True}}))
    assert await settle(lambda: speaker.flushes == 1)
    assert speaker.dropped == 960 * 2
    assert hooks.last_state == "listening"
    said = [e for e in of(events, Transcript) if e.role == "assistant"]
    assert said and said[-1].text.endswith("…")


async def test_blocking_tool_round_trip(live_env):
    app, live, client, *_ , events = live_env
    assert await live.open()
    session = client.sessions[0]
    session.push(msg({"tool_call": {"function_calls": [{"id": "c1", "name": "get_price",
                                                        "args": {"symbol": "XAUUSD"}}]}}))
    assert await settle(lambda: session.tool_responses)
    [response] = session.tool_responses[0]
    assert (response.id, response.name) == ("c1", "get_price")
    assert response.response["ok"] is True and response.response["data"]["price"] == 2650.5
    assert response.scheduling is None
    started = of(events, ToolStarted)
    assert started and started[0].source == "live" and started[0].call_id == "c1"


async def test_non_blocking_tool_answers_when_idle_and_separately(live_env):
    _, live, client, *_ = live_env
    assert await live.open()
    session = client.sessions[0]
    session.push(msg({"tool_call": {"function_calls": [
        {"id": "a1", "name": "analyze_market", "args": {}},
        {"id": "p1", "name": "get_price", "args": {}}]}}))
    assert await settle(lambda: len(session.tool_responses) == 2)
    first, second = session.tool_responses
    assert [r.id for r in first] == ["p1"]          # the quick BLOCKING one is not held back
    assert [r.id for r in second] == ["a1"]
    assert str(second[0].scheduling.value) == "WHEN_IDLE"


async def test_fallback_model_gets_no_behavior_and_sequential_calls(live_env):
    app, live, client, *_ = live_env
    client.plan = [ApiError(404, "models/gemini-3.8-live is not found"), FakeLiveSession()]
    assert await live.open()
    assert [m for m, _ in client.connects] == ["gemini-3.8-live", "gemini-3.1-flash-live-preview"]
    declarations = client.connects[1][1].tools[0].function_declarations
    assert all(d.behavior is None for d in declarations)
    session = client.sessions[0]
    session.push(msg({"tool_call": {"function_calls": [{"id": "a1", "name": "analyze_market", "args": {}}]}}))
    assert await settle(lambda: session.tool_responses)
    assert session.tool_responses[0][0].scheduling is None   # 3.1: no async scheduling
    assert app.db.query("SELECT * FROM activity WHERE name='live_setup'")


async def test_go_away_reconnects_with_resumption_handle(live_env):
    _, live, client, *_ = live_env
    assert await live.open()
    first = client.sessions[0]
    first.push(msg({"session_resumption_update": {"new_handle": "handle-1", "resumable": True}}),
               msg({"go_away": {"time_left": "5s"}}))
    assert await settle(lambda: len(client.sessions) == 2 and live.ready)
    assert first.closed
    assert client.connects[1][1].session_resumption.handle == "handle-1"
    assert live.connects == 2 and not live.degraded


async def test_go_away_mid_turn_waits_for_the_turn_to_end(live_env):
    _, live, client, speaker, *_ = live_env
    assert await live.open()
    first = client.sessions[0]
    first.push(msg({"session_resumption_update": {"new_handle": "handle-2", "resumable": True}}),
               audio_msg(), msg({"go_away": {"time_left": "30s"}}))
    await settle(lambda: speaker.chunks)
    await asyncio.sleep(0.05)
    assert len(client.sessions) == 1               # the answer is still being spoken: no cut
    first.push(audio_msg(), idle_msg())
    assert await settle(lambda: len(client.sessions) == 2 and live.ready)
    assert len(speaker.chunks) == 2 and client.connects[1][1].session_resumption.handle == "handle-2"


async def test_dropped_socket_reconnects_with_handle(live_env):
    _, live, client, *_ = live_env
    assert await live.open()
    first = client.sessions[0]
    first.push(msg({"session_resumption_update": {"new_handle": "h-9", "resumable": True}}))
    await settle(lambda: live._handle == "h-9")  # noqa: SLF001
    first.drop()
    assert await settle(lambda: len(client.sessions) == 2 and live.ready)
    assert client.connects[1][1].session_resumption.handle == "h-9"
    assert not live.degraded


async def test_non_resumable_update_keeps_previous_handle(live_env):
    _, live, client, *_ = live_env
    assert await live.open()
    session = client.sessions[0]
    session.push(msg({"session_resumption_update": {"new_handle": "h-1", "resumable": True}}),
                 msg({"session_resumption_update": {"new_handle": "", "resumable": False}}))
    await settle()
    assert live._handle == "h-1"  # noqa: SLF001


async def test_watchdog_hands_the_turn_to_the_cascade(live_env):
    app, live, client, speaker, hooks, events = live_env
    app.config.set("voice.watchdog_s", 0.1)
    app.config.set("voice.selftest", {"ok": True, "cer": 0.1})   # Live's text is trusted only after a pass
    assert await live.open()
    session = client.sessions[0]
    session.push(msg({"server_content": {"input_transcription": {"text": "نرخی زێڕ چەندە؟"}}}))
    await settle()
    live.note_end_of_speech(b"\x07\x00" * 1600, time.perf_counter())
    assert await settle(lambda: hooks.stalls, timeout=1.0)
    pcm, text, _eos, published = hooks.stalls[0]
    # What Live heard is published once (Transcript) and handed over as text:
    # the cascade needs no second STT call and stores no duplicate turn.
    assert pcm == b"\x07\x00" * 1600 and text == "نرخی زێڕ چەندە؟" and published
    assert [e.text for e in of(events, Transcript) if e.role == "user"] == ["نرخی زێڕ چەندە؟"]
    assert live.degraded and "0.1 s" in live.degraded_reason
    assert app.db.query("SELECT * FROM timings WHERE stage='live_watchdog'")


async def test_watchdog_keeps_a_transcript_finished_before_local_end_of_speech(live_env):
    """The server may finalize the transcript before the local VAD's 600 ms
    ran out: that text must still reach the cascade (and not be re-published)."""
    app, live, client, speaker, hooks, events = live_env
    app.config.set("voice.watchdog_s", 0.1)
    app.config.set("voice.selftest", {"ok": True, "cer": 0.1})
    assert await live.open()
    live.note_speech_start()
    client.sessions[0].push(msg({"server_content": {"input_transcription": {"text": "کرۆم بکەرەوە",
                                                                           "finished": True}}}))
    assert await settle(lambda: of(events, Transcript))
    live.note_end_of_speech(b"\x07\x00" * 800, time.perf_counter())
    assert await settle(lambda: hooks.stalls, timeout=1.0)
    _pcm, text, _eos, published = hooks.stalls[0]
    assert text == "کرۆم بکەرەوە" and published
    assert len([e for e in of(events, Transcript) if e.role == "user"]) == 1


async def test_a_sound_without_any_transcript_is_not_a_stall(live_env):
    """Review 2026-09-24: any unanswered sound (cough, door, TV) dropped Live for
    the whole window and spent STT on the noise. No words heard -> no stall."""
    app, live, client, speaker, hooks, events = live_env
    app.config.set("voice.watchdog_s", 0.1)
    assert await live.open()
    live.note_speech_start()
    live.note_end_of_speech(b"\x08\x00" * 800, time.perf_counter())
    await asyncio.sleep(0.3)
    assert not hooks.stalls and not live.degraded
    # Hybrid VAD: the local end of speech was sent as audio_stream_end.
    assert any(r.get("audio_stream_end") for r in client.sessions[0].realtime)


async def test_an_untrusted_transcript_is_not_handed_over(live_env):
    """Without a passing self-test, the cascade must run STT itself."""
    app, live, client, speaker, hooks, events = live_env
    app.config.set("voice.watchdog_s", 0.1)
    assert await live.open()
    client.sessions[0].push(msg({"server_content": {"input_transcription": {"text": "Kashmir Chand"}}}))
    await settle()
    live.note_end_of_speech(b"\x07\x00" * 800, time.perf_counter())
    assert await settle(lambda: hooks.stalls, timeout=1.0)
    _pcm, text, _eos, published = hooks.stalls[0]
    assert text == "" and published


async def test_spoken_yes_for_a_pending_confirmation_is_not_a_stall(live_env):
    """A "بەڵێ" that answers a worker's confirmation owes no reply: the
    watchdog must neither degrade Live nor send "بەڵێ" to the brain."""
    app, live, client, speaker, hooks, events = live_env
    app.config.set("voice.watchdog_s", 0.1)
    assert await live.open()
    question = asyncio.ensure_future(app.confirm.confirm("فایلەکە بسڕمەوە؟", tool_name="files"))
    assert await settle(lambda: app.confirm.has_pending)
    live.note_speech_start()
    client.sessions[0].push(msg({"server_content": {"input_transcription": {"text": "بەڵێ", "finished": True}}}))
    live.note_end_of_speech(b"\x00\x00" * 800, time.perf_counter())
    assert await asyncio.wait_for(question, 2) is True
    await asyncio.sleep(0.3)
    assert not hooks.stalls and not live.degraded


async def test_interim_transcription_is_only_a_caption(live_env):
    app, live, client, speaker, hooks, events = live_env
    assert await live.open()
    client.sessions[0].push(msg({"server_content": {"interim_input_transcription": {"text": "نرخی زێ"}}}))
    assert await settle(lambda: any(isinstance(e, Caption) and e.text == "نرخی زێ" for e in events))
    assert not of(events, Transcript) and not any(isinstance(e, Caption) and e.final for e in events)


async def test_watchdog_is_disarmed_by_audio(live_env):
    app, live, client, speaker, hooks, _ = live_env
    app.config.set("voice.watchdog_s", 0.1)
    assert await live.open()
    live.note_end_of_speech(b"\x00\x00" * 160, time.perf_counter())
    client.sessions[0].push(audio_msg())
    await asyncio.sleep(0.25)
    assert not hooks.stalls and not live.degraded


async def test_auth_failure_reports_once_without_retry_loop(live_env):
    _, live, client, _, hooks, _ = live_env
    client.plan = [ApiError(403, "API key not valid")]
    assert not await live.open()
    assert len(client.connects) == 1 and live.degraded and hooks.failures and "auth" in hooks.failures[0]


async def test_say_sends_client_content_when_idle_and_waits_when_busy(live_env):
    _, live, client, _, hooks, _ = live_env
    assert await live.open()
    session = client.sessions[0]
    assert await live.say("زێڕ گەیشتە ٢٧٠٠", source="alert")
    turns, complete = session.client_content[0]
    assert complete is True and "زێڕ گەیشتە ٢٧٠٠" in turns.parts[0].text and turns.parts[0].text.startswith("[SAM]")
    session.push(idle_msg())
    await settle()
    session.push(audio_msg())                     # the model is talking now
    await settle(lambda: live._model_active)  # noqa: SLF001
    assert await live.say("ئاگاداری دووەم", source="alert")
    assert len(session.client_content) == 1       # queued, not interrupting
    session.push(idle_msg())
    assert await settle(lambda: len(session.client_content) == 2)
    # a confirmation question during a running turn goes to TTS instead
    session.push(audio_msg())
    await settle(lambda: live._model_active)  # noqa: SLF001
    await live.say("دابخەم؟", source="confirm")
    assert hooks.fallback_spoken == [("دابخەم؟", "confirm")]


async def test_say_times_out_to_tts_when_the_model_never_goes_idle(live_env):
    _, live, client, _, hooks, _ = live_env
    assert await live.open()
    client.sessions[0].push(audio_msg())
    await settle(lambda: live._model_active)  # noqa: SLF001
    await live.say("ئاگادارکردنەوە", source="alert")
    assert await settle(lambda: hooks.fallback_spoken, timeout=2.0)


async def test_confirmation_survives_server_cancellation_and_reports_result(live_env):
    app, live, client, _, _, events = live_env
    assert await live.open()
    session = client.sessions[0]
    session.push(msg({"tool_call": {"function_calls": [{"id": "w1", "name": "close_window",
                                                        "args": {"target": "Notepad"}}]}}))
    assert await settle(lambda: of(events, ConfirmRequest))
    assert of(events, ConfirmRequest)[0].question_ckb == "Notepad دابخەم؟"
    # the user's spoken answer is a new turn: the server cancels the pending call
    session.push(msg({"tool_call_cancellation": {"ids": ["w1"]}}),
                 msg({"server_content": {"input_transcription": {"text": "بەڵێ", "finished": True}}}))
    assert await settle(lambda: CLOSED == ["Notepad"])
    await settle()
    assert session.client_content == []           # the model is replying to "بەڵێ": do not interrupt it
    session.push(audio_msg(), idle_msg())
    assert await settle(lambda: session.client_content)
    note = session.client_content[-1][0].parts[0].text
    assert note.startswith("[SAM]") and "close_window" in note and "closed Notepad" in note
    assert session.tool_responses == []           # the cancelled id is not answered


async def test_cancellation_before_start_cancels_the_call(live_env):
    app, live, client, *_ = live_env
    ran = []

    @tool("slow_tool", description="slow", blocking=True)
    async def slow_tool(ctx):
        ran.append(1)
        return ok("done")

    gate = asyncio.Event()
    original = app.tools.dispatch

    async def delayed_dispatch(*a, **kw):
        await gate.wait()      # still validating when the cancellation arrives
        return await original(*a, **kw)

    app.tools.add(slow_tool, owner="test")
    app.tools.dispatch = delayed_dispatch
    assert await live.open()
    session = client.sessions[0]
    session.push(msg({"tool_call": {"function_calls": [{"id": "s1", "name": "slow_tool", "args": {}}]}}),
                 msg({"tool_call_cancellation": {"ids": ["s1"]}}))
    await settle(lambda: not live._calls)  # noqa: SLF001
    gate.set()
    await settle()
    assert ran == [] and session.tool_responses == []


async def test_stop_output_suppresses_rest_of_turn(live_env):
    _, live, client, speaker, _, _ = live_env
    assert await live.open()
    session = client.sessions[0]
    session.push(audio_msg())
    await settle(lambda: speaker.chunks)
    live.stop_output()
    session.push(audio_msg(), audio_msg())
    await settle()
    assert len(speaker.chunks) == 1 and speaker.flushes == 1
    session.push(idle_msg(), audio_msg())             # next turn is heard again
    assert await settle(lambda: len(speaker.chunks) == 2)


async def test_undeclared_arguments_are_dropped_before_dispatch(live_env):
    """Gemini once called the parameterless tv_open with {"reason": ...}
    (brain builder, 2026-09-24); the handler has no **kwargs."""
    app, live, client, *_ = live_env

    @tool("tv_open", description="Open TradingView.", blocking=True)
    async def tv_open(ctx):
        return ok("TradingView is open")

    app.tools.add(tv_open, owner="test")
    assert await live.open()
    session = client.sessions[0]
    session.push(msg({"tool_call": {"function_calls": [
        {"id": "t1", "name": "tv_open", "args": {"reason": "user asked"}},
        {"id": "p1", "name": "get_price", "args": {"symbol": "XAUUSD", "verbose": True}}]}}))
    assert await settle(lambda: session.tool_responses)
    by_id = {r.id: r.response for r in session.tool_responses[0]}
    assert by_id["t1"]["ok"] is True and by_id["p1"]["ok"] is True
    assert by_id["p1"]["data"]["symbol"] == "XAUUSD"


async def test_setup_message_on_the_wire(live_env):
    """The exact JSON the SDK would send for our config (google-genai 2.25.0
    connect(): _t_live_connect_config -> _LiveConnectParameters_to_mldev),
    built offline: tools keep their JSON schema and BLOCKING/NON_BLOCKING on
    3.8; the 3.1 fallback gets no behavior. (Acceptance by the real service is
    unproven: no Gemini key on this PC yet.)"""
    from google import genai
    from google.genai import _common, _live_converters, types
    from google.genai.live import _t_live_connect_config

    from sam.voice.live_config import build_live_config
    app, *_ = live_env
    client = genai.Client(api_key=FAKE_GEMINI)       # constructing a client makes no request
    setups = {}
    for model in ("gemini-3.8-live", "gemini-3.1-flash-live-preview"):
        config = await _t_live_connect_config(client._api_client, build_live_config(app, model, handle="h-1"))
        request = _common.convert_to_dict(_live_converters._LiveConnectParameters_to_mldev(
            api_client=client._api_client,
            from_object=types.LiveConnectParameters(model=f"models/{model}", config=config).model_dump(
                exclude_none=True)))
        setups[model] = request["setup"]
    setup = setups["gemini-3.8-live"]
    assert setup["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert setup["generationConfig"]["speechConfig"]["voice_config"]["prebuilt_voice_config"]["voice_name"] == "Kore"
    assert setup["inputAudioTranscription"] == {} and setup["outputAudioTranscription"] == {}
    assert setup["realtimeInputConfig"]["automatic_activity_detection"]["silence_duration_ms"] == 600
    assert "sliding_window" in setup["contextWindowCompression"]
    assert setup["sessionResumption"] == {"handle": "h-1"}
    assert "thinkingConfig" not in setup["generationConfig"]        # 3.8 Live: thinking level must be omitted
    decls = {d["name"]: d for d in setup["tools"][0]["functionDeclarations"]}
    assert decls["get_price"]["parameters_json_schema"]["properties"]["symbol"] == {"type": "string"}
    assert decls["get_price"]["behavior"] == "BLOCKING" and decls["analyze_market"]["behavior"] == "NON_BLOCKING"
    fallback = {d["name"]: d for d in setups["gemini-3.1-flash-live-preview"]["tools"][0]["functionDeclarations"]}
    assert all("behavior" not in d for d in fallback.values())
    assert FAKE_GEMINI not in repr(setups)


async def test_usage_is_counted_per_connection(live_env):
    app, live, client, *_ = live_env
    assert await live.open()
    client.sessions[0].push(msg({"usage_metadata": {"prompt_token_count": 10, "response_token_count": 4}}))
    await settle()
    await live.close()
    row = app.db.query_one("SELECT * FROM usage_counters WHERE provider='gemini' AND kind='live'")
    assert row and row["model"] == "gemini-3.8-live" and row["tokens_in"] == 10 and row["tokens_out"] == 4
