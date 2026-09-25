from __future__ import annotations

import asyncio

from brain_helpers import CALLS, Reply, brain_app, collect, tool_names, tool_results, user_text

from sam.brain.conversation import ACKS_DO, ACKS_LOOK, SORANI_NO_MODEL
from sam.brain.llm import LLMError
from sam.events import Caption, ConfirmRequest, Error, SpeakRequest, ToolFinished, Transcript, VoiceState


def sorani_router(req):
    """A fake model that 'understands' Sorani: it picks open_app for
    'ترەیدینگ ڤیو بکەرەوە' only if that tool was offered (no keyword gating
    on SAM's side), then words the result."""
    if tool_results(req):
        return Reply(text="کرایەوە، چارتەکە ئامادەیە.")
    if "ترەیدینگ ڤیو" in user_text(req) and "open_app" in tool_names(req):
        return Reply(text="باشە، ئێستا.", calls=[("open_app", {"name": "ترەیدینگ ڤیو"})])
    return Reply(text="سڵاو، باشم سوپاس. تۆ چۆنی؟")


async def drain(stream) -> list[str]:
    return [piece async for piece in stream]


async def test_sorani_command_runs_the_tool_and_answers_in_chunks(make_app):
    app, backend = brain_app(make_app, default=sorani_router)
    finished = collect(app.bus, ToolFinished)
    chunks = await drain(app.conversation.respond_stream("ترەیدینگ ڤیو بکەرەوە", source="cascade"))
    # The model's own words next to its tool call ("باشە، ئێستا.") are not spoken:
    # the cached acknowledgement is (review 2026-09-24: the user heard two answers).
    assert chunks[0] in ACKS_DO and chunks[1:] == ["کرایەوە، چارتەکە ئامادەیە."]
    assert CALLS == [("open_app", {"name": "ترەیدینگ ڤیو"})]
    assert finished and finished[0].name == "open_app" and finished[0].ok and finished[0].source == "cascade"
    first = backend.requests[0]
    # The core tier (compact) + more_tools on every turn: no keyword gating.
    from sam.brain.responder import CORE_TOOLS
    offered = set(tool_names(first))
    assert offered == ({n for n in CORE_TOOLS if n in app.tools.names()} | {"more_tools"})
    assert first.messages[0]["role"] == "system" and "RESPOND IN CENTRAL KURDISH" in first.messages[0]["content"]
    # Second request carries the assistant tool call and the tool result.
    second = backend.requests[1]
    assert any(m.get("tool_calls") for m in second.messages if m["role"] == "assistant")
    assert tool_results(second)[0]["ok"] is True


async def test_turns_are_stored_once_in_one_continuous_conversation(make_app):
    app, backend = brain_app(make_app, default=sorani_router)
    # The voice engine publishes the user transcript itself, then asks us.
    app.bus.publish(Transcript(role="user", text="سڵاو، چۆنی؟", source="cascade"))
    await drain(app.conversation.respond_stream("سڵاو، چۆنی؟", source="cascade"))
    app.bus.publish(Transcript(role="user", text="ترەیدینگ ڤیو بکەرەوە", source="cascade"))
    await drain(app.conversation.respond_stream("ترەیدینگ ڤیو بکەرەوە", source="cascade"))
    # ...and publishes the assistant text again when playback ends.
    stored = app.memory.recent_turns(app.conversation.conversation_id, limit=1, roles=("assistant",))[0]["text"]
    app.bus.publish(Transcript(role="assistant", text=stored, source="cascade"))
    cid = app.conversation.conversation_id
    assert app.db.scalar("SELECT COUNT(*) FROM conversations") == 1
    roles = [(t["role"], t["text"][:12]) for t in app.memory.recent_turns(cid, limit=20)]
    assert [r for r, _ in roles] == ["user", "assistant", "user", "tool", "assistant"]
    tool_turn = app.memory.recent_turns(cid, limit=20, roles=("tool",))[0]
    assert tool_turn["meta"]["name"] == "open_app" and tool_turn["meta"]["args"] == {"name": "ترەیدینگ ڤیو"}
    # The second request saw the first exchange as history (continuity).
    history = [m["content"] for m in backend.requests[1].messages if m["role"] in ("user", "assistant")]
    assert history[0] == "سڵاو، چۆنی؟" and history[-1] == "ترەیدینگ ڤیو بکەرەوە"
    assert history.count("ترەیدینگ ڤیو بکەرەوە") == 1


async def test_handle_text_publishes_captions_transcripts_and_state(make_app):
    app, _ = brain_app(make_app, default=sorani_router)
    captions = collect(app.bus, Caption)
    transcripts = collect(app.bus, Transcript)
    states = collect(app.bus, VoiceState)
    reply = await app.submit_text("ترەیدینگ ڤیو بکەرەوە")
    assert reply == "کرایەوە، چارتەکە ئامادەیە."
    assert captions[0].role == "user" and captions[-1].final and captions[-1].text == reply
    assert [(t.role, t.source) for t in transcripts] == [("user", "text"), ("assistant", "text")]
    assert [s.state for s in states] == ["thinking", "idle"]
    conv = app.memory.get_conversation(app.conversation.conversation_id)
    assert conv["source"] == "text" and conv["title"].startswith("ترەیدینگ")


async def test_text_mode_uses_text_style_and_voice_mode_is_speakable(make_app):
    app, backend = brain_app(make_app, default=Reply(text="**باشە.** ئەمە [لینک](https://x.y) بوو."))
    spoken = await drain(app.conversation.respond_stream("شتێک بڵێ", source="cascade"))
    assert spoken == ["باشە.", "ئەمە لینک بوو."]
    typed = await app.conversation.handle_text("شتێک بڵێ")
    assert "**" in typed
    assert "Speaking style" in backend.requests[0].messages[0]["content"]
    assert "Writing style" in backend.requests[1].messages[0]["content"]


async def test_no_model_gives_an_honest_sorani_apology(make_app):
    app, _ = brain_app(make_app, default=LLMError("server", "down", provider="groq", model="x"))
    errors = collect(app.bus, Error)
    chunks = await drain(app.conversation.respond_stream("سڵاو", source="cascade"))
    assert chunks == [SORANI_NO_MODEL]
    assert errors and errors[0].where == "conversation"
    bare = make_app(backends={})
    from sam.brain import conversation, memory, persona
    for module in (memory, persona, conversation):
        module.register(bare)
    assert await bare.conversation.handle_text("سڵاو") == SORANI_NO_MODEL


async def test_tool_rounds_are_capped_then_a_wording_round_is_forced(make_app):
    app, backend = brain_app(make_app, default=lambda req: Reply(calls=[("screen_look", {"window": str(len(req.messages))})])
                             if req.tool_choice != "none" else Reply(text="نەمتوانی تەواوی بکەم."))
    app.config.set("conversation.max_tool_rounds", 3)
    chunks = await drain(app.conversation.respond_stream("شتێک بکە", source="text"))
    assert chunks == ["نەمتوانی تەواوی بکەم."]
    assert len(backend.requests) == 4 and backend.requests[-1].tool_choice == "none"
    assert [n for n, _ in CALLS] == ["screen_look"] * 3


async def test_identical_repeat_calls_are_not_executed_twice(make_app):
    app, _ = brain_app(make_app, [Reply(calls=[("open_app", {"name": "Chrome"})]),
                                  Reply(calls=[("open_app", {"name": "Chrome"})]),
                                  Reply(text="کرۆم کرایەوە.")])
    chunks = await drain(app.conversation.respond_stream("کرۆم بکەرەوە", source="text"))
    assert chunks == ["کرۆم کرایەوە."]
    assert CALLS == [("open_app", {"name": "Chrome"})]


async def test_silent_model_after_tools_gets_an_honest_fallback(make_app):
    # Every rung stays silent (default=""): the empty answer is re-asked on the
    # remaining rungs (gpt-oss-20b is the text ladder's last resort) first.
    app, _ = brain_app(make_app, [Reply(calls=[("open_app", {"name": "missing app"})]), Reply(text="")],
                       default=Reply(text=""))
    chunks = await drain(app.conversation.respond_stream("بەرنامەکە بکەرەوە", source="text"))
    # The tool's own honest outcome in Sorani (sam/brain/outcome.py), not a bare "not done".
    assert chunks == ["نەمتوانی بەرنامەکە بکەمەوە."]
    app2, _ = brain_app(make_app, [Reply(calls=[("tv_open", {})]), Reply(text="")], default=Reply(text=""))
    assert await drain(app2.conversation.respond_stream("ترەیدینگ ڤیو", source="text")) == ["ترەیدینگ ڤیو ئامادەیە."]


async def test_confirmation_is_answered_by_the_users_typed_yes(make_app):
    app, _ = brain_app(make_app, [Reply(calls=[("delete_thing", {"what": "فایلە کۆنەکان"})]), Reply(text="سڕانەوە.")])
    requests = collect(app.bus, ConfirmRequest)

    async def say_yes():
        while not requests:
            await asyncio.sleep(0.01)
        assert await app.conversation.handle_text("بەڵێ") == ""

    answer = asyncio.create_task(say_yes())
    chunks = await drain(app.conversation.respond_stream("فایلە کۆنەکان بسڕەوە", source="cascade"))
    await answer
    assert chunks[1:] == ["سڕانەوە."] and chunks[0] in ACKS_DO     # spoken acknowledgement first
    assert CALLS == [("delete_thing", {"what": "فایلە کۆنەکان"})]
    assert requests[0].question_ckb == "فایلە کۆنەکان بسڕمەوە؟"


async def test_untrusted_screen_text_reaches_the_model_only_as_tool_data(make_app):
    app, backend = brain_app(make_app, [Reply(calls=[("screen_look", {})]), Reply(text="سێ دوگمە هەیە.")])
    await drain(app.conversation.respond_stream("سەیری شاشە بکە", source="text"))
    result = tool_results(backend.requests[1])[0]
    assert "untrusted" in result["data"]
    assert all("Ignore previous" not in str(m.get("content")) for m in backend.requests[1].messages
               if m["role"] in ("system", "user"))


async def test_sleep_ends_the_conversation_and_extracts_facts(make_app):
    app, backend = brain_app(make_app, default='{"facts": [{"text": "ناوی سامییە", "kind": "person"}]}')
    cid = app.conversation.ensure_conversation("cascade")
    for text in ("ناوم سامییە", "من ترەیدەرم"):
        app.bus.publish(Transcript(role="user", text=text, source="cascade", conversation_id=cid))
        app.bus.publish(Transcript(role="assistant", text="باشە " + text, source="cascade", conversation_id=cid))
    app.bus.publish(VoiceState(state="sleeping"))
    for _ in range(50):
        if app.memory.get_conversation(cid)["facts_extracted"]:
            break
        await asyncio.sleep(0.01)
    assert app.conversation.conversation_id is None
    conv = app.memory.get_conversation(cid)
    assert conv["ended_at"] is not None and conv["facts_extracted"] == 1
    assert [f["text"] for f in app.memory.list_facts()] == ["ناوی سامییە"]
    # The next utterance starts a new conversation that knows the previous one.
    app.bus.publish(Transcript(role="user", text="دووبارە سڵاو", source="cascade"))
    assert app.conversation.conversation_id != cid
    app.conversation.conversation_id = None
    fresh = app.conversation.new_conversation("voice")
    assert "Previous conversation" in app.conversation.context_for_prompt("voice") and fresh


async def test_idle_conversation_is_closed_by_the_watcher(make_app):
    app, _ = brain_app(make_app, default='{"facts": []}')
    cid = app.conversation.ensure_conversation("text")
    app.config.set("conversation.idle_timeout_s", 1)
    app.conversation._last_activity -= 5
    task = asyncio.create_task(app.conversation.idle_watch(interval_s=0.01))
    for _ in range(100):
        if app.conversation.conversation_id is None:
            break
        await asyncio.sleep(0.01)
    task.cancel()
    assert app.conversation.conversation_id is None
    assert app.memory.get_conversation(cid)["ended_at"] is not None


async def test_older_turns_are_summarised_in_the_background(make_app):
    app, backend = brain_app(make_app, default="User asked about gold levels and opened TradingView.")
    app.config.set("conversation.history_turns", 4)
    cid = app.conversation.ensure_conversation("live")
    for i in range(8):
        app.bus.publish(Transcript(role="user", text=f"پرسیاری {i}", source="live", conversation_id=cid))
        app.bus.publish(Transcript(role="assistant", text=f"وەڵامی {i}", source="live", conversation_id=cid))
    # Acceptance 2026-09-24: never a model call in the middle of the exchange ...
    await app.conversation.background_tick()
    assert app.memory.get_conversation(cid)["summary"] == "" and backend.requests == []
    # ... but in the next pause (brain/budget.py quiet_s), with one request.
    app.conversation._last_activity -= 60                                     # noqa: SLF001
    await app.conversation.background_tick()
    assert app.memory.get_conversation(cid)["summary"].startswith("User asked about gold")
    assert len(backend.requests) == 1
    context = app.conversation.context_for_prompt("text")
    assert "Earlier in this conversation" in context
    upto = app.db.scalar("SELECT summary_upto FROM brain_conversation_state WHERE conversation_id=?", (cid,))
    assert upto > 0


async def test_summary_falls_back_to_extractive_without_a_model(make_app):
    app, _ = brain_app(make_app, default=LLMError("server", "down", provider="groq", model="x"))
    app.config.set("conversation.history_turns", 2)
    cid = app.conversation.ensure_conversation("live")
    for i in range(6):
        app.memory.add_turn(cid, "user", f"داواکاری {i}", source="live")
        app.memory.add_turn(cid, "assistant", f"وەڵام {i}", source="live")
    summary = await app.conversation.summarize(cid)
    assert summary.startswith("User asked: داواکاری 0")


async def test_typed_text_goes_into_an_open_live_session(make_app):
    app, backend = brain_app(make_app)

    class FakeVoice:
        live_session_open = True
        state = "listening"
        engine_name = "live"

        def __init__(self):
            self.sent = []

        async def send_text(self, text):
            self.sent.append(text)
            return True

    app.voice = FakeVoice()
    assert await app.submit_text("نرخی زێڕ چەندە؟") == ""
    assert app.voice.sent == ["نرخی زێڕ چەندە؟"] and backend.requests == []
    app.voice.live_session_open = False
    assert await app.submit_text("نرخی زێڕ چەندە؟") == "باشە."
    assert len(backend.requests) == 1


async def test_typed_replies_are_spoken_only_when_enabled(make_app):
    app, _ = brain_app(make_app)
    speak = collect(app.bus, SpeakRequest)
    await app.submit_text("سڵاو")
    assert speak == []
    app.config.set("conversation.speak_typed_replies", True)
    await app.submit_text("سڵاو دیسان")
    assert speak and speak[0].text_ckb == "باشە."


async def test_context_for_prompt_lists_recent_actions(make_app):
    app, _ = brain_app(make_app, default=sorani_router)
    await drain(app.conversation.respond_stream("ترەیدینگ ڤیو بکەرەوە", source="live"))
    context = app.conversation.context_for_prompt("voice")
    assert "Recent actions" in context and "open_app: ok" in context
    assert "Last turns" in context


# --- measured failure modes (2026-09-24) -------------------------------------------------------

async def test_undeclared_arguments_are_dropped_before_dispatch(make_app):
    # Gemini via OmniRoute sent tv_open({"reason": ...}) although tv_open declares no parameters.
    app, _ = brain_app(make_app, [Reply(calls=[("tv_open", {"reason": "بەکارهێنەر داوای کرد"}),
                                               ("open_app", {"name": "Chrome", "why": "x"})]),
                                  Reply(text="هەردووکیان کرانەوە.")])
    chunks = await drain(app.conversation.respond_stream("ترەیدینگ ڤیو و کرۆم بکەرەوە", source="text"))
    assert chunks == ["هەردووکیان کرانەوە."]
    assert CALLS == [("tv_open", {}), ("open_app", {"name": "Chrome"})]
    from sam.brain.conversation import clean_tool_args
    assert clean_tool_args(app.tools, "nope", {"a": 1}) == {"a": 1}


async def test_an_empty_answer_falls_through_to_the_next_rung(make_app):
    app, backend = brain_app(make_app, [Reply(text=""), Reply(text="سڵاو، فەرموو.")])
    app.config.set("conversation.ladder.text", ["groq:openai/gpt-oss-20b", "groq:openai/gpt-oss-120b"])
    chunks = await drain(app.conversation.respond_stream("سڵاو", source="text"))
    assert chunks == ["سڵاو، فەرموو."]
    assert backend.models == ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]


async def test_streaming_mode_and_its_silent_empty_stream_fallback(make_app):
    app, backend = brain_app(make_app, [Reply(text=""), Reply(text="ئەمە وەڵامە. دووەم ڕستە.")])
    app.config.set("conversation.stream", True)
    app.config.set("conversation.ladder.voice", ["groq:openai/gpt-oss-20b", "groq:openai/gpt-oss-120b"])
    chunks = await drain(app.conversation.respond_stream("شتێک", source="cascade"))
    assert chunks == ["ئەمە وەڵامە.", "دووەم ڕستە."]
    assert backend.models == ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
    # A normal streamed tool loop still works.
    app2, _ = brain_app(make_app, default=sorani_router)
    app2.config.set("conversation.stream", True)
    assert await drain(app2.conversation.respond_stream("ترەیدینگ ڤیو بکەرەوە", source="cascade")) == [
        "باشە، ئێستا.", "کرایەوە، چارتەکە ئامادەیە."]


async def test_nothing_said_and_nothing_done_is_not_reported_as_done(make_app):
    from sam.brain.conversation import SORANI_NOT_UNDERSTOOD
    app, _ = brain_app(make_app, default=Reply(text=""))
    assert await drain(app.conversation.respond_stream("هممم", source="cascade")) == [SORANI_NOT_UNDERSTOOD]


async def test_voice_picks_tools_fast_and_words_results_with_the_quality_ladder(make_app):
    app, backend = brain_app(make_app, [Reply(calls=[("tv_open", {})]), Reply(text="کرایەوە.")])
    chunks = await drain(app.conversation.respond_stream("ترەیدینگ ڤیو بکەرەوە", source="cascade"))
    assert chunks[0] in ACKS_DO and chunks[1:] == ["کرایەوە."]
    # Round 1 on the picker ladder (Groq gpt-oss-20b first without a Gemini key),
    # the wording round on the quality ladder (omniroute/gemini unconfigured here
    # -> gpt-oss-120b, the first Groq rung of the wording ladder).
    assert backend.models == ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
    voice = app.conversation._ladder("voice")
    assert voice[:2] == ["groq:openai/gpt-oss-20b", "groq:openai/gpt-oss-120b"]
    assert voice.index("omniroute:sam-fast") == 2 and "omniroute:gemini/gemini-3-flash-preview" in voice
    text = app.conversation._ladder("text")
    assert text[:2] == ["groq:openai/gpt-oss-120b", "groq:openai/gpt-oss-20b"]
    wording = app.conversation._ladder("voice", after_tools=True)
    assert wording[0] == "omniroute:sam-fast" and wording[-2:] == ["groq:openai/gpt-oss-120b", "groq:openai/gpt-oss-20b"]
    assert app.conversation._ladder("text", after_tools=True) == wording
    # Typed text: no acknowledgement.
    app2, backend2 = brain_app(make_app, [Reply(calls=[("tv_open", {})]), Reply(text="کرایەوە.")])
    assert await drain(app2.conversation.respond_stream("ترەیدینگ ڤیو بکەرەوە", source="text")) == ["کرایەوە."]
    assert backend2.models == ["openai/gpt-oss-120b", "openai/gpt-oss-120b"]


async def test_acknowledgements_rotate_and_fit_the_tool(make_app):
    app, _ = brain_app(make_app)
    from sam.brain.llm import ToolCall
    do = [app.conversation._ack([ToolCall(id="1", name="open_app", arguments={})]) for _ in range(6)]
    assert all(a in ACKS_DO for a in do)
    assert all(a != b for a, b in zip(do, do[1:]))
    look = app.conversation._ack([ToolCall(id="2", name="analyze_market", arguments={})])
    assert look in ACKS_LOOK


async def test_wording_failure_after_a_tool_reports_the_tool_outcome(make_app):
    app, _ = brain_app(make_app, [Reply(calls=[("tv_open", {})])] + [LLMError("server", "x", provider="groq", model="m")] * 4)
    chunks = await drain(app.conversation.respond_stream("ترەیدینگ ڤیو بکەرەوە", source="cascade"))
    assert chunks[0] in ACKS_DO and chunks[-1] == "ترەیدینگ ڤیو ئامادەیە."
    assert CALLS == [("tv_open", {})]


async def test_a_request_that_got_no_model_is_left_out_of_later_prompts(make_app):
    """Integration smoke 2026-09-24: after three "no model" replies the next
    question was answered by running the stale "هێڵەکانت بسڕەوە" from history."""
    from sam.brain.conversation import drop_abandoned

    down = LLMError("server", "503", provider="groq", model="x")   # quick: retried once per rung, then rests
    app, backend = brain_app(make_app, [down, down, down, down, Reply(text="نرخی زێڕ ٤٢٦٠ دۆلارە.")])
    assert await drain(app.conversation.respond_stream("هێڵەکانت بسڕەوە", source="text")) == [SORANI_NO_MODEL]
    app.llm._cooldown.clear()   # the failed rungs rest now (repeated 5xx); pretend the rest is over
    await drain(app.conversation.respond_stream("نرخی زێڕ چەندە؟", source="text"))
    history = [m["content"] for m in backend.requests[-1].messages if m["role"] in ("user", "assistant")]
    assert history == ["نرخی زێڕ چەندە؟"]
    # Answered exchanges stay; only the unanswered request and its apology go.
    turns = [{"role": "user", "text": "a"}, {"role": "assistant", "text": "b"},
             {"role": "user", "text": "c"}, {"role": "assistant", "text": SORANI_NO_MODEL}]
    assert drop_abandoned(turns) == turns[:2]


async def test_wording_failure_speaks_the_tools_own_sorani_summary(make_app):
    from sam.brain.tools import ok, tool

    @tool("get_price", description="Price.", description_ckb="نرخ")
    async def fake_price(ctx):
        return ok("نرخی زێڕ ئێستا 4276.18.", price=4276.18)

    app, _ = brain_app(make_app, [Reply(calls=[("get_price", {})])] + [LLMError("server", "x", provider="groq", model="m")] * 4,
                       tools=(fake_price,))
    chunks = await drain(app.conversation.respond_stream("نرخی زێڕ چەندە؟", source="cascade"))
    assert chunks[-1] == "نرخی زێڕ ئێستا 4276.18."


async def test_each_model_request_of_a_turn_is_capped(make_app):
    app, backend = brain_app(make_app, [Reply(text="سڵاو.")])
    app.config.set("conversation.llm_timeout_s", 7)
    assert await drain(app.conversation.respond_stream("سڵاو", source="text")) == ["سڵاو."]
    assert backend.requests[0].timeout_s == 7
