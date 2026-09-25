"""Acceptance fixes 2026-09-24 (brain): slow Gemini rungs never eat the whole
first round, more_tools never becomes a false «تەواو بوو», background model
calls respect the quota budget (the user's evening test burnt the free quotas
on summaries/extraction while he talked), natural yes+verb answers, search
result links, and a new key clears the old key's rests."""

from __future__ import annotations

import asyncio
import time

import pytest
from brain_helpers import Reply, ScriptedBackend
from conftest import FakeBackend, rate_limited

from sam.brain import ladders, taint
from sam.brain.budget import BackgroundBudget
from sam.brain.confirm import ASK_AGAIN_CKB, classify_answer
from sam.brain.llm import LLMError, LLMResponse
from sam.brain.outcome import DONE, NO_RESULTS, tool_sentence
from sam.brain.responder import SORANI_NO_MODEL
from sam.events import Transcript


async def drain(stream):
    return [piece async for piece in stream]


def _app(make_app, backends):
    import importlib

    app = make_app(backends=backends)
    for name in ("memory", "persona", "worker", "conversation"):
        importlib.import_module(f"sam.brain.{name}").register(app)
    return app


class Hanging(FakeBackend):
    """A Gemini that never answers (sam2.log 2026-09-24: 30 s timeouts)."""

    async def complete(self, model, req):
        self.calls.append((model, req))
        await asyncio.sleep(60)
        return LLMResponse(text="late", provider=self.provider, model=model)


@pytest.fixture
def fast_ladders(monkeypatch):
    """The real ladder rules in fewer seconds (LLMClient never starts a rung
    with less than 0.5 s left, so the scale stops there)."""
    monkeypatch.setattr(ladders, "GEMINI_CAP_S", 0.7)
    monkeypatch.setattr(ladders, "HEAD_MAX_S", 0.9)
    monkeypatch.setattr(ladders, "FAST_RESERVE_S", 0.5)
    monkeypatch.setattr(ladders, "MIN_SLOW_RUNG_S", 0.6)


# -- finding 1: two hanging Gemini rungs + a healthy Groq ------------------------------------------------------
async def test_hanging_gemini_rungs_leave_the_round_to_groq(make_app, fast_ladders):
    gemini = Hanging("gemini")
    groq = ScriptedBackend([Reply(text="سڵاو، فەرموو."), Reply(text="باشم، سوپاس."), Reply(text="فەرموو.")],
                           provider="groq")
    app = _app(make_app, {"gemini": gemini, "groq": groq})
    app.config.set("conversation.picker_deadline_s", 2.0)
    started = time.perf_counter()
    first = await drain(app.conversation.respond_stream("سڵاو", source="text"))
    took = time.perf_counter() - started
    assert first == ["سڵاو، فەرموو."], first
    assert took < 1.5                                   # the round's deadline is 2.0: Groq got its turn in time
    assert [m for m, _ in gemini.calls] == ["gemini-3.5-flash-lite"]      # the 2nd slow rung was not started
    assert app.llm.cooling("gemini:gemini-3.5-flash-lite")                 # it missed its cap: it rests
    # The other Gemini model (its own quota) gets one bounded try, then both rest.
    started = time.perf_counter()
    second = await drain(app.conversation.respond_stream("چۆنی؟", source="text"))
    assert second == ["باشم، سوپاس."] and time.perf_counter() - started < 1.5
    started = time.perf_counter()
    third = await drain(app.conversation.respond_stream("ئەی تۆ؟", source="text"))
    assert third == ["فەرموو."] and time.perf_counter() - started < 0.5          # no wait on a resting Gemini
    assert [m for m, _ in gemini.calls] == ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]


def test_head_budget_always_leaves_room_for_a_fast_picker():
    assert ladders.head_budget(ladders.PICKER_DEADLINE_S) <= ladders.PICKER_DEADLINE_S - ladders.FAST_RESERVE_S
    assert ladders.head_budget(ladders.WORDING_DEADLINE_S) <= ladders.WORDING_DEADLINE_S - ladders.FAST_RESERVE_S
    assert ladders.GEMINI_CAP_S <= ladders.HEAD_MAX_S


async def test_a_deadline_cut_after_most_of_the_cap_blames_the_rung(make_app):
    class Slow(FakeBackend):
        async def complete(self, model, req):
            await asyncio.sleep(5)
            return LLMResponse(text="late", provider=self.provider, model=model)

    app = make_app(backends={"gemini": Slow("gemini")})
    with pytest.raises(LLMError):
        await app.llm.chat([{"role": "user", "content": "x"}], ladder=["gemini:m"],
                           rung_timeouts={"gemini": 0.8}, deadline_s=0.7)
    assert app.llm.cooling("gemini:m")                    # it had 0.7 of its 0.8 s and said nothing


# -- finding 4: more_tools then no model ----------------------------------------------------------------------
@pytest.mark.parametrize("source", ["text", "cascade"])
async def test_more_tools_then_no_model_never_says_done(make_app, source):
    from sam.brain.tools import ok, tool

    @tool("files", description="Files.", params={"type": "object", "properties": {"path": {"type": "string"}}})
    async def fake_files(ctx, path=""):
        return ok("listed")

    groq = ScriptedBackend([Reply(calls=[("more_tools", {"tools": ["files"]})])],
                           default=rate_limited("groq", "x"), provider="groq")
    app = _app(make_app, {"groq": groq})
    app.tools.add(fake_files, owner="test")
    chunks = await drain(app.conversation.respond_stream("فایلەکانم پیشان بدە", source=source))
    assert DONE not in chunks and chunks[-1] == SORANI_NO_MODEL, chunks
    assert tool_sentence("more_tools", {}, {"ok": True, "summary": "attached"}) != DONE


def test_outcome_sentences_do_not_claim_results_on_screen():
    assert tool_sentence("web_search", {}, {"ok": True, "summary": "No results for 'x'."}) == NO_RESULTS
    for name in ("web_search", "recall", "list_alerts", "strategy_list", "strategy_get", "theory_info"):
        assert "شاشە" not in tool_sentence(name, {}, {"ok": True, "summary": "5 things"})


# -- background budget ------------------------------------------------------------------------------------------
def _quiet(app):
    app.conversation._last_activity = time.time() - 600                          # noqa: SLF001


async def test_summary_waits_for_a_pause_and_folds_without_a_model_under_pressure(make_app):
    groq = ScriptedBackend(default="User asked about gold.", provider="groq")
    app = _app(make_app, {"groq": groq})
    app.config.set("conversation.history_turns", 2)
    cid = app.conversation.ensure_conversation("cascade")
    for i in range(6):
        app.bus.publish(Transcript(role="user", text=f"داواکاری {i}", source="cascade", conversation_id=cid))
        app.bus.publish(Transcript(role="assistant", text=f"وەڵام {i}", source="cascade", conversation_id=cid))
    await app.conversation.background_tick()
    assert groq.requests == [] and app.memory.get_conversation(cid)["summary"] == ""      # live: nothing yet
    app.llm._cooldown["groq:openai/gpt-oss-120b"] = time.monotonic() + 60                # noqa: SLF001
    _quiet(app)
    await app.conversation.background_tick()
    assert groq.requests == []                                                          # a rung rests: no request
    assert app.memory.get_conversation(cid)["summary"].startswith("User asked: داواکاری 0")
    counts = app.conversation.budget.counts_today()
    assert counts["summary"]["skipped"] == 1 and "ran" not in counts["summary"]


async def test_extraction_is_deferred_under_pressure_and_runs_once_later(make_app):
    groq = ScriptedBackend(default='{"facts": [{"text": "ناوی سامییە", "kind": "person"}]}', provider="groq")
    app = _app(make_app, {"groq": groq})
    cid = app.conversation.ensure_conversation("cascade")
    for text in ("ناوم سامییە", "من ترەیدەرم"):
        app.bus.publish(Transcript(role="user", text=text, source="cascade", conversation_id=cid))
    app.llm._cooldown["groq:openai/gpt-oss-20b"] = time.monotonic() + 60                 # noqa: SLF001
    await app.conversation.on_sleep()
    assert groq.requests == [] and cid in app.conversation._extract_due                  # noqa: SLF001
    app.llm._cooldown.clear()                                                           # noqa: SLF001
    _quiet(app)
    await app.conversation.background_tick()
    assert len(groq.requests) == 1 and app.memory.get_conversation(cid)["facts_extracted"] == 1
    assert groq.models == ["openai/gpt-oss-20b"]                                        # exactly one rung asked
    counts = app.conversation.budget.counts_today()["extract"]
    assert counts == {"deferred": 1, "ran": 1}


async def test_background_calls_stop_at_the_daily_budget(make_app):
    app = _app(make_app, {"groq": ScriptedBackend(provider="groq")})
    budget: BackgroundBudget = app.conversation.budget
    _quiet(app)
    refs = ["groq:openai/gpt-oss-20b"]
    assert budget.reason_to_skip("summary", refs) is None
    app.config.set("brain.background.daily_max", 2)
    budget.record("summary", "ran")
    budget.record("extract", "ran")
    assert budget.reason_to_skip("summary", refs) == "daily budget used"
    assert budget.reason_to_skip("reword", refs, allow_live=True) is None          # its own bucket
    app.conversation.active_turns = 1
    assert budget.reason_to_skip("reword", refs, allow_live=True) is None          # a reword is part of a turn
    app.config.set("brain.background.daily_max", 99)
    assert budget.reason_to_skip("summary", refs) == "live"                        # never during a turn


async def test_reword_is_skipped_while_any_rung_rests(make_app):
    groq = ScriptedBackend([Reply(text="RSI یان Relative Strength Index یه‌کێ لە کۆنترۆڵەکانی فینانسی")],
                           provider="groq")
    gemini = ScriptedBackend([Reply(text="RSI پێوەرێکی هێزی نرخە.")], provider="gemini")
    app = _app(make_app, {"groq": groq, "gemini": gemini})
    app.config.set("conversation.ladder.voice", ["groq:openai/gpt-oss-20b"])
    app.config.set("conversation.ladder.reply", ["gemini:gemini-3.5-flash-lite"])
    app.llm._cooldown["gemini:gemini-3.1-flash-lite"] = time.monotonic() + 60          # noqa: SLF001
    chunks = await drain(app.conversation.respond_stream("RSI چییە؟", source="cascade"))
    assert gemini.requests == [] and "Relative Strength" in " ".join(chunks)
    assert app.conversation.budget.counts_today()["reword"] == {"skipped": 1}


# -- confirmations -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("answer,question,expected", [
    ("بەڵێ، بینێرە", "ئەم نامەیە بنێرم؟", True),
    ("بینێرە", "ئەم نامەیە بنێرم؟", True),
    ("بیسڕەوە", "report.docx بسڕمەوە؟", True),
    ("بەڵێ دایبخە", "پەنجەرەکە دابخەم؟", True),
    ("بیسڕەوە", "ئەم نامەیە بنێرم؟", None),              # another action's verb never approves
    ("بینێرە", "", None),
    ("مەینێرە", "ئەم نامەیە بنێرم؟", False),
    ("باشە", "ئەم نامەیە بنێرم؟", None),
    ("بەڵێ", "", True),
])
def test_yes_plus_the_actions_own_verb(answer, question, expected):
    assert classify_answer(answer, question) is expected


async def test_voice_answer_uses_the_pending_question_and_short_unclear_text_asks_again(make_app):
    app = _app(make_app, {"groq": ScriptedBackend(provider="groq")})
    pending = asyncio.ensure_future(app.confirm.confirm("ئەم نامەیە بنێرم؟", tool_name="type_text"))
    await asyncio.sleep(0)
    assert app.confirm.needs_clear_answer("باشە") and not app.confirm.needs_clear_answer("بینێرە")
    assert await app.conversation.handle_text("باشە") == ASK_AGAIN_CKB
    assert app.confirm.offer_transcript("بەڵێ بینێرە") is True
    assert await pending is True


# -- taint: the scope's own search results --------------------------------------------------------------------------
def test_fetching_a_search_result_link_needs_no_question():
    state = taint.TaintState(user_text="هەواڵی زێڕ بگەڕێ")
    taint.note(state, "web_search", {"ok": True, "data": {"untrusted": [
        {"title": "Gold", "url": "https://www.reuters.com/markets/gold/", "snippet": "..."}]}})
    assert state.tainted
    assert taint.check("fetch_page", {"url": "https://www.reuters.com/markets/gold"}, state) is None
    assert taint.check("fetch_page", {"url": "https://www.reuters.com/markets/gold?q=secret"}, state)[0] == "confirm"
    assert taint.check("fetch_page", {"url": "https://attacker.example/x"}, state)[0] == "confirm"
    assert taint.check("open_url", {"url": "https://www.reuters.com/markets/gold/"}, state)[0] == "confirm"
    page = taint.TaintState()
    taint.note(page, "fetch_page", {"ok": True, "data": {"untrusted": "see https://a.example", "url": "https://a.example"}})
    assert page.tainted and page.result_urls == set()                  # links inside pages are never trusted


# -- rests after a new key --------------------------------------------------------------------------------------------
async def test_a_wrong_key_rest_ends_when_the_key_is_saved_or_tested(make_app):
    backend = FakeBackend("gemini", {"m": [LLMError("auth", "bad key", provider="gemini", model="m", status=401)]})
    app = make_app(backends={"gemini": backend})
    with pytest.raises(LLMError):
        await app.llm.chat([{"role": "user", "content": "x"}], ladder=["gemini:m"])
    assert app.llm.cooling("gemini:m")
    result = await app.llm.test_provider("gemini")                    # list_models works now
    assert result["ok"] and not app.llm.cooling("gemini:m")
    app.llm._cooldown["gemini:m"] = time.monotonic() + 600              # noqa: SLF001
    app.llm._strikes["gemini:m"] = 2                                    # noqa: SLF001
    assert app.llm.reset_provider("gemini") == ["gemini:m"]
    assert not app.llm.cooling("gemini:m") and app.llm.strikes("gemini:m") == 0
    from sam.brain.llm import LLMClient
    fresh = LLMClient(app.config, app.secrets, db=app.db, backends={"gemini": backend})
    assert not fresh.cooling("gemini:m")                                # also forgotten on disk
