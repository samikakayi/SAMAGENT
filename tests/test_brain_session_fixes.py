"""The user's live session of 2026-09-25 (KurdishTTS STT transcripts from the DB):

- «وەڵاهی جارێ ترێیت ملیۆم لۆ بکەوە بڕۆ سەر چار چی دەکەی؟» -> TradingView, by the fast path;
- «بڕۆ 100 چار 3 خولەکی یەکسەر لە گوڵت.» -> the chart became a symbol named "100";
- «کڕۆکڕۆک قەشمەر بڕۆ سەر چاوتی گوڵ» -> gold, calmly (insults are not part of the command);
- «کوڕە دەنگی بنەکەرە!» -> SAM muted the WINDOWS volume instead of its own voice;
- two utterances close together -> two overlapping model turns and two answers.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from brain_helpers import Reply, ScriptedBackend, brain_app, user_text
from fastpath_corpus import COLLOQUIAL, CORPUS, SESSION_2026_09_25, evaluate

from sam.brain import fastpath
from sam.brain.intents import match
from sam.brain.tools import ok, tool
from sam.trading.symbols import (chart_symbol, mentioned_instruments, resolve_instrument, trading_context,
                                 user_named)

SEEN: list[tuple[str, dict[str, Any]]] = []


@tool("tv_open", description="open TradingView")
async def fake_tv_open(ctx) -> dict[str, Any]:
    SEEN.append(("tv_open", {}))
    return ok("ترەیدینگ ڤیو ئامادەیە.", state="connected")


@tool("tv_set_chart", description="chart", params={"type": "object", "properties": {
    "symbol": {"type": "string"}, "timeframe": {"type": "string"}}})
async def fake_set_chart(ctx, symbol: str | None = None, timeframe: str | None = None) -> dict[str, Any]:
    SEEN.append(("tv_set_chart", {k: v for k, v in {"symbol": symbol, "timeframe": timeframe}.items() if v}))
    return ok("چارتەکە گۆڕا بۆ زێڕ.", changed=True)


@tool("system_control", description="Windows volume", params={"type": "object", "properties": {
    "action": {"type": "string"}, "level": {"type": "integer"}}, "required": ["action"]})
async def fake_system_control(ctx, action: str, level: int | None = None) -> dict[str, Any]:
    SEEN.append(("system_control", {"action": action}))
    return ok("Volume is 0% and muted.")


class Voice:
    """Stands in for app.voice: counts stop_speaking calls (the voice contract 3.1)."""

    def __init__(self) -> None:
        self.stopped = 0
        self.state = "idle"

    async def stop_speaking(self) -> None:
        self.stopped += 1


@pytest.fixture
def fast(make_app):
    app, backend = brain_app(make_app, tools=(fake_tv_open, fake_set_chart, fake_system_control), fastpath=True)
    app.voice = Voice()
    SEEN.clear()
    return app, backend


async def turn(app: Any, text: str, source: str = "text") -> list[str]:
    return [c async for c in app.conversation.respond_stream(text, source=source)]


# --- the fast path -------------------------------------------------------------------------------------------------
def test_the_session_and_colloquial_rows_keep_precision_at_one():
    assert len(COLLOQUIAL) >= 80 and sum(1 for _, label, _ in COLLOQUIAL if label) >= 40
    for rows in (SESSION_2026_09_25, COLLOQUIAL, CORPUS):
        result = evaluate(match, rows)
        assert result["fp"] == 0 and result["precision"] == 1.0, result["wrong"]
    assert evaluate(match, SESSION_2026_09_25)["recall"] == 1.0
    assert evaluate(match, COLLOQUIAL)["recall"] == 1.0


async def test_tradingview_from_the_real_stt_words_needs_no_model(fast):
    app, backend = fast
    chunks = await turn(app, "وەڵاهی جارێ ترێیت ملیۆم لۆ بکەوە بڕۆ سەر چار چی دەکەی؟")
    assert SEEN == [("tv_open", {})] and backend.requests == []
    assert "ترەیدینگ ڤیو" in " ".join(chunks)


async def test_an_insulting_request_is_just_done(fast):
    app, backend = fast
    chunks = await turn(app, "کڕۆکڕۆک قەشمەر بڕۆ سەر چاوتی گوڵ")
    assert SEEN == [("tv_set_chart", {"symbol": "گوڵ"})] and backend.requests == []
    assert "قەشمەر" not in " ".join(chunks)                       # never mirrors the insult


@pytest.mark.parametrize("text,timeframe", [("زێڕ لەسەر ٣ خولەکی", "M3"), ("گۆڵد لەسەر 3 خولەکی دابنێ", "M3"),
                                            ("چارتەکە بکە بە چار سەعات", "H4"), ("زێڕ لەسەر نیو سەعات", "M30"),
                                            ("گۆڵت لەسەر یەک سەعات پیشان بدە", "H1"),
                                            ("زێڕ لەسەر ٤ سەعاتی پیشان بدە", "H4"),
                                            ("گوڵد لەسەر ڕۆژانە پیشان بدە", "D1"), ("بیکە بە پازدە خولەکی", "M15")])
def test_spoken_timeframes(text, timeframe):
    intent = match(text)
    assert intent is not None and intent.name == "set_chart" and intent.args["timeframe"] == timeframe


def test_the_chart_takes_tradingview_only_intervals():
    from sam.trading.tv_parse import parse_tv_resolution

    assert [parse_tv_resolution(v) for v in ("M3", "3m", "H2", "h3", "45min", "M7")] == ["3", "3", "120", "180",
                                                                                         "45", None]


@pytest.mark.parametrize("text", ["گوڵ پیشان بدە", "گوڵێک بکڕە", "گوڵ چەندە", "بڕۆ سەر گوڵ", "گۆڵ بکەرەوە"])
async def test_the_flower_word_is_gold_only_next_to_the_chart(fast, text):
    app, backend = fast
    await turn(app, text)
    assert SEEN == [] and len(backend.requests) >= 1


def test_the_gold_words_resolve_to_gold():
    for word in ("گوڵ", "گوڵت", "گوڵد", "گۆڵت", "گۆڵد", "گولد", "گۆڵ", "ئاڵتون", "ئاڵتوون", "زێڕ", "زیڕ", "ذهب",
                 "gold", "xau", "XAUUSD", "xau/usd"):
        assert resolve_instrument(word) == "XAUUSD", word
    assert resolve_instrument("گوڵ", context=False) is None and resolve_instrument("گۆڵ", context=False) is None
    assert resolve_instrument("گوڵت", context=False) == "XAUUSD"
    assert mentioned_instruments("کڕۆکڕۆک قەشمەر بڕۆ سەر چاوتی گوڵ") == {"XAUUSD"}
    assert mentioned_instruments("گوڵێکی جوان بۆ دایکم بکڕە") == set()
    assert mentioned_instruments("نرخی گوڵ چەندە") == {"XAUUSD"} and trading_context("گوڵ لەسەر ١٥ خولەک")
    assert not trading_context("گوڵەکان ئاو بدە")


@pytest.mark.parametrize("symbol", ["100", "3", "z", "ab", "ین", "banana", "شتێکی نەناسراو", "AAPL", ""])
def test_a_chart_symbol_must_be_a_known_instrument(symbol):
    assert chart_symbol(symbol) is None


def test_the_users_own_feeds_count_as_known():
    learned = {"XAUUSD": "PEPPERSTONE:XAUUSD", "AAPL": "NASDAQ:AAPL"}
    assert chart_symbol("NASDAQ:AAPL", learned=learned) == "AAPL" and chart_symbol("aapl", learned=learned) == "AAPL"
    assert chart_symbol("GER40", mapped={"GER40": {"tv": "PEPPERSTONE:GER40"}}) == "GER40"
    assert chart_symbol("گوڵ") == "XAUUSD" and chart_symbol("PEPPERSTONE:XAUUSD") == "XAUUSD"


def test_digits_in_the_users_words_never_name_a_symbol(make_app):
    app, _ = brain_app(make_app, tools=())
    cid = app.conversation.ensure_conversation("cascade")
    app.memory.add_turn(cid, "user", "بڕۆ 100 چار 3 خولەکی یەکسەر", source="cascade")
    assert user_named(app, "100", "cascade") is False
    app.memory.add_turn(cid, "user", "show NAS100 please", source="text")
    assert user_named(app, "NAS100", "text") is True


# --- SAM's own voice, not the Windows volume -------------------------------------------------------------------------
async def test_be_quiet_stops_sams_own_voice_and_never_mutes_windows(fast):
    app, backend = fast
    chunks = await turn(app, "کوڕە دەنگی بنەکەرە!", source="cascade")
    assert app.voice.stopped == 1 and SEEN == [] and backend.requests == []
    assert "".join(chunks).strip() == ""                             # the user asked for quiet: nothing is said
    typed = await turn(app, "دەنگت بنەکەرە")
    assert " ".join(typed) == "باشە، بێدەنگ بووم." and app.voice.stopped == 2


@pytest.mark.parametrize("text", ["بێدەنگ بە", "دەنگ مەکە", "قسە مەکە", "بێدەنگ بە ئیتر", "shut up", "stop talking"])
def test_quiet_phrases_are_sams_voice(text):
    intent = match(text)
    assert intent is not None and intent.name == "quiet" and intent.tool == "stop_speaking"


@pytest.mark.parametrize("text", ["دەنگی کۆمپیوتەر بنەکەرە", "دەنگەکە کەم بکەرەوە", "دەنگت خۆشە", "باسی زێڕ مەکە"])
def test_the_computer_volume_and_other_sentences_are_not_quiet(text):
    assert match(text) is None


def test_the_model_is_told_the_difference(make_app):
    app, _ = brain_app(make_app, tools=())
    voice = app.persona.system_instruction("voice")
    assert "stop_speaking" in voice and "Windows volume only if he names the computer" in voice
    assert "insults you" in voice                                    # calm, never mirrors
    spec = app.tools.get("stop_speaking")
    assert spec is not None and "never touches the computer's volume" in spec.description
    assert "stop_speaking" in app.config.get("conversation.core_tools")


# --- one answer per request -------------------------------------------------------------------------------------------
class GatedBackend(ScriptedBackend):
    """The first model request waits until ``gate`` is set (a slow free model)."""

    def __init__(self, steps: list[Any]) -> None:
        super().__init__(steps)
        self.gate = asyncio.Event()
        self.cancelled = 0

    async def complete(self, model: str, req: Any) -> Any:
        if not self.requests:
            self._record(model, req)
            reply = self._next(req)
            try:
                await self.gate.wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
            return self._response(model, reply, self._calls(reply))
        return await super().complete(model, req)


def gated_app(make_app, steps, tools=()):
    backend = GatedBackend(steps)
    app = make_app(backends={"groq": backend})
    app.config.set("brain.fastpath.enabled", False)
    import importlib

    for name in ("memory", "persona", "worker", "conversation"):
        importlib.import_module(f"sam.brain.{name}").register(app)
    for fn in tools:
        app.tools.add(fn, owner="test")
    return app, backend


async def test_a_newer_request_replaces_one_that_is_still_thinking(make_app):
    app, backend = gated_app(make_app, ["یەکەم وەڵام.", "باشە، هەردووکیانم کرد."])
    first = asyncio.ensure_future(turn(app, "کوڕە دەنگی"))
    for _ in range(50):
        if backend.requests:
            break
        await asyncio.sleep(0.01)
    second = await turn(app, "چارتەکە بکە بە سێ خولەک")
    older = await asyncio.wait_for(first, 5)
    assert older == [] and backend.cancelled == 1                 # stopped at once, said nothing
    assert " ".join(second) == "باشە، هەردووکیانم کرد."
    heard = user_text(backend.requests[-1])
    assert "کوڕە دەنگی" in heard and "چارتەکە بکە بە سێ خولەک" in heard     # the newest answers both
    replies = app.memory.recent_turns(app.conversation.conversation_id, roles=("assistant",))
    assert [r["text"] for r in replies] == ["باشە، هەردووکیانم کرد."]            # one answer stored


async def test_a_turn_whose_tool_ran_reports_it_once_without_a_second_model_round(make_app):
    released = asyncio.Event()
    ran: list[str] = []

    @tool("mute_sound", description="slow tool", risk="safe")
    async def slow_tool(ctx) -> dict[str, Any]:
        ran.append("mute")
        await released.wait()
        return ok("دەنگەکە کپ کرا.")

    backend = ScriptedBackend([Reply(calls=[("mute_sound", {})]), "چارتەکە گۆڕا بۆ سێ خولەک."],
                              default="هەرگیز نابێت ئەمە بڵێت.")
    app = make_app(backends={"groq": backend})
    app.config.set("brain.fastpath.enabled", False)
    import importlib

    for name in ("memory", "persona", "worker", "conversation"):
        importlib.import_module(f"sam.brain.{name}").register(app)
    app.tools.add(slow_tool, owner="test")
    first = asyncio.ensure_future(turn(app, "دەنگەکە کپ بکە", source="cascade"))
    for _ in range(100):
        if ran:
            break
        await asyncio.sleep(0.01)
    second = await turn(app, "چارتەکە بکە بە سێ خولەک")
    released.set()
    older = await asyncio.wait_for(first, 5)
    assert "دەنگەکە کپ کرا." in " ".join(older)                    # its own result, once
    assert "هەرگیز" not in " ".join(older)
    assert len(backend.requests) == 2                                # no wording round for the older turn
    assert " ".join(second) == "چارتەکە گۆڕا بۆ سێ خولەک."


async def test_a_replaced_fast_command_does_not_run(make_app):
    app, backend = brain_app(make_app, tools=(fake_set_chart,), fastpath=True)
    SEEN.clear()
    intent = match("گۆڵد لەسەر ١٥ خولەک پیشان بدە")
    out = [piece async for piece in fastpath.run(app, intent, source="text", proceed=lambda: False)]
    assert out == [] and SEEN == []
