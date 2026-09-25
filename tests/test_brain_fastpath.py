"""No-AI fast path: the labelled corpus (precision / recall), and whole turns
through ``respond_stream`` with the real tool names (fake handlers) and a model
backend that must never be asked."""

from __future__ import annotations

import time
from typing import Any

import pytest
from brain_helpers import brain_app
from fastpath_corpus import (CORPUS, HELD_OUT, HELD_OUT_FIRST_RUN, REVIEW_FIRST_RUN, REVIEW_HELD_OUT,
                             REVIEW_PROBES, evaluate)

from sam.brain import fastpath
from sam.brain.intents import match
from sam.brain.tools import fail, ok, tool

SEEN: list[tuple[str, dict[str, Any]]] = []


def _record(name: str, args: dict[str, Any]) -> None:
    SEEN.append((name, {k: v for k, v in args.items() if v is not None}))


@tool("get_price", description="price", params={"type": "object", "properties": {"symbol": {"type": "string"}}})
async def fake_get_price(ctx, symbol: str = "") -> dict[str, Any]:
    _record("get_price", {"symbol": symbol})
    return ok("زێڕ ئێستا لەسەر ٤٣١٢ مامەڵە دەکرێت.", symbol="XAUUSD", bid=4312.4)


@tool("tv_open", description="open TradingView")
async def fake_tv_open(ctx) -> dict[str, Any]:
    _record("tv_open", {})
    return ok("ترەیدینگ ڤیو ئامادەیە. چارت: زێڕ، پازدە خولەک.", state="connected")


@tool("open_app", description="open an app", params={"type": "object", "properties": {"name": {"type": "string"}},
                                                    "required": ["name"]})
async def fake_open_app(ctx, name: str) -> dict[str, Any]:
    _record("open_app", {"name": name})
    if name == "Excel":
        return fail("No installed app matches 'Excel'.", state="not_found")
    return ok(f"{name} is open.", state="focused" if name == "Notepad" else "started")


@tool("tv_set_chart", description="chart", params={"type": "object", "properties": {
    "symbol": {"type": "string"}, "timeframe": {"type": "string"}}})
async def fake_set_chart(ctx, symbol: str | None = None, timeframe: str | None = None) -> dict[str, Any]:
    _record("tv_set_chart", {"symbol": symbol, "timeframe": timeframe})
    return ok("چارتەکە گۆڕا بۆ زێڕ لەسەر پازدە خولەک.", changed=True)


@tool("analyze_market", description="analysis", params={"type": "object", "properties": {
    "symbol": {"type": "string"}, "timeframes": {"type": "array", "items": {"type": "string"}},
    "draw": {"type": "string"}, "vision": {"type": "boolean"}}}, blocking=False)
async def fake_analyze(ctx, symbol: str = "", timeframes: list[str] | None = None, draw: str = "full",
                       vision: bool = True) -> dict[str, Any]:
    _record("analyze_market", {"symbol": symbol, "timeframes": timeframes, "draw": draw, "vision": vision})
    return ok("زێڕ لە ڕەوتی سەرەوەدایە؛ نزیکترین پشتگیری ٤٢٩٠ و بەرگری ٤٣٣٠، کێشرانە سەر چارت.", verdict="WAIT")


@tool("clear_my_drawings", description="clear", params={"type": "object", "properties": {"tag": {"type": "string"}}})
async def fake_clear(ctx, tag: str | None = None) -> dict[str, Any]:
    _record("clear_my_drawings", {"tag": tag})
    return ok("٥ نیشانەی خۆمم سڕییەوە؛ دەستم لە هێڵەکانی تۆ نەدا.", removed=5)


@tool("list_alerts", description="alerts", params={"type": "object", "properties": {"status": {"type": "string"}}})
async def fake_list_alerts(ctx, status: str = "active") -> dict[str, Any]:
    _record("list_alerts", {"status": status})
    return ok("2 ئاگادارکردنەوە (active).", alerts=[
        {"id": 1, "what_ckb": "کاتێک زێڕ گەیشتە 4,300"}, {"id": 2, "what_ckb": "کاتێک زیو گەیشتە 52"}])


@tool("cancel_alert", description="cancel", params={"type": "object", "properties": {"alert_id": {"type": "string"}},
                                                  "required": ["alert_id"]})
async def fake_cancel(ctx, alert_id: str) -> dict[str, Any]:
    _record("cancel_alert", {"alert_id": alert_id})
    if alert_id == "all":
        return ok("2 ئاگادارکردنەوە هەڵوەشێنرایەوە.", cancelled=2)
    return fail(f"No active alert {alert_id}.")


TOOLS = (fake_get_price, fake_tv_open, fake_open_app, fake_set_chart, fake_analyze, fake_clear, fake_list_alerts,
         fake_cancel)


@pytest.fixture
def fast(make_app):
    app, backend = brain_app(make_app, tools=TOOLS, fastpath=True)
    SEEN.clear()
    return app, backend


async def turn(app: Any, text: str, source: str = "text") -> list[str]:
    return [c async for c in app.conversation.respond_stream(text, source=source)]


# --- the corpus ------------------------------------------------------------------------------------------

def test_corpus_precision_and_recall():
    """>= 150 labelled utterances; a wrong firing is worse than a miss."""
    result = evaluate(match)
    assert result["rows"] >= 150 and result["negative"] >= 100
    assert result["precision"] >= 0.98, result["wrong"]
    assert result["recall"] >= 0.9, result["wrong"]


def test_held_out_rows_were_scored_before_tuning():
    assert HELD_OUT_FIRST_RUN["precision"] == 1.0 and HELD_OUT_FIRST_RUN["rows"] == len(HELD_OUT)
    result = evaluate(match, HELD_OUT)
    assert result["precision"] >= 0.98 and result["recall"] >= HELD_OUT_FIRST_RUN["recall"]


def test_the_reviewers_held_out_rows():
    """The independent review's 77 adversarial rows: 21 of 64 negatives fired
    on the first run (precision 0.38 on that set); none may fire now."""
    assert REVIEW_FIRST_RUN["rows"] == len(REVIEW_HELD_OUT) and REVIEW_FIRST_RUN["fp"] == 21
    result = evaluate(match, REVIEW_HELD_OUT)
    assert result["fp"] == 0 and result["recall"] == 1.0, result["wrong"]
    probes = evaluate(match, REVIEW_PROBES)
    assert probes["fp"] == 0 and probes["recall"] == 1.0, probes["wrong"]


@pytest.mark.parametrize("text", ["cancel the alert", "remove my alert", "stop the alarm", "stop alerts",
                                  "ئاگادارکردنەوەکە بسڕەوە"])
def test_one_alert_never_means_every_alert(text):
    assert match(text) is None


@pytest.mark.parametrize("text", ["draw support at 60", "هێڵی پشتگیری نەوت لە ٦٠ بکێشە", "gold 240", "بیتکۆین ٦٠",
                                  "analyze", "market analysis", "delete the chart", "نەرمی زێڕ چەندە",
                                  "how much gold"])
async def test_prices_bare_nouns_and_look_alikes_go_to_the_model(fast, text):
    app, backend = fast
    await turn(app, text)
    assert SEEN == [] and len(backend.requests) >= 1


def test_a_timeframe_needs_a_unit_or_a_timeframe_word():
    assert match("change the timeframe to 60").args == {"timeframe": "H1"}
    assert match("draw support and resistance on 4 hours").args["timeframes"] == ["H4"]
    assert match("draw support and resistance on H4").args["timeframes"] == ["H4"]
    assert match("draw lines at 240") is None and match("switch to 240") is None


def test_matching_is_instant():
    texts = [row[0] for row in CORPUS]
    match(texts[0])                                   # tables are built once
    started = time.perf_counter()
    for text in texts:
        match(text)
    assert (time.perf_counter() - started) / len(texts) < 0.01


@pytest.mark.parametrize("text", ["", "   ", "؟", "سام", "زێڕ " * 20])
def test_empty_or_long_input_never_matches(text):
    assert match(text) is None


# --- whole turns, no model -------------------------------------------------------------------------------

@pytest.mark.parametrize("text,tool_name,args,reply", [
    ("نرخی زێڕ چەندە", "get_price", {"symbol": "زێڕ"}, "زێڕ ئێستا لەسەر ٤٣١٢ مامەڵە دەکرێت."),
    ("نەخنەشکی زێڕ چەندە", "get_price", {"symbol": "زێڕ"}, "زێڕ ئێستا لەسەر ٤٣١٢ مامەڵە دەکرێت."),
    ("ترەیدینگ ڤیو بکەرەوە", "tv_open", {}, "ترەیدینگ ڤیو ئامادەیە. چارت: زێڕ، پازدە خولەک."),
    ("کرۆم بکەرەوە", "open_app", {"name": "Google Chrome"}, "کرۆم کرایەوە."),
    ("open notepad", "open_app", {"name": "Notepad"}, "نۆتپاد پێشتر کرابووەوە؛ هێنامە پێشەوە."),
    ("ئێکسڵ بکەرەوە", "open_app", {"name": "Excel"}, "ئێکسڵ لەسەر ئەم کۆمپیوتەرە نەدۆزرایەوە."),
    ("گۆڵد لەسەر ١٥ خولەک پیشان بدە", "tv_set_chart", {"symbol": "گۆڵد", "timeframe": "M15"},
     "چارتەکە گۆڕا بۆ زێڕ لەسەر پازدە خولەک."),
    ("هێڵەکانت بسڕەوە", "clear_my_drawings", {}, "٥ نیشانەی خۆمم سڕییەوە؛ دەستم لە هێڵەکانی تۆ نەدا."),
    ("ئاگادارکردنەوەکانم پیشان بدە", "list_alerts", {"status": "active"},
     "دوو ئاگادارکردنەوەی چالاکت هەیە: کاتێک زێڕ گەیشتە 4,300؛ کاتێک زیو گەیشتە 52."),
    ("هەموو ئاگادارکردنەوەکان هەڵبوەشێنەوە", "cancel_alert", {"alert_id": "all"}, "2 ئاگادارکردنەوە هەڵوەشێنرایەوە."),
    ("cancel alert 7", "cancel_alert", {"alert_id": "7"}, "ئاگادارکردنەوەی ژمارە ٧ نەدۆزرایەوە."),
])
async def test_common_commands_run_the_tool_without_any_model(fast, text, tool_name, args, reply):
    app, backend = fast
    chunks = await turn(app, text)
    assert backend.requests == [] and SEEN == [(tool_name, args)]
    assert " ".join(chunks).strip() == reply
    assert app.db.usage_for("fastpath", match(text).name)["requests"] == 1        # counted: zero quota used


async def test_analysis_speaks_first_then_the_engines_summary_without_vision(fast):
    app, backend = fast
    chunks = await turn(app, "شیکاری زێڕ بکە و ئاستەکان بکێشە", source="cascade")
    assert chunks[0] == fastpath.ACK_LOOK and "نزیکترین پشتگیری" in " ".join(chunks[1:])
    assert SEEN == [("analyze_market", {"symbol": "زێڕ", "draw": "full", "vision": False})]
    assert backend.requests == []


async def test_draw_levels_uses_the_symbol_on_the_users_chart(fast):
    app, backend = fast

    class Tv:
        connected = True

        async def chart_state(self):
            return {"symbol": "BINANCE:BTCUSDT", "canonical": "BTCUSD"}

    app.trading.tv = Tv()
    await turn(app, "هێڵی پشتگیری و بەرگری بکێشە")
    assert SEEN == [("analyze_market", {"symbol": "BTCUSD", "draw": "levels", "vision": False})]


async def test_the_user_turn_is_stored_before_the_tool_runs(make_app):
    """tv_set_chart only applies a symbol the user named (symbols.user_named
    reads the stored turn): the fast path must store it first."""
    seen_turns = []

    @tool("tv_set_chart", description="chart", params={"type": "object", "properties": {
        "symbol": {"type": "string"}, "timeframe": {"type": "string"}}})
    async def checking(ctx, symbol: str | None = None, timeframe: str | None = None) -> dict[str, Any]:
        rows = ctx.app.memory.recent_turns(ctx.app.conversation.conversation_id, limit=1, roles=("user",))
        seen_turns.append(rows[-1]["text"] if rows else "")
        return ok("چارتەکە گۆڕا.")

    app, backend = brain_app(make_app, tools=(checking,), fastpath=True)
    await turn(app, "بیتکۆین پیشان بدە")
    assert seen_turns == ["بیتکۆین پیشان بدە"] and backend.requests == []


async def test_questions_and_off_switch_go_to_the_model(fast):
    app, backend = fast
    await turn(app, "بۆچی زێڕ دابەزی؟")
    assert len(backend.requests) >= 1 and SEEN == []
    backend.requests.clear()
    app.config.set("brain.fastpath.enabled", False)
    await turn(app, "نرخی زێڕ چەندە")
    assert len(backend.requests) >= 1


async def test_a_missing_tool_leaves_the_turn_to_the_model(make_app):
    app, backend = brain_app(make_app, tools=(), fastpath=True)
    await turn(app, "نرخی زێڕ چەندە")
    assert len(backend.requests) >= 1


def test_replies_for_declined_and_blocked_results():
    intent = match("کرۆم بکەرەوە")
    assert fastpath.reply(intent, intent.args, {"ok": False, "summary": "x", "data": {"declined": True}}) == \
        "باشە، نەمکرد."
    assert fastpath.reply(intent, intent.args, {"ok": False, "summary": "x", "data": {"blocked": True}}).startswith(
        "ئەمە ڕێگەپێدراو نییە")
    english = match("open chrome")
    assert fastpath.reply(english, english.args, {"ok": True, "summary": "Google Chrome is open.",
                                                  "data": {"state": "started"}}) == "کرۆم کرایەوە."
    assert fastpath.reply(match("بوەستە"), {}, {"ok": True, "summary": "Stopped.", "data": {}}) == "ڕاگیرا."
    assert fastpath._alerts_reply([]) == "هیچ ئاگادارکردنەوەیەکی چالاکت نییە."   # noqa: SLF001


def test_a_cancel_count_is_not_read_as_a_stopped_tool():
    """cancel_alert returns data.cancelled = <count>; only the registry's
    ``cancelled: True`` means stop_all interrupted the tool (outcome.py said
    «ڕاگیرا.» for «هەموو ئاگادارکردنەوەکان هەڵبوەشێنەوە» before)."""
    from sam.brain.outcome import own_sentence, tool_sentence

    counted = {"ok": True, "summary": "2 ئاگادارکردنەوە هەڵوەشێنرایەوە.", "data": {"cancelled": 2}}
    assert tool_sentence("cancel_alert", {"alert_id": "all"}, counted) == "2 ئاگادارکردنەوە هەڵوەشێنرایەوە."
    assert own_sentence("cancel_alert", {"alert_id": "all"}, counted) == "2 ئاگادارکردنەوە هەڵوەشێنرایەوە."
    stopped = {"ok": False, "summary": "Stopped by the user.", "data": {"cancelled": True}}
    assert tool_sentence("get_price", {}, stopped) == "ڕاگیرا."
