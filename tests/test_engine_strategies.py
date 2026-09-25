"""Strategy cards: ingest (one fake LLM call, Sorani text), versions, status,
search, prompt index, and the strategy_save/strategy_get tools."""

from __future__ import annotations

import json

from conftest import FakeBackend

from sam.brain.llm import LLMError
from sam.textnorm import is_arabic_script
from sam.trading import tools as trading_tools

SOURCE = ("ستراتیژی ئاسیا سویپ: لە چارتی یەک کاتژمێر ترێند دیاری دەکەم. دوای ئەوەی نرخ نزمایی سیشنی ئاسیا "
          "ڕادەماڵێت، لەسەر پازدە خولەک چاوەڕێی شکانی پێکهاتە دەکەم و لە ناو FVG دەچمە ژوورەوە. ستۆپ لە ژێر "
          "نزمایی ڕاماڵین، ئامانج یەک بە دوو. تەنها لە سیشنی لەندەن.")

MODEL_CARD = {
    "title_ckb": "ئاسیا سویپ", "title_en": "Asia Sweep", "markets": ["gold"], "direction": "both",
    "timeframes": {"bias": "H1", "setup": "15m", "entry": ""}, "sessions": ["London"],
    "rules": [
        {"kind": "bias", "text_ckb": "ترێندی یەک کاتژمێر", "text_en": "H1 trend sets the bias", "predicate": "trend_is",
         "params": [{"name": "tf", "value": "H1"}, {"name": "direction", "value": "setup"}]},
        {"kind": "setup", "text_ckb": "ڕاماڵینی نزمایی ئاسیا", "text_en": "Asian low is swept", "predicate": "swept",
         "params": [{"name": "tf", "value": "M15"}, {"name": "level", "value": "asian_low"},
                    {"name": "within_bars", "value": "12"}]},
        {"kind": "trigger", "text_ckb": "شکانی پێکهاتە لە پازدە خولەک", "text_en": "M15 MSS", "predicate": "mss_or_bos",
         "params": [{"name": "tf", "value": "M15"}]},
        {"kind": "entry", "text_ckb": "چوونەژوورەوە لە ناو FVG", "text_en": "enter inside the FVG", "predicate": "in_fvg",
         "params": []},
        {"kind": "setup", "text_ckb": "هێڵی جادوویی", "text_en": "magic line", "predicate": "magic_line", "params": []},
        {"kind": "stop", "text_ckb": "ستۆپ لە ژێر نزمایی ڕاماڵین", "text_en": "stop below the sweep low",
         "predicate": "", "params": []},
        {"kind": "filter", "text_ckb": "تەنها لە سیشنی لەندەن", "text_en": "London session only",
         "predicate": "session_is", "params": [{"name": "session", "value": "london"}]},
    ],
    "risk": {"target_rr": 2, "min_rr": 2}, "management": "", "theories": ["ICT", "liquidity"],
    "missing": ["entry_timeframe"], "summary_ckb": "ڕاماڵینی ئاسیا و شکانی پێکهاتە لە لەندەن.",
}


def build(make_app, replies):
    backend = FakeBackend("gemini", script={"gemini-3.5-flash-lite": replies})
    app = make_app(backends={"gemini": backend})
    trading_tools.register(app)
    return app, backend


async def test_ingest_turns_sorani_text_into_a_checkable_draft(make_app):
    app, backend = build(make_app, [json.dumps(MODEL_CARD, ensure_ascii=False)])
    result = await app.trading.strategies.ingest(SOURCE)
    card = result["card"]
    assert card["id"] == "asia-sweep" and card["status"] == "draft" and card["version"] == 1
    assert card["source_text"] == SOURCE, "the user's words are kept verbatim"
    # entry is inferred from the model's own M15 trigger rule
    assert card["markets"] == ["XAUUSD"] and card["timeframes"] == {"bias": "H1", "setup": "M15", "entry": "M15"}
    assert card["sessions"] == ["london"] and card["theories"] == ["ict", "liquidity"]
    checks = {r["id"]: r["check"] for r in card["rules"]}
    assert checks["r2"]["params"]["within_bars"] == 12 and checks["r2"]["params"]["reclaim"] is True
    assert checks["r5"] is None, "an unknown predicate becomes a chart-judged rule, never a guess"
    assert checks["r6"] is None
    assert result["missing"] == [], "the model's self-reported gap is filled by its own rules"
    readback = result["readback_ckb"]
    assert is_arabic_script(readback) and "ئاسیا سویپ" in readback and "بڵێ «بەڵێ»" in readback
    assert "5 مەرجیان خۆم بە ژمارە دەپشکنم" in readback
    request = backend.calls[0][1]
    assert request.json_schema is not None and "swept(" in request.messages[0]["content"]
    assert SOURCE in request.messages[1]["content"]


async def test_updates_are_new_versions_and_activation_indexes_the_card(make_app):
    app, backend = build(make_app, [json.dumps(MODEL_CARD, ensure_ascii=False)])
    store = app.trading.strategies
    first = (await store.ingest(SOURCE))["card"]
    second = (await store.ingest("ئامانج یەک بە سێ", strategy_id=first["id"]))["card"]
    assert second["id"] == first["id"] and second["version"] == 2
    assert [v["version"] for v in store.versions(first["id"])] == [2, 1]
    assert SOURCE in second["source_text"] and "یەک بە سێ" in second["source_text"]
    assert "existing card" in backend.calls[1][1].messages[1]["content"]
    assert store.index_for_prompt() == ""
    store.set_status(first["id"], "active")
    assert store.index_for_prompt().startswith("asia-sweep — ئاسیا سویپ")
    assert store.get("asia-sweep")["status"] == "active"
    assert [c["id"] for c in store.list("active")] == ["asia-sweep"]


async def test_search_finds_cards_by_sorani_words(make_app):
    app, _ = build(make_app, [json.dumps(MODEL_CARD, ensure_ascii=False)])
    await app.trading.strategies.ingest(SOURCE)
    assert app.trading.strategies.search("ڕاماڵینی ئاسیا")[0]["id"] == "asia-sweep"
    assert app.trading.strategies.search("لەندەن")[0]["id"] == "asia-sweep"
    assert app.trading.strategies.get("ئاسیا سویپ")["id"] == "asia-sweep"  # a title works as an id


async def test_the_tool_confirms_without_a_second_model_call(make_app):
    app, backend = build(make_app, [json.dumps(MODEL_CARD, ensure_ascii=False)])
    saved = await app.tools.dispatch("strategy_save", {"text": SOURCE}, source="text")
    assert saved["ok"] and saved["data"]["status"] == "draft" and is_arabic_script(saved["summary"])
    calls = len(backend.calls)
    confirmed = await app.tools.dispatch("strategy_save", {"text": "بەڵێ", "strategy_id": saved["data"]["strategy_id"],
                                                           "status": "active"}, source="text")
    assert confirmed["ok"] and confirmed["data"]["status"] == "active" and len(backend.calls) == calls
    got = await app.tools.dispatch("strategy_get", {"strategy_id": "asia-sweep"}, source="text")
    assert got["ok"] and got["data"]["rules"][1]["check"] == "swept" and got["data"]["rules"][5]["check"] == "chart"
    listed = await app.tools.dispatch("strategy_list", {"status": "active"}, source="text")
    assert listed["data"]["strategies"][0]["id"] == "asia-sweep"


async def test_a_model_failure_saves_nothing(make_app):
    app, _ = build(make_app, [LLMError("rate_limit", "quota", provider="gemini", model="x", status=429)])
    result = await app.tools.dispatch("strategy_save", {"text": SOURCE}, source="text")
    assert result["ok"] is False and "Nothing was saved" in result["summary"]
    assert app.trading.strategies.list() == []


def test_gaps_are_filled_from_the_models_own_rule_mapping():
    """Shape of a real sam-fast reply (live 2026-09-24): timeframes.entry empty
    while the entry/trigger rules say tf=M15, and 'TP 1:2' as rr_at_least(2)."""
    from sam.trading.strategies import card_missing, normalize_card
    live = {"title_ckb": "ستراتیژی ئاسیا سویپ", "title_en": "Asian Sweep Strategy", "timeframes": {"bias": "H1"},
            "sessions": ["london"], "risk": {}, "rules": [
                {"kind": "bias", "text_ckb": "ئاراستە لە H1", "predicate": "trend_is",
                 "params": [{"name": "tf", "value": "H1"}, {"name": "direction", "value": "setup"}]},
                {"kind": "trigger", "text_ckb": "MSS لە M15", "predicate": "mss_or_bos(tf=M15, direction=setup)"},
                {"kind": "entry", "text_ckb": "ناو FVG", "check": {"predicate": "in_fvg", "params": {"tf": "M15"}}},
                {"kind": "stop", "text_ckb": "ستۆپ", "predicate": "", "params": []},
                {"kind": "target", "text_ckb": "یەک بە دوو", "predicate": "rr_at_least",
                 "params": [{"name": "value", "value": "2.0"}]}]}
    card = normalize_card(live, source_text="x")
    assert card["timeframes"] == {"bias": "H1", "setup": None, "entry": "M15"}
    assert card["risk"]["target_rr"] == 2.0
    assert card["rules"][1]["check"] == {"predicate": "mss_or_bos",
                                         "params": {"tf": "M15", "direction": "setup", "within_bars": 10}}
    assert card["rules"][2]["check"]["predicate"] == "in_fvg"
    assert card_missing(card) == []


def test_a_missing_stop_is_asked_for_in_sorani():
    from sam.trading.strategies import card_missing, normalize_card, readback_ckb
    card = normalize_card({"title_ckb": "تاقی", "title_en": "t", "timeframes": {"entry": "M5"}, "risk": {"target_rr": 2},
                           "rules": [{"kind": "entry", "text_ckb": "چوونەژوورەوە", "predicate": "in_fvg"}]},
                          source_text="x")
    missing = card_missing(card)
    assert missing == ["stop"]
    assert readback_ckb(card, missing).endswith("تەنها ئەمەم پێ بڵێ: ستۆپ لە کوێ دادەنێیت؟")
