"""analyze_market with a fake TradingView bridge and a fake MT5 feed: data
source choice, feed-offset alignment, drawing vs dry run, vision for chart
rules, journal, timings, and the package wiring/tools."""

from __future__ import annotations

import json
import time

import pytest
from conftest import FakeBackend
from engine_helpers import FakeFeed, FakeTV, wave_bars

from sam.textnorm import is_arabic_script
from sam.trading import tools as trading_tools
from sam.trading.analyze import analyze_market, compact_report

NOW = time.time()  # analyze_market judges freshness against the real clock
OFFSET = 1.5       # TradingView's feed sits 1.5 above the broker (measured order of magnitude)


def mt5_bars(tfs=("H1", "M15", "M5", "M1")):
    return {tf: wave_bars(600, tf, end=NOW, base=2600.0) for tf in tfs}


def chart_bars(tf="M15"):
    return [{**b, "open": b["open"] + OFFSET, "high": b["high"] + OFFSET, "low": b["low"] + OFFSET,
             "close": b["close"] + OFFSET} for b in wave_bars(600, tf, end=NOW, base=2600.0)]


def setup(make_app, *, tv=None, feed=None, backends=None):
    app = make_app(backends=backends)
    trading_tools.register(app)
    app.trading.mt5 = feed if feed is not None else FakeFeed(mt5_bars(), now=NOW)
    app.trading.tv = tv
    return app


async def test_chart_bars_are_used_and_mt5_is_aligned_to_the_chart_feed(make_app):
    tv = FakeTV(symbol="OANDA:XAUUSD", timeframe="M15", bars=chart_bars())
    feed = FakeFeed(mt5_bars(), now=NOW)
    app = setup(make_app, tv=tv, feed=feed)
    report = await analyze_market(app, "گۆڵد", draw="full")
    assert report["data_source"] == "tradingview" and report["sources"]["M15"] == "tradingview"
    assert report["sources"]["H1"] == "mt5" and ("bars", "XAUUSD", "M15") not in feed.calls
    assert report["feed_offset"] == pytest.approx(OFFSET, abs=0.3)
    assert abs(report["price"] - tv._bars[-1]["close"]) < 1.0, "numbers are in the chart's price space"
    assert tv.cleared == ["analysis"], "only SAM's previous analysis drawings are cleared"
    assert tv.drawn and {d["tag"] for d in tv.drawn} == {f"analysis:{report['analysis_id']}"}
    assert report["drawn"] and report["drawing"]["dry_run"] is False
    row = app.db.query_one("SELECT verdict, symbol, drawn FROM analyses WHERE id=?", (report["analysis_id"],))
    assert row["symbol"] == "XAUUSD" and row["drawn"] == len(report["drawn"])
    stages = {r["stage"] for r in app.db.query("SELECT stage FROM timings")}
    assert {"analysis_engine", "analysis_total", "mt5_fetch", "tv_cdp"} <= stages
    assert report["engine_ms"] < 1500 and is_arabic_script(report["summary_ckb"])
    assert len(json.dumps(compact_report(report), ensure_ascii=False)) < 6000


async def test_another_symbol_on_the_chart_means_mt5_only_and_a_dry_run(make_app):
    tv = FakeTV(symbol="FX:EURUSD", timeframe="M15", bars=chart_bars())
    app = setup(make_app, tv=tv)
    report = await analyze_market(app, "XAUUSD", draw="levels")
    assert report["data_source"] == "mt5" and not report["feed_offset"]
    assert tv.drawn == [] and report["drawing"]["dry_run"] is True
    assert report["drawing"]["draw_plan"] and all(i["kind"] == "horizontal_line" for i in report["drawing"]["draw_plan"])


async def test_without_mt5_the_chart_bars_are_resampled(make_app):
    tv = FakeTV(symbol="TVC:GOLD", timeframe="M5", bars=chart_bars("M5"))
    feed = FakeFeed(mt5_bars(), now=NOW)
    feed.fail = True
    app = setup(make_app, tv=tv, feed=feed)
    report = await analyze_market(app, "XAUUSD", timeframes=["H1", "M15", "M5"], draw="none")
    assert report["sources"] == {"M5": "tradingview", "H1": "tradingview-resampled", "M15": "tradingview-resampled"}
    assert tv.drawn == []


async def test_no_data_at_all_is_an_honest_error(make_app):
    feed = FakeFeed(mt5_bars(), now=NOW)
    feed.fail = True
    app = setup(make_app, tv=None, feed=feed)
    with pytest.raises(RuntimeError, match="no market data"):
        await analyze_market(app, "XAUUSD")
    result = await app.tools.dispatch("analyze_market", {"symbol": "XAUUSD"}, source="text")
    assert result["ok"] is False and "Analysis failed" in result["summary"]


VISION_REPLY = json.dumps({"verdicts": [{"rule_id": "r2", "verdict": "pass", "why": "strong displacement candle"}]})


async def test_one_vision_call_judges_chart_rules_and_a_setup_is_drawn(make_app):
    backend = FakeBackend("gemini", script={"gemini-3.5-flash-lite": [VISION_REPLY]})
    tv = FakeTV(symbol="OANDA:XAUUSD", timeframe="M15", bars=chart_bars())
    app = setup(make_app, tv=tv, backends={"gemini": backend})
    card = app.trading.strategies.save({
        "title_ckb": "ترێند و دیسپلەیسمێنت", "title_en": "Trend displacement", "status": "active",
        "timeframes": {"bias": "H1", "entry": "M5"}, "risk": {"target_rr": 2.0},
        "rules": [{"id": "r1", "kind": "bias", "text_ckb": "ترێندی یەک کاتژمێر سەرەوە", "text_en": "H1 up",
                   "check": {"predicate": "trend_is", "params": {"tf": "H1", "direction": "up"}}},
                  {"id": "r2", "kind": "setup", "text_ckb": "دیسپلەیسمێنتی پاک", "text_en": "clean displacement",
                   "check": None}]})
    report = await analyze_market(app, None, strategy_id=card["id"], draw="full")
    rules = {r["id"]: r for r in report["strategy"]["rules"]}
    assert rules["r2"]["passed"] is True and rules["r2"]["how"] == "vision" and tv.screenshots == 1
    request = backend.calls[0][1]
    assert request.json_schema is not None and request.has_images()
    assert "never read prices" in request.messages[0]["content"]
    assert report["verdict"] == "SETUP" and report["direction"] == "long" and report["rr"] == 2.0
    assert any(d["kind"] == "long_position" for d in tv.drawn)
    assert "چوونەژوورەوە" in report["summary_ckb"] and "بڕیاری کۆتایی هی خۆتە" in report["summary_ckb"]


async def test_a_failed_vision_call_leaves_the_rule_unknown(make_app):
    from sam.brain.llm import LLMError
    backend = FakeBackend("gemini", script={"gemini-3.5-flash-lite": [LLMError("server", "boom", provider="gemini", model="x")]})
    tv = FakeTV(symbol="OANDA:XAUUSD", timeframe="M15", bars=chart_bars())
    app = setup(make_app, tv=tv, backends={"gemini": backend})
    card = app.trading.strategies.save({"title_ckb": "تاقی", "title_en": "t", "rules": [
        {"id": "r1", "kind": "setup", "text_ckb": "شێوەی سەر و شان", "check": None}]})
    report = await analyze_market(app, "XAUUSD", strategy_id=card["id"], draw="none")
    assert report["vision"]["ok"] is False and report["strategy"]["rules"][0]["passed"] is None
    assert report["verdict"] == "WAIT"


async def test_register_wires_the_package_and_tools(make_app):
    app = make_app()
    trading_tools.register(app)
    for slot in ("mt5", "engine", "theories", "strategies", "monitor"):
        assert getattr(app.trading, slot) is not None
    names = set(app.tools.names())
    assert {"get_price", "analyze_market", "set_alert", "list_alerts", "cancel_alert", "strategy_save",
            "strategy_list", "strategy_get", "theory_info"} <= names
    assert app.tools.get("analyze_market").blocking is False and app.tools.get("get_price").blocking is True
    assert all(app.tools.get(n).examples_ckb for n in ("analyze_market", "set_alert", "get_price"))
    assert app.config.get("trading.analysis_timeframes") == ["H1", "M15", "M5", "M1"]
    assert app.db.schema_version("trading") == 1
    app.trading.mt5 = FakeFeed(mt5_bars(), now=NOW)
    await trading_tools.start(app)
    assert app.trading.monitor._task is not None
    await trading_tools.stop(app)
    assert app.trading.monitor._task is None and app.trading.mt5.closed


async def test_price_analysis_and_theory_tools(make_app):
    tv = FakeTV(symbol="OANDA:XAUUSD", timeframe="M15", bars=chart_bars())
    app = setup(make_app, tv=tv)
    price = await app.tools.dispatch("get_price", {"symbol": "زێڕ"}, source="text")
    # Spoken the way a trader says it: the Kurdish name and a rounded price (the exact bid stays in data).
    assert price["ok"] and price["summary"].startswith("زێڕ ئێستا لەسەر") and price["data"]["chart_price"]
    assert "." not in price["summary"].split("لەسەر")[1].split()[0]
    result = await app.tools.dispatch("analyze_market", {"symbol": "gold", "draw": "none", "theory": "ئێلیۆت"},
                                      source="text")
    assert result["ok"] and result["data"]["verdict"] in ("WAIT", "NO_TRADE", "SETUP")
    assert is_arabic_script(result["summary"]) and result["data"]["theory"]["status"] == "AVAILABLE"
    info = await app.tools.dispatch("theory_info", {"name": "وایکۆف"}, source="text")
    assert info["ok"] and info["data"]["id"] == "wyckoff" and info["summary"].startswith("وایکۆف")
    catalogue = await app.tools.dispatch("theory_info", {}, source="text")
    assert len(catalogue["data"]["theories"]) == 40
