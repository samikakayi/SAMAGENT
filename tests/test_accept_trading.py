"""Acceptance fixes 2026-09-24 (trading): real tickers are never "corrected"
into another market, drawing re-adoption needs the same kind and a label, the
user's own TradingView feed is never learned from SAM's own switch, and a
frozen MT5 feed neither shifts nor draws levels on a live chart."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from trading_chart_helpers import FakeCdpServer, FakeProc

from sam.trading.analyze import NOT_DRAWN_CKB, draw_report, feed_offset, stale_open_mt5, stale_reason
from sam.trading.symbols import resolve_instrument
from sam.trading.tradingview import TradingViewBridge
from sam.trading.tv_feeds import LEARNED_KEY, SAM_SET_KEY, FeedMemory
from sam.trading.tv_owner import match_shapes, same_kind
from sam.trading.tv_parse import tv_symbol_for


# -- symbols (verify3/us500_probe.py: US500 and UK100 answered with the Nasdaq price) -----------------------------
@pytest.mark.parametrize("ticker", ["US500", "UK100", "UKOIL", "GER40", "SPX500", "USDT", "ETCUSD", "LTCUSD",
                                    "EURGBP", "JP225"])
def test_real_tickers_are_never_read_as_another_market(ticker):
    assert resolve_instrument(ticker) not in ("NAS100", "USOIL", "USDX", "XAUUSD", "BTCUSD", "ETHUSD")
    assert tv_symbol_for(ticker)[0] not in ("OANDA:NAS100USD", "TVC:USOIL")


@pytest.mark.parametrize("text,expected", [("XUUSD", "XAUUSD"), ("goldd", "XAUUSD"), ("nasdak", "NAS100"),
                                           ("US100", "NAS100"), ("silverr", "XAGUSD"), ("GBPUSX", "GBPUSD")])
def test_near_misses_of_names_are_still_understood(text, expected):
    assert resolve_instrument(text) == expected


# -- drawing re-adoption --------------------------------------------------------------------------------------
def test_readoption_needs_the_same_kind_and_a_label():
    user_ray = {"id": "userX", "name": "horizontal_ray", "text": "", "points": [{"price": 4285.52}]}
    unlabelled = {"db_id": 1, "kind": "horizontal_line", "text": "", "points": [{"price": 4285.52}]}
    assert match_shapes([unlabelled], [user_ray], {"old1"}) == {}                  # verify review's case
    labelled = {"db_id": 2, "kind": "horizontal_line", "text": "بەرگری M15", "points": [{"price": 4285.52}]}
    user_line = {"id": "userY", "name": "horizontal_ray", "text": "بەرگری M15", "points": [{"price": 4285.52}]}
    assert match_shapes([labelled], [user_line], set()) == {}                     # another kind
    own = {"id": "new7", "name": "LineToolHorzLine", "text": "بەرگری M15", "points": [{"price": 4285.52}]}
    assert match_shapes([labelled], [user_line, own], set()) == {2: "new7"}
    assert not same_kind("horizontal_line", "") and same_kind("rectangle", "rectangle")


# -- the user's own feed ---------------------------------------------------------------------------------------
def test_a_symbol_sam_set_is_not_learned_after_a_restart(make_app):
    config = make_app().config
    FeedMemory(config, resolve_instrument).note_sam_set("OANDA:XAUUSD")        # SAM switched the chart at 17:19
    after_restart = FeedMemory(config, resolve_instrument)
    assert after_restart.observe("OANDA:XAUUSD") is None                       # first look: still SAM's choice
    assert (config.get(LEARNED_KEY, {}) or {}).get("XAUUSD") is None
    assert after_restart.observe("PEPPERSTONE:XAUUSD") == "XAUUSD"             # the user changed it himself
    assert config.get(LEARNED_KEY)["XAUUSD"] == "PEPPERSTONE:XAUUSD"
    assert after_restart.observe("OANDA:XAUUSD") == "XAUUSD"                   # ... and back, by hand
    assert config.get(LEARNED_KEY)["XAUUSD"] == "OANDA:XAUUSD"
    assert "OANDA:XAUUSD" not in (config.get(SAM_SET_KEY) or {})


def test_sams_own_default_feed_learned_by_the_old_code_is_dropped_once(make_app):
    from sam.trading.tv_feeds import VERSION_KEY
    from sam.trading.tv_parse import tv_symbol_for

    config = make_app().config
    config.set(LEARNED_KEY, {"XAUUSD": "OANDA:XAUUSD", "BTCUSD": "COINBASE:BTCUSD"})   # the evening's DB
    memory = FeedMemory(config, resolve_instrument)
    assert config.get(LEARNED_KEY) == {"BTCUSD": "COINBASE:BTCUSD"} and config.get(VERSION_KEY) == 2
    assert memory.observe("PEPPERSTONE:XAUUSD") == "XAUUSD"                     # the user's own gold chart
    assert tv_symbol_for("گۆڵد", current="BINANCE:BTCUSDT", learned=config.get(LEARNED_KEY))[0] \
        == "PEPPERSTONE:XAUUSD"
    config.set(LEARNED_KEY, {"XAUUSD": "OANDA:XAUUSD"})                          # learned by hand, later
    FeedMemory(config, resolve_instrument)
    assert config.get(LEARNED_KEY) == {"XAUUSD": "OANDA:XAUUSD"}                 # the repair ran only once


def test_the_users_chart_at_start_is_learned(make_app):
    config = make_app().config
    assert FeedMemory(config, resolve_instrument).observe("PEPPERSTONE:XAUUSD") == "XAUUSD"
    assert config.get(LEARNED_KEY) == {"XAUUSD": "PEPPERSTONE:XAUUSD"}


@pytest.fixture
async def server():
    fake = FakeCdpServer()
    await fake.start()
    yield fake
    await fake.stop()


async def test_the_bridge_remembers_its_own_switch_across_a_restart(make_app, server):
    app = make_app()
    first = TradingViewBridge(app, port=server.port, proc=FakeProc(pids=[11]))
    try:
        result = await first.set_symbol("OANDA:XAUUSD")
        assert result["ok"] and server.chart.symbol == "OANDA:XAUUSD"
    finally:
        await first.close()
    second = TradingViewBridge(app, port=server.port, proc=FakeProc(pids=[11]))
    try:
        await second.chart_state()
        assert (app.config.get(LEARNED_KEY, {}) or {}).get("XAUUSD") != "OANDA:XAUUSD"
    finally:
        await second.close()
        server.chart.symbol = "TVC:GOLD"


# -- a frozen MT5 feed ----------------------------------------------------------------------------------------------
def _chart_bars(close: float, now: float) -> list[dict]:
    return [{"time": int(now) - 900 * i, "open": close, "high": close + 1, "low": close - 1, "close": close}
            for i in range(20, -1, -1)]


def test_the_feed_offset_is_refused_when_mt5_is_frozen_or_far_off():
    now = time.time()
    bars = _chart_bars(4300.0, now)
    offset, refused = feed_offset(bars, {"bid": 4299.4, "time": now - 5})
    assert refused is None and round(offset, 2) == 0.6
    offset, refused = feed_offset(bars, {"bid": 4250.0, "time": now - 7200})      # MT5 froze 2 h ago
    assert refused and round(offset, 1) == 50.0
    assert feed_offset(bars, {"bid": 4000.0, "time": now})[1]                     # 7%: not a feed difference
    weekend = _chart_bars(4300.0, now - 2 * 86400)                                 # both stopped together
    assert feed_offset(weekend, {"bid": 4299.4, "time": now - 2 * 86400 + 60})[1] is None


async def test_a_stale_mt5_feed_on_an_open_market_is_not_drawn():
    report = {"symbol": "XAUUSD", "price": 4300.0, "stale": True, "market_closed": False,
              "sources": {"M15": "tradingview", "H1": "mt5"}, "verdict": "WAIT",
              "resistance": [{"price": 4310.0, "tf": "H1"}], "support": [{"price": 4290.0, "tf": "H1"}], "zones": []}
    assert stale_open_mt5(report)
    app = SimpleNamespace(trading=SimpleNamespace(tv=object()))
    result = await draw_report(app, report, "levels", {"matches": True}, "analysis:1", None)
    assert result["reason"] == "stale_feed" and result["drawn"] == [] and result["dry_run"]
    assert "مێتاتڕەیدەر" in NOT_DRAWN_CKB["stale_feed"]
    closed = {**report, "market_closed": True}
    assert not stale_open_mt5(closed)                                  # a closed market's levels are still drawn
    assert not stale_open_mt5({**report, "sources": {"M15": "tradingview"}})


async def test_a_frozen_chart_is_named_as_the_stale_feed_not_mt5():
    """Live 2026-09-24: the chart's M1 series was 7.8 h old while MT5 ticked."""
    report = {"symbol": "XAUUSD", "price": 4266.0, "stale": True, "market_closed": False,
              "sources": {"M1": "tradingview", "M15": "mt5", "H1": "mt5"}, "stale_timeframes": ["M1"],
              "verdict": "WAIT", "resistance": [{"price": 4268.0, "tf": "M15"}], "support": [], "zones": []}
    assert stale_reason(report) == "stale_chart"
    app = SimpleNamespace(trading=SimpleNamespace(tv=object()))
    result = await draw_report(app, report, "levels", {"matches": True}, "analysis:1", None)
    assert result["reason"] == "stale_chart" and result["drawn"] == []
    assert "ترەیدینگ ڤیو" in NOT_DRAWN_CKB["stale_chart"]
    assert stale_reason({**report, "stale_timeframes": ["M1", "H1"]}) == "stale_feed"
    assert stale_reason({**report, "market_closed": True}) is None
