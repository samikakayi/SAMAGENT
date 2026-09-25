"""Repair review 2026-09-24 (trading-hands-ui lens): symbol resolution and the
user-named check, far prices, alert wicks and restarts, archived strategy
cards, instrument equivalence, per-side S/R, closed-market drawing, clutter,
the broker offset after DST, spoken prices."""

from __future__ import annotations

import time

import pytest
from engine_helpers import FakeMT5Module
from trading_chart_helpers import FakeCdpServer, FakeProc

from sam.brain import conversation as conversation_mod, memory as memory_mod
from sam.trading import chart_tools
from sam.trading.engine.drawplan import build_draw_plan, plan_zones
from sam.trading.engine.sorani import spoken_price, spoken_summary
from sam.trading.symbols import resolve_instrument, same_instrument
from sam.trading.tv_parse import tv_symbol_for


# -- symbols ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("زێڕەکە", "XAUUSD"), ("گۆڵت", "XAUUSD"), ("خاو یوئێزدی", "XAUUSD"), ("XUUSD", "XAUUSD"), ("ZEUR", "XAUUSD"),
    ("ZIW", "XAGUSD"), ("NAWT", "USOIL"), ("Bitkoin", "BTCUSD"), ("altun", "XAUUSD"), ("BINANCE:BTCUSDT", "BTCUSD"),
    ("z", None), ("x", None), ("gol", None), ("شتێکی نەناسراو", None),
])
def test_one_resolver_for_every_trading_tool(text, expected):
    assert resolve_instrument(text) == expected


def test_short_tickers_never_pass_through_and_the_users_feed_is_kept():
    assert tv_symbol_for("z") == (None, "unknown")
    assert tv_symbol_for("گۆڵد", current="BINANCE:BTCUSDT",
                         learned={"XAUUSD": "PEPPERSTONE:XAUUSD"}) == ("PEPPERSTONE:XAUUSD", "learned")
    assert same_instrument("OANDA:NAS100USD", "NAS100") and same_instrument("OANDA:US30USD", "US30")


@pytest.fixture
async def server():
    fake = FakeCdpServer()
    await fake.start()
    yield fake
    await fake.stop()


@pytest.fixture
async def chart_app(make_app, server):
    app = make_app()
    memory_mod.register(app)
    conversation_mod.register(app)
    chart_tools.register(app)
    tv = app.trading.tv
    tv._port = server.port
    tv.proc = FakeProc(pids=[11])
    tv.poll_s, tv.launch_timeout_s, tv.ready_timeout_s, tv.close_timeout_s = 0.02, 2.0, 1.0, 0.5
    yield app
    await chart_tools.stop(app)


def said(app, text):
    cid = app.conversation.ensure_conversation("cascade")
    app.memory.add_turn(cid, "user", text, source="cascade")


async def call(app, name, args, source="cascade"):
    return await app.tools.dispatch(name, args, source=source)


async def test_a_symbol_the_user_never_named_is_not_applied(chart_app, server):
    """Live 2026-09-24: "put it on the 1-minute timeframe" -> tv_set_chart(symbol='z', '1') -> BATS:Z."""
    said(chart_app, "بۆم بخەرە سەر تایمفرێمی یەک خولەکی")
    result = await call(chart_app, "tv_set_chart", {"symbol": "z", "timeframe": "1"})
    assert result["ok"] and server.chart.symbol == "TVC:GOLD" and server.chart.resolution == "1"
    assert result["data"]["symbol_ignored"]["requested"] == "z"
    refused = await call(chart_app, "tv_set_chart", {"symbol": "BTCUSD"})
    assert not refused["ok"] and refused["data"]["error"] == "symbol_not_named" and server.chart.symbol == "TVC:GOLD"
    said(chart_app, "بیتکۆین پیشان بدە")
    assert (await call(chart_app, "tv_set_chart", {"symbol": "BTCUSD"}))["ok"]
    assert server.chart.symbol == "BINANCE:BTCUSDT"


async def test_the_spoken_summary_names_the_instrument_in_kurdish(chart_app, server):
    said(chart_app, "گۆڵد لەسەر پازدە خولەک پیشان بدە")
    result = await call(chart_app, "tv_set_chart", {"symbol": "گۆڵد", "timeframe": "15"})
    assert result["ok"] and "زێڕ" in result["summary"] and "TVC" not in result["summary"]


async def test_prices_far_from_the_chart_are_not_drawn_unless_the_user_said_them(chart_app, server):
    said(chart_app, "هێڵی پشتگیری و بەرگری بکێشە")
    far = await call(chart_app, "draw_on_chart", {"items": [{"kind": "horizontal_line", "points": [{"price": 2650}]}]})
    assert not far["ok"] and far["data"]["far_prices"] == [2650.0]
    said(chart_app, "هێڵێک لەسەر ٢٦٥٠ بکێشە")
    assert (await call(chart_app, "draw_on_chart", {"items": [{"kind": "horizontal_line",
                                                                 "points": [{"price": 2650}]}]}))["ok"]


# -- alerts -------------------------------------------------------------------------------------------------------
class Feed:
    def __init__(self):
        self.price = 0.0
        self.m1 = []

    async def tick(self, symbol):
        return {"symbol": symbol, "bid": self.price, "ask": self.price + 0.2, "spread": 0.2, "time": 0, "digits": 2}

    async def bars(self, symbol, tf, count):
        return self.m1[-count:]


def monitor_app(make_app, clock):
    from sam.trading import tools as trading_tools
    from sam.trading.monitor import Monitor

    app = make_app()
    trading_tools.register(app)
    feed = Feed()
    app.trading.mt5 = feed
    app.trading.monitor = Monitor(app, clock=lambda: clock[0])
    return app, feed


async def test_a_wick_from_before_the_alert_does_not_fire_it(make_app):
    clock = [1_790_000_440.0]                           # 12:00:40 in its minute
    app, feed = monitor_app(make_app, clock)
    feed.price = 2698.5
    feed.m1 = [{"time": 1_790_000_400, "open": 2698.0, "high": 2700.2, "low": 2697.9, "close": 2698.5, "volume": 5}]
    app.trading.monitor.add({"kind": "price_cross", "level": 2700, "direction": "up"}, price=2698.5)
    fired = []
    for price in (2698.6, 2698.7, 2698.8):
        clock[0] += 2
        feed.price = price
        fired += await app.trading.monitor.check_once(clock[0])
    assert fired == []
    clock[0] += 60                                      # a new minute whose wick really touches 2700
    feed.m1.append({"time": 1_790_000_460, "open": 2698.8, "high": 2700.1, "low": 2698.7, "close": 2699.0, "volume": 3})
    feed.price = 2699.0
    fired = await app.trading.monitor.check_once(clock[0])
    assert len(fired) == 1


async def test_a_repeat_alert_does_not_fire_again_after_a_restart(make_app):
    from sam.trading.monitor import Monitor

    clock = [1_790_000_000.0]
    app, feed = monitor_app(make_app, clock)
    feed.price = 2690.0
    alert = app.trading.monitor.add({"kind": "price_cross", "level": 2700, "direction": "up", "repeat": True},
                                    price=2690.0)
    await app.trading.monitor.check_once(clock[0])
    clock[0] += 2
    feed.price = 2701.0
    assert len(await app.trading.monitor.check_once(clock[0])) == 1
    restarted = Monitor(app, clock=lambda: clock[0])    # SAM restarted; price still above the level
    clock[0] += 120
    assert await restarted.check_once(clock[0]) == []
    assert app.trading.monitor.get(alert["id"])["fire_count"] == 1


# -- strategies -------------------------------------------------------------------------------------------------
async def test_a_guessed_id_never_activates_an_archived_card(make_app):
    from sam.trading import tools as trading_tools

    app = make_app()
    trading_tools.register(app)
    store = app.trading.strategies
    store.save({"id": "v1-workflow-strategy", "title_en": "Workflow strategy", "title_ckb": "ستراتیژی",
                "status": "archived", "rules": []})
    assert store.get("strategy") is None and store.get("my gold strategy") is None
    result = await app.tools.dispatch("strategy_save", {"text": "بەڵێ", "strategy_id": "strategy", "status": "active"},
                                      source="text")
    assert not result["ok"] and store.get("v1-workflow-strategy")["status"] == "archived"
    exact = await app.tools.dispatch("strategy_save", {"text": "بەڵێ", "strategy_id": "v1-workflow-strategy",
                                                       "status": "active"}, source="text")
    assert not exact["ok"] and store.get("v1-workflow-strategy")["status"] == "archived"


# -- engine: levels, plans, words ------------------------------------------------------------------------------------
def test_levels_are_ranked_per_side_with_day_anchors():
    from datetime import UTC, datetime

    from sam.trading.engine.analysis import snr_levels
    from sam.trading.engine.types import Candle

    start = datetime(2026, 9, 21, tzinfo=UTC).timestamp()
    candles = []
    for i in range(600):     # days ranging 4300-4360 with many touches above, then one dip to 4244 just now
        base = 4330 + 25 * ((i // 7) % 2) - 12
        low = 4244.5 if i == 599 else base - 3    # too recent to be a confirmed swing: only the day low catches it
        candles.append(Candle(time=datetime.fromtimestamp(start + i * 900, UTC), open=base, high=base + 6, low=low,
                              close=base + 1, volume=10))
    candles.append(Candle(time=datetime.fromtimestamp(start + 600 * 900, UTC), open=4271, high=4273, low=4270,
                          close=4271.7, volume=10))
    levels = snr_levels(candles, "M15")
    assert any(lv.price < 4271.7 for lv in levels)                       # support below price exists
    assert any((lv.metadata or {}).get("anchor", "").endswith("_low") and lv.price < 4250 for lv in levels)


def test_overlapping_zones_merge_and_one_zone_per_side_is_drawn():
    zones = [{"kind": "order_block", "tf": "M5", "low": 4265.79, "high": 4272.76, "direction": "bullish"},
             {"kind": "order_block", "tf": "M1", "low": 4264.99, "high": 4267.17, "direction": "bullish"},
             {"kind": "fvg", "tf": "M1", "low": 4266.0, "high": 4267.0, "direction": "bullish"},
             {"kind": "order_block", "tf": "M5", "low": 4290.0, "high": 4295.0, "direction": "bearish"}]
    planned = plan_zones(zones, 4271.0)
    assert len(planned) == 2 and {z["direction"] for z in planned} == {"bullish", "bearish"}
    report = {"price": 4271.0, "zones": zones, "resistance": [{"price": 4285.52, "tf": "H1"}, {"price": 4287.12, "tf": "M15"}],
              "support": [], "indicators": {"M15": {"atr14": 8.0}}, "verdict": "WAIT"}
    items = build_draw_plan(report, "full")
    lines = [i for i in items if i["kind"] == "horizontal_line"]
    boxes = [i for i in items if i["kind"] == "rectangle"]
    assert len(lines) == 1                                                # 1.6 apart < 0.3 ATR: one line
    assert len(boxes) == 2 and all(b["style"]["extend_right"] for b in boxes)
    assert {b["style"]["label_valign"] for b in boxes} == {"top", "bottom"}


def test_spoken_prices_are_rounded_and_the_closed_market_names_levels():
    assert spoken_price(4269.81) == "4270" and spoken_price(1.085431) == "1.0854" and spoken_price(152.36) == "152.4"
    text = spoken_summary({"symbol": "XAUUSD", "price": 4271.44, "market_closed": True, "stale": True,
                           "resistance": [{"price": 4285.5}], "support": [{"price": 4244.49}]})
    assert "داخراوە" in text and "4286" in text and "4244" in text and "دوای کرانەوەی بازاڕ دەکەم" not in text


# -- MT5 broker offset -----------------------------------------------------------------------------------------------
async def test_the_broker_offset_is_reverified_after_a_dst_switch():
    from sam.trading.mt5 import MT5Feed

    now = [time.time()]
    module = FakeMT5Module(now=now[0], offset=10800)
    feed = MT5Feed(None, module_loader=lambda: module, is_terminal_running=lambda: True, clock=lambda: now[0])
    assert await feed.connect() and feed.broker_offset_s == 10800
    now[0] += 2 * 86400
    module.now, module.offset = now[0], 7200            # the broker moved to UTC+2 while SAM kept running
    tick = await feed.tick("XAUUSD")
    assert feed.broker_offset_s == 7200 and abs(now[0] - tick["time"]) < 5
    await feed.close()
