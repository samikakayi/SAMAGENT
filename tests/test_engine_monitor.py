"""Market monitor on a synthetic feed: fires once, never twice, catches wicks,
honours candle closes/expiry/repeat, speaks Sorani, and the alert tools."""

from __future__ import annotations

import pytest
from engine_helpers import FakeFeed, flat_bars

from sam.events import Alert, SpeakRequest
from sam.textnorm import is_arabic_script
from sam.trading import tools as trading_tools
from sam.trading.monitor import AlertError, Monitor

T0 = 1_790_000_000.0  # Monday 14:13 UTC


def setup_app(make_app, feed: FakeFeed):
    app = make_app()
    trading_tools.register(app)
    app.trading.mt5 = feed
    app.trading.monitor = Monitor(app, clock=lambda: T0)
    events: list = []
    app.bus.subscribe((Alert, SpeakRequest), events.append)
    return app, events


def m1(price: float, *, end: float, high: float | None = None, low: float | None = None) -> list[dict]:
    bars = flat_bars(3, "M1", end=end, price=price, spread=0.1)
    bars[-1] = {**bars[-1], "high": high if high is not None else price + 0.1, "low": low if low is not None else price - 0.1}
    return bars


async def test_a_price_cross_fires_once_and_is_spoken(make_app):
    feed = FakeFeed({"M1": m1(2690.0, end=T0)}, price=2690.0)
    app, events = setup_app(make_app, feed)
    alert = app.trading.monitor.add({"kind": "price_cross", "symbol": "gold", "level": 2700, "direction": "up"}, price=2690.0)
    assert alert["symbol"] == "XAUUSD" and alert["status"] == "active"
    assert await app.trading.monitor.check_once(now=T0 + 2) == []
    feed.price = 2701.0
    feed.bars_by_tf["M1"] = m1(2701.0, end=T0 + 4)
    fired = await app.trading.monitor.check_once(now=T0 + 4)
    assert len(fired) == 1 and "2700" in fired[0]["text_ckb"] and is_arabic_script(fired[0]["text_ckb"])
    assert [type(e).__name__ for e in events] == ["Alert", "SpeakRequest"]
    assert events[1].source == "alert" and events[1].text_ckb.startswith("ئاگاداری: زێڕ")
    row = app.trading.monitor.get(alert["id"])
    assert row["status"] == "fired" and row["fire_count"] == 1 and row["last_value"] == 2701.0
    assert await app.trading.monitor.check_once(now=T0 + 6) == []
    assert app.db.query("SELECT kind FROM activity WHERE kind='alert'")


async def test_a_wick_between_polls_is_caught_but_old_prices_are_not(make_app):
    feed = FakeFeed({"M1": m1(2690.0, end=T0, high=2705.0)}, price=2690.0)  # the high happened before the alert
    app, _ = setup_app(make_app, feed)
    app.trading.monitor.add({"kind": "price_cross", "level": 2700, "direction": "up"}, price=2690.0)
    assert await app.trading.monitor.check_once(now=T0 + 1) == [], "a wick older than the alert must not fire it"
    feed.bars_by_tf["M1"] = m1(2694.0, end=T0 + 61, high=2702.5)
    feed.price = 2694.0
    fired = await app.trading.monitor.check_once(now=T0 + 62)
    assert len(fired) == 1, "price touched 2700 between two polls (M1 high 2702.5)"


async def test_already_beyond_the_level_waits_for_a_real_cross(make_app):
    feed = FakeFeed({"M1": m1(2710.0, end=T0)}, price=2710.0)
    app, _ = setup_app(make_app, feed)
    app.trading.monitor.add({"kind": "price_cross", "level": 2700, "direction": "up"}, price=2710.0)
    assert await app.trading.monitor.check_once(now=T0 + 1) == []
    feed.price = 2695.0
    feed.bars_by_tf["M1"] = m1(2695.0, end=T0 + 61)
    assert await app.trading.monitor.check_once(now=T0 + 62) == []
    feed.price = 2700.5
    feed.bars_by_tf["M1"] = m1(2700.5, end=T0 + 121)
    assert len(await app.trading.monitor.check_once(now=T0 + 122)) == 1


async def test_repeat_alerts_rearm_after_moving_away_and_respect_the_cooldown(make_app):
    feed = FakeFeed({"M1": m1(2690.0, end=T0)}, price=2690.0)
    app, _ = setup_app(make_app, feed)
    alert = app.trading.monitor.add({"kind": "price_cross", "level": 2700, "direction": "up", "repeat": True}, price=2690.0)

    async def at(price: float, now: float) -> int:
        feed.price = price
        feed.bars_by_tf["M1"] = m1(price, end=now)
        return len(await app.trading.monitor.check_once(now=now))

    assert await at(2701.0, T0 + 10) == 1
    assert await at(2702.0, T0 + 20) == 0          # still above: no double fire
    assert await at(2699.0, T0 + 80) == 0          # moved away below -> re-armed
    assert await at(2700.2, T0 + 140) == 1         # second cross, after the 60 s cooldown
    assert app.trading.monitor.get(alert["id"])["status"] == "active"
    assert app.trading.monitor.get(alert["id"])["fire_count"] == 2


async def test_expired_alerts_never_fire(make_app):
    feed = FakeFeed({"M1": m1(2690.0, end=T0)}, price=2690.0)
    app, events = setup_app(make_app, feed)
    alert = app.trading.monitor.add({"kind": "price_cross", "level": 2700, "expires_in_hours": 1}, price=2690.0)
    feed.price = 2705.0
    assert await app.trading.monitor.check_once(now=T0 + 7200) == []
    assert app.trading.monitor.get(alert["id"])["status"] == "expired" and events == []


def closes(values: list[float], *, tf: str, end: float) -> list[dict]:
    bars = flat_bars(len(values), tf, end=end, price=values[0])
    return [{**b, "open": v, "high": v + 0.3, "low": v - 0.3, "close": v} for b, v in zip(bars, values)]


async def test_a_candle_close_cross_fires_once_per_bar(make_app):
    # bars: ..., prior close 2698, closed 2701 (closes after the alert), forming bar
    end = T0 + 900
    feed = FakeFeed({"M15": closes([2695.0] * 20 + [2698.0, 2701.0, 2701.5], tf="M15", end=end)}, price=2701.5)
    app, _ = setup_app(make_app, feed)
    app.trading.monitor.add({"kind": "candle_close", "level": 2700, "timeframe": "15m"}, price=2699.0)
    fired = await app.trading.monitor.check_once(now=end + 5)
    assert len(fired) == 1 and "پازدە خولەک" in fired[0]["text_ckb"] and "داخرا" in fired[0]["text_ckb"]
    assert await app.trading.monitor.check_once(now=end + 10) == []


async def test_a_cross_that_closed_before_the_alert_existed_is_ignored(make_app):
    end = T0 - 3600  # the crossing bar closed an hour before the alert
    feed = FakeFeed({"M15": closes([2695.0] * 20 + [2698.0, 2701.0, 2701.5], tf="M15", end=end)}, price=2701.5)
    app, _ = setup_app(make_app, feed)
    app.trading.monitor.add({"kind": "candle_close", "level": 2700, "timeframe": "M15"}, price=2701.5)
    assert await app.trading.monitor.check_once(now=T0 + 5) == []


async def test_zone_touch_and_volume_spike(make_app):
    volume_bars = flat_bars(25, "M5", end=T0 + 300, price=2690.0, volume=100.0)
    volume_bars[-2] = {**volume_bars[-2], "volume": 320.0}  # the last CLOSED bar
    feed = FakeFeed({"M1": m1(2690.0, end=T0), "M5": volume_bars}, price=2690.0)
    app, _ = setup_app(make_app, feed)
    app.trading.monitor.add({"kind": "zone_touch", "low": 2685, "high": 2680}, price=2690.0)
    app.trading.monitor.add({"kind": "volume_spike", "timeframe": "M5", "k": 3, "n": 20})
    fired = await app.trading.monitor.check_once(now=T0 + 301)
    assert [f["kind"] for f in fired] == ["volume_spike"] and "تیک ڤۆلیۆم" in fired[0]["text_ckb"]
    feed.price = 2684.0
    feed.bars_by_tf["M1"] = m1(2684.0, end=T0 + 400)
    fired = await app.trading.monitor.check_once(now=T0 + 401)
    assert [f["kind"] for f in fired] == ["zone_touch"] and "2680" in fired[0]["text_ckb"]


async def test_a_strategy_state_alert_speaks_when_its_rules_become_true(make_app):
    feed = FakeFeed({"M5": flat_bars(200, "M5", end=T0, price=2690.0)}, price=2690.0, now=T0)
    app, _ = setup_app(make_app, feed)
    card = app.trading.strategies.save({"title_ckb": "شکانی ٢٧٠٠", "title_en": "Break 2700", "status": "active",
                                        "timeframes": {"entry": "M5"}, "rules": [
                                            {"id": "r1", "kind": "trigger", "text_ckb": "نرخ لە سەرووی 2700",
                                             "check": {"predicate": "price_above", "params": {"level": 2700}}}]})
    app.trading.monitor.add({"kind": "strategy_state", "strategy_id": card["id"]})
    assert await app.trading.monitor.check_once(now=T0 + 1) == []
    feed.price = 2705.0
    feed.bars_by_tf["M5"] = flat_bars(200, "M5", end=T0 + 61, price=2705.0)
    fired = await app.trading.monitor.check_once(now=T0 + 62)
    assert len(fired) == 1 and "شکانی ٢٧٠٠" in fired[0]["text_ckb"]
    assert await app.trading.monitor.check_once(now=T0 + 130) == []


async def test_invalid_specs_are_refused(make_app):
    app, _ = setup_app(make_app, FakeFeed({"M1": m1(1.0, end=T0)}, price=1.0))
    with pytest.raises(AlertError):
        app.trading.monitor.add({"kind": "price_cross"})
    with pytest.raises(AlertError):
        app.trading.monitor.add({"kind": "teleport", "level": 1})
    with pytest.raises(AlertError):
        app.trading.monitor.add({"kind": "strategy_state", "strategy_id": "nope"})


async def test_alert_tools_set_list_and_cancel(make_app):
    feed = FakeFeed({"M1": m1(2690.0, end=T0)}, price=2690.0)
    app, _ = setup_app(make_app, feed)
    result = await app.tools.dispatch("set_alert", {"kind": "price_cross", "symbol": "زێڕ", "level": "2700"}, source="text")
    assert result["ok"] and result["data"]["symbol"] == "XAUUSD" and result["data"]["price_now"] == 2690.0
    assert result["summary"].startswith("ئاگادارکردنەوەکە دانرا") and "2700" in result["summary"]
    listed = await app.tools.dispatch("list_alerts", {}, source="text")
    assert listed["data"]["alerts"][0]["id"] == result["data"]["alert_id"]
    cancelled = await app.tools.dispatch("cancel_alert", {"alert_id": "all"}, source="text")
    assert cancelled["ok"] and cancelled["data"]["cancelled"] == 1
    assert (await app.tools.dispatch("cancel_alert", {"alert_id": "7"}, source="text"))["ok"] is False


async def test_a_tradingview_sourced_alert_uses_the_chart_price(make_app):
    from engine_helpers import FakeTV
    feed = FakeFeed({"M1": m1(2600.0, end=T0)}, price=2600.0)  # MT5 would never reach the level
    app, _ = setup_app(make_app, feed)
    chart = flat_bars(30, "M1", end=T0, price=2690.0)
    app.trading.tv = FakeTV(symbol="OANDA:XAUUSD", timeframe="M1", bars=chart)
    app.trading.monitor.add({"kind": "price_cross", "level": 2700, "direction": "up", "source": "tradingview"},
                            price=2690.0)
    assert await app.trading.monitor.check_once(now=T0 + 1) == []
    app.trading.tv._bars = flat_bars(30, "M1", end=T0 + 60, price=2701.0)
    fired = await app.trading.monitor.check_once(now=T0 + 61)
    assert len(fired) == 1 and "2701" in fired[0]["text_ckb"]


async def test_a_level_with_decimals_is_followed_by_the_price_at_the_same_precision(make_app):
    """Live run 2026-09-24: «گەیشتە سەرووی 4265.36؛ نرخی ئێستا 4265» sounded like a contradiction."""
    feed = FakeFeed({"M1": m1(2700.0, end=T0)}, price=2700.0)
    app, events = setup_app(make_app, feed)
    app.trading.monitor.add({"kind": "price_cross", "symbol": "gold", "level": 2700.35, "direction": "up"},
                            price=2700.0)
    feed.price = 2700.41
    feed.bars_by_tf["M1"] = m1(2700.41, end=T0 + 4)
    fired = await app.trading.monitor.check_once(now=T0 + 4)
    assert len(fired) == 1 and "2700.35" in fired[0]["text_ckb"] and "2700.41" in fired[0]["text_ckb"]
