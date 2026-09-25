"""MT5 feed: broker offset, symbol map, read-only proxy, UTC conversion, reconnect."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from engine_helpers import FakeMT5Module

from sam.trading.mt5 import MT5Feed, ReadOnlyMT5, compute_broker_offset, rank_broker_symbols

NOW = 1_790_000_000.0


def test_offset_is_three_hours_today_and_two_in_winter():
    assert compute_broker_offset([NOW + 10800 + 1.4], NOW) == (10800, True)
    assert compute_broker_offset([NOW + 7200 - 2.0], NOW) == (7200, True)
    # The newest of several symbols wins (a quiet symbol's older tick is ignored).
    assert compute_broker_offset([NOW + 10800 - 400, NOW + 10800 - 0.5], NOW) == (10800, True)


def test_a_frozen_weekend_tick_is_not_trusted_as_the_clock():
    offset, verified = compute_broker_offset([NOW + 10800 - 3000], NOW)
    assert verified is False
    assert compute_broker_offset([], NOW) == (None, False)
    assert compute_broker_offset([NOW - 3 * 86400], NOW) == (None, False)


def test_broker_symbols_rank_exact_then_suffixed():
    names = [("XAUUSD.m.e", True), ("XAUUSD.crp", False), ("XAUUSD", True), ("EURUSD", True)]
    assert rank_broker_symbols(names, "XAUUSD")[:3] == ["XAUUSD", "XAUUSD.m.e", "XAUUSD.crp"]
    assert rank_broker_symbols([("BTCUSD.crp", True)], "BTCUSD") == ["BTCUSD.crp"]


def test_the_proxy_exposes_market_data_only():
    fake = FakeMT5Module(now=NOW)
    proxy = ReadOnlyMT5(fake)
    assert proxy.TIMEFRAME_M15 == 15 and proxy.symbol_info("XAUUSD").name == "XAUUSD"
    for forbidden in ("order_send", "order_check", "positions_get", "account_info"):
        with pytest.raises(AttributeError):
            getattr(proxy, forbidden)
    assert fake.orders == []


def test_no_order_function_is_called_anywhere_in_the_trading_package():
    root = Path(__file__).resolve().parent.parent / "sam" / "trading"
    pattern = re.compile(r"\.(order_send|order_check|order_calc_margin|positions_get|position_close)\s*\(")
    offenders = [str(p) for p in root.rglob("*.py") if pattern.search(p.read_text(encoding="utf-8"))]
    assert offenders == []


def feed_for(fake: FakeMT5Module, **kwargs) -> MT5Feed:
    return MT5Feed(None, module_loader=lambda: fake, is_terminal_running=lambda: True, clock=lambda: fake.now, **kwargs)


async def test_bars_and_ticks_come_out_in_true_utc():
    fake = FakeMT5Module(now=NOW, offset=10800)
    feed = feed_for(fake)
    try:
        assert await feed.connect() is True
        status = await feed.status()
        assert status["broker_offset_s"] == 10800 and status["server_time_ok"] is True
        assert "XAUUSD.m.e" in status["symbols"]
        bars = await feed.bars("TVC:GOLD", "M15", 50)
        assert len(bars) == 50
        assert bars[-1]["time"] <= NOW < bars[-1]["time"] + 900  # the forming bar contains 'now'
        assert bars[-1]["volume"] == 59.0  # tick volume when real volume is 0
        tick = await feed.tick("OANDA:XAUUSD")
        assert tick["symbol"] == "XAUUSD" and abs(tick["time"] - (NOW - 1.0)) < 0.01
        assert tick["spread"] == pytest.approx(0.2)
    finally:
        await feed.close()


async def test_an_explicit_broker_symbol_is_kept():
    feed = feed_for(FakeMT5Module(now=NOW))
    try:
        assert await feed.resolve_symbol("XAUUSD.m.e") == "XAUUSD.m.e"
        assert await feed.resolve_symbol("زێڕ") == "XAUUSD"
        assert await feed.resolve_symbol("NOPE123") is None
    finally:
        await feed.close()


async def test_one_session_is_reused_and_an_ipc_failure_reconnects_once():
    fake = FakeMT5Module(now=NOW)
    feed = feed_for(fake)
    try:
        await feed.connect()
        await feed.bars("XAUUSD", "M5", 20)
        await feed.bars("XAUUSD", "H1", 20)
        assert fake.initialized == 1
        fake.fail_next_rates = 1
        bars = await feed.bars("XAUUSD", "M1", 20)
        assert len(bars) == 20 and fake.initialized == 2
    finally:
        await feed.close()


async def test_a_closed_terminal_is_not_launched():
    fake = FakeMT5Module(now=NOW)
    feed = MT5Feed(None, module_loader=lambda: fake, is_terminal_running=lambda: False, clock=lambda: NOW)
    try:
        assert await feed.connect() is False
        assert fake.initialized == 0
        with pytest.raises(Exception, match="not running"):
            await feed.bars("XAUUSD", "M5", 10)
    finally:
        await feed.close()


async def test_a_frozen_clock_uses_the_saved_offset(make_app):
    app = make_app()
    app.config.set("trading.mt5_offset_s", 10800)
    fake = FakeMT5Module(now=NOW, offset=10800, tick_age=3000)  # market closed: last tick 50 min old
    feed = MT5Feed(app, module_loader=lambda: fake, is_terminal_running=lambda: True, clock=lambda: NOW)
    try:
        assert await feed.connect() is True
        assert feed.broker_offset_s == 10800 and feed.offset_source == "saved" and not feed.offset_verified
    finally:
        await feed.close()
