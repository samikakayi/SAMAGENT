"""tv_set_chart never applies a symbol that is not a known instrument, and the
TradingView restart follows the user's full-authority setting.

Live session 2026-09-25: «بڕۆ 100 چار 3 خولەکی یەکسەر لە گوڵت.» ("go to the gold
chart, 3 minutes"; STT wrote "100") -> tv_set_chart(symbol='100', timeframe=3m)
and the user's chart switched to a symbol named "100" (the "user named it" check
passed because "100" was in the text)."""

from __future__ import annotations

import pytest
from trading_chart_helpers import FakeCdpServer, FakeProc

from sam.brain import conversation as conversation_mod, memory as memory_mod
from sam.brain.confirm import AUTHORITY_KEY, bind_authority
from sam.events import ConfirmRequest
from sam.textnorm import is_arabic_script
from sam.trading import chart_tools


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


async def test_a_number_from_speech_to_text_uses_the_instrument_the_user_named(chart_app, server):
    server.chart.symbol = "BINANCE:BTCUSDT"
    said(chart_app, "بڕۆ 100 چار 3 خولەکی یەکسەر لە گوڵت.")
    result = await call(chart_app, "tv_set_chart", {"symbol": "100", "timeframe": "3"})
    assert result["ok"], result
    assert server.chart.symbol != "100" and "XAUUSD" in server.chart.symbol and server.chart.resolution == "3"
    assert result["data"]["symbol_from_words"] == {"requested": "100", "used": "XAUUSD"}
    assert "زێڕ" in result["summary"] and "سێ خولەک" in result["summary"]


async def test_an_unknown_symbol_keeps_the_chart_and_asks_which_one(chart_app, server):
    said(chart_app, "بڕۆ 100 چار 3 خولەکی یەکسەر")
    result = await call(chart_app, "tv_set_chart", {"symbol": "100", "timeframe": "3"})
    assert result["ok"] and server.chart.symbol == "TVC:GOLD" and server.chart.resolution == "3"
    assert result["data"]["symbol_ignored"]["requested"] == "100"
    assert "مەبەستت زێڕە؟" in result["summary"] and is_arabic_script(result["summary"])
    refused = await call(chart_app, "tv_set_chart", {"symbol": "100"})
    assert not refused["ok"] and refused["data"]["error"] == "unknown_symbol" and "مەبەستت" in refused["summary"]
    assert server.chart.symbol == "TVC:GOLD"


@pytest.mark.parametrize("symbol", ["100", "z", "ab", "banana", "شتێکی نەناسراو", "AAPL"])
async def test_numbers_letters_and_unknown_words_are_never_applied(chart_app, server, symbol):
    said(chart_app, f"{symbol} پیشان بدە")
    result = await call(chart_app, "tv_set_chart", {"symbol": symbol})
    assert not result["ok"] and server.chart.symbol == "TVC:GOLD"


async def test_the_flower_word_is_gold_on_the_chart(chart_app, server):
    server.chart.symbol = "BINANCE:BTCUSDT"
    said(chart_app, "کڕۆکڕۆک قەشمەر بڕۆ سەر چاوتی گوڵ")
    result = await call(chart_app, "tv_set_chart", {"symbol": "گوڵ"})
    assert result["ok"], result
    assert "XAUUSD" in server.chart.symbol and "زێڕ" in result["summary"]


async def test_the_users_learned_gold_feed_is_used(chart_app, server):
    chart_app.config.set("trading.tv_learned_symbols", {"XAUUSD": "PEPPERSTONE:XAUUSD"})
    server.chart.symbol = "BINANCE:BTCUSDT"
    said(chart_app, "بڕۆ سەر چارتی گوڵت")
    result = await call(chart_app, "tv_set_chart", {"symbol": "گوڵت"})
    assert result["ok"] and server.chart.symbol == "PEPPERSTONE:XAUUSD"


# --- full authority: the TradingView restart for its port -----------------------------------------------------------
def port_less(app, server):
    server.online = False
    proc = app.trading.tv.proc
    proc.pids = [11, 12]
    proc.on_activate = lambda: setattr(server, "online", True)
    return proc


async def test_with_full_authority_tradingview_is_restarted_without_a_question(chart_app, server):
    bind_authority(chart_app)                                  # sam.brain.persona.register does this in SAM
    assert chart_app.config.get(AUTHORITY_KEY) is True          # the default
    proc = port_less(chart_app, server)
    questions: list[str] = []
    chart_app.bus.subscribe(ConfirmRequest, lambda ev: questions.append(ev.question_ckb))
    result = await call(chart_app, "tv_open", {})
    assert result["ok"] and questions == [] and proc.closed == [[11, 12]]
    assert "دووبارە کردمەوە" in result["summary"]                 # it says what it did, in one clause
    assert result["data"]["acted_without_asking"] is True
    rows = chart_app.db.query("SELECT * FROM activity WHERE kind='confirm' AND source='authority'")
    assert rows and rows[-1]["name"] == "tv_open"


async def test_a_chart_change_that_restarted_tradingview_says_so(chart_app, server):
    bind_authority(chart_app)
    port_less(chart_app, server)
    said(chart_app, "گۆڵد لەسەر پازدە خولەک پیشان بدە")
    result = await call(chart_app, "tv_set_chart", {"symbol": "گۆڵد", "timeframe": "15"})
    assert result["ok"], result
    assert result["summary"].startswith(chart_tools.RESTARTED_NOTE_CKB) and result["data"]["acted_without_asking"]


async def test_without_full_authority_the_restart_is_asked(chart_app, server):
    bind_authority(chart_app)
    chart_app.config.set(AUTHORITY_KEY, False)
    proc = port_less(chart_app, server)
    questions: list[str] = []

    def on_request(event: ConfirmRequest) -> None:
        questions.append(event.question_ckb)
        chart_app.confirm.resolve(event.confirm_id, False)

    chart_app.bus.subscribe(ConfirmRequest, on_request)
    declined = await call(chart_app, "tv_open", {})
    assert not declined["ok"] and questions and proc.closed == []
