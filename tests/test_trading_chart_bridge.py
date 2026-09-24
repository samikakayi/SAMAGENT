"""Chart bridge end to end against the fake DevTools server: injection, chart state,
symbol/timeframe changes, bars, drawing ownership, reconnects, screenshots, start/restart."""

from __future__ import annotations

import pytest
from trading_chart_helpers import FakeCdpServer, FakeProc, ProtocolFail

from sam.trading.tradingview import CONFIRM_RESTART_CKB, TradingViewBridge, TvError

AUMID = "TradingView.Desktop_n534cwy3pjxzj!TradingView.Desktop"


@pytest.fixture
async def server():
    fake = FakeCdpServer()
    await fake.start()
    yield fake
    await fake.stop()


def make_bridge(app, server, proc=None) -> TradingViewBridge:
    tv = TradingViewBridge(app, port=server.port, proc=proc or FakeProc(pids=[11, 12]))
    tv.poll_s, tv.launch_timeout_s, tv.ready_timeout_s, tv.close_timeout_s = 0.02, 2.0, 1.0, 0.5
    return tv


@pytest.fixture
async def tv(make_app, server):
    app = make_app()
    bridge = make_bridge(app, server)
    yield bridge
    await bridge.close()


def owned(tv) -> list[dict]:
    return tv.app.db.query("SELECT * FROM drawings ORDER BY id")


async def test_connect_injects_the_bundle_once_per_page(tv, server):
    assert await tv.connect()
    await tv.chart_state()
    await tv.bars(50)
    await tv.draw("horizontal_line", [{"price": 4260}], text="SAM test")
    assert server.chart.injections == 1 and tv.injections == 1 and server.ws_opened == 1
    assert tv.connected


async def test_chart_state_follows_the_contract(tv, server):
    state = await tv.chart_state()
    last = server.chart.bars[-1]
    assert state["symbol"] == "TVC:GOLD" and state["canonical"] == "XAUUSD"
    assert state["timeframe"] == "M1" and state["resolution"] == "1" and state["timeframe_ckb"] == "یەک خولەک"
    assert state["last_bar"] == {"time": last[0], "open": last[1], "high": last[2], "low": last[3], "close": last[4],
                                 "volume": last[5]}
    assert state["price"] == last[4] and state["bar_count"] == 300
    assert state["visible_range"] == {"from": server.chart.bars[0][0], "to": last[0]}
    assert (state["my_drawings"], state["all_drawings"], state["user_drawings"]) == (0, 2, 2)
    assert "Volume" in state["studies"] and state["visible"] is True and state["failed"] is False
    rows = tv.app.db.query("SELECT stage, extra FROM timings WHERE stage='tv_cdp'")
    assert rows and '"op": "chart_state"' in rows[-1]["extra"]


async def test_set_timeframe_sorani_then_back(tv, server):
    result = await tv.set_timeframe("١٥ خولەک")
    assert result["ok"] and result["changed"] and result["resolution"] == "15" and result["timeframe"] == "M15"
    assert server.chart.resolution == "15"
    back = await tv.set_timeframe("1")
    assert back["ok"] and back["resolution"] == "1" and server.chart.resolution == "1"
    same = await tv.set_timeframe("1m")
    assert same["ok"] and not same["changed"]


async def test_set_timeframe_unknown_or_refused(tv, server):
    calls_before = len(server.chart.calls)
    unknown = await tv.set_timeframe("banana")
    assert not unknown["ok"] and unknown["error"] == "unknown_timeframe"
    assert len(server.chart.calls) == calls_before          # nothing was sent to the chart
    server.chart.allowed_resolutions.discard("45")
    refused = await tv.set_timeframe("45")
    assert not refused["ok"] and refused["error"] == "timeframe_failed" and refused["restored"]
    assert server.chart.resolution == "1"


async def test_set_symbol_keeps_the_users_gold_feed(tv, server):
    result = await tv.set_symbol("گۆڵد")
    assert result["ok"] and not result["changed"] and result["symbol"] == "TVC:GOLD" and result["reason"] == "same"
    assert not any(fn == "setSymbol" for fn, _ in server.chart.calls)


async def test_set_symbol_switches_and_verifies(tv, server):
    btc = await tv.set_symbol("بیتکۆین")
    assert btc["ok"] and btc["changed"] and btc["symbol"] == "BINANCE:BTCUSDT" and server.chart.symbol == "BINANCE:BTCUSDT"
    gold = await tv.set_symbol("زێڕ")
    assert gold["ok"] and gold["symbol"] == "OANDA:XAUUSD"
    explicit = await tv.set_symbol("TVC:GOLD")
    assert explicit["ok"] and explicit["reason"] == "explicit" and server.chart.symbol == "TVC:GOLD"
    apple = await tv.set_symbol("AAPL")
    assert apple["ok"] and apple["symbol"] == "NASDAQ:AAPL"


async def test_set_symbol_failure_is_restored_and_unknown_is_not_sent(tv, server):
    bad = await tv.set_symbol("BAD:XXXX")
    assert not bad["ok"] and bad["error"] == "symbol_failed" and bad["restored"] and server.chart.symbol == "TVC:GOLD"
    unknown = await tv.set_symbol("شتێکی نەناسراو")
    assert not unknown["ok"] and unknown["error"] == "unknown_symbol"


async def test_bars_are_utc_ascending_bars(tv, server):
    bars = await tv.bars(120)
    assert len(bars) == 120 and tv.last_bars_source == "series"
    assert [b["time"] for b in bars] == sorted(b["time"] for b in bars)
    assert bars[-1]["time"] == server.chart.bars[-1][0] and isinstance(bars[-1]["close"], float)
    assert set(bars[0]) == {"time", "open", "high", "low", "close", "volume"}
    assert len(await tv.bars(100000)) == 300                  # only what the chart has loaded


async def test_draw_many_records_ownership_and_reports_errors(tv, server):
    server.chart.fail_kinds.add("fib_retracement")
    result = await tv.draw_many([
        {"kind": "horizontal_line", "points": [{"price": 4262}], "text": "بەرگری"},
        {"kind": "trend_line", "points": [{"price": 4250, "bars_ago": 40}, {"price": 4258}]},
        {"kind": "rectangle", "points": [{"price": 4240}, {"price": 4245}], "text": "zone"},
        {"kind": "fib_retracement", "points": [{"price": 4240}, {"price": 4270}]},
        {"kind": "horizontal_line", "points": [{"price": 1.5}]},
    ], tag="levels")
    assert result["ok"] and result["drawn"] == 3 and result["kinds"] == ["horizontal_line", "trend_line", "rectangle"]
    assert [e["index"] for e in result["errors"]] == [4, 3]
    rows = owned(tv)
    assert [r["tv_id"] for r in rows] == result["ids"] and {r["tag"] for r in rows} == {"levels"}
    assert {r["symbol"] for r in rows} == {"TVC:GOLD"} and {r["timeframe"] for r in rows} == {"M1"}
    assert rows[0]["text"] == "بەرگری" and '"price": 4262' in rows[0]["points"]
    assert server.chart.drawn_specs[0]["overrides"]["linecolor"] == "#ef5350"     # resistance colour
    assert server.chart.drawn_specs[2]["points"][0]["bars_ago"] == 50             # timeless zone spans 50 bars


async def test_draw_single_contract_shape(tv):
    good = await tv.draw("arrow_up", [{"price": 4250, "bars_ago": 3}], text="buy zone?")
    assert good["ok"] and good["id"] and good["db_id"]
    bad = await tv.draw("trend_line", [{"price": 4250}])
    assert not bad["ok"] and "exactly 2 points" in bad["error"]


async def test_clear_removes_only_sam_drawings(tv, server):
    await tv.draw_many([{"kind": "horizontal_line", "points": [{"price": 4260}]},
                        {"kind": "text", "points": [{"price": 4255}], "text": "SAM test"}])
    assert set(server.chart.shapes) == {"USERa1", "USERb2", "sam001", "sam002"}
    assert await tv.clear_my_drawings() == 2
    assert set(server.chart.shapes) == {"USERa1", "USERb2"}                     # the user's drawings survive
    removes = [args for fn, args in server.chart.calls if fn == "remove"]
    assert removes == [[["sam001", "sam002"]]]                                  # only SAM ids were ever sent
    assert all(r["removed_at"] for r in owned(tv))
    assert await tv.clear_my_drawings() == 0


async def test_user_deleted_sam_drawing_is_pruned(tv, server):
    from sam.trading import tv_owner
    await tv.draw_many([{"kind": "horizontal_line", "points": [{"price": 4260}]},
                        {"kind": "horizontal_line", "points": [{"price": 4250}]}])
    del server.chart.shapes["sam001"]                                           # the user deleted it by hand
    state = await tv.chart_state()
    assert state["my_drawings"] == 1 and state["user_drawings"] == 2
    # Forgotten only after repeated misses spread over time, never right after a
    # symbol change (drawings may still be loading: review 2026-09-24).
    assert [r["tv_id"] for r in await tv.my_drawings()] == ["sam001", "sam002"]
    clock = [1000.0]
    tv.owner._clock = lambda: clock[0]        # noqa: SLF001
    tv.owner.changed_at = 0.0
    for _ in range(tv_owner.MISSES):
        clock[0] += tv_owner.MISS_SPAN_S
        rows = await tv.my_drawings()
    assert [r["tv_id"] for r in rows] == ["sam002"]
    result = await tv.clear()
    assert result["removed"] == 1 and result["pruned"] == 0


async def test_drawings_recreated_under_new_ids_are_readopted_and_cleared(tv, server):
    """Live 2026-09-24: after a symbol round trip TradingView re-created SAM's
    lines under new ids; SAM said it had none and the lines stayed."""
    await tv.draw_many([{"kind": "horizontal_line", "points": [{"price": 4260}], "text": "بەرگری M15"},
                        {"kind": "horizontal_line", "points": [{"price": 4250}], "text": "پشتگیری M15"}])
    new_ids = [server.chart.recreate("sam001"), server.chart.recreate("sam002")]
    await tv.set_symbol("BINANCE:BTCUSDT")
    await tv.set_symbol("TVC:GOLD")
    state = await tv.chart_state()                  # right after the switch: nothing is forgotten
    assert state["my_drawings"] == 2
    assert sorted(r["tv_id"] for r in await tv.my_drawings()) == sorted(new_ids)
    result = await tv.clear()
    assert result["removed"] == 2
    assert set(server.chart.shapes) == {"USERa1", "USERb2"}                    # the user's own stayed


async def test_drawings_on_another_symbol_are_kept_until_visible(tv, server):
    await tv.draw("horizontal_line", [{"price": 4260}])
    gold_shapes = dict(server.chart.shapes)
    server.chart.symbol, server.chart.shapes = "BINANCE:BTCUSDT", {}              # drawings are per symbol
    result = await tv.clear()
    assert result == {"removed": 0, "pruned": 0, "other_symbol": 1, "failed": 0, "remaining_on_chart": 0}
    assert owned(tv)[0]["removed_at"] is None
    assert len(await tv.my_drawings(symbol="gold")) == 1 and await tv.my_drawings(symbol="BTCUSD") == []
    server.chart.symbol, server.chart.shapes = "TVC:GOLD", gold_shapes
    assert await tv.clear_my_drawings() == 1


async def test_clear_by_tag_prefix(tv, server):
    await tv.draw("horizontal_line", [{"price": 4260}], tag="levels")
    await tv.draw("horizontal_line", [{"price": 4261}], tag="analysis:1")
    await tv.draw("horizontal_line", [{"price": 4262}], tag="analysis:2")
    assert await tv.clear_my_drawings(tag="analysis") == 2
    assert [r["tag"] for r in owned(tv) if r["removed_at"] is None] == ["levels"]


async def test_reconnects_after_the_socket_drops(tv, server):
    await tv.chart_state()
    await server.drop_all()
    state = await tv.chart_state()
    assert state["symbol"] == "TVC:GOLD" and server.ws_opened == 2
    assert server.chart.injections == 1                                          # same page: no re-injection


async def test_reinjects_after_page_reload_and_retries_lost_context(tv, server):
    await tv.chart_state()
    server.chart.injected = False                                                # page reloaded
    await tv.chart_state()
    assert server.chart.injections == 2
    server.chart.throw_next = ProtocolFail("Execution context was destroyed.")
    assert (await tv.chart_state())["symbol"] == "TVC:GOLD"


async def test_foreign_page_is_refused(tv, server):
    await tv.chart_state()
    server.chart.url = "https://evil.example/chart/x/"
    with pytest.raises(TvError) as info:
        await tv.chart_state()
    assert info.value.code == "foreign_page" and not tv.connected
    with pytest.raises(TvError) as info:                                         # no chart target any more
        await tv.chart_state()
    assert info.value.code == "no_chart"


async def test_not_ready_chart(tv, server):
    server.chart.ready = False
    assert not await tv.connect(wait_ready_s=0.1)


async def test_screenshot_is_clipped_and_scaled(tv, server):
    data = await tv.screenshot(max_width=1440)
    assert data[:2] == b"\xff\xd8"                                              # JPEG
    params = server.screenshot_params[-1]
    clip = params["clip"]
    assert (clip["x"], clip["y"], clip["width"], clip["height"]) == (56, 42, 1541, 866)
    assert clip["scale"] == pytest.approx(1440 / (1541 * 1.75), rel=1e-6)
    assert params["format"] == "jpeg" and params["quality"] == 80
    await tv.screenshot(fmt="png", max_width=0, clip_chart=False)
    params = server.screenshot_params[-1]
    assert params["format"] == "png" and params["clip"]["width"] == 1646 and params["clip"]["scale"] == 1.0


async def test_evaluate_is_host_checked(tv, server):
    assert await tv.evaluate("document.visibilityState") == "visible"
    server.chart.url = "https://evil.example/x"
    with pytest.raises(TvError):
        await tv.evaluate("document.visibilityState")


# -- ensure_running ---------------------------------------------------------------------------------------------
async def test_ensure_running_connected_focuses(make_app, server):
    proc = FakeProc(pids=[11])
    tv = make_bridge(make_app(), server, proc)
    result = await tv.ensure_running()
    assert result["ok"] and result["state"] == "connected" and proc.focused == 1 and proc.activations == []
    background = await tv.ensure_running(focus=False)
    assert background["ok"] and proc.focused == 1
    await tv.close()


async def test_ensure_running_starts_tradingview_with_the_port(make_app, server):
    server.online = False
    proc = FakeProc(pids=[], on_activate=lambda: setattr(server, "online", True))
    tv = make_bridge(make_app(), server, proc)
    result = await tv.ensure_running()
    assert result["ok"] and result["state"] == "started"
    assert proc.activations == [(AUMID, f"--remote-debugging-port={server.port}")]
    await tv.close()


async def test_running_without_port_needs_restart_or_confirmation(make_app, server):
    server.online = False
    proc = FakeProc(pids=[11, 12], on_activate=lambda: setattr(server, "online", True))
    tv = make_bridge(make_app(), server, proc)
    plain = await tv.ensure_running()
    assert not plain["ok"] and plain["state"] == "needs_restart" and proc.closed == []

    asked: list[str] = []

    async def say_no(question: str) -> bool:
        asked.append(question)
        return False

    declined = await tv.ensure_running(allow_restart=True, confirm=say_no)
    assert declined["state"] == "needs_restart" and declined["declined"] and asked == [CONFIRM_RESTART_CKB]
    assert proc.closed == [] and proc.activations == []

    async def say_yes(question: str) -> bool:
        return True

    restarted = await tv.ensure_running(allow_restart=True, confirm=say_yes)
    assert restarted["ok"] and restarted["state"] == "restarted"
    assert proc.closed == [[11, 12]] and len(proc.activations) == 1
    activity = tv.app.db.query("SELECT name FROM activity WHERE name='tradingview_restart'")
    assert activity
    await tv.close()


async def test_not_installed_and_discovery_fallback(make_app, server):
    server.online = False
    proc = FakeProc(pids=[], installed=False)
    tv = make_bridge(make_app(), server, proc)
    result = await tv.ensure_running()
    assert not result["ok"] and result["state"] == "not_installed"

    class Store(FakeProc):
        def activate(self, aumid: str, arguments: str) -> int:
            if aumid == AUMID:
                self.activations.append((aumid, arguments))
                raise OSError("package missing")
            return super().activate(aumid, arguments)

    store = Store(pids=[], on_activate=lambda: setattr(server, "online", True))
    store.discovered = [AUMID, "31178TradingViewInc.TradingView_q4jpyh43s5mv6!TradingView.Desktop"]
    tv2 = make_bridge(tv.app, server, store)
    result = await tv2.ensure_running()
    assert result["ok"] and [a for a, _ in store.activations] == [AUMID, store.discovered[1]]
    await tv2.close()


async def test_other_app_on_port(make_app):
    other = FakeCdpServer(user_agent="Mozilla/5.0 Chrome/146 Safari/537.36")
    await other.start()
    try:
        tv = make_bridge(make_app(), other, FakeProc(pids=[]))
        result = await tv.ensure_running()
        assert not result["ok"] and result["state"] == "failed" and "another application" in result["detail"]
    finally:
        await other.stop()


async def test_probe_never_launches(make_app, server):
    server.online = False
    proc = FakeProc(pids=[11])
    tv = make_bridge(make_app(), server, proc)
    assert (await tv.probe())["state"] == "needs_restart"
    proc.pids = []
    assert (await tv.probe())["state"] == "not_running"
    assert proc.activations == [] and proc.closed == []
    server.online = True
    assert (await tv.probe())["state"] == "connected"
    await tv.close()
