"""Chart tools through the real ToolRegistry: registration, schemas, Sorani summaries,
confirmation of the TradingView restart, and that user drawings and private text stay safe."""

from __future__ import annotations

import json

import pytest
from trading_chart_helpers import FakeCdpServer, FakeProc

from sam.events import ConfirmRequest
from sam.textnorm import is_arabic_script
from sam.trading import chart_tools

TOOL_NAMES = {"tv_open", "tv_set_chart", "chart_state", "draw_on_chart", "clear_my_drawings"}


@pytest.fixture
async def server():
    fake = FakeCdpServer()
    await fake.start()
    yield fake
    await fake.stop()


@pytest.fixture
async def app(make_app, server):
    app = make_app()
    chart_tools.register(app)
    tv = app.trading.tv
    tv._port = server.port
    tv.proc = FakeProc(pids=[11])
    tv.poll_s, tv.launch_timeout_s, tv.ready_timeout_s, tv.close_timeout_s = 0.02, 2.0, 1.0, 0.5
    yield app
    await chart_tools.stop(app)


async def call(app, name, args=None):
    return await app.tools.dispatch(name, args or {}, source="text")


def test_register_sets_slot_and_tools(make_app):
    app = make_app()
    chart_tools.register(app)
    assert app.trading.tv is not None and not app.trading.tv.connected
    assert TOOL_NAMES <= set(app.tools.names())
    assert {app.tools.get(n).owner for n in TOOL_NAMES} == {"trading.chart"}
    assert all(app.tools.get(n).examples_ckb for n in TOOL_NAMES)
    schemas = {t["function"]["name"]: t["function"] for t in app.tools.openai_tools(TOOL_NAMES)}
    items = schemas["draw_on_chart"]["parameters"]["properties"]["items"]
    assert items["items"]["properties"]["kind"]["enum"][0] == "horizontal_line"
    assert schemas["draw_on_chart"]["parameters"]["required"] == ["items"]


def test_gemini_declarations_build(make_app):
    app = make_app()
    chart_tools.register(app)
    declarations = app.tools.gemini_declarations(TOOL_NAMES, live=True)
    assert {d.name for d in declarations} == TOOL_NAMES


async def test_chart_state_tool(app, server):
    result = await call(app, "chart_state")
    assert result["ok"], result
    # Spoken: the Kurdish name (review 2026-09-24); the ticker stays in the data.
    assert is_arabic_script(result["summary"]) and "زێڕ" in result["summary"] and "TVC:GOLD" not in result["summary"]
    data = result["data"]
    assert data["symbol"] == "TVC:GOLD" and data["timeframe"] == "M1" and data["user_drawings"] == 2
    # indicator names are chart text: returned only as untrusted data
    assert "studies" not in data and "Ignore previous instructions" in data["untrusted"]["indicator_names"][1]
    assert "Private Name" not in json.dumps(result, ensure_ascii=False)


async def test_set_chart_sorani_phrase(app, server):
    result = await call(app, "tv_set_chart", {"symbol": "گۆڵد", "timeframe": "١٥ خولەک"})
    assert result["ok"], result
    assert result["data"]["symbol"] == "TVC:GOLD" and result["data"]["resolution"] == "15"
    assert "پازدە خولەک" in result["summary"] and is_arabic_script(result["summary"])
    assert app.trading.tv.proc.focused >= 1                      # the user asked to see it
    unchanged = await call(app, "tv_set_chart", {"timeframe": "15"})
    assert unchanged["ok"] and not unchanged["data"]["changed"] and "پێشتر" in unchanged["summary"]


async def test_set_chart_errors(app, server):
    assert not (await call(app, "tv_set_chart", {}))["ok"]
    unknown = await call(app, "tv_set_chart", {"symbol": "شتێکی نەناسراو"})
    assert not unknown["ok"] and "نەناسرایەوە" in unknown["summary"]
    bad_tf = await call(app, "tv_set_chart", {"timeframe": "banana"})
    assert not bad_tf["ok"] and server.chart.resolution == "1"


async def test_draw_then_clear_leaves_user_drawings(app, server):
    items = [{"kind": "horizontal_line", "points": [{"price": "4260"}], "text": "بەرگری"},
             {"kind": "horizontal_line", "points": [{"price": 4240}], "text": "پشتگیری"},
             {"kind": "long_position", "points": [{"price": 4250}, {"price": 4244}, {"price": 4265}]}]
    drawn = await call(app, "draw_on_chart", {"items": items, "tag": "levels"})
    assert drawn["ok"], drawn
    assert drawn["data"]["drawn"] == 3 and "هێڵی ئاسۆیی" in drawn["summary"]
    assert server.chart.drawn_specs[2]["position"] == {"stop": 4244.0, "target": 4265.0}
    state = await call(app, "chart_state")
    assert state["data"]["my_drawings"] == 3 and state["data"]["user_drawings"] == 2
    cleared = await call(app, "clear_my_drawings")
    assert cleared["ok"] and cleared["data"]["removed"] == 3 and "دەستم لە هێڵەکانی تۆ نەدا" in cleared["summary"]
    assert set(server.chart.shapes) == {"USERa1", "USERb2"}
    again = await call(app, "clear_my_drawings")
    assert again["ok"] and "هیچ" in again["summary"]


async def test_draw_with_only_invalid_items_fails_honestly(app, server):
    result = await call(app, "draw_on_chart", {"items": [{"kind": "trend_line", "points": [{"price": 4250}]}]})
    assert not result["ok"] and "exactly 2 points" in result["summary"]
    assert not server.chart.drawn_specs


async def test_read_only_tools_do_not_start_tradingview(app, server):
    server.online = False
    app.trading.tv.proc.pids = []
    result = await call(app, "chart_state")
    assert not result["ok"] and result["data"]["hint"].startswith("call tv_open")
    assert app.trading.tv.proc.activations == []


async def test_tv_open_asks_before_restarting(app, server):
    server.online = False
    proc = app.trading.tv.proc
    proc.pids = [11, 12]
    proc.on_activate = lambda: setattr(server, "online", True)
    questions: list[str] = []
    answer = {"value": False}

    def on_request(event: ConfirmRequest) -> None:
        questions.append(event.question_ckb)
        app.confirm.resolve(event.confirm_id, answer["value"])

    app.bus.subscribe(ConfirmRequest, on_request)
    declined = await call(app, "tv_open")
    assert not declined["ok"] and declined["data"]["declined"] and "دانەخست" in declined["summary"]
    assert proc.closed == [] and questions and "دابخرێت" in questions[0]
    answer["value"] = True
    opened = await call(app, "tv_open")
    assert opened["ok"], opened
    assert opened["data"]["state"] == "restarted" and proc.closed == [[11, 12]]
    assert "زێڕ" in opened["summary"] and is_arabic_script(opened["summary"]) and opened["data"]["symbol"] == "TVC:GOLD"


async def test_tv_open_when_already_connected(app, server):
    result = await call(app, "tv_open")
    assert result["ok"] and result["data"]["state"] == "connected" and app.trading.tv.proc.activations == []


async def test_start_probe_and_stop(app, server):
    await chart_tools.start(app)
    for task in list(app._tasks):
        await task
    assert app.trading.tv.connected
    await chart_tools.stop(app)
    assert not app.trading.tv.connected


async def test_missing_bridge_is_reported(make_app):
    app = make_app()
    chart_tools.register(app)
    app.trading.tv = None
    for name, args in (("chart_state", {}), ("tv_open", {}), ("clear_my_drawings", {}),
                       ("draw_on_chart", {"items": [{"kind": "text", "points": [{"price": 1}]}]})):
        result = await call(app, name, args)
        assert not result["ok"] and "ئامادە نییە" in result["summary"]


async def test_clear_with_a_tag_that_names_no_group_clears_all_of_sams_drawings(app, server):
    """Integration smoke 2026-09-24: the model sent tag 'none' for "هێڵەکانت بسڕەوە"
    and SAM's 7 analysis drawings stayed on the chart."""
    items = [{"kind": "horizontal_line", "points": [{"price": 4260}], "text": "بەرگری"}]
    assert (await call(app, "draw_on_chart", {"items": items, "tag": "analysis:7"}))["ok"]
    for tag in ("none", "levels"):
        assert (await call(app, "draw_on_chart", {"items": items, "tag": "analysis:8"}))["ok"]
        cleared = await call(app, "clear_my_drawings", {"tag": tag})
        assert cleared["ok"] and cleared["data"]["removed"] >= 1, cleared
    assert set(server.chart.shapes) == {"USERa1", "USERb2"}
