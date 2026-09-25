"""Chart bridge: the DevTools client (discovery rules, evaluate, errors, socket loss)."""

from __future__ import annotations

import asyncio

import pytest
from trading_chart_helpers import CHART_TARGET_ID, FakeCdpServer, JsThrow, ProtocolFail, free_port

from sam.trading.cdp import (CdpClosed, CdpError, CdpJsError, CdpProtocolError, CdpSession, CdpTimeout,
                             CdpUnavailable, browser_version, is_tradingview_browser, is_tradingview_url,
                             list_chart_targets, local_ws_url)


@pytest.fixture
async def server():
    fake = FakeCdpServer()
    await fake.start()
    yield fake
    await fake.stop()


@pytest.mark.parametrize("url,expected", [
    ("https://www.tradingview.com/chart/b2R792Bm/", True),
    ("https://tradingview.com/chart/x/", True),
    ("https://www.tradingview.com/the-leap/amp-futures/", False),
    ("http://www.tradingview.com/chart/x/", False),
    ("https://tradingview.com.evil.example/chart/x/", False),
    ("https://eviltradingview.com/chart/x/", False),
    ("file:///C:/Program%20Files/WindowsApps/TradingView/index.html", False),
    ("", False),
])
def test_is_tradingview_url(url, expected):
    assert is_tradingview_url(url) is expected


def test_browser_check_and_local_ws_url():
    assert is_tradingview_browser({"User-Agent": "Mozilla/5.0 TradingView/3.4.1 Chrome/146 TVDesktop/3.4.1"})
    assert not is_tradingview_browser({"User-Agent": "Mozilla/5.0 Chrome/146 Safari/537.36", "Browser": "Chrome/146"})
    assert local_ws_url(9222, CHART_TARGET_ID) == f"ws://127.0.0.1:9222/devtools/page/{CHART_TARGET_ID}"
    with pytest.raises(CdpError):
        local_ws_url(9222, "../../evil")
    with pytest.raises(CdpError):
        CdpSession("ws://192.168.1.5:9222/devtools/page/" + CHART_TARGET_ID)


async def test_list_chart_targets_keeps_only_chart_pages_and_rebuilds_ws_url(server):
    targets = await list_chart_targets(server.port)
    assert [t.id for t in targets] == [CHART_TARGET_ID]
    # the page advertised ws://evil.example/...; the client only ever uses 127.0.0.1
    assert targets[0].ws_url == f"ws://127.0.0.1:{server.port}/devtools/page/{CHART_TARGET_ID}"
    assert not hasattr(targets[0], "title")   # titles carry the account holder's name: never kept
    assert server.http_paths[:2] == ["/json/version", "/json/list"]


async def test_other_app_on_the_port_is_refused():
    fake = FakeCdpServer(user_agent="Mozilla/5.0 Chrome/146 Safari/537.36")
    await fake.start()
    try:
        with pytest.raises(CdpUnavailable, match="another application"):
            await list_chart_targets(fake.port)
    finally:
        await fake.stop()


async def test_closed_port_is_unavailable():
    port = free_port()
    assert await browser_version(port, timeout_s=0.5) is None
    with pytest.raises(CdpUnavailable):
        await list_chart_targets(port, timeout_s=0.5)


async def test_evaluate_value_js_error_protocol_error_and_timeout(server):
    session = CdpSession(local_ws_url(server.port, CHART_TARGET_ID))
    await session.connect()
    try:
        assert await session.evaluate("location.protocol + '//' + location.host + location.pathname") == server.chart.url
        server.chart.throw_next = JsThrow("boom")
        with pytest.raises(CdpJsError, match="boom"):
            await session.evaluate("anything")
        server.chart.throw_next = ProtocolFail("Execution context was destroyed.")
        with pytest.raises(CdpProtocolError) as info:
            await session.evaluate("anything")
        assert info.value.context_lost
        with pytest.raises(CdpProtocolError, match="wasn't found"):
            await session.call("Network.getAllCookies")
        server.hang_methods.add("Runtime.evaluate")
        with pytest.raises(CdpTimeout):
            await session.evaluate("document.visibilityState", timeout_s=0.2)
    finally:
        await session.close()
    assert session.closed


async def test_concurrent_calls_are_matched_by_id(server):
    session = CdpSession(local_ws_url(server.port, CHART_TARGET_ID))
    await session.connect()
    try:
        results = await asyncio.gather(*(session.evaluate("document.visibilityState") for _ in range(20)))
        assert results == ["visible"] * 20
    finally:
        await session.close()


async def test_socket_loss_fails_pending_calls(server):
    session = CdpSession(local_ws_url(server.port, CHART_TARGET_ID))
    await session.connect()
    server.hang_methods.add("Runtime.evaluate")
    pending = asyncio.ensure_future(session.evaluate("document.visibilityState", timeout_s=5))
    await asyncio.sleep(0.05)
    await server.drop_all()
    with pytest.raises(CdpClosed):
        await pending
    assert session.closed
    with pytest.raises(CdpClosed):
        await session.call("Runtime.evaluate", {"expression": "1"})
    await session.close()
