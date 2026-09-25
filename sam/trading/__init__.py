"""SAM 2 trading: TradingView Desktop chart bridge, MT5 feed, analysis engine,
strategy cards, theories and the market monitor.

Two builders share this package, so its wiring lives in two entry modules
listed in ``sam.app.PACKAGES`` (see docs/CONTRACTS.md):

- ``sam.trading.chart_tools`` (chart bridge): sets ``app.trading.tv`` and
  registers tv_open, tv_set_chart, chart_state, draw_on_chart, clear_my_drawings.
- ``sam.trading.tools`` (engine): sets ``app.trading.mt5 / engine / theories /
  strategies / monitor`` and registers get_price, analyze_market, set_alert,
  list_alerts, cancel_alert, strategy_save, strategy_list, strategy_get,
  theory_info.

Shared vocabulary (bars, drawing kinds, timeframes, symbol aliases) is in
``sam.trading.common``. SAM 2 never places, modifies or closes orders.
"""
