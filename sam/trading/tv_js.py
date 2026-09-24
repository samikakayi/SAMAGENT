"""JavaScript helper bundle injected into the TradingView chart page (``window.__sam``).

Kept to the chart API only: ``TradingViewApi.activeChart()`` public methods plus
the main series (the verified path for bars). It never reads cookies, storage,
account, broker or network data.

Facts measured on TradingView Desktop 3.4.1 (2026-09-24, this PC):

- ``activeChart().exportData`` is a stub that rejects with "Data export is not
  supported", so ``bars()`` tries it at most once per page and then uses
  ``mainSeries().bars()`` (``valueAt(i)`` -> [time, o, h, l, c, v]; 300 bars in
  0.3 ms). Bar times are UTC seconds (``_convertTimeToPublic`` is the identity
  on intraday bars).
- ``setSymbol``/``setResolution`` are async and resolve after the series has
  completed or errored; a bad symbol leaves ``mainSeries().isFailed()`` true, so
  the helper restores the previous symbol/resolution instead of leaving the
  user's chart on an error screen.
- ``createShape``/``createMultipointShape`` drew all ten DRAWING_KINDS; points
  read back exactly. Position tools take stop/target as tick distances
  (``stopLevel``/``profitLevel``; tick = minmov / pricescale, 0.001 on TVC:GOLD).
- ``removeEntity(id, {disableUndo: true})`` removes cleanly.

Python calls functions through ``call_expression`` so every call re-checks the
page host and reports ``{__sam_missing: true}`` after a page reload (the bundle
is then re-injected once).
"""

from __future__ import annotations

import json
from typing import Any

JS_VERSION = 2   # 2: shapesDetailed (drawing re-adoption, sam/trading/tv_owner.py)
BUNDLE_MARKER = "/*sam-bundle*/"

# Host check shared by every expression: anything but tradingview.com is refused.
_HOST_CHECK = "if (!/(^|\\.)tradingview\\.com$/.test(location.hostname)) return {__sam_error: 'foreign_page'};"

LOCATION_EXPRESSION = "location.protocol + '//' + location.host + location.pathname"
VISIBILITY_EXPRESSION = "document.visibilityState"

BUNDLE = BUNDLE_MARKER + r"""
(() => {
  const V = __VERSION__;
  if (!/(^|\.)tradingview\.com$/.test(location.hostname)) return 'foreign_page';
  if (window.__sam && window.__sam.v === V) return 'present';
  const api = () => window.TradingViewApi;
  const widget = () => api()._activeChartWidgetWV.value()._chartWidget;
  const chart = () => api().activeChart();
  const series = () => widget().model().mainSeries();
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const msg = e => String((e && e.message) || e).slice(0, 300);
  const withTimeout = (p, ms, what) => Promise.race([Promise.resolve(p), new Promise((_, rej) =>
      setTimeout(() => rej(new Error(what + ' timed out after ' + ms + ' ms')), ms))]);

  function apiReady() {
    try {
      const a = api();
      if (!a || typeof a.activeChart !== 'function' || !a._activeChartWidgetWV) return false;
      const c = a.activeChart();
      return !!(c && c.symbol() && series());
    } catch (e) { return false; }
  }
  function ready() {
    if (!apiReady()) return false;
    try { const s = series(); return !s.isFailed() && !s.isLoading() && s.bars().size() > 0; }
    catch (e) { return false; }
  }
  async function waitFor(pred, ms) {
    const t0 = Date.now();
    while (!pred()) { if (Date.now() - t0 > ms) return false; await sleep(100); }
    return true;
  }
  async function needReady(ms) { if (!(await waitFor(ready, ms))) throw new Error('not_ready'); }
  async function needApi(ms) { if (!(await waitFor(apiReady, ms))) throw new Error('not_ready'); }
  function tick() {
    const si = series().symbolInfo() || {};
    const ps = Number(si.pricescale) || 100, mm = Number(si.minmov) || 1;
    return mm / ps;
  }
  const num = x => (x == null || !isFinite(x)) ? 0 : Number(x);
  const row = v => ({t: v[0], o: v[1], h: v[2], l: v[3], c: v[4], v: num(v[5])});

  function state() {
    const c = chart(), s = series(), b = s.bars(), last = b.last(), si = s.symbolInfo() || {};
    let vr = null, vpr = null, tz = '';
    try { vr = c.getVisibleRange(); } catch (e) {}
    try { vpr = c.getVisiblePriceRange(); } catch (e) {}
    try { tz = c.getTimezone(); } catch (e) {}
    return {symbol: c.symbol(), resolution: c.resolution(), description: String(si.description || ''),
      type: String(si.type || ''), visible_range: vr, visible_price_range: vpr, bar_count: b.size(),
      last_bar: last ? row(last.value) : null, studies: c.getAllStudies().map(x => String(x.name || '')),
      shapes: c.getAllShapes().map(x => ({id: String(x.id), name: String(x.name || '')})), tick: tick(),
      timezone: tz, visibility: document.visibilityState, loading: s.isLoading(), failed: s.isFailed()};
  }
  async function settle(ms) {
    const t0 = Date.now();
    await sleep(0);
    while (Date.now() - t0 < ms) {
      const s = series();
      if (s.isFailed()) return 'failed';
      if (!s.isLoading() && s.bars().size() > 0) return 'ok';
      await sleep(50);
    }
    return 'timeout';
  }
  async function change(kind, value, ms) {
    await needApi(ms);   // not needReady: a chart stuck on a bad symbol must still be fixable
    const c = chart();
    const read = () => kind === 'symbol' ? c.symbol() : c.resolution();
    const apply = v => kind === 'symbol' ? c.setSymbol(v, {}) : c.setResolution(v, {});
    const before = read();
    let error = null;
    try { await withTimeout(apply(value), ms, kind); } catch (e) { error = msg(e); }
    const settled = await settle(ms);
    if (error || settled === 'failed') {
      let restored = false;
      if (before && read() !== before) {
        try { await withTimeout(apply(before), ms, 'restore'); restored = (await settle(ms)) === 'ok'; } catch (e) {}
      }
      return {ok: false, error: error || (kind + ' failed to load'), restored, before, symbol: c.symbol(),
              resolution: c.resolution()};
    }
    return {ok: true, before, symbol: c.symbol(), resolution: c.resolution(), settled};
  }
  function fromExport(ex, n) {
    if (!ex || !Array.isArray(ex.schema) || !ex.data || !ex.data.length) return null;
    const idx = {};
    ex.schema.forEach((f, i) => {
      if (f.type === 'time') idx.t = i;
      const k = String(f.plotTitle || '').toLowerCase();
      if (['open', 'high', 'low', 'close', 'volume'].includes(k)) idx[k[0]] = i;
    });
    if ([idx.t, idx.o, idx.h, idx.l, idx.c].some(x => x == null)) return null;
    return Array.from(ex.data).slice(-n).map(r => ({t: r[idx.t], o: r[idx.o], h: r[idx.h], l: r[idx.l],
      c: r[idx.c], v: idx.v == null ? 0 : num(r[idx.v])}));
  }
  async function bars(n) {
    await needReady(5000);
    const w = window.__sam;
    if (w._exportOk !== false) {
      try {
        const ex = await withTimeout(chart().exportData({includeTime: true, includeSeries: true,
          includedStudies: [], includeDisplayedValues: false}), 2000, 'exportData');
        const rows = fromExport(ex, n);
        if (rows) { w._exportOk = true; return {source: 'exportData', rows, loaded: ex.data.length}; }
      } catch (e) {}
      w._exportOk = false;
    }
    const b = series().bars(), first = b.firstIndex(), last = b.lastIndex(), out = [];
    for (let i = Math.max(first, last - n + 1); i <= last; i++) {
      const v = b.valueAt(i);
      if (v && v[0] != null) out.push(row(v));
    }
    return {source: 'series', rows: out, loaded: b.size()};
  }
  function resolveTime(p) {
    if (p.time != null) return Number(p.time);
    const b = series().bars(), last = b.lastIndex(), ago = Math.max(0, Math.floor(p.bars_ago || 0));
    const lv = b.valueAt(last);
    if (!lv) throw new Error('the chart has no bars');
    if (last - ago >= b.firstIndex()) { const v = b.valueAt(last - ago); if (v) return v[0]; }
    const pv = b.valueAt(last - 1);
    return lv[0] - ago * (pv ? lv[0] - pv[0] : 60);
  }
  async function drawOne(spec) {
    const c = chart();
    const pts = spec.points.map(p => ({time: resolveTime(p), price: Number(p.price)}));
    const opts = {shape: spec.kind, lock: !!spec.lock, disableSelection: false,
                  overrides: Object.assign({}, spec.overrides || {})};
    if (spec.text) opts.text = spec.text;
    if (spec.position) {
      const t = tick(), e = pts[0].price;
      opts.overrides.stopLevel = Math.max(1, Math.round(Math.abs(e - spec.position.stop) / t));
      opts.overrides.profitLevel = Math.max(1, Math.round(Math.abs(spec.position.target - e) / t));
    }
    const id = pts.length === 1 ? await c.createShape(pts[0], opts) : await c.createMultipointShape(pts, opts);
    if (!id) throw new Error('TradingView did not create the ' + spec.kind);
    const shape = c.getShapeById(id);   // throws "There is no such shape" when it was not created
    return {id: String(id), points: shape.getPoints()};
  }
  async function draw(specs) {
    await needReady(5000);
    const t0 = performance.now(), results = [];
    for (const spec of specs) {
      try { results.push(Object.assign({ok: true}, await drawOne(spec))); }
      catch (e) { results.push({ok: false, error: msg(e)}); }
    }
    return {results, ms: performance.now() - t0, symbol: chart().symbol(), resolution: chart().resolution()};
  }
  function shapes() { return chart().getAllShapes().map(x => ({id: String(x.id), name: String(x.name || '')})); }
  function shapesDetailed() {
    const c = chart();
    return c.getAllShapes().map(x => {
      let text = null, points = null;
      try { const sh = c.getShapeById(x.id); points = sh.getPoints(); const p = sh.getProperties(); text = p ? p.text : null; }
      catch (e) {}
      return {id: String(x.id), name: String(x.name || ''), text: text == null ? '' : String(text), points};
    });
  }
  function remove(ids) {
    const c = chart(), present = new Set(shapes().map(s => s.id));
    const removed = [], missing = [], failed = [];
    for (const id of ids) {
      if (!present.has(String(id))) { missing.push(id); continue; }
      try { c.removeEntity(String(id), {disableUndo: true}); removed.push(id); } catch (e) { failed.push(id); }
    }
    const after = new Set(shapes().map(s => s.id));
    return {removed: removed.filter(id => !after.has(String(id))),
            failed: failed.concat(removed.filter(id => after.has(String(id)))), missing, remaining: after.size,
            symbol: c.symbol()};
  }
  function chartRect() {
    const el = document.querySelector('.chart-container.active') || document.querySelector('.chart-markup-table')
      || document.querySelector('.layout__area--center');
    const r = el ? el.getBoundingClientRect() : {x: 0, y: 0, width: window.innerWidth, height: window.innerHeight};
    return {x: r.x, y: r.y, width: r.width, height: r.height, dpr: window.devicePixelRatio || 1,
            visibility: document.visibilityState, found: !!el, inner: [window.innerWidth, window.innerHeight]};
  }
  const guard = fn => async (...args) => {
    if (!/(^|\.)tradingview\.com$/.test(location.hostname)) return {__sam_error: 'foreign_page'};
    try { return await fn(...args); } catch (e) { return {__sam_error: msg(e)}; }
  };
  window.__sam = {v: V, _exportOk: null,
    ready: guard(async () => ready()),
    state: guard(async () => { await needApi(5000); await waitFor(ready, 3000); return state(); }),
    setSymbol: guard((v, ms) => change('symbol', v, ms)),
    setResolution: guard((v, ms) => change('resolution', v, ms)),
    bars: guard(bars), draw: guard(draw), remove: guard(async ids => remove(ids)),
    shapes: guard(async () => ({symbol: chart().symbol(), shapes: shapes()})),
    shapesDetailed: guard(async () => ({symbol: chart().symbol(), loading: series().isLoading(),
                                        shapes: shapesDetailed()})),
    chartRect: guard(async () => chartRect())};
  return 'installed';
})()
""".replace("__VERSION__", str(JS_VERSION))


def call_expression(fn: str, *args: Any) -> str:
    """Expression calling ``window.__sam[fn](...args)``.

    Arguments travel as JSON (ASCII-escaped, a valid JS literal), never as
    string-formatted code. Returns ``{__sam_missing: true}`` when the bundle
    is absent (fresh page) so the caller injects it once and retries.
    """
    if not fn.isidentifier():
        raise ValueError("bad helper name")
    payload = json.dumps(list(args), ensure_ascii=True, allow_nan=False)
    return ("(async () => { " + _HOST_CHECK +
            f" const s = window.__sam; if (!s || s.v !== {JS_VERSION}) return {{__sam_missing: true}};"
            f" return await s[{json.dumps(fn)}](...{payload}); }})()")


__all__ = ["BUNDLE", "BUNDLE_MARKER", "JS_VERSION", "call_expression", "LOCATION_EXPRESSION",
           "VISIBILITY_EXPRESSION"]
