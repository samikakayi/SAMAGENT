"""Which TradingView feed is the user's own for an instrument ("learned" feeds).

``tv_symbol_for(..., learned=)`` keeps the user's own feed when SAM switches
the chart back to an instrument (his gold chart is PEPPERSTONE:XAUUSD and his
drawings live there). The feed is learned from what ``chart_state`` sees --
but only when it is the user's choice.

Verify review 2026-09-24: SAM had switched the chart to OANDA:XAUUSD at 17:19
and the chart was still on it later. The old guard skipped only the symbol SAM
set last *in this process*, so after a restart the first ``chart_state`` would
have learned SAM's own OANDA:XAUUSD as "the user's gold feed", and «گۆڵد»
would then keep going to OANDA. Now:

- every symbol SAM sets is remembered in setting ``trading.tv_sam_set_symbols``
  ({symbol: time}, kept ``KEEP_S``), so a restart still knows it;
- a symbol is learned when the user changed the chart to it (an observed
  change that SAM did not make), or when it is on the chart at the first look
  and SAM did not put it there.

One-time repair (adversarial review 2026-09-24): the evening test had already
stored ``{"XAUUSD": "OANDA:XAUUSD"}`` -- SAM's own default feed, learned by the
old code at 20:53 -- so «گۆڵد» kept going to OANDA instead of the user's
PEPPERSTONE:XAUUSD. Learned entries from before this memory existed that equal
SAM's default feed are dropped once (``trading.tv_feeds_version`` 2). That
never loses a real choice: without the entry SAM uses the same default feed.
"""

from __future__ import annotations

import time
from typing import Any, Callable

SAM_SET_KEY = "trading.tv_sam_set_symbols"
LEARNED_KEY = "trading.tv_learned_symbols"
VERSION_KEY = "trading.tv_feeds_version"
VERSION = 2
KEEP_S = 30 * 86400.0     # a symbol SAM set a month ago is no longer "SAM's"
MAX_REMEMBERED = 24


class FeedMemory:
    """Per-bridge observer. ``resolve`` maps a chart symbol to SAM's canonical
    instrument ('PEPPERSTONE:XAUUSD' -> 'XAUUSD') or None."""

    def __init__(self, config: Any, resolve: Callable[[str], str | None], *,
                 clock: Callable[[], float] = time.time, defaults: dict[str, str] | None = None) -> None:
        self.config = config
        self.resolve = resolve
        self.clock = clock
        self.last_seen = ""        # the chart symbol at the previous observation
        self.sam_set = ""          # the symbol SAM set last in this process
        self._repair_legacy(defaults)

    def _repair_legacy(self, defaults: dict[str, str] | None) -> None:
        """Drop learned entries stored before ``SAM_SET_KEY`` existed that are
        SAM's own default feed (see the module docstring). Runs once."""
        try:
            if int(self.config.get(VERSION_KEY, 0) or 0) >= VERSION:
                return
            if defaults is None:
                from .tv_parse import TV_DEFAULT_SYMBOLS
                defaults = TV_DEFAULT_SYMBOLS
            learned = dict(self.config.get(LEARNED_KEY, {}) or {})
            kept = {k: v for k, v in learned.items() if v != defaults.get(k)}
            if kept != learned:
                self.config.set(LEARNED_KEY, kept)
            self.config.set(VERSION_KEY, VERSION)
        except Exception:  # noqa: BLE001 - a repair never stops the chart bridge
            pass

    # -- SAM's own changes ---------------------------------------------------------------------------------------
    def _sam_symbols(self) -> dict[str, float]:
        try:
            value = self.config.get(SAM_SET_KEY, {}) or {}
        except Exception:  # noqa: BLE001
            value = {}
        now = self.clock()
        return {str(k): float(v) for k, v in dict(value).items()
                if isinstance(v, (int, float)) and now - float(v) < KEEP_S}

    def note_sam_set(self, symbol: str) -> None:
        """SAM itself put ``symbol`` on the chart (set_symbol)."""
        if not symbol:
            return
        self.sam_set = symbol
        remembered = self._sam_symbols()
        remembered[symbol] = self.clock()
        newest = dict(sorted(remembered.items(), key=lambda item: item[1])[-MAX_REMEMBERED:])
        try:
            self.config.set(SAM_SET_KEY, newest)
        except Exception:  # noqa: BLE001 - memory only makes learning safer, never required
            pass

    def sam_made(self, symbol: str) -> bool:
        return bool(symbol) and (symbol == self.sam_set or symbol in self._sam_symbols())

    # -- observations --------------------------------------------------------------------------------------------
    def observe(self, symbol: str) -> str | None:
        """Called with the chart's symbol on every chart_state. Returns the
        canonical instrument it learned for, or None."""
        if not symbol or ":" not in symbol or symbol == self.last_seen:
            return None
        previous, self.last_seen = self.last_seen, symbol
        if symbol == self.sam_set:
            return None                                   # SAM's own change, just now
        user_changed = bool(previous)                     # the chart moved and SAM did not move it
        if not user_changed and self.sam_made(symbol):
            return None                                   # first look after a start: still SAM's choice
        canonical = self.resolve(symbol)
        if not canonical:
            return None
        if user_changed:
            self._forget_sam_set(symbol)                  # the user chose it himself now
        learned = dict(self.config.get(LEARNED_KEY, {}) or {})
        if learned.get(canonical) != symbol:
            learned[canonical] = symbol
            self.config.set(LEARNED_KEY, learned)
        return canonical

    def _forget_sam_set(self, symbol: str) -> None:
        remembered = self._sam_symbols()
        if symbol in remembered:
            remembered.pop(symbol)
            try:
                self.config.set(SAM_SET_KEY, remembered)
            except Exception:  # noqa: BLE001
                pass


__all__ = ["FeedMemory", "SAM_SET_KEY", "LEARNED_KEY", "KEEP_S"]
