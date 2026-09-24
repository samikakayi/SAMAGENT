"""Which chart drawings are SAM's: pruning with patience, and re-adoption.

The ``drawings`` table holds the TradingView id of everything SAM drew. Two
live findings (repair review 2026-09-24, live_idchange2.py / live_analyze.py
on PEPPERSTONE:XAUUSD):

1. Right after a symbol change, bars are loaded but the drawings sometimes are
   not yet (one round trip: the line did not reappear within 8 s; later trips
   18-172 ms). The old rule forgot every owned row whose id was missing from
   ``getAllShapes()`` for the symbol on screen -- and chart_state runs right
   after set_symbol, in analyze_market and every 2 s in the alert monitor --
   so SAM "lost" its lines and «بیانسڕەوە» answered that there were none
   while the lines stayed on the user's chart.
2. TradingView sometimes re-creates drawings with NEW ids after a round trip
   (a horizontal line's stored time moved from 1790258400 to 1789978500).

So a missing row is only forgotten after ``MISSES`` observations spread over
``MISS_SPAN_S`` and never within ``SETTLE_S`` of a symbol change or while
the chart loads; and before anything is forgotten or cleared, shapes on the
chart that match a missing row (same kind, same label text, same prices) are
re-adopted under their new id. User drawings never match: SAM's labels are
its own words («بەرگری M15», «ئامانجی 1») and the prices must be equal.
"""

from __future__ import annotations

import time
from typing import Any

SETTLE_S = 10.0       # after a symbol change: drawings may still be loading
MISSES = 3            # consecutive misses before a missing owned row is forgotten
MISS_SPAN_S = 6.0     # ... spread over at least this long


def _prices(points: Any) -> list[float]:
    out = []
    for point in points or []:
        try:
            out.append(float(point.get("price")))
        except (AttributeError, TypeError, ValueError):
            continue
    return out


def _same_prices(a: list[float], b: list[float]) -> bool:
    if not a or len(a) != len(b):
        return False
    return all(abs(x - y) <= max(1e-6, abs(x) * 1e-7) for x, y in zip(a, b))


class Ownership:
    """Per-bridge memory of symbol changes and misses (in memory: a restart
    starts with a clean slate, which only makes pruning slower, never wrong)."""

    def __init__(self, clock: Any = time.monotonic) -> None:
        self._clock = clock
        self.changed_at = 0.0
        self._symbol = ""
        self._misses: dict[int, tuple[int, float]] = {}

    def note_symbol(self, symbol: str) -> None:
        if symbol and symbol != self._symbol:
            self._symbol = symbol
            self.changed_at = self._clock()

    def settling(self) -> bool:
        return self._clock() - self.changed_at < SETTLE_S

    def forget_now(self, db_id: int) -> None:
        self._misses.pop(db_id, None)

    def review(self, rows: list[dict[str, Any]], present: set[str], symbol: str, *,
               loading: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """(missing rows on this symbol that may be re-adopted, rows to forget now)."""
        self.note_symbol(symbol)
        missing = [r for r in rows if r["tv_id"] not in present and r["symbol"] == symbol]
        for row in rows:
            if row["tv_id"] in present:
                self._misses.pop(row["db_id"], None)
        if not missing:
            return [], []
        patient = loading or self.settling()
        forget: list[dict[str, Any]] = []
        now = self._clock()
        for row in missing:
            if patient:
                continue
            count, first = self._misses.get(row["db_id"], (0, now))
            count += 1
            self._misses[row["db_id"]] = (count, first)
            if count >= MISSES and now - first >= MISS_SPAN_S:
                forget.append(row)
                self._misses.pop(row["db_id"], None)
        return missing, forget


def match_shapes(missing: list[dict[str, Any]], shapes: list[dict[str, Any]],
                 owned_ids: set[str]) -> dict[int, str]:
    """{db_id: new tv_id} for missing rows whose drawing is on the chart under
    another id (same label text and same prices; never an id SAM already owns)."""
    taken = set(owned_ids)
    found: dict[int, str] = {}
    for row in missing:
        want_prices = _prices(row.get("points"))
        label = str(row.get("text") or "")
        for shape in shapes:
            sid = str(shape.get("id"))
            if sid in taken:
                continue
            if label and str(shape.get("text") or "") != label:
                continue
            if not _same_prices(want_prices, _prices(shape.get("points"))):
                continue
            found[int(row["db_id"])] = sid
            taken.add(sid)
            break
    return found


__all__ = ["Ownership", "match_shapes", "SETTLE_S", "MISSES", "MISS_SPAN_S"]
