"""SAM 2 trading engine: deterministic analysis over closed candles.

Ported from v1 ``sam_backend/trading/{types,indicators,patterns,analysis,
analyst,geometry,research,replay}.py`` with the audit fixes of
reports/trading-intelligence.json (UTC bars, real order blocks, report keys
the drawing plan reads, Sorani summaries). No module here does I/O except
``core.Engine.fetch_mt5``; everything else is pure and unit-tested.

``Engine`` is imported lazily: ``sam.trading.predicates`` imports the pure
modules of this package, and ``core`` imports ``predicates``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["Engine"]


def __getattr__(name: str) -> Any:
    if name == "Engine":
        from .core import Engine
        return Engine
    raise AttributeError(name)
