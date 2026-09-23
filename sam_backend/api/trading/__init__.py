"""The trading API, split by what each part is responsible for.

One entry point, because the composition root should not have to know how
many parts there are -- only that the trading surface gets registered.
"""

from __future__ import annotations

from fastapi import FastAPI

from ..services import AppServices
from .chart import register_chart_routes
from .market import register_market_routes
from .record import register_record_routes
from .replay import register_replay_routes
from .strategies import register_strategy_routes

__all__ = ["register_trading_routes"]


def register_trading_routes(application: FastAPI, sv: AppServices) -> None:
    register_market_routes(application, sv)
    register_record_routes(application, sv)
    register_chart_routes(application, sv)
    register_strategy_routes(application, sv)
    register_replay_routes(application, sv)
