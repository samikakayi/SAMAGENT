"""The trading record: setups SAM found, and what the user wrote down.

Almost entirely persistence. A setup is created from the last verified
analysis and may be watched; a journal entry is the user's own note.

Creating a setup depends on market analysis having run: the last verified
result lives on the shared TradingService, not here, and POST /setups
refuses with a 409 when there is none. That ordering is why both groups
must keep taking the same service instance rather than building their own.
"""

from __future__ import annotations

from ...schemas import JournalCreate, SetupCreateRequest, SetupMonitorRequest
from ..services import AppServices
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Query
from typing import Any


def register_record_routes(application: FastAPI, sv: AppServices) -> None:
    database = sv.database
    trading = sv.trading
    @application.get("/api/trading/setups")
    async def trading_setups(limit: int = Query(100, ge=1, le=500)) -> dict[str, Any]:
        return {"setups": database.list_trading_setups(limit)}
    @application.post("/api/trading/setups", status_code=201)
    async def create_trading_setup(payload: SetupCreateRequest) -> dict[str, Any]:
        try:
            setup = trading.create_setup_from_last_analysis(payload.theory)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        database.add_audit("trading_setup", "created", "Setup saved from the latest verified analysis", actor="user", details={"setup_id": setup["id"], "theory": payload.theory})
        return {"setup": setup}
    @application.get("/api/trading/setups/{setup_id}")
    async def trading_setup(setup_id: str) -> dict[str, Any]:
        setup = database.get_trading_setup(setup_id)
        if setup is None:
            raise HTTPException(404, "Setup not found")
        return {"setup": setup, "events": database.list_setup_events(setup_id)}
    @application.post("/api/trading/setups/{setup_id}/monitor")
    async def monitor_setup(setup_id: str, payload: SetupMonitorRequest) -> dict[str, Any]:
        setup = database.set_setup_monitoring(setup_id, payload.enabled)
        if setup is None:
            raise HTTPException(404, "Setup not found")
        database.add_audit("setup_monitor", "started" if payload.enabled else "stopped", "Setup monitoring changed", actor="user", details={"setup_id": setup_id, "enabled": payload.enabled})
        return {"setup": setup}
    @application.get("/api/trading/journal")
    async def journal(query: str = "", limit: int = Query(100, ge=1, le=500)) -> dict[str, Any]:
        return {"entries": database.list_trading_journal(query, limit)}
    @application.post("/api/trading/journal", status_code=201)
    async def create_journal_entry(payload: JournalCreate) -> dict[str, Any]:
        return {"entry": database.add_trading_journal(payload.symbol, payload.theory, payload.payload, payload.setup_id)}
