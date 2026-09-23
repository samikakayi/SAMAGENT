"""The machine SAM runs on: terminal, desktop control and the live socket."""

from __future__ import annotations

import asyncio
import json
import time

from ..contracts import ExecutionStatus
from .services import AppServices
from fastapi import Request
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from pathlib import Path
from typing import Any


def register_desktop_routes(application: FastAPI, sv: AppServices) -> None:
    settings = sv.settings
    database = sv.database
    cancellation = sv.cancellation
    trading = sv.trading
    windows = sv.windows
    task_store = sv.task_store
    agent_api = sv.agent_api
    @application.get("/api/mt5")
    async def api_mt5() -> dict[str, Any]:
        result = await asyncio.to_thread(trading.market_snapshot, "XAUUSD", ["M1", "M5", "M15", "H1", "H4", "D1", "W1", "MN1"])
        if not result.data:
            return {"connected": False, "mt5_error": result.error}
        data = result.data
        first = next(iter(data["timeframes"].values()))["metadata"]
        point = float(first.get("point") or 1)
        ohlc = {
            timeframe: {"o": item["open"], "h": item["high"], "l": item["low"], "c": item["close"], "time": item["time"]}
            for timeframe, item in data["timeframes"].items()
        }
        return {
            "connected": True,
            "broker": data["feed"],
            "symbol": data["symbol"],
            "bid": data["bid"],
            "ask": data["ask"],
            "spread": round(data["spread"] / point, 2) if data["spread"] is not None and point else None,
            "ohlc": ohlc,
            "session": ", ".join(data["session"]["active"]) or "CLOSED/TRANSITION",
            "provider_metadata": first,
            "verified": result.verified,
        }

    @application.get("/api/desktop/status")
    async def api_desktop_status() -> dict[str, Any]:
        # Only these three matter here, and asking for them by name avoids
        # opening a handle to every process on the machine.
        wanted = ("tradingview", "terminal64", "python")
        # observe() makes blocking Win32 calls; on the event loop it stalled
        # every other request behind it.
        process_result, state = await asyncio.gather(
            asyncio.to_thread(windows.list_processes, "", wanted),
            asyncio.to_thread(trading.tradingview.observe),
        )
        processes = []
        if isinstance(process_result.data, dict):
            processes = process_result.data.get("processes", [])
        return {
            "processes": processes,
            "tradingview": {
                "running": state.running,
                "windows": [{"hwnd": state.window_handle, "title": state.title}] if state.window_handle else [],
                "interactive": state.interactive,
                "symbol": state.symbol,
                "timeframe": state.timeframe,
                "timeframe_verified": state.timeframe_verified,
            },
        }

    @application.post("/api/abort")
    async def api_abort() -> dict[str, Any]:
        outcome = cancellation.emergency_stop("emergency_stop")
        stopped_monitors = 0
        for setup in database.list_trading_setups(500):
            if setup["monitor_enabled"] and database.set_setup_monitoring(setup["id"], False):
                stopped_monitors += 1
        database.add_audit("emergency_stop", "executed", "Emergency stop cancelled active work", actor="user", details={**outcome, "stopped_monitors": stopped_monitors})
        return {"aborted": True, **outcome, "stopped_monitors": stopped_monitors}

    @application.post("/api/desktop/action")
    async def api_desktop_action(request: Request) -> dict[str, Any]:
        payload = await request.json()
        action = str(payload.get("action", ""))
        if action == "focus_tradingview":
            trading.refresh_permissions()
            result = trading.tradingview.launch()
            return {"ok": result.status in {ExecutionStatus.SUCCESS, ExecutionStatus.PARTIAL}, "action": action, "error": result.error, "verified": result.verified}
        if action == "capture_screen":
            windows.refresh_permissions(computer_control=settings.computer_control_enabled, screen_access=settings.screen_access_enabled)
            result = windows.capture_screen()
            response = {"ok": result.status in {ExecutionStatus.SUCCESS, ExecutionStatus.PARTIAL}, "action": action, "error": result.error, "verified": result.verified}
            if result.data and isinstance(result.data, dict) and result.data.get("path"):
                import base64
                response["image_b64"] = base64.b64encode(Path(result.data["path"]).read_bytes()).decode("ascii")
            return response
        return {"ok": False, "error": f"Unknown action {action}"}
    @application.websocket("/ws/live")
    async def live_socket(websocket: WebSocket):
        origin = websocket.headers.get("origin")
        client_host = (websocket.client.host if websocket.client else "").lower()
        if (origin and origin not in settings.cors_origins) or client_host not in {"127.0.0.1", "::1", "testclient"}:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        await agent_api.hub.add(websocket)
        await websocket.send_json({
            "type": "state",
            "state": "IDLE",
            "tasks": cancellation.snapshot(),
            "agent_tasks": task_store.list(limit=10),
            "trading": database.get_trading_context(),
        })
        try:
            while True:
                message = await websocket.receive_json()
                message_type = str(message.get("type", "ping"))
                if message_type == "cancel":
                    task_id = str(message.get("task_id", ""))
                    await websocket.send_json({"type": "cancelled", "task_id": task_id, "cancelled": cancellation.cancel(task_id, "live_socket")})
                elif message_type == "emergency_stop":
                    await websocket.send_json({"type": "emergency_stop", **cancellation.emergency_stop("live_socket")})
                else:
                    await websocket.send_json({
                        "type": "state",
                        "state": "IDLE" if not cancellation.snapshot()["active_tasks"] else "ACTING",
                        "tasks": cancellation.snapshot(),
                        "trading": database.get_trading_context(),
                    })
        except WebSocketDisconnect:
            return
        finally:
            await agent_api.hub.remove(websocket)
