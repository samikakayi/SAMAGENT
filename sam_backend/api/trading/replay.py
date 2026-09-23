"""Replaying recorded history to research it, bar by bar."""

from __future__ import annotations

import asyncio

from ...schemas import ReplayControlRequest, ReplayScanRequest, ReplayStartRequest
from ..services import AppServices
from fastapi import FastAPI
from typing import Any


def register_replay_routes(application: FastAPI, sv: AppServices) -> None:
    replay = sv.replay
    @application.get("/api/research/replay/capability")
    async def replay_capability() -> dict[str, Any]:
        return replay.capability()
    @application.post("/api/research/replay/start")
    async def replay_start(payload: ReplayStartRequest) -> dict[str, Any]:
        result = await asyncio.to_thread(
            replay.start, payload.symbol, payload.timeframe,
            count=payload.count, start_offset=payload.start_offset, session_id=payload.session_id,
        )
        return result.as_dict()
    @application.post("/api/research/replay/control")
    async def replay_control(payload: ReplayControlRequest) -> dict[str, Any]:
        result = await asyncio.to_thread(replay.control, payload.action,
                                         session_id=payload.session_id, bars=payload.bars)
        return result.as_dict()
    @application.post("/api/research/replay/scan")
    async def replay_scan(payload: ReplayScanRequest) -> dict[str, Any]:
        result = await asyncio.to_thread(
            replay.run_scan, payload.trigger, session_id=payload.session_id, bars=payload.bars,
            stop_atr_multiple=payload.stop_atr_multiple, reward_multiple=payload.reward_multiple,
        )
        return result.as_dict()
