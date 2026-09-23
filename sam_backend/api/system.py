"""Is this install healthy, what can it do, and what is in the workspace."""

from __future__ import annotations

import importlib.util

import asyncio
import time

from ..config import is_elevated_windows_process
from ..dpi import status as dpi_status
from .services import AppServices
from fastapi import HTTPException
from fastapi import Query
from typing import Any


def register_system_routes(application: FastAPI, sv: AppServices) -> None:
    settings = sv.settings
    database = sv.database
    policy = sv.policy
    cancellation = sv.cancellation
    tools = sv.tools
    @application.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "name": "SAM",
            "version": "2.0.0",
            "uptime_seconds": round(time.monotonic() - application.state.started_at, 2),
            "workspace": str(settings.workspace_root),
            "database": "ok",
            "audit_chain_valid": database.verify_audit_chain(),
            "running_elevated": is_elevated_windows_process(),
            "screen_coordinates": dpi_status(),
            "model_mode": settings.model_mode,
            "computer_control": settings.computer_control_enabled,
            "screen_access": settings.screen_access_enabled,
            "active_tasks": cancellation.snapshot(),
        }

    @application.get("/api/config")
    async def public_config() -> dict[str, Any]:
        return {
            **settings.public_dict(),
            "name": "SAM",
            "capabilities": {
                "chat": True, "voice_input": "browser-realtime-vad", "voice_output": "browser-streamed-chunks", "persistent_memory": True,
                "files": True, "terminal": True, "python": True, "browser": True, "apps": True,
                "planning": True, "approvals": True, "audit": True, "trading_brain": True,
                "metatrader5_read_only": importlib.util.find_spec("MetaTrader5") is not None,
                "tradingview_observer": True, "emergency_stop": True, "model_router": True,
                "openrouter": bool(settings.openrouter_api_key), "litellm_gateway": True,
            },
            "security": {
                "loopback_only": True, "workspace_containment": True, "single_use_approvals": True,
                "audit_hash_chain": True, "safe_mode": not settings.allow_unsafe_system_actions,
            },
        }

    @application.get("/api/capabilities")
    async def capabilities() -> dict[str, Any]:
        config = await public_config()
        return {"capabilities": config["capabilities"], "security": config["security"]}
    @application.get("/api/workspace/tree")
    async def workspace_tree(path: str = ".", max_depth: int = Query(2, ge=0, le=8)) -> dict[str, Any]:
        decision = policy.evaluate("list_files", {"path": path, "max_depth": max_depth})
        if not decision.allowed or decision.approval_required:
            raise HTTPException(403, decision.reason)
        result = await asyncio.to_thread(tools.execute, "list_files", {"path": path, "max_depth": max_depth}, approved=False)
        if not result.ok:
            raise HTTPException(400, result.error)
        return result.output
    @application.get("/api/tools/manifests")
    async def tool_manifests() -> dict[str, Any]:
        return {"tools": tools.manifests}
