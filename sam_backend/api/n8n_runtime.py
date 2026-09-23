"""Start, stop and inspect the one n8n SAM installed.

Three routes with no parameters at all. That is the security design, not an
oversight: there is no path, command, port or argument a caller can supply, so
there is nothing here to inject into. Start and stop are still mutations, so
they go through the same approval policy every other mutation uses rather
than growing a private one.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .services import AppServices


class RuntimeActionRequest(BaseModel):
    # The only field: the approval this action was granted, once a human has
    # decided it. Deliberately nothing else -- no path, no port, no command.
    approval_id: str | None = Field(default=None, max_length=64)


def register_n8n_runtime_routes(application: FastAPI, sv: AppServices) -> None:
    runtime = sv.n8n_runtime

    def gate(tool: str, approval_id: str | None, summary: dict[str, Any]):
        """The same approval records every other mutation in SAM uses."""
        arguments = {"action": tool}
        decision = sv.policy.evaluate(tool, arguments)
        if not decision.approval_required:
            return None
        if not approval_id:
            approval = sv.database.create_approval(
                conversation_id=None, tool_name=tool, tool_call_id=f"ui_{tool}",
                risk_level=decision.risk_level.value, reason=decision.reason,
                arguments=arguments, ttl_minutes=sv.settings.approval_ttl_minutes,
            )
            sv.database.add_audit(
                "approval", "pending", decision.reason, tool_name=tool,
                risk_level=decision.risk_level.value,
                details={"approval_id": approval["id"], **summary},
            )
            return {"approval_required": True, "approval_id": approval["id"],
                    "reason": decision.reason, "risk_level": decision.risk_level.value, **summary}
        claimed = sv.database.authorize_approval(approval_id, "approved")
        if claimed is None:
            raise HTTPException(404, {"error": "Approval not found.", "code": "NOT_FOUND"})
        if claimed.get("tool_name") != tool or claimed.get("arguments") != arguments:
            raise HTTPException(409, {"error": "That approval was granted for a different action.",
                                      "code": "VALIDATION_FAILED"})
        if claimed.get("status") != "executing":
            raise HTTPException(409, {"error": f"Approval is already {claimed.get('status')}.",
                                      "code": "VALIDATION_FAILED"})
        return None

    def spend(approval_id: str | None, result: dict[str, Any]) -> dict[str, Any]:
        if approval_id:
            sv.database.set_approval_result(approval_id, {"ok": True})
        return result

    @application.get("/api/n8n/runtime")
    async def runtime_status() -> dict[str, Any]:
        return (await asyncio.to_thread(runtime.status)).as_dict()

    @application.post("/api/n8n/runtime/start")
    async def runtime_start(payload: RuntimeActionRequest) -> dict[str, Any]:
        current = await asyncio.to_thread(runtime.status)
        pending = await asyncio.to_thread(lambda: gate(
            "n8n_runtime_start", payload.approval_id,
            {"url": current.url, "version": current.record.version,
             "runtime_path": current.record.runtime_path, "state": current.state.value}))
        if pending:
            return pending
        return spend(payload.approval_id, (await asyncio.to_thread(runtime.start)).as_dict())

    @application.post("/api/n8n/runtime/stop")
    async def runtime_stop(payload: RuntimeActionRequest) -> dict[str, Any]:
        current = await asyncio.to_thread(runtime.status)
        pending = await asyncio.to_thread(lambda: gate(
            "n8n_runtime_stop", payload.approval_id,
            {"url": current.url, "pid": current.record.pid, "state": current.state.value}))
        if pending:
            return pending
        return spend(payload.approval_id, (await asyncio.to_thread(runtime.stop)).as_dict())
