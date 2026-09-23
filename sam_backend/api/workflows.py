"""What the Workflow Intelligence panel can ask for.

Reading is open: searching the library, inspecting a workflow and preparing an
artifact all create nothing. Importing and activating go through the same
approval path every other mutation uses, so this module never grows its own
weaker gate -- it calls the tool layer's policy rather than deciding for itself.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..workflows import WorkflowError
from .services import AppServices


class PrepareRequest(BaseModel):
    workflow_id: str = Field(min_length=1, max_length=200)
    name: str | None = Field(default=None, max_length=200)
    credential_mapping: dict[str, str] | None = None


class ImportRequest(BaseModel):
    workflow_sha256: str = Field(min_length=64, max_length=64)
    # Returned by the first call, which creates the approval. Sent back once a
    # human has decided it, through the approval endpoints every other
    # mutation already uses.
    approval_id: str | None = Field(default=None, max_length=64)


class ActivateRequest(BaseModel):
    workflow_id: str = Field(min_length=1, max_length=200)
    active: bool
    approval_id: str | None = Field(default=None, max_length=64)


def register_workflow_routes(application: FastAPI, sv: AppServices) -> None:
    workflows = sv.workflows

    def gate(tool: str, arguments: dict[str, Any], approval_id: str | None, summary: dict[str, Any]):
        """Ask for a decision, or claim one that was already made.

        This deliberately reuses the approval records, the single-use claim and
        the audit trail that every other mutation in SAM goes through. Inventing
        a second, weaker gate inside the n8n integration is exactly how an
        automation engine turns into a side door.
        """
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
            return {
                "approval_required": True, "approval_id": approval["id"],
                "reason": decision.reason, "risk_level": decision.risk_level.value, **summary,
            }
        claimed = sv.database.authorize_approval(approval_id, "approved")
        if claimed is None:
            raise HTTPException(404, {"error": "Approval not found.", "code": "NOT_FOUND"})
        # The approval must be for this exact action, not merely any approval.
        if claimed.get("tool_name") != tool or claimed.get("arguments") != arguments:
            raise HTTPException(409, {"error": "That approval was granted for a different action.",
                                      "code": "VALIDATION_FAILED"})
        # `authorize_approval` only moves pending -> executing, so a second
        # attempt finds it already spent. Recording the result below is what
        # closes that window, exactly as the tool execution path does.
        if claimed.get("status") != "executing":
            raise HTTPException(409, {"error": f"Approval is already {claimed.get('status')}.",
                                      "code": "VALIDATION_FAILED"})
        return None

    def spend(approval_id: str | None, result: dict[str, Any]) -> dict[str, Any]:
        """Mark the approval used, so it cannot authorise a second attempt."""
        if approval_id:
            sv.database.set_approval_result(approval_id, {"ok": True})
        return result

    def answer(action) -> Any:
        """One translation from SAM's workflow errors to HTTP, with no traceback."""
        try:
            return action()
        except WorkflowError as exc:
            status = {
                "NOT_CONFIGURED": 409, "NOT_FOUND": 404, "AUTH": 502,
                "VALIDATION_FAILED": 422, "INVALID_WORKFLOW": 422,
                "UNSUPPORTED_OPERATION": 501, "TOO_LARGE": 413, "RATE_LIMIT": 429,
            }.get(exc.code.value, 502)
            raise HTTPException(status, exc.as_dict()) from exc

    @application.get("/api/workflows/status")
    async def workflow_status() -> dict[str, Any]:
        return await asyncio.to_thread(workflows.status)

    @application.get("/api/workflows/search")
    async def workflow_search(
        query: str = "", category: str = "", service: str = "",
        trigger: str = "", limit: int = 5,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            lambda: answer(lambda: workflows.search(
                query, category=category, service=service, trigger=trigger, limit=limit)))

    @application.get("/api/workflows/categories")
    async def workflow_categories() -> dict[str, Any]:
        return await asyncio.to_thread(
            lambda: answer(lambda: {"categories": workflows.library.get_categories()}))

    @application.get("/api/workflows/{workflow_id}/inspect")
    async def workflow_inspect(workflow_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(
            lambda: answer(lambda: workflows.inspect_workflow(workflow_id)))

    @application.post("/api/workflows/prepare")
    async def workflow_prepare(payload: PrepareRequest) -> dict[str, Any]:
        return await asyncio.to_thread(lambda: answer(lambda: workflows.prepare_workflow(
            payload.workflow_id, name=payload.name or "",
            credential_mapping=payload.credential_mapping)))

    @application.get("/api/workflows/n8n/workflows")
    async def n8n_workflows(limit: int = 20) -> dict[str, Any]:
        return await asyncio.to_thread(
            lambda: answer(lambda: {"workflows": workflows.n8n.list_workflows(limit)}))

    @application.get("/api/workflows/n8n/credentials")
    async def n8n_credentials() -> dict[str, Any]:
        # Names, ids and types. n8n keeps the values and is never asked for them.
        return await asyncio.to_thread(
            lambda: answer(lambda: {"credentials": workflows.n8n.list_credentials()}))

    @application.get("/api/workflows/runs")
    async def workflow_runs(workflow_id: str = "", limit: int = 5) -> dict[str, Any]:
        return await asyncio.to_thread(
            lambda: answer(lambda: workflows.run_status(workflow_id, limit)))

    @application.post("/api/workflows/import")
    async def workflow_import(payload: ImportRequest) -> dict[str, Any]:
        """Create the prepared artifact in n8n, inactive.

        Approval is enforced where every other mutation's is -- the policy and
        the tool layer -- so this route asks for the decision rather than
        making one of its own.
        """
        artifact = await asyncio.to_thread(
            lambda: answer(lambda: workflows.artifact(payload.workflow_sha256)))
        pending = await asyncio.to_thread(lambda: gate(
            "workflow_import", {"workflow_sha256": payload.workflow_sha256}, payload.approval_id,
            {
                "workflow_sha256": artifact.sha256, "name": artifact.name,
                "target_instance": workflows.target,
                "risk": artifact.inspection.risk.as_dict(),
                "unresolved_credentials": [item.as_dict() for item in artifact.unresolved_credentials],
            }))
        if pending:
            return pending
        return await asyncio.to_thread(lambda: spend(
            payload.approval_id, answer(lambda: workflows.import_workflow(payload.workflow_sha256))))

    @application.post("/api/workflows/activate")
    async def workflow_activate(payload: ActivateRequest) -> dict[str, Any]:
        arguments = {"workflow_id": payload.workflow_id, "active": payload.active}
        pending = await asyncio.to_thread(lambda: gate(
            "workflow_activate", arguments, payload.approval_id,
            {"workflow_id": payload.workflow_id, "active": payload.active,
             "target_instance": workflows.target}))
        if pending:
            return pending
        return await asyncio.to_thread(lambda: spend(
            payload.approval_id,
            answer(lambda: workflows.set_active(payload.workflow_id, payload.active))))
