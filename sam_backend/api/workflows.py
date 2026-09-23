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


class ActivateRequest(BaseModel):
    workflow_id: str = Field(min_length=1, max_length=200)
    active: bool


def register_workflow_routes(application: FastAPI, sv: AppServices) -> None:
    workflows = sv.workflows

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
        decision = sv.policy.evaluate("workflow_import", {"workflow_sha256": payload.workflow_sha256})
        if decision.approval_required:
            artifact = await asyncio.to_thread(lambda: answer(lambda: workflows.artifact(payload.workflow_sha256)))
            return {
                "approval_required": True, "reason": decision.reason,
                "risk_level": decision.risk_level.value,
                "workflow_sha256": artifact.sha256, "name": artifact.name,
                "target_instance": workflows.target,
                "risk": artifact.inspection.risk.as_dict(),
                "unresolved_credentials": [item.as_dict() for item in artifact.unresolved_credentials],
            }
        return await asyncio.to_thread(
            lambda: answer(lambda: workflows.import_workflow(payload.workflow_sha256)))

    @application.post("/api/workflows/activate")
    async def workflow_activate(payload: ActivateRequest) -> dict[str, Any]:
        decision = sv.policy.evaluate(
            "workflow_activate", {"workflow_id": payload.workflow_id, "active": payload.active})
        if decision.approval_required:
            return {
                "approval_required": True, "reason": decision.reason,
                "risk_level": decision.risk_level.value,
                "workflow_id": payload.workflow_id, "active": payload.active,
                "target_instance": workflows.target,
            }
        return await asyncio.to_thread(
            lambda: answer(lambda: workflows.set_active(payload.workflow_id, payload.active)))
