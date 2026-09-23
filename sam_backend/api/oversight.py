"""What a human approves, audits and remembers."""

from __future__ import annotations

from ..schemas import ApprovalDecision, MemoryCreate
from .services import AppServices
from fastapi import HTTPException
from fastapi import Query
from typing import Any


def register_oversight_routes(application: FastAPI, sv: AppServices) -> None:
    database = sv.database
    policy = sv.policy
    agent = sv.agent
    agent_api = sv.agent_api
    @application.get("/api/approvals")
    async def list_approvals(status: str | None = Query(default=None), limit: int = Query(100, ge=1, le=500)) -> dict[str, Any]:
        records = database.list_approvals(status, limit)
        return {"approvals": [agent.public_approval(record) for record in records]}

    @application.get("/api/approvals/{approval_id}")
    async def get_approval(approval_id: str) -> dict[str, Any]:
        record = database.get_approval(approval_id)
        if record is None:
            raise HTTPException(404, "Approval not found")
        return {"approval": agent.public_approval(record)}

    @application.post("/api/approvals/{approval_id}/decision")
    async def decide_approval(approval_id: str, payload: ApprovalDecision) -> dict[str, Any]:
        """Resolve any pending approval, from whichever panel raised it.

        Chat turns and autonomous runs both queue approvals into one table, so
        this dispatches on the recorded owner rather than assuming a
        conversation exists.
        """
        record = database.get_approval(approval_id)
        if record is None:
            raise HTTPException(404, "Approval not found")
        owning_task = record.get("task_id")
        if owning_task:
            agent_api.schedule_approval(owning_task, approval_id, payload.decision, payload.note)
            return {
                "approval": agent.public_approval(database.get_approval(approval_id) or {}),
                "task_id": owning_task,
                "accepted": True,
            }
        try:
            result = await agent.resolve_approval(approval_id, payload.decision, payload.note)
            record = database.get_approval(approval_id)
            return {"approval": agent.public_approval(record or {}), "agent_response": result}
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @application.get("/api/audit")
    async def audit(limit: int = Query(200, ge=1, le=1000), event_type: str | None = None) -> dict[str, Any]:
        return {"entries": database.list_audit(limit, event_type), "chain_valid": database.verify_audit_chain()}

    @application.get("/api/audit/verify")
    async def verify_audit() -> dict[str, Any]:
        return {"valid": database.verify_audit_chain()}

    @application.get("/api/memories")
    async def memories(query: str = "", limit: int = Query(20, ge=1, le=100)) -> dict[str, Any]:
        return {"memories": database.list_memories(query, limit)}

    @application.post("/api/memories", status_code=201)
    async def create_memory(payload: MemoryCreate) -> dict[str, Any]:
        if policy.contains_embedded_secret({"content": payload.content}):
            raise HTTPException(400, "SAM will not store likely credentials in durable memory")
        memory = database.add_memory(payload.content, payload.tags, payload.importance, payload.source_conversation_id, payload.domain)
        database.add_audit("memory", "created", "Local memory created", actor="user", details={"memory_id": memory["id"], "tags": payload.tags})
        return {"memory": memory}

    @application.delete("/api/memories/{memory_id}")
    async def delete_memory(memory_id: str) -> dict[str, Any]:
        if not database.delete_memory(memory_id):
            raise HTTPException(404, "Memory not found")
        database.add_audit("memory", "deleted", "Local memory deleted", actor="user", details={"memory_id": memory_id})
        return {"deleted": True, "id": memory_id}
