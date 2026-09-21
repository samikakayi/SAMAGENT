"""HTTP surface for autonomous runs: tasks, project map, environment, live feed.

Everything here is an adapter between HTTP and the orchestrator. It owns no
run state -- the orchestrator and the task store do -- and no policy. What it
does own is the plumbing an HTTP process needs around long-lived work: the
registry that keeps background drivers alive, and the fan-out that lets every
open UI watch a run.

Wired from create_app; the pre-existing /api/approvals decision endpoint
reaches in through schedule_approval() so both panels resolve the same way.
"""

from __future__ import annotations

import asyncio
from functools import partial
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import APIRouter, HTTPException, Query, WebSocket

from .autonomy import AutonomousOrchestrator, TaskAlreadyRunning
from .cancellation import CancellationManager
from .capabilities import CapabilityRegistry, probe_providers
from .config import Settings
from .db import Database
from .project_map import ProjectScanner
from .schemas import AgentTaskApproval, AgentTaskCreate
from .verification import VerificationEngine


class LiveEventHub:
    """Fan-out of agent activity to every connected UI.

    A browser that has gone away must never stall a running task, so sends
    are best-effort and a socket that raises is dropped.
    """

    def __init__(self) -> None:
        self._sockets: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def add(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._sockets.add(websocket)

    async def remove(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._sockets.discard(websocket)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            targets = list(self._sockets)
        dead: list[WebSocket] = []
        for socket in targets:
            try:
                await socket.send_json(payload)
            except Exception:  # noqa: BLE001 - any transport error means "gone"
                dead.append(socket)
        if dead:
            async with self._lock:
                for socket in dead:
                    self._sockets.discard(socket)


class AgentApi:
    """Builds the router and the helpers create_app wires in."""

    def __init__(
        self,
        *,
        settings: Settings,
        database: Database,
        orchestrator: AutonomousOrchestrator,
        scanner: ProjectScanner,
        verifier: VerificationEngine,
        capabilities: CapabilityRegistry,
        cancellation: CancellationManager,
        adapters: Callable[[], Any],
    ) -> None:
        self.settings = settings
        self.database = database
        self.orchestrator = orchestrator
        self.scanner = scanner
        self.verifier = verifier
        self.capabilities = capabilities
        self.cancellation = cancellation
        self._adapters = adapters
        self.hub = LiveEventHub()
        # Strong references to detached drivers: without them asyncio may
        # garbage-collect a run mid-flight and it would simply disappear.
        self.runs: set[asyncio.Task[Any]] = set()
        # The orchestrator emits; the API decides where emissions go.
        orchestrator.broadcast = self.hub.broadcast
        self.router = self._build_router()

    # -- shared plumbing ---------------------------------------------------
    def drive(self, coroutine: Awaitable[Any], *, task_id: str, what: str) -> None:
        """Run orchestrator work detached from the request that asked for it."""

        async def guarded() -> None:
            try:
                await coroutine
            except TaskAlreadyRunning:
                pass  # a concurrent caller already owns this task
            except Exception as exc:  # noqa: BLE001 - a crashed driver must leave a trace
                self.database.add_audit(
                    "task", "error", f"{what} crashed: {exc}",
                    details={"task_id": task_id, "error_type": type(exc).__name__},
                )

        task = asyncio.ensure_future(guarded())
        self.runs.add(task)
        task.add_done_callback(self.runs.discard)

    def schedule_approval(self, task_id: str, approval_id: str, decision: str, note: str) -> None:
        """Hand an approved/denied step back to the orchestrator, detached:
        resolving drives the run to its next pause or end, which can take
        minutes."""
        self.drive(
            self.orchestrator.resolve_approval(task_id, approval_id, decision, note),
            task_id=task_id, what="Approval resolution",
        )

    async def startup(self) -> None:
        interrupted = await self.orchestrator.mark_interrupted()
        if interrupted:
            self.database.add_audit(
                "task", "interrupted", f"{len(interrupted)} autonomous run(s) were interrupted by a restart",
                details={"task_ids": interrupted},
            )

    def shutdown(self) -> None:
        for task in list(self.runs):
            task.cancel()

    # -- routes ------------------------------------------------------------
    def _build_router(self) -> APIRouter:
        router = APIRouter()
        orchestrator = self.orchestrator
        store = orchestrator.store

        def task_or_404(task_id: str):
            task = store.get(task_id)
            if task is None:
                raise HTTPException(404, "Task not found")
            return task

        @router.post("/api/tasks", status_code=202)
        async def start_agent_task(payload: AgentTaskCreate) -> dict[str, Any]:
            """Start an autonomous run and return immediately; progress arrives
            over /ws/live and /api/tasks/{id}."""
            task = store.create(payload.goal, conversation_id=payload.conversation_id, constraints=payload.constraints)
            self.drive(orchestrator.run(task), task_id=task.id, what="Autonomous run")
            return {"task": task.public_dict(), "task_id": task.id}

        @router.get("/api/tasks")
        async def list_agent_tasks(
            limit: int = Query(default=30, ge=1, le=200), state: str | None = None,
        ) -> dict[str, Any]:
            return {"tasks": store.list(limit=limit, state=state)}

        @router.get("/api/tasks/{task_id}")
        async def get_agent_task(task_id: str) -> dict[str, Any]:
            return {"task": task_or_404(task_id).public_dict()}

        @router.post("/api/tasks/{task_id}/approvals")
        async def resolve_agent_task_approval(task_id: str, payload: AgentTaskApproval) -> dict[str, Any]:
            task_or_404(task_id)
            if self.database.get_approval(payload.approval_id) is None:
                raise HTTPException(404, "Approval not found")
            self.schedule_approval(task_id, payload.approval_id, payload.decision, payload.note)
            return {"accepted": True, "task_id": task_id, "decision": payload.decision}

        @router.post("/api/tasks/{task_id}/resume", status_code=202)
        async def resume_agent_task(task_id: str) -> dict[str, Any]:
            task = task_or_404(task_id)
            if task.terminal:
                return {"resumed": False, "task_id": task_id, "reason": "The task already finished."}
            if orchestrator.driving(task_id):
                return {"resumed": False, "task_id": task_id, "reason": "The task is already running."}
            self.drive(orchestrator.resume(task_id), task_id=task_id, what="Resume")
            return {"resumed": True, "task_id": task_id}

        @router.post("/api/tasks/{task_id}/cancel")
        async def cancel_task(task_id: str) -> dict[str, Any]:
            """Stop an autonomous run, or a chat turn by its token."""
            outcome = await orchestrator.cancel(task_id)
            if outcome is not None:
                if outcome["cancelled"]:
                    self.database.add_audit(
                        "task", "cancelled", "User stopped an autonomous run", actor="user",
                        details={"task_id": task_id},
                    )
                return outcome
            if self.cancellation.cancel(task_id, "user"):
                return {"cancelled": True, "task_id": task_id}
            raise HTTPException(404, "Active task not found")

        @router.get("/api/tasks/{task_id}/diff")
        async def get_agent_task_diff(task_id: str) -> dict[str, Any]:
            return await asyncio.to_thread(orchestrator.diff, task_or_404(task_id))

        @router.post("/api/tasks/{task_id}/rollback")
        async def rollback_agent_task(task_id: str) -> dict[str, Any]:
            outcome = await orchestrator.rollback(task_id)
            if outcome is None:
                raise HTTPException(404, "Task not found")
            return outcome

        @router.post("/api/tasks/{task_id}/retry", status_code=202)
        async def retry_agent_task(task_id: str) -> dict[str, Any]:
            task = await orchestrator.retry(task_id)
            if task is None:
                raise HTTPException(404, "Task not found")
            self.drive(orchestrator.run(task), task_id=task.id, what="Retry")
            return {"task_id": task.id, "retry_of": task_id, "task": task.public_dict()}

        @router.get("/api/project/map")
        async def get_project_map(refresh: bool = False, detail: str = "summary") -> dict[str, Any]:
            project_map = await asyncio.to_thread(
                partial(self.scanner.scan, Path(self.settings.workspace_root), force=refresh)
            )
            if detail == "full":
                return {"map": project_map.as_dict()}
            payload = project_map.as_dict()
            payload.pop("tree", None)
            payload["summary"] = project_map.summary_text()
            return {"map": payload}

        @router.get("/api/providers/resolution")
        async def get_provider_resolution(refresh: bool = False) -> dict[str, Any]:
            """Which real model the next autonomous run would use, and why.

            Costs no tokens: served from the bounded verdict cache, or from
            the provider's free catalogue and key endpoints when refreshed.
            """
            resolution = await orchestrator.health.resolve(self._adapters(), refresh=refresh)
            return {"resolution": resolution.as_dict()}

        @router.get("/api/environment")
        async def get_environment() -> dict[str, Any]:
            """Which developer tools and model providers are actually usable."""
            providers = await probe_providers(self._adapters(), self.settings)
            project_map = await asyncio.to_thread(self.scanner.scan, Path(self.settings.workspace_root))
            return {
                "tools": self.capabilities.as_dict(),
                "providers": [item.as_dict() for item in providers],
                "checks": self.verifier.available_checks(project_map),
            }

        return router
