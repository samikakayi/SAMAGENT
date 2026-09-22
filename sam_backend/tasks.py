"""Autonomous task state: the durable record of one goal being worked on.

A chat turn is ephemeral; an autonomous run is not. It spans many tool calls,
may pause for approval, and must survive a UI refresh or a backend restart.
Every state transition is persisted so an interrupted run can be inspected and
resumed instead of silently vanishing.

Transitions are validated, not merely recorded: an illegal jump (say
COMPLETED -> EXECUTING) means the orchestrator has a bug, and surfacing that
immediately is better than persisting an incoherent history.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from .db import Database


class TaskState(StrEnum):
    IDLE = "IDLE"
    UNDERSTANDING = "UNDERSTANDING"
    SCANNING = "SCANNING"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    OBSERVING = "OBSERVING"
    VALIDATING = "VALIDATING"
    FIXING = "FIXING"
    RETRYING = "RETRYING"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_STATES = {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED}

# The orchestrator's legal moves. Anything outside this graph is a bug.
ALLOWED_TRANSITIONS: dict[TaskState, set[TaskState]] = {
    TaskState.IDLE: {TaskState.UNDERSTANDING, TaskState.CANCELLED, TaskState.FAILED},
    TaskState.UNDERSTANDING: {TaskState.SCANNING, TaskState.PLANNING, TaskState.FAILED, TaskState.CANCELLED},
    TaskState.SCANNING: {TaskState.PLANNING, TaskState.FAILED, TaskState.CANCELLED},
    TaskState.PLANNING: {TaskState.EXECUTING, TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED},
    TaskState.EXECUTING: {
        TaskState.OBSERVING, TaskState.WAITING_FOR_APPROVAL, TaskState.VALIDATING,
        TaskState.PLANNING, TaskState.FAILED, TaskState.CANCELLED,
    },
    TaskState.OBSERVING: {
        TaskState.EXECUTING, TaskState.VALIDATING, TaskState.FIXING, TaskState.PLANNING,
        TaskState.FAILED, TaskState.CANCELLED,
    },
    TaskState.VALIDATING: {
        TaskState.COMPLETED, TaskState.FIXING, TaskState.EXECUTING, TaskState.PLANNING,
        TaskState.FAILED, TaskState.CANCELLED,
    },
    TaskState.FIXING: {TaskState.RETRYING, TaskState.EXECUTING, TaskState.FAILED, TaskState.CANCELLED},
    TaskState.RETRYING: {
        TaskState.EXECUTING, TaskState.VALIDATING, TaskState.PLANNING,
        TaskState.FAILED, TaskState.CANCELLED,
    },
    TaskState.WAITING_FOR_APPROVAL: {
        TaskState.EXECUTING, TaskState.OBSERVING, TaskState.PLANNING,
        TaskState.FAILED, TaskState.CANCELLED,
    },
    TaskState.COMPLETED: set(),
    TaskState.FAILED: set(),
    TaskState.CANCELLED: set(),
}


class IllegalTransition(RuntimeError):
    """Raised when the orchestrator attempts a move outside the state graph."""


@dataclass(slots=True)
class TaskStep:
    """One planned unit of work, with its own outcome."""

    index: int
    text: str
    kind: str = "action"
    status: str = "pending"  # pending | active | done | failed | skipped
    tool: str | None = None
    detail: str = ""
    started_at: float | None = None
    finished_at: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TaskEvent:
    """A timeline entry. This is what the activity UI renders."""

    kind: str  # state | thought | action | tool | result | error | fix | fallback | success | approval
    message: str
    at: float = field(default_factory=time.time)
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentTask:
    """The full record of one autonomous run."""

    id: str
    goal: str
    state: TaskState = TaskState.IDLE
    conversation_id: str | None = None
    constraints: list[str] = field(default_factory=list)
    plan: list[TaskStep] = field(default_factory=list)
    events: list[TaskEvent] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    # Every validation attempt, plus the raw output of the agent's own
    # run_tests calls -- the history a re-planning run leaves behind.
    test_results: list[dict[str, Any]] = field(default_factory=list)
    # The verdict that decided completion: the last verification report, or
    # None when the run ended before anything could be verified.
    verification: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    tool_calls: int = 0
    retries: int = 0
    replans: int = 0
    completion_status: str = ""
    summary: str = ""
    # Pre-edit snapshots of every file the run touched, so its work can be
    # undone precisely -- back to the state the user had, not to git HEAD.
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    # Paths the user already had uncommitted changes in when the run began.
    # The agent's own diff is always reported separately from these.
    preexisting_changes: list[str] = field(default_factory=list)
    rolled_back: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def current_step(self) -> TaskStep | None:
        return next((step for step in self.plan if step.status in {"pending", "active"}), None)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        payload["terminal"] = self.terminal
        return payload

    def public_dict(self, *, event_limit: int = 200) -> dict[str, Any]:
        """UI-facing view: the timeline is capped so one long run cannot
        balloon a websocket frame."""
        payload = self.as_dict()
        payload["events"] = payload["events"][-event_limit:]
        payload["event_count"] = len(self.events)
        return payload


VERIFIED_STATUS = "completed_verified"


def _unevidenced_claim(completion_status: Any, verification: Any) -> bool:
    """A record asserting it was verified with nothing to show for it."""
    return completion_status == VERIFIED_STATUS and verification is None


class TaskStore:
    """Persistence and transition validation for AgentTask records."""

    def __init__(self, database: Database) -> None:
        self.database = database
        self._migrate()

    def _migrate(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS agent_tasks (
            id TEXT PRIMARY KEY,
            conversation_id TEXT,
            goal TEXT NOT NULL,
            state TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_agent_tasks_updated ON agent_tasks(updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_agent_tasks_state ON agent_tasks(state);
        """
        with self.database.write() as connection:
            connection.executescript(schema)

    # -- lifecycle ---------------------------------------------------------
    def create(
        self, goal: str, *, conversation_id: str | None = None, constraints: list[str] | None = None,
    ) -> AgentTask:
        goal = goal.strip()
        if not goal:
            raise ValueError("A task needs a goal")
        task = AgentTask(
            id=f"task_{uuid.uuid4().hex}",
            goal=goal,
            conversation_id=conversation_id,
            constraints=list(constraints or []),
        )
        task.events.append(TaskEvent("state", "Task created", detail={"state": TaskState.IDLE.value}))
        self.save(task)
        return task

    def save(self, task: AgentTask) -> AgentTask:
        with self.database.write() as connection:
            # The one authoritative rule: a run may not claim it was verified
            # without the report that says so. The sole exception is a row
            # that already made that claim before the rule existed -- it has
            # to stay writable or rolling it back would fail -- and whether
            # that applies is read from the stored row, never from the
            # caller, so no field on the task in hand can grant it. Repairing
            # such a row therefore ends the exemption by itself.
            if _unevidenced_claim(task.completion_status, task.verification):
                row = connection.execute(
                    "SELECT payload_json FROM agent_tasks WHERE id = ?", (task.id,)
                ).fetchone()
                stored = json.loads(row["payload_json"]) if row else {}
                if not _unevidenced_claim(stored.get("completion_status"), stored.get("verification")):
                    raise ValueError(
                        f"{task.id} cannot be saved as {VERIFIED_STATUS} "
                        "without the verification report that earned it"
                    )
            task.updated_at = time.time()
            payload = json.dumps(task.as_dict())
            connection.execute(
                """
                INSERT INTO agent_tasks (id, conversation_id, goal, state, payload_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    conversation_id=excluded.conversation_id,
                    goal=excluded.goal,
                    state=excluded.state,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    task.id, task.conversation_id, task.goal, task.state.value, payload,
                    _iso(task.created_at), _iso(task.updated_at),
                ),
            )
        return task

    def get(self, task_id: str) -> AgentTask | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM agent_tasks WHERE id = ?", (task_id,)
            ).fetchone()
        return _from_payload(row["payload_json"]) if row else None

    def list(self, *, limit: int = 50, state: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT payload_json FROM agent_tasks"
        params: list[Any] = []
        if state:
            query += " WHERE state = ?"
            params.append(state)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with self.database.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        tasks = [_from_payload(row["payload_json"]) for row in rows]
        # A list view only needs headline fields; timelines can be large.
        return [
            {
                "id": task.id, "goal": task.goal, "state": task.state.value,
                "completion_status": task.completion_status, "summary": task.summary,
                "modified_files": task.modified_files, "tool_calls": task.tool_calls,
                "retries": task.retries, "errors": task.errors[-3:],
                "steps_total": len(task.plan),
                "steps_done": sum(1 for step in task.plan if step.status == "done"),
                "created_at": task.created_at, "updated_at": task.updated_at,
                "terminal": task.terminal,
            }
            for task in tasks
        ]

    def resumable(self) -> list[AgentTask]:
        """Tasks that were mid-flight when the process stopped."""
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM agent_tasks WHERE state NOT IN (?, ?, ?) ORDER BY updated_at DESC",
                (TaskState.COMPLETED.value, TaskState.FAILED.value, TaskState.CANCELLED.value),
            ).fetchall()
        return [_from_payload(row["payload_json"]) for row in rows]

    # -- mutation ----------------------------------------------------------
    def transition(self, task: AgentTask, state: TaskState, message: str = "", **detail: Any) -> AgentTask:
        if state != task.state and state not in ALLOWED_TRANSITIONS[task.state]:
            raise IllegalTransition(f"{task.state.value} -> {state.value} is not a legal task transition")
        task.state = state
        task.events.append(
            TaskEvent("state", message or f"State: {state.value}", detail={"state": state.value, **detail})
        )
        return self.save(task)

    def record(self, task: AgentTask, kind: str, message: str, **detail: Any) -> AgentTask:
        task.events.append(TaskEvent(kind, message, detail=detail))
        return self.save(task)

    def set_plan(self, task: AgentTask, steps: list[TaskStep]) -> AgentTask:
        task.plan = steps
        task.events.append(
            TaskEvent("plan", f"Plan with {len(steps)} step(s)", detail={"steps": [step.as_dict() for step in steps]})
        )
        return self.save(task)

    def note_files(self, task: AgentTask, paths: list[str]) -> AgentTask:
        for path in paths:
            if path and path not in task.modified_files:
                task.modified_files.append(path)
        return self.save(task)


def _iso(value: float) -> str:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(value, UTC).isoformat()


def _from_payload(payload: str) -> AgentTask:
    data = json.loads(payload)
    task = AgentTask(
        id=data["id"],
        goal=data["goal"],
        state=TaskState(data["state"]),
        conversation_id=data.get("conversation_id"),
        constraints=list(data.get("constraints") or []),
        plan=[TaskStep(**step) for step in data.get("plan") or []],
        events=[TaskEvent(**event) for event in data.get("events") or []],
        observations=list(data.get("observations") or []),
        modified_files=list(data.get("modified_files") or []),
        test_results=list(data.get("test_results") or []),
        verification=data.get("verification"),
        errors=list(data.get("errors") or []),
        tool_calls=int(data.get("tool_calls") or 0),
        retries=int(data.get("retries") or 0),
        replans=int(data.get("replans") or 0),
        completion_status=str(data.get("completion_status") or ""),
        summary=str(data.get("summary") or ""),
        checkpoints=list(data.get("checkpoints") or []),
        preexisting_changes=list(data.get("preexisting_changes") or []),
        rolled_back=bool(data.get("rolled_back", False)),
        created_at=float(data.get("created_at") or time.time()),
        updated_at=float(data.get("updated_at") or time.time()),
    )
    return task
