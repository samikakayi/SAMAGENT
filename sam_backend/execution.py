"""One owner of "safely run a tool call".

The chat loop and the autonomous orchestrator both need the same sequence:
evaluate policy, queue an approval when required, re-check a resolved
approval (single use, hash-bound to the exact request, policy evaluated
again), execute, audit, and keep sensitive output out of anything a model or
a UI will see. When each loop carried its own copy they drifted -- an
orchestrator approval could not be resolved from the chat endpoint, and a
re-approved approval would have replayed its tool. This module is now the
only place that sequence lives; the loops decide what to *say* about an
outcome, never how to reach it.

Data flow: ToolCall -> gate() -> (blocked | awaiting approval | allowed)
                              -> execute() -> executed outcome
           approval id -> resolve() -> (denied | mismatch | blocked | executed)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .config import Settings
from .db import Database
from .models import ToolCall
from .policy import PolicyDecision, RiskPolicy
from .tools import ToolRegistry, ToolResult

BeforeExecute = Callable[[ToolCall], Awaitable[None]]


class ApprovalStateError(RuntimeError):
    """The approval is not in the state the decision requires (already
    decided, expired). Raised rather than returned because acting on it would
    be a replay, and every caller must refuse the same way."""


@dataclass(slots=True)
class ToolOutcome:
    """What happened to one tool call, in terms both loops understand."""

    call: ToolCall
    decision: PolicyDecision
    status: str  # allowed | blocked | awaiting_approval | executed | denied | binding_mismatch | policy_blocked
    result: ToolResult | None = None
    approval: dict[str, Any] | None = None
    safe_arguments: dict[str, Any] = field(default_factory=dict)

    @property
    def approval_id(self) -> str | None:
        return self.approval["id"] if self.approval else None

    @property
    def sensitive(self) -> bool:
        return bool(self.decision.sensitive or (self.result is not None and self.result.sensitive))

    @property
    def ok(self) -> bool:
        return self.status == "executed" and self.result is not None and self.result.ok

    @property
    def error(self) -> str | None:
        return self.result.error if self.result is not None else None

    def public_result(self) -> dict[str, Any]:
        """The result as the UI and the audit trail may see it."""
        return ToolExecutor.public_result(self.result) if self.result is not None else {}

    def model_text(self) -> str:
        """The result as the model (and persisted history) may see it."""
        if self.result is None:
            return ""
        if self.sensitive:
            return "[Sensitive tool result omitted from persistent history.]"
        return self.result.model_text()


class ToolExecutor:
    """Policy, approvals, execution and audit for tool calls -- once."""

    _AUDITED_RESULT_KEYS = frozenset({
        "path", "bytes", "created", "overwritten", "deleted", "recoverable", "backup",
        "before_sha256", "after_sha256", "exit_code", "cwd", "url", "opened", "application", "pid",
    })

    def __init__(self, settings: Settings, database: Database, tools: ToolRegistry, policy: RiskPolicy) -> None:
        self.settings = settings
        self.database = database
        self.tools = tools
        self.policy = policy

    # -- shaping ---------------------------------------------------------
    @staticmethod
    def public_result(result: ToolResult) -> dict[str, Any]:
        if result.sensitive:
            return {"ok": result.ok, "output": "[Sensitive output hidden]", "error": result.error,
                    "sensitive": True, "truncated": result.truncated}
        return result.as_dict()

    def public_approval(self, record: dict[str, Any]) -> dict[str, Any]:
        """An approval record with its arguments sanitised and its result reduced."""
        return {
            **{key: value for key, value in record.items() if key not in {"arguments", "result"}},
            "arguments": self.policy.sanitize_arguments(record.get("arguments") or {}),
            "result": None if record.get("result") is None else {
                "ok": record["result"].get("ok"), "error": record["result"].get("error"),
                "sensitive": record["result"].get("sensitive", False),
            },
        }

    @classmethod
    def _audit_metadata(cls, result: ToolResult) -> dict[str, Any]:
        if result.sensitive or not isinstance(result.output, dict):
            return {"sensitive": result.sensitive}
        return {key: value for key, value in result.output.items() if key in cls._AUDITED_RESULT_KEYS}

    # -- the sequence ----------------------------------------------------
    async def gate(self, call: ToolCall, *, conversation_id: str | None, task_id: str | None = None) -> ToolOutcome:
        """Decide: run now, ask first, or refuse. Nothing is executed here."""
        decision = self.policy.evaluate(call.name, call.arguments)
        safe = self.policy.sanitize_arguments(call.arguments)
        scope = {"task_id": task_id} if task_id else {}

        if not decision.allowed:
            self.database.add_audit(
                "tool", "blocked", decision.reason, conversation_id=conversation_id, tool_name=call.name,
                risk_level=decision.risk_level.value, details={**scope, "arguments": safe},
            )
            return ToolOutcome(
                call, decision, "blocked", safe_arguments=safe,
                result=ToolResult(False, error=decision.reason, sensitive=decision.sensitive),
            )

        if decision.approval_required:
            approval = self.database.create_approval(
                conversation_id=conversation_id, tool_name=call.name, tool_call_id=call.id,
                risk_level=decision.risk_level.value, reason=decision.reason, arguments=call.arguments,
                ttl_minutes=self.settings.approval_ttl_minutes, task_id=task_id,
            )
            self.database.add_audit(
                "approval", "pending", decision.reason, conversation_id=conversation_id, tool_name=call.name,
                risk_level=decision.risk_level.value,
                details={**scope, "approval_id": approval["id"], "request_hash": approval["request_hash"], "arguments": safe},
            )
            return ToolOutcome(call, decision, "awaiting_approval", approval=approval, safe_arguments=safe)

        return ToolOutcome(call, decision, "allowed", safe_arguments=safe)

    async def execute(
        self, outcome: ToolOutcome, *, conversation_id: str | None, task_id: str | None = None, approved: bool = False,
    ) -> ToolOutcome:
        """Run a call that gate() (or resolve()) has already cleared."""
        call, decision = outcome.call, outcome.decision
        result = await asyncio.to_thread(self.tools.execute, call.name, call.arguments, approved=approved)
        # A sensitive decision taints the result even when the tool itself
        # did not flag it: the file was sensitive before it was read.
        result.sensitive = result.sensitive or decision.sensitive
        self.database.add_audit(
            "tool", "completed" if result.ok else "failed",
            f"Tool {call.name} {'completed' if result.ok else 'failed'}",
            conversation_id=conversation_id, tool_name=call.name, risk_level=decision.risk_level.value,
            details={
                **({"task_id": task_id} if task_id else {}),
                "arguments": outcome.safe_arguments, "ok": result.ok, "error": result.error,
                "truncated": result.truncated, "sensitive": result.sensitive,
                "result_metadata": self._audit_metadata(result),
            },
        )
        return ToolOutcome(call, decision, "executed", result=result, approval=outcome.approval,
                           safe_arguments=outcome.safe_arguments)

    async def run(self, call: ToolCall, *, conversation_id: str | None, task_id: str | None = None) -> ToolOutcome:
        outcome = await self.gate(call, conversation_id=conversation_id, task_id=task_id)
        if outcome.status != "allowed":
            return outcome
        return await self.execute(outcome, conversation_id=conversation_id, task_id=task_id)

    async def resolve(
        self, approval_id: str, decision: str, note: str = "", *, before_execute: BeforeExecute | None = None,
    ) -> ToolOutcome:
        """Apply the user's decision to a pending approval, then act on it.

        Single use: an approval that is no longer pending is refused, so a
        second click cannot replay the tool. Hash-bound: the stored request
        must still match the tool, arguments and call it was issued for.
        Re-checked: policy runs again at execution time, not approval time.
        """
        record = self.database.authorize_approval(approval_id, decision, note)
        if record is None:
            raise KeyError("Approval not found")
        expected_status = "executing" if decision == "approved" else "denied"
        if record["status"] != expected_status:
            raise ApprovalStateError(f"Approval is already {record['status']}")

        call = ToolCall(record["tool_call_id"], record["tool_name"], record["arguments"])
        conversation_id = record.get("conversation_id")
        task_id = record.get("task_id")
        scope = {"task_id": task_id} if task_id else {}
        fresh = self.policy.evaluate(call.name, call.arguments)
        safe = self.policy.sanitize_arguments(call.arguments)

        if decision == "denied":
            self.database.add_audit(
                "approval", "denied", "User denied tool execution", actor="user", conversation_id=conversation_id,
                tool_name=call.name, risk_level=record["risk_level"],
                details={**scope, "approval_id": approval_id, "note": note[:1000]},
            )
            return ToolOutcome(call, fresh, "denied", approval=record, safe_arguments=safe,
                               result=ToolResult(False, error="The user denied this action."))

        expected_hash = self.database.approval_hash(conversation_id, call.name, call.arguments, call.id)
        if expected_hash != record["request_hash"]:
            self.database.set_approval_result(approval_id, {"ok": False, "error": "Approval binding mismatch"}, "blocked")
            self.database.add_audit(
                "approval", "blocked", "Approval binding mismatch", conversation_id=conversation_id,
                tool_name=call.name, risk_level="critical", details={**scope, "approval_id": approval_id},
            )
            return ToolOutcome(call, fresh, "binding_mismatch", approval=record, safe_arguments=safe,
                               result=ToolResult(False, error="Approval binding mismatch"))

        if not fresh.allowed:
            self.database.set_approval_result(approval_id, {"ok": False, "error": fresh.reason}, "blocked")
            return ToolOutcome(call, fresh, "policy_blocked", approval=record, safe_arguments=safe,
                               result=ToolResult(False, error=fresh.reason, sensitive=fresh.sensitive))

        if before_execute is not None:
            await before_execute(call)
        outcome = await self.execute(
            ToolOutcome(call, fresh, "allowed", approval=record, safe_arguments=safe),
            conversation_id=conversation_id, task_id=task_id, approved=True,
        )
        result = outcome.result
        assert result is not None
        self.database.set_approval_result(approval_id, self.public_result(result), "executed" if result.ok else "failed")
        self.database.add_audit(
            "approval", "executed" if result.ok else "failed", "Approved tool call executed", actor="user",
            conversation_id=conversation_id, tool_name=call.name, risk_level=fresh.risk_level.value,
            details={
                **scope, "approval_id": approval_id, "request_hash": record["request_hash"],
                "ok": result.ok, "error": result.error, "sensitive": result.sensitive,
                "result_metadata": self._audit_metadata(result),
            },
        )
        return outcome
