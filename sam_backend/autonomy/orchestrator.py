"""The autonomous orchestrator: the loop that actually does the work.

    understand -> scan -> plan -> execute -> observe -> validate
                                   ^                       |
                                   |                       v
                               retry <- fix <-------- failure

Every stage is persisted through the TaskStore, so a run can be inspected
live, survive a restart, and resume after an approval pause. The loop is
bounded in three independent ways -- tool steps, self-heal retries and
re-plans -- because an agent that cannot stop is more dangerous than one that
stops early and says why.

Completion is earned, never assumed: a task reaches COMPLETED only when the
verification engine actually ran the project's checks and they passed, or
when there was genuinely nothing to verify and that is stated plainly.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..cancellation import CancellationManager
from ..capabilities import CapabilityRegistry
from ..config import Settings
from ..context_engine import ContextEngine
from ..db import Database
from ..execution import ApprovalStateError, ToolExecutor, ToolOutcome
from ..models import ModelError, ToolCall
from ..planner import Planner
from ..policy import RiskPolicy
from ..project_map import ProjectScanner
from ..provider_health import ProviderHealth
from ..tasks import AgentTask, TaskState, TaskStep, TaskStore
from ..tools import ToolRegistry
from ..ui_review import review_screenshot
from ..verification import CheckOutcome, CheckResult, UiSmokeRunner, VerificationEngine, is_ui_work
from .checkpoints import MODIFYING_TOOLS, WorkspaceCheckpoints

# How much of a file's text an observation may carry back to the model. Large
# enough to act on, small enough that one read cannot swamp the next prompt.
OBSERVATION_CONTENT_LIMIT = 4000

EXECUTOR_SYSTEM_PROMPT = """You are SAM, an autonomous software engineering agent executing one step of an approved plan.

You act by calling exactly one tool per turn. You do not ask the user questions; you inspect the project and act.

Rules:
- Inspect before you change. Read a file before editing it.
- Make the smallest correct change. Preserve the surrounding style.
- Never claim something works because you wrote code: run the project's checks with run_tests.
- If a tool fails, read the error and address its actual cause rather than retrying identically.
- When the current step is genuinely finished and nothing further is needed, reply with a short plain-text summary and no tool call.
- Never print, copy or store credentials."""



class TaskAlreadyRunning(RuntimeError):
    """Raised when a second driver tries to take a task that is already running.

    Two drivers on one task would each hold their own copy loaded from the
    database and overwrite each other's progress, so the second caller is
    refused rather than queued.
    """


class AutonomousOrchestrator:
    """Drives an AgentTask from a goal to a verified outcome."""

    def __init__(
        self,
        settings: Settings,
        database: Database,
        tools: ToolRegistry,
        policy: RiskPolicy,
        router: Any,
        *,
        store: TaskStore | None = None,
        scanner: ProjectScanner | None = None,
        verifier: VerificationEngine | None = None,
        capabilities: CapabilityRegistry | None = None,
        cancellation: CancellationManager | None = None,
        health: ProviderHealth | None = None,
        broadcast: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        self.settings = settings
        self.database = database
        self.tools = tools
        self.policy = policy
        self.router = router
        self.store = store or TaskStore(database)
        self.scanner = scanner or tools.scanner
        self.verifier = verifier or tools.verifier
        self.capabilities = capabilities or tools.capabilities
        self.cancellation = cancellation
        self.executor = ToolExecutor(settings, database, tools, policy)
        self.health = health or ProviderHealth(settings)
        # The model this run resolved to. None until preflight has run; every
        # model call routes through it so a run cannot drift onto a different
        # model than the one the operator was shown.
        self.active_route: tuple[str, str] | None = None
        self.planner = Planner(router)
        self.context = ContextEngine(settings)
        self.broadcast = broadcast
        # Bounds. Independent, because exhausting one should not silently
        # consume the budget of another.
        self.max_steps = max(4, int(getattr(settings, "max_tool_iterations", 8)) * 3)
        self.max_retries = 3
        self.max_replans = 2
        # Exactly one driver per task id. The set is the authority; the lock
        # only keeps check-and-claim atomic.
        self._driving: set[str] = set()
        self._guard = asyncio.Lock()

    # -- helpers -----------------------------------------------------------
    @property
    def workspace(self) -> Path:
        return Path(self.settings.workspace_root)

    async def _publish(self, payload: dict[str, Any]) -> None:
        """Best-effort: a dead socket must never stop a run."""
        if self.broadcast is None:
            return
        try:
            await self.broadcast(payload)
        except Exception:  # noqa: BLE001 - any transport error means "gone"
            pass

    async def _emit(self, task: AgentTask, kind: str, message: str, **detail: Any) -> None:
        self.store.record(task, kind, message, **detail)
        await self._publish({
            "type": "task_event", "task_id": task.id, "state": task.state.value,
            "event": {"kind": kind, "message": message, "detail": detail, "at": time.time()},
        })

    async def _transition(self, task: AgentTask, state: TaskState, message: str = "", **detail: Any) -> None:
        self.store.transition(task, state, message, **detail)
        await self._publish({
            "type": "task_state", "task_id": task.id, "state": state.value,
            "message": message or state.value,
        })

    @asynccontextmanager
    async def _exclusive(self, task_id: str):
        """Claim sole ownership of driving one task, or refuse."""
        async with self._guard:
            if task_id in self._driving:
                raise TaskAlreadyRunning(f"{task_id} is already running")
            self._driving.add(task_id)
        try:
            yield
        finally:
            async with self._guard:
                self._driving.discard(task_id)

    def driving(self, task_id: str) -> bool:
        return task_id in self._driving

    def _register_token(self, task: AgentTask) -> None:
        """Publish the task to the cancellation manager so Stop can reach it.

        Without this the shared /api/tasks/{id}/cancel endpoint has no token
        to cancel and an autonomous run cannot be stopped at all.
        """
        if self.cancellation is not None and self.cancellation.get(task.id) is None:
            self.cancellation.create(task.id)

    def _release_token(self, task: AgentTask) -> None:
        """Drop the token once the run is over.

        A paused run keeps its token: waiting for approval is still stoppable.
        """
        if self.cancellation is not None and task.terminal:
            self.cancellation.complete(task.id)

    def _cancelled(self, task: AgentTask) -> bool:
        if self.cancellation is None:
            return False
        token = self.cancellation.get(task.id)
        return token is not None and token.cancelled

    async def _finish_cancelled(self, task: AgentTask) -> AgentTask:
        await self._transition(task, TaskState.CANCELLED, "Stopped at your request")
        task.completion_status = "cancelled"
        task.summary = (
            f"Stopped by you after {task.tool_calls} tool step(s). "
            f"{sum(1 for item in task.plan if item.status == 'done')} of {len(task.plan)} "
            "planned steps had completed."
        )
        await self._emit(task, "error", "Run stopped at your request")
        self.store.save(task)
        self._release_token(task)
        return task

    # -- public API --------------------------------------------------------
    async def cancel(self, task_id: str) -> dict[str, Any] | None:
        """Stop a run. Owns the whole outcome, whatever the task is doing.

        A running task notices the cancelled token at its next checkpoint
        and finalises itself. A task paused for approval has no driver to
        notice anything, so it is finalised here -- otherwise Stop would
        flip a token nobody reads and leave the task waiting forever.

        Returns None for an id this orchestrator does not own.
        """
        task = self.store.get(task_id)
        if task is None:
            return None
        if task.terminal:
            return {"cancelled": False, "task_id": task_id, "state": task.state.value,
                    "reason": "The task is no longer running."}
        if self.cancellation is not None:
            self._register_token(task)
            self.cancellation.cancel(task_id, "user")
        try:
            async with self._exclusive(task_id):
                await self._finish_cancelled(task)
        except TaskAlreadyRunning:
            # A driver holds the task; it will read the cancelled token at
            # its next checkpoint and finalise the run itself.
            pass
        return {"cancelled": True, "task_id": task_id, "state": task.state.value}

    async def start(
        self, goal: str, *, conversation_id: str | None = None, constraints: list[str] | None = None,
    ) -> AgentTask:
        task = self.store.create(goal, conversation_id=conversation_id, constraints=constraints)
        self.database.add_audit(
            "task", "created", f"Autonomous task created: {goal[:200]}",
            conversation_id=conversation_id, details={"task_id": task.id},
        )
        return await self.run(task)

    async def run(self, task: AgentTask) -> AgentTask:
        """Drive the task to a terminal state, or to an approval pause."""
        async with self._exclusive(task.id):
            self._register_token(task)
            try:
                return await self._run_unguarded(task)
            finally:
                self._release_token(task)

    async def _run_unguarded(self, task: AgentTask) -> AgentTask:
        try:
            if task.state is TaskState.IDLE:
                await self._understand(task)
                if not await self._preflight(task):
                    return task
                await self._scan(task)
                await self._plan(task)
            return await self._drive(task)
        except asyncio.CancelledError:
            await self._transition(task, TaskState.CANCELLED, "Task cancelled")
            task.completion_status = "cancelled"
            self.store.save(task)
            raise
        except Exception as exc:  # noqa: BLE001 - the loop must never crash the server
            task.errors.append(f"{type(exc).__name__}: {exc}")
            await self._transition(task, TaskState.FAILED, f"Run failed: {type(exc).__name__}: {exc}")
            task.completion_status = "failed"
            task.summary = f"The run stopped on an unexpected error: {exc}"
            self.store.save(task)
            return task

    async def resume(self, task_id: str) -> AgentTask | None:
        """Continue a task that was interrupted (a restart, say)."""
        task = self.store.get(task_id)
        if task is None or task.terminal:
            return task
        async with self._exclusive(task_id):
            self._register_token(task)
            try:
                if task.state is TaskState.WAITING_FOR_APPROVAL:
                    await self._transition(task, TaskState.EXECUTING, "Resuming")
                return await self._drive(task)
            finally:
                self._release_token(task)

    async def resolve_approval(self, task_id: str, approval_id: str, decision: str, note: str = "") -> AgentTask | None:
        """Apply the user's decision, then carry on.

        The approved call is re-checked before it runs: the request hash must
        still bind to this exact tool and arguments, and policy is evaluated a
        second time. An approval is permission for one specific action, not a
        standing licence.
        """
        task = self.store.get(task_id)
        if task is None:
            return None
        if task.terminal:
            return task

        async with self._exclusive(task_id):
            self._register_token(task)
            try:
                return await self._apply_approval(task, approval_id, decision, note)
            finally:
                self._release_token(task)

    async def _apply_approval(
        self, task: AgentTask, approval_id: str, decision: str, note: str,
    ) -> AgentTask | None:
        step = task.current_step
        try:
            outcome = await self.executor.resolve(
                approval_id, decision, note, before_execute=partial(self._checkpoint, task),
            )
        except ApprovalStateError as exc:
            # A second click on an already-decided approval must not replay
            # the tool; the run simply carries on from where it is.
            await self._emit(task, "error", str(exc))
            await self._transition(task, TaskState.EXECUTING, "Continuing")
            return await self._drive(task)
        name = outcome.call.name

        if outcome.status == "denied":
            await self._emit(task, "approval", f"You denied {name}", tool=name, approved=False)
            task.observations.append(f"The user denied {name}; find another route or stop.")
            if step is not None:
                step.status = "failed"
                step.detail = "Denied by the user."
            self.store.save(task)
            await self._transition(task, TaskState.EXECUTING, "Continuing after a denial")
            return await self._drive(task)

        if outcome.status == "binding_mismatch":
            await self._emit(task, "error", "Approval binding mismatch; the action was not executed")
            await self._transition(task, TaskState.FAILED, "Approval binding mismatch")
            task.completion_status = "failed"
            task.summary = "An approval no longer matched the action it was issued for, so nothing was executed."
            return self.store.save(task)

        if outcome.status == "policy_blocked":
            await self._emit(task, "error", f"Policy now blocks {name}: {outcome.error}")
            await self._transition(task, TaskState.EXECUTING, "Continuing after a policy block")
            return await self._drive(task)

        await self._transition(task, TaskState.EXECUTING, f"Approved: ran {name}")
        if step is not None:
            await self._observe(task, step, outcome)
        return await self._drive(task)

    # -- stages ------------------------------------------------------------
    async def _preflight(self, task: AgentTask) -> bool:
        """Settle which real model will drive this run, before spending one.

        Discovering mid-run that the configured model cannot be paid for wastes
        the work already done and leaves a half-finished task. The check costs
        no tokens, and its outcome is recorded so the operator can see which
        model actually ran and why.
        """
        adapters = getattr(self.router, "adapters", None)
        if adapters is None:
            # Nothing to interrogate. Preflight exists to avoid wasted work,
            # never to invent a blocker, so an unanswerable check proceeds and
            # lets the first real model call report the truth.
            return True
        resolution = await self.health.resolve(adapters)
        if resolution.blocked:
            self.active_route = None
            await self._emit(
                task, "error",
                f"Provider unavailable: {resolution.fallback_reason}",
                primary=resolution.primary.as_dict(),
                fallback=resolution.fallback.as_dict() if resolution.fallback else None,
            )
            await self._transition(task, TaskState.FAILED, "No usable model provider")
            task.completion_status = "provider_unavailable"
            task.summary = (
                f"The run did not start: {resolution.fallback_reason} "
                "No work was attempted and nothing was changed."
            )
            self.store.save(task)
            return False

        active = resolution.active
        assert active is not None
        self.active_route = (active.provider, active.model)
        if resolution.as_dict()["fallback_engaged"]:
            # Naming both models matters: an agent quietly running on a
            # different brain than the operator configured is worse than one
            # that stops and says so.
            await self._emit(
                task, "fallback",
                f"Primary model unavailable: {resolution.primary.model} "
                f"({resolution.primary.reason or resolution.primary.availability.value}). "
                f"Using configured fallback: {active.model}",
                primary=resolution.primary.as_dict(), active=active.as_dict(),
            )
        else:
            await self._emit(task, "thought", f"Model: {active.model} ({active.provider})",
                             active=active.as_dict())
        return True

    async def _understand(self, task: AgentTask) -> None:
        await self._transition(task, TaskState.UNDERSTANDING, "Understanding the request")
        await self._emit(task, "thought", f"Goal: {task.goal[:300]}")

    async def _scan(self, task: AgentTask) -> None:
        await self._transition(task, TaskState.SCANNING, "Scanning the project")
        project_map = await asyncio.to_thread(self.scanner.scan, self.workspace)
        if project_map.git.get("repository"):
            # What the user already had in flight. Anything the run touches
            # in this set is reported as "also had your changes" so the
            # agent's work is never confused with the user's.
            task.preexisting_changes = list(project_map.git.get("changed_paths") or [])
            self.store.save(task)
        await self._emit(
            task, "result",
            f"Scanned {project_map.file_count} files"
            + (f"; {len(project_map.api_routes)} API route(s)" if project_map.api_routes else ""),
            file_count=project_map.file_count,
            languages=project_map.languages,
            commands=project_map.commands,
        )

    async def _plan(self, task: AgentTask) -> None:
        await self._transition(task, TaskState.PLANNING, "Building a plan")
        project_map = self.scanner.cached(self.workspace)
        provider, model = self.active_route or (None, None)
        result = await self.planner.plan(
            task.goal,
            provider=provider, model=model,
            project_map=project_map,
            capabilities_summary=self.capabilities.summary_text(),
            constraints=task.constraints,
            task_id=task.id,
        )
        self.store.set_plan(task, result.steps)
        if result.understanding:
            task.observations.append(result.understanding)
        if result.note:
            await self._emit(task, "thought", result.note)
        await self._emit(
            task, "plan", f"Planned {len(result.steps)} step(s) [{result.source}]",
            steps=[step.as_dict() for step in result.steps],
        )

    async def _drive(self, task: AgentTask) -> AgentTask:
        """Execute steps until the plan is done, then validate and heal."""
        while True:
            # A stage may have ended the run (model unreachable, retry budget
            # spent). Continuing to drive it would attempt an illegal
            # transition out of a terminal state.
            if task.terminal:
                return self.store.save(task)
            if self._cancelled(task):
                return await self._finish_cancelled(task)

            step = task.current_step
            if step is not None:
                if task.tool_calls >= self.max_steps:
                    return await self._stop_bounded(task)
                if task.state is not TaskState.EXECUTING:
                    await self._transition(task, TaskState.EXECUTING, f"Step {step.index}: {step.text[:160]}")
                approval_id = await self._execute_step(task, step)
                if approval_id:
                    await self._transition(
                        task, TaskState.WAITING_FOR_APPROVAL,
                        "Waiting for your approval to continue",
                        approval_id=approval_id,
                    )
                    task.completion_status = "awaiting_approval"
                    return self.store.save(task)
                continue

            # Plan exhausted: prove it worked.
            report = await self._validate(task)
            if report is None or report.ok:
                return await self._complete(task, report)
            healed = await self._heal(task, report)
            if not healed:
                return task
        # unreachable

    async def _execute_step(self, task: AgentTask, step: TaskStep) -> str | None:
        """Advance one step. Returns the approval id if it paused, else None."""
        step.status = "active"
        step.started_at = step.started_at or time.time()
        self.store.save(task)

        messages = self.context.build(
            task=task,
            step=step,
            system_prompt=EXECUTOR_SYSTEM_PROMPT,
            project_map=self.scanner.cached(self.workspace),
            capabilities_summary=self.capabilities.summary_text(),
        )
        try:
            provider, model = self.active_route or (None, None)
            turn, choice, _fallbacks = await self.router.complete(
                message=step.text,
                messages=messages,
                tools=self.tools.specs,
                provider=provider,
                model=model,
                conversation_id=task.conversation_id,
                task_id=task.id,
            )
        except ModelError as exc:
            if self.active_route:
                # A real failure is better evidence than any preflight; record
                # it so the next run does not probe a known-dead model again.
                self.health.record_failure(*self.active_route, exc)
            step.status = "failed"
            step.detail = str(exc)[:500]
            task.errors.append(f"Model unavailable: {exc}")
            self.store.save(task)
            await self._emit(task, "error", f"No model could handle this step: {exc}")
            await self._transition(task, TaskState.FAILED, "No model provider was reachable")
            task.completion_status = "failed"
            task.summary = "The run stopped because no configured model provider was reachable."
            self.store.save(task)
            return None

        # A model call can take a while; a Stop pressed during it should not
        # be spent on another tool. The drive loop finalises the cancellation.
        if self._cancelled(task):
            return None

        calls = turn.tool_calls[:1]
        if not calls:
            step.status = "done"
            step.finished_at = time.time()
            step.detail = (turn.content or "").strip()[:500]
            if step.detail:
                task.observations.append(step.detail)
            self.store.save(task)
            await self._emit(task, "success", f"Step {step.index} complete: {step.detail[:200] or 'no action needed'}")
            return None

        call = calls[0]
        return await self._run_tool(task, step, call, provider=getattr(choice, "provider", ""))

    async def _run_tool(self, task: AgentTask, step: TaskStep, call: ToolCall, *, provider: str = "") -> str | None:
        gated = await self.executor.gate(call, conversation_id=task.conversation_id, task_id=task.id)
        decision = gated.decision
        step.tool = call.name
        task.tool_calls += 1
        await self._emit(
            task, "tool", f"{call.name}", tool=call.name, arguments=gated.safe_arguments,
            risk=decision.risk_level.value, provider=provider,
        )

        if gated.status == "blocked":
            step.status = "failed"
            step.detail = decision.reason[:500]
            task.errors.append(f"{call.name} blocked: {decision.reason}")
            self.store.save(task)
            await self._emit(task, "error", f"{call.name} was blocked: {decision.reason}", tool=call.name)
            # A blocked tool is information, not a dead end: the next turn
            # sees the refusal and can choose a permitted route.
            return None

        if gated.status == "awaiting_approval":
            await self._emit(
                task, "approval", f"{call.name} needs your approval: {decision.reason}",
                tool=call.name, approval_id=gated.approval_id, risk=decision.risk_level.value,
                arguments=gated.safe_arguments,
            )
            return gated.approval_id

        if self._cancelled(task):
            return None
        await self._checkpoint(task, call)
        outcome = await self.executor.execute(gated, conversation_id=task.conversation_id, task_id=task.id)
        return await self._observe(task, step, outcome)

    async def _observe(self, task: AgentTask, step: TaskStep, outcome: ToolOutcome) -> str | None:
        call, result = outcome.call, outcome.result
        assert result is not None
        await self._transition(task, TaskState.OBSERVING, f"Observing {call.name}")
        self._record_files(task, call, result)

        if result.ok:
            # A sensitive result (an approved credential-file read, a
            # screenshot, the clipboard) must never enter the timeline, the
            # persisted observations, or the next prompt. The agent is told
            # the call succeeded and nothing more.
            sensitive = outcome.sensitive
            summary = (
                f"succeeded; its output is sensitive and was withheld from context"
                if sensitive else self._summarise_result(call.name, result)
            )
            task.observations.append(f"{call.name}: {summary}")
            await self._emit(
                task, "result", f"{call.name}: {summary}", tool=call.name, ok=True, sensitive=sensitive,
            )
            if call.name == "run_tests" and isinstance(result.output, dict):
                task.test_results.append(result.output)
            step.status = "done"
            step.finished_at = time.time()
            step.detail = summary[:500]
            self.store.save(task)
            await self._transition(task, TaskState.EXECUTING, "Continuing")
            return None

        error = result.error or "The tool reported a failure without a message."
        if outcome.sensitive:
            error = "the call failed; its details are sensitive and were withheld"
        task.errors.append(f"{call.name}: {error}")
        await self._emit(task, "error", f"{call.name} failed: {error[:300]}", tool=call.name, ok=False)
        if call.name == "run_tests" and isinstance(result.output, dict):
            task.test_results.append(result.output)
        # The step stays open so the next turn can react to the error. Its
        # own retry budget is what stops this from looping forever.
        step.detail = error[:500]
        # A check that ran and reported failing tests is information, not a
        # broken tool. Counting it against the tool-failure budget would kill
        # the ordinary debugging loop -- run, see the failure, fix, run again
        # -- which is the whole point of the self-healing stage. Verification
        # still has to pass before a run may call itself verified.
        if call.name != "run_tests":
            task.retries += 1
        self.store.save(task)
        if task.retries > self.max_retries:
            step.status = "failed"
            self.store.save(task)
            await self._transition(task, TaskState.FAILED, "Too many failed tool attempts")
            task.completion_status = "failed"
            task.summary = f"Stopped after {task.retries} failed tool attempts. Last error: {error[:300]}"
            self.store.save(task)
            return None
        await self._transition(task, TaskState.EXECUTING, "Retrying after a tool failure")
        return None

    async def _validate(self, task: AgentTask) -> Any:
        await self._transition(task, TaskState.VALIDATING, "Verifying the work")
        project_map = await asyncio.to_thread(partial(self.scanner.scan, self.workspace, force=True))
        report = await asyncio.to_thread(self.verifier.verify, project_map, self.workspace)
        if is_ui_work(task.modified_files):
            await self._validate_ui(task, project_map, report)
        task.test_results.append(report.as_dict())
        await self._emit(
            task, "result" if report.ok else "error",
            report.summary_text()[:800], verified=report.verified, ok=report.ok,
        )
        return report

    async def _validate_ui(self, task: AgentTask, project_map: Any, report: Any) -> None:
        """UI work is only done when the page opens cleanly and looks right.

        Two checks join the report. The browser smoke is deterministic: a
        console error or failed request fails it outright. The visual review
        is a model's judgement of the screenshot; when no model can give one
        it is recorded as skipped, so the run says "not visually reviewed"
        rather than pretending.
        """
        await self._emit(task, "action", "Opening the page in an isolated browser")
        screenshot_dir = Path(self.settings.data_dir) / "ui-review"
        smoke = await asyncio.to_thread(
            UiSmokeRunner(self.verifier).run, project_map, self.workspace,
            screenshot_dir=screenshot_dir, name=f"{task.id}-{len(task.test_results) + 1}",
        )
        report.checks.append(smoke.check)
        if smoke.check.outcome is CheckOutcome.SKIPPED:
            await self._emit(task, "error", f"Browser check skipped: {smoke.check.reason}")
            return
        await self._emit(
            task, "result" if smoke.check.ok else "error", smoke.check.summary,
            url=smoke.url, screenshot=str(smoke.screenshot) if smoke.screenshot else None,
            problems=smoke.check.failures,
        )
        if smoke.screenshot is None:
            report.checks.append(CheckResult(kind="ui_review", command="", outcome=CheckOutcome.SKIPPED,
                                             reason="no screenshot was captured"))
            await self._emit(task, "error", "Visual review skipped: no screenshot was captured")
            return
        review = await review_screenshot(
            self.router, screenshot=smoke.screenshot, goal=task.goal,
            console_errors=smoke.check.failures, task_id=task.id,
        )
        report.checks.append(review.check)
        if review.outcome is CheckOutcome.SKIPPED:
            await self._emit(task, "error", f"Visual review skipped: {review.reason}")
        elif review.outcome is CheckOutcome.PASSED:
            await self._emit(task, "result", f"Visual review passed ({review.provider or 'model'})")
        else:
            await self._emit(task, "error", f"Visual review found {len(review.findings)} problem(s)", findings=review.findings)


    async def _heal(self, task: AgentTask, report: Any) -> bool:
        """Diagnose, re-plan and retry. False means the run is over."""
        if task.replans >= self.max_replans:
            await self._transition(task, TaskState.FAILED, "Verification still failing after re-planning")
            task.completion_status = "failed"
            task.summary = (
                f"The work did not pass verification after {task.replans} re-plan(s).\n{report.summary_text()[:800]}"
            )
            self.store.save(task)
            return False

        await self._transition(task, TaskState.FIXING, "Diagnosing the failure")
        failure_text = report.summary_text()
        for check in report.failures:
            if check.stderr_tail or check.stdout_tail:
                failure_text += "\n" + (check.stderr_tail or check.stdout_tail)[-2000:]
        provider, model = self.active_route or (None, None)
        revised = await self.planner.replan(
            task.goal, failure=failure_text, attempted=task.plan,
            provider=provider, model=model,
            project_map=self.scanner.cached(self.workspace), task_id=task.id,
        )
        task.replans += 1
        # Completed work stays in the record; the revised steps are appended
        # so the timeline shows the whole history rather than a rewrite.
        offset = len(task.plan)
        for position, step in enumerate(revised.steps, start=1):
            step.index = offset + position
        task.plan.extend(revised.steps)
        self.store.save(task)
        await self._emit(
            task, "fix", f"Re-planned {len(revised.steps)} step(s) after a failed verification",
            steps=[step.as_dict() for step in revised.steps], diagnosis=revised.understanding[:500],
        )
        await self._transition(task, TaskState.RETRYING, "Retrying with a revised plan")
        await self._transition(task, TaskState.EXECUTING, "Executing the revised plan")
        return True

    async def _complete(self, task: AgentTask, report: Any) -> AgentTask:
        verified = bool(report is not None and report.verified)
        await self._transition(task, TaskState.COMPLETED, "Task complete")
        task.completion_status = "completed_verified" if verified else "completed_unverified"
        lines = [f"Goal: {task.goal}"]
        if task.modified_files:
            lines.append("Files changed: " + ", ".join(task.modified_files[:20]))
        else:
            lines.append("Files changed: none")
        if report is not None:
            lines.append("Verification:\n" + report.summary_text())
            if not verified:
                lines.append(
                    "Note: nothing could be independently verified, so this is reported as complete-but-unverified."
                )
        task.summary = "\n".join(lines)
        self.store.save(task)
        self.database.add_audit(
            "task", "completed", f"Autonomous task finished: {task.completion_status}",
            conversation_id=task.conversation_id,
            details={"task_id": task.id, "verified": verified, "files": task.modified_files[:20]},
        )
        await self._emit(task, "success", "Task complete", verified=verified)
        return task

    async def _stop_bounded(self, task: AgentTask) -> AgentTask:
        await self._transition(task, TaskState.FAILED, "Step budget exhausted")
        task.completion_status = "bounded"
        task.summary = (
            f"Stopped after {task.tool_calls} tool steps to keep the run bounded. "
            f"{sum(1 for step in task.plan if step.status == 'done')} of {len(task.plan)} planned steps completed."
        )
        self.store.save(task)
        await self._emit(task, "error", task.summary)
        return task

    # -- checkpoints, diff, rollback --------------------------------------
    # The mechanics live in WorkspaceCheckpoints; what stays here is the
    # orchestration around them: when to snapshot, what to tell the user, and
    # what must not happen while a run is still driving.

    def _checkpoints_for(self, task: AgentTask) -> WorkspaceCheckpoints:
        return WorkspaceCheckpoints(self.workspace, Path(self.settings.data_dir) / "checkpoints" / task.id)

    async def _checkpoint(self, task: AgentTask, call: ToolCall) -> None:
        """Snapshot a file the first time this run is about to change it."""
        entry = await asyncio.to_thread(
            self._checkpoints_for(task).snapshot, task.checkpoints, call.name, call.arguments,
        )
        if entry is None:
            return
        first = not task.checkpoints
        task.checkpoints.append(entry)
        self.store.save(task)
        if first:
            await self._emit(task, "result", "Checkpoint recorded before the first file change", path=entry["path"])

    def diff(self, task: AgentTask) -> dict[str, Any]:
        """The run's own changes, file by file, against its checkpoints."""
        files = self._checkpoints_for(task).diff(task.checkpoints, task.preexisting_changes)
        return {"task_id": task.id, "files": files, "rolled_back": task.rolled_back}

    async def rollback(self, task_id: str) -> dict[str, Any] | None:
        """Undo every file change this run made, restoring the checkpoints."""
        task = self.store.get(task_id)
        if task is None:
            return None
        if self.driving(task_id):
            return {"rolled_back": False, "task_id": task_id, "reason": "Stop the task before rolling it back."}
        if task.rolled_back:
            return {"rolled_back": False, "task_id": task_id, "reason": "This run was already rolled back."}
        restored = await asyncio.to_thread(self._checkpoints_for(task).restore, task.checkpoints)
        task.rolled_back = True
        await self._emit(task, "fix", f"Rolled back {len(restored)} file(s) to their checkpoints", files=restored)
        self.store.save(task)
        self.database.add_audit(
            "task", "rolled_back", "User rolled back an autonomous run", actor="user",
            details={"task_id": task_id, "files": restored},
        )
        return {"rolled_back": True, "task_id": task_id, "files": restored}

    async def retry(self, task_id: str) -> AgentTask | None:
        """A fresh run of the same goal. The old record is left as history."""
        previous = self.store.get(task_id)
        if previous is None:
            return None
        task = self.store.create(previous.goal, conversation_id=previous.conversation_id, constraints=previous.constraints)
        self.store.record(task, "thought", f"Retry of {previous.id}", previous_task=previous.id)
        return task

    async def mark_interrupted(self) -> list[str]:
        """Called once at startup for runs the previous process left mid-flight.

        A task that was executing is not resumed automatically: the tool it
        was in the middle of may already have had its effect, and repeating
        it blind is exactly the kind of silent double-write an agent must not
        do. It is failed honestly with Retry available. A task waiting for
        approval is left waiting -- nothing was half-done.
        """
        interrupted: list[str] = []
        for task in self.store.resumable():
            if task.state is TaskState.WAITING_FOR_APPROVAL:
                continue
            was = task.state.value.lower()
            await self._transition(task, TaskState.FAILED, "Interrupted by a restart")
            task.completion_status = "interrupted"
            task.summary = (
                f"SAM restarted while this run was {was}. Nothing after that point was "
                "executed. Review the diff, roll back if needed, or retry."
            )
            self.store.save(task)
            interrupted.append(task.id)
        return interrupted

    # -- small helpers -----------------------------------------------------
    def _record_files(self, task: AgentTask, call: ToolCall, result: Any) -> None:
        if call.name not in {"write_file", "replace_text", "delete_path"}:
            return
        output = result.output if isinstance(result.output, dict) else {}
        path = str(output.get("path") or call.arguments.get("path") or "").strip()
        if path:
            self.store.note_files(task, [path])

    @staticmethod
    def _summarise_result(name: str, result: Any) -> str:
        output = result.output
        if isinstance(output, dict):
            if name == "run_tests":
                return str(output.get("summary") or "checks finished")[:400]
            if name == "project_map":
                return f"{output.get('file_count', '?')} files, commands {output.get('commands', {})}"
            if "content" in output:
                # The point of reading a file is the text inside it. Summarising
                # to a path told the model nothing and left it re-reading the
                # same file instead of acting on what it had asked for.
                content = str(output.get("content") or "")
                return f"{output.get('path')}:\n{content[:OBSERVATION_CONTENT_LIMIT]}"
            if "path" in output:
                return f"{output.get('path')} ({output.get('bytes', '?')} bytes)"
            if "exit_code" in output:
                return f"exit {output.get('exit_code')}"
            if "commits" in output:
                return f"{len(output['commits'])} commit(s)"
            if "changed" in output:
                return f"{len(output['changed'])} changed path(s) on {output.get('branch', '?')}"
        text = result.model_text() if hasattr(result, "model_text") else str(output)
        return text.strip()[:400] or "done"
