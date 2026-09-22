"""The autonomous loop, end to end, against a scripted model.

These are the tests that decide whether SAM is an agent or a chatbot: a goal
must travel through planning, real tool execution, real verification and
self-correction, and must refuse to call itself done when it is not.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from sam_backend.autonomy import AutonomousOrchestrator
from sam_backend.capabilities import CapabilityRegistry
from sam_backend.config import Settings
from sam_backend.db import Database
from sam_backend.models import AssistantTurn, ModelError, ToolCall
from sam_backend.policy import RiskPolicy
from sam_backend.project_map import ProjectScanner
from sam_backend.tasks import TaskState, TaskStore
from sam_backend.tools import ToolRegistry
from sam_backend.verification import VerificationEngine


class RouteChoice:
    provider = "fake"
    model = "fake-model"


class ScriptedRouter:
    """Returns pre-scripted turns; records what it was asked."""

    def __init__(self, turns: list[Any]) -> None:
        self.turns = list(turns)
        self.prompts: list[str] = []
        self.tool_specs_seen: list[int] = []

    async def complete(self, *, message, messages, tools, provider, model, conversation_id, task_id=None):
        self.prompts.append("\n".join(str(item.get("content", "")) for item in messages))
        self.tool_specs_seen.append(len(tools))
        if not self.turns:
            return AssistantTurn("Nothing further to do."), RouteChoice(), []
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn, RouteChoice(), []


def plan_turn(*steps: tuple[str, str]) -> AssistantTurn:
    payload = {
        "understanding": "Understood the goal.",
        "steps": [{"text": text, "kind": kind} for text, kind in steps],
    }
    return AssistantTurn(f"Here is the plan:\n```json\n{json.dumps(payload)}\n```")


def tool_turn(name: str, arguments: dict[str, Any], call_id: str = "call_1") -> AssistantTurn:
    return AssistantTurn("Working on it.", [ToolCall(call_id, name, arguments)])


def done_turn(text: str = "Step finished.") -> AssistantTurn:
    return AssistantTurn(text)


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    space = tmp_path / "workspace"
    space.mkdir()
    return space


@pytest.fixture()
def settings(tmp_path: Path, workspace: Path) -> Settings:
    return Settings(
        project_root=tmp_path,
        workspace_root=workspace,
        data_dir=tmp_path / "data",
        default_provider="ollama",
        default_model="fake",
        permission_mode="trusted",
    )


def build_orchestrator(settings: Settings, router: ScriptedRouter) -> AutonomousOrchestrator:
    database = Database(Path(settings.data_dir) / "sam.sqlite3")
    scanner = ProjectScanner(fresh_seconds=0.0)
    verifier = VerificationEngine(timeout=120)
    capabilities = CapabilityRegistry()
    tools = ToolRegistry(settings, database, scanner=scanner, verifier=verifier, capabilities=capabilities)
    return AutonomousOrchestrator(
        settings, database, tools, RiskPolicy(settings), router,
        store=TaskStore(database), scanner=scanner, verifier=verifier, capabilities=capabilities,
    )


def add_passing_suite(workspace: Path) -> None:
    (workspace / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (workspace / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")


# -- the happy path --------------------------------------------------------

def test_a_goal_is_planned_executed_and_verified(settings: Settings, workspace: Path):
    add_passing_suite(workspace)
    router = ScriptedRouter([
        plan_turn(("Create greeting.txt", "edit"), ("Confirm the file exists", "verify")),
        tool_turn("write_file", {"path": "greeting.txt", "content": "hello"}),
        tool_turn("read_file", {"path": "greeting.txt"}, "call_2"),
    ])
    orchestrator = build_orchestrator(settings, router)

    task = asyncio.run(orchestrator.start("Create a greeting file"))

    assert task.state is TaskState.COMPLETED
    assert task.completion_status == "completed_verified"
    assert (workspace / "greeting.txt").read_text(encoding="utf-8") == "hello"
    assert "greeting.txt" in " ".join(task.modified_files)
    # completed_verified must carry the evidence that earned it: the verdict
    # is the same report the completion path read, not a second opinion.
    assert task.verification is not None and task.verification["verified"] is True
    assert any(check["outcome"] == "PASSED" for check in task.verification["checks"])


def test_the_verdict_survives_a_reload_and_is_absent_when_nothing_ran(settings: Settings, workspace: Path):
    """The evidence lives in the task record, not only in memory."""
    add_passing_suite(workspace)
    router = ScriptedRouter([
        plan_turn(("Write a file", "edit")),
        tool_turn("write_file", {"path": "a.txt", "content": "x"}),
    ])
    orchestrator = build_orchestrator(settings, router)
    task = asyncio.run(orchestrator.start("Write a file"))

    reloaded = orchestrator.store.get(task.id)
    assert reloaded.completion_status == "completed_verified"
    assert reloaded.verification == task.verification
    assert reloaded.verification["verified"] is True

    # A run that never reached validation invents nothing.
    never_validated = orchestrator.store.create("Not started")
    assert never_validated.verification is None
    assert orchestrator.store.get(never_validated.id).verification is None


def test_a_run_that_fails_verification_keeps_the_failing_evidence(settings: Settings, workspace: Path):
    """Honest either way: the verdict is recorded when it refuses the work too."""
    (workspace / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (workspace / "test_bad.py").write_text("def test_bad():\n    assert False\n", encoding="utf-8")
    router = ScriptedRouter([
        plan_turn(("Write a file", "edit")),
        tool_turn("write_file", {"path": "a.txt", "content": "x"}),
        plan_turn(("Try again", "edit")),
        tool_turn("write_file", {"path": "a.txt", "content": "y"}),
    ])
    orchestrator = build_orchestrator(settings, router)
    orchestrator.max_replans = 0

    task = asyncio.run(orchestrator.start("Write a file"))

    assert task.completion_status != "completed_verified"
    assert task.verification is not None and task.verification["verified"] is False
    assert any(check["outcome"] == "FAILED" for check in task.verification["checks"])


def test_the_persisted_verdict_masks_credential_shaped_output(settings: Settings, workspace: Path):
    """Command output is persisted and rendered, so it is redacted first."""
    from sam_backend.verification import CheckOutcome, CheckResult, VerificationReport

    leaked = "OPENROUTER_API_KEY=sk-or-v1-VERIFICATION-SENTINEL-0123456789"
    report = VerificationReport(checks=[CheckResult(
        kind="test", command="pytest", outcome=CheckOutcome.FAILED,
        stdout_tail=leaked, stderr_tail=leaked, failures=[leaked],
    )])

    payload = json.dumps(report.as_dict())

    assert "sk-or-v1-VERIFICATION-SENTINEL-0123456789" not in payload
    assert "[REDACTED]" in payload
    # The re-planner still needs the real text, so the object keeps it.
    assert report.checks[0].stdout_tail == leaked


@pytest.mark.parametrize("status, verification, accepted", [
    ("completed_verified", {"verified": True, "checks": []}, True),
    ("completed_verified", None, False),
    ("completed_unverified", None, True),
    ("provider_unavailable", None, True),
    ("failed", None, True),
])
def test_the_store_refuses_a_verified_claim_without_its_evidence(settings: Settings, status, verification, accepted):
    """The rule lives at the write boundary, so no caller can route around it."""
    store = TaskStore(Database(settings.database_path))
    task = store.create("Some goal")
    task.completion_status = status
    task.verification = verification

    if accepted:
        assert store.save(task).completion_status == status
    else:
        with pytest.raises(ValueError, match="without the verification report"):
            store.save(task)


def seed_legacy_row(store: "TaskStore", goal: str = "Old run") -> str:
    """A row of the shape written before the rule existed."""
    task = store.create(goal)
    stale = task.as_dict() | {"completion_status": "completed_verified", "verification": None}
    with store.database.write() as connection:
        connection.execute("UPDATE agent_tasks SET payload_json=? WHERE id=?", (json.dumps(stale), task.id))
    return task.id


def test_a_legacy_verified_row_keeps_working_without_being_rewritten(settings: Settings):
    """Records written before the rule stay readable, and still writable:
    rolling one back must not fail because of history."""
    store = TaskStore(Database(settings.database_path))
    task_id = seed_legacy_row(store)

    loaded = store.get(task_id)
    assert loaded.completion_status == "completed_verified"
    assert loaded.verification is None, "nothing is invented on read"
    # Reading must not smuggle exemption state onto the object or the record.
    assert "legacy_unverified_claim" not in loaded.as_dict()
    assert not hasattr(loaded, "legacy_unverified_claim"), "no caller-settable bypass exists"

    # The resave a rollback performs: unrelated metadata, same inconsistency.
    loaded.rolled_back = True
    store.save(loaded)
    assert store.get(task_id).rolled_back is True

    # And it does not travel: a new task cannot make the same claim, even
    # with every field a caller can reach set to imitate the legacy row.
    fresh = store.create("New run")
    fresh.completion_status = "completed_verified"
    for spec in dataclasses.fields(loaded):
        if spec.name not in {"id", "goal", "created_at"}:
            setattr(fresh, spec.name, getattr(loaded, spec.name))
    with pytest.raises(ValueError):
        store.save(fresh)


def test_repairing_a_legacy_row_ends_its_exemption(settings: Settings):
    """The pass is granted by the stored row, so fixing it withdraws the pass."""
    store = TaskStore(Database(settings.database_path))
    task_id = seed_legacy_row(store, "Row to repair")

    repaired = store.get(task_id)
    repaired.verification = {"verified": True, "checks": []}
    store.save(repaired)
    assert store.get(task_id).verification == {"verified": True, "checks": []}

    # Now that the stored row is sound, it cannot regress to the old shape.
    regressed = store.get(task_id)
    regressed.verification = None
    with pytest.raises(ValueError, match="without the verification report"):
        store.save(regressed)
    assert store.get(task_id).verification is not None, "the sound row is untouched"


def test_rollback_resaves_a_legacy_row_without_tripping_the_rule(settings: Settings, workspace: Path):
    """The real hazard: rollback writes back whatever task it restored."""
    store = TaskStore(Database(settings.database_path))
    orchestrator = build_orchestrator(settings, ScriptedRouter([]))
    orchestrator.store = store
    task_id = seed_legacy_row(store, "Legacy run to roll back")

    outcome = asyncio.run(orchestrator.rollback(task_id))

    assert outcome["rolled_back"] is True
    assert store.get(task_id).rolled_back is True


def test_the_whole_run_is_persisted_and_replayable(settings: Settings, workspace: Path):
    add_passing_suite(workspace)
    router = ScriptedRouter([
        plan_turn(("Write a file", "edit")),
        tool_turn("write_file", {"path": "a.txt", "content": "x"}),
    ])
    orchestrator = build_orchestrator(settings, router)
    task = asyncio.run(orchestrator.start("Write a file"))

    reloaded = orchestrator.store.get(task.id)
    assert reloaded is not None
    kinds = [event.kind for event in reloaded.events]
    # A user must be able to see the shape of what happened afterwards.
    for expected in ("state", "plan", "tool", "result", "success"):
        assert expected in kinds, f"missing {expected} in {kinds}"
    assert reloaded.state is TaskState.COMPLETED


def test_planning_asks_without_tools_but_execution_offers_them(settings: Settings, workspace: Path):
    """Offering tools during planning invites acting before thinking."""
    add_passing_suite(workspace)
    router = ScriptedRouter([
        plan_turn(("Inspect the project", "inspect")),
        tool_turn("project_map", {"path": "."}),
    ])
    orchestrator = build_orchestrator(settings, router)
    asyncio.run(orchestrator.start("Look around"))

    assert router.tool_specs_seen[0] == 0, "the planner must not be given tools"
    assert router.tool_specs_seen[1] > 0, "the executor must be given tools"


# -- self-correction -------------------------------------------------------

def test_a_failing_suite_triggers_replanning_and_a_real_fix(settings: Settings, workspace: Path):
    """The loop must detect a genuine test failure, re-plan, fix it, and retest."""
    (workspace / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (workspace / "test_feature.py").write_text(
        "from feature import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n", encoding="utf-8",
    )
    # The first implementation is wrong, so the suite fails for real.
    (workspace / "feature.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")

    router = ScriptedRouter([
        plan_turn(("Implement add()", "edit")),
        done_turn("add() already exists."),
        # After validation fails, the planner is asked for a corrected plan.
        plan_turn(("Fix the arithmetic in feature.py", "edit")),
        tool_turn("write_file", {"path": "feature.py", "content": "def add(a, b):\n    return a + b\n"}, "call_fix"),
    ])
    orchestrator = build_orchestrator(settings, router)

    async def drive() -> Any:
        task = await orchestrator.start("Make the feature tests pass")
        # Overwriting an existing file is gated, so the fix pauses for
        # approval. Approving it must then run that exact call and continue.
        assert task.state is TaskState.WAITING_FOR_APPROVAL
        approval_id = next(
            event.detail["approval_id"] for event in reversed(task.events) if event.kind == "approval"
        )
        return await orchestrator.resolve_approval(task.id, approval_id, "approved")

    task = asyncio.run(drive())

    assert task.state is TaskState.COMPLETED
    assert task.completion_status == "completed_verified"
    assert task.replans == 1
    assert (workspace / "feature.py").read_text(encoding="utf-8").endswith("a + b\n")
    assert any(event.kind == "fix" for event in task.events)
    # The re-planner must have been told what actually broke.
    assert any("test_add" in prompt or "failed" in prompt.lower() for prompt in router.prompts)


def test_denying_an_approval_does_not_execute_the_action(settings: Settings, workspace: Path):
    add_passing_suite(workspace)
    (workspace / "precious.txt").write_text("original", encoding="utf-8")
    router = ScriptedRouter([
        plan_turn(("Overwrite precious.txt", "edit")),
        tool_turn("write_file", {"path": "precious.txt", "content": "clobbered"}),
        done_turn("Left the file alone."),
    ])
    orchestrator = build_orchestrator(settings, router)

    async def drive() -> Any:
        task = await orchestrator.start("Overwrite a file")
        approval_id = next(
            event.detail["approval_id"] for event in reversed(task.events) if event.kind == "approval"
        )
        return await orchestrator.resolve_approval(task.id, approval_id, "denied")

    task = asyncio.run(drive())

    assert (workspace / "precious.txt").read_text(encoding="utf-8") == "original"
    assert task.state is TaskState.COMPLETED
    assert any("denied" in observation.lower() for observation in task.observations)


def test_a_persistently_failing_run_stops_and_admits_it(settings: Settings, workspace: Path):
    """Never claim success. After the re-plan budget, fail honestly."""
    (workspace / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (workspace / "test_broken.py").write_text("def test_bad():\n    assert 1 == 2\n", encoding="utf-8")

    router = ScriptedRouter([plan_turn(("Try something", "edit"))] + [done_turn()] * 20)
    orchestrator = build_orchestrator(settings, router)

    task = asyncio.run(orchestrator.start("Fix the unfixable"))

    assert task.state is TaskState.FAILED
    assert task.completion_status == "failed"
    assert task.replans == orchestrator.max_replans
    assert "did not pass verification" in task.summary


def test_a_tool_failure_is_retried_then_abandoned(settings: Settings, workspace: Path):
    add_passing_suite(workspace)
    # Reading a file that does not exist fails every time.
    router = ScriptedRouter(
        [plan_turn(("Read a missing file", "inspect"))]
        + [tool_turn("read_file", {"path": "nope.txt"}, f"call_{index}") for index in range(10)]
    )
    orchestrator = build_orchestrator(settings, router)

    task = asyncio.run(orchestrator.start("Read something missing"))

    assert task.state is TaskState.FAILED
    assert task.retries > orchestrator.max_retries
    assert any(event.kind == "error" for event in task.events)


# -- safety and bounds -----------------------------------------------------

def test_a_high_risk_tool_pauses_for_approval_instead_of_acting(tmp_path: Path, workspace: Path):
    guarded = Settings(
        project_root=tmp_path, workspace_root=workspace, data_dir=tmp_path / "data",
        default_provider="ollama", default_model="fake", permission_mode="guarded",
    )
    add_passing_suite(workspace)
    router = ScriptedRouter([
        plan_turn(("Run a shell command", "command")),
        tool_turn("run_terminal", {"command": "echo hello", "cwd": "."}),
    ])
    orchestrator = build_orchestrator(guarded, router)

    task = asyncio.run(orchestrator.start("Run a command"))

    assert task.state is TaskState.WAITING_FOR_APPROVAL
    assert task.completion_status == "awaiting_approval"
    approval_events = [event for event in task.events if event.kind == "approval"]
    assert approval_events and approval_events[0].detail.get("approval_id")
    # The run is durable across the pause.
    assert orchestrator.store.get(task.id).state is TaskState.WAITING_FOR_APPROVAL


def test_a_blocked_tool_does_not_end_the_run(settings: Settings, workspace: Path):
    """A refusal is information the agent can route around."""
    add_passing_suite(workspace)
    router = ScriptedRouter([
        plan_turn(("Try a blocked action", "command")),
        tool_turn("open_url", {"url": "file:///etc/passwd"}),
        done_turn("Used a permitted route instead."),
    ])
    orchestrator = build_orchestrator(settings, router)

    task = asyncio.run(orchestrator.start("Try something disallowed"))

    assert task.state is TaskState.COMPLETED
    assert any("blocked" in error.lower() for error in task.errors)


def test_a_long_plan_is_stopped_by_the_step_budget(settings: Settings, workspace: Path):
    """A plan larger than the step budget must stop and say so, not grind on."""
    add_passing_suite(workspace)
    router = ScriptedRouter(
        [plan_turn(*[(f"Write file {index}", "edit") for index in range(10)])]
        + [tool_turn("write_file", {"path": f"f{index}.txt", "content": "x"}, f"c{index}") for index in range(20)]
    )
    orchestrator = build_orchestrator(settings, router)
    orchestrator.max_steps = 3

    task = asyncio.run(orchestrator.start("Do far too much"))

    assert task.completion_status == "bounded"
    assert task.tool_calls <= orchestrator.max_steps
    assert "bounded" in task.summary.lower()
    # Work already done is reported, not discarded.
    assert "of 10 planned steps completed" in task.summary


def test_an_unreachable_model_fails_cleanly(settings: Settings, workspace: Path):
    add_passing_suite(workspace)
    router = ScriptedRouter([plan_turn(("Do a thing", "edit")), ModelError("no provider reachable")])
    orchestrator = build_orchestrator(settings, router)

    task = asyncio.run(orchestrator.start("Do a thing"))

    assert task.state is TaskState.FAILED
    assert task.completion_status == "provider_unavailable", "the provider failed, not the task"
    assert "reachable" in task.summary


# -- honesty about verification -------------------------------------------

def test_a_project_with_no_checks_completes_but_says_it_is_unverified(settings: Settings, workspace: Path):
    (workspace / "notes.md").write_text("just notes", encoding="utf-8")
    router = ScriptedRouter([
        plan_turn(("Write a note", "edit")),
        tool_turn("write_file", {"path": "note.txt", "content": "hi"}),
    ])
    orchestrator = build_orchestrator(settings, router)

    task = asyncio.run(orchestrator.start("Write a note"))

    assert task.state is TaskState.COMPLETED
    assert task.completion_status == "completed_unverified"
    assert "unverified" in task.summary


def test_the_executor_context_stays_focused_not_a_transcript_dump(settings: Settings, workspace: Path):
    """Context is assembled per step; it must carry the plan and the goal
    without replaying an entire conversation."""
    add_passing_suite(workspace)
    router = ScriptedRouter([
        plan_turn(("Step one", "inspect"), ("Step two", "edit")),
        done_turn("one done"),
        done_turn("two done"),
    ])
    orchestrator = build_orchestrator(settings, router)
    asyncio.run(orchestrator.start("A focused goal"))

    execution_prompt = router.prompts[1]
    assert "GOAL: A focused goal" in execution_prompt
    assert "PLAN:" in execution_prompt
    assert "CURRENT STEP" in execution_prompt
    assert len(execution_prompt) < 20_000


def test_observations_from_earlier_steps_reach_later_ones(settings: Settings, workspace: Path):
    add_passing_suite(workspace)
    (workspace / "config.ini").write_text("port=9999\n", encoding="utf-8")
    router = ScriptedRouter([
        plan_turn(("Read the config", "inspect"), ("Act on it", "edit")),
        tool_turn("read_file", {"path": "config.ini"}),
        done_turn("acted"),
    ])
    orchestrator = build_orchestrator(settings, router)
    asyncio.run(orchestrator.start("Use the config"))

    assert any("OBSERVED SO FAR" in prompt and "read_file" in prompt for prompt in router.prompts[2:])


# -- secret hygiene --------------------------------------------------------

def test_a_sensitive_tool_result_never_enters_the_timeline_or_context(tmp_path: Path, workspace: Path):
    """An approved credential read must not leak through observations."""
    guarded = Settings(
        project_root=tmp_path, workspace_root=workspace, data_dir=tmp_path / "data",
        default_provider="ollama", default_model="fake", permission_mode="guarded",
    )
    add_passing_suite(workspace)
    secret_value = "sk-live-this-must-never-be-echoed-anywhere-000"
    (workspace / ".env").write_text(f"OPENAI_API_KEY={secret_value}\n", encoding="utf-8")

    router = ScriptedRouter([
        plan_turn(("Read the env file", "inspect")),
        tool_turn("read_file", {"path": ".env"}),
        done_turn("Read it."),
    ])
    orchestrator = build_orchestrator(guarded, router)

    async def drive() -> Any:
        task = await orchestrator.start("Inspect configuration")
        assert task.state is TaskState.WAITING_FOR_APPROVAL
        approval_id = next(
            event.detail["approval_id"] for event in reversed(task.events) if event.kind == "approval"
        )
        return await orchestrator.resolve_approval(task.id, approval_id, "approved")

    task = asyncio.run(drive())

    serialized = json.dumps(orchestrator.store.get(task.id).as_dict())
    assert secret_value not in serialized, "the secret reached the persisted task record"
    assert any("sensitive" in observation for observation in task.observations)
    # Nothing that reached the model on later turns may carry it either.
    assert all(secret_value not in prompt for prompt in router.prompts)


# -- what the model is told about a tool result ----------------------------
# The agent acts on nothing else, and a wrong answer here is silent: the run
# continues, having been told something useless. This seam shipped a real
# defect once, when a file read was summarised down to its path.

class FakeResult:
    def __init__(self, output, error=None):
        self.output, self.error = output, error

    def model_text(self):
        return str(self.output)


class FakeOutcome:
    def __init__(self, name, output, error=None, sensitive=False):
        self.call = type("Call", (), {"name": name})()
        self.result = FakeResult(output, error)
        self.sensitive = sensitive


@pytest.mark.parametrize("name, output, expected", [
    ("read_file", {"path": "a.py", "content": "print(1)"}, "a.py:\nprint(1)"),
    ("run_tests", {"summary": "3 passed"}, "3 passed"),
    ("run_tests", {}, "checks finished"),
    ("project_map", {"file_count": 12, "commands": {}}, "12 files, commands {}"),
    ("write_file", {"path": "b.txt", "bytes": 9}, "b.txt (9 bytes)"),
    ("run_terminal", {"exit_code": 0}, "exit 0"),
    ("git_log", {"commits": [1, 2]}, "2 commit(s)"),
    ("git_status", {"changed": ["x"], "branch": "main"}, "1 changed path(s) on main"),
    ("anything", "plain text", "plain text"),
    ("anything", "", "done"),
])
def test_a_tool_result_is_described_by_what_the_model_needs_from_it(name, output, expected):
    from sam_backend.autonomy.observations import summarise_result

    assert summarise_result(name, FakeResult(output)) == expected


def test_a_file_read_carries_its_text_but_cannot_swamp_the_prompt():
    """The defect this seam exists to prevent, and the bound that limits it."""
    from sam_backend.autonomy.observations import OBSERVATION_CONTENT_LIMIT, summarise_result

    summary = summarise_result("read_file", FakeResult({"path": "big.py", "content": "x" * 10_000}))

    assert summary.startswith("big.py:\n"), "the model must know which file it is reading"
    assert "x" * 100 in summary, "the text itself is the point of a read"
    assert len(summary) <= OBSERVATION_CONTENT_LIMIT + len("big.py:\n")


@pytest.mark.parametrize("sensitive, output, error, describes", [
    (True, {"content": "sk-or-v1-OBSERVATION-SENTINEL-0123456789"}, None, "success"),
    (True, None, "sk-or-v1-OBSERVATION-SENTINEL-0123456789", "failure"),
])
def test_a_sensitive_result_never_reaches_the_model(sensitive, output, error, describes):
    from sam_backend.autonomy.observations import describe_failure, describe_success

    outcome = FakeOutcome("read_file", output, error, sensitive=sensitive)
    told = describe_success(outcome) if describes == "success" else describe_failure(outcome)

    assert "sk-or-v1-OBSERVATION-SENTINEL-0123456789" not in told
    assert "withheld" in told


def test_a_failure_without_a_message_still_says_something():
    from sam_backend.autonomy.observations import describe_failure

    assert describe_failure(FakeOutcome("run_terminal", None, error=None)) == \
        "The tool reported a failure without a message."


# -- who is driving, and has it been told to stop --------------------------
# Leaf mechanics: the orchestrator asks these and draws its own conclusions.
# Tested directly because the exclusivity contract is subtle and the token
# lifetime has a deliberate exception for a paused run.

def test_only_one_driver_may_hold_a_task_and_the_claim_is_released():
    from sam_backend.autonomy.control import RunControl, TaskAlreadyRunning

    control = RunControl()

    async def scenario():
        async with control.exclusive("t1"):
            assert control.driving("t1")
            with pytest.raises(TaskAlreadyRunning, match="already running"):
                async with control.exclusive("t1"):
                    pass
            # A different task is unaffected by the claim on this one.
            async with control.exclusive("t2"):
                assert control.driving("t2")
        assert not control.driving("t1") and not control.driving("t2")

    asyncio.run(scenario())


def test_a_claim_is_released_even_when_the_run_raises():
    from sam_backend.autonomy.control import RunControl

    control = RunControl()

    async def scenario():
        with pytest.raises(RuntimeError):
            async with control.exclusive("t1"):
                raise RuntimeError("the run blew up")
        assert not control.driving("t1"), "a crashed driver must not hold the task forever"

    asyncio.run(scenario())


def test_the_stop_token_outlives_a_pause_but_not_a_finished_run(settings: Settings):
    from sam_backend.cancellation import CancellationManager
    from sam_backend.autonomy.control import RunControl

    manager = CancellationManager()
    control = RunControl(manager)
    store = TaskStore(Database(settings.database_path))
    task = store.create("Stoppable")

    control.arm(task)
    assert manager.get(task.id) is not None and not control.cancelled(task)

    # Paused, not finished: Stop must still have something to flip.
    control.release(task)
    assert manager.get(task.id) is not None, "a waiting run stays stoppable"

    control.request_stop(task)
    assert control.cancelled(task)

    task.state = TaskState.COMPLETED
    control.release(task)
    assert manager.get(task.id) is None, "a finished run drops its token"


def test_without_a_cancellation_manager_nothing_is_ever_cancelled(settings: Settings):
    """The orchestrator is constructed without one in some tests; it must not crash."""
    from sam_backend.autonomy.control import RunControl

    control = RunControl(None)
    task = TaskStore(Database(settings.database_path)).create("No manager")

    control.arm(task)
    control.request_stop(task)
    control.release(task)
    assert control.cancelled(task) is False
