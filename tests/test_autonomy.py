"""The autonomous loop, end to end, against a scripted model.

These are the tests that decide whether SAM is an agent or a chatbot: a goal
must travel through planning, real tool execution, real verification and
self-correction, and must refuse to call itself done when it is not.
"""

from __future__ import annotations

import asyncio
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
    # The verification engine really ran the project's suite.
    assert any(
        check.get("outcome") == "PASSED"
        for result in task.test_results
        for check in (result.get("checks") or [])
    )


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
