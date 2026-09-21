"""Task state machine, persistence, and the verification engine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sam_backend.db import Database
from sam_backend.project_map import ProjectScanner
from sam_backend.tasks import (
    AgentTask,
    IllegalTransition,
    TaskState,
    TaskStep,
    TaskStore,
)
from sam_backend.verification import (
    CheckOutcome,
    VerificationEngine,
    VerificationReport,
    parse_test_output,
    parse_typecheck_output,
)


@pytest.fixture()
def store(tmp_path: Path) -> TaskStore:
    return TaskStore(Database(tmp_path / "sam.sqlite3"))


# -- task state machine ----------------------------------------------------

def test_a_task_starts_idle_and_persists(store: TaskStore):
    task = store.create("Fix the login page", conversation_id="conv_1")

    reloaded = store.get(task.id)
    assert reloaded is not None
    assert reloaded.goal == "Fix the login page"
    assert reloaded.state is TaskState.IDLE
    assert reloaded.conversation_id == "conv_1"


def test_a_task_needs_a_goal(store: TaskStore):
    with pytest.raises(ValueError):
        store.create("   ")


def test_the_full_happy_path_is_a_legal_walk(store: TaskStore):
    task = store.create("Add a feature")
    for state in (
        TaskState.UNDERSTANDING, TaskState.SCANNING, TaskState.PLANNING,
        TaskState.EXECUTING, TaskState.OBSERVING, TaskState.VALIDATING, TaskState.COMPLETED,
    ):
        store.transition(task, state)
    assert task.state is TaskState.COMPLETED
    assert task.terminal


def test_a_failing_run_can_self_heal_and_retry(store: TaskStore):
    task = store.create("Fix a bug")
    for state in (
        TaskState.UNDERSTANDING, TaskState.PLANNING, TaskState.EXECUTING,
        TaskState.VALIDATING, TaskState.FIXING, TaskState.RETRYING, TaskState.VALIDATING,
        TaskState.COMPLETED,
    ):
        store.transition(task, state)
    assert task.state is TaskState.COMPLETED


def test_an_illegal_transition_is_refused_rather_than_recorded(store: TaskStore):
    """A bad jump means the orchestrator has a bug; persisting it would hide that."""
    task = store.create("Do something")
    store.transition(task, TaskState.UNDERSTANDING)
    store.transition(task, TaskState.PLANNING)
    store.transition(task, TaskState.EXECUTING)
    store.transition(task, TaskState.VALIDATING)
    store.transition(task, TaskState.COMPLETED)

    with pytest.raises(IllegalTransition):
        store.transition(task, TaskState.EXECUTING)
    assert store.get(task.id).state is TaskState.COMPLETED


def test_planning_cannot_be_skipped_straight_from_idle(store: TaskStore):
    task = store.create("Do something")
    with pytest.raises(IllegalTransition):
        store.transition(task, TaskState.EXECUTING)


def test_any_state_can_be_cancelled(store: TaskStore):
    task = store.create("Long job")
    store.transition(task, TaskState.UNDERSTANDING)
    store.transition(task, TaskState.PLANNING)
    store.transition(task, TaskState.EXECUTING)
    store.transition(task, TaskState.CANCELLED)
    assert task.terminal


def test_a_plan_and_its_progress_survive_a_reload(store: TaskStore):
    task = store.create("Ship it")
    store.set_plan(task, [
        TaskStep(index=1, text="Read auth.py", kind="inspect"),
        TaskStep(index=2, text="Patch the handler", kind="edit"),
    ])
    task.plan[0].status = "done"
    store.note_files(task, ["sam_backend/auth.py", "sam_backend/auth.py"])
    store.save(task)

    reloaded = store.get(task.id)
    assert [step.status for step in reloaded.plan] == ["done", "pending"]
    assert reloaded.current_step.text == "Patch the handler"
    # Duplicate paths are recorded once.
    assert reloaded.modified_files == ["sam_backend/auth.py"]


def test_an_interrupted_task_is_listed_as_resumable(store: TaskStore):
    running = store.create("Interrupted work")
    store.transition(running, TaskState.UNDERSTANDING)
    store.transition(running, TaskState.PLANNING)
    store.transition(running, TaskState.EXECUTING)

    done = store.create("Finished work")
    store.transition(done, TaskState.UNDERSTANDING)
    store.transition(done, TaskState.PLANNING)
    store.transition(done, TaskState.COMPLETED)

    resumable = [task.id for task in store.resumable()]
    assert running.id in resumable
    assert done.id not in resumable


def test_the_list_view_summarises_without_dumping_timelines(store: TaskStore):
    task = store.create("Summarise me")
    store.set_plan(task, [TaskStep(index=1, text="a"), TaskStep(index=2, text="b")])
    task.plan[0].status = "done"
    store.save(task)

    row = store.list()[0]
    assert row["steps_total"] == 2 and row["steps_done"] == 1
    assert "events" not in row


def test_the_public_view_caps_a_runaway_timeline(store: TaskStore):
    task = store.create("Chatty task")
    for index in range(500):
        task.events.append(task.events[0].__class__("tool", f"step {index}"))

    payload = task.public_dict(event_limit=50)
    assert len(payload["events"]) == 50
    assert payload["event_count"] == 501
    assert payload["events"][-1]["message"] == "step 499"


# -- verification engine ---------------------------------------------------

def test_pytest_output_is_parsed_in_quiet_and_verbose_form():
    assert parse_test_output("9 passed, 1 warning in 0.10s")[:2] == (9, 0)
    assert parse_test_output("==== 5 failed, 443 passed, 9 skipped in 42.81s ====")[:2] == (443, 5)
    # pytest counts collection errors separately; they are failures too.
    assert parse_test_output("=== 2 errors, 3 passed in 1.0s ===")[:2] == (3, 2)


def test_other_runners_are_parsed_too():
    assert parse_test_output("Tests:       2 failed, 1 skipped, 7 passed, 10 total")[:2] == (7, 2)
    assert parse_test_output("Tests  3 failed | 20 passed")[:2] == (20, 3)
    assert parse_test_output("test result: FAILED. 12 passed; 3 failed")[:2] == (12, 3)


def test_an_unrecognised_runner_reports_unknown_rather_than_guessing():
    """A confident wrong count is worse than an honest None."""
    assert parse_test_output("Everything looks fine to me")[:2] == (None, None)


def test_typescript_errors_are_extracted():
    output = "src/a.ts(12,5): error TS2345: Argument of type 'string' is not assignable.\nDone."
    assert parse_typecheck_output(output) == [
        "src/a.ts(12,5): error TS2345: Argument of type 'string' is not assignable."
    ]


def test_a_passing_command_is_reported_as_passed(tmp_path: Path):
    result = VerificationEngine().run_check("test", "python -c \"print('1 passed')\"", tmp_path)
    assert result.outcome is CheckOutcome.PASSED
    assert result.exit_code == 0
    assert result.ok


def test_a_nonzero_exit_fails_even_when_output_looks_clean(tmp_path: Path):
    """The exit code is the authority; text is only ever supporting detail."""
    result = VerificationEngine().run_check("test", "python -c \"print('9 passed'); raise SystemExit(1)\"", tmp_path)
    assert result.outcome is CheckOutcome.FAILED
    assert result.passed_count == 9
    assert result.blocking


def test_a_zero_exit_still_fails_when_the_runner_reported_failures(tmp_path: Path):
    """Some runners exit 0 on failure; the parsed tally must not be ignored."""
    result = VerificationEngine().run_check("test", "python -c \"print('2 failed, 3 passed')\"", tmp_path)
    assert result.exit_code == 0
    assert result.outcome is CheckOutcome.FAILED


def test_an_unknown_runner_is_skipped_not_executed(tmp_path: Path):
    marker = tmp_path / "executed.txt"
    result = VerificationEngine().run_check("test", f"curl -o {marker} http://example.com", tmp_path)
    assert result.outcome is CheckOutcome.SKIPPED
    assert not marker.exists()


def test_a_missing_runner_is_skipped_with_a_reason(tmp_path: Path):
    result = VerificationEngine().run_check("test", "jest --ci", tmp_path)
    assert result.outcome is CheckOutcome.SKIPPED
    assert "not installed" in result.reason


def test_a_hanging_command_times_out_instead_of_blocking_forever(tmp_path: Path):
    result = VerificationEngine(timeout=1).run_check(
        "test", "python -c \"import time; time.sleep(30)\"", tmp_path,
    )
    assert result.outcome is CheckOutcome.TIMEOUT
    assert result.blocking


def test_a_project_with_no_checks_is_skipped_never_assumed_green(tmp_path: Path):
    (tmp_path / "notes.txt").write_text("no code here", encoding="utf-8")
    project_map = ProjectScanner().scan(tmp_path)

    report = VerificationEngine().verify(project_map, tmp_path)

    assert report.checks[0].outcome is CheckOutcome.SKIPPED
    assert report.ok, "a skip is not a failure"
    assert not report.verified, "but it is not a verified pass either"


def test_a_report_is_only_verified_when_something_actually_passed():
    from sam_backend.verification import CheckResult

    skipped_only = VerificationReport(checks=[
        CheckResult(kind="test", command="", outcome=CheckOutcome.SKIPPED, reason="none"),
    ])
    assert skipped_only.ok and not skipped_only.verified

    real_pass = VerificationReport(checks=[
        CheckResult(kind="test", command="pytest", outcome=CheckOutcome.PASSED, exit_code=0),
    ])
    assert real_pass.ok and real_pass.verified


def test_a_real_pytest_suite_is_detected_and_run(tmp_path: Path):
    """End to end: discovered command -> executed -> parsed verdict."""
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "test_sample.py").write_text(
        "def test_one():\n    assert True\n\n\ndef test_two():\n    assert True\n", encoding="utf-8",
    )
    project_map = ProjectScanner().scan(tmp_path)
    assert project_map.commands["test"] == "python -m pytest"

    report = VerificationEngine(timeout=120).verify(project_map, tmp_path, kinds=["test"])

    assert report.verified, report.summary_text()
    assert report.checks[0].passed_count == 2


def test_a_failing_suite_surfaces_the_failing_test_name(tmp_path: Path):
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "test_broken.py").write_text(
        "def test_good():\n    assert True\n\n\ndef test_bad():\n    assert 1 == 2\n", encoding="utf-8",
    )
    project_map = ProjectScanner().scan(tmp_path)

    report = VerificationEngine(timeout=120).verify(project_map, tmp_path, kinds=["test"])

    assert not report.ok
    check = report.checks[0]
    assert check.failed_count == 1 and check.passed_count == 1
    assert any("test_bad" in failure for failure in check.failures)
    assert json.dumps(report.as_dict())
