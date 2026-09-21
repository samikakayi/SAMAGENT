"""Checkpoints, scoped diff, rollback, retry and restart handling.

An agent that edits files must be able to show exactly what it changed --
and only that -- and to put it back. Through the real HTTP surface.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from fastapi.testclient import TestClient

from tests.test_task_lifecycle import ScriptedAdapter, make_client, pending_approval_id, wait_for


def writer(paths: dict[str, str], delay: float = 0.0) -> ScriptedAdapter:
    """Writes each path once, in order."""
    items = list(paths.items())
    return ScriptedAdapter(
        steps=len(items), tool="write_file", delay=delay,
        arguments=lambda turn: {"path": items[turn - 1][0], "content": items[turn - 1][1]},
    )


def approve_all(client: TestClient, task_id: str, *, rounds: int) -> dict:
    """Approve each *new* pause. Re-posting an id already decided is a replay
    the executor refuses, so the helper must never do it."""
    approved: set[str] = set()
    deadline = time.time() + 60
    while len(approved) < rounds and time.time() < deadline:
        task = wait_for(client, task_id, state="WAITING_FOR_APPROVAL")
        if task["terminal"]:
            break
        approval_id = pending_approval_id(task)
        if approval_id in approved:
            time.sleep(0.15)
            continue
        approved.add(approval_id)
        client.post(f"/api/approvals/{approval_id}/decision", json={"decision": "approved"})
    return wait_for(client, task_id)


def test_a_run_checkpoints_before_its_first_edit_and_the_diff_is_scoped_to_its_files(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "untouched.txt").write_text("leave me\n", encoding="utf-8")

    with make_client(tmp_path, writer({"created.txt": "new file\n"})) as client:
        task_id = client.post("/api/tasks", json={"goal": "create"}).json()["task_id"]
        task = wait_for(client, task_id)
        assert task["state"] == "COMPLETED"
        assert any("Checkpoint recorded" in event["message"] for event in task["events"])

        diff = client.get(f"/api/tasks/{task_id}/diff").json()
        assert [item["path"] for item in diff["files"]] == ["created.txt"], "only files this run touched"
        created = diff["files"][0]
        assert created["status"] == "created"
        assert "+new file" in created["diff"] and "/dev/null" in created["diff"]
        assert "untouched" not in str(diff)


def test_rollback_restores_the_users_pre_task_content(tmp_path: Path):
    """The checkpoint is what the user had, including uncommitted edits."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    original = "the user's uncommitted draft\n"
    (workspace / "notes.txt").write_text(original, encoding="utf-8")

    with make_client(tmp_path, writer({"notes.txt": "agent overwrote this\n"}), mode="guarded") as client:
        task_id = client.post("/api/tasks", json={"goal": "overwrite"}).json()["task_id"]
        task = approve_all(client, task_id, rounds=1)
        assert task["terminal"]
        assert (workspace / "notes.txt").read_text(encoding="utf-8") == "agent overwrote this\n"

        diff = client.get(f"/api/tasks/{task_id}/diff").json()["files"][0]
        assert diff["status"] == "modified"
        assert "-the user" in diff["diff"] and "+agent overwrote this" in diff["diff"]

        outcome = client.post(f"/api/tasks/{task_id}/rollback").json()
        assert outcome["rolled_back"] is True and outcome["files"] == ["notes.txt"]
        assert (workspace / "notes.txt").read_text(encoding="utf-8") == original
        # Honest the second time.
        assert client.post(f"/api/tasks/{task_id}/rollback").json()["rolled_back"] is False
        assert client.get(f"/api/tasks/{task_id}/diff").json()["rolled_back"] is True


def test_rollback_removes_files_the_run_created(tmp_path: Path):
    workspace = tmp_path / "workspace"
    with make_client(tmp_path, writer({"a.txt": "a", "b.txt": "b"})) as client:
        task_id = client.post("/api/tasks", json={"goal": "create two"}).json()["task_id"]
        wait_for(client, task_id)
        assert (workspace / "a.txt").exists() and (workspace / "b.txt").exists()

        outcome = client.post(f"/api/tasks/{task_id}/rollback").json()

        assert sorted(outcome["files"]) == ["a.txt", "b.txt"]
        assert not (workspace / "a.txt").exists() and not (workspace / "b.txt").exists()


def test_rollback_is_refused_while_the_run_is_live(tmp_path: Path):
    adapter = writer({f"f{index}.txt": "x" for index in range(6)}, delay=0.4)
    with make_client(tmp_path, adapter) as client:
        task_id = client.post("/api/tasks", json={"goal": "long"}).json()["task_id"]
        wait_for(client, task_id, state="EXECUTING")

        outcome = client.post(f"/api/tasks/{task_id}/rollback").json()

        assert outcome["rolled_back"] is False and "Stop the task" in outcome["reason"]
        client.post(f"/api/tasks/{task_id}/cancel")
        wait_for(client, task_id)
        assert client.post(f"/api/tasks/{task_id}/rollback").json()["rolled_back"] is True


def test_the_diff_flags_files_the_user_already_had_changes_in(tmp_path: Path):
    """In a git workspace, an edit to an already-dirty file is labelled as such."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    (workspace / "clean.txt").write_text("committed\n", encoding="utf-8")
    subprocess.run(["git", "add", "clean.txt"], cwd=workspace, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "clean"], cwd=workspace, check=True)
    (workspace / "dirty.txt").write_text("user edit in progress\n", encoding="utf-8")  # the user's change

    with make_client(tmp_path, writer({"dirty.txt": "agent\n", "clean.txt": "agent\n"}), mode="guarded") as client:
        task_id = client.post("/api/tasks", json={"goal": "edit both"}).json()["task_id"]
        task = approve_all(client, task_id, rounds=2)
        assert task["terminal"]

        flags = {
            item["path"]: item["had_user_changes"]
            for item in client.get(f"/api/tasks/{task_id}/diff").json()["files"]
        }
        assert flags == {"dirty.txt": True, "clean.txt": False}
        assert "dirty.txt" in task["preexisting_changes"]


def test_retry_starts_a_fresh_run_of_the_same_goal(tmp_path: Path):
    with make_client(tmp_path, writer({"a.txt": "a", "b.txt": "b"})) as client:
        first = client.post("/api/tasks", json={"goal": "make files"}).json()["task_id"]
        wait_for(client, first)

        response = client.post(f"/api/tasks/{first}/retry")
        assert response.status_code == 202
        second = response.json()["task_id"]
        assert second != first and response.json()["retry_of"] == first

        task = wait_for(client, second)
        assert task["goal"] == "make files" and task["terminal"]
        assert any("Retry of" in event["message"] for event in task["events"])
        assert client.get(f"/api/tasks/{first}").json()["task"]["state"] == "COMPLETED", "history intact"


def test_unknown_ids_are_404_on_every_new_endpoint(tmp_path: Path):
    with make_client(tmp_path, writer({"a.txt": "a"})) as client:
        assert client.get("/api/tasks/task_nope/diff").status_code == 404
        assert client.post("/api/tasks/task_nope/rollback").status_code == 404
        assert client.post("/api/tasks/task_nope/retry").status_code == 404


def test_a_restart_marks_mid_flight_runs_interrupted_but_leaves_paused_ones_waiting(tmp_path: Path):
    from sam_backend.db import Database
    from sam_backend.tasks import TaskState, TaskStore

    (tmp_path / "workspace").mkdir()
    # What a killed process leaves behind: rows mid-flight in the database.
    store = TaskStore(Database(tmp_path / "data" / "sam.sqlite3"))
    executing = store.create("was executing")
    for state in (TaskState.UNDERSTANDING, TaskState.PLANNING, TaskState.EXECUTING):
        store.transition(executing, state)
    paused = store.create("was paused")
    for state in (TaskState.UNDERSTANDING, TaskState.PLANNING, TaskState.EXECUTING, TaskState.WAITING_FOR_APPROVAL):
        store.transition(paused, state)
    finished = store.create("was done")
    for state in (TaskState.UNDERSTANDING, TaskState.PLANNING, TaskState.COMPLETED):
        store.transition(finished, state)

    with make_client(tmp_path, writer({"a.txt": "a"})) as client:  # startup hook runs here
        rows = {row["id"]: row for row in client.get("/api/tasks?limit=10").json()["tasks"]}

        assert rows[executing.id]["state"] == "FAILED"
        assert rows[executing.id]["completion_status"] == "interrupted"
        assert "restarted" in rows[executing.id]["summary"]
        assert rows[paused.id]["state"] == "WAITING_FOR_APPROVAL", "nothing was half-done; keep waiting"
        assert rows[finished.id]["state"] == "COMPLETED"
        assert client.post(f"/api/tasks/{executing.id}/retry").status_code == 202
