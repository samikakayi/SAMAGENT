"""An approval is single use -- including when two callers arrive together.

The sequential case was already covered: claim, execute, claim again, refused.
The concurrent case was not, and it is the one that matters. `authorize_approval`
is atomic, but it returns the record whether this caller won the transition or
merely found it already taken -- and "already taken" during an in-flight
execution reads as `executing`, which every caller was treating as success.
"""

from __future__ import annotations

import concurrent.futures as futures
import threading

import pytest

from sam_backend.db import Database


def approval(database, tool="workflow_import", arguments=None):
    return database.create_approval(
        conversation_id=None, tool_name=tool, tool_call_id=f"ui_{tool}",
        risk_level="high", reason="soak", arguments=arguments or {"workflow_sha256": "a" * 64},
        ttl_minutes=30,
    )["id"]


def test_only_one_caller_claims_an_approval(tmp_path):
    database = Database(tmp_path / "approvals.db")
    approval_id = database.create_approval(
        conversation_id=None, tool_name="workflow_import", tool_call_id="ui",
        risk_level="high", reason="soak", arguments={}, ttl_minutes=30)["id"]

    first = database.authorize_approval(approval_id, "approved")
    second = database.authorize_approval(approval_id, "approved")

    assert first["claimed"] is True, "the first caller performed the transition"
    assert second["claimed"] is False, "the second found it already taken"
    assert second["status"] == "executing", "and can still say what state it is in"


def test_concurrent_claims_produce_exactly_one_winner(tmp_path):
    """Two threads, one approval. Exactly one may be told it claimed it."""
    database = Database(tmp_path / "approvals.db")
    approval_id = approval(database)
    barrier = threading.Barrier(2)

    def claim(_index):
        barrier.wait(timeout=10)
        return database.authorize_approval(approval_id, "approved")

    with futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = [f.result() for f in [pool.submit(claim, i) for i in (0, 1)]]

    winners = [r for r in results if r and r.get("claimed")]
    assert len(winners) == 1, f"{len(winners)} callers were told they claimed it"


def test_a_denial_is_also_single_use(tmp_path):
    database = Database(tmp_path / "approvals.db")
    approval_id = approval(database)

    first = database.authorize_approval(approval_id, "denied")
    second = database.authorize_approval(approval_id, "denied")

    assert first["claimed"] is True and first["status"] == "denied"
    assert second["claimed"] is False


def test_an_expired_approval_is_never_claimed(tmp_path):
    database = Database(tmp_path / "approvals.db")
    approval_id = database.create_approval(
        conversation_id=None, tool_name="workflow_import", tool_call_id="ui",
        risk_level="high", reason="soak", arguments={}, ttl_minutes=0)["id"]

    result = database.authorize_approval(approval_id, "approved")

    assert result["claimed"] is False
    assert result["status"] == "expired"


def test_a_missing_approval_is_still_none(tmp_path):
    database = Database(tmp_path / "approvals.db")

    assert database.authorize_approval("apr_does_not_exist", "approved") is None


# --- and the same thing over the real HTTP surface ------------------------------


def _prepared(app):
    from sam_backend.workflows import prepare
    from sam_backend.workflows.models import WorkflowProvenance

    artifact = prepare(
        {"name": "single use", "nodes": [{"name": "S", "type": "n8n-nodes-base.manualTrigger",
                                          "parameters": {}}], "connections": {}},
        WorkflowProvenance(source="library", source_repository="test/library"))
    app.state.workflows.remember_artifact(artifact)
    return artifact.sha256


class SlowN8n:
    """An n8n slow enough that two claims genuinely overlap."""

    configured = True
    base_url = "http://fake:5678"

    def __init__(self):
        self.created: list[dict] = []
        self.started = threading.Event()

    def create_workflow(self, body):
        self.started.set()
        import time

        time.sleep(0.4)
        self.created.append(body)
        return {"id": f"wf-{len(self.created)}", "name": body.get("name", ""), "active": False}

    def list_credentials(self):
        return []


def test_two_overlapping_imports_create_one_workflow(client, app):
    """The defect: both callers passed the gate and n8n was asked twice."""
    fake = SlowN8n()
    app.state.workflows._n8n = fake
    sha = _prepared(app)

    opened = client.post("/api/workflows/import", json={"workflow_sha256": sha}).json()
    assert opened["approval_required"] is True
    barrier = threading.Barrier(2)

    def claim(_index):
        barrier.wait(timeout=10)
        return client.post("/api/workflows/import",
                           json={"workflow_sha256": sha, "approval_id": opened["approval_id"]})

    with futures.ThreadPoolExecutor(max_workers=2) as pool:
        responses = [f.result() for f in [pool.submit(claim, i) for i in (0, 1)]]

    accepted = [r for r in responses if r.status_code == 200]
    assert len(accepted) == 1, f"{len(accepted)} imports were accepted from one approval"
    assert len(fake.created) == 1, f"n8n was asked to create {len(fake.created)} workflows"
    assert sorted(r.status_code for r in responses) == [200, 409]
