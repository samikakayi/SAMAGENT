"""The facade the tools and the API both call.

Holds the one library provider, the one n8n client, and the prepared artifacts
between `prepare` and `import`. Keeping the artifact here rather than passing
it through the model is what makes hash-bound approval mean anything: the
bytes that were inspected are the bytes that get sent, because the model never
holds them in between -- it holds a hash.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .inspector import inspect
from .library import GitHubWorkflowLibrary
from .models import (
    LibraryState,
    WorkflowArtifact,
    WorkflowError,
    WorkflowErrorCode,
    WorkflowProvenance,
)
from .n8n import N8nClient
from .service import approval_fingerprint, prepare

# Prepared artifacts are short-lived by design: an approval that survived a
# week would be authorising something nobody remembers looking at.
ARTIFACT_TTL_SECONDS = 60 * 60
MAX_ARTIFACTS = 25


class WorkflowIntelligence:
    """Search, understand, prepare, and -- only with approval -- import."""

    def __init__(self, settings: Any, *, library: Any = None, n8n: Any = None) -> None:
        self.settings = settings
        self.library = library or GitHubWorkflowLibrary(Path(getattr(settings, "data_dir", ".")))
        self._n8n = n8n
        self._artifacts: dict[str, tuple[float, WorkflowArtifact]] = {}

    # -- the configured instance, and only that one ------------------------
    @property
    def n8n(self) -> N8nClient:
        """Built from settings every time, so a credential change takes effect.

        The base URL comes from settings alone. No workflow, no tool argument
        and no model output can redirect this.
        """
        if self._n8n is not None:
            return self._n8n
        return N8nClient(getattr(self.settings, "n8n_base_url", "") or "",
                         getattr(self.settings, "n8n_api_key", None))

    # -- read ---------------------------------------------------------------
    def search(self, query: str = "", **filters: Any) -> dict[str, Any]:
        found = self.library.search(query, **filters)
        return {
            "query": query, "count": len(found),
            "library_state": getattr(self.library, "state", LibraryState.AVAILABLE).value,
            "results": [item.as_dict() for item in found],
        }

    def inspect_workflow(self, workflow_id: str) -> dict[str, Any]:
        workflow, provenance = self.library.get_workflow(workflow_id)
        report = inspect(workflow)
        return {
            "workflow_id": workflow_id, "name": str(workflow.get("name") or workflow_id),
            "provenance": provenance.as_dict(), "inspection": report.as_dict(),
        }

    # -- prepare ------------------------------------------------------------
    def prepare_workflow(
        self, workflow_id: str, *, name: str = "",
        credential_mapping: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Produce the exact artifact an import would send, and hash it."""
        workflow, provenance = self.library.get_workflow(workflow_id)
        candidate = dict(workflow)
        if name:
            candidate["name"] = name[:200]
        available: list[dict[str, str]] = []
        if credential_mapping and self.n8n.configured:
            try:
                available = self.n8n.list_credentials()
            except WorkflowError:
                available = []  # mapping is then checked against nothing, not against a guess
        artifact = prepare(candidate, provenance, original=workflow,
                           credential_mapping=credential_mapping, available_credentials=available)
        self._remember(artifact)
        return self._describe(artifact)

    def _remember(self, artifact: WorkflowArtifact) -> None:
        now = time.time()
        self._artifacts = {
            key: value for key, value in self._artifacts.items()
            if now - value[0] < ARTIFACT_TTL_SECONDS
        }
        if len(self._artifacts) >= MAX_ARTIFACTS:
            oldest = min(self._artifacts, key=lambda key: self._artifacts[key][0])
            self._artifacts.pop(oldest, None)
        self._artifacts[artifact.sha256] = (now, artifact)

    def artifact(self, workflow_sha256: str) -> WorkflowArtifact:
        entry = self._artifacts.get(str(workflow_sha256 or "").lower())
        if entry is None:
            raise WorkflowError(
                "No prepared workflow with that hash. Prepare it again -- an artifact expires, and a "
                "hash that is not held here is one SAM never inspected.",
                WorkflowErrorCode.NOT_FOUND)
        return entry[1]

    def _describe(self, artifact: WorkflowArtifact) -> dict[str, Any]:
        payload = artifact.as_dict()
        payload["approval_fingerprint"] = approval_fingerprint(
            workflow_sha=artifact.sha256, operation="workflow_import", target=self.target)
        payload["target_instance"] = self.target
        payload["importable"] = artifact.validation.ok and not artifact.unresolved_credentials
        return payload

    @property
    def target(self) -> str:
        return getattr(self.settings, "n8n_base_url", "") or "(no n8n configured)"

    # -- mutate -------------------------------------------------------------
    def import_workflow(self, workflow_sha256: str) -> dict[str, Any]:
        """Create the prepared artifact in n8n, inactive.

        Refuses on an unmapped credential: importing a workflow that cannot run
        leaves a broken automation someone has to find and clean up.
        """
        artifact = self.artifact(workflow_sha256)
        if not artifact.validation.ok:
            raise WorkflowError(
                "This workflow did not validate: " + "; ".join(artifact.validation.errors[:3]),
                WorkflowErrorCode.VALIDATION_FAILED)
        if artifact.unresolved_credentials:
            missing = ", ".join(item.credential_type for item in artifact.unresolved_credentials)
            raise WorkflowError(
                f"Map these credentials to ones that exist in your n8n first: {missing}.",
                WorkflowErrorCode.VALIDATION_FAILED)
        created = self.n8n.create_workflow(artifact.workflow)
        return {
            "imported": True, "workflow_id": created["id"], "name": created["name"],
            # n8n creates inactive and SAM never asks otherwise; activation is
            # a separate decision with its own approval.
            "active": bool(created.get("active")), "target_instance": self.target,
            "workflow_sha256": artifact.sha256, "provenance": artifact.provenance.as_dict(),
            "risk": artifact.inspection.risk.as_dict(),
        }

    def set_active(self, workflow_id: str, active: bool) -> dict[str, Any]:
        result = self.n8n.set_active(workflow_id, active)
        return {**result, "target_instance": self.target}

    def run_status(self, workflow_id: str = "", limit: int = 5) -> dict[str, Any]:
        executions = self.n8n.executions(workflow_id, limit)
        return {
            "workflow_id": workflow_id, "count": len(executions),
            "target_instance": self.target,
            "executions": [item.as_dict() for item in executions],
            # n8n reporting success says the workflow ran, not that it achieved
            # what the task wanted. Verification stays SAM's own job.
            "note": "An execution status is what n8n did, not evidence the task's goal was met.",
        }

    # -- status -------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        library: dict[str, Any]
        try:
            library = self.library.status()
        except Exception:  # noqa: BLE001 - a status panel must not fail
            library = {"state": LibraryState.UNAVAILABLE.value}
        return {"library": library, "n8n": self.n8n.status()}
