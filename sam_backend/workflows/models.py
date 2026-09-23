"""What SAM knows about an automation workflow.

These are SAM's own types. The workflow library's file layout and n8n's REST
shapes both get converted into these at their boundaries, so the rest of SAM
never learns either vocabulary -- a second library, or a different automation
engine, changes one adapter rather than the agent.

Every workflow here came from somewhere else, so provenance travels with the
artifact rather than being recorded beside it. After adaptation the original
hash is still attached: "what did SAM change, and change from what" has to
stay answerable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class RiskFlag(StrEnum):
    """One specific reason a workflow is not merely reading something."""

    READ_ONLY = "READ_ONLY"
    NETWORK_READ = "NETWORK_READ"
    EXTERNAL_WRITE = "EXTERNAL_WRITE"
    CREDENTIAL_SENSITIVE = "CREDENTIAL_SENSITIVE"
    DATABASE_WRITE = "DATABASE_WRITE"
    FILESYSTEM_WRITE = "FILESYSTEM_WRITE"
    CODE_EXECUTION = "CODE_EXECUTION"
    SHELL_EXECUTION = "SHELL_EXECUTION"
    WEBHOOK_EXPOSURE = "WEBHOOK_EXPOSURE"
    SUBWORKFLOW_EXECUTION = "SUBWORKFLOW_EXECUTION"
    UNKNOWN_NODE = "UNKNOWN_NODE"
    FINANCIAL_ACTION = "FINANCIAL_ACTION"
    HIGH_IMPACT = "HIGH_IMPACT"


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        return {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}[self.value]


class LibraryState(StrEnum):
    """Whether what SAM just read is current, remembered, or absent."""

    AVAILABLE = "LIBRARY_AVAILABLE"
    STALE_CACHE = "LIBRARY_STALE_CACHE"
    UNAVAILABLE = "LIBRARY_UNAVAILABLE"


class WorkflowErrorCode(StrEnum):
    """The same vocabulary the rest of SAM already uses for failures."""

    AUTH = "AUTH"
    RATE_LIMIT = "RATE_LIMIT"
    TIMEOUT = "TIMEOUT"
    NETWORK = "NETWORK"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    NOT_FOUND = "NOT_FOUND"
    INVALID_WORKFLOW = "INVALID_WORKFLOW"
    UNSUPPORTED_NODE = "UNSUPPORTED_NODE"
    UNSUPPORTED_OPERATION = "UNSUPPORTED_OPERATION"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    TOO_LARGE = "TOO_LARGE"


class WorkflowError(RuntimeError):
    """A workflow operation failed, with a category the caller can branch on."""

    def __init__(self, message: str, code: WorkflowErrorCode = WorkflowErrorCode.NETWORK) -> None:
        super().__init__(message)
        self.code = code

    def as_dict(self) -> dict[str, str]:
        return {"error": str(self), "code": self.code.value}


def canonical_json(workflow: dict[str, Any]) -> str:
    """One byte-for-byte form of a workflow, so a hash means something.

    Sorted keys and fixed separators, because two dictionaries that differ
    only in key order are the same workflow and must not produce two hashes --
    an approval bound to a hash would otherwise expire on a re-serialisation.
    """
    return json.dumps(workflow, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def workflow_sha256(workflow: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(workflow).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class WorkflowProvenance:
    """Where this workflow came from, kept even after SAM rewrites it."""

    source: str
    source_repository: str = ""
    source_path: str = ""
    source_workflow_id: str = ""
    source_commit_sha: str = ""
    retrieved_at: str = ""
    original_sha256: str = ""
    adapted_sha256: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "source": self.source, "source_repository": self.source_repository,
            "source_path": self.source_path, "source_workflow_id": self.source_workflow_id,
            "source_commit_sha": self.source_commit_sha, "retrieved_at": self.retrieved_at,
            "original_sha256": self.original_sha256, "adapted_sha256": self.adapted_sha256,
        }


@dataclass(frozen=True, slots=True)
class WorkflowSummary:
    """Enough to choose a candidate, small enough to show ten of them.

    Deliberately not the workflow itself: a search that returned full JSON
    would put megabytes in front of the model to answer "which of these?"
    """

    workflow_id: str
    title: str
    description: str = ""
    services: tuple[str, ...] = ()
    trigger: str = "unknown"
    complexity: str = "unknown"
    category: str = ""
    source: str = ""
    source_path: str = ""
    size_bytes: int = 0
    match_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id, "title": self.title, "description": self.description,
            "services": list(self.services), "trigger": self.trigger, "complexity": self.complexity,
            "category": self.category, "source": self.source, "source_path": self.source_path,
            "size_bytes": self.size_bytes, "match_reason": self.match_reason,
        }


@dataclass(frozen=True, slots=True)
class CredentialRequirement:
    """A credential the workflow needs, and whether it has been mapped yet.

    `foreign_id` is what the exporting instance called it. It is recorded so a
    reviewer can see it, and never sent to n8n: an ID from someone else's
    instance means nothing on this one, and reusing it blindly would either
    fail or -- worse -- attach a real local credential nobody chose.
    """

    credential_type: str
    node_names: tuple[str, ...] = ()
    foreign_id: str = ""
    foreign_name: str = ""
    mapped_id: str = ""

    @property
    def resolved(self) -> bool:
        return bool(self.mapped_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "credential_type": self.credential_type, "node_names": list(self.node_names),
            "foreign_id": self.foreign_id, "foreign_name": self.foreign_name,
            "mapped_id": self.mapped_id, "resolved": self.resolved,
        }


@dataclass(frozen=True, slots=True)
class NodeFinding:
    """One node, and what the inspector could tell about it without running it."""

    name: str
    node_type: str
    category: str
    flags: tuple[RiskFlag, ...] = ()
    detail: str = ""
    disabled: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "node_type": self.node_type, "category": self.category,
            "flags": [flag.value for flag in self.flags], "detail": self.detail,
            "disabled": self.disabled,
        }


@dataclass(frozen=True, slots=True)
class WorkflowRiskAssessment:
    level: RiskLevel
    flags: tuple[RiskFlag, ...]
    reasons: tuple[str, ...] = ()
    # True when something could not be resolved -- an unavailable subworkflow,
    # say. The assessment is then a floor, not a verdict.
    incomplete: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value, "flags": [flag.value for flag in self.flags],
            "reasons": list(self.reasons), "incomplete": self.incomplete,
        }


@dataclass(frozen=True, slots=True)
class WorkflowInspection:
    """The deterministic reading of a workflow. No model, no execution."""

    node_count: int
    node_types: tuple[str, ...]
    triggers: tuple[str, ...]
    services: tuple[str, ...]
    nodes: tuple[NodeFinding, ...]
    credentials: tuple[CredentialRequirement, ...]
    http_destinations: tuple[str, ...]
    expressions: int
    disabled_nodes: tuple[str, ...]
    disconnected_nodes: tuple[str, ...]
    subworkflows: tuple[str, ...]
    risk: WorkflowRiskAssessment
    code_previews: tuple[dict[str, str], ...] = ()
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_count": self.node_count, "node_types": list(self.node_types),
            "triggers": list(self.triggers), "services": list(self.services),
            "nodes": [node.as_dict() for node in self.nodes],
            "credentials": [item.as_dict() for item in self.credentials],
            "http_destinations": list(self.http_destinations), "expressions": self.expressions,
            "disabled_nodes": list(self.disabled_nodes),
            "disconnected_nodes": list(self.disconnected_nodes),
            "subworkflows": list(self.subworkflows), "risk": self.risk.as_dict(),
            "code_previews": [dict(item) for item in self.code_previews],
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class WorkflowValidation:
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": list(self.errors), "warnings": list(self.warnings)}


@dataclass(frozen=True, slots=True)
class WorkflowDiff:
    """What changed, in words a reviewer can act on rather than raw JSON."""

    nodes_added: tuple[str, ...] = ()
    nodes_removed: tuple[str, ...] = ()
    nodes_changed: tuple[str, ...] = ()
    services_added: tuple[str, ...] = ()
    services_removed: tuple[str, ...] = ()
    credentials_added: tuple[str, ...] = ()
    credentials_removed: tuple[str, ...] = ()
    trigger_changed: str = ""
    risk_changed: str = ""
    summary: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "nodes_added": list(self.nodes_added), "nodes_removed": list(self.nodes_removed),
            "nodes_changed": list(self.nodes_changed), "services_added": list(self.services_added),
            "services_removed": list(self.services_removed),
            "credentials_added": list(self.credentials_added),
            "credentials_removed": list(self.credentials_removed),
            "trigger_changed": self.trigger_changed, "risk_changed": self.risk_changed,
            "summary": list(self.summary),
        }


@dataclass(slots=True)
class WorkflowArtifact:
    """A concrete workflow SAM is prepared to do something with.

    The hash is of `workflow` as it stands. An approval binds to that hash, so
    editing anything here produces a different artifact that the old approval
    does not cover.
    """

    workflow: dict[str, Any]
    provenance: WorkflowProvenance
    inspection: WorkflowInspection
    validation: WorkflowValidation
    diff: WorkflowDiff | None = None
    credentials: tuple[CredentialRequirement, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def sha256(self) -> str:
        return workflow_sha256(self.workflow)

    @property
    def name(self) -> str:
        return str(self.workflow.get("name") or "Untitled workflow")

    @property
    def unresolved_credentials(self) -> tuple[CredentialRequirement, ...]:
        return tuple(item for item in self.credentials if not item.resolved)

    def as_dict(self, *, include_workflow: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name, "sha256": self.sha256,
            "provenance": self.provenance.as_dict(),
            "inspection": self.inspection.as_dict(),
            "validation": self.validation.as_dict(),
            "credentials": [item.as_dict() for item in self.credentials],
            "unresolved_credentials": [item.as_dict() for item in self.unresolved_credentials],
            "notes": list(self.notes),
        }
        if self.diff is not None:
            payload["diff"] = self.diff.as_dict()
        if include_workflow:
            payload["workflow"] = self.workflow
        return payload


@dataclass(frozen=True, slots=True)
class N8nExecutionStatus:
    execution_id: str
    workflow_id: str
    status: str
    started_at: str = ""
    finished_at: str = ""
    duration_ms: int | None = None
    failed_node: str = ""
    error: str = ""
    outputs: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id, "workflow_id": self.workflow_id,
            "status": self.status, "started_at": self.started_at, "finished_at": self.finished_at,
            "duration_ms": self.duration_ms, "failed_node": self.failed_node,
            "error": self.error, "outputs": [dict(item) for item in self.outputs],
        }
