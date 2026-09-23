"""Workflow Intelligence: find an automation, understand it, then decide.

SAM does the reasoning and holds the gate; n8n runs the automation. Nothing
here executes a workflow's contents to understand it, and nothing imported is
active until someone says so separately.
"""

from __future__ import annotations

from .inspector import assess, inspect
from .intelligence import WorkflowIntelligence
from .library import GitHubWorkflowLibrary, WorkflowLibraryProvider
from .models import (
    LibraryState,
    N8nExecutionStatus,
    RiskFlag,
    RiskLevel,
    WorkflowArtifact,
    WorkflowError,
    WorkflowErrorCode,
    WorkflowProvenance,
    WorkflowSummary,
    workflow_sha256,
)
from .n8n import N8nClient
from .service import approval_fingerprint, diff, prepare, sanitize_for_model, validate

__all__ = [
    "GitHubWorkflowLibrary", "LibraryState", "N8nClient", "N8nExecutionStatus",
    "RiskFlag", "RiskLevel", "WorkflowArtifact", "WorkflowError", "WorkflowErrorCode",
    "WorkflowIntelligence", "WorkflowLibraryProvider", "WorkflowProvenance", "WorkflowSummary",
    "approval_fingerprint", "assess", "diff", "inspect", "prepare", "sanitize_for_model",
    "validate", "workflow_sha256",
]
