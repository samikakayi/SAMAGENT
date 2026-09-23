"""Workflow Intelligence: find an automation, understand it, then decide.

SAM does the reasoning and holds the gate; n8n runs the automation. Nothing
here executes a workflow's contents to understand it, and nothing imported is
active until someone says so separately.
"""

from __future__ import annotations

from .goals import GoalPlan, GoalReading, generate_workflow, plan_goal, read_goal, score_candidate
from .inspector import activation, assess, inspect
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
from .n8n import REQUIRED_SCOPES, N8nClient
from .service import approval_fingerprint, diff, prepare, sanitize_for_model, validate

__all__ = [
    "REQUIRED_SCOPES", "GitHubWorkflowLibrary", "GoalPlan", "GoalReading",
    "LibraryState", "N8nClient", "N8nExecutionStatus",
    "RiskFlag", "RiskLevel", "WorkflowArtifact", "WorkflowError", "WorkflowErrorCode",
    "WorkflowIntelligence", "WorkflowLibraryProvider", "WorkflowProvenance", "WorkflowSummary",
    "activation", "approval_fingerprint", "assess", "diff", "generate_workflow", "inspect",
    "plan_goal", "prepare", "read_goal", "sanitize_for_model", "score_candidate",
    "validate", "workflow_sha256",
]
