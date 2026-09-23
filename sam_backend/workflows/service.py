"""Turning a candidate workflow into something SAM is willing to act on.

The order matters and is enforced here rather than left to a caller:

    fetch -> inspect -> validate -> diff -> hash -> approve -> import

A model may propose an adaptation, but it never gains authority by proposing
one: the deterministic validator and the risk engine both run again afterwards
on whatever came back, and the approval binds to the hash of the exact bytes
that would be sent. Change one character and the old approval no longer covers
it -- which is the point, because otherwise "approve this harmless workflow"
could be followed by sending a different one.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from ..policy import redact_secrets
from .inspector import inspect
from .models import (
    CredentialRequirement,
    RiskLevel,
    WorkflowArtifact,
    WorkflowDiff,
    WorkflowError,
    WorkflowErrorCode,
    WorkflowInspection,
    WorkflowProvenance,
    WorkflowValidation,
    canonical_json,
    workflow_sha256,
)

MAX_WORKFLOW_BYTES = 1_500_000
MAX_NODES = 400
MAX_NAME_CHARS = 200

# Long random-looking literals inside a workflow are usually a key someone
# pasted where a credential reference belonged.
_SECRET_SHAPED = re.compile(
    r"(sk-[A-Za-z0-9._\-]{16,}|gsk_[A-Za-z0-9._\-]{16,}|AIza[A-Za-z0-9._\-]{16,}"
    r"|ghp_[A-Za-z0-9]{16,}|xox[baprs]-[A-Za-z0-9-]{10,}|eyJ[A-Za-z0-9._\-]{20,})"
)


def validate(workflow: Any) -> WorkflowValidation:
    """Everything that must be true before this is worth approving."""
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(workflow, dict):
        return WorkflowValidation(False, ("The workflow is not a JSON object.",))

    size = len(canonical_json(workflow))
    if size > MAX_WORKFLOW_BYTES:
        errors.append(f"The workflow is {size} bytes, over the {MAX_WORKFLOW_BYTES} limit.")

    name = workflow.get("name")
    if not isinstance(name, str) or not name.strip():
        errors.append("The workflow has no name.")
    elif len(name) > MAX_NAME_CHARS:
        errors.append(f"The workflow name is longer than {MAX_NAME_CHARS} characters.")

    nodes = workflow.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        errors.append("The workflow has no nodes.")
        nodes = []
    elif len(nodes) > MAX_NODES:
        errors.append(f"The workflow declares {len(nodes)} nodes, over the {MAX_NODES} limit.")

    seen: set[str] = set()
    for index, node in enumerate(nodes if isinstance(nodes, list) else []):
        if not isinstance(node, dict):
            errors.append(f"Node {index} is not an object.")
            continue
        node_name = str(node.get("name") or "")
        if not node_name:
            errors.append(f"Node {index} has no name.")
        elif node_name in seen:
            errors.append(f"Two nodes are both named {node_name!r}; n8n needs them unique.")
        seen.add(node_name)
        # n8n exports reference nodes by id in some versions and by name in
        # others, so both are legitimate targets for a connection.
        if node.get("id"):
            seen.add(str(node["id"]))
        if not str(node.get("type") or ""):
            errors.append(f"Node {node_name or index} has no type.")

    connections = workflow.get("connections")
    if connections is None:
        warnings.append("The workflow has no connections block.")
    elif not isinstance(connections, dict):
        errors.append("The connections block is not an object.")
    else:
        for source, outputs in connections.items():
            if source not in seen:
                errors.append(f"Connections reference {source!r}, which is not a node in this workflow.")
            for branches in (outputs or {}).values() if isinstance(outputs, dict) else []:
                for branch in branches if isinstance(branches, list) else []:
                    for link in branch if isinstance(branch, list) else []:
                        target = str(link.get("node") or "") if isinstance(link, dict) else ""
                        if target and target not in seen:
                            errors.append(f"A connection points at {target!r}, which does not exist.")

    leaked = _SECRET_SHAPED.findall(canonical_json(workflow))
    if leaked:
        # Refused, not redacted: a literal key in a workflow is a mistake that
        # should be fixed at the source, not quietly carried into n8n.
        errors.append(f"The workflow contains {len(leaked)} literal secret-shaped value(s); "
                      "credentials belong in n8n, not in workflow JSON.")

    return WorkflowValidation(not errors, tuple(errors[:20]), tuple(warnings[:20]))


def _service_set(inspection: WorkflowInspection) -> set[str]:
    return set(inspection.services)


def diff(original: dict[str, Any], adapted: dict[str, Any],
         before: WorkflowInspection, after: WorkflowInspection) -> WorkflowDiff:
    """What actually changed, described the way a reviewer would ask."""
    def names(workflow: dict[str, Any]) -> dict[str, str]:
        nodes = workflow.get("nodes") if isinstance(workflow.get("nodes"), list) else []
        return {str(n.get("name") or ""): canonical_json(n) for n in nodes if isinstance(n, dict)}

    old, new = names(original), names(adapted)
    added = tuple(sorted(set(new) - set(old)))
    removed = tuple(sorted(set(old) - set(new)))
    changed = tuple(sorted(name for name in set(old) & set(new) if old[name] != new[name]))

    old_services, new_services = _service_set(before), _service_set(after)
    old_creds = {item.credential_type for item in before.credentials}
    new_creds = {item.credential_type for item in after.credentials}

    trigger_changed = ""
    if set(before.triggers) != set(after.triggers):
        trigger_changed = f"{', '.join(before.triggers) or 'none'} -> {', '.join(after.triggers) or 'none'}"
    risk_changed = ""
    if before.risk.level is not after.risk.level:
        risk_changed = f"{before.risk.level.value} -> {after.risk.level.value}"

    summary: list[str] = []
    for name in added:
        summary.append(f"Added node {name}")
    for name in removed:
        summary.append(f"Removed node {name}")
    for name in changed:
        summary.append(f"Changed node {name}")
    for service in sorted(new_services - old_services):
        summary.append(f"Now uses {service}")
    for service in sorted(old_services - new_services):
        summary.append(f"No longer uses {service}")
    for credential in sorted(new_creds - old_creds):
        summary.append(f"Requires a {credential} credential")
    if trigger_changed:
        summary.append(f"Trigger: {trigger_changed}")
    if risk_changed:
        summary.append(f"Risk: {risk_changed}")
    if not summary:
        summary.append("No structural change.")

    return WorkflowDiff(
        nodes_added=added, nodes_removed=removed, nodes_changed=changed,
        services_added=tuple(sorted(new_services - old_services)),
        services_removed=tuple(sorted(old_services - new_services)),
        credentials_added=tuple(sorted(new_creds - old_creds)),
        credentials_removed=tuple(sorted(old_creds - new_creds)),
        trigger_changed=trigger_changed, risk_changed=risk_changed,
        summary=tuple(summary[:40]),
    )


def strip_foreign_credentials(workflow: dict[str, Any]) -> dict[str, Any]:
    """Drop credential IDs that belong to someone else's n8n.

    An ID from the exporting instance is meaningless here at best. At worst it
    silently matches a real local credential nobody chose to attach, so the
    reference is reduced to its type and the mapping is left to be made
    explicitly.
    """
    import copy

    cleaned = copy.deepcopy(workflow)
    for node in cleaned.get("nodes", []) if isinstance(cleaned.get("nodes"), list) else []:
        if not isinstance(node, dict) or not isinstance(node.get("credentials"), dict):
            continue
        for credential_type, reference in list(node["credentials"].items()):
            if isinstance(reference, dict):
                node["credentials"][credential_type] = {
                    key: value for key, value in reference.items() if key not in {"id"}
                }
    return cleaned


def apply_credential_mapping(
    workflow: dict[str, Any], mapping: dict[str, str], available: list[dict[str, str]] | None = None,
) -> tuple[dict[str, Any], tuple[CredentialRequirement, ...]]:
    """Attach the local credentials an operator chose, and report what is missing."""
    import copy

    known = {item["id"]: item for item in (available or []) if item.get("id")}
    mapped = copy.deepcopy(workflow)
    requirements: dict[str, CredentialRequirement] = {}
    for node in mapped.get("nodes", []) if isinstance(mapped.get("nodes"), list) else []:
        if not isinstance(node, dict) or not isinstance(node.get("credentials"), dict):
            continue
        for credential_type, reference in list(node["credentials"].items()):
            chosen = mapping.get(str(credential_type), "")
            if chosen and known and chosen not in known:
                raise WorkflowError(
                    f"No credential {chosen!r} exists in the configured n8n instance.",
                    WorkflowErrorCode.VALIDATION_FAILED)
            if chosen:
                node["credentials"][credential_type] = {
                    "id": chosen, "name": known.get(chosen, {}).get("name", chosen),
                }
            existing = requirements.get(str(credential_type))
            requirements[str(credential_type)] = CredentialRequirement(
                credential_type=str(credential_type),
                node_names=tuple(sorted(set((existing.node_names if existing else ()) + (str(node.get("name") or ""),)))),
                foreign_name=str(reference.get("name") or "") if isinstance(reference, dict) else "",
                mapped_id=chosen,
            )
    return mapped, tuple(requirements[key] for key in sorted(requirements))


def sanitize_for_model(workflow: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Blank out anything credential-shaped before a workflow reaches a model.

    A user's own workflow can hold tokens, internal hostnames and private
    constants. Adaptation is worth doing, but not at the price of posting those
    to a remote provider.
    """
    text = canonical_json(workflow)
    cleaned, count = redact_secrets(text)
    cleaned = _SECRET_SHAPED.sub("[REDACTED]", cleaned)
    import json

    try:
        return json.loads(cleaned), count
    except ValueError:
        # Redaction broke the JSON, so send nothing rather than something wrong.
        raise WorkflowError("The workflow could not be safely redacted for a model.",
                            WorkflowErrorCode.VALIDATION_FAILED) from None


def approval_fingerprint(*, workflow_sha: str, operation: str, target: str, inputs: Any = None) -> str:
    """What an approval is actually for.

    Binding the hash alone would let an approved import become an approved
    activation, or an approval for one instance authorise another. All four
    travel together, so changing any one of them needs a new decision.
    """
    payload = canonical_json({
        "workflow_sha256": workflow_sha, "operation": operation,
        "target": target, "inputs": inputs if inputs is not None else {},
    })
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prepare(
    workflow: dict[str, Any], provenance: WorkflowProvenance, *,
    original: dict[str, Any] | None = None,
    credential_mapping: dict[str, str] | None = None,
    available_credentials: list[dict[str, str]] | None = None,
) -> WorkflowArtifact:
    """Build the artifact SAM would import, with everything a reviewer needs.

    Runs after any model involvement, never instead of it: the inspection and
    validation here describe the bytes that would actually be sent.
    """
    candidate = strip_foreign_credentials(workflow)
    candidate, requirements = apply_credential_mapping(
        candidate, credential_mapping or {}, available_credentials)

    inspection = inspect(candidate)
    validation = validate(candidate)
    change: WorkflowDiff | None = None
    if original is not None:
        change = diff(original, candidate, inspect(original), inspection)

    notes: list[str] = []
    unresolved = [item for item in requirements if not item.resolved]
    if unresolved:
        notes.append(
            "Unmapped credentials: " + ", ".join(item.credential_type for item in unresolved)
            + ". n8n keeps credential values; map each to one that already exists there.")
    if inspection.risk.level.rank >= RiskLevel.HIGH.rank:
        notes.append(f"Risk {inspection.risk.level.value}: this needs a deliberate decision, not a glance.")
    if inspection.risk.incomplete:
        notes.append("A referenced subworkflow could not be read, so the risk shown is a floor.")

    return WorkflowArtifact(
        workflow=candidate,
        provenance=WorkflowProvenance(
            **{**provenance.as_dict(), "adapted_sha256": workflow_sha256(candidate)}),
        inspection=inspection, validation=validation, diff=change,
        credentials=requirements, notes=tuple(notes),
    )
