"""From "build me an automation that..." to something worth approving.

The primitives already existed -- search, inspect, prepare, approve, import --
but a user had to drive them one at a time and know which to call next. This
is the missing operation between them, and it is deliberately *planning* only:
it reads the library, chooses, adapts, validates and hashes, and then stops.
Every way to change the outside world stays behind the approval-bound tools it
already lived behind.

Two rules shape everything here.

Search first, generate second: a workflow somebody already runs has been
debugged by reality, and adapting one is cheaper and safer than inventing one.
Generation is the fallback, not the reflex.

Generated is not trusted. A workflow SAM wrote and a workflow a stranger wrote
go through the identical inspect -> validate -> risk -> credential -> hash
pipeline, because the author of a workflow is not evidence about what it does.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from inspect import signature
from typing import Any

from .adaptation import AdaptationRejected, AdaptationUnavailable, adaptation_needed
from .inspector import activation, inspect
from .models import (
    RiskFlag,
    RiskLevel,
    WorkflowArtifact,
    WorkflowError,
    WorkflowErrorCode,
    WorkflowInspection,
    WorkflowProvenance,
    WorkflowSummary,
)
from .service import prepare, sanitize_for_model

# How much of the library reaches a decision, and how little reaches a model.
# The library holds thousands of workflows; shortlisting is what keeps the
# context bounded no matter how big it grows.
SHORTLIST = 5
INSPECT_DEPTH = 3

# Below this, the best candidate is not actually about the goal, and adapting
# it would be worse than writing two honest nodes.
MINIMUM_USEFUL_SCORE = 3

# Scoring weights. Deliberately small integers with named reasons rather than a
# confidence percentage: a reviewer can check "matched gmail, matched drive"
# and cannot check "87%".
SERVICE_MATCH = 3
TRIGGER_MATCH = 2
CREDENTIAL_COST = 1
COMPLEXITY_FREE_NODES = 8
RISK_PENALTY = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 2, RiskLevel.HIGH: 6, RiskLevel.CRITICAL: 12}

_STOPWORDS = frozenset({
    "a", "an", "and", "are", "automation", "build", "can", "create", "every",
    "for", "from", "get", "goal", "i", "in", "into", "is", "it", "make", "me",
    "my", "new", "of", "on", "or", "please", "sam", "set", "so", "start",
    "stores", "that", "the", "then", "to", "want", "when", "which", "with",
    "workflow", "would",
})

# Words in a goal that name how it should start.
_TRIGGER_WORDS: dict[str, tuple[str, ...]] = {
    "scheduled": ("schedule", "scheduled", "daily", "hourly", "weekly", "cron", "every"),
    "webhook": ("webhook", "http", "endpoint", "callback"),
    "manual": ("manual", "manually", "button", "on demand"),
    "triggered": ("when", "whenever", "watches", "watch", "incoming", "arrives"),
}

_FIELD_ASSIGNMENT = re.compile(r"([A-Za-z_][A-Za-z0-9_]{0,60})\s*=\s*([^\s,;]{1,120})")


@dataclass(frozen=True, slots=True)
class GoalReading:
    """What SAM believes was asked for, stated so it can be contradicted."""

    goal: str
    keywords: tuple[str, ...]
    services: tuple[str, ...]
    trigger: str
    fields: tuple[tuple[str, str], ...] = ()
    ambiguous: bool = False
    clarification: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal, "keywords": list(self.keywords), "services": list(self.services),
            "trigger": self.trigger, "fields": [list(pair) for pair in self.fields],
            "ambiguous": self.ambiguous, "clarification": self.clarification,
        }


@dataclass(slots=True)
class Candidate:
    """One library workflow, scored against the goal, with the reasons shown."""

    summary: WorkflowSummary
    score: int
    reasons: tuple[str, ...] = ()
    concerns: tuple[str, ...] = ()
    # The subset of concerns that are about fit rather than safety. Risk is
    # a reason to read carefully, not a reason to rewrite: adapting a
    # workflow does not make it less dangerous, and treating every MEDIUM
    # as a gap would send almost every real workflow to a model.
    fit_gaps: tuple[str, ...] = ()
    inspection: WorkflowInspection | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = {
            **self.summary.as_dict(), "score": self.score,
            "reasons": list(self.reasons), "concerns": list(self.concerns),
            "fit_gaps": list(self.fit_gaps),
        }
        if self.inspection is not None:
            payload["risk"] = self.inspection.risk.as_dict()
            payload["node_count"] = self.inspection.node_count
            payload["credentials"] = [item.credential_type for item in self.inspection.credentials]
        return payload


@dataclass(slots=True)
class GoalPlan:
    """Everything a person needs to decide, and nothing that acts."""

    reading: GoalReading
    candidates: list[Candidate] = field(default_factory=list)
    origin: str = "none"  # library | generated | none
    selected: Candidate | None = None
    selection_reason: str = ""
    adaptations: tuple[str, ...] = ()
    # PROPOSED | NOT_NEEDED | UNAVAILABLE | REJECTED -- what became of the
    # model step, separate from what the deterministic pipeline then found.
    adaptation: str = "NOT_NEEDED"
    adaptation_reason: str = ""
    adaptation_claims: dict[str, Any] = field(default_factory=dict)
    artifact: WorkflowArtifact | None = None
    activation: dict[str, Any] = field(default_factory=dict)
    next_action: str = ""
    blockers: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "goal": self.reading.goal,
            "understood": self.reading.as_dict(),
            "candidates": [item.as_dict() for item in self.candidates],
            "origin": self.origin,
            "selected": self.selected.as_dict() if self.selected else None,
            "selection_reason": self.selection_reason,
            "adaptations": list(self.adaptations),
            "adaptation": self.adaptation,
            "adaptation_reason": self.adaptation_reason,
            # Labelled claims, never findings: the risk and credential
            # numbers a reviewer reads come from `prepared`, not from here.
            "adaptation_claims": self.adaptation_claims,
            "activation": self.activation,
            "next_action": self.next_action,
            "blockers": list(self.blockers),
        }
        if self.artifact is not None:
            payload["prepared"] = {
                "name": self.artifact.name,
                "sha256": self.artifact.sha256,
                "inspection": self.artifact.inspection.as_dict(),
                "validation": self.artifact.validation.as_dict(),
                "diff": self.artifact.diff.as_dict() if self.artifact.diff else None,
                "credentials": [item.as_dict() for item in self.artifact.credentials],
                "unresolved_credentials": [item.as_dict() for item in self.artifact.unresolved_credentials],
                "notes": list(self.artifact.notes),
                "provenance": self.artifact.provenance.as_dict(),
            }
        return payload


# --- reading the goal ---------------------------------------------------------


def read_goal(goal: str, known_services: frozenset[str] | None = None) -> GoalReading:
    """Turn a sentence into the few facts the search and the score can use."""
    text = str(goal or "").strip()
    words = [word for word in re.split(r"[^A-Za-z0-9_]+", text.lower()) if word]
    keywords = tuple(dict.fromkeys(word for word in words if len(word) > 2 and word not in _STOPWORDS))[:10]

    lowered = text.lower()
    trigger = ""
    for name, markers in _TRIGGER_WORDS.items():
        if any(marker in lowered for marker in markers):
            trigger = name
            break

    # Only words that match a vocabulary SAM actually has are called services.
    # Returning the leading keywords here instead would put "invoices" and
    # "them" under a heading that says SAM recognised an integration, which is
    # a small lie that a reviewer would reasonably rely on.
    services = tuple(word for word in keywords if known_services and word in known_services)

    fields = tuple((match.group(1), match.group(2)) for match in _FIELD_ASSIGNMENT.finditer(text))[:10]

    # "Ambiguous" means SAM genuinely cannot act, not that it is unsure.
    ambiguous = len(keywords) < 2
    clarification = ""
    if ambiguous:
        clarification = (
            "That goal names too little to search for. Say which service or which outcome is "
            "involved -- for example 'watch Gmail for invoices and save them to Drive'.")
    return GoalReading(text, keywords, services, trigger, fields, ambiguous, clarification)


# --- scoring ------------------------------------------------------------------


def score_candidate(summary: WorkflowSummary, reading: GoalReading,
                    inspection: WorkflowInspection | None = None) -> Candidate:
    """Rank on things a reviewer can verify, never on a model's opinion."""
    reasons: list[str] = []
    concerns: list[str] = []
    gaps: list[str] = []
    score = 0

    haystack = f"{summary.title} {' '.join(summary.services)} {summary.category}".lower()
    matched = [word for word in reading.keywords if word in haystack]
    score += SERVICE_MATCH * len(matched)
    if matched:
        reasons.append("matches " + ", ".join(matched[:5]))

    if reading.trigger and summary.trigger == reading.trigger:
        score += TRIGGER_MATCH
        reasons.append(f"starts the way the goal asks ({summary.trigger})")
    elif reading.trigger and summary.trigger not in ("", "unknown"):
        concerns.append(f"starts on {summary.trigger}, the goal asked for {reading.trigger}")
        gaps.append(f"it starts on {summary.trigger} and the goal asked for {reading.trigger}")

    if inspection is not None:
        penalty = RISK_PENALTY.get(inspection.risk.level, 0)
        score -= penalty
        if penalty:
            concerns.append(f"risk {inspection.risk.level.value}")
        credentials = len(inspection.credentials)
        score -= CREDENTIAL_COST * credentials
        if credentials:
            concerns.append(f"needs {credentials} credential(s)")
        extra = max(0, inspection.node_count - COMPLEXITY_FREE_NODES)
        score -= extra
        if extra:
            concerns.append(f"{inspection.node_count} nodes to review")
        if inspection.risk.incomplete:
            concerns.append("contains a subworkflow that could not be read")
        # An unreadable node is the one concern that should stop a candidate
        # being chosen automatically, however well it matches the words.
        if RiskFlag.UNKNOWN_NODE in inspection.risk.flags:
            score -= 10
            concerns.append("contains a node SAM cannot classify")
            gaps.append("it contains a node SAM cannot classify")

    return Candidate(summary, score, tuple(reasons), tuple(concerns), tuple(gaps), inspection)


# --- generating, when nothing fits --------------------------------------------


def generate_workflow(reading: GoalReading, name: str = "") -> dict[str, Any]:
    """Build the smallest honest workflow that expresses the goal.

    Deliberately minimal and side-effect free: a manual trigger and a Set node.
    SAM will not invent a workflow that sends, writes or calls anything -- if
    the goal needs that, the right answer is a library workflow a human reviews,
    not a guess that quietly reaches the outside world.
    """
    assignments = [
        {"id": f"f{index}", "name": key[:60], "value": _coerce(value), "type": _value_type(value)}
        for index, (key, value) in enumerate(reading.fields[:10])
    ]
    if not assignments:
        assignments = [{"id": "f0", "name": "sam_goal", "value": reading.goal[:300], "type": "string"}]

    return {
        "name": (name or f"SAM: {reading.goal}")[:200],
        "nodes": [
            {
                "name": "When clicking Execute", "type": "n8n-nodes-base.manualTrigger",
                "typeVersion": 1, "position": [0, 0], "parameters": {},
            },
            {
                "name": "Set fields", "type": "n8n-nodes-base.set",
                "typeVersion": 3.4, "position": [220, 0],
                "parameters": {"assignments": {"assignments": assignments}, "options": {}},
            },
        ],
        "connections": {
            "When clicking Execute": {"main": [[{"node": "Set fields", "type": "main", "index": 0}]]},
        },
        "settings": {},
    }


def _value_type(value: str) -> str:
    lowered = str(value).strip().lower()
    if lowered in ("true", "false"):
        return "boolean"
    try:
        float(lowered)
    except ValueError:
        return "string"
    return "number"


def _coerce(value: str) -> Any:
    kind = _value_type(value)
    if kind == "boolean":
        return str(value).strip().lower() == "true"
    if kind == "number":
        number = float(value)
        return int(number) if number.is_integer() else number
    return str(value)[:300]


# --- the plan -----------------------------------------------------------------


def plan_goal(
    goal: str, intelligence: Any, *, name: str = "",
    credential_mapping: dict[str, str] | None = None,
    adapt: Any = None, customize: bool = False,
) -> GoalPlan:
    """Search, choose, adapt, validate and hash -- then stop.

    `adapt` is the seam a model plugs into: it receives a *redacted* copy of the
    candidate and may return a replacement. Whatever comes back is put through
    the same validator and risk engine as everything else, and its own account
    of what it changed is ignored in favour of the diff.
    """
    reading = read_goal(goal, _known_services(intelligence))
    plan = GoalPlan(reading=reading)
    if reading.ambiguous:
        plan.next_action = "clarify"
        plan.blockers = (reading.clarification,)
        return plan

    plan.candidates = _shortlist(reading, intelligence)
    best = plan.candidates[0] if plan.candidates else None

    original: dict[str, Any] | None = None
    if best is not None and best.score >= MINIMUM_USEFUL_SCORE:
        workflow, provenance = intelligence.library.get_workflow(best.summary.workflow_id)
        original = workflow
        candidate = dict(workflow)
        plan.origin = "library"
        plan.selected = best
        plan.selection_reason = _explain(best, plan.candidates)
    else:
        candidate = generate_workflow(reading, name)
        provenance = WorkflowProvenance(source="generated",
                                        source_repository="SAM (no library workflow fitted this goal)")
        plan.origin = "generated"
        plan.selection_reason = _explain_generation(best)

    if name:
        candidate["name"] = name[:200]

    adaptations: list[str] = []
    if adapt is not None:
        wanted, why = adaptation_needed(
            origin=plan.origin,
            score=best.score if (best is not None and plan.origin == "library") else 0,
            concerns=best.fit_gaps if (best is not None and plan.origin == "library") else (),
            requested=customize,
        )
        plan.adaptation_reason = why
        if wanted:
            candidate, adaptations, plan.adaptation = _adapt(
                candidate, reading, adapt,
                inspect(candidate) if plan.origin == "library" else None)
            claims = getattr(adapt, "last_proposal", None)
            if plan.adaptation == "PROPOSED" and claims is not None:
                plan.adaptation_claims = claims.as_dict()
        else:
            # Not an omission: declining to spend a completion is the decision.
            plan.adaptation = "NOT_NEEDED"
    else:
        plan.adaptation = "NOT_NEEDED"
        plan.adaptation_reason = "Model adaptation is not enabled for this request."

    available: list[dict[str, str]] = []
    if credential_mapping and intelligence.n8n.configured:
        try:
            available = intelligence.n8n.list_credentials()
        except WorkflowError:
            available = []

    artifact = prepare(candidate, provenance, original=original,
                       credential_mapping=credential_mapping, available_credentials=available)
    intelligence.remember_artifact(artifact)

    plan.artifact = artifact
    plan.adaptations = tuple(adaptations)
    plan.activation = activation(artifact.inspection)
    plan.next_action, plan.blockers = _next_action(artifact, intelligence)
    return plan


def _known_services(intelligence: Any) -> frozenset[str]:
    try:
        categories = intelligence.library.get_categories() or []
    except Exception:  # noqa: BLE001 - a planner must not fail on a cold cache
        return frozenset()
    return frozenset(str(item).lower() for item in categories)


def _shortlist(reading: GoalReading, intelligence: Any) -> list[Candidate]:
    """Top summaries, then a deeper look at only the few that could win."""
    try:
        found = intelligence.library.search(
            " ".join(reading.keywords), trigger=reading.trigger, limit=SHORTLIST)
    except Exception:  # noqa: BLE001 - an unreachable library means generate, not fail
        return []

    scored = [score_candidate(summary, reading) for summary in found]
    scored.sort(key=lambda item: (-item.score, item.summary.size_bytes))

    # Only the plausible few are fetched and inspected. This is what keeps a
    # library of thousands from ever becoming a context problem.
    for candidate in scored[:INSPECT_DEPTH]:
        try:
            workflow, _ = intelligence.library.get_workflow(candidate.summary.workflow_id)
        except Exception:  # noqa: BLE001 - one bad file must not sink the search
            continue
        rescored = score_candidate(candidate.summary, reading, inspect(workflow))
        candidate.score, candidate.reasons = rescored.score, rescored.reasons
        candidate.concerns, candidate.inspection = rescored.concerns, rescored.inspection
        candidate.fit_gaps = rescored.fit_gaps

    scored.sort(key=lambda item: (-item.score, item.summary.size_bytes))
    return scored


def _adapt(candidate: dict[str, Any], reading: GoalReading, adapt: Any,
           inspection: Any = None) -> tuple[dict[str, Any], list[str], str]:
    """Let something else propose a change, then forget that it did.

    The proposal is only accepted as far as "these are bytes to inspect". It is
    redacted on the way out so a workflow's own constants never reach a remote
    model, and it is re-read from scratch on the way back.
    """
    redacted, _ = sanitize_for_model(candidate)
    # The inspection is a courtesy, not a contract, so an adapter may take two
    # arguments or three. Asking the signature rather than catching TypeError
    # matters: a TypeError raised *inside* a three-argument adapter would
    # otherwise look like an arity mismatch and call it a second time.
    try:
        takes_inspection = len(signature(adapt).parameters) >= 3
    except (TypeError, ValueError):
        takes_inspection = False
    try:
        proposed = adapt(redacted, reading.goal, inspection) if takes_inspection             else adapt(redacted, reading.goal)
    except AdaptationUnavailable as exc:
        return candidate, [f"Model adaptation was unavailable, so the workflow is unchanged: {exc}"], "UNAVAILABLE"
    except AdaptationRejected as exc:
        return candidate, [f"The model's proposal was rejected ({exc}), so the workflow is unchanged."], "REJECTED"
    except Exception as exc:  # noqa: BLE001 - an adapter is third-party by nature
        return candidate, [f"An adaptation was proposed but could not be read ({type(exc).__name__}); "
                           "the original workflow is used unchanged."], "REJECTED"
    if not isinstance(proposed, dict) or not proposed.get("nodes"):
        return candidate, ["The proposed adaptation was not a workflow, so it was discarded."], "REJECTED"
    return proposed, ["A model proposed an adaptation; it was re-inspected and re-validated from "
                      "scratch, and the diff below is what actually changed."], "PROPOSED"


def _explain(best: Candidate, candidates: list[Candidate]) -> str:
    parts = [f"{best.summary.title!r} scored {best.score}"]
    if best.reasons:
        parts.append("because it " + "; ".join(best.reasons))
    runner_up = candidates[1] if len(candidates) > 1 else None
    if runner_up is not None:
        # A tie broken by size is not "scoring higher", and saying it did
        # would be inventing a distinction the scorer never made.
        parts.append(
            f"level with {runner_up.summary.title!r} on {best.score}, and chosen for being smaller"
            if runner_up.score == best.score
            else f"ahead of {runner_up.summary.title!r} on {runner_up.score}")
    if best.concerns:
        parts.append("Worth knowing: " + "; ".join(best.concerns) + ".")
    return ". ".join(parts[:3]) + ("" if parts[-1].endswith(".") else ".")


def _explain_generation(best: Candidate | None) -> str:
    if best is None:
        return ("Nothing in the library matched this goal, so SAM wrote the smallest workflow that "
                "expresses it: a manual trigger and a Set node, with no external calls.")
    return (f"The closest library workflow, {best.summary.title!r}, scored {best.score}, below the "
            f"threshold of {MINIMUM_USEFUL_SCORE}, so adapting it would have been guesswork. SAM "
            "wrote a minimal workflow instead, with no external calls.")


def _next_action(artifact: WorkflowArtifact, intelligence: Any) -> tuple[str, tuple[str, ...]]:
    """The one thing that may happen next, and what is in the way."""
    blockers: list[str] = []
    if not artifact.validation.ok:
        return "fix_validation", tuple(artifact.validation.errors[:5])
    if artifact.unresolved_credentials:
        return "map_credentials", tuple(
            f"{item.credential_type} must be mapped to a credential that already exists in n8n."
            for item in artifact.unresolved_credentials)
    if not intelligence.n8n.configured:
        blockers.append("No n8n instance is configured, so there is nowhere to import this yet.")
        return "configure_n8n", tuple(blockers)
    if artifact.inspection.risk.level.rank >= RiskLevel.HIGH.rank:
        blockers.append(
            f"Risk is {artifact.inspection.risk.level.value}; read the flags before approving.")
    # Import is the only next step, and it is still gated by an approval bound
    # to this exact hash. Planning never becomes importing on its own.
    return "request_import_approval", tuple(blockers)


def missing_intelligence() -> WorkflowError:
    return WorkflowError("Workflow Intelligence is not available.", WorkflowErrorCode.NOT_CONFIGURED)
