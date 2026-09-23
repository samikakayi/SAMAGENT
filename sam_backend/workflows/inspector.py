"""Read a workflow without running any of it.

Everything here is static. A workflow may carry JavaScript, Python, shell
commands and expressions; all of it is treated as text. Nothing is evaluated,
no URL is fetched, no command is run -- understanding what a workflow *would*
do must not involve doing it.

The risk engine is deterministic for the same reason the inspector is: a model
asked "is this dangerous?" can be talked out of its answer by the workflow it
is reading. Node types are matched against a table, and a node the table does
not know is UNKNOWN_NODE -- never assumed safe, because the whole point of an
unknown node is that its behaviour is unknown.
"""

from __future__ import annotations

import re
from typing import Any

from .models import (
    CredentialRequirement,
    NodeFinding,
    RiskFlag,
    RiskLevel,
    WorkflowInspection,
    WorkflowRiskAssessment,
)

# Bounds, so a pathological file cannot exhaust memory or the model's context.
MAX_NODES = 400
MAX_CODE_PREVIEW_CHARS = 600
MAX_LISTED = 40

_PREFIX = "n8n-nodes-base."
_LANGCHAIN = "@n8n/n8n-nodes-langchain."

# What a node type means, by family. Kept as data because the interesting
# question -- "does this write to the outside world?" -- is a property of the
# node, not something to re-derive with cleverness each time.
READ_ONLY_NODES = {
    "set", "code.read", "merge", "if", "switch", "filter", "itemlists", "splitinbatches",
    "sort", "limit", "aggregate", "summarize", "datetime", "renamekeys", "splitout",
    "noop", "stickynote", "html", "xml", "markdown", "crypto", "editimage",
    "converttofile", "extractfromfile", "compression", "removeduplicates",
    "stopanderror", "wait", "executiondata", "n8ntrainingcustomerdatastore",
}
TRIGGER_NODES = {
    "manualtrigger", "scheduletrigger", "cron", "interval", "webhook", "errortrigger",
    "executeworkflowtrigger", "emailreadimap", "formtrigger", "n8ntrigger", "workflowtrigger",
    "localfiletrigger", "sseTrigger".lower(), "rabbitmqtrigger", "mqtttrigger", "kafkatrigger",
}
# Sending a message is an external write, whatever the product is called.
MESSAGING_NODES = {
    "gmail", "emailsend", "sendemail", "telegram", "slack", "discord", "whatsapp", "twilio",
    "mattermost", "rocketchat", "pushover", "pushbullet", "signl4", "msteams", "microsoftteams",
    "matrix", "vonage", "messagebird", "plivo", "gotify", "line", "webex", "ciscowebex",
}
DATABASE_NODES = {
    "postgres", "mysql", "mongodb", "redis", "microsoftsql", "snowflake", "questdb",
    "timescaledb", "cratedb", "elasticsearch", "supabase", "clickhouse", "cassandra",
}
FILESYSTEM_NODES = {"readwritefile", "readbinaryfile", "writebinaryfile", "filemaker", "ftp", "ssh"}
CODE_NODES = {"code", "function", "functionitem"}
SHELL_NODES = {"executecommand"}
SUBWORKFLOW_NODES = {"executeworkflow", "toolworkflow"}
# Money moves here. Treated as high-impact regardless of the operation named,
# because SAM's own trading safety must not be reachable around the side.
FINANCIAL_NODES = {
    "stripe", "paypal", "coinbase", "binance", "kraken", "brex", "chargebee", "quickbooks",
    "xero", "wise", "plaid", "square", "wooCommerce".lower(), "shopify", "invoiceninja",
    "metatrader", "alpaca", "interactivebrokers", "oanda", "bitstamp", "coingecko",
}
# Writing to someone's account or storage.
EXTERNAL_WRITE_NODES = {
    "googlesheets", "googledrive", "googledocs", "googlecalendar", "airtable", "notion",
    "hubspot", "salesforce", "pipedrive", "zendesk", "jira", "github", "gitlab", "trello",
    "asana", "clickup", "monday", "todoist", "dropbox", "box", "s3", "awss3", "nextcloud",
    "wordpress", "webflow", "strapi", "contentful", "baserow", "nocodb", "seatable",
}
NETWORK_READ_NODES = {"httprequest", "graphql", "rss", "rssfeedread", "webhookresponse", "respondtowebhook"}
MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

_EXPRESSION = re.compile(r"\{\{.*?\}\}", re.DOTALL)


def _short_type(node_type: str) -> str:
    """`n8n-nodes-base.httpRequest` -> `httprequest`."""
    text = str(node_type or "")
    for prefix in (_PREFIX, _LANGCHAIN):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text.split(".")[-1].strip().lower()


def _is_trigger(short: str) -> bool:
    """n8n names every trigger the same way, so the suffix is the rule.

    A table of trigger nodes goes out of date every release -- `gmailTrigger`
    was missing from one and a perfectly ordinary Gmail workflow came back
    UNKNOWN_NODE, which inflates the level and teaches a reviewer to ignore it.
    """
    return short in TRIGGER_NODES or short.endswith("trigger")


def _is_known(node_type: str) -> bool:
    """Whether this node comes from a namespace whose behaviour SAM can reason about."""
    text = str(node_type or "")
    return text.startswith(_PREFIX) or text.startswith(_LANGCHAIN)


def _classify(node: dict[str, Any]) -> NodeFinding:
    """One node's meaning, from its type and -- where safe -- its parameters."""
    node_type = str(node.get("type") or "")
    short = _short_type(node_type)
    name = str(node.get("name") or short or "unnamed")
    disabled = bool(node.get("disabled"))
    parameters = node.get("parameters") if isinstance(node.get("parameters"), dict) else {}
    flags: list[RiskFlag] = []
    detail = ""
    category = "other"

    if not _is_known(node_type):
        # A community or unrecognised node could do anything. Unknown is not safe.
        return NodeFinding(name, node_type, "unknown", (RiskFlag.UNKNOWN_NODE,),
                           "Not a bundled n8n node; its behaviour cannot be determined statically.", disabled)

    if short in SHELL_NODES:
        category, flags = "shell", [RiskFlag.SHELL_EXECUTION, RiskFlag.HIGH_IMPACT]
        detail = "Runs a shell command on the n8n host."
    elif short in CODE_NODES:
        category, flags = "code", [RiskFlag.CODE_EXECUTION]
        detail = f"Runs {parameters.get('language') or 'JavaScript'} inside n8n."
    elif short in SUBWORKFLOW_NODES:
        category, flags = "subworkflow", [RiskFlag.SUBWORKFLOW_EXECUTION]
        detail = "Executes another workflow, whose behaviour is not in this file."
    elif short in FINANCIAL_NODES:
        category, flags = "financial", [RiskFlag.FINANCIAL_ACTION, RiskFlag.EXTERNAL_WRITE, RiskFlag.HIGH_IMPACT]
        detail = "Touches money, an exchange, or a payment account."
    elif short in MESSAGING_NODES:
        category, flags = "messaging", [RiskFlag.EXTERNAL_WRITE]
        detail = "Sends a message to a real recipient."
    elif short in DATABASE_NODES:
        category, flags = "database", [RiskFlag.DATABASE_WRITE, RiskFlag.EXTERNAL_WRITE]
        detail = "Connects to a database; a write cannot be ruled out statically."
    elif short in FILESYSTEM_NODES:
        category, flags = "filesystem", [RiskFlag.FILESYSTEM_WRITE]
        detail = "Reads or writes files on the n8n host or a remote file service."
    elif short in {"webhook"}:
        category, flags = "trigger", [RiskFlag.WEBHOOK_EXPOSURE]
        detail = "Activating this publishes an endpoint that anyone who learns the URL can call."
    elif short in NETWORK_READ_NODES:
        category = "http"
        method = str(parameters.get("method") or "GET").upper()
        url = str(parameters.get("url") or "")
        dynamic = bool(_EXPRESSION.search(url) or _EXPRESSION.search(str(parameters.get("method") or "")))
        if dynamic:
            # An expression decides the target at run time, so neither the
            # destination nor the verb is knowable now.
            flags = [RiskFlag.NETWORK_READ, RiskFlag.EXTERNAL_WRITE]
            detail = "URL or method is an expression, so the destination is decided at run time."
        elif method in MUTATING_METHODS:
            flags = [RiskFlag.EXTERNAL_WRITE]
            detail = f"{method} request: this changes something on the far side."
        else:
            flags = [RiskFlag.NETWORK_READ]
            detail = f"{method} request."
    elif short in EXTERNAL_WRITE_NODES:
        category, flags = "service", [RiskFlag.EXTERNAL_WRITE]
        detail = "Writes into an external account or document store."
    elif _is_trigger(short):
        category, flags = "trigger", []
        detail = "Starts the workflow."
    elif short in READ_ONLY_NODES:
        category, flags = "transform", [RiskFlag.READ_ONLY]
        detail = "Transforms data in place."
    else:
        # A bundled node SAM has no entry for. Known namespace, unknown effect:
        # still not assumed safe.
        category, flags = "unclassified", [RiskFlag.UNKNOWN_NODE]
        detail = "A bundled node SAM has no classification for; treated as unknown."

    if node.get("credentials"):
        flags.append(RiskFlag.CREDENTIAL_SENSITIVE)
    return NodeFinding(name, node_type, category, tuple(dict.fromkeys(flags)), detail, disabled)


def _credentials(nodes: list[dict[str, Any]]) -> tuple[CredentialRequirement, ...]:
    """Which credential types the workflow needs, and what the exporter called them."""
    found: dict[str, dict[str, Any]] = {}
    for node in nodes:
        block = node.get("credentials")
        if not isinstance(block, dict):
            continue
        for credential_type, reference in block.items():
            entry = found.setdefault(str(credential_type), {"nodes": [], "id": "", "name": ""})
            entry["nodes"].append(str(node.get("name") or "unnamed"))
            if isinstance(reference, dict):
                entry["id"] = entry["id"] or str(reference.get("id") or "")
                entry["name"] = entry["name"] or str(reference.get("name") or "")
            elif isinstance(reference, str):
                entry["name"] = entry["name"] or reference
    return tuple(
        CredentialRequirement(
            credential_type=credential_type,
            node_names=tuple(entry["nodes"][:MAX_LISTED]),
            foreign_id=entry["id"],
            foreign_name=entry["name"],
        )
        for credential_type, entry in sorted(found.items())
    )


def _count_expressions(value: Any, budget: int = 5000) -> int:
    """How much of this workflow is decided at run time rather than now."""
    if budget <= 0:
        return 0
    if isinstance(value, str):
        return len(_EXPRESSION.findall(value))
    if isinstance(value, dict):
        total = 0
        for item in value.values():
            total += _count_expressions(item, budget - total)
        return total
    if isinstance(value, list):
        total = 0
        for item in value:
            total += _count_expressions(item, budget - total)
        return total
    return 0


def _disconnected(nodes: list[dict[str, Any]], connections: dict[str, Any]) -> tuple[str, ...]:
    """Nodes nothing reaches and which reach nothing -- often leftovers."""
    # Real exports key connections by node id as often as by name.
    named = {str(node.get("name") or "") for node in nodes}
    named |= {str(node.get("id") or "") for node in nodes if node.get("id")}
    linked: set[str] = set()
    for source, outputs in (connections or {}).items():
        if not isinstance(outputs, dict):
            continue
        linked.add(str(source))
        for branches in outputs.values():
            for branch in branches if isinstance(branches, list) else []:
                for link in branch if isinstance(branch, list) else []:
                    if isinstance(link, dict) and link.get("node"):
                        linked.add(str(link["node"]))
    by_id = {str(node.get("id") or ""): str(node.get("name") or "") for node in nodes if node.get("id")}
    unlinked = {by_id.get(name, name) for name in named if name and name not in linked}
    connected_names = {by_id.get(item, item) for item in linked}
    return tuple(sorted(name for name in unlinked if name and name not in connected_names))


def _level(flags: set[RiskFlag], incomplete: bool) -> RiskLevel:
    """One overall answer, erring upward.

    Shell and money are critical because getting them wrong is not recoverable
    by apologising. Unknown sits at high rather than critical: it means SAM
    cannot tell, which warrants a human, not alarm.
    """
    if flags & {RiskFlag.SHELL_EXECUTION, RiskFlag.FINANCIAL_ACTION}:
        return RiskLevel.CRITICAL
    if flags & {RiskFlag.CODE_EXECUTION, RiskFlag.UNKNOWN_NODE, RiskFlag.DATABASE_WRITE,
                RiskFlag.FILESYSTEM_WRITE, RiskFlag.SUBWORKFLOW_EXECUTION} or incomplete:
        return RiskLevel.HIGH
    if flags & {RiskFlag.EXTERNAL_WRITE, RiskFlag.WEBHOOK_EXPOSURE}:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def assess(findings: list[NodeFinding], *, incomplete: bool = False) -> WorkflowRiskAssessment:
    """Combine per-node findings into one level, ignoring disabled nodes."""
    flags: set[RiskFlag] = set()
    reasons: list[str] = []
    for finding in findings:
        if finding.disabled:
            continue
        for flag in finding.flags:
            if flag is RiskFlag.READ_ONLY:
                continue
            if flag not in flags:
                reasons.append(f"{finding.name}: {finding.detail}")
            flags.add(flag)
    if not flags:
        flags.add(RiskFlag.READ_ONLY)
        reasons.append("Every node only transforms data already in the workflow.")
    if incomplete:
        reasons.append("A referenced subworkflow could not be read, so this is a floor, not a verdict.")
    return WorkflowRiskAssessment(
        level=_level(flags, incomplete),
        flags=tuple(sorted(flags, key=lambda item: item.value)),
        reasons=tuple(reasons[:MAX_LISTED]),
        incomplete=incomplete,
    )


def inspect(workflow: dict[str, Any], *, subworkflows_resolved: bool = True) -> WorkflowInspection:
    """Everything SAM can say about a workflow without executing any of it."""
    raw_nodes = workflow.get("nodes")
    nodes = [node for node in raw_nodes if isinstance(node, dict)] if isinstance(raw_nodes, list) else []
    if len(nodes) > MAX_NODES:
        nodes = nodes[:MAX_NODES]
    connections = workflow.get("connections") if isinstance(workflow.get("connections"), dict) else {}

    findings = [_classify(node) for node in nodes]
    triggers = tuple(sorted({
        finding.node_type for finding, node in zip(findings, nodes)
        if _is_trigger(_short_type(node.get("type") or "")) or finding.category == "trigger"
    }))
    services = tuple(sorted({
        _short_type(node.get("type") or "") for node in nodes
        if _short_type(node.get("type") or "") not in READ_ONLY_NODES
    }))[:MAX_LISTED]

    destinations: list[str] = []
    previews: list[dict[str, str]] = []
    subworkflows: list[str] = []
    for node, finding in zip(nodes, findings):
        parameters = node.get("parameters") if isinstance(node.get("parameters"), dict) else {}
        if finding.category == "http" and parameters.get("url"):
            destinations.append(str(parameters["url"])[:300])
        if finding.category == "code":
            # Shown, never run: a reviewer must be able to read what would execute.
            code = str(parameters.get("jsCode") or parameters.get("pythonCode") or parameters.get("functionCode") or "")
            previews.append({
                "node": finding.name,
                "language": str(parameters.get("language") or "javaScript"),
                "code": code[:MAX_CODE_PREVIEW_CHARS],
                "truncated": "true" if len(code) > MAX_CODE_PREVIEW_CHARS else "false",
            })
        if finding.category == "subworkflow":
            reference = parameters.get("workflowId")
            if isinstance(reference, dict):
                reference = reference.get("value")
            subworkflows.append(str(reference or "unspecified"))

    incomplete = bool(subworkflows) and not subworkflows_resolved
    notes: list[str] = []
    if len(raw_nodes or []) > MAX_NODES:
        notes.append(f"Only the first {MAX_NODES} nodes were inspected; the file declares {len(raw_nodes)}.")

    return WorkflowInspection(
        node_count=len(nodes),
        node_types=tuple(sorted({str(node.get("type") or "") for node in nodes}))[:MAX_LISTED],
        triggers=triggers,
        services=services,
        nodes=tuple(findings[:MAX_LISTED]),
        credentials=_credentials(nodes),
        http_destinations=tuple(dict.fromkeys(destinations))[:MAX_LISTED],
        expressions=_count_expressions(workflow),
        disabled_nodes=tuple(finding.name for finding in findings if finding.disabled)[:MAX_LISTED],
        disconnected_nodes=_disconnected(nodes, connections)[:MAX_LISTED],
        subworkflows=tuple(dict.fromkeys(subworkflows))[:MAX_LISTED],
        risk=assess(findings, incomplete=incomplete),
        code_previews=tuple(previews[:10]),
        notes=tuple(notes),
    )


# What activation would actually do, per trigger family. The manual trigger is
# the interesting one: n8n refuses to activate a workflow that has only that,
# and the public API has no endpoint to run one on demand, so a workflow can be
# perfectly valid, imported, and still have no supported way to start. Saying
# so before the attempt is the difference between an explanation and an opaque
# HTTP 400 from n8n's own validator.
_MANUAL_TRIGGERS = frozenset({"manualtrigger", "executeworkflowtrigger"})
_WEBHOOK_TRIGGERS = frozenset({"webhook", "chattrigger", "formtrigger"})
_SCHEDULE_TRIGGERS = frozenset({"scheduletrigger", "cron", "interval"})


def activation(inspection: WorkflowInspection) -> dict[str, Any]:
    """Whether this workflow can be activated, and what happens if it is."""
    shorts = {_short_type(trigger) for trigger in inspection.triggers}
    if not shorts:
        return {
            "can_activate": False, "kind": "none",
            "reason": "This workflow has no trigger node, so n8n has nothing to activate.",
            "manual_run_supported": False,
        }
    if shorts <= _MANUAL_TRIGGERS:
        return {
            "can_activate": False, "kind": "manual",
            "reason": ("This workflow only starts manually. n8n refuses to activate a manual-only "
                       "workflow, and its public API has no endpoint for running one on demand, so "
                       "SAM cannot start it either -- open it in the n8n editor and press Execute."),
            "manual_run_supported": False,
        }
    if shorts & _WEBHOOK_TRIGGERS:
        return {
            "can_activate": True, "kind": "webhook",
            "reason": ("Activating publishes a webhook URL on the n8n instance. Anything that can "
                       "reach that URL can start this workflow."),
            "manual_run_supported": False,
        }
    if shorts & _SCHEDULE_TRIGGERS:
        return {
            "can_activate": True, "kind": "schedule",
            "reason": "Activating starts a schedule; the workflow then runs unattended until deactivated.",
            "manual_run_supported": False,
        }
    return {
        "can_activate": True, "kind": "polling",
        "reason": ("Activating starts a trigger that polls or listens against a third-party service, "
                   "using whichever credential is attached to it."),
        "manual_run_supported": False,
    }
