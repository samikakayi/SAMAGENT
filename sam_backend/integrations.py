"""Whether SAM's outside connections will still work tomorrow.

`ProviderHealth` answers "can this model drive a run". That is a different
question from "is this credential still good", and the acceptance work showed
why the second one needs its own answer: a token can expire silently, a key
can be revoked in a provider's console, and an API key can carry far more
authority than the product ever uses. None of those show up as a model
problem until something fails at the worst moment.

Two axes are reported separately because a credential can be perfectly healthy
*and* expiring next week, or working *and* overprivileged. Collapsing them into
one enum would force a choice between telling the truth about reachability and
telling the truth about hygiene.

Nothing here returns a secret. Expiry and scope are only ever reported when
they were recorded as fact; SAM never guesses when a token dies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from .config import Settings
from .secrets import SecretStore, openai_compatible_status, openrouter_status
from .workflows.n8n import REQUIRED_SCOPES as N8N_REQUIRED_SCOPES

# A dashboard is not worth a provider's rate limit. Every check here is a
# catalogue or key GET, but even those are cached: refreshing is something the
# operator asks for, not something a panel does on a timer.
DEFAULT_TTL_SECONDS = 300.0

# Where the non-secret half of a credential lives. The value itself stays in
# SecretStore; this records only what a reviewer needs in order to judge it.
METADATA_SETTING_KEY = "credential_metadata"

EXPIRING_SOON_DAYS = 14
EXPIRING_URGENT_DAYS = 7


class IntegrationState(StrEnum):
    """Reachability and authentication -- what a probe just found."""

    NOT_CONFIGURED = "NOT_CONFIGURED"
    HEALTHY = "HEALTHY"
    AUTH_ERROR = "AUTH_ERROR"
    QUOTA = "QUOTA"
    RATE_LIMITED = "RATE_LIMITED"
    UNREACHABLE = "UNREACHABLE"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    UNKNOWN = "UNKNOWN"


class ExpiryState(StrEnum):
    """What is known about when the credential stops working."""

    # Not a failure: most providers simply never say. Saying so is honest;
    # inventing a date would not be.
    UNKNOWN_EXPIRY = "UNKNOWN_EXPIRY"
    OK = "OK"
    EXPIRING_SOON = "EXPIRING_SOON"
    EXPIRED = "EXPIRED"


class PrivilegeState(StrEnum):
    """Whether the credential can do more than SAM ever asks of it."""

    UNKNOWN = "UNKNOWN"
    LEAST_PRIVILEGE = "LEAST_PRIVILEGE"
    OVERPRIVILEGED = "OVERPRIVILEGED"


# The provider vocabularies already in use, mapped into one. Keeping the
# translation here means the existing status helpers stay untouched.
_PROVIDER_STATE: dict[str, IntegrationState] = {
    "CONNECTED": IntegrationState.HEALTHY,
    "UNCONFIGURED": IntegrationState.NOT_CONFIGURED,
    "NOT_CONFIGURED": IntegrationState.NOT_CONFIGURED,
    "AUTH_FAILED": IntegrationState.AUTH_ERROR,
    "AUTH_ERROR": IntegrationState.AUTH_ERROR,
    "RATE_LIMITED": IntegrationState.RATE_LIMITED,
    "QUOTA": IntegrationState.QUOTA,
    "UNREACHABLE": IntegrationState.UNREACHABLE,
    "DOWN": IntegrationState.UNREACHABLE,
    "NO_MODELS": IntegrationState.MODEL_UNAVAILABLE,
    "INCOMPATIBLE": IntegrationState.UNKNOWN,
    "ERROR": IntegrationState.UNREACHABLE,
}


def _parse_expiry(value: Any) -> datetime | None:
    """Accept what providers actually emit: ISO strings or unix seconds.

    Including unix seconds that arrive *as a string*. n8n publishes an API
    key's expiry as an epoch integer, and any JSON schema that types the field
    as text -- as the credential-metadata schema does, so one field can hold
    either shape -- hands it over as "1797915600". Reading only the numeric
    type meant a key with a perfectly good expiry date reported
    UNKNOWN_EXPIRY, which is the one answer this module exists to avoid.
    """
    if value in (None, "", 0):
        return None
    text = str(value).strip()
    if isinstance(value, (int, float)) or text.lstrip("-").isdigit():
        try:
            return datetime.fromtimestamp(float(text), tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


@dataclass(slots=True)
class CredentialFacts:
    """The non-secret half of a credential: enough to judge, never to use."""

    configured: bool = False
    fingerprint: str = ""
    expires_at: str = ""
    days_until_expiry: int | None = None
    expiry: ExpiryState = ExpiryState.UNKNOWN_EXPIRY
    privilege: PrivilegeState = PrivilegeState.UNKNOWN
    scopes: tuple[str, ...] = ()
    scope_summary: str = ""
    extra_scopes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "configured": self.configured, "fingerprint": self.fingerprint,
            "expires_at": self.expires_at, "days_until_expiry": self.days_until_expiry,
            "expiry": self.expiry.value, "privilege": self.privilege.value,
            "scopes": list(self.scopes), "scope_summary": self.scope_summary,
            "extra_scopes": list(self.extra_scopes),
        }


@dataclass(slots=True)
class IntegrationReport:
    """One integration, as a panel should show it."""

    name: str
    state: IntegrationState
    detail: str = ""
    latency_ms: float | None = None
    checked_at: float = field(default_factory=time.time)
    credential: CredentialFacts = field(default_factory=CredentialFacts)
    warnings: tuple[str, ...] = ()
    recommended_action: str = ""

    @property
    def needs_attention(self) -> bool:
        return bool(self.warnings) or self.state not in (
            IntegrationState.HEALTHY, IntegrationState.NOT_CONFIGURED)

    def as_dict(self) -> dict[str, Any]:
        return {
            "integration": self.name, "state": self.state.value, "detail": self.detail,
            "latency_ms": self.latency_ms,
            "checked_at": datetime.fromtimestamp(self.checked_at, tz=UTC).isoformat(),
            "credential": self.credential.as_dict(),
            "warnings": list(self.warnings), "recommended_action": self.recommended_action,
            "needs_attention": self.needs_attention,
        }


def describe_expiry(expires_at: Any, *, now: datetime | None = None) -> tuple[ExpiryState, str, int | None]:
    """Turn a recorded expiry into a state, a date and a countdown."""
    parsed = _parse_expiry(expires_at)
    if parsed is None:
        return ExpiryState.UNKNOWN_EXPIRY, "", None
    moment = now or datetime.now(tz=UTC)
    days = (parsed - moment).days
    if parsed <= moment:
        return ExpiryState.EXPIRED, parsed.date().isoformat(), days
    state = ExpiryState.EXPIRING_SOON if days <= EXPIRING_SOON_DAYS else ExpiryState.OK
    return state, parsed.date().isoformat(), days


def describe_privilege(scopes: Any, required: frozenset[str]) -> tuple[PrivilegeState, tuple[str, ...], str]:
    """Compare what a credential may do against what SAM ever asks it to.

    Anything beyond the required set is authority nobody is using, which is
    exactly the kind of thing that is only noticed after it is abused.
    """
    if not scopes:
        return PrivilegeState.UNKNOWN, (), "No scope list was recorded for this credential."
    granted = {str(scope) for scope in scopes if str(scope)}
    extra = tuple(sorted(granted - required))
    if extra:
        return (PrivilegeState.OVERPRIVILEGED, extra,
                f"{len(granted)} scopes granted; {len(extra)} beyond the {len(required)} SAM uses.")
    return (PrivilegeState.LEAST_PRIVILEGE, (),
            f"{len(granted)} scopes granted, all of them used by SAM.")


def read_metadata(database: Any) -> dict[str, dict[str, Any]]:
    """Recorded, non-secret credential metadata, keyed by credential name."""
    try:
        stored = database.get_settings().get(METADATA_SETTING_KEY) or {}
    except Exception:  # noqa: BLE001 - a health panel must not fail on storage
        return {}
    return stored if isinstance(stored, dict) else {}


def record_metadata(database: Any, name: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Remember scope/expiry facts for a credential SAM did not issue.

    n8n's public API exposes no endpoint for reading an API key's own scopes
    or expiry, so the only honest moment to learn them is when the operator
    stores the key. Only the three non-secret fields are kept -- anything else
    offered is dropped rather than trusted.
    """
    allowed = {
        "scopes": [str(scope)[:100] for scope in (metadata.get("scopes") or [])][:200],
        "created_at": str(metadata.get("created_at") or "")[:40],
        "expires_at": str(metadata.get("expires_at") or "")[:40],
    }
    current = read_metadata(database)
    current[name] = {key: value for key, value in allowed.items() if value}
    database.update_settings({METADATA_SETTING_KEY: current})
    return current[name]


class IntegrationHealth:
    """Cached, cheap answers about every outside connection SAM depends on."""

    def __init__(self, settings: Settings, secrets: SecretStore, database: Any,
                 workflows: Any = None, *, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> None:
        self.settings = settings
        self.secrets = secrets
        self.database = database
        self.workflows = workflows
        self.ttl_seconds = ttl_seconds
        self._cache: tuple[float, list[IntegrationReport]] | None = None

    def invalidate(self) -> None:
        self._cache = None

    # -- per-integration checks -------------------------------------------
    async def _openrouter(self) -> IntegrationReport:
        raw = await openrouter_status(
            self.settings.openrouter_base_url, self.settings.openrouter_api_key,
            {"X-Title": self.settings.openrouter_title})
        report = self._from_provider("openrouter", raw)
        if raw.get("is_free_tier") and not raw.get("limit_remaining"):
            # Not an error yet, but every paid model will fail on this key.
            report.warnings = (*report.warnings,
                               "This is a free-tier key with no paid credit; billed models will fail.")
            report.recommended_action = report.recommended_action or (
                "Add credit, or keep routing on the FREE or BALANCED profile.")
        return report

    async def _openai_compatible(self, name: str, base_url: str, key: str | None) -> IntegrationReport:
        return self._from_provider(name, await openai_compatible_status(name, base_url, key))

    def _from_provider(self, name: str, raw: dict[str, Any]) -> IntegrationReport:
        state = _PROVIDER_STATE.get(str(raw.get("status") or ""), IntegrationState.UNKNOWN)
        report = IntegrationReport(
            name=name, state=state, detail=str(raw.get("detail") or ""),
            latency_ms=raw.get("latency_ms"),
            credential=self._credential_facts(f"{name}_api_key"),
        )
        return self._advise(report)

    def _n8n(self) -> IntegrationReport:
        """n8n answers synchronously; the caller runs this off the loop."""
        client = getattr(self.workflows, "n8n", None)
        raw = client.status() if client is not None else {"status": "NOT_CONFIGURED"}
        state = _PROVIDER_STATE.get(str(raw.get("status") or ""), IntegrationState.UNKNOWN)
        report = IntegrationReport(
            name="n8n", state=state, detail=str(raw.get("detail") or ""),
            credential=self._credential_facts("n8n_api_key", required=N8N_REQUIRED_SCOPES),
        )
        return self._advise(report)

    # -- credential facts ---------------------------------------------------
    def _credential_facts(self, name: str, *, required: frozenset[str] | None = None) -> CredentialFacts:
        try:
            status = self.secrets.public_status().get(name) or {}
        except Exception:  # noqa: BLE001
            status = {}
        facts = CredentialFacts(
            configured=bool(status.get("configured")),
            fingerprint=str(status.get("fingerprint") or ""),
        )
        metadata = read_metadata(self.database).get(name) or {}
        facts.expiry, facts.expires_at, facts.days_until_expiry = describe_expiry(metadata.get("expires_at"))
        if required is not None:
            facts.privilege, facts.extra_scopes, facts.scope_summary = describe_privilege(
                metadata.get("scopes"), required)
            facts.scopes = tuple(str(scope) for scope in (metadata.get("scopes") or []))
        return facts

    @staticmethod
    def _advise(report: IntegrationReport) -> IntegrationReport:
        """One place that decides what a reviewer should be told to do."""
        warnings = list(report.warnings)
        action = report.recommended_action
        facts = report.credential

        if report.state is IntegrationState.AUTH_ERROR:
            action = action or f"Re-issue the {report.name} credential and store it again in Settings."
        elif report.state is IntegrationState.QUOTA:
            action = action or f"{report.name} reports no remaining quota for this credential."
        elif report.state is IntegrationState.UNREACHABLE:
            action = action or f"{report.name} did not answer; check that it is running and reachable."

        if facts.expiry is ExpiryState.EXPIRED:
            warnings.append(f"The credential expired on {facts.expires_at}.")
            action = f"Issue a new {report.name} credential; this one has expired."
        elif facts.expiry is ExpiryState.EXPIRING_SOON:
            days = facts.days_until_expiry
            urgency = "in less than a week" if (days is not None and days <= EXPIRING_URGENT_DAYS) else f"in {days} days"
            warnings.append(f"The credential expires {urgency}, on {facts.expires_at}.")
            action = action or f"Plan a {report.name} credential rotation before {facts.expires_at}."

        if facts.privilege is PrivilegeState.OVERPRIVILEGED:
            warnings.append(
                "The credential grants " + ", ".join(facts.extra_scopes[:6])
                + (" and more" if len(facts.extra_scopes) > 6 else "")
                + " which SAM never uses.")
            action = action or "Re-issue the key with only the scopes SAM needs."
        elif facts.configured and facts.privilege is PrivilegeState.UNKNOWN and report.name == "n8n":
            warnings.append("No scope list was recorded, so over-privilege cannot be ruled out.")
            action = action or "Record the key's scopes and expiry when storing it, so this can be checked."

        report.warnings = tuple(warnings)
        report.recommended_action = action
        return report

    # -- public -------------------------------------------------------------
    async def reports(self, *, refresh: bool = False) -> list[IntegrationReport]:
        import asyncio

        if not refresh and self._cache and time.time() - self._cache[0] < self.ttl_seconds:
            return self._cache[1]
        openrouter, groq, gemini, n8n = await asyncio.gather(
            self._openrouter(),
            self._openai_compatible("groq", self.settings.groq_base_url, self.settings.groq_api_key),
            self._openai_compatible("gemini", self.settings.gemini_base_url, self.settings.gemini_api_key),
            asyncio.to_thread(self._n8n),
        )
        found = [openrouter, groq, gemini, n8n]
        self._cache = (time.time(), found)
        return found

    async def summary(self, *, refresh: bool = False) -> dict[str, Any]:
        found = await self.reports(refresh=refresh)
        cached_at = self._cache[0] if self._cache else time.time()
        return {
            "integrations": [item.as_dict() for item in found],
            "needs_attention": [item.name for item in found if item.needs_attention],
            "checked_at": datetime.fromtimestamp(cached_at, tz=UTC).isoformat(),
            "ttl_seconds": self.ttl_seconds,
            # Said plainly so nobody reads a cached panel as a live one.
            "note": "States are cached; press Refresh to re-check. No completion is ever spent on this panel.",
        }


def clear_metadata(database: Any, name: str) -> bool:
    """Forget a credential's recorded facts when the credential itself goes."""
    current = read_metadata(database)
    if name not in current:
        return False
    current.pop(name, None)
    database.update_settings({METADATA_SETTING_KEY: current})
    return True
