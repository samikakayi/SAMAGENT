"""Which model will actually drive a run, and whether it can.

An autonomous run is expensive to start and embarrassing to abandon halfway,
so the model is checked before planning rather than discovered mid-flight.

The check is deliberately free. OpenRouter publishes a model catalogue and a
key summary, both zero-token GETs, and between them they answer every question
that matters: is the model real, does it support tool calling, how much
context does it have, is it billed, and does this account have anything to
bill against. A paid model on a free-tier key is known to be unusable without
spending a cent to find out.

Fallback here is between REAL models only. There is no scripted provider in
this module, and a fallback is always recorded and shown -- an agent quietly
running on a different brain than the operator configured is worse than one
that stops.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from typing import Any

import httpx

from .config import Settings
from .models import ErrorCategory, ModelError

OPENROUTER_CATALOGUE_TTL = 600.0
# SAM's executor sends the plan, observations and every tool schema each turn.
MINIMUM_CONTEXT_TOKENS = 16_000


class Availability(StrEnum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE_AUTH = "UNAVAILABLE_AUTH"
    UNAVAILABLE_QUOTA = "UNAVAILABLE_QUOTA"
    UNAVAILABLE_RATE_LIMIT = "UNAVAILABLE_RATE_LIMIT"
    UNAVAILABLE_NETWORK = "UNAVAILABLE_NETWORK"
    UNSUPPORTED = "UNSUPPORTED"
    UNKNOWN = "UNKNOWN"


# How long a verdict may be trusted. A confirmed quota or auth problem will not
# fix itself in seconds, and re-probing a paid model after a credit failure is
# exactly the waste this cache exists to prevent. Transient faults expire fast
# so a recovered provider is picked up quickly. Nothing is cached forever.
VERDICT_TTL_SECONDS: dict[Availability, float] = {
    Availability.AVAILABLE: 300.0,
    Availability.UNAVAILABLE_AUTH: 600.0,
    Availability.UNAVAILABLE_QUOTA: 600.0,
    Availability.UNAVAILABLE_RATE_LIMIT: 60.0,
    Availability.UNAVAILABLE_NETWORK: 30.0,
    Availability.UNSUPPORTED: 900.0,
    Availability.UNKNOWN: 60.0,
}

# Verdicts definite enough that sending another request is pure waste, and
# the category a skipped request reports.
CONFIRMED_UNAVAILABLE: dict[Availability, ErrorCategory] = {
    Availability.UNAVAILABLE_QUOTA: ErrorCategory.QUOTA,
    Availability.UNAVAILABLE_AUTH: ErrorCategory.AUTH,
}

# A runtime failure is evidence too: fold it into the same vocabulary so the
# cache learns from real requests, not only from preflight.
CATEGORY_TO_AVAILABILITY: dict[ErrorCategory, Availability] = {
    ErrorCategory.AUTH: Availability.UNAVAILABLE_AUTH,
    ErrorCategory.QUOTA: Availability.UNAVAILABLE_QUOTA,
    ErrorCategory.RATE_LIMIT: Availability.UNAVAILABLE_RATE_LIMIT,
    ErrorCategory.NETWORK: Availability.UNAVAILABLE_NETWORK,
    ErrorCategory.TIMEOUT: Availability.UNAVAILABLE_NETWORK,
    ErrorCategory.NOT_CONFIGURED: Availability.UNAVAILABLE_AUTH,
}


@dataclass(slots=True)
class ModelCapability:
    """What one model is, and whether it can be used right now."""

    provider: str
    model: str
    availability: Availability = Availability.UNKNOWN
    reason: str = ""
    supports_tools: bool = False
    context_length: int = 0
    cost_class: str = "unknown"  # free | paid | unknown
    checked_at: float = field(default_factory=time.time)

    @property
    def usable(self) -> bool:
        """Available AND able to do what an autonomous run requires."""
        return (
            self.availability is Availability.AVAILABLE
            and self.supports_tools
            and self.context_length >= MINIMUM_CONTEXT_TOKENS
        )

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "availability": self.availability.value, "usable": self.usable}


@dataclass(slots=True)
class ModelResolution:
    """Which model a run will use, and why -- the UI renders this directly."""

    primary: ModelCapability
    fallback: ModelCapability | None
    active: ModelCapability | None
    fallback_enabled: bool
    fallback_reason: str = ""

    @property
    def blocked(self) -> bool:
        return self.active is None

    @property
    def fallback_engaged(self) -> bool:
        return self.active is not None and self.active is self.fallback

    def as_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary.as_dict(),
            "fallback": self.fallback.as_dict() if self.fallback else None,
            "active": self.active.as_dict() if self.active else None,
            "fallback_enabled": self.fallback_enabled,
            "fallback_engaged": self.fallback_engaged,
            "fallback_reason": self.fallback_reason,
            "blocked": self.blocked,
        }


class ProviderHealth:
    """Cheap, cached answers about whether a model can drive a run."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._verdicts: dict[tuple[str, str], ModelCapability] = {}
        self._catalogue: tuple[float, dict[str, dict[str, Any]], str] | None = None

    # -- cache -------------------------------------------------------------
    def cached(self, provider: str, model: str) -> ModelCapability | None:
        entry = self._verdicts.get((provider, model))
        if entry is None:
            return None
        if time.time() - entry.checked_at > VERDICT_TTL_SECONDS[entry.availability]:
            del self._verdicts[(provider, model)]
            return None
        return entry

    def remember(self, capability: ModelCapability) -> ModelCapability:
        capability.checked_at = time.time()
        self._verdicts[(capability.provider, capability.model)] = capability
        return capability

    def record_failure(self, provider: str, model: str, error: ModelError) -> None:
        """Teach the cache from a real request that failed, keeping whatever
        the catalogue already said about the model itself."""
        availability = CATEGORY_TO_AVAILABILITY.get(error.category)
        if availability is None:
            return
        known = self.cached(provider, model) or ModelCapability(provider, model)
        self.remember(replace(known, availability=availability, reason=str(error)[:300]))

    def invalidate(self) -> None:
        self._verdicts.clear()
        self._catalogue = None

    # -- OpenRouter: catalogue + key, both free ----------------------------
    async def _openrouter_catalogue(self) -> tuple[dict[str, dict[str, Any]], Availability, str]:
        """The catalogue and a note about the account, or why neither could
        be fetched. Only a successful fetch is cached."""
        if self._catalogue and time.time() - self._catalogue[0] < OPENROUTER_CATALOGUE_TTL:
            return self._catalogue[1], Availability.AVAILABLE, self._catalogue[2]
        key = self.settings.openrouter_api_key
        if not key:
            return {}, Availability.UNAVAILABLE_AUTH, "No OpenRouter credential is configured."
        headers = {"Authorization": f"Bearer {key}"}
        base = self.settings.openrouter_base_url
        try:
            async with httpx.AsyncClient(timeout=20, trust_env=False) as client:
                key_response = await client.get(f"{base}/key", headers=headers)
                if key_response.status_code in (401, 403):
                    return {}, Availability.UNAVAILABLE_AUTH, "The OpenRouter credential was rejected."
                key_response.raise_for_status()
                account = key_response.json().get("data") or {}
                catalogue_response = await client.get(f"{base}/models", headers=headers)
                catalogue_response.raise_for_status()
                entries = catalogue_response.json().get("data") or []
        except httpx.HTTPError as exc:
            return {}, Availability.UNAVAILABLE_NETWORK, f"OpenRouter was unreachable: {type(exc).__name__}"
        except ValueError:
            return {}, Availability.UNKNOWN, "OpenRouter returned an unreadable catalogue."

        catalogue = {str(item.get("id")): item for item in entries if item.get("id")}
        # A free-tier key cannot pay for a billed model. Knowing that here
        # saves starting a run that would die on its first paid request.
        broke = bool(account.get("is_free_tier")) and not account.get("limit_remaining")
        note = "free-tier key with no paid credit" if broke else ""
        self._catalogue = (time.time(), catalogue, note)
        return catalogue, Availability.AVAILABLE, note

    async def _check_openrouter(self, model: str) -> ModelCapability:
        catalogue, account_state, note = await self._openrouter_catalogue()
        if account_state is not Availability.AVAILABLE:
            return ModelCapability("openrouter", model, account_state, note)
        entry = catalogue.get(model)
        if entry is None:
            return ModelCapability("openrouter", model, Availability.UNSUPPORTED,
                                   f"{model} is not in this provider's catalogue.")
        pricing = entry.get("pricing") or {}
        try:
            billed = float(pricing.get("prompt") or 0) > 0 or float(pricing.get("completion") or 0) > 0
        except (TypeError, ValueError):
            billed = False
        capability = ModelCapability(
            "openrouter", model, Availability.AVAILABLE,
            supports_tools="tools" in (entry.get("supported_parameters") or []),
            context_length=int(entry.get("context_length") or 0),
            cost_class="paid" if billed else "free",
        )
        if billed and note:
            capability.availability = Availability.UNAVAILABLE_QUOTA
            capability.reason = f"{model} is billed and the account has no paid credit ({note})."
        elif not capability.supports_tools:
            capability.availability = Availability.UNSUPPORTED
            capability.reason = f"{model} does not support tool calling, which an autonomous run requires."
        elif capability.context_length < MINIMUM_CONTEXT_TOKENS:
            capability.availability = Availability.UNSUPPORTED
            capability.reason = (
                f"{model} offers {capability.context_length} tokens of context; "
                f"an autonomous run needs at least {MINIMUM_CONTEXT_TOKENS}."
            )
        return capability

    async def _check_listed(self, provider: str, model: str, adapters: Any) -> ModelCapability:
        """Providers without a rich catalogue: does it serve this model at all?"""
        try:
            listed = await adapters.get(provider).list_models()
        except Exception as exc:  # noqa: BLE001 - adapter/transport errors vary
            return ModelCapability(provider, model, Availability.UNAVAILABLE_NETWORK,
                                   f"{provider} was unreachable: {type(exc).__name__}")
        names = {str(item.get("id") or item.get("name")) for item in listed or []}
        if not names:
            return ModelCapability(provider, model, Availability.UNAVAILABLE_NETWORK,
                                   f"{provider} published no models.")
        if model not in names:
            return ModelCapability(provider, model, Availability.UNSUPPORTED,
                                   f"{provider} does not serve {model}.")
        # A local runtime bills nothing and its served models take tools.
        return ModelCapability(provider, model, Availability.AVAILABLE,
                               supports_tools=True, context_length=MINIMUM_CONTEXT_TOKENS, cost_class="free")

    # -- public ------------------------------------------------------------
    async def capability(self, provider: str, model: str, adapters: Any, *, refresh: bool = False) -> ModelCapability:
        if not model:
            return ModelCapability(provider, model, Availability.UNSUPPORTED, "No model is configured.")
        if provider in ("", "auto"):
            # The router picks per request, so there is nothing to interrogate
            # yet. UNKNOWN lets the run proceed rather than inventing a fault.
            return ModelCapability(provider, model, Availability.UNKNOWN,
                                   "Automatic routing chooses the model per request.")
        if not refresh and (cached := self.cached(provider, model)) is not None:
            return cached
        if provider == "openrouter":
            capability = await self._check_openrouter(model)
        else:
            capability = await self._check_listed(provider, model, adapters)
        return self.remember(capability)

    async def resolve(self, adapters: Any, *, refresh: bool = False) -> ModelResolution:
        """Decide which real model drives the next run."""
        provider = self.settings.default_provider
        primary = await self.capability(provider, self.settings.default_model, adapters, refresh=refresh)
        fallback = None
        if self.settings.fallback_model and self.settings.fallback_model != primary.model:
            fallback = await self.capability(provider, self.settings.fallback_model, adapters, refresh=refresh)
        enabled = self.settings.fallback_enabled

        # A non-answer must not become an outage: an UNKNOWN primary proceeds
        # and the first real request decides.
        if primary.usable or primary.availability is Availability.UNKNOWN:
            return ModelResolution(primary, fallback, primary, enabled)
        problem = f"{primary.model} is unavailable ({primary.reason or primary.availability.value})"
        if not enabled:
            # Honest stop. Silently switching models is the thing this policy
            # exists to prevent.
            return ModelResolution(primary, fallback, None, enabled, f"{problem} and fallback is disabled.")
        if fallback is not None and fallback.usable:
            return ModelResolution(primary, fallback, fallback, enabled, f"{problem}.")
        if fallback is None:
            return ModelResolution(primary, fallback, None, enabled, f"{problem} and no fallback is configured.")
        return ModelResolution(
            primary, fallback, None, enabled,
            f"{problem} and the fallback {fallback.model} is also unavailable "
            f"({fallback.reason or fallback.availability.value}).",
        )
