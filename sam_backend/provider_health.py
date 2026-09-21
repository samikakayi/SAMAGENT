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
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import httpx

from .models import ErrorCategory, ModelError

OPENROUTER_CATALOGUE_TTL = 600.0
PROBE_TIMEOUT_SECONDS = 20.0
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
    supports_structured_output: bool = False
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
        payload = asdict(self)
        payload["availability"] = self.availability.value
        payload["usable"] = self.usable
        return payload


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

    def as_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary.as_dict(),
            "fallback": self.fallback.as_dict() if self.fallback else None,
            "active": self.active.as_dict() if self.active else None,
            "fallback_enabled": self.fallback_enabled,
            "fallback_engaged": bool(self.active and self.fallback and self.active.model == self.fallback.model
                                     and self.active.model != self.primary.model),
            "fallback_reason": self.fallback_reason,
            "blocked": self.blocked,
        }


class ProviderHealth:
    """Cheap, cached answers about whether a model can drive a run."""

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self._verdicts: dict[tuple[str, str], ModelCapability] = {}
        self._catalogue: tuple[float, dict[str, dict[str, Any]], Availability, str] | None = None

    # -- cache -------------------------------------------------------------
    def cached(self, provider: str, model: str) -> ModelCapability | None:
        entry = self._verdicts.get((provider, model))
        if entry is None:
            return None
        if time.time() - entry.checked_at > VERDICT_TTL_SECONDS[entry.availability]:
            self._verdicts.pop((provider, model), None)
            return None
        return entry

    def remember(self, capability: ModelCapability) -> ModelCapability:
        capability.checked_at = time.time()
        self._verdicts[(capability.provider, capability.model)] = capability
        return capability

    def record_failure(self, provider: str, model: str, error: ModelError) -> None:
        """Teach the cache from a real request that failed.

        Without this, a run that dies on a credit error would be followed by
        another run that probes the same paid model all over again.
        """
        availability = CATEGORY_TO_AVAILABILITY.get(getattr(error, "category", ErrorCategory.UNKNOWN))
        if availability is None:
            return
        known = self.cached(provider, model)
        self.remember(ModelCapability(
            provider=provider, model=model, availability=availability,
            reason=str(error)[:300],
            supports_tools=known.supports_tools if known else False,
            supports_structured_output=known.supports_structured_output if known else False,
            context_length=known.context_length if known else 0,
            cost_class=known.cost_class if known else "unknown",
        ))

    def invalidate(self) -> None:
        self._verdicts.clear()
        self._catalogue = None

    # -- OpenRouter: catalogue + key, both free ----------------------------
    async def _openrouter_catalogue(self) -> tuple[dict[str, dict[str, Any]], Availability, str]:
        if self._catalogue and time.time() - self._catalogue[0] < OPENROUTER_CATALOGUE_TTL:
            return self._catalogue[1], self._catalogue[2], self._catalogue[3]
        key = getattr(self.settings, "openrouter_api_key", None)
        if not key:
            return {}, Availability.UNAVAILABLE_AUTH, "No OpenRouter credential is configured."
        headers = {"Authorization": f"Bearer {key}"}
        base = getattr(self.settings, "openrouter_base_url", "https://openrouter.ai/api/v1")
        try:
            async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_SECONDS, trust_env=False) as client:
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
        remaining = account.get("limit_remaining")
        broke = bool(account.get("is_free_tier")) and not remaining
        note = "free-tier key with no paid credit" if broke else ""
        self._catalogue = (time.time(), catalogue, Availability.AVAILABLE, note)
        return catalogue, Availability.AVAILABLE, note

    async def _check_openrouter(self, model: str) -> ModelCapability:
        catalogue, account_state, note = await self._openrouter_catalogue()
        if account_state is not Availability.AVAILABLE:
            return ModelCapability("openrouter", model, account_state, note)
        entry = catalogue.get(model)
        if entry is None:
            return ModelCapability("openrouter", model, Availability.UNSUPPORTED,
                                   f"{model} is not in this provider's catalogue.")
        parameters = entry.get("supported_parameters") or []
        pricing = entry.get("pricing") or {}
        try:
            billed = float(pricing.get("prompt") or 0) > 0 or float(pricing.get("completion") or 0) > 0
        except (TypeError, ValueError):
            billed = False
        capability = ModelCapability(
            provider="openrouter", model=model,
            availability=Availability.AVAILABLE, reason="",
            supports_tools="tools" in parameters,
            supports_structured_output="structured_outputs" in parameters or "response_format" in parameters,
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
        return ModelCapability(
            provider, model, Availability.AVAILABLE, "",
            supports_tools=True, supports_structured_output=True,
            context_length=MINIMUM_CONTEXT_TOKENS, cost_class="free",
        )

    # -- public ------------------------------------------------------------
    async def capability(self, provider: str, model: str, adapters: Any, *, refresh: bool = False) -> ModelCapability:
        if not model:
            return ModelCapability(provider, model, Availability.UNSUPPORTED, "No model is configured.")
        if not refresh:
            cached = self.cached(provider, model)
            if cached is not None:
                return cached
        if provider == "openrouter":
            capability = await self._check_openrouter(model)
        else:
            capability = await self._check_listed(provider, model, adapters)
        return self.remember(capability)

    async def resolve(self, adapters: Any, *, refresh: bool = False) -> ModelResolution:
        """Decide which real model drives the next run."""
        provider = str(getattr(self.settings, "default_provider", "") or "")
        primary_model = str(getattr(self.settings, "default_model", "") or "")
        primary = await self.capability(provider, primary_model, adapters, refresh=refresh)

        fallback_model = str(getattr(self.settings, "fallback_model", "") or "")
        enabled = bool(getattr(self.settings, "fallback_enabled", False))
        fallback: ModelCapability | None = None
        if fallback_model and fallback_model != primary_model:
            fallback = await self.capability(provider, fallback_model, adapters, refresh=refresh)

        if primary.usable:
            return ModelResolution(primary, fallback, primary, enabled)
        if primary.availability is Availability.UNKNOWN:
            # The check could not reach a verdict. Refusing to start on a
            # non-answer would turn a diagnostic into an outage, so the run
            # proceeds and the first real request decides.
            return ModelResolution(primary, fallback, primary, enabled)
        if not enabled:
            # Honest stop. Silently switching models is the thing this policy
            # exists to prevent.
            return ModelResolution(primary, fallback, None, enabled,
                                   f"{primary.model} is unavailable ({primary.reason or primary.availability.value}) "
                                   "and fallback is disabled.")
        if fallback is not None and fallback.usable:
            return ModelResolution(primary, fallback, fallback, enabled,
                                   f"{primary.model} is unavailable: {primary.reason or primary.availability.value}")
        return ModelResolution(primary, fallback, None, enabled,
                               f"{primary.model} is unavailable ({primary.reason or primary.availability.value}) "
                               "and no usable fallback is configured.")
