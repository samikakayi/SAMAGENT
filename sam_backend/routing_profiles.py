"""Which models a run is allowed to reach, and in what order.

A routing profile answers one question: may this run spend money? It does not
choose a model -- `ModelRouter` still does that, with the same health checks,
retries and fallback events as before. This module only says which candidates
the router is permitted to consider, and in which order.

    PREMIUM   the configured chain, exactly as it behaved before this existed
    BALANCED  the free candidates first, then that same configured chain
    FREE      the free candidates and nothing else

FREE fails closed. If no free candidate survives, the run stops with
`provider_unavailable` rather than quietly reaching for a paid model: a profile
that silently spent money when it ran out of free options would be worse than
no profile at all.

What "free" means here is deliberately narrow. It is a routing policy, not a
billing guarantee -- SAM promises never to *choose* a model it knows to be
paid while FREE is active, and cannot promise what a third party charges under
future account terms. Eligibility is therefore explicit configuration rather
than a guess from a model name, with one exception: OpenRouter documents the
`:free` suffix as a contract, so a slug without it is known to be paid and is
refused even if an operator lists it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

PROFILES = ("FREE", "BALANCED", "PREMIUM")
DEFAULT_PROFILE = "PREMIUM"
# The most a run may carry, so a long list cannot turn one prompt into dozens
# of provider calls.
MAX_FREE_CANDIDATES = 12


class Eligibility(StrEnum):
    """What is actually known about a candidate's cost."""

    FREE = "free"            # a provider contract says so
    PAID = "paid"            # a provider contract says so
    UNVERIFIED = "unverified"  # the operator asserts it; SAM cannot confirm it


@dataclass(frozen=True, slots=True)
class FreeCandidate:
    provider: str
    model: str

    @property
    def reference(self) -> str:
        return f"{self.provider}/{self.model}"

    def as_dict(self) -> dict[str, str]:
        return {"provider": self.provider, "model": self.model,
                "reference": self.reference, "eligibility": eligibility(self.provider, self.model).value}


def normalise_profile(value: object) -> str:
    text = str(value or "").strip().upper()
    return text if text in PROFILES else DEFAULT_PROFILE


def eligibility(provider: str, model: str) -> Eligibility:
    """What SAM actually knows about this candidate's cost.

    Only claims a contract it can point at. OpenRouter publishes the `:free`
    suffix, so its absence means paid. No other provider here exposes a
    per-model free flag -- Groq and Gemini meter free use by account and rate
    rather than by model -- so those stay UNVERIFIED however they are spelled.
    """
    name = (provider or "").strip().lower()
    slug = (model or "").strip()
    if name == "openrouter":
        return Eligibility.FREE if slug.endswith(":free") else Eligibility.PAID
    return Eligibility.UNVERIFIED


def parse_candidates(raw: object) -> list[FreeCandidate]:
    """Read the persisted ordered list, keeping the operator's order.

    Accepts "provider/model" strings or mappings, ignores anything malformed
    rather than failing a whole settings load, and de-duplicates so one entry
    listed twice is not called twice.
    """
    items = raw if isinstance(raw, list) else []
    candidates: list[FreeCandidate] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        provider = model = ""
        if isinstance(item, str) and "/" in item:
            provider, _, model = item.partition("/")
        elif isinstance(item, dict):
            provider, model = str(item.get("provider") or ""), str(item.get("model") or "")
        provider, model = provider.strip().lower(), model.strip()
        if not provider or not model:
            continue
        key = (provider, model)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(FreeCandidate(provider, model))
        if len(candidates) >= MAX_FREE_CANDIDATES:
            break
    return candidates


def spendable(profile: str) -> bool:
    """Whether this profile may reach a model that is not free."""
    return normalise_profile(profile) != "FREE"


def usable_free_candidates(candidates: list[FreeCandidate]) -> tuple[list[FreeCandidate], list[dict[str, str]]]:
    """The candidates FREE may use, and why any were refused.

    A candidate known to be paid is dropped here rather than at call time, so
    the refusal is visible in the resolution output instead of showing up as a
    surprise charge.
    """
    usable: list[FreeCandidate] = []
    refused: list[dict[str, str]] = []
    for candidate in candidates:
        if eligibility(candidate.provider, candidate.model) is Eligibility.PAID:
            refused.append({
                "reference": candidate.reference,
                "reason": "the provider publishes this model as paid, so FREE will not select it",
            })
        else:
            usable.append(candidate)
    return usable, refused
