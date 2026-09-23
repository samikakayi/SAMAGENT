from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from .config import Settings
from .db import Database
from .models import AdapterRegistry, AssistantTurn, ErrorCategory, ModelError
from .provider_health import CONFIRMED_UNAVAILABLE, ProviderHealth
from .routing_profiles import (
    FreeCandidate, normalise_profile, parse_candidates, usable_free_candidates,
)


@dataclass(slots=True)
class TaskProfile:
    complexity: str
    needs_tools: bool
    needs_vision: bool
    privacy_sensitive: bool
    realtime: bool
    deterministic_candidate: bool


@dataclass(slots=True)
class RouteChoice:
    provider: str
    model: str
    reason: str


class ModelRouter:
    def __init__(
        self, settings: Settings, adapters: AdapterRegistry, database: Database,
        health: ProviderHealth | None = None,
    ) -> None:
        self.settings = settings
        self.adapters = adapters
        self.database = database
        # Shared with the orchestrator's preflight: a quota or credential
        # verdict learned anywhere is honoured everywhere.
        self.health = health or ProviderHealth(settings)
        self._health_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self.failures: dict[str, int] = {}

    @staticmethod
    def profile(message: str) -> TaskProfile:
        lowered = message.lower()
        needs_vision = any(token in lowered for token in ("screenshot", "screen", "chart image", "سەیری شاشە", "وێنە", "چارتەکە"))
        realtime = any(token in lowered for token in ("realtime", "live voice", "لایڤ", "دەنگ"))
        privacy = any(token in lowered for token in ("credential", "secret", "password", "api key", ".env", "نهێنی", "پاسوۆرد"))
        complex_markers = (
            "compare", "analyze", "analysis", "strategy", "theory", "wyckoff", "ict", "smc", "backtest",
            "بەراورد", "شیکاری", "تیۆری", "ستراتیژی",
        )
        complex_count = sum(token in lowered for token in complex_markers)
        complexity = "strong" if complex_count >= 2 or len(message) > 1200 else "balanced" if complex_count else "fast"
        deterministic = bool(re.search(r"(?i)(open|focus|switch|set|بکەرەوە|بچۆ)\s+.{0,30}(tradingview|timeframe|[0-9]+m|[0-9]+h)", message))
        tool_markers = (
            "file", "folder", "directory", "workspace", "project", "terminal", "powershell", "command", "shell",
            "python", "script", "code", "browser", "website", "url", "launch", "application", "search", "find",
            "read ", "write ", "edit", "change", "create", "delete", "remove", "copy", "move", "rename", "list",
            "inspect", "run ", "execute", "remember", "memory", "plan", "desktop", "screen", "tradingview", "download",
            "install", "do it", "فایل", "فۆڵدەر", "پرۆژە", "تێرمیناڵ", "پاوەرشێڵ", "پایتۆن", "کۆد", "براوزەر",
            "وێبسایت", "بکەرەوە", "بدۆزەوە", "بخوێنەوە", "بنووسە", "دەستکاری", "دروست بکە", "بسڕەوە",
            "بگوازەوە", "لیست", "پلان", "شاشە", "ئەپ", "داگرە", "دابمەزرێنە", "بیکە",
        )
        needs_tools = needs_vision or deterministic or any(token in lowered for token in tool_markers)
        return TaskProfile(complexity, needs_tools, needs_vision, privacy, realtime, deterministic)

    async def _models(self, provider: str) -> list[dict[str, Any]]:
        cached = self._health_cache.get(provider)
        cache_ttl = 30 if cached and cached[1] else 3
        if cached and time.monotonic() - cached[0] < cache_ttl:
            return cached[1]
        try:
            models = await self.adapters.get(provider).list_models()
        except Exception:
            models = []
        # A transient timeout must not make a known local model disappear for
        # 30 seconds while its single CPU inference slot is busy.
        if not models and cached and cached[1]:
            return cached[1]
        self._health_cache[provider] = (time.monotonic(), models)
        return models

    def budget_state(self) -> dict[str, Any]:
        summary = self.database.model_cost_summary()
        daily = float(summary["today"]["cost_usd"])
        monthly = float(summary["month"]["cost_usd"])
        daily_ratio = daily / self.settings.daily_budget_usd if self.settings.daily_budget_usd > 0 else 0.0
        monthly_ratio = monthly / self.settings.monthly_budget_usd if self.settings.monthly_budget_usd > 0 else 0.0
        ratio = max(daily_ratio, monthly_ratio)
        if ratio >= self.settings.budget_hard_ratio:
            mode = "LOCAL_ONLY"
        elif ratio >= self.settings.budget_warning_ratio:
            mode = "LOCAL_FIRST"
        else:
            mode = "NORMAL"
        return {
            "mode": mode,
            "daily_ratio": daily_ratio,
            "monthly_ratio": monthly_ratio,
            "daily_budget_usd": self.settings.daily_budget_usd,
            "monthly_budget_usd": self.settings.monthly_budget_usd,
            **summary,
        }

    def _openrouter_slug(self, candidate: str | None) -> str:
        value = str(candidate or "").strip()
        # OpenRouter ids look like "anthropic/claude-sonnet-4.5". Local Ollama
        # tags like "qwen3.5:4b" must never be forwarded there.
        if "/" in value and not value.lower().startswith("qwen"):
            return value
        return ""

    def _model_for(self, provider: str, profile: TaskProfile, explicit_model: str | None = None) -> str:
        if provider == "openrouter":
            saved = self._openrouter_slug(explicit_model) or self._openrouter_slug(self.settings.default_model)
            if profile.needs_vision:
                return saved or self.settings.openrouter_vision_model
            if profile.complexity == "strong":
                return saved or self.settings.openrouter_strong_model
            return saved or self.settings.openrouter_fast_model
        if explicit_model:
            return explicit_model
        if provider == "ollama":
            return self.settings.default_model
        if provider == "openai":
            return self.settings.openai_model
        if provider == "litellm":
            if profile.needs_vision:
                return self.settings.litellm_vision_model
            return self.settings.litellm_strong_model if profile.complexity == "strong" else self.settings.litellm_fast_model
        raise ModelError(f"Unsupported route provider: {provider}")

    async def route(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> tuple[TaskProfile, list[RouteChoice]]:
        profile = self.profile(message)
        mode = self.settings.model_mode
        explicit = provider and provider.lower() not in {"auto", ""}
        if explicit:
            selected = provider.lower()
            return profile, [RouteChoice(selected, self._model_for(selected, profile, model), "Explicit provider selection")]
        budget = self.budget_state()
        if budget["mode"] == "LOCAL_ONLY":
            mode = "LOCAL_ONLY"
        elif budget["mode"] == "LOCAL_FIRST" and mode == "AUTO":
            mode = "LOCAL_FIRST"

        ollama_available = bool(await self._models("ollama"))
        litellm_available = bool(await self._models("litellm"))
        openrouter_configured = bool(self.settings.openrouter_api_key)
        openai_configured = bool(self.settings.openai_api_key)
        local = [RouteChoice("ollama", self._model_for("ollama", profile), "Local privacy/cost route")] if ollama_available else []
        cloud: list[RouteChoice] = []
        if litellm_available:
            cloud.append(RouteChoice("litellm", self._model_for("litellm", profile), "Normalized LiteLLM gateway"))
        if openrouter_configured:
            cloud.append(RouteChoice("openrouter", self._model_for("openrouter", profile), "OpenRouter cloud fallback"))
        if openai_configured:
            cloud.append(RouteChoice("openai", self._model_for("openai", profile), "Direct OpenAI fallback"))

        if mode == "LOCAL_ONLY":
            choices = local
        elif mode == "CLOUD_ONLY":
            choices = cloud
        elif mode == "MANUAL":
            selected = self.settings.default_provider
            choices = [RouteChoice(selected, self._model_for(selected, profile, model), "Manual configured route")]
        elif mode == "LOCAL_FIRST" or profile.privacy_sensitive or profile.complexity == "fast":
            choices = local + cloud
        else:
            choices = cloud + local if profile.complexity == "strong" or profile.needs_vision else local + cloud
        # Stable de-duplication, then penalize repeatedly failing providers.
        deduplicated: dict[tuple[str, str], RouteChoice] = {}
        for choice in choices:
            deduplicated.setdefault((choice.provider, choice.model), choice)
        ordered = list(deduplicated.values())
        ordered.sort(key=lambda choice: self.failures.get(choice.provider, 0))
        ordered = self.apply_profile(ordered)
        if not ordered:
            raise self._no_route_error()
        return profile, ordered

    def _no_route_error(self) -> ModelError:
        """Say which rule left nothing to call, rather than one generic line.

        FREE running out of candidates is a different situation from having no
        provider configured at all, and reporting it as the latter would hide
        that SAM refused to spend money on purpose.
        """
        selected = normalise_profile(getattr(self.settings, "routing_profile", None))
        if selected != "FREE":
            return ModelError(
                "No AI model route is available. Start Ollama with a downloaded model, "
                "configure LiteLLM, or set OPENROUTER_API_KEY."
            )
        usable, refused = usable_free_candidates(self.free_candidates())
        if not self.free_candidates():
            detail = "no free-tier candidates are configured"
        elif not usable:
            detail = "every configured candidate is published by its provider as paid"
        else:
            detail = "every free-tier candidate was unavailable"
        if refused:
            detail += f" ({len(refused)} refused as paid)"
        return ModelError(
            f"FREE routing has nothing left to call: {detail}. SAM will not fall back to a "
            "paid model while FREE is active; add a free-tier candidate or switch to BALANCED.",
            ErrorCategory.NOT_CONFIGURED,
        )

    # -- routing profile ---------------------------------------------------
    def free_candidates(self) -> list[FreeCandidate]:
        return parse_candidates(getattr(self.settings, "free_candidates", []))

    def apply_profile(self, configured: list[RouteChoice]) -> list[RouteChoice]:
        """Narrow or widen the configured chain according to the profile.

        PREMIUM returns the chain untouched, so an installation that never set
        a profile routes exactly as it did before profiles existed. FREE keeps
        only the free candidates, in the operator's order -- no failure-count
        reordering, because the configured order is the whole promise. BALANCED
        puts those first and keeps the configured chain behind them.
        """
        selected = normalise_profile(getattr(self.settings, "routing_profile", None))
        if selected == "PREMIUM":
            return configured
        usable, _refused = usable_free_candidates(self.free_candidates())
        free = [
            RouteChoice(candidate.provider, candidate.model, f"{selected} free-tier candidate")
            for candidate in usable
        ]
        if selected == "FREE":
            return free
        seen = {(choice.provider, choice.model) for choice in free}
        return free + [c for c in configured if (c.provider, c.model) not in seen]

    def profile_state(self) -> dict[str, Any]:
        """Why the next run will reach what it reaches, without secrets."""
        selected = normalise_profile(getattr(self.settings, "routing_profile", None))
        usable, refused = usable_free_candidates(self.free_candidates())
        return {
            "routing_profile": selected,
            "free_candidates": [candidate.as_dict() for candidate in usable],
            "refused_candidates": refused,
            "paid_fallback_allowed": selected != "FREE",
        }

    async def complete(
        self,
        *,
        message: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        provider: str | None,
        model: str | None,
        conversation_id: str | None,
        task_id: str | None = None,
    ) -> tuple[AssistantTurn, RouteChoice, list[dict[str, str]]]:
        _, choices = await self.route(message, provider=provider, model=model)
        failures: list[dict[str, str]] = []
        for choice in choices:
            verdict = self.health.cached(choice.provider, choice.model)
            if verdict is not None and verdict.availability in CONFIRMED_UNAVAILABLE:
                # Confirmed quota or credential failure: sending another
                # request cannot succeed and only burns the account's goodwill.
                failures.append({
                    "provider": choice.provider, "model": choice.model,
                    "error": f"skipped, {verdict.reason}",
                    "category": str(CONFIRMED_UNAVAILABLE[verdict.availability]),
                })
                continue
            try:
                turn = await self.adapters.get(choice.provider).complete(messages, tools, choice.model)
                self.failures[choice.provider] = max(0, self.failures.get(choice.provider, 0) - 1)
                raw = turn.raw or {}
                usage = raw.get("usage") or {}
                input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
                output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
                cost = usage.get("cost", usage.get("total_cost"))
                try:
                    cost_value = float(cost) if cost is not None else None
                except (TypeError, ValueError):
                    cost_value = None
                self.database.add_model_usage(
                    provider=choice.provider,
                    model=str(raw.get("model") or choice.model),
                    route_mode=self.settings.model_mode,
                    input_tokens=int(input_tokens) if input_tokens is not None else None,
                    output_tokens=int(output_tokens) if output_tokens is not None else None,
                    cost_usd=cost_value,
                    task_id=task_id,
                    conversation_id=conversation_id,
                    metadata={"outcome": "succeeded", "reason": choice.reason, "fallbacks_before_success": failures, "usage_cost_reported": cost is not None},
                )
                turn.raw = {**raw, "route": {"provider": choice.provider, "model": choice.model, "reason": choice.reason}, "fallbacks": failures}
                return turn, choice, failures
            except ModelError as exc:
                self.failures[choice.provider] = self.failures.get(choice.provider, 0) + 1
                self.health.record_failure(choice.provider, choice.model, exc)
                category = getattr(exc, "category", ErrorCategory.UNKNOWN)
                failures.append({
                    "provider": choice.provider, "model": choice.model,
                    "error": str(exc), "category": str(category),
                })
                # A failed attempt is still a fact worth recording: without it
                # an operator sees only the final error and cannot tell an
                # expired key from an unreachable host.
                self.database.add_model_usage(
                    provider=choice.provider, model=choice.model, route_mode=self.settings.model_mode,
                    input_tokens=None, output_tokens=None, cost_usd=None,
                    task_id=task_id, conversation_id=conversation_id,
                    metadata={"outcome": "failed", "error_category": str(category), "reason": choice.reason},
                )
        # Every real route failed. Raising keeps production honest: there is no
        # scripted provider in this chain to quietly fall back to.
        raise ModelError(
            "All model routes failed: " + " | ".join(f"{item['provider']}: {item['error']}" for item in failures),
            ErrorCategory(failures[-1]["category"]) if failures else ErrorCategory.UNKNOWN,
        )
