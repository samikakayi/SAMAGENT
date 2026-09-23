"""Which models exist, which are reachable, and what they cost."""

from __future__ import annotations

import asyncio

from ..integrations import clear_metadata, record_metadata
from ..schemas import CredentialRequest
from ..secrets import ollama_status, openai_compatible_status, openrouter_status, start_ollama
from .services import AppServices
from fastapi import HTTPException
from typing import Any



MODEL_DISCOVERY_TIMEOUT_SECONDS = 3.0


async def _discover_provider_models(adapters: Any, provider: str) -> list[dict[str, Any]]:
    """Keep an offline provider from stalling local status and model discovery."""
    try:
        models = await asyncio.wait_for(
            adapters.get(provider).list_models(),
            timeout=MODEL_DISCOVERY_TIMEOUT_SECONDS,
        )
    except Exception:
        return []
    return models if isinstance(models, list) else []


def register_provider_routes(application: FastAPI, sv: AppServices) -> None:
    settings = sv.settings
    database = sv.database
    secret_store = sv.secrets
    router = sv.router
    voice = sv.voice
    apply_stored_credentials = sv.apply_stored_credentials
    rebuild_adapters = sv.rebuild_adapters
    @application.get("/api/models")
    async def models() -> dict[str, Any]:
        # Read the live registry once, so a rebuild mid-request cannot make
        # one half of the answer describe a different set of adapters.
        registry = application.state.adapters

        async def gather(provider: str):
            return provider, await _discover_provider_models(registry, provider)

        provider_names = getattr(registry, "providers", ["ollama", "openai"])
        pairs = await asyncio.gather(*(gather(provider) for provider in provider_names))
        providers = {provider: items for provider, items in pairs}
        return {
            "providers": providers,
            "models": [item for items in providers.values() for item in items],
            "defaults": {
                "provider": settings.default_provider,
                "model": settings.default_model,
                "mode": settings.model_mode,
                "openai_model": settings.openai_model,
                "litellm_fast": settings.litellm_fast_model,
                "litellm_strong": settings.litellm_strong_model,
                "openrouter_fast": settings.openrouter_fast_model,
                "openrouter_strong": settings.openrouter_strong_model,
            },
            "budget": router.budget_state(),
        }
    @application.get("/api/router/status")
    async def router_status() -> dict[str, Any]:
        registry = application.state.adapters
        ollama_available, litellm_available = await asyncio.gather(
            _discover_provider_models(registry, "ollama"),
            _discover_provider_models(registry, "litellm"),
        )
        return {
            "mode": settings.model_mode,
            "providers": getattr(registry, "providers", ["ollama", "openai"]),
            "configured": {
                "ollama": bool(ollama_available),
                "litellm": bool(litellm_available),
                "openrouter": bool(settings.openrouter_api_key),
                "openai": bool(settings.openai_api_key),
            },
            "failure_history": router.failures,
            "budget": router.budget_state(),
        }

    @application.get("/api/cost")
    async def cost_status() -> dict[str, Any]:
        return router.budget_state()
    @application.get("/api/providers/status")
    async def provider_status() -> dict[str, Any]:
        openrouter, ollama = await asyncio.gather(
            openrouter_status(
                settings.openrouter_base_url, settings.openrouter_api_key,
                {"X-Title": settings.openrouter_title, "HTTP-Referer": settings.openrouter_http_referer or ""},
            ),
            ollama_status(settings.ollama_base_url),
        )
        sorani = await asyncio.to_thread(voice.sorani_status)
        # Both are keyed OpenAI-compatible providers, so one helper classifies
        # them the same way and neither costs a completion to check.
        groq, gemini = await asyncio.gather(
            openai_compatible_status("groq", settings.groq_base_url, settings.groq_api_key),
            openai_compatible_status("gemini", settings.gemini_base_url, settings.gemini_api_key),
        )
        return {
            "openrouter": openrouter,
            "ollama": ollama,
            "groq": groq,
            "gemini": gemini,
            # What the next run may reach, and which candidates FREE refused.
            "routing": router.profile_state(),
            "litellm": {
                "provider": "litellm", "base_url": settings.litellm_base_url,
                "key_configured": bool(settings.litellm_api_key),
                "status": "CONNECTED" if await _discover_provider_models(application.state.adapters, "litellm") else "DOWN",
            },
            # States only: configured / connected / unconfigured / auth_failed /
            # rate_limited / error. No credential is ever part of this payload.
            "sorani": sorani,
            "credentials": secret_store.public_status(),
            "model_mode": settings.model_mode,
        }

    @application.post("/api/providers/credentials")
    async def configure_credential(payload: CredentialRequest) -> dict[str, Any]:
        try:
            stored = secret_store.set(payload.name, payload.value)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        apply_stored_credentials()
        rebuild_adapters()
        recorded: dict[str, Any] = {}
        if payload.metadata is not None:
            # Scope and expiry are what let the health panel say "this key can
            # do more than SAM uses" or "this expires in six days". They are
            # not secret, and they are kept apart from the value that is.
            recorded = record_metadata(database, payload.name,
                                       payload.metadata.model_dump(exclude_none=True))
        sv.integrations.invalidate()
        database.add_audit(
            "credentials", "success", f"Stored {payload.name} in the local secret store",
            actor="user", details={"name": payload.name, "fingerprint": stored["fingerprint"],
                                   "metadata_recorded": sorted(recorded)},
        )
        # The response carries presence and fingerprint only, never the value.
        status = await openrouter_status(
            settings.openrouter_base_url, settings.openrouter_api_key,
            {"X-Title": settings.openrouter_title},
        ) if payload.name == "openrouter_api_key" else None
        return {"stored": True, "name": payload.name, "fingerprint": stored["fingerprint"],
                "reloaded_without_restart": True, "health": status}

    @application.delete("/api/providers/credentials/{name}")
    async def clear_credential(name: str) -> dict[str, Any]:
        removed = secret_store.clear(name)
        apply_stored_credentials()
        rebuild_adapters()
        # Scope and expiry described the key that just went away; leaving them
        # behind would let the health panel describe a credential nobody has.
        clear_metadata(database, name)
        sv.integrations.invalidate()
        database.add_audit("credentials", "success" if removed else "failed",
                           f"Cleared {name}", actor="user", details={"name": name})
        return {"cleared": removed, "name": name, "credentials": secret_store.public_status()}

    @application.post("/api/providers/ollama/start")
    async def start_local_ollama() -> dict[str, Any]:
        current = await ollama_status(settings.ollama_base_url)
        if current["status"] == "CONNECTED":
            return {"already_running": True, **current}
        launch = await asyncio.to_thread(start_ollama, settings.project_root)
        if not launch.get("started"):
            return {"already_running": False, "started": False, **launch, **current}
        for _ in range(20):
            await asyncio.sleep(0.75)
            current = await ollama_status(settings.ollama_base_url)
            if current["status"] in {"CONNECTED", "NO_MODELS"}:
                break
        database.add_audit("providers", "success", "Started the local Ollama daemon", actor="user")
        return {"already_running": False, "started": True, **current}
