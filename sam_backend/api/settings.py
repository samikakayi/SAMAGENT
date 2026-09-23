"""Reading and changing the non-secret runtime configuration."""

from __future__ import annotations


from ..schemas import SettingsUpdate
from .services import AppServices
from fastapi import HTTPException
from typing import Any


def register_settings_routes(application: FastAPI, sv: AppServices) -> None:
    settings = sv.settings
    database = sv.database
    trading = sv.trading
    windows = sv.windows
    provider_health = sv.provider_health
    @application.get("/api/settings")
    async def get_settings() -> dict[str, Any]:
        return {"runtime": settings.public_dict(), "overrides": database.get_settings()}

    @application.put("/api/settings")
    async def update_settings(payload: SettingsUpdate) -> dict[str, Any]:
        values = payload.provided()
        if "fallback_enabled" in values or "fallback_model" in values:
            # Half the pair arrives at a time, so the rule is checked against
            # the settings the change would actually produce. An armed switch
            # with nothing behind it is the failure this prevents.
            if values.get("fallback_enabled", settings.fallback_enabled) and not values.get("fallback_model", settings.fallback_model):
                raise HTTPException(422, "fallback_enabled requires a fallback_model")
        # Renaming a model leaves cached verdicts answering a question nobody
        # is asking. Toggling the switch does not: resolve() reads it live.
        stale_verdicts = any(
            key in values and values[key] != getattr(settings, key)
            for key in ("default_provider", "default_model", "fallback_model")
        )
        for key, value in values.items():
            setattr(settings, key, value)
        if stale_verdicts:
            provider_health.invalidate()
        overrides = database.update_settings(values)
        trading.refresh_permissions()
        windows.refresh_permissions(
            computer_control=settings.computer_control_enabled,
            screen_access=settings.screen_access_enabled,
        )
        database.add_audit("settings", "updated", "Non-secret runtime settings updated", actor="user", details={"keys": sorted(values)})
        return {"runtime": settings.public_dict(), "overrides": overrides}
