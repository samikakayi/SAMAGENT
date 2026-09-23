"""What the Integration Health panel can ask for.

Read-only, and cheap on purpose: the default answer is whatever was cached,
and re-probing is something the operator asks for by pressing Refresh. A panel
that quietly re-checked four providers on every render would spend a free tier
to look busy.

No route here can return a credential. The health of a key and the value of a
key are different facts, and only the first one is anybody's business here.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from .services import AppServices


def register_integration_routes(application: FastAPI, sv: AppServices) -> None:
    integrations = sv.integrations

    @application.get("/api/integrations")
    async def integration_health(refresh: bool = False) -> dict[str, Any]:
        return await integrations.summary(refresh=refresh)
