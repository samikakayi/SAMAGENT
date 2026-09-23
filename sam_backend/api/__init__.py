"""The HTTP surface, grouped by what each group is responsible for.

create_app builds the services; these modules say what the web exposes of
them. Every group is registered the same way -- register(application,
services) -- and takes its dependencies from one typed object rather than
reaching for module state, so what a group touches is visible at its top.
"""

from __future__ import annotations

from .conversations import register_conversation_routes
from .desktop import register_desktop_routes
from .oversight import register_oversight_routes
from .providers import register_provider_routes
from .security import register_local_request_guard
from .services import AppServices
from .settings import register_settings_routes
from .system import register_system_routes
from .trading import register_trading_routes
from .voice import register_voice_routes
from .workflows import register_workflow_routes

__all__ = [
    "AppServices",
    "register_conversation_routes",
    "register_desktop_routes",
    "register_oversight_routes",
    "register_local_request_guard",
    "register_provider_routes",
    "register_settings_routes",
    "register_system_routes",
    "register_trading_routes",
    "register_voice_routes",
    "register_workflow_routes",
]
