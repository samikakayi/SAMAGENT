from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .agent import AgentService
from .agent_api import AgentApi
from .api import (
    AppServices,
    register_conversation_routes,
    register_desktop_routes,
    register_oversight_routes,
    register_local_request_guard,
    register_provider_routes,
    register_settings_routes,
    register_system_routes,
    register_trading_routes,
    register_voice_routes,
    register_workflow_routes,
)
from .autonomy import AutonomousOrchestrator
from .cancellation import CancellationManager
from .capabilities import CapabilityRegistry
from .config import Settings
from .dpi import ensure_dpi_awareness
from .db import Database
from .models import AdapterRegistry
from .policy import RiskPolicy
from .project_map import ProjectScanner
from .provider_health import ProviderHealth
from .routing import ModelRouter
from .schemas import SettingsUpdate
from .secrets import SecretStore, resolve_credential
from .tasks import TaskStore
from .tools import ToolRegistry
from .trading.replay import BarReplayResearch
from .trading.service import TradingService
from .verification import VerificationEngine
from .voice import VoiceService
from .workflows import WorkflowIntelligence
from .windows_control import WindowsController


def create_app(settings: Settings | None = None, adapters: AdapterRegistry | None = None) -> FastAPI:
    ensure_dpi_awareness()
    settings = settings or Settings.from_env()
    settings.prepare()
    database = Database(settings.database_path)
    # Apply only the non-secret settings that were saved through the local UI.
    # The update schema is the one list of what may be persisted, so a setting
    # the API accepts cannot be silently lost on the next restart.
    for key, value in database.get_settings().items():
        if key in SettingsUpdate.model_fields:
            setattr(settings, key, value)
    policy = RiskPolicy(settings)
    cancellation = CancellationManager()
    trading = TradingService(settings, database, cancellation)
    windows = WindowsController(
        settings.data_dir,
        computer_control=settings.computer_control_enabled,
        screen_access=settings.screen_access_enabled,
    )
    # One scanner/verifier/registry is shared by the tools and the
    # orchestrator so a project is fingerprinted once per change, not once per
    # caller.
    scanner = ProjectScanner()
    verifier = VerificationEngine()
    capability_registry = CapabilityRegistry()
    tools = ToolRegistry(
        settings, database, trading=trading, windows=windows, cancellation=cancellation,
        scanner=scanner, verifier=verifier, capabilities=capability_registry,
    )
    # Credentials must be resolved BEFORE the adapters are built: an adapter
    # captures the key at construction, so building first leaves a stored
    # credential unused until something happens to rebuild the registry.
    secret_store = SecretStore(settings.data_dir)

    def apply_stored_credentials() -> None:
        """Environment wins; the local store fills in what it does not set."""
        for name, attribute in (
            ("openrouter_api_key", "openrouter_api_key"),
            ("openai_api_key", "openai_api_key"),
            ("litellm_api_key", "litellm_api_key"),
            ("groq_api_key", "groq_api_key"),
            ("gemini_api_key", "gemini_api_key"),
            ("n8n_api_key", "n8n_api_key"),
        ):
            value, _ = resolve_credential(name, secret_store)
            setattr(settings, attribute, value)

    apply_stored_credentials()
    adapters = adapters or AdapterRegistry(settings)
    provider_health = ProviderHealth(settings)
    router = ModelRouter(settings, adapters, database, provider_health)
    agent = AgentService(settings, database, tools, policy, adapters, router, trading, cancellation)
    task_store = TaskStore(database)
    orchestrator = AutonomousOrchestrator(
        settings, database, tools, policy, router,
        store=task_store, scanner=scanner, verifier=verifier, capabilities=capability_registry,
        cancellation=cancellation, health=provider_health,
    )
    agent_api = AgentApi(
        settings=settings, database=database, orchestrator=orchestrator, scanner=scanner,
        verifier=verifier, capabilities=capability_registry, cancellation=cancellation,
        adapters=lambda: application.state.adapters,
    )
    # The Sorani providers are configured by key, so the voice service resolves
    # them from the store on use rather than at construction.
    voice = VoiceService(settings, secret_store)
    replay = BarReplayResearch(trading.market_data, trading.tradingview)
    # Workflow Intelligence: SAM reasons and gates, n8n executes.
    workflow_intelligence = WorkflowIntelligence(settings)

    def rebuild_adapters() -> None:
        """Pick up a new credential without restarting the process."""
        refreshed = AdapterRegistry(settings)
        application.state.adapters = refreshed
        router.adapters = refreshed
        agent.adapters = refreshed
        router._health_cache.clear()
        router.failures.clear()
        provider_health.invalidate()

    monitor_stop = asyncio.Event()

    async def setup_monitor_worker() -> None:
        while not monitor_stop.is_set():
            try:
                await asyncio.wait_for(monitor_stop.wait(), timeout=max(3, settings.setup_monitor_interval_seconds))
            except TimeoutError:
                transitions = await asyncio.to_thread(trading.poll_monitors)
                for transition in transitions:
                    database.add_audit(
                        "setup_monitor",
                        "transition",
                        transition["reason"],
                        details={"setup_id": transition["setup"]["id"], "state": transition["setup"]["state"]},
                    )

    @asynccontextmanager
    async def lifespan(active_app: FastAPI):
        monitor_stop.clear()
        await agent_api.startup()
        active_app.state.monitor_task = asyncio.create_task(setup_monitor_worker())
        try:
            yield
        finally:
            monitor_stop.set()
            task = getattr(active_app.state, "monitor_task", None)
            if task:
                await task
            agent_api.shutdown()

    application = FastAPI(
        title="SAM Local Agent API",
        version="2.0.0",
        description="Local-first realtime Windows agent and deterministic TradingView trading brain.",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
        lifespan=lifespan,
    )
    application.state.settings = settings
    application.state.database = database
    application.state.policy = policy
    application.state.tools = tools
    application.state.adapters = adapters
    application.state.agent = agent
    application.state.router = router
    application.state.trading = trading
    application.state.workflows = workflow_intelligence
    application.state.windows = windows
    application.state.cancellation = cancellation
    application.state.voice = voice
    application.state.secrets = secret_store
    application.state.replay = replay
    application.state.orchestrator = orchestrator
    application.state.tasks = task_store
    application.state.scanner = scanner
    application.state.capabilities = capability_registry
    application.state.agent_api = agent_api
    application.include_router(agent_api.router)
    application.state.started_at = time.monotonic()

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Accept"],
    )

    register_local_request_guard(application, settings)

    services = AppServices(
        settings=settings, database=database, policy=policy, cancellation=cancellation,
        trading=trading, windows=windows, scanner=scanner, verifier=verifier,
        capabilities=capability_registry, tools=tools, secrets=secret_store,
        provider_health=provider_health, router=router, agent=agent,
        task_store=task_store, orchestrator=orchestrator, agent_api=agent_api,
        voice=voice, replay=replay,
        apply_stored_credentials=apply_stored_credentials,
        workflows=workflow_intelligence,
        rebuild_adapters=rebuild_adapters,
    )
    register_system_routes(application, services)
    register_conversation_routes(application, services)
    register_oversight_routes(application, services)
    register_settings_routes(application, services)
    register_workflow_routes(application, services)
    register_provider_routes(application, services)
    register_trading_routes(application, services)
    register_voice_routes(application, services)
    register_desktop_routes(application, services)


    frontend_candidates = [
        settings.project_root / "frontend" / "dist",
        settings.project_root / "frontend",
        settings.project_root / "static",
    ]
    frontend = next((candidate for candidate in frontend_candidates if (candidate / "index.html").exists()), None)
    if frontend:
        assets = frontend / "assets"
        if assets.exists():
            application.mount("/assets", StaticFiles(directory=assets), name="assets")

        @application.get("/")
        async def frontend_index():
            return FileResponse(frontend / "index.html", headers={"Cache-Control": "no-store"})

        application.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    else:
        @application.get("/")
        async def api_landing() -> dict[str, str]:
            return {"name": "SAM", "status": "backend ready", "docs": "/api/docs"}

    return application

