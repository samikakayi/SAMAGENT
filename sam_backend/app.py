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
    register_integration_routes,
    register_n8n_runtime_routes,
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
from .capability_brief import CapabilityBrief
from .config import Settings
from .dpi import ensure_dpi_awareness
from .integrations import IntegrationHealth
from .n8n_runtime import ManagedN8nRuntime
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
from .voice_session import VoiceConversationController
from .workflows import WorkflowIntelligence
from .windows_control import WindowsController


def import_numpy_before_any_thread() -> None:
    """Import numpy once, here, before anything can import it concurrently.

    The SAM started at 03:16:58 came up with a dead wake listener ("cannot
    import name '__cpu_features__' from partially initialized module
    'numpy._core._multiarray_umath'") and a /api/trading/status that answered
    500 (MetaTrader5 missing `shutdown`), until a restart. numpy's first import
    had been taken by several threads at once, by two routes: faster_whisper on
    the wake warm thread through `import numpy`, and MetaTrader5 on a request
    thread through its C `import_array()`, which imports
    `numpy._core._multiarray_umath` directly. Module locks are taken child
    first, so the two routes lock numpy's modules in opposite orders; the
    import system breaks that deadlock by handing one thread a half-built
    module, and numpy's core refuses to initialise twice ("cannot load module
    more than once per process"), so numpy stays broken for the whole process.

    Measured in fresh processes: those two imports started 0-0.4 s apart failed
    1 run in 40 with exactly those errors, and four threads entering numpy by
    different submodules at once failed 30 in 30; with numpy imported first,
    0 in 190. Once numpy is whole in sys.modules, every later route finds it
    there and takes no lock, so numpy is the one import this needs: six
    MetaTrader5 imports racing only each other failed 0 in 30. It costs about
    0.16 s of startup. Nothing heavier is imported for it, since most installs
    never use the local voice or a broker.
    """
    try:
        import numpy  # noqa: F401
    except ImportError:
        # A requirement, but its absence is for the features that use it to
        # report, not a reason for the server not to start.
        pass


def create_app(settings: Settings | None = None, adapters: AdapterRegistry | None = None) -> FastAPI:
    # First, on the thread that builds the app: the lifespan, the wake
    # listener and every request thread come after.
    import_numpy_before_any_thread()
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
    # Hands-free voice drives the ordinary agent: it captures speech and
    # hands the text to the same `agent.chat` the typed box uses, so a
    # spoken request gets no authority a typed one would not have.
    voice_session = VoiceConversationController(settings, voice, agent)
    # The chat model is told what SAM can do from live state -- the voice keys,
    # the wake listener, the desktop switches -- so it cannot deny having a
    # voice it has. Attached here because those services exist only now; the
    # orchestrator behind Autopilot is always built above.
    agent.capability_brief = CapabilityBrief(
        settings, voice=voice, voice_session=voice_session, trading=trading, autonomy=True,
    )
    replay = BarReplayResearch(trading.market_data, trading.tradingview)
    # Workflow Intelligence: SAM reasons and gates, n8n executes.
    workflow_intelligence = WorkflowIntelligence(settings, router=router)

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
        # The wake listener lives and dies with the application -- no service,
        # no scheduled task -- and only starts when the user has asked for it.
        if settings.hands_free_enabled:
            await asyncio.to_thread(voice_session.start)
        try:
            yield
        finally:
            await asyncio.to_thread(voice_session.stop)
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
    application.state.voice_session = voice_session
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

    # One health view over every outside connection, sharing the secret store
    # and the n8n client that already exist rather than opening its own.
    integrations = IntegrationHealth(settings, secret_store, database, workflow_intelligence)
    # The one n8n SAM installed. It knows a single location and a single
    # command; nothing a caller sends can change either.
    n8n_runtime = ManagedN8nRuntime(database)
    application.state.integrations = integrations

    services = AppServices(
        settings=settings, database=database, policy=policy, cancellation=cancellation,
        trading=trading, windows=windows, scanner=scanner, verifier=verifier,
        capabilities=capability_registry, tools=tools, secrets=secret_store,
        provider_health=provider_health, integrations=integrations,
        n8n_runtime=n8n_runtime, router=router, agent=agent,
        task_store=task_store, orchestrator=orchestrator, agent_api=agent_api,
        voice=voice, voice_session=voice_session, replay=replay,
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
    register_integration_routes(application, services)
    register_n8n_runtime_routes(application, services)
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

