from __future__ import annotations

import asyncio
import importlib.util
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from .agent import AgentService
from .cancellation import CancellationManager
from .config import Settings, is_elevated_windows_process
from .contracts import ExecutionStatus
from .dpi import ensure_dpi_awareness, status as dpi_status
from .db import Database
from .models import AdapterRegistry
from .policy import RiskPolicy
from .routing import ModelRouter
from .schemas import (
    ApprovalDecision,
    BacktestRequest,
    ComposedStrategyRequest,
    CredentialRequest,
    ChartLayerRequest,
    ChatRequest,
    ClearDrawingsRequest,
    ConversationCreate,
    ConversationUpdate,
    CustomTheoryCreate,
    DrawAnalysisRequest,
    DrawLevelRequest,
    GannDrawRequest,
    JournalCreate,
    MarketSnapshotRequest,
    MemoryCreate,
    PitchforkDrawRequest,
    ReplayControlRequest,
    ReplayScanRequest,
    ReplayStartRequest,
    SettingsUpdate,
    SetupCreateRequest,
    SetupMonitorRequest,
    TradingAnalysisRequest,
    TradingViewActionRequest,
    TwoAnchorDrawRequest,
    VoiceListenRequest,
    VoiceSpeakRequest,
)
from .tools import ToolRegistry
from . import sorani as sorani_speech
from .secrets import SecretStore, ollama_status, openrouter_status, resolve_credential, start_ollama
from .trading.replay import BarReplayResearch
from .voice import VoiceService
from .trading.service import TradingService
from .windows_control import WindowsController


MODEL_DISCOVERY_TIMEOUT_SECONDS = 3.0
VOICE_UPLOAD_MAX_BYTES = 8 * 1024 * 1024


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


def _composed_conditions(payload: Any) -> list[dict[str, Any]]:
    """Map a composed strategy onto the executable predicates SAM supports.

    Only phrases that correspond to a real predicate become conditions; free text
    is preserved as documentation rather than pretending to be executable.
    """
    lowered = " ".join(
        str(part).lower() for part in (payload.context, payload.setup, payload.confirmation, payload.entry_trigger)
    )
    conditions: list[dict[str, Any]] = []
    timeframes = payload.timeframes or ["H1", "M15", "M5"]
    higher = timeframes[0]
    lower = timeframes[-1]
    if payload.direction in {"LONG", "BOTH"} and "demand" in lowered or "bullish" in lowered:
        conditions.append({"predicate": "trend_is", "value": "BULLISH", "timeframe": higher})
    elif payload.direction == "SHORT" or "supply" in lowered or "bearish" in lowered:
        conditions.append({"predicate": "trend_is", "value": "BEARISH", "timeframe": higher})
    if "sweep" in lowered or "liquidity" in lowered:
        conditions.append({"predicate": "has_liquidity_sweep", "timeframe": timeframes[min(1, len(timeframes) - 1)]})
    if "mss" in lowered or "choch" in lowered or "shift" in lowered:
        conditions.append({"predicate": "has_mss", "timeframe": lower})
    if "bos" in lowered or "break" in lowered:
        conditions.append({"predicate": "has_bos", "timeframe": lower})
    if "fvg" in lowered or "imbalance" in lowered or "gap" in lowered:
        conditions.append({"predicate": "has_active_fvg", "timeframe": lower})
    if not conditions:
        # A strategy must be executable to be saved at all.
        conditions.append({"predicate": "trend_is", "value": "BULLISH" if payload.direction != "SHORT" else "BEARISH",
                           "timeframe": higher})
    return conditions


def create_app(settings: Settings | None = None, adapters: AdapterRegistry | None = None) -> FastAPI:
    ensure_dpi_awareness()
    settings = settings or Settings.from_env()
    settings.prepare()
    database = Database(settings.database_path)
    # Apply only the non-secret settings that were saved through the local UI.
    persisted_settings = database.get_settings()
    for key in {
        "default_provider", "default_model", "model_mode", "openai_model", "permission_mode",
        "openrouter_fast_model", "openrouter_strong_model", "openrouter_vision_model",
        "max_tool_iterations", "command_timeout_seconds", "daily_budget_usd", "monthly_budget_usd",
        "computer_control_enabled", "screen_access_enabled", "default_trading_theory", "minimum_rr",
        "voice_mode", "voice_language", "voice_vad_threshold", "voice_silence_ms", "voice_wake_word",
    }:
        if key in persisted_settings:
            setattr(settings, key, persisted_settings[key])
    policy = RiskPolicy(settings)
    cancellation = CancellationManager()
    trading = TradingService(settings, database, cancellation)
    windows = WindowsController(
        settings.data_dir,
        computer_control=settings.computer_control_enabled,
        screen_access=settings.screen_access_enabled,
    )
    tools = ToolRegistry(settings, database, trading=trading, windows=windows, cancellation=cancellation)
    adapters = adapters or AdapterRegistry(settings)
    router = ModelRouter(settings, adapters, database)
    agent = AgentService(settings, database, tools, policy, adapters, router, trading, cancellation)
    secret_store = SecretStore(settings.data_dir)
    # The Sorani providers are configured by key, so the voice service resolves
    # them from the store on use rather than at construction.
    voice = VoiceService(settings, secret_store)
    replay = BarReplayResearch(trading.market_data, trading.tradingview)

    def apply_stored_credentials() -> None:
        """Environment wins; the local store fills in what it does not set."""
        for name, attribute in (
            ("openrouter_api_key", "openrouter_api_key"),
            ("openai_api_key", "openai_api_key"),
            ("litellm_api_key", "litellm_api_key"),
        ):
            value, _ = resolve_credential(name, secret_store)
            setattr(settings, attribute, value)

    def rebuild_adapters() -> None:
        """Pick up a new credential without restarting the process."""
        refreshed = AdapterRegistry(settings)
        application.state.adapters = refreshed
        router.adapters = refreshed
        agent.adapters = refreshed
        router._health_cache.clear()
        router.failures.clear()

    apply_stored_credentials()

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
        active_app.state.monitor_task = asyncio.create_task(setup_monitor_worker())
        try:
            yield
        finally:
            monitor_stop.set()
            task = getattr(active_app.state, "monitor_task", None)
            if task:
                await task

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
    application.state.windows = windows
    application.state.cancellation = cancellation
    application.state.voice = voice
    application.state.secrets = secret_store
    application.state.replay = replay
    application.state.started_at = time.monotonic()

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Accept"],
    )

    @application.middleware("http")
    async def local_request_guard(request: Request, call_next):
        host_header = request.headers.get("host", "").split(":", 1)[0].strip("[]").lower()
        client_host = (request.client.host if request.client else "").lower()
        allowed_hosts = {"127.0.0.1", "localhost", "::1", "testserver", "testclient"}
        if host_header and host_header not in allowed_hosts:
            return JSONResponse({"detail": "SAM only accepts loopback requests."}, status_code=403)
        if client_host and client_host not in allowed_hosts:
            return JSONResponse({"detail": "SAM only accepts local clients."}, status_code=403)
        origin = request.headers.get("origin")
        if origin and origin not in settings.cors_origins:
            return JSONResponse({"detail": "Origin is not allowed."}, status_code=403)
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Permissions-Policy", "microphone=(self), camera=(), geolocation=()")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; connect-src 'self'; media-src 'self' blob:; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
        )
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    @application.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "name": "SAM",
            "version": "2.0.0",
            "uptime_seconds": round(time.monotonic() - application.state.started_at, 2),
            "workspace": str(settings.workspace_root),
            "database": "ok",
            "audit_chain_valid": database.verify_audit_chain(),
            "running_elevated": is_elevated_windows_process(),
            "screen_coordinates": dpi_status(),
            "model_mode": settings.model_mode,
            "computer_control": settings.computer_control_enabled,
            "screen_access": settings.screen_access_enabled,
            "active_tasks": cancellation.snapshot(),
        }

    @application.get("/api/config")
    async def public_config() -> dict[str, Any]:
        return {
            **settings.public_dict(),
            "name": "SAM",
            "capabilities": {
                "chat": True, "voice_input": "browser-realtime-vad", "voice_output": "browser-streamed-chunks", "persistent_memory": True,
                "files": True, "terminal": True, "python": True, "browser": True, "apps": True,
                "planning": True, "approvals": True, "audit": True, "trading_brain": True,
                "metatrader5_read_only": importlib.util.find_spec("MetaTrader5") is not None,
                "tradingview_observer": True, "emergency_stop": True, "model_router": True,
                "openrouter": bool(settings.openrouter_api_key), "litellm_gateway": True,
            },
            "security": {
                "loopback_only": True, "workspace_containment": True, "single_use_approvals": True,
                "audit_hash_chain": True, "safe_mode": not settings.allow_unsafe_system_actions,
            },
        }

    @application.get("/api/capabilities")
    async def capabilities() -> dict[str, Any]:
        config = await public_config()
        return {"capabilities": config["capabilities"], "security": config["security"]}

    @application.get("/api/models")
    async def models() -> dict[str, Any]:
        async def gather(provider: str):
            return provider, await _discover_provider_models(adapters, provider)

        provider_names = getattr(adapters, "providers", ["ollama", "openai"])
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

    @application.get("/api/conversations")
    async def list_conversations(limit: int = Query(100, ge=1, le=500)) -> dict[str, Any]:
        return {"conversations": database.list_conversations(limit)}

    @application.post("/api/conversations", status_code=201)
    async def create_conversation(payload: ConversationCreate) -> dict[str, Any]:
        default_models = {
            "auto": "auto",
            "ollama": settings.default_model,
            "litellm": settings.litellm_fast_model,
            "openrouter": (
                settings.default_model
                if "/" in str(settings.default_model or "") and not str(settings.default_model).lower().startswith("qwen")
                else settings.openrouter_fast_model
            ),
            "openai": settings.openai_model,
        }
        model = payload.model or default_models[payload.provider]
        conversation = database.create_conversation(payload.title, payload.provider, model)
        database.add_audit("conversation", "created", "Conversation created", actor="user", conversation_id=conversation["id"])
        return {"conversation": conversation}

    @application.get("/api/conversations/{conversation_id}")
    async def get_conversation(conversation_id: str) -> dict[str, Any]:
        conversation = database.get_conversation(conversation_id)
        if conversation is None:
            raise HTTPException(404, "Conversation not found")
        return {"conversation": conversation, "messages": database.list_messages(conversation_id)}

    @application.patch("/api/conversations/{conversation_id}")
    async def update_conversation(conversation_id: str, payload: ConversationUpdate) -> dict[str, Any]:
        conversation = database.update_conversation_title(conversation_id, payload.title)
        if conversation is None:
            raise HTTPException(404, "Conversation not found")
        database.add_audit(
            "conversation",
            "renamed",
            "Conversation renamed",
            actor="user",
            conversation_id=conversation_id,
            details={"conversation_id": conversation_id, "title_characters": len(conversation["title"])},
        )
        return {"conversation": conversation}

    @application.get("/api/conversations/{conversation_id}/messages")
    async def conversation_messages(conversation_id: str, limit: int = Query(200, ge=1, le=1000)) -> dict[str, Any]:
        if database.get_conversation(conversation_id) is None:
            raise HTTPException(404, "Conversation not found")
        return {"messages": database.list_messages(conversation_id, limit)}

    @application.delete("/api/conversations/{conversation_id}")
    async def delete_conversation(conversation_id: str) -> dict[str, Any]:
        if not database.delete_conversation(conversation_id):
            raise HTTPException(404, "Conversation not found")
        database.add_audit("conversation", "deleted", "Conversation and its messages deleted", actor="user", details={"conversation_id": conversation_id})
        return {"deleted": True, "id": conversation_id}

    @application.post("/api/chat")
    async def chat(payload: ChatRequest) -> dict[str, Any]:
        try:
            return await agent.chat(
                payload.message, conversation_id=payload.conversation_id, provider=payload.provider, model=payload.model,
            )
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @application.post("/api/chat/stream")
    async def chat_stream(payload: ChatRequest) -> StreamingResponse:
        async def events():
            yield "event: status\ndata: " + json.dumps({"status": "thinking"}) + "\n\n"
            try:
                result = await agent.chat(
                    payload.message, conversation_id=payload.conversation_id, provider=payload.provider, model=payload.model,
                )
                yield "event: result\ndata: " + json.dumps(result, ensure_ascii=False, default=str) + "\n\n"
            except Exception as exc:
                yield "event: error\ndata: " + json.dumps({"error": str(exc)}) + "\n\n"
        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @application.websocket("/ws/chat")
    async def chat_socket(websocket: WebSocket):
        origin = websocket.headers.get("origin")
        client_host = (websocket.client.host if websocket.client else "").lower()
        if (origin and origin not in settings.cors_origins) or client_host not in {"127.0.0.1", "::1", "testclient"}:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        try:
            while True:
                payload = ChatRequest.model_validate(await websocket.receive_json())
                await websocket.send_json({"type": "status", "status": "thinking"})
                result = await agent.chat(
                    payload.message, conversation_id=payload.conversation_id, provider=payload.provider, model=payload.model,
                )
                await websocket.send_json({"type": "result", **result})
        except WebSocketDisconnect:
            return
        except Exception as exc:
            await websocket.send_json({"type": "error", "error": str(exc)})

    @application.get("/api/approvals")
    async def list_approvals(status: str | None = Query(default=None), limit: int = Query(100, ge=1, le=500)) -> dict[str, Any]:
        records = database.list_approvals(status, limit)
        return {"approvals": [agent.public_approval(record) for record in records]}

    @application.get("/api/approvals/{approval_id}")
    async def get_approval(approval_id: str) -> dict[str, Any]:
        record = database.get_approval(approval_id)
        if record is None:
            raise HTTPException(404, "Approval not found")
        return {"approval": agent.public_approval(record)}

    @application.post("/api/approvals/{approval_id}/decision")
    async def decide_approval(approval_id: str, payload: ApprovalDecision) -> dict[str, Any]:
        try:
            result = await agent.resolve_approval(approval_id, payload.decision, payload.note)
            record = database.get_approval(approval_id)
            return {"approval": agent.public_approval(record or {}), "agent_response": result}
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @application.get("/api/audit")
    async def audit(limit: int = Query(200, ge=1, le=1000), event_type: str | None = None) -> dict[str, Any]:
        return {"entries": database.list_audit(limit, event_type), "chain_valid": database.verify_audit_chain()}

    @application.get("/api/audit/verify")
    async def verify_audit() -> dict[str, Any]:
        return {"valid": database.verify_audit_chain()}

    @application.get("/api/memories")
    async def memories(query: str = "", limit: int = Query(20, ge=1, le=100)) -> dict[str, Any]:
        return {"memories": database.list_memories(query, limit)}

    @application.post("/api/memories", status_code=201)
    async def create_memory(payload: MemoryCreate) -> dict[str, Any]:
        if policy.contains_embedded_secret({"content": payload.content}):
            raise HTTPException(400, "SAM will not store likely credentials in durable memory")
        memory = database.add_memory(payload.content, payload.tags, payload.importance, payload.source_conversation_id, payload.domain)
        database.add_audit("memory", "created", "Local memory created", actor="user", details={"memory_id": memory["id"], "tags": payload.tags})
        return {"memory": memory}

    @application.delete("/api/memories/{memory_id}")
    async def delete_memory(memory_id: str) -> dict[str, Any]:
        if not database.delete_memory(memory_id):
            raise HTTPException(404, "Memory not found")
        database.add_audit("memory", "deleted", "Local memory deleted", actor="user", details={"memory_id": memory_id})
        return {"deleted": True, "id": memory_id}

    @application.get("/api/workspace/tree")
    async def workspace_tree(path: str = ".", max_depth: int = Query(2, ge=0, le=8)) -> dict[str, Any]:
        decision = policy.evaluate("list_files", {"path": path, "max_depth": max_depth})
        if not decision.allowed or decision.approval_required:
            raise HTTPException(403, decision.reason)
        result = await asyncio.to_thread(tools.execute, "list_files", {"path": path, "max_depth": max_depth}, approved=False)
        if not result.ok:
            raise HTTPException(400, result.error)
        return result.output

    @application.get("/api/settings")
    async def get_settings() -> dict[str, Any]:
        return {"runtime": settings.public_dict(), "overrides": database.get_settings()}

    @application.put("/api/settings")
    async def update_settings(payload: SettingsUpdate) -> dict[str, Any]:
        values = payload.provided()
        for key, value in values.items():
            setattr(settings, key, value)
        overrides = database.update_settings(values)
        trading.refresh_permissions()
        windows.refresh_permissions(
            computer_control=settings.computer_control_enabled,
            screen_access=settings.screen_access_enabled,
        )
        database.add_audit("settings", "updated", "Non-secret runtime settings updated", actor="user", details={"keys": sorted(values)})
        return {"runtime": settings.public_dict(), "overrides": overrides}


    @application.get("/api/tools/manifests")
    async def tool_manifests() -> dict[str, Any]:
        return {"tools": tools.manifests}

    @application.get("/api/router/status")
    async def router_status() -> dict[str, Any]:
        ollama_available, litellm_available = await asyncio.gather(
            _discover_provider_models(adapters, "ollama"),
            _discover_provider_models(adapters, "litellm"),
        )
        return {
            "mode": settings.model_mode,
            "providers": getattr(adapters, "providers", ["ollama", "openai"]),
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

    @application.get("/api/trading/status")
    async def trading_status(symbol: str = "XAUUSD") -> dict[str, Any]:
        return await asyncio.to_thread(trading.status, symbol)

    @application.get("/api/trading/capabilities")
    async def trading_capabilities(symbol: str = "XAUUSD") -> dict[str, Any]:
        return await asyncio.to_thread(trading.market_data.providers["metatrader5"].capabilities, symbol)

    @application.post("/api/trading/snapshot")
    async def trading_snapshot(payload: MarketSnapshotRequest) -> dict[str, Any]:
        return (await asyncio.to_thread(trading.market_snapshot, payload.symbol, payload.timeframes)).as_dict()

    @application.post("/api/trading/analyze")
    async def trading_analyze(payload: TradingAnalysisRequest) -> dict[str, Any]:
        token = cancellation.create()
        try:
            result = await asyncio.to_thread(
                trading.analyze,
                payload.symbol,
                payload.timeframes,
                payload.theories,
                count=payload.count,
                minimum_rr=payload.minimum_rr,
                task_id=token.task_id,
            )
            response = result.as_dict()
            response["task_id"] = token.task_id
            return response
        except RuntimeError as exc:
            if token.cancelled:
                return {"status": "CANCELLED", "executed": True, "verified": False, "data": None, "error": str(exc), "task_id": token.task_id}
            raise HTTPException(500, str(exc)) from exc
        finally:
            cancellation.complete(token.task_id)

    @application.get("/api/trading/theories")
    async def trading_theories() -> dict[str, Any]:
        return {"built_in": trading.knowledge.list(), "custom": database.list_custom_theories()}

    @application.post("/api/trading/theories/custom", status_code=201)
    async def save_theory(payload: CustomTheoryCreate) -> dict[str, Any]:
        if policy.contains_embedded_secret(payload.definition):
            raise HTTPException(400, "Custom theories cannot contain credentials")
        try:
            theory = trading.save_custom_theory(payload.definition)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        database.add_audit("custom_theory", "created", "Versioned custom theory saved", actor="user", details={"theory_id": theory["id"], "name": theory["name"], "version": theory["version"]})
        return {"theory": theory}

    @application.get("/api/trading/skills")
    async def trading_skills() -> dict[str, Any]:
        return {"skills": trading.skills.list()}

    @application.get("/api/trading/context")
    async def trading_context() -> dict[str, Any]:
        return {"context": database.get_trading_context()}

    @application.get("/api/trading/setups")
    async def trading_setups(limit: int = Query(100, ge=1, le=500)) -> dict[str, Any]:
        return {"setups": database.list_trading_setups(limit)}

    @application.post("/api/trading/setups", status_code=201)
    async def create_trading_setup(payload: SetupCreateRequest) -> dict[str, Any]:
        try:
            setup = trading.create_setup_from_last_analysis(payload.theory)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        database.add_audit("trading_setup", "created", "Setup saved from the latest verified analysis", actor="user", details={"setup_id": setup["id"], "theory": payload.theory})
        return {"setup": setup}

    @application.get("/api/trading/setups/{setup_id}")
    async def trading_setup(setup_id: str) -> dict[str, Any]:
        setup = database.get_trading_setup(setup_id)
        if setup is None:
            raise HTTPException(404, "Setup not found")
        return {"setup": setup, "events": database.list_setup_events(setup_id)}

    @application.post("/api/trading/setups/{setup_id}/monitor")
    async def monitor_setup(setup_id: str, payload: SetupMonitorRequest) -> dict[str, Any]:
        setup = database.set_setup_monitoring(setup_id, payload.enabled)
        if setup is None:
            raise HTTPException(404, "Setup not found")
        database.add_audit("setup_monitor", "started" if payload.enabled else "stopped", "Setup monitoring changed", actor="user", details={"setup_id": setup_id, "enabled": payload.enabled})
        return {"setup": setup}

    @application.get("/api/trading/journal")
    async def journal(query: str = "", limit: int = Query(100, ge=1, le=500)) -> dict[str, Any]:
        return {"entries": database.list_trading_journal(query, limit)}

    @application.post("/api/trading/journal", status_code=201)
    async def create_journal_entry(payload: JournalCreate) -> dict[str, Any]:
        return {"entry": database.add_trading_journal(payload.symbol, payload.theory, payload.payload, payload.setup_id)}

    @application.get("/api/tradingview/state")
    async def tradingview_state() -> dict[str, Any]:
        return trading.tradingview.observe().as_dict()

    @application.post("/api/tradingview/action")
    async def tradingview_action(payload: TradingViewActionRequest) -> dict[str, Any]:
        trading.refresh_permissions()
        if payload.action in {"launch", "focus"}:
            result = await asyncio.to_thread(
                trading.tradingview.launch if payload.action == "launch" else trading.tradingview.focus
            )
        elif payload.action == "set_symbol":
            if not payload.symbol:
                raise HTTPException(400, "symbol is required")
            result = await asyncio.to_thread(trading.tradingview.set_symbol, payload.symbol)
        elif payload.action == "set_timeframe":
            if not payload.timeframe:
                raise HTTPException(400, "timeframe is required")
            result = await asyncio.to_thread(trading.tradingview.set_timeframe, payload.timeframe)
        elif payload.action == "capture":
            result = await asyncio.to_thread(trading.tradingview.capture)
        elif payload.action == "auto_calibrate":
            # Screen capture and OCR are blocking; they must never run on the loop.
            result = await asyncio.to_thread(trading.calibrate_chart)
        elif payload.action == "verify_calibration":
            result = await asyncio.to_thread(trading.verify_calibration)
        else:
            anchors = (payload.price_a, payload.y_a, payload.price_b, payload.y_b)
            if any(value is None for value in anchors):
                raise HTTPException(400, "price_a, y_a, price_b, and y_b are required")
            result = trading.tradingview.calibrate(*anchors)  # type: ignore[arg-type]
        database.add_audit("tradingview", result.status.value.lower(), f"TradingView {payload.action}", actor="user", details={"verified": result.verified, "error_code": result.error_code})
        return result.as_dict()

    @application.get("/api/tradingview/drawings")
    async def list_drawings(
        symbol: str | None = None,
        layer: str | None = None,
        theory: str | None = None,
        setup_id: str | None = None,
        visible_only: bool = False,
    ) -> dict[str, Any]:
        result = await asyncio.to_thread(
            trading.list_drawings, symbol=symbol, layer=layer, theory=theory,
            setup_id=setup_id, visible_only=visible_only,
        )
        return result.as_dict()

    @application.post("/api/tradingview/drawings")
    async def draw_level(payload: DrawLevelRequest) -> dict[str, Any]:
        trading.refresh_permissions()
        result = await asyncio.to_thread(
            trading.draw_annotation, payload.annotation, payload.price,
            label=payload.label, theory=payload.theory, setup_id=payload.setup_id,
            layer=payload.layer, symbol=payload.symbol,
        )
        database.add_audit(
            "tradingview", result.status.value.lower(), f"Draw {payload.annotation} at {payload.price}",
            actor="user", details={"verified": result.verified, "error_code": result.error_code},
        )
        return result.as_dict()

    @application.post("/api/tradingview/drawings/analysis")
    async def draw_analysis(payload: DrawAnalysisRequest) -> dict[str, Any]:
        trading.refresh_permissions()
        result = await asyncio.to_thread(trading.draw_analysis, None, theory=payload.theory, setup_id=payload.setup_id)
        database.add_audit(
            "tradingview", result.status.value.lower(), "Draw analysis plan",
            actor="user", details={"verified": result.verified, "error_code": result.error_code},
        )
        return result.as_dict()

    @application.post("/api/tradingview/drawings/clear")
    async def clear_drawings(payload: ClearDrawingsRequest) -> dict[str, Any]:
        trading.refresh_permissions()
        result = await asyncio.to_thread(
            trading.clear_drawings, symbol=payload.symbol, layer=payload.layer,
            theory=payload.theory, setup_id=payload.setup_id, all_owned=payload.all_owned,
        )
        database.add_audit(
            "tradingview", result.status.value.lower(), "Clear SAM drawings",
            actor="user", details={"verified": result.verified, "error_code": result.error_code},
        )
        return result.as_dict()

    @application.post("/api/tradingview/layers")
    async def set_chart_layer(payload: ChartLayerRequest) -> dict[str, Any]:
        result = await asyncio.to_thread(trading.set_layer_visibility, payload.layer, payload.visible, payload.symbol)
        return result.as_dict()

    @application.get("/api/trading/triggers")
    async def list_triggers() -> dict[str, Any]:
        return (await asyncio.to_thread(trading.list_entry_triggers)).as_dict()

    @application.post("/api/trading/backtest")
    async def run_backtest(payload: BacktestRequest) -> dict[str, Any]:
        result = await asyncio.to_thread(
            trading.backtest, payload.symbol, payload.timeframe,
            trigger=payload.trigger, count=payload.count,
            stop_atr_multiple=payload.stop_atr_multiple,
            reward_multiple=payload.reward_multiple, max_bars=payload.max_bars,
        )
        database.add_audit(
            "research", result.status.value.lower(),
            f"Backtest {payload.trigger or 'all triggers'} on {payload.symbol} {payload.timeframe}",
            actor="user", details={"verified": result.verified, "error_code": result.error_code},
        )
        return result.as_dict()

    @application.post("/api/tradingview/drawings/object")
    async def draw_object(payload: TwoAnchorDrawRequest) -> dict[str, Any]:
        trading.refresh_permissions()
        result = await asyncio.to_thread(
            trading.draw_two_anchor, payload.annotation,
            payload.price_a, payload.minutes_a, payload.price_b, payload.minutes_b,
            label=payload.label, theory=payload.theory, setup_id=payload.setup_id, layer=payload.layer,
        )
        database.add_audit(
            "tradingview", result.status.value.lower(), f"Draw {payload.annotation} (two anchors)",
            actor="user", details={"verified": result.verified, "error_code": result.error_code},
        )
        return result.as_dict()

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
        return {
            "openrouter": openrouter,
            "ollama": ollama,
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
        database.add_audit(
            "credentials", "success", f"Stored {payload.name} in the local secret store",
            actor="user", details={"name": payload.name, "fingerprint": stored["fingerprint"]},
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

    @application.get("/api/trading/gann")
    async def gann_analysis(symbol: str = "XAUUSD", timeframe: str = "M15") -> dict[str, Any]:
        return (await asyncio.to_thread(trading.gann_analysis, symbol, timeframe)).as_dict()

    @application.post("/api/tradingview/drawings/gann")
    async def draw_gann(payload: GannDrawRequest) -> dict[str, Any]:
        trading.refresh_permissions()
        result = await asyncio.to_thread(
            trading.draw_gann_fan, payload.symbol, payload.timeframe,
            max_rays=payload.max_rays, setup_id=payload.setup_id,
        )
        database.add_audit("tradingview", result.status.value.lower(), "Draw Gann fan",
                           actor="user", details={"verified": result.verified, "error_code": result.error_code})
        return result.as_dict()

    @application.get("/api/trading/pitchfork")
    async def pitchfork_analysis(symbol: str = "XAUUSD", timeframe: str = "M15", variant: str = "andrews") -> dict[str, Any]:
        return (await asyncio.to_thread(trading.pitchfork_analysis, symbol, timeframe, variant=variant)).as_dict()

    @application.post("/api/tradingview/drawings/pitchfork")
    async def draw_pitchfork(payload: PitchforkDrawRequest) -> dict[str, Any]:
        trading.refresh_permissions()
        result = await asyncio.to_thread(
            trading.draw_pitchfork, payload.symbol, payload.timeframe,
            variant=payload.variant, setup_id=payload.setup_id,
        )
        database.add_audit("tradingview", result.status.value.lower(), f"Draw {payload.variant} pitchfork",
                           actor="user", details={"verified": result.verified, "error_code": result.error_code})
        return result.as_dict()

    @application.get("/api/research/replay/capability")
    async def replay_capability() -> dict[str, Any]:
        return replay.capability()

    @application.post("/api/research/replay/start")
    async def replay_start(payload: ReplayStartRequest) -> dict[str, Any]:
        result = await asyncio.to_thread(
            replay.start, payload.symbol, payload.timeframe,
            count=payload.count, start_offset=payload.start_offset, session_id=payload.session_id,
        )
        return result.as_dict()

    @application.post("/api/research/replay/control")
    async def replay_control(payload: ReplayControlRequest) -> dict[str, Any]:
        result = await asyncio.to_thread(replay.control, payload.action,
                                         session_id=payload.session_id, bars=payload.bars)
        return result.as_dict()

    @application.post("/api/research/replay/scan")
    async def replay_scan(payload: ReplayScanRequest) -> dict[str, Any]:
        result = await asyncio.to_thread(
            replay.run_scan, payload.trigger, session_id=payload.session_id, bars=payload.bars,
            stop_atr_multiple=payload.stop_atr_multiple, reward_multiple=payload.reward_multiple,
        )
        return result.as_dict()

    @application.post("/api/trading/strategies", status_code=201)
    async def save_strategy(payload: ComposedStrategyRequest) -> dict[str, Any]:
        """Persist a composed strategy as a versioned custom theory."""
        from .trading.research import TRIGGER_REGISTRY

        if payload.entry_trigger and payload.entry_trigger not in TRIGGER_REGISTRY:
            raise HTTPException(400, f"Unknown entry trigger: {payload.entry_trigger}")
        definition = {
            "name": payload.name,
            "description": " + ".join(
                part for part in (payload.context, payload.setup, payload.confirmation, payload.entry_trigger) if part
            ) or payload.name,
            "kind": "composed_strategy",
            "context": payload.context, "setup": payload.setup, "confirmation": payload.confirmation,
            "entry_trigger": payload.entry_trigger, "entry": payload.entry_trigger or payload.confirmation,
            "invalidation": payload.invalidation or "A close through the setup level.",
            "stop": payload.stop, "targets": payload.targets or ["Nearest external liquidity"],
            "timeframes": payload.timeframes or ["H1", "M15", "M5"],
            "minimum_rr": payload.minimum_rr, "direction": payload.direction,
            "conditions": _composed_conditions(payload),
        }
        try:
            saved = await asyncio.to_thread(trading.save_custom_theory, definition)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        database.add_audit("trading", "success", f"Saved composed strategy {payload.name}",
                           actor="user", details={"version": saved.get("version")})
        return saved

    @application.get("/api/trading/strategies")
    async def list_strategies() -> dict[str, Any]:
        theories = await asyncio.to_thread(database.list_custom_theories)
        return {"strategies": theories, "count": len(theories)}

    @application.get("/api/voice/capabilities")
    async def voice_capabilities() -> dict[str, Any]:
        local = await asyncio.to_thread(voice.capabilities)
        return {
            **local,
            "browser_fallback": {
                "vad": True, "partial_transcript": True, "continuous": True,
                "barge_in": True, "streaming_tts_chunks": True,
            },
        }

    def _require_stt(language: str | None) -> None:
        spoken = language or settings.voice_language
        if sorani_speech.is_sorani(spoken):
            # A live health probe used to transcribe silence before the user's
            # audio. That cost a full KurdishTTS round trip for nothing.
            if not voice.sorani_input_configured():
                raise HTTPException(503, sorani_speech.MESSAGE_NO_PROVIDER)
            return
        support = voice.capabilities()
        if support["stt"]["state"] != "AVAILABLE":
            raise HTTPException(503, support["stt"].get("reason") or "Local speech recognition is unavailable.")

    @application.post("/api/voice/listen")
    async def voice_listen(payload: VoiceListenRequest) -> dict[str, Any]:
        await asyncio.to_thread(_require_stt, payload.language)
        result = await asyncio.to_thread(
            voice.listen_once,
            max_seconds=payload.max_seconds,
            device=payload.device,
            language=payload.language,
        )
        database.add_audit(
            "voice", "listen", "Captured one utterance", actor="user",
            details={"captured": result.get("captured"), "seconds": result.get("seconds")},
        )
        return result

    @application.post("/api/voice/transcribe")
    async def voice_transcribe(
        file: UploadFile = File(...),
        language: str | None = Form(None),
    ) -> dict[str, Any]:
        """Transcribe a browser-recorded WAV. Used for Sorani, which Web Speech cannot do."""
        await asyncio.to_thread(_require_stt, language)
        payload = await file.read()
        if len(payload) > VOICE_UPLOAD_MAX_BYTES:
            raise HTTPException(413, "دەنگەکە زۆر گەورەیە.")
        if not payload:
            raise HTTPException(400, sorani_speech.MESSAGE_EMPTY_AUDIO)
        try:
            result = await asyncio.to_thread(voice.transcribe_audio_bytes, payload, language)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        database.add_audit(
            "voice", "transcribe", "Transcribed an uploaded utterance", actor="user",
            details={"captured": result.get("captured"), "seconds": result.get("seconds"),
                     "engine": result.get("engine")},
        )
        return result

    @application.post("/api/voice/speak")
    async def voice_speak(payload: VoiceSpeakRequest) -> dict[str, Any]:
        return await asyncio.to_thread(voice.speak, payload.text, payload.language)

    @application.post("/api/voice/tts")
    async def voice_tts(payload: VoiceSpeakRequest) -> Response:
        """Return Sorani WAV for browser playback. Does not play on the server speakers."""
        result = await asyncio.to_thread(voice.synthesize, payload.text, payload.language)
        audio = result.get("audio")
        if not audio:
            raise HTTPException(503, result.get("error") or sorani_speech.MESSAGE_NO_PROVIDER)
        headers = {
            "X-SAM-Engine": str(result.get("engine") or ""),
            "X-SAM-Language": str(result.get("language") or ""),
        }
        if result.get("speaker_id"):
            headers["X-SAM-Speaker"] = str(result["speaker_id"])
        return Response(content=audio, media_type="audio/wav", headers=headers)

    @application.get("/api/voice/sorani/speakers")
    async def sorani_speakers() -> dict[str, Any]:
        """The Sorani voices available for replies. Never returns a credential."""
        def read() -> dict[str, Any]:
            _, tts_router = voice.sorani_stack()
            health = tts_router.status()["primary"]
            if health["status"] != "CONNECTED":
                return {"status": health["status"], "detail": health.get("detail"), "speakers": []}
            provider = tts_router.primary
            return {
                "status": "CONNECTED",
                "selected": provider.resolve_speaker(),
                "speakers": [speaker.as_dict() for speaker in provider.available_speakers()],
            }

        return await asyncio.to_thread(read)

    @application.post("/api/voice/interrupt")
    async def voice_interrupt() -> dict[str, Any]:
        was_speaking = voice.barge_in.interrupt("api")
        # Barge-in must also stop whatever work the spoken answer came from.
        cancelled = cancellation.emergency_stop("voice_barge_in") if was_speaking else {}
        return {"was_speaking": was_speaking, "interruptions": voice.barge_in.interruptions, "cancelled": cancelled}

    @application.get("/api/mt5")
    async def api_mt5() -> dict[str, Any]:
        result = await asyncio.to_thread(trading.market_snapshot, "XAUUSD", ["M1", "M5", "M15", "H1", "H4", "D1", "W1", "MN1"])
        if not result.data:
            return {"connected": False, "mt5_error": result.error}
        data = result.data
        first = next(iter(data["timeframes"].values()))["metadata"]
        point = float(first.get("point") or 1)
        ohlc = {
            timeframe: {"o": item["open"], "h": item["high"], "l": item["low"], "c": item["close"], "time": item["time"]}
            for timeframe, item in data["timeframes"].items()
        }
        return {
            "connected": True,
            "broker": data["feed"],
            "symbol": data["symbol"],
            "bid": data["bid"],
            "ask": data["ask"],
            "spread": round(data["spread"] / point, 2) if data["spread"] is not None and point else None,
            "ohlc": ohlc,
            "session": ", ".join(data["session"]["active"]) or "CLOSED/TRANSITION",
            "provider_metadata": first,
            "verified": result.verified,
        }

    @application.get("/api/desktop/status")
    async def api_desktop_status() -> dict[str, Any]:
        # Only these three matter here, and asking for them by name avoids
        # opening a handle to every process on the machine.
        wanted = ("tradingview", "terminal64", "python")
        # observe() makes blocking Win32 calls; on the event loop it stalled
        # every other request behind it.
        process_result, state = await asyncio.gather(
            asyncio.to_thread(windows.list_processes, "", wanted),
            asyncio.to_thread(trading.tradingview.observe),
        )
        processes = []
        if isinstance(process_result.data, dict):
            processes = process_result.data.get("processes", [])
        return {
            "processes": processes,
            "tradingview": {
                "running": state.running,
                "windows": [{"hwnd": state.window_handle, "title": state.title}] if state.window_handle else [],
                "interactive": state.interactive,
                "symbol": state.symbol,
                "timeframe": state.timeframe,
                "timeframe_verified": state.timeframe_verified,
            },
        }

    @application.post("/api/abort")
    async def api_abort() -> dict[str, Any]:
        outcome = cancellation.emergency_stop("emergency_stop")
        stopped_monitors = 0
        for setup in database.list_trading_setups(500):
            if setup["monitor_enabled"] and database.set_setup_monitoring(setup["id"], False):
                stopped_monitors += 1
        database.add_audit("emergency_stop", "executed", "Emergency stop cancelled active work", actor="user", details={**outcome, "stopped_monitors": stopped_monitors})
        return {"aborted": True, **outcome, "stopped_monitors": stopped_monitors}

    @application.post("/api/tasks/{task_id}/cancel")
    async def cancel_task(task_id: str) -> dict[str, Any]:
        cancelled = cancellation.cancel(task_id, "user")
        if not cancelled:
            raise HTTPException(404, "Active task not found")
        return {"cancelled": True, "task_id": task_id}

    @application.post("/api/desktop/action")
    async def api_desktop_action(request: Request) -> dict[str, Any]:
        payload = await request.json()
        action = str(payload.get("action", ""))
        if action == "focus_tradingview":
            trading.refresh_permissions()
            result = trading.tradingview.launch()
            return {"ok": result.status in {ExecutionStatus.SUCCESS, ExecutionStatus.PARTIAL}, "action": action, "error": result.error, "verified": result.verified}
        if action == "capture_screen":
            windows.refresh_permissions(computer_control=settings.computer_control_enabled, screen_access=settings.screen_access_enabled)
            result = windows.capture_screen()
            response = {"ok": result.status in {ExecutionStatus.SUCCESS, ExecutionStatus.PARTIAL}, "action": action, "error": result.error, "verified": result.verified}
            if result.data and isinstance(result.data, dict) and result.data.get("path"):
                import base64
                response["image_b64"] = base64.b64encode(Path(result.data["path"]).read_bytes()).decode("ascii")
            return response
        return {"ok": False, "error": f"Unknown action {action}"}

    @application.websocket("/ws/live")
    async def live_socket(websocket: WebSocket):
        origin = websocket.headers.get("origin")
        client_host = (websocket.client.host if websocket.client else "").lower()
        if (origin and origin not in settings.cors_origins) or client_host not in {"127.0.0.1", "::1", "testclient"}:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        await websocket.send_json({
            "type": "state",
            "state": "IDLE",
            "tasks": cancellation.snapshot(),
            "trading": database.get_trading_context(),
        })
        try:
            while True:
                message = await websocket.receive_json()
                message_type = str(message.get("type", "ping"))
                if message_type == "cancel":
                    task_id = str(message.get("task_id", ""))
                    await websocket.send_json({"type": "cancelled", "task_id": task_id, "cancelled": cancellation.cancel(task_id, "live_socket")})
                elif message_type == "emergency_stop":
                    await websocket.send_json({"type": "emergency_stop", **cancellation.emergency_stop("live_socket")})
                else:
                    await websocket.send_json({
                        "type": "state",
                        "state": "IDLE" if not cancellation.snapshot()["active_tasks"] else "ACTING",
                        "tasks": cancellation.snapshot(),
                        "trading": database.get_trading_context(),
                    })
        except WebSocketDisconnect:
            return

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


app = create_app()
