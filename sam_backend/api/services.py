"""The services one running SAM is made of.

Passed whole to each route group, which names the few it needs at the top of
its register function. A typed record rather than a lookup table: a reader
can see every dependency the application has, and a route group cannot
quietly acquire one that is not declared here.

The adapter registry is deliberately absent: it is replaced wholesale when a
credential changes, so the only authoritative answer to "which registry is
current" is application.state.adapters. Holding it here would hand out a
snapshot that silently goes stale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..agent import AgentService
from ..agent_api import AgentApi
from ..autonomy import AutonomousOrchestrator
from ..cancellation import CancellationManager
from ..capabilities import CapabilityRegistry
from ..config import Settings
from ..db import Database
from ..policy import RiskPolicy
from ..project_map import ProjectScanner
from ..provider_health import ProviderHealth
from ..routing import ModelRouter
from ..secrets import SecretStore
from ..tasks import TaskStore
from ..tools import ToolRegistry
from ..trading.replay import BarReplayResearch
from ..trading.service import TradingService
from ..verification import VerificationEngine
from ..voice import VoiceService
from ..windows_control import WindowsController


@dataclass(slots=True)
class AppServices:
    settings: Settings
    database: Database
    policy: RiskPolicy
    cancellation: CancellationManager
    trading: TradingService
    windows: WindowsController
    scanner: ProjectScanner
    verifier: VerificationEngine
    capabilities: CapabilityRegistry
    tools: ToolRegistry
    secrets: SecretStore
    provider_health: ProviderHealth
    router: ModelRouter
    agent: AgentService
    task_store: TaskStore
    orchestrator: AutonomousOrchestrator
    agent_api: AgentApi
    voice: VoiceService
    replay: BarReplayResearch
    # Re-resolve credentials and swap the adapter registry in place, so a new
    # key takes effect without a restart.
    apply_stored_credentials: Callable[[], None]
    rebuild_adapters: Callable[[], None]
