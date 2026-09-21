"""Capability discovery: what this machine can actually do, right now.

The agent should reach for a specialised tool when one exists rather than
hand-rolling the same work through generic code. That decision needs facts,
so this module probes the environment once and caches the answer: which
developer CLIs are installed, which model providers are reachable, whether
browser automation and screen OCR are usable.

Probes are cheap and defensive. A capability that cannot be confirmed is
reported UNCONFIGURED with a reason -- never assumed present, because acting
on an assumed tool fails later and more confusingly than declining up front.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from importlib.util import find_spec
from typing import Any

from .contracts import CapabilityState

PROBE_TIMEOUT = 6
PROBE_TTL_SECONDS = 300.0


@dataclass(slots=True)
class Capability:
    name: str
    category: str  # cli | runtime | provider | automation | vision | vcs
    state: CapabilityState
    detail: str = ""
    version: str = ""
    path: str = ""

    @property
    def available(self) -> bool:
        return self.state is CapabilityState.AVAILABLE

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        payload["available"] = self.available
        return payload


# (executable, category, version flag). Probed in one pass at startup.
CLI_PROBES: tuple[tuple[str, str, str], ...] = (
    ("git", "vcs", "--version"),
    ("node", "runtime", "--version"),
    ("npm", "cli", "--version"),
    ("pnpm", "cli", "--version"),
    ("yarn", "cli", "--version"),
    ("bun", "cli", "--version"),
    ("python", "runtime", "--version"),
    ("pip", "cli", "--version"),
    ("uv", "cli", "--version"),
    ("poetry", "cli", "--version"),
    ("go", "runtime", "version"),
    ("cargo", "cli", "--version"),
    ("dotnet", "runtime", "--version"),
    ("java", "runtime", "-version"),
    ("docker", "cli", "--version"),
    ("gh", "cli", "--version"),
    ("rg", "cli", "--version"),
    ("tsc", "cli", "--version"),
)

PYTHON_MODULE_PROBES: tuple[tuple[str, str, str], ...] = (
    ("playwright", "automation", "Isolated browser automation"),
    ("pytest", "cli", "Python test runner"),
    ("winsdk", "vision", "Windows screen OCR bindings"),
    ("pywinauto", "automation", "Windows UI Automation"),
    ("faster_whisper", "automation", "Local speech recognition"),
    ("piper", "automation", "Local speech synthesis"),
    ("MetaTrader5", "automation", "Read-only market data"),
)


def _probe_executable(name: str, version_flag: str) -> Capability:
    path = shutil.which(name)
    category = next((item[1] for item in CLI_PROBES if item[0] == name), "cli")
    if not path:
        return Capability(name, category, CapabilityState.UNAVAILABLE, detail=f"{name} is not on PATH")
    try:
        completed = subprocess.run(
            [path, version_flag], capture_output=True, text=True, timeout=PROBE_TIMEOUT,
            encoding="utf-8", errors="replace",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return Capability(name, category, CapabilityState.UNCONFIGURED, detail=str(exc), path=path)
    # Several tools (java, go) print their version to stderr.
    output = (completed.stdout or completed.stderr or "").strip().splitlines()
    version = output[0].strip() if output else ""
    return Capability(name, category, CapabilityState.AVAILABLE, version=version, path=path)


def _probe_module(module: str, category: str, detail: str) -> Capability:
    try:
        found = find_spec(module) is not None
    except (ImportError, ValueError):
        found = False
    return Capability(
        module, category,
        CapabilityState.AVAILABLE if found else CapabilityState.UNCONFIGURED,
        detail=detail if found else f"{module} is not installed",
    )


class CapabilityRegistry:
    """Caches one environment probe and answers questions about it."""

    def __init__(self) -> None:
        self._capabilities: dict[str, Capability] = {}
        self._probed_at: float = 0.0

    def refresh(self) -> dict[str, Capability]:
        found: dict[str, Capability] = {}
        for name, _category, flag in CLI_PROBES:
            found[name] = _probe_executable(name, flag)
        for module, category, detail in PYTHON_MODULE_PROBES:
            found[module] = _probe_module(module, category, detail)
        self._capabilities = found
        self._probed_at = time.time()
        return found

    def all(self) -> dict[str, Capability]:
        if not self._capabilities or (time.time() - self._probed_at) > PROBE_TTL_SECONDS:
            self.refresh()
        return self._capabilities

    def get(self, name: str) -> Capability | None:
        return self.all().get(name)

    def available(self, category: str | None = None) -> list[Capability]:
        return [
            capability for capability in self.all().values()
            if capability.available and (category is None or capability.category == category)
        ]

    def as_dict(self) -> dict[str, Any]:
        capabilities = self.all()
        return {
            "probed_at": self._probed_at,
            "available": sorted(name for name, item in capabilities.items() if item.available),
            "missing": sorted(name for name, item in capabilities.items() if not item.available),
            "capabilities": {name: item.as_dict() for name, item in sorted(capabilities.items())},
        }

    def summary_text(self) -> str:
        """One line for the model's context: what it may reach for."""
        available = sorted(item.name for item in self.available())
        return "Installed tools: " + (", ".join(available) if available else "none detected")


@dataclass(slots=True)
class ProviderStatus:
    """Reachability of one model provider, kept separate from CLI probes
    because it depends on network and credentials rather than PATH."""

    name: str
    state: CapabilityState
    detail: str = ""
    models: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        return payload


async def probe_providers(adapters: Any, settings: Any) -> list[ProviderStatus]:
    """Ask each configured adapter whether it can actually serve a request.

    Credentials are never echoed: only whether one is configured at all.
    """
    statuses: list[ProviderStatus] = []
    configured = {
        "ollama": True,
        "litellm": bool(getattr(settings, "litellm_base_url", "")),
        "openrouter": bool(getattr(settings, "openrouter_api_key", None) or os.getenv("OPENROUTER_API_KEY")),
        "openai": bool(getattr(settings, "openai_api_key", None) or os.getenv("OPENAI_API_KEY")),
    }
    for name, is_configured in configured.items():
        if not is_configured:
            statuses.append(ProviderStatus(name, CapabilityState.UNCONFIGURED, "No credential configured"))
            continue
        try:
            adapter = adapters.get(name)
        except Exception as exc:  # noqa: BLE001 - adapter lookup is provider-specific
            statuses.append(ProviderStatus(name, CapabilityState.UNAVAILABLE, str(exc)[:200]))
            continue
        try:
            models = await adapter.list_models()
            identifiers = [str(model.get("id", "")) for model in (models or []) if isinstance(model, dict)]
            statuses.append(
                ProviderStatus(
                    name,
                    CapabilityState.AVAILABLE if identifiers else CapabilityState.UNCONFIGURED,
                    f"{len(identifiers)} model(s)" if identifiers else "Reachable but no models are published",
                    models=identifiers[:25],
                )
            )
        except Exception as exc:  # noqa: BLE001 - network/provider errors vary widely
            statuses.append(ProviderStatus(name, CapabilityState.UNAVAILABLE, str(exc)[:200]))
    return statuses
