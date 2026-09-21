from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Callable


class ExecutionStatus(StrEnum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"


class CapabilityState(StrEnum):
    AVAILABLE = "AVAILABLE"
    PARTIALLY_AVAILABLE = "PARTIALLY_AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    UNCONFIGURED = "UNCONFIGURED"


class PermissionDisposition(StrEnum):
    ALLOW = "ALLOW"
    ASK = "ASK"
    DENY = "DENY"


@dataclass(slots=True)
class StandardResult:
    status: ExecutionStatus
    executed: bool
    verified: bool
    data: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    observations: list[str] = field(default_factory=list)
    error_code: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload

    @classmethod
    def success(
        cls,
        data: Any,
        *,
        verified: bool = True,
        started_at: float | None = None,
        observations: list[str] | None = None,
    ) -> "StandardResult":
        return cls(
            ExecutionStatus.SUCCESS,
            True,
            verified,
            data=data,
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2) if started_at else 0.0,
            observations=observations or [],
        )

    @classmethod
    def failure(
        cls,
        error: str,
        *,
        status: ExecutionStatus = ExecutionStatus.FAILED,
        executed: bool = False,
        error_code: str | None = None,
        started_at: float | None = None,
        observations: list[str] | None = None,
    ) -> "StandardResult":
        return cls(
            status,
            executed,
            False,
            error=error,
            error_code=error_code,
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2) if started_at else 0.0,
            observations=observations or [],
        )


@dataclass(slots=True)
class ToolManifest:
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    permission_class: str
    timeout_seconds: int
    cancellable: bool
    max_retries: int
    verification: str
    audit_behavior: str
    secret_policy: str
    error_codes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SkillDefinition:
    id: str
    name: str
    category: str
    version: str
    required_tools: list[str]
    optional_tools: list[str]
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    prerequisites: list[str]
    data_requirements: list[str]
    model_requirements: list[str]
    permission_requirements: list[str]
    cancellable: bool
    timeout_seconds: int
    max_retries: int
    verification_policy: str
    failure_states: list[str]
    health: CapabilityState = CapabilityState.AVAILABLE
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["health"] = self.health.value
        return payload


class SkillRegistry:
    def __init__(self) -> None:
        self._definitions: dict[str, SkillDefinition] = {}
        self._handlers: dict[str, Callable[..., Any]] = {}

    def register(self, definition: SkillDefinition, handler: Callable[..., Any] | None = None) -> None:
        if definition.id in self._definitions:
            raise ValueError(f"Duplicate skill id: {definition.id}")
        self._definitions[definition.id] = definition
        if handler is not None:
            self._handlers[definition.id] = handler

    def get(self, skill_id: str) -> SkillDefinition:
        try:
            return self._definitions[skill_id]
        except KeyError as exc:
            raise KeyError(f"Unknown skill: {skill_id}") from exc

    def handler(self, skill_id: str) -> Callable[..., Any] | None:
        return self._handlers.get(skill_id)

    def list(self, category: str | None = None) -> list[dict[str, Any]]:
        definitions = self._definitions.values()
        if category:
            definitions = (item for item in definitions if item.category == category)
        return [item.as_dict() for item in sorted(definitions, key=lambda item: (item.category, item.name))]

