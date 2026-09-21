from __future__ import annotations

import asyncio
import threading
import uuid
from dataclasses import dataclass, field


@dataclass(slots=True)
class CancellationToken:
    task_id: str
    reason: str | None = None
    _event: threading.Event = field(default_factory=threading.Event)

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self, reason: str = "user") -> None:
        self.reason = reason
        self._event.set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise asyncio.CancelledError(self.reason or "cancelled")


class CancellationManager:
    """Owns independent task tokens and a persistent emergency-stop generation."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tokens: dict[str, CancellationToken] = {}
        self._emergency_generation = 0

    def create(self, task_id: str | None = None) -> CancellationToken:
        token = CancellationToken(task_id or f"task_{uuid.uuid4().hex}")
        with self._lock:
            self._tokens[token.task_id] = token
        return token

    def get(self, task_id: str) -> CancellationToken | None:
        with self._lock:
            return self._tokens.get(task_id)

    def complete(self, task_id: str) -> None:
        with self._lock:
            self._tokens.pop(task_id, None)

    def cancel(self, task_id: str, reason: str = "user") -> bool:
        token = self.get(task_id)
        if token is None:
            return False
        token.cancel(reason)
        return True

    def emergency_stop(self, reason: str = "emergency_stop") -> dict[str, int]:
        with self._lock:
            tokens = list(self._tokens.values())
            self._emergency_generation += 1
            generation = self._emergency_generation
        for token in tokens:
            token.cancel(reason)
        return {"cancelled_tasks": len(tokens), "generation": generation}

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "active_tasks": [token.task_id for token in self._tokens.values() if not token.cancelled],
                "cancelled_tasks": [token.task_id for token in self._tokens.values() if token.cancelled],
                "emergency_generation": self._emergency_generation,
            }

