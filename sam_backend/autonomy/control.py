"""Who is driving this task, and has it been told to stop.

Two mechanisms with one purpose, kept away from the run loop because they
are about the run rather than part of it.

Exactly one driver may hold a task. Two would each load their own copy from
the database and overwrite each other's progress, so the second caller is
refused rather than queued. And a cancellation token has to exist before
Stop is pressed, or the shared cancel endpoint has nothing to flip and an
autonomous run cannot be stopped at all.

Nothing here decides anything about the run itself: no transitions, no
persistence, no events. The orchestrator asks these questions and draws its
own conclusions.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from ..cancellation import CancellationManager
from ..tasks import AgentTask


class TaskAlreadyRunning(RuntimeError):
    """Raised when a second driver tries to take a task that is already running.

    Two drivers on one task would each hold their own copy loaded from the
    database and overwrite each other's progress, so the second caller is
    refused rather than queued.
    """


class RunControl:
    """Ownership of a running task, and the channel that stops it."""

    def __init__(self, cancellation: CancellationManager | None = None) -> None:
        self.cancellation = cancellation
        # The set is the authority; the lock only keeps check-and-claim atomic.
        self._driving: set[str] = set()
        self._guard = asyncio.Lock()

    @asynccontextmanager
    async def exclusive(self, task_id: str):
        """Claim sole ownership of driving one task, or refuse."""
        async with self._guard:
            if task_id in self._driving:
                raise TaskAlreadyRunning(f"{task_id} is already running")
            self._driving.add(task_id)
        try:
            yield
        finally:
            async with self._guard:
                self._driving.discard(task_id)

    def driving(self, task_id: str) -> bool:
        return task_id in self._driving

    def arm(self, task: AgentTask) -> None:
        """Publish the task to the cancellation manager so Stop can reach it."""
        if self.cancellation is not None and self.cancellation.get(task.id) is None:
            self.cancellation.create(task.id)

    def release(self, task: AgentTask) -> None:
        """Drop the token once the run is over.

        A paused run keeps its token: waiting for approval is still stoppable.
        """
        if self.cancellation is not None and task.terminal:
            self.cancellation.complete(task.id)

    def cancelled(self, task: AgentTask) -> bool:
        if self.cancellation is None:
            return False
        token = self.cancellation.get(task.id)
        return token is not None and token.cancelled

    def request_stop(self, task: AgentTask) -> None:
        """Flip the token. Whether anyone is listening is the caller's problem."""
        if self.cancellation is None:
            return
        self.arm(task)
        self.cancellation.cancel(task.id, "user")
