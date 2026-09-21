"""The autonomous runtime: driving a goal to a verified outcome.

    orchestrator.py  the state machine -- understand, scan, plan, execute,
                     observe, validate, heal, and the lifecycle around it
                     (ownership, cancellation, approvals, restart handling)
    checkpoints.py   workspace snapshot / diff / restore mechanics

The drive loop and the lifecycle stay together on purpose: run -> drive ->
execute -> observe -> drive is one state machine with one owner, and splitting
it would only trade clarity for two objects that call each other back.
"""

from .checkpoints import WorkspaceCheckpoints
from .orchestrator import AutonomousOrchestrator, TaskAlreadyRunning

__all__ = ["AutonomousOrchestrator", "TaskAlreadyRunning", "WorkspaceCheckpoints"]
