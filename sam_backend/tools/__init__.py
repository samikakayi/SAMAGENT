"""SAM's tool layer.

    registry.py   the engine: dispatch, workspace containment, process
                  execution, and every tool handler
    catalogue.py  the data: model-facing schemas and per-tool permission /
                  manifest metadata

The handlers stay together in the registry because they share its containment
and process helpers; splitting them by domain would scatter that shared core
without making any one group easier to understand.
"""

from .catalogue import tool_manifests, tool_specs
from .registry import ToolRegistry, ToolResult

__all__ = ["ToolRegistry", "ToolResult", "tool_specs", "tool_manifests"]
