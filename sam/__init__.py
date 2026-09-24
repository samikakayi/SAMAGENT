"""SAM 2 -- a voice-first Sorani desktop assistant and trading analyst.

One process: a PySide6 UI thread and an asyncio core thread (see
``sam.bridge``). Core modules never import Qt. Package contracts live in
``docs/CONTRACTS.md``; the design in ``docs/DESIGN.md``.
"""

__all__ = ["__version__", "APP_NAME"]

__version__ = "2.0.0-dev"
APP_NAME = "SAM"
