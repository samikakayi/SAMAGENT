"""SAM's knowledge library: the user's own books and documents (PDF, DOCX,
TXT, MD), indexed locally with page citations -- plus the ``run_python``
tool, which is registered from here (``sam.app.PACKAGES`` entry
``sam.knowledge``; CONTRACTS section 9).

``register`` is fast and side-effect free (settings, schema, the
``app.knowledge`` slot, tools); ``start`` re-indexes files that changed on
disk, a while after start-up so it never competes with the island.

For the brain (analyze_market / strategy answers)::

    from sam.knowledge import passages_for, context_for_prompt
    passages = await passages_for(app, "order block entry rules", k=4, max_chars=2400)
    block = context_for_prompt(passages)        # "" when the library has nothing relevant
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger("sam.knowledge")

REFRESH_DELAY_S = 25.0


def register(app: Any) -> None:
    from ..hands import python_tool
    from .library import DEFAULTS, Library
    from .schema import MIGRATIONS
    from .tools import TOOLS

    app.config.register_defaults(DEFAULTS)
    app.db.ensure_schema("knowledge", MIGRATIONS)
    app.knowledge = Library(app)
    for handler in TOOLS:
        app.tools.add(handler, owner="knowledge")
    python_tool.register(app)


async def start(app: Any) -> None:
    library = getattr(app, "knowledge", None)
    if library is None or not app.config.get("knowledge.refresh_on_start", True):
        return

    async def later() -> None:
        await asyncio.sleep(REFRESH_DELAY_S)
        try:
            result = await library.refresh()
            if result.get("changed") or result.get("missing"):
                log.info("knowledge refresh: %s", result)
        except Exception:  # noqa: BLE001
            log.warning("knowledge refresh failed", exc_info=True)

    app.spawn(later(), "knowledge-refresh")


async def stop(app: Any) -> None:
    library = getattr(app, "knowledge", None)
    if library is not None:
        library.cancel()


async def passages_for(app: Any, question: str, *, k: int = 4, max_chars: int = 2400,
                       min_score: float = 0.35) -> list[dict[str, Any]]:
    """Best library passages for ``question`` ([] when the library is empty,
    not loaded, or nothing relevant). Each: title, page, page_end, section,
    citation ("«Title», p. 12"), citation_ckb, text, score, document_id."""
    library = getattr(app, "knowledge", None)
    if library is None:
        return []
    try:
        return await library.passages_for(question, k=k, max_chars=max_chars, min_score=min_score)
    except Exception:  # noqa: BLE001 - the library must never break an analysis
        log.warning("knowledge passages failed", exc_info=True)
        return []


def context_for_prompt(passages: list[dict[str, Any]]) -> str:
    from .library import Library

    return Library.context_for_prompt(passages)


__all__ = ["context_for_prompt", "passages_for", "register", "start", "stop"]
