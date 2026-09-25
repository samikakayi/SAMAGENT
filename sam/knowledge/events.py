"""Knowledge-library events (compatible additions: subclasses of
``sam.events.Event``, forwarded to the UI by ``UiAdapter`` like every event)."""

from __future__ import annotations

from dataclasses import dataclass

from ..events import Event


@dataclass(frozen=True, slots=True)
class LibraryChanged(Event):
    """A document was queued, indexed, re-indexed, failed or removed."""

    document_id: int = 0
    status: str = ""          # queued|indexing|ready|partial|failed|missing|removed|duplicate|unchanged
    title: str = ""
    detail: str = ""


__all__ = ["LibraryChanged"]
