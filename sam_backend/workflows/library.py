"""Where candidate workflows come from.

`WorkflowLibraryProvider` is the seam. Today one implementation reads a public
GitHub repository of n8n workflows; a private company library, a curated SAM
set, or the user's own n8n could implement the same three methods without the
agent learning anything new.

The corpus is ~37 MB across ~2000 files, so none of it is vendored. Two small
index files and one directory listing are fetched and cached, which is enough
to search by name, service, trigger and category; a workflow's actual JSON is
fetched only when someone asks for that one. That ordering is the whole point:
search thousands, read one.

Everything fetched here is untrusted third-party data. It is size-capped
before parsing and hashed on arrival, and nothing in it is executed.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from .models import (
    LibraryState,
    WorkflowError,
    WorkflowErrorCode,
    WorkflowProvenance,
    WorkflowSummary,
    workflow_sha256,
)

SOURCE_REPOSITORY = "Zie619/n8n-workflows"
RAW_BASE = "https://raw.githubusercontent.com/Zie619/n8n-workflows"
API_BASE = "https://api.github.com/repos/Zie619/n8n-workflows"
BRANCH = "main"

# A workflow that will not fit in a prompt is not a candidate, and a huge one
# is a denial-of-service risk before it is anything else.
MAX_WORKFLOW_BYTES = 1_500_000
MAX_INDEX_BYTES = 8_000_000
TIMEOUT_SECONDS = 20.0
INDEX_TTL_SECONDS = 24 * 60 * 60
DEFAULT_LIMIT = 5
MAX_LIMIT = 10

# `0001_Telegram_Schedule_Automation_Scheduled.json` -> id, words, trigger.
_FILENAME = re.compile(r"^(?P<id>\d+)_(?P<body>.+?)(?:_(?P<trigger>Scheduled|Triggered|Webhook|Manual))?\.json$", re.I)
_TRIGGERS = {"scheduled", "triggered", "webhook", "manual"}


@dataclass(frozen=True, slots=True)
class LibraryEntry:
    """One indexed workflow: everything known before its JSON is fetched."""

    workflow_id: str
    filename: str
    path: str
    category: str
    title: str
    services: tuple[str, ...]
    trigger: str
    size_bytes: int

    @property
    def complexity(self) -> str:
        """A blunt proxy: file size tracks node count closely enough to sort by."""
        if self.size_bytes < 6_000:
            return "simple"
        return "moderate" if self.size_bytes < 25_000 else "complex"

    def summary(self, match_reason: str = "") -> WorkflowSummary:
        return WorkflowSummary(
            workflow_id=self.workflow_id, title=self.title,
            description=f"{self.category} workflow using {', '.join(self.services[:4]) or 'built-in nodes'}.",
            services=self.services, trigger=self.trigger, complexity=self.complexity,
            category=self.category, source=SOURCE_REPOSITORY, source_path=self.path,
            size_bytes=self.size_bytes, match_reason=match_reason,
        )


class WorkflowLibraryProvider(Protocol):
    def search(self, query: str = "", **filters: Any) -> list[WorkflowSummary]: ...
    def get_workflow(self, workflow_id: str) -> tuple[dict[str, Any], WorkflowProvenance]: ...
    def get_categories(self) -> list[str]: ...


def _parse_filename(filename: str, category: str, path: str, size: int) -> LibraryEntry | None:
    match = _FILENAME.match(filename)
    if not match:
        return None
    body = match.group("body").replace("_", " ").strip()
    trigger = (match.group("trigger") or "").lower() or "unknown"
    words = [word for word in match.group("body").split("_") if word]
    # The leading words are service names; the rest describes the purpose.
    services = tuple(word.lower() for word in words[:3] if word.lower() not in _TRIGGERS)
    return LibraryEntry(
        workflow_id=f"{match.group('id')}_{words[0] if words else 'workflow'}".lower(),
        filename=filename, path=path, category=category,
        title=body, services=services, trigger=trigger, size_bytes=size,
    )


class GitHubWorkflowLibrary:
    """The public n8n workflow collection, read through GitHub's own API.

    The repository ships a FastAPI search service, but no hosted instance of it
    is advertised, so SAM does not pretend one exists: it reads the repository
    contents directly, which is a supported and stable interface.
    """

    def __init__(self, cache_dir: Path, *, client_factory: Any = None) -> None:
        # Outside the repository: a cache is data, not source.
        self.cache_dir = Path(cache_dir) / "workflow-library"
        self._client_factory = client_factory or (
            lambda: httpx.Client(timeout=TIMEOUT_SECONDS, trust_env=False, follow_redirects=True)
        )
        self._entries: list[LibraryEntry] | None = None
        self._state = LibraryState.UNAVAILABLE
        self._retrieved_at = ""
        self._commit_sha = ""

    # -- state -------------------------------------------------------------
    @property
    def state(self) -> LibraryState:
        return self._state

    def status(self) -> dict[str, Any]:
        entries = self._entries or []
        return {
            "state": self._state.value, "source_repository": SOURCE_REPOSITORY,
            "indexed_workflows": len(entries), "retrieved_at": self._retrieved_at,
            "commit_sha": self._commit_sha, "cache_dir": str(self.cache_dir),
        }

    # -- cache -------------------------------------------------------------
    def _cache_path(self, name: str) -> Path:
        return self.cache_dir / name

    def _read_cache(self, name: str, ttl: float) -> Any | None:
        path = self._cache_path(name)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        fresh = (time.time() - float(payload.get("retrieved_at_epoch") or 0)) < ttl
        self._state = LibraryState.AVAILABLE if fresh else LibraryState.STALE_CACHE
        self._retrieved_at = str(payload.get("retrieved_at") or "")
        self._commit_sha = str(payload.get("commit_sha") or "")
        return payload.get("data")

    def _write_cache(self, name: str, data: Any, commit_sha: str = "") -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_path(name).write_text(json.dumps({
                "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "retrieved_at_epoch": time.time(),
                "source_repository": SOURCE_REPOSITORY, "commit_sha": commit_sha, "data": data,
            }), encoding="utf-8")
        except OSError:
            pass  # a cache that cannot be written is a slower library, not a failure

    # -- fetching ----------------------------------------------------------
    def _get(self, client: Any, url: str, *, limit: int) -> str:
        response = client.get(url, headers={"User-Agent": "SAM-workflow-intelligence",
                                            "Accept": "application/vnd.github+json"})
        if response.status_code == 404:
            raise WorkflowError(f"Not found in the workflow library: {url}", WorkflowErrorCode.NOT_FOUND)
        if response.status_code in (401, 403):
            raise WorkflowError("The workflow library refused the request (rate limit or auth).",
                                WorkflowErrorCode.RATE_LIMIT)
        response.raise_for_status()
        text = response.text
        if len(text) > limit:
            raise WorkflowError(f"Library response exceeds {limit} bytes and was not parsed.",
                                WorkflowErrorCode.TOO_LARGE)
        return text

    def _build_index(self) -> list[LibraryEntry]:
        """One directory listing plus one category map, cached for a day."""
        categories: dict[str, str] = {}
        entries: list[LibraryEntry] = []
        with self._client_factory() as client:
            tree_raw = self._get(client, f"{API_BASE}/git/trees/{BRANCH}?recursive=1", limit=MAX_INDEX_BYTES)
            tree = json.loads(tree_raw)
            commit_sha = str(tree.get("sha") or "")
            try:
                mapping = json.loads(self._get(
                    client, f"{RAW_BASE}/{BRANCH}/context/search_categories.json", limit=MAX_INDEX_BYTES))
                categories = {
                    str(item.get("filename")): str(item.get("category") or "")
                    for item in mapping if isinstance(item, dict)
                }
            except (WorkflowError, httpx.HTTPError, ValueError):
                categories = {}  # a missing category map costs a filter, not the search
        for node in tree.get("tree", []):
            path = str(node.get("path") or "")
            if node.get("type") != "blob" or not path.startswith("workflows/") or not path.endswith(".json"):
                continue
            filename = path.rsplit("/", 1)[-1]
            entry = _parse_filename(filename, categories.get(filename, path.split("/")[1]),
                                    path, int(node.get("size") or 0))
            if entry:
                entries.append(entry)
        self._commit_sha = commit_sha
        self._write_cache(
            "index.json",
            [asdict(entry) | {"services": list(entry.services)} for entry in entries],
            commit_sha,
        )
        self._state = LibraryState.AVAILABLE
        self._retrieved_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return entries

    def _index(self, *, refresh: bool = False) -> list[LibraryEntry]:
        if self._entries is not None and not refresh:
            return self._entries
        if not refresh:
            cached = self._read_cache("index.json", INDEX_TTL_SECONDS)
            if cached:
                self._entries = [LibraryEntry(**{**item, "services": tuple(item.get("services") or ())})
                                 for item in cached]
                return self._entries
        try:
            self._entries = self._build_index()
        except (WorkflowError, httpx.HTTPError, ValueError) as exc:
            cached = self._read_cache("index.json", INDEX_TTL_SECONDS)
            if cached:
                # Usable, but say so: stale is not the same as current.
                self._state = LibraryState.STALE_CACHE
                self._entries = [LibraryEntry(**{**item, "services": tuple(item.get("services") or ())})
                                 for item in cached]
                return self._entries
            self._state = LibraryState.UNAVAILABLE
            code = exc.code if isinstance(exc, WorkflowError) else WorkflowErrorCode.NETWORK
            raise WorkflowError(f"The workflow library is unavailable: {type(exc).__name__}", code) from exc
        return self._entries

    # -- provider surface --------------------------------------------------
    def get_categories(self) -> list[str]:
        return sorted({entry.category for entry in self._index() if entry.category})

    def search(
        self, query: str = "", *, category: str = "", service: str = "",
        trigger: str = "", complexity: str = "", limit: int = DEFAULT_LIMIT, **_: Any,
    ) -> list[WorkflowSummary]:
        """Rank by how many asked-for words a candidate actually matches.

        No opaque score: a candidate is shortlisted because specific terms
        appear in its name or services, and the summary says which.
        """
        entries = self._index()
        terms = [word for word in re.split(r"[\s,]+", str(query or "").lower()) if len(word) > 1][:8]
        bounded = max(1, min(int(limit or DEFAULT_LIMIT), MAX_LIMIT))
        scored: list[tuple[int, int, LibraryEntry, str]] = []
        for entry in entries:
            if category and entry.category.lower() != category.lower():
                continue
            if service and not any(service.lower() in item for item in entry.services):
                continue
            if trigger and entry.trigger != trigger.lower():
                continue
            if complexity and entry.complexity != complexity.lower():
                continue
            haystack = f"{entry.title} {' '.join(entry.services)} {entry.category}".lower()
            hits = [term for term in terms if term in haystack]
            if terms and not hits:
                continue
            reason = ("matched " + ", ".join(hits)) if hits else f"in {entry.category or 'the library'}"
            # Prefer more matched terms, then the simpler workflow: fewer moving
            # parts is easier to review and adapt.
            scored.append((-len(hits), entry.size_bytes, entry, reason))
        scored.sort(key=lambda item: (item[0], item[1]))
        return [entry.summary(reason) for _, _, entry, reason in scored[:bounded]]

    def get_workflow(self, workflow_id: str) -> tuple[dict[str, Any], WorkflowProvenance]:
        """Fetch exactly one workflow, size-capped, hashed on arrival."""
        entry = next((item for item in self._index() if item.workflow_id == str(workflow_id).lower()), None)
        if entry is None:
            raise WorkflowError(f"No workflow {workflow_id!r} in the library index.", WorkflowErrorCode.NOT_FOUND)
        if entry.size_bytes > MAX_WORKFLOW_BYTES:
            raise WorkflowError(
                f"{entry.filename} is {entry.size_bytes} bytes, over the {MAX_WORKFLOW_BYTES} limit.",
                WorkflowErrorCode.TOO_LARGE)
        with self._client_factory() as client:
            raw = self._get(client, f"{RAW_BASE}/{BRANCH}/{entry.path}", limit=MAX_WORKFLOW_BYTES)
        try:
            workflow = json.loads(raw)
        except ValueError as exc:
            raise WorkflowError(f"{entry.filename} is not valid JSON.", WorkflowErrorCode.INVALID_WORKFLOW) from exc
        if not isinstance(workflow, dict):
            raise WorkflowError(f"{entry.filename} is not a workflow object.", WorkflowErrorCode.INVALID_WORKFLOW)
        provenance = WorkflowProvenance(
            source="library", source_repository=SOURCE_REPOSITORY, source_path=entry.path,
            source_workflow_id=entry.workflow_id, source_commit_sha=self._commit_sha,
            retrieved_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            original_sha256=workflow_sha256(workflow),
        )
        return workflow, provenance

    def refresh(self) -> dict[str, Any]:
        self._index(refresh=True)
        return self.status()
