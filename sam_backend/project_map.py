"""Project intelligence: a cached, structured understanding of a codebase.

The agent must answer "what is this project?" before it can plan work on it.
Rescanning thousands of files every turn would be slow and wasteful, so the
scan is cached and invalidated from a cheap directory fingerprint rather than
a full re-read.

Secret hygiene: environment files contribute variable NAMES only. No value
from a .env-style file ever enters the map, because the map is summarised
into model prompts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Directories that carry no project meaning and are expensive to walk.
IGNORED_DIRECTORIES = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "env", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".next", ".nuxt",
    "dist", "build", "out", "target", "coverage", "htmlcov", ".gradle",
    ".idea", ".vscode", "obj", ".cache", ".parcel-cache", "vendor",
    "site-packages", ".turbo", ".svelte-kit", "__snapshots__",
}
# Scanning is bounded so a mistakenly wide root cannot stall the agent.
MAX_FILES = 20_000
MAX_DEPTH = 12
MAX_MANIFEST_BYTES = 512_000
MAX_SOURCE_SCAN_BYTES = 400_000

LANGUAGE_BY_SUFFIX = {
    ".py": "Python", ".ts": "TypeScript", ".tsx": "TypeScript", ".js": "JavaScript",
    ".jsx": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript", ".go": "Go",
    ".rs": "Rust", ".java": "Java", ".kt": "Kotlin", ".cs": "C#", ".rb": "Ruby",
    ".php": "PHP", ".swift": "Swift", ".c": "C", ".h": "C", ".cpp": "C++",
    ".hpp": "C++", ".sh": "Shell", ".ps1": "PowerShell", ".sql": "SQL",
    ".html": "HTML", ".css": "CSS", ".scss": "CSS", ".vue": "Vue", ".svelte": "Svelte",
}

MANIFEST_FILES = {
    "package.json", "requirements.txt", "requirements-lock.txt", "pyproject.toml",
    "setup.py", "setup.cfg", "Pipfile", "poetry.lock", "uv.lock", "go.mod",
    "Cargo.toml", "pom.xml", "build.gradle", "Gemfile", "composer.json", "pubspec.yaml",
}
LOCK_FILES = {
    "package-lock.json": "npm", "pnpm-lock.yaml": "pnpm", "yarn.lock": "yarn",
    "bun.lockb": "bun", "poetry.lock": "poetry", "uv.lock": "uv",
    "requirements-lock.txt": "pip", "Cargo.lock": "cargo", "go.sum": "go",
}
TEST_DIRECTORY_NAMES = {"tests", "test", "__tests__", "spec", "e2e"}
TEST_FILE_PATTERN = re.compile(r"(?:^|[._-])(?:test|spec)(?:[._-]|$)|^test_|_test\.", re.IGNORECASE)

# Route declarations worth surfacing. Deliberately conservative: a fabricated
# route is worse than a missing one, because the agent may act on it.
ROUTE_PATTERNS = (
    # FastAPI / Flask: @app.get("/path"), @router.websocket("/path")
    re.compile(r"@(?:\w+)\.(get|post|put|patch|delete|websocket)\(\s*[\"']([^\"']+)[\"']", re.IGNORECASE),
    # Express / Koa: app.get("/path", ...), router.post('/path', ...)
    re.compile(r"\b(?:app|router)\.(get|post|put|patch|delete)\(\s*[\"']([^\"']+)[\"']", re.IGNORECASE),
)
ENV_NAME_PATTERN = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", re.MULTILINE)


@dataclass(slots=True)
class ProjectMap:
    """A structured, model-friendly summary of one project root."""

    root: str
    generated_at: float
    fingerprint: str
    file_count: int
    truncated: bool
    languages: dict[str, int] = field(default_factory=dict)
    tree: dict[str, Any] = field(default_factory=dict)
    manifests: list[str] = field(default_factory=list)
    package_managers: list[str] = field(default_factory=list)
    dependencies: dict[str, list[str]] = field(default_factory=dict)
    scripts: dict[str, str] = field(default_factory=dict)
    env_var_names: list[str] = field(default_factory=list)
    api_routes: list[dict[str, str]] = field(default_factory=list)
    test_suites: list[str] = field(default_factory=list)
    commands: dict[str, str] = field(default_factory=dict)
    entry_points: list[str] = field(default_factory=list)
    components: dict[str, list[str]] = field(default_factory=dict)
    git: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary_text(self, *, max_routes: int = 25) -> str:
        """Compact prose for a model prompt; context budget matters here."""
        lines = [
            f"Project root: {self.root}",
            f"Files scanned: {self.file_count}{' (truncated)' if self.truncated else ''}",
        ]
        if self.languages:
            ranked = sorted(self.languages.items(), key=lambda item: -item[1])[:6]
            lines.append("Languages: " + ", ".join(f"{name} ({count})" for name, count in ranked))
        if self.package_managers:
            lines.append("Package managers: " + ", ".join(self.package_managers))
        for area, paths in self.components.items():
            if paths:
                lines.append(f"{area}: " + ", ".join(paths[:8]))
        if self.commands:
            lines.append("Commands: " + "; ".join(f"{key}={value}" for key, value in self.commands.items()))
        if self.test_suites:
            lines.append("Test suites: " + ", ".join(self.test_suites[:8]))
        if self.api_routes:
            shown = self.api_routes[:max_routes]
            lines.append(
                f"API routes ({len(self.api_routes)} found): "
                + ", ".join(f"{route['method'].upper()} {route['path']}" for route in shown)
            )
        if self.env_var_names:
            lines.append(
                f"Env var names ({len(self.env_var_names)}, values never read): "
                + ", ".join(self.env_var_names[:20])
            )
        if self.git:
            branch = self.git.get("branch") or "unknown"
            dirty = self.git.get("dirty_files") or 0
            lines.append(f"Git: branch {branch}, {dirty} modified/untracked file(s)")
        for note in self.notes:
            lines.append(f"Note: {note}")
        return "\n".join(lines)


def _is_ignored(name: str) -> bool:
    return name in IGNORED_DIRECTORIES or (name.startswith(".") and name not in {".github", ".freebuff"})


def gitignored_directories(root: Path) -> set[str]:
    """Directory paths the project itself declares as not-source.

    A scratch directory named in .gitignore (build output, a test sandbox, a
    downloaded model cache) is not part of the project's identity. Walking it
    wastes time and -- worse -- lets a stray manifest inside it masquerade as
    the project's own, so the root .gitignore is treated as authoritative.

    Only unambiguous directory entries are honoured: a trailing slash, no
    glob characters, no negation. Anything cleverer risks ignoring real
    source, which is a far costlier mistake than walking one extra folder.
    """
    ignore_file = root / ".gitignore"
    if not ignore_file.is_file():
        return set()
    found: set[str] = set()
    for raw_line in _read_text(ignore_file, 200_000).splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "!")) or not line.endswith("/"):
            continue
        candidate = line.rstrip("/").lstrip("/")
        if not candidate or any(character in candidate for character in "*?[]"):
            continue
        found.add(candidate.replace("\\", "/"))
    return found


def _read_text(path: Path, limit: int = MAX_MANIFEST_BYTES) -> str:
    try:
        if path.stat().st_size > limit:
            return ""
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _load_json(path: Path) -> Any:
    try:
        return json.loads(_read_text(path) or "null")
    except json.JSONDecodeError:
        return None


def _declared_package_manager(package_json: dict[str, Any]) -> str | None:
    declared = package_json.get("packageManager")
    if isinstance(declared, str) and declared:
        return declared.split("@", 1)[0]
    return None


def run_git(root: Path, *args: str, timeout: int = 10) -> str:
    """Read-only git helper. Returns stdout, or "" when git is unusable."""
    try:
        completed = subprocess.run(
            ["git", *args], cwd=str(root), capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace",
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout if completed.returncode == 0 else ""


class ProjectScanner:
    """Builds and caches ProjectMap instances, one per root."""

    def __init__(
        self, *, max_depth: int = MAX_DEPTH, fresh_seconds: float = 2.0,
    ) -> None:
        self.max_depth = max_depth
        # Within this window a cached map is reused without even
        # fingerprinting. A single task calls scan() many times; walking the
        # tree each time would dominate its runtime for no new information.
        self.fresh_seconds = fresh_seconds
        self._cache: dict[str, ProjectMap] = {}

    # -- walking and fingerprinting ---------------------------------------
    def _walk(self, root: Path, declared_ignores: set[str] | None = None):
        """Depth-bounded walk that skips ignored directories entirely."""
        declared_ignores = declared_ignores or set()
        stack: list[tuple[Path, int]] = [(root, 0)]
        while stack:
            current, depth = stack.pop()
            try:
                entries = list(os.scandir(current))
            except OSError:
                continue
            directories: list[str] = []
            files: list[str] = []
            for entry in entries:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        if _is_ignored(entry.name):
                            continue
                        parent = current.relative_to(root).as_posix()
                        relative = entry.name if parent == "." else f"{parent}/{entry.name}"
                        if entry.name in declared_ignores or relative in declared_ignores:
                            continue
                        directories.append(entry.name)
                    elif entry.is_file(follow_symlinks=False):
                        files.append(entry.name)
                except (OSError, ValueError):
                    continue
            yield current, directories, files, depth
            if depth < self.max_depth:
                for name in directories:
                    stack.append((current / name, depth + 1))

    def fingerprint(self, root: Path, declared_ignores: set[str] | None = None) -> tuple[str, int, bool]:
        """Cheap change signal: (digest, file count, truncated).

        Every file's name, size and mtime feeds the digest. A false "changed"
        only costs one rescan, while a false "unchanged" would make the agent
        act on a stale map -- so this does not sample.
        """
        digest = hashlib.sha256()
        count = 0
        truncated = False
        for path, _directories, files, _depth in self._walk(root, declared_ignores):
            try:
                digest.update(str(path.relative_to(root)).encode("utf-8", "replace"))
            except ValueError:
                continue
            for file_name in sorted(files):
                count += 1
                if count > MAX_FILES:
                    truncated = True
                    break
                try:
                    stat = (path / file_name).stat()
                except OSError:
                    continue
                digest.update(file_name.encode("utf-8", "replace"))
                digest.update(f"{int(stat.st_mtime)}:{stat.st_size}".encode())
            if truncated:
                break
        return digest.hexdigest()[:32], count, truncated

    # -- public API --------------------------------------------------------
    def scan(self, root: Path, *, force: bool = False) -> ProjectMap:
        root = Path(root).expanduser().resolve()
        key = str(root)
        cached = self._cache.get(key)
        if (
            cached is not None
            and not force
            and (time.time() - cached.generated_at) < self.fresh_seconds
        ):
            return cached
        declared_ignores = gitignored_directories(root)
        digest, count, truncated = self.fingerprint(root, declared_ignores)
        if cached is not None and not force and cached.fingerprint == digest:
            return cached
        project_map = self._build(root, digest, count, truncated, declared_ignores)
        self._cache[key] = project_map
        return project_map

    def cached(self, root: Path) -> ProjectMap | None:
        return self._cache.get(str(Path(root).expanduser().resolve()))

    # -- building ----------------------------------------------------------
    def _build(
        self, root: Path, digest: str, count: int, truncated: bool, declared_ignores: set[str] | None = None,
    ) -> ProjectMap:
        languages: dict[str, int] = {}
        manifests: list[str] = []
        env_files: list[Path] = []
        test_suites: set[str] = set()
        source_files: list[Path] = []
        tree: dict[str, Any] = {}
        notes: list[str] = []

        for path, directories, files, depth in self._walk(root, declared_ignores):
            relative = path.relative_to(root)
            if depth <= 2:
                node = tree
                for part in relative.parts:
                    node = node.setdefault(part, {})
                for name in sorted(directories):
                    node.setdefault(name, {})
            relative_text = str(relative).replace("\\", "/")
            if path.name in TEST_DIRECTORY_NAMES and any(TEST_FILE_PATTERN.search(name) for name in files):
                test_suites.add(relative_text)
            for file_name in files:
                suffix = Path(file_name).suffix.lower()
                language = LANGUAGE_BY_SUFFIX.get(suffix)
                if language:
                    languages[language] = languages.get(language, 0) + 1
                    source_files.append(path / file_name)
                if file_name in MANIFEST_FILES or file_name.endswith(".csproj"):
                    manifests.append(file_name if relative_text == "." else f"{relative_text}/{file_name}")
                if file_name.startswith(".env"):
                    env_files.append(path / file_name)
                if TEST_FILE_PATTERN.search(file_name) and suffix in LANGUAGE_BY_SUFFIX:
                    test_suites.add(relative_text)

        package_managers, dependencies, scripts, commands, entry_points = self._analyze_manifests(root, manifests)
        self._infer_commands(root, commands)

        env_var_names: list[str] = []
        for env_file in env_files:
            # Names only. A value from a .env never enters the map.
            for name in ENV_NAME_PATTERN.findall(_read_text(env_file, 200_000)):
                if name not in env_var_names:
                    env_var_names.append(name)
        if env_files:
            notes.append("Environment variable values are deliberately never read into the project map.")
        if truncated:
            notes.append(f"Scan stopped at the {MAX_FILES}-file cap; the map may be incomplete.")

        return ProjectMap(
            root=str(root),
            generated_at=time.time(),
            fingerprint=digest,
            file_count=count,
            truncated=truncated,
            languages=languages,
            tree=tree,
            manifests=sorted(manifests),
            package_managers=sorted(package_managers),
            dependencies=dependencies,
            scripts=scripts,
            env_var_names=env_var_names,
            api_routes=self._scan_routes(root, source_files),
            test_suites=sorted(test_suites),
            commands=commands,
            entry_points=sorted(entry_points),
            components=self._classify_components(root, tree),
            git=self.git_state(root),
            notes=notes,
        )

    @staticmethod
    def _infer_commands(root: Path, commands: dict[str, str]) -> None:
        """Fill gaps from conventions the manifests did not already cover."""
        if (root / "pytest.ini").exists() or (root / "conftest.py").exists():
            commands.setdefault("test", "python -m pytest")
        pyproject = root / "pyproject.toml"
        if pyproject.exists() and "pytest" in _read_text(pyproject):
            commands.setdefault("test", "python -m pytest")
        for script in sorted(root.glob("*.ps1")):
            name = script.stem.lower()
            if name in {"run-tests", "test"}:
                commands.setdefault("test", f".\\{script.name}")
            elif name in {"start", "run", "dev"}:
                commands.setdefault("dev", f".\\{script.name}")
            elif name == "setup":
                commands.setdefault("setup", f".\\{script.name}")

    def _analyze_manifests(
        self, root: Path, manifests: list[str],
    ) -> tuple[set[str], dict[str, list[str]], dict[str, str], dict[str, str], set[str]]:
        package_managers: set[str] = set()
        dependencies: dict[str, list[str]] = {}
        scripts: dict[str, str] = {}
        commands: dict[str, str] = {}
        entry_points: set[str] = set()

        for lock_name, manager in LOCK_FILES.items():
            if (root / lock_name).exists():
                package_managers.add(manager)

        # Shallowest first: the root manifest defines the project's commands.
        # A manifest buried in a subdirectory still contributes dependencies,
        # but it must not dictate how the project as a whole is built or tested.
        for relative in sorted(manifests, key=lambda item: (item.count("/"), item)):
            path = root / relative
            name = path.name
            if name == "package.json":
                data = _load_json(path)
                if not isinstance(data, dict):
                    continue
                runner = _declared_package_manager(data) or "npm"
                package_managers.add(runner)
                node_dependencies: list[str] = []
                for key in ("dependencies", "devDependencies"):
                    section = data.get(key)
                    if isinstance(section, dict):
                        node_dependencies.extend(sorted(section))
                if node_dependencies:
                    dependencies.setdefault("node", []).extend(node_dependencies)
                package_scripts = data.get("scripts")
                if isinstance(package_scripts, dict):
                    for script_name, script_body in package_scripts.items():
                        scripts[str(script_name)] = str(script_body)
                    for intent, candidates in (
                        ("test", ("test",)),
                        ("build", ("build",)),
                        ("dev", ("dev", "start", "serve")),
                        ("lint", ("lint",)),
                        ("typecheck", ("typecheck", "type-check", "tsc")),
                    ):
                        for candidate in candidates:
                            if candidate in package_scripts:
                                commands.setdefault(intent, f"{runner} run {candidate}")
                                break
                main_entry = data.get("main")
                if isinstance(main_entry, str):
                    entry_points.add(main_entry)
            elif name in {"requirements.txt", "requirements-lock.txt"}:
                package_managers.add("pip")
                requirements: list[str] = []
                for line in _read_text(path).splitlines():
                    stripped = line.strip()
                    if stripped and not stripped.startswith(("#", "-")):
                        requirements.append(re.split(r"[<>=!;\[ ]", stripped, maxsplit=1)[0])
                if requirements:
                    dependencies.setdefault("python", []).extend(requirements)
            elif name == "pyproject.toml":
                package_managers.add("uv" if (root / "uv.lock").exists() else "pip")
            elif name == "go.mod":
                package_managers.add("go")
                commands.setdefault("test", "go test ./...")
                commands.setdefault("build", "go build ./...")
            elif name == "Cargo.toml":
                package_managers.add("cargo")
                commands.setdefault("test", "cargo test")
                commands.setdefault("build", "cargo build")

        for key, values in dependencies.items():
            dependencies[key] = sorted(dict.fromkeys(values))
        for candidate in ("main.py", "app.py", "manage.py", "index.js", "server.js"):
            if (root / candidate).exists():
                entry_points.add(candidate)
        return package_managers, dependencies, scripts, commands, entry_points

    @staticmethod
    def _scan_routes(root: Path, source_files: list[Path]) -> list[dict[str, str]]:
        routes: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for path in source_files:
            try:
                if path.stat().st_size > MAX_SOURCE_SCAN_BYTES:
                    continue
            except OSError:
                continue
            text = _read_text(path, MAX_SOURCE_SCAN_BYTES)
            if not text or ("app." not in text and "router." not in text and "@" not in text):
                continue
            for pattern in ROUTE_PATTERNS:
                for method, route_path in pattern.findall(text):
                    if not route_path.startswith("/"):
                        continue
                    key = (method.lower(), route_path)
                    if key in seen:
                        continue
                    seen.add(key)
                    try:
                        relative = str(path.relative_to(root)).replace("\\", "/")
                    except ValueError:
                        relative = path.name
                    routes.append({"method": method.lower(), "path": route_path, "file": relative})
        routes.sort(key=lambda item: (item["path"], item["method"]))
        return routes

    @staticmethod
    def _classify_components(root: Path, tree: dict[str, Any]) -> dict[str, list[str]]:
        """Map conventional directory names onto architectural areas."""
        buckets = {
            "frontend": ("frontend", "client", "web", "ui", "pages", "components"),
            "backend": ("backend", "server", "api"),
            "agent": ("agent", "agents", "runtime", "orchestrator"),
            "tests": ("tests", "test", "__tests__", "e2e"),
            "docs": ("docs", "documentation"),
            "infra": ("infra", "deploy", "terraform", "k8s", ".github"),
        }
        found: dict[str, list[str]] = {}
        for area, candidates in buckets.items():
            matches: list[str] = []
            for candidate in candidates:
                if (root / candidate).is_dir():
                    matches.append(candidate)
            for name in tree:
                if name.lower() in candidates and name not in matches:
                    matches.append(name)
            if matches:
                found[area] = sorted(dict.fromkeys(matches))
        return found

    @staticmethod
    def git_state(root: Path) -> dict[str, Any]:
        if not (root / ".git").exists():
            return {"repository": False}
        status = run_git(root, "status", "--porcelain")
        branch = run_git(root, "rev-parse", "--abbrev-ref", "HEAD").strip()
        head = run_git(root, "rev-parse", "--short", "HEAD").strip()
        changed = [line[3:].strip() for line in status.splitlines() if line.strip()]
        return {
            "repository": True,
            "branch": branch or None,
            "head": head or None,
            "dirty_files": len(changed),
            "changed_paths": changed[:50],
            "clean": not changed,
        }
