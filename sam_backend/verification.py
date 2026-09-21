"""Verification: proving work actually holds, rather than asserting it did.

"Done" is only meaningful if something checked it. This module turns the
project map's discovered commands into runnable checks, executes them with
bounded time and captured output, and parses the result into a structured
verdict the orchestrator can branch on.

A check that could not be run is reported as SKIPPED with a reason. It is
never silently treated as a pass -- a fabricated green is worse than an
honest "I could not verify this".
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from .project_map import ProjectMap

MAX_CAPTURED_CHARS = 20_000
DEFAULT_TIMEOUT = 600


class CheckOutcome(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"


@dataclass(slots=True)
class CheckResult:
    """One executed (or deliberately skipped) verification check."""

    kind: str  # test | build | lint | typecheck | smoke
    command: str
    outcome: CheckOutcome
    exit_code: int | None = None
    duration_ms: float = 0.0
    passed_count: int | None = None
    failed_count: int | None = None
    summary: str = ""
    failures: list[str] = field(default_factory=list)
    stdout_tail: str = ""
    stderr_tail: str = ""
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.outcome is CheckOutcome.PASSED

    @property
    def blocking(self) -> bool:
        """Did this check actively prove a problem, rather than not run?"""
        return self.outcome in {CheckOutcome.FAILED, CheckOutcome.ERROR, CheckOutcome.TIMEOUT}

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outcome"] = self.outcome.value
        payload["ok"] = self.ok
        return payload


@dataclass(slots=True)
class VerificationReport:
    """The aggregate verdict over every check run for one task."""

    checks: list[CheckResult] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    @property
    def ok(self) -> bool:
        """True when nothing failed. Skipped checks do not fail a report, but
        they are surfaced so the caller can say what was not verified."""
        return not any(check.blocking for check in self.checks)

    @property
    def verified(self) -> bool:
        """True only when at least one check actually ran and all passed."""
        return self.ok and any(check.outcome is CheckOutcome.PASSED for check in self.checks)

    @property
    def failures(self) -> list[CheckResult]:
        return [check for check in self.checks if check.blocking]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "verified": self.verified,
            "checks": [check.as_dict() for check in self.checks],
            "started_at": self.started_at,
        }

    def summary_text(self) -> str:
        if not self.checks:
            return "No verification checks were available for this project."
        lines = []
        for check in self.checks:
            if check.outcome is CheckOutcome.SKIPPED:
                lines.append(f"{check.kind}: SKIPPED ({check.reason})")
            elif check.passed_count is not None or check.failed_count is not None:
                lines.append(
                    f"{check.kind}: {check.outcome.value} "
                    f"({check.passed_count or 0} passed, {check.failed_count or 0} failed) via `{check.command}`"
                )
            else:
                lines.append(f"{check.kind}: {check.outcome.value} via `{check.command}`")
        for check in self.failures:
            for failure in check.failures[:10]:
                lines.append(f"  - {failure}")
        return "\n".join(lines)


# -- output parsing --------------------------------------------------------
# Each parser is intentionally narrow: it recognises its own runner's summary
# line and returns None otherwise, so an unknown runner falls back to the exit
# code rather than being mis-parsed into a confident wrong number.

_PYTEST_TALLY = re.compile(r"(\d+)\s+(passed|failed|errors?|skipped|xfailed|xpassed)\b")
_PYTEST_LINE = re.compile(r"^(FAILED|ERROR)\s+(\S+.*)$", re.MULTILINE)
_JEST_SUMMARY = re.compile(r"Tests:\s+(?:(\d+) failed,\s*)?(?:(\d+) skipped,\s*)?(?:(\d+) passed,\s*)?(\d+) total")
_VITEST_SUMMARY = re.compile(r"Tests\s+(?:(\d+) failed \| )?(\d+) passed")
_GO_FAIL = re.compile(r"^--- FAIL: (\S+)", re.MULTILINE)
_CARGO_SUMMARY = re.compile(r"test result: \w+\. (\d+) passed; (\d+) failed")
_TSC_ERROR = re.compile(r"^(\S+\(\d+,\d+\): error TS\d+: .*)$", re.MULTILINE)


def parse_test_output(text: str) -> tuple[int | None, int | None, list[str]]:
    """Best-effort (passed, failed, failure names) from a test runner's output."""
    failures: list[str] = []

    match = _JEST_SUMMARY.search(text)
    if match:
        failed = int(match.group(1) or 0)
        passed = int(match.group(3) or 0)
        failures = [line.strip() for line in re.findall(r"^\s*●\s+(.+)$", text, re.MULTILINE)][:20]
        return passed, failed, failures

    match = _VITEST_SUMMARY.search(text)
    if match:
        return int(match.group(2) or 0), int(match.group(1) or 0), failures

    match = _CARGO_SUMMARY.search(text)
    if match:
        return int(match.group(1)), int(match.group(2)), failures

    if "--- FAIL:" in text or "\nok  \t" in text or "\nFAIL\t" in text:
        failures = _GO_FAIL.findall(text)[:20]
        return None, len(failures) or None, failures

    # pytest's tally is on its last summary line, which is decorated with "="
    # in default mode but bare under -q ("9 passed, 1 warning in 0.10s"), so
    # the counts are read from the line's own tokens rather than its framing.
    for line in reversed(text.strip().splitlines()[-20:]):
        tokens = _PYTEST_TALLY.findall(line)
        if not tokens:
            continue
        tally: dict[str, int] = {}
        for count, label in tokens:
            label = "error" if label.startswith("error") else label
            tally[label] = tally.get(label, 0) + int(count)
        if not ({"passed", "failed", "error"} & tally.keys()):
            continue
        failures = [f"{kind} {name}".strip() for kind, name in _PYTEST_LINE.findall(text)][:20]
        return tally.get("passed", 0), tally.get("failed", 0) + tally.get("error", 0), failures
    return None, None, failures


def parse_typecheck_output(text: str) -> list[str]:
    return _TSC_ERROR.findall(text)[:20]


def _tail(text: str, limit: int = 4000) -> str:
    text = text.strip()
    return text if len(text) <= limit else "...\n" + text[-limit:]


class VerificationEngine:
    """Runs a project's own checks and reports structured verdicts."""

    # Only commands a project itself declares are runnable, and only when
    # their runner is one of these. An arbitrary string from a manifest is
    # not executed blindly.
    ALLOWED_RUNNERS = {
        "python", "python3", "py", "pytest", "npm", "pnpm", "yarn", "bun", "npx",
        "node", "go", "cargo", "dotnet", "mvn", "gradle", "make", "tsc", "ruff",
        "eslint", "mypy", "jest", "vitest", "uv", "poetry",
    }

    def __init__(self, *, timeout: int = DEFAULT_TIMEOUT, python_executable: str | None = None) -> None:
        self.timeout = timeout
        self.python_executable = python_executable

    def available_checks(self, project_map: ProjectMap) -> dict[str, str]:
        """Which check kinds this project actually declares."""
        return {
            kind: command
            for kind, command in project_map.commands.items()
            if kind in {"test", "build", "lint", "typecheck"} and command
        }

    def run_check(self, kind: str, command: str, cwd: Path, *, timeout: int | None = None) -> CheckResult:
        started = time.perf_counter()
        argv = self._argv(command, cwd)
        if argv is None:
            return CheckResult(
                kind=kind, command=command, outcome=CheckOutcome.SKIPPED,
                reason="The declared command is not a recognised test/build runner, so it was not executed.",
            )
        try:
            completed = subprocess.run(
                argv, cwd=str(cwd), capture_output=True, text=True,
                timeout=timeout or self.timeout, encoding="utf-8", errors="replace",
                env=self._environment(),
            )
        except FileNotFoundError:
            return CheckResult(
                kind=kind, command=command, outcome=CheckOutcome.SKIPPED,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                reason=f"{argv[0]} is not installed on this machine.",
            )
        except subprocess.TimeoutExpired:
            return CheckResult(
                kind=kind, command=command, outcome=CheckOutcome.TIMEOUT,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                reason=f"The command exceeded {timeout or self.timeout}s.",
                summary="Timed out before producing a verdict.",
            )
        except OSError as exc:
            return CheckResult(
                kind=kind, command=command, outcome=CheckOutcome.ERROR,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                reason=str(exc),
            )

        duration = round((time.perf_counter() - started) * 1000, 2)
        combined = f"{completed.stdout}\n{completed.stderr}"[:MAX_CAPTURED_CHARS]
        passed_count = failed_count = None
        failures: list[str] = []
        if kind == "test":
            passed_count, failed_count, failures = parse_test_output(combined)
        elif kind == "typecheck":
            failures = parse_typecheck_output(combined)

        # The exit code is the authority. Parsed counts add detail; they never
        # override a non-zero exit into a pass.
        outcome = CheckOutcome.PASSED if completed.returncode == 0 else CheckOutcome.FAILED
        if completed.returncode == 0 and failed_count:
            outcome = CheckOutcome.FAILED
        return CheckResult(
            kind=kind,
            command=command,
            outcome=outcome,
            exit_code=completed.returncode,
            duration_ms=duration,
            passed_count=passed_count,
            failed_count=failed_count,
            failures=failures,
            summary=self._headline(combined, outcome),
            stdout_tail=_tail(completed.stdout),
            stderr_tail=_tail(completed.stderr),
        )

    def verify(
        self, project_map: ProjectMap, cwd: Path, *, kinds: list[str] | None = None,
    ) -> VerificationReport:
        report = VerificationReport()
        available = self.available_checks(project_map)
        wanted = kinds or ["typecheck", "lint", "build", "test"]
        for kind in wanted:
            command = available.get(kind)
            if not command:
                continue
            report.checks.append(self.run_check(kind, command, cwd))
        if not report.checks:
            report.checks.append(
                CheckResult(
                    kind="test", command="", outcome=CheckOutcome.SKIPPED,
                    reason="This project declares no test, build, lint or typecheck command.",
                )
            )
        return report

    # -- internals ---------------------------------------------------------
    @staticmethod
    def split_command(command: str) -> list[str] | None:
        """Split a command line without mangling Windows paths.

        shlex in posix mode eats backslashes ("C:\\venv\\python" -> "C:venvpython");
        in non-posix mode it keeps the quote characters inside the tokens, so
        `python -c "print(1)"` is passed to the child with literal quotes and
        fails to parse. Disabling escapes while keeping posix quote handling
        gets both right.
        """
        lexer = shlex.shlex(command, posix=True)
        lexer.whitespace_split = True
        lexer.escape = ""
        try:
            return list(lexer)
        except ValueError:
            return None

    def _resolve_python(self, cwd: Path) -> str:
        """The interpreter that owns this project, not whatever is on PATH.

        Running a project's tests with an unrelated interpreter produces
        confusing "No module named pytest" failures that look like the code
        is broken when only the environment is wrong.
        """
        if self.python_executable:
            return self.python_executable
        names = ("Scripts/python.exe",) if os.name == "nt" else ("bin/python3", "bin/python")
        for environment in (".venv", "venv", "env"):
            for name in names:
                candidate = cwd / environment / name
                if candidate.is_file():
                    return str(candidate)
        return sys.executable

    def _argv(self, command: str, cwd: Path) -> list[str] | None:
        argv = self.split_command(command)
        if not argv:
            return None
        # PowerShell wrappers (".\run-tests.ps1") are project scripts, not
        # runners this engine can reason about; the caller may still run them
        # through the approved terminal tool.
        head = argv[0].strip('"').strip("'")
        runner = Path(head).name.lower()
        runner = runner[:-4] if runner.endswith(".exe") else runner
        if runner not in self.ALLOWED_RUNNERS:
            return None
        if runner in {"python", "python3", "py"}:
            argv[0] = self._resolve_python(cwd)
            return argv
        # Popen does not consult PATHEXT, so on Windows a bare "npm" is not
        # found even though "npm.cmd" is on PATH -- and the check would be
        # reported as "not installed". which() resolves it the way a shell would.
        resolved = shutil.which(head)
        if resolved:
            argv[0] = resolved
        return argv

    @staticmethod
    def _environment() -> dict[str, str]:
        environment = dict(os.environ)
        # Deterministic, non-interactive output from common runners.
        environment.update({"CI": "1", "FORCE_COLOR": "0", "NO_COLOR": "1", "PYTHONIOENCODING": "utf-8"})
        return environment

    @staticmethod
    def _headline(text: str, outcome: CheckOutcome) -> str:
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        if not lines:
            return outcome.value
        if outcome is CheckOutcome.PASSED:
            return lines[-1][:300]
        for line in reversed(lines):
            if re.search(r"(?i)\b(error|failed|failure|exception|traceback)\b", line):
                return line[:300]
        return lines[-1][:300]


# -- UI smoke ------------------------------------------------------------
UI_SUFFIXES = {".html", ".htm", ".css", ".scss", ".sass", ".less", ".vue", ".svelte", ".jsx", ".tsx"}
# Plain .js/.ts is ambiguous (a Node server is also .js); it only counts as UI
# work when it lives somewhere a front end conventionally lives.
SCRIPT_SUFFIXES = {".js", ".ts", ".mjs"}
UI_DIRECTORIES = {"frontend", "client", "web", "ui", "pages", "components", "public", "static", "app", "views", "templates"}
DEFAULT_UI_PORTS = (3000, 5173, 8080, 8000, 4200, 4321, 1234)


def is_ui_work(paths: list[str]) -> bool:
    """Did these edits plausibly change what a browser renders?"""
    for raw in paths:
        path = Path(str(raw).replace("\\", "/"))
        suffix = path.suffix.lower()
        parts = {part.lower() for part in path.parts[:-1]}
        if suffix in UI_SUFFIXES:
            return True
        if suffix in SCRIPT_SUFFIXES and parts & UI_DIRECTORIES:
            return True
    return False


def _url_answers(url: str, timeout: float = 2.0) -> bool:
    """Any HTTP response at all means something is listening and serving.

    A raw TCP connect goes first with a short timeout: some Windows setups
    silently drop connections to closed local ports instead of refusing them,
    and paying a full HTTP timeout per dead candidate made readiness polling
    crawl.
    """
    parsed = urllib.parse.urlsplit(url)
    try:
        socket.create_connection((parsed.hostname or "127.0.0.1", parsed.port or 80), timeout=0.5).close()
    except OSError:
        return False
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except urllib.error.HTTPError:
        return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _first_answering(urls: list[str]) -> str | None:
    """Probe every candidate at once; the first live one wins."""
    with ThreadPoolExecutor(max_workers=max(1, len(urls))) as pool:
        results = list(pool.map(_url_answers, urls))
    return next((url for url, alive in zip(urls, results) if alive), None)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@dataclass(slots=True)
class UiTarget:
    """Where the page can be reached, and whether this engine started it."""

    url: str | None
    process: Any = None
    started: bool = False
    reason: str = ""

    def stop(self) -> None:
        """Only stop what this engine started; a user's own dev server is theirs."""
        if self.process is None or not self.started:
            return
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW, check=False,
                )
            self.process.kill()
        except OSError:
            pass
        self.process = None


@dataclass(slots=True)
class UiSmoke:
    check: CheckResult
    screenshot: Path | None = None
    url: str | None = None


class UiSmokeRunner:
    """Bring the project's page up, open it in an isolated browser, judge it."""

    def __init__(self, engine: "VerificationEngine", *, ready_timeout: float = 40.0) -> None:
        self.engine = engine
        self.ready_timeout = ready_timeout

    # -- server ----------------------------------------------------------
    def _candidate_urls(self, command: str, project_map: ProjectMap) -> list[str]:
        # "npm run dev" says nothing about the port; the script it runs
        # ("vite --port 3458") usually does.
        text = command
        script_match = re.match(r"^\s*(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?(\S+)", command)
        if script_match:
            text += " " + project_map.scripts.get(script_match.group(1), "")
        ports = [int(match) for match in re.findall(r"\b(\d{4,5})\b", text) if 1024 <= int(match) <= 65535]
        for default in DEFAULT_UI_PORTS:
            if default not in ports:
                ports.append(default)
        return [f"http://127.0.0.1:{port}/" for port in ports]

    def _spawn(self, argv: list[str], cwd: Path) -> Any:
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        return subprocess.Popen(
            argv, cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=self.engine._environment(), creationflags=flags, start_new_session=os.name != "nt",
        )

    def _wait_ready(self, urls: list[str], process: Any | None) -> str | None:
        deadline = time.monotonic() + self.ready_timeout
        while time.monotonic() < deadline:
            url = _first_answering(urls)
            if url:
                return url
            if process is not None and process.poll() is not None:
                return None  # the server exited before answering
            time.sleep(0.5)
        return None

    def bring_up(self, project_map: ProjectMap, cwd: Path) -> UiTarget:
        """Reuse a running dev server, start the declared one, or serve the
        static page with Python. Each fallback is reported, not hidden."""
        dev_command = project_map.commands.get("dev", "")
        if dev_command:
            urls = self._candidate_urls(dev_command, project_map)
            already = _first_answering(urls)
            if already:
                return UiTarget(url=already, started=False)
            argv = self.engine._argv(dev_command, cwd)
            if argv is None:
                return UiTarget(url=None, reason=f"The dev command `{dev_command}` is not a recognised runner, so it was not started.")
            try:
                process = self._spawn(argv, cwd)
            except OSError as exc:
                return UiTarget(url=None, reason=f"The dev server could not start: {exc}")
            url = self._wait_ready(urls, process)
            target = UiTarget(url=url, process=process, started=True)
            if url is None:
                target.stop()
                target.reason = f"`{dev_command}` did not answer on any expected port within {self.ready_timeout:.0f}s."
            return target

        # No dev command: a static page can still be served and judged.
        static_root = next(
            (cwd / candidate for candidate in ("", "frontend", "public", "static", "dist", "web", "client")
             if (cwd / candidate / "index.html").is_file()),
            None,
        )
        if static_root is None:
            return UiTarget(url=None, reason="No dev command is declared and no index.html was found to serve.")
        port = _free_port()
        try:
            process = self._spawn(
                [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1", "--directory", str(static_root)],
                cwd,
            )
        except OSError as exc:
            return UiTarget(url=None, reason=f"The static server could not start: {exc}")
        url = self._wait_ready([f"http://127.0.0.1:{port}/"], process)
        target = UiTarget(url=url, process=process, started=True)
        if url is None:
            target.stop()
            target.reason = "The static server did not become ready."
        return target

    # -- browser ---------------------------------------------------------
    def run(self, project_map: ProjectMap, cwd: Path, *, screenshot_dir: Path, name: str) -> UiSmoke:
        started = time.perf_counter()
        target = self.bring_up(project_map, cwd)
        if target.url is None:
            return UiSmoke(CheckResult(kind="ui_smoke", command="", outcome=CheckOutcome.SKIPPED, reason=target.reason))
        try:
            from .browser_automation import BrowserAutomationError, run_browser_workflow

            screenshot_dir.mkdir(parents=True, exist_ok=True)
            screenshot = screenshot_dir / f"{name}.png"
            try:
                output = run_browser_workflow(
                    {"url": target.url, "headless": True, "timeout_ms": 20_000,
                     "actions": [{"type": "screenshot", "path": str(screenshot), "full_page": True}]},
                    workspace=cwd, max_output_chars=MAX_CAPTURED_CHARS,
                    screenshot_root=screenshot_dir, collect_console=True,
                )
            except BrowserAutomationError as exc:
                return UiSmoke(
                    CheckResult(kind="ui_smoke", command=target.url, outcome=CheckOutcome.SKIPPED,
                                duration_ms=round((time.perf_counter() - started) * 1000, 2), reason=str(exc)),
                    url=target.url,
                )
            problems = list(output.get("console_errors") or []) + list(output.get("failed_requests") or [])
            outcome = CheckOutcome.FAILED if problems else CheckOutcome.PASSED
            check = CheckResult(
                kind="ui_smoke", command=target.url, outcome=outcome, exit_code=0 if not problems else 1,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                failed_count=len(problems) or None,
                failures=problems[:20],
                summary=(f"{len(problems)} browser problem(s) on {target.url}" if problems
                         else f"Page loaded cleanly: {output.get('title') or target.url}"),
                stderr_tail="\n".join(problems)[:4000],
            )
            return UiSmoke(check, screenshot=screenshot if screenshot.is_file() else None, url=target.url)
        finally:
            target.stop()
