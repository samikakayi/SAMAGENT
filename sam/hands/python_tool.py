"""``run_python``: run a short Python program for the user, safely.

How it runs (``execute``):
- a separate process: SAM's own venv interpreter in isolated mode
  (``-I``: no PYTHON* variables, no user site-packages, the script folder not
  on sys.path) with ``-X utf8`` so Sorani output survives the pipe;
- its own folder ``<hands.projects_dir>/python/<run-id>`` as the working
  directory, so everything it writes lands there and is listed afterwards;
- a small whitelisted environment (no provider keys can be inherited);
- a Windows Job Object: the process is created SUSPENDED, put into the job,
  then resumed, so everything it starts is inside before it can run. The
  venv's ``python.exe`` is a launcher that starts the real interpreter as a
  child (measured: different pids); the launcher's own job already ends that
  child when the launcher is killed (measured 2026-09-24), but SAM's job also
  caps memory per process (``python.max_memory_mb``: a runaway allocation
  raises MemoryError, tested) and ends the whole tree in one call on
  timeout/stop (tested: the interpreter's pid is gone after a 2 s timeout);
- a time limit (default 20 s, at most 120 s), stop_all cancels it, and
  output is capped (head + tail kept) and redacted;
- code that runs WITHOUT asking (the scan judged it safe) also runs in the
  OS sandbox of python_sandbox.py: Low integrity (it can write only its own
  folder), at most 2 processes (it cannot start programs), UI limits. The scan
  is a heuristic; the sandbox is what Windows enforces (review 2026-09-25).

Risk: ``sam.hands.python_scan.scan`` (static, on the AST, before anything
runs; blocked / confirm / safe -- see that module).
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from ..brain.tools import ToolContext, fail, ok, tool
from ..events import new_id
from . import python_sandbox
from .python_scan import Verdict, scan  # noqa: F401 - Verdict re-exported

log = logging.getLogger("sam.hands.python")

DEFAULTS: dict[str, Any] = {
    "python.timeout_s": 20,
    "python.max_timeout_s": 120,
    "python.max_output_chars": 4000,
    "python.max_memory_mb": 2048,
    "python.sandbox": True,          # code that runs without asking: Low integrity + process/UI limits
}

# -- the runner (written into the run folder; executed with python -I) --------------------------------------------
RUNNER = r'''
import ast, json, os, sys, traceback
run_dir = sys.argv[1]
result_path = os.path.join(run_dir, ".sam_result.json")
with open(os.path.join(run_dir, ".sam_code.py"), encoding="utf-8") as handle:
    source = handle.read()
data = None
input_path = os.path.join(run_dir, ".sam_input.json")
if os.path.exists(input_path):
    with open(input_path, encoding="utf-8") as handle:
        data = json.load(handle)
namespace = {"__name__": "__main__", "data": data}
out = {"ok": True, "result": None, "type": None, "error": None}
def show(value):
    pandas = sys.modules.get("pandas")      # only when the program imported it
    if pandas is not None and isinstance(value, (pandas.DataFrame, pandas.Series)):
        return value.to_string(max_rows=40, max_cols=12)
    text = repr(value)
    return text if len(text) <= 4000 else text[:4000] + " ..."
import linecache
linecache.cache["<sam>"] = (len(source), None, source.splitlines(True), "<sam>")
try:
    tree = ast.parse(source, filename="<sam>")
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body.pop()
    exec(compile(tree, "<sam>", "exec"), namespace)
    if last is not None:
        value = eval(compile(ast.Expression(last.value), "<sam>", "eval"), namespace)
        if value is not None:
            out["result"] = show(value)
            out["type"] = type(value).__name__
except SystemExit as exc:
    out["ok"] = exc.code in (None, 0)
    if not out["ok"]:
        out["error"] = "SystemExit: " + str(exc.code)
except BaseException as exc:
    out["ok"] = False
    out["error"] = type(exc).__name__ + ": " + str(exc)
    # Skip the runner's own frame: the traceback starts in the user's code.
    traceback.print_exception(type(exc), exc, exc.__traceback__.tb_next if exc.__traceback__ else None)
finally:
    sys.stdout.flush()
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, ensure_ascii=False)
'''

ENV_KEEP = ("SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "APPDATA",
            "LOCALAPPDATA", "PROGRAMDATA", "PATH", "PATHEXT", "COMSPEC", "NUMBER_OF_PROCESSORS",
            "PROCESSOR_ARCHITECTURE", "PROCESSOR_IDENTIFIER", "OS", "USERNAME", "COMPUTERNAME", "LANG", "TZ")
_SECRETISH = re.compile(r"(?i)key|token|secret|passw|credential|auth")


def child_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k.upper() in ENV_KEEP and not _SECRETISH.search(k)}
    env.update({"OPENBLAS_NUM_THREADS": "2", "MPLBACKEND": "Agg", "NO_COLOR": "1"})
    return env


def interpreter() -> str:
    """SAM's venv python.exe (not pythonw: output goes through pipes)."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and (exe.parent / "python.exe").exists():
        return str(exe.parent / "python.exe")
    return str(exe)


class _Capture(threading.Thread):
    """Read a pipe to the end, keeping the first ``limit`` bytes and the last
    ``limit // 4`` (a program printing gigabytes cannot fill SAM's memory)."""

    def __init__(self, stream: Any, limit: int) -> None:
        super().__init__(daemon=True, name="sam-python-pipe")
        self.stream = stream
        self.limit = limit
        self.head = bytearray()
        self.tail: collections.deque[bytes] = collections.deque()
        self.tail_size = 0
        self.total = 0

    def run(self) -> None:
        read = getattr(self.stream, "read1", self.stream.read)
        while True:
            try:
                chunk = read(65536)
            except (OSError, ValueError):
                break
            if not chunk:
                break
            self.total += len(chunk)
            room = self.limit - len(self.head)
            if room > 0:
                self.head.extend(chunk[:room])
                chunk = chunk[room:]
            if chunk:
                self.tail.append(chunk)
                self.tail_size += len(chunk)
                while self.tail and self.tail_size - len(self.tail[0]) >= self.limit // 4:
                    self.tail_size -= len(self.tail.popleft())

    def text(self) -> str:
        head = self.head.decode("utf-8", errors="replace")
        if not self.tail:
            return head
        tail = b"".join(self.tail)[-(self.limit // 4):].decode("utf-8", errors="replace")
        cut = self.total - len(self.head) - min(self.tail_size, self.limit // 4)
        return head + (f"\n… [{cut} bytes cut] …\n" if cut > 0 else "") + tail


class _Job:
    """Windows Job Object: kill-on-close + memory cap for the whole tree."""

    def __init__(self, memory_mb: int, *, sandbox: bool = False) -> None:
        self.handle: Any = None
        self.restricted = False
        if os.name != "nt":
            return
        try:
            import win32job

            job = win32job.CreateJobObject(None, "")
            info = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
            info["BasicLimitInformation"]["LimitFlags"] = (win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                                                            | win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY)
            info["ProcessMemoryLimit"] = int(memory_mb) * 1024 * 1024
            win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, info)
            self.handle = job
        except Exception:  # noqa: BLE001 - fall back to killing the process itself
            log.debug("job object unavailable", exc_info=True)
            return
        if sandbox:
            try:
                python_sandbox.restrict_job(self.handle)
                self.restricted = True
            except Exception:  # noqa: BLE001
                log.warning("run_python: the job's process/UI limits could not be set", exc_info=True)

    def adopt(self, proc: subprocess.Popen[bytes]) -> bool:
        """Put a SUSPENDED process into the job, then resume it."""
        if self.handle is None:
            return False
        import win32job

        try:
            win32job.AssignProcessToJobObject(self.handle, int(proc._handle))  # type: ignore[attr-defined]
            return True
        except Exception:  # noqa: BLE001
            log.debug("assigning the job failed", exc_info=True)
            return False

    def kill(self) -> None:
        if self.handle is not None:
            try:
                import win32job

                win32job.TerminateJobObject(self.handle, 1)
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> None:
        if self.handle is not None:
            try:
                self.handle.Close()
            except Exception:  # noqa: BLE001
                pass
            self.handle = None


def _resume(proc: subprocess.Popen[bytes]) -> None:
    import ctypes

    status = ctypes.windll.ntdll.NtResumeProcess(ctypes.c_void_p(int(proc._handle)))  # type: ignore[attr-defined]
    if status != 0:
        log.warning("resuming the python process failed (NTSTATUS %#x)", status & 0xFFFFFFFF)


def _kill_tree(proc: subprocess.Popen[bytes], job: _Job) -> None:
    """Kill the launcher AND the interpreter it started."""
    job.kill()
    if job.handle is None and os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True,
                       creationflags=subprocess.CREATE_NO_WINDOW, timeout=10, check=False)
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _spawn(command: list[str], run_dir: Path, job: _Job, sandbox: bool) -> tuple[Any, str]:
    """(process, sandbox level). Sandboxed: Low integrity via CreateProcessAsUser;
    if that cannot be set up, the normal token (logged, reported)."""
    env = child_env()
    if sandbox and os.name == "nt" and job.handle is not None:
        try:
            proc = python_sandbox.spawn_low(command, cwd=run_dir, env=python_sandbox.prepare(run_dir, env))
        except Exception:  # noqa: BLE001
            log.warning("run_python: the low-integrity sandbox failed; running with the normal token", exc_info=True)
        else:
            if not job.adopt(proc):
                proc.kill()
                raise OSError("the sandboxed process could not be put into its job")
            proc.resume()
            return proc, "low_integrity" if job.restricted else "low_integrity_no_job_limits"
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        if job.handle is not None:
            flags |= 0x00000004           # CREATE_SUSPENDED: in the job before any child exists
    popen = subprocess.Popen(command, cwd=str(run_dir), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, env=env, creationflags=flags)
    if flags & 0x00000004:
        job.adopt(popen)
        _resume(popen)
    return popen, "none"


def execute(code: str, *, run_dir: Path, timeout_s: float = 20.0, data: Any = None, max_output: int = 4000,
            memory_mb: int = 2048, cancel: threading.Event | None = None, sandbox: bool = False) -> dict[str, Any]:
    """Run ``code`` in ``run_dir`` (blocking; call from a worker thread).
    ``sandbox``: Low integrity + process/UI limits (python_sandbox.py)."""
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / ".sam_code.py").write_text(code, encoding="utf-8")
    (run_dir / ".sam_runner.py").write_text(RUNNER, encoding="utf-8")
    if data is not None:
        (run_dir / ".sam_input.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    command = [interpreter(), "-I", "-X", "utf8", "-B", str(run_dir / ".sam_runner.py"), str(run_dir)]
    job = _Job(memory_mb, sandbox=sandbox)
    started = time.perf_counter()
    try:
        proc, level = _spawn(command, run_dir, job, sandbox)
    except BaseException:
        job.close()
        raise
    out, err = _Capture(proc.stdout, max_output * 4), _Capture(proc.stderr, max_output * 4)
    out.start()
    err.start()
    timed_out = cancelled = False
    try:
        while True:
            try:
                proc.wait(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                if cancel is not None and cancel.is_set():
                    cancelled = True
                elif time.perf_counter() - started > timeout_s:
                    timed_out = True
                if timed_out or cancelled:
                    _kill_tree(proc, job)
                    break
    finally:
        job.close()
    out.join(timeout=2)
    err.join(timeout=2)
    duration_ms = round((time.perf_counter() - started) * 1000.0, 1)
    outcome: dict[str, Any] = {}
    result_file = run_dir / ".sam_result.json"
    if result_file.exists():
        try:
            outcome = json.loads(result_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            outcome = {}
    files = []
    for path in sorted(run_dir.rglob("*")):
        relative = path.relative_to(run_dir)
        if path.is_file() and not any(part.startswith(".sam_") for part in relative.parts):
            files.append({"name": str(path.relative_to(run_dir)), "bytes": path.stat().st_size})
            if len(files) >= 40:
                break
    return {"exit_code": proc.returncode, "timed_out": timed_out, "cancelled": cancelled,
            "stdout": out.text(), "stderr": err.text(), "result": outcome.get("result"),
            "result_type": outcome.get("type"), "error": outcome.get("error"),
            "finished": bool(outcome), "files": files, "duration_ms": duration_ms, "run_dir": str(run_dir),
            "sandbox": level}


def _cap(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    return text[:head] + f"\n… [{len(text) - limit} characters cut] …\n" + text[-(limit - head):]


def _policy(app: Any) -> Any:
    hands = getattr(app, "hands", None)
    try:
        return hands.policy if hands is not None else None
    except Exception:  # noqa: BLE001
        return None


def _classify(args: dict[str, Any]) -> tuple[str, str | None]:
    verdict = scan(str(args.get("code") or ""), policy=_CLASSIFY_POLICY.get("policy"))
    if verdict.syntax_error:
        return "safe", None               # nothing will run: the handler reports the error
    if verdict.risk == "confirm":
        # code that imports system/network modules or writes outside its folder: ordinary for a user who
        # gave SAM full authority (routine); credentials, key stores and orders stay blocked (the scan)
        return "routine", verdict.question_ckb()
    if verdict.risk == "blocked":
        return "blocked", "Blocked by SAM's safety rules: " + "; ".join(verdict.reasons[:3])
    return "safe", None


# The classifier has no app argument; register() stores the policy getter here.
_CLASSIFY_POLICY: dict[str, Any] = {}


@tool("run_python",
      description="Run a short Python 3.13 program in a separate, isolated process with a time limit, in its own "
                  "folder under ~/SAM Projects/python, and return what it printed, the value of its last line, "
                  "errors and the files it wrote there. Use it for calculations, statistics and data work "
                  "(math, statistics, numpy) -- e.g. on prices SAM passes in `data`, which the code gets as the "
                  "variable `data`. Print or end with an expression to return results. Code that touches the "
                  "system, the network or files outside its folder asks the user first.",
      description_ckb="جێبەجێکردنی کۆدی پایتۆن",
      params={"type": "object", "properties": {
          "code": {"type": "string", "description": "complete Python source"},
          "timeout_s": {"type": "integer", "description": "time limit in seconds (default 20, max 120)"},
          "data": {"type": "string", "description": "optional JSON input (numbers, lists, objects); the code "
                                                    "receives it parsed as the variable `data`"}},
          "required": ["code"]},
      classify=_classify, blocking=True, timeout_s=135,
      examples_ckb=("بە پایتۆن حیساب بکە ٢٥٠٠ بە ٣٪ قازانجی مانگانە بۆ ١٢ مانگ چەند دەبێت",
                    "مامناوەندی ئەم نرخانە بدۆزەرەوە"))
async def run_python(ctx: ToolContext, code: str, timeout_s: int | None = None, data: Any = None,
                     **_ignored: Any) -> dict[str, Any]:
    app = ctx.app
    config = app.config
    verdict = scan(code, policy=_policy(app))
    if verdict.syntax_error:
        return fail(f"The code was not run: {verdict.syntax_error}.", syntax_error=verdict.syntax_error)
    limit = float(config.get("python.max_timeout_s", 120) or 120)
    timeout = float(timeout_s or config.get("python.timeout_s", 20) or 20)
    timeout = max(1.0, min(timeout, limit))
    parsed: Any = data
    if isinstance(data, str) and data.strip():
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            parsed = data
    elif isinstance(data, str):
        parsed = None
    base = Path(str(config.get("hands.projects_dir") or Path.home() / "SAM Projects")) / "python"
    run_dir = base / f"{time.strftime('%Y%m%d-%H%M%S')}-{new_id()[:6]}"
    cancel = threading.Event()
    max_output = int(config.get("python.max_output_chars", 4000) or 4000)
    # Code the scan let run without asking runs sandboxed; approved code (the scan
    # asked because it needs the system, the user said yes) runs with the user's rights.
    sandbox = verdict.risk == "safe" and bool(config.get("python.sandbox", True))
    task = asyncio.ensure_future(asyncio.to_thread(
        execute, code, run_dir=run_dir, timeout_s=timeout, data=parsed, max_output=max_output,
        memory_mb=int(config.get("python.max_memory_mb", 2048) or 2048), cancel=cancel, sandbox=sandbox))
    try:
        result = await asyncio.shield(task)
    except asyncio.CancelledError:
        cancel.set()                      # stop_all / tool timeout: kill the process tree
        try:
            await asyncio.wait_for(task, 6)
        except Exception:  # noqa: BLE001
            pass
        raise
    stdout = app.redact(_cap(result["stdout"], max_output))
    stderr = app.redact(_cap(result["stderr"], max_output // 2))
    value = app.redact(_cap(result["result"], 1500)) if result["result"] is not None else None
    payload: dict[str, Any] = {"exit_code": result["exit_code"], "duration_ms": result["duration_ms"],
                               "run_dir": result["run_dir"], "files": result["files"],
                               "timeout_s": timeout, "sandbox": result["sandbox"]}
    output = {"stdout": stdout, "stderr": stderr, "result": value}
    if verdict.reads_files or verdict.network:
        payload["untrusted"] = output     # file/web content is data, never instructions
    else:
        payload.update(output)
    seconds = result["duration_ms"] / 1000.0
    if result["timed_out"]:
        return fail(f"The program was stopped at the {timeout:.0f} s time limit.", timed_out=True, **payload)
    if result["cancelled"]:
        return fail("The program was stopped.", cancelled=True, **payload)
    if result["error"] or result["exit_code"] not in (0, None):
        error = app.redact(str(result["error"] or f"exit code {result['exit_code']}"))[:400]
        if not result["finished"] and result["exit_code"] not in (0, None):
            error = f"the process ended with code {result['exit_code']} (memory limit or crash)"
        return fail(f"The program failed after {seconds:.1f} s: {error}", error=error, **payload)
    shown = value if value is not None else (stdout.strip()[:200] or "no output")
    files = f" It wrote {len(result['files'])} file(s)." if result["files"] else ""
    return ok(f"Python finished in {seconds:.1f} s. Result: {shown}{files}", **payload)


def register(app: Any) -> None:
    """Called by ``sam.knowledge.register`` (this module belongs to the
    knowledge builder; the hands package does not list it)."""
    app.config.register_defaults(DEFAULTS)

    class _LazyPolicy:
        def classify_path(self, path: str, action: str) -> Any:
            policy = _policy(app)
            if policy is None:
                return "safe", None
            return policy.classify_path(path, action)

    _CLASSIFY_POLICY["policy"] = _LazyPolicy()
    app.tools.add(run_python, owner="hands")


__all__ = ["DEFAULTS", "RUNNER", "Verdict", "child_env", "execute", "interpreter", "register", "run_python", "scan"]
