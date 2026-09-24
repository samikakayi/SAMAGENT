"""The OS-level sandbox for ``run_python`` code that runs without asking.

The static scan (python_scan.py) decides whether a program asks first; a
static scan of Python can only be a heuristic, so code it judged safe also
runs with fewer rights than the user (review 2026-09-25):

- LOW integrity: the child gets a copy of SAM's token lowered to the Low
  mandatory level (S-1-16-4096). Windows then refuses its writes to anything
  the user owns at the normal (Medium) level -- documents, the desktop, SAM's
  data, the registry -- except its own run folder, which is labelled Low
  first. Measured on this PC (2026-09-25): a write to the home folder raised
  PermissionError, a write in the run folder worked, numpy imported and ran,
  and start-up took the same ~0.3 s.
- a process cap in the Job Object: 2 = the venv launcher and the interpreter
  it starts, so a safe program cannot start other programs. Measured on this
  PC (2026-09-25): the job holds 3 processes (Windows adds a console host,
  which the cap does not count); cap 1 -> the launcher cannot start the
  interpreter; cap 2 -> the program runs and ``cmd`` is refused (8 of 8 runs);
  cap 3 -> ``cmd`` starts;
- UI limits: no clipboard, no other processes' windows, no desktop switching,
  no logoff/shutdown, no system settings.

Code the user approved (the scan asked, the user said yes) runs at the normal
level: it was asked because it needs the system. If the sandbox cannot be set
up the run falls back to the normal token and says so (``sandbox`` in the
result), because a missing sandbox must never make the tool unusable silently.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger("sam.hands.python")

LOW_INTEGRITY_SID = "S-1-16-4096"
PROCESS_LIMIT = 2        # launcher + interpreter; the console host is not counted (measured 2026-09-25)
CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_SUSPENDED = 0x00000004
CREATE_UNICODE_ENVIRONMENT = 0x00000400


def ui_limits() -> int:
    import win32job

    return (win32job.JOB_OBJECT_UILIMIT_DESKTOP | win32job.JOB_OBJECT_UILIMIT_DISPLAYSETTINGS
            | win32job.JOB_OBJECT_UILIMIT_EXITWINDOWS | win32job.JOB_OBJECT_UILIMIT_GLOBALATOMS
            | win32job.JOB_OBJECT_UILIMIT_HANDLES | win32job.JOB_OBJECT_UILIMIT_READCLIPBOARD
            | win32job.JOB_OBJECT_UILIMIT_SYSTEMPARAMETERS | win32job.JOB_OBJECT_UILIMIT_WRITECLIPBOARD)


def restrict_job(job: Any) -> None:
    """Add the sandbox limits (process cap, UI limits) to an existing job."""
    import win32job

    info = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
    info["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_ACTIVE_PROCESS
    info["BasicLimitInformation"]["ActiveProcessLimit"] = PROCESS_LIMIT
    win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, info)
    win32job.SetInformationJobObject(job, win32job.JobObjectBasicUIRestrictions, {"UIRestrictionsClass": ui_limits()})


def label_low(folder: Path) -> None:
    """Give ``folder`` (and what is created in it later) the Low integrity label,
    so a Low-integrity program may write there and nowhere else of the user's."""
    import win32security

    sid = win32security.ConvertStringSidToSid(LOW_INTEGRITY_SID)
    sacl = win32security.ACL()
    sacl.AddMandatoryAce(win32security.ACL_REVISION,
                         win32security.OBJECT_INHERIT_ACE | win32security.CONTAINER_INHERIT_ACE,
                         win32security.SYSTEM_MANDATORY_LABEL_NO_WRITE_UP, sid)
    win32security.SetNamedSecurityInfo(str(folder), win32security.SE_FILE_OBJECT,
                                       win32security.LABEL_SECURITY_INFORMATION, None, None, None, sacl)


def low_token() -> Any:
    """A primary token like SAM's own, lowered to the Low integrity level."""
    import win32api
    import win32con
    import win32security

    own = win32security.OpenProcessToken(win32api.GetCurrentProcess(),
                                         win32con.TOKEN_DUPLICATE | win32con.TOKEN_QUERY
                                         | win32con.TOKEN_ADJUST_DEFAULT | win32con.TOKEN_ASSIGN_PRIMARY)
    try:
        token = win32security.DuplicateTokenEx(own, win32security.SecurityImpersonation,
                                               win32con.TOKEN_QUERY | win32con.TOKEN_DUPLICATE
                                               | win32con.TOKEN_ASSIGN_PRIMARY | win32con.TOKEN_ADJUST_DEFAULT,
                                               win32security.TokenPrimary)
    finally:
        own.Close()
    sid = win32security.ConvertStringSidToSid(LOW_INTEGRITY_SID)
    win32security.SetTokenInformation(token, win32security.TokenIntegrityLevel, (sid, win32security.SE_GROUP_INTEGRITY))
    return token


class LowProcess:
    """The part of ``subprocess.Popen`` that ``python_tool.execute`` uses, for a
    process started with ``CreateProcessAsUser`` (Popen cannot take a token)."""

    def __init__(self, process: Any, thread: Any, pid: int, stdout: Any, stderr: Any) -> None:
        self._process = process               # PyHANDLE: kept alive with this object
        self._thread = thread
        self._handle = int(process)
        self.pid = pid
        self.stdout = stdout
        self.stderr = stderr
        self.returncode: int | None = None
        self._lock = threading.Lock()

    def resume(self) -> None:
        import win32process

        win32process.ResumeThread(self._thread)

    def poll(self) -> int | None:
        import win32event
        import win32process

        with self._lock:
            if self.returncode is None and win32event.WaitForSingleObject(self._process, 0) == win32event.WAIT_OBJECT_0:
                self.returncode = int(win32process.GetExitCodeProcess(self._process))
            return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        import win32event

        millis = win32event.INFINITE if timeout is None else max(0, int(timeout * 1000))
        if win32event.WaitForSingleObject(self._process, millis) == win32event.WAIT_TIMEOUT:
            raise subprocess.TimeoutExpired("python", timeout or 0)
        code = self.poll()
        return 0 if code is None else code

    def kill(self) -> None:
        import win32process

        try:
            win32process.TerminateProcess(self._process, 1)
        except Exception:  # noqa: BLE001 - already gone
            pass


def spawn_low(command: list[str], *, cwd: Path, env: dict[str, str]) -> LowProcess:
    """Start ``command`` SUSPENDED at Low integrity with piped stdout/stderr
    (the caller puts it into its job, then calls ``resume``)."""
    import msvcrt

    import pywintypes
    import win32api
    import win32con
    import win32file
    import win32pipe
    import win32process

    token = low_token()
    inherit = pywintypes.SECURITY_ATTRIBUTES()
    inherit.bInheritHandle = True
    out_r, out_w = win32pipe.CreatePipe(inherit, 0)
    err_r, err_w = win32pipe.CreatePipe(inherit, 0)
    for handle in (out_r, err_r):   # SAM's ends stay in SAM
        win32api.SetHandleInformation(handle, win32con.HANDLE_FLAG_INHERIT, 0)
    null = win32file.CreateFile("NUL", win32con.GENERIC_READ, win32con.FILE_SHARE_READ | win32con.FILE_SHARE_WRITE,
                                inherit, win32con.OPEN_EXISTING, 0, None)
    info = win32process.STARTUPINFO()
    info.dwFlags = win32con.STARTF_USESTDHANDLES
    info.hStdInput, info.hStdOutput, info.hStdError = null, out_w, err_w
    flags = CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP | CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT
    try:
        process, thread, pid, _tid = win32process.CreateProcessAsUser(
            token, None, subprocess.list2cmdline(command), None, None, True, flags, env, str(cwd), info)
    except Exception:
        for handle in (out_r, err_r):
            handle.Close()
        raise
    finally:
        for handle in (out_w, err_w, null):   # the child has its copies
            handle.Close()
        token.Close()
    stdout = os.fdopen(msvcrt.open_osfhandle(out_r.Detach(), os.O_RDONLY), "rb", buffering=0)
    stderr = os.fdopen(msvcrt.open_osfhandle(err_r.Detach(), os.O_RDONLY), "rb", buffering=0)
    return LowProcess(process, thread, pid, stdout, stderr)


def prepare(run_dir: Path, env: dict[str, str]) -> dict[str, str]:
    """Label the run folder Low and point the child's temp/cache folders into
    it (a Low process cannot write %TEMP% or ~/.matplotlib). Returns the env."""
    label_low(run_dir)
    temp = run_dir / ".sam_tmp"
    temp.mkdir(exist_ok=True)
    out = dict(env)
    out.update({"TEMP": str(temp), "TMP": str(temp), "MPLCONFIGDIR": str(temp)})
    return out


__all__ = ["LowProcess", "PROCESS_LIMIT", "label_low", "low_token", "prepare", "restrict_job", "spawn_low",
           "ui_limits"]
