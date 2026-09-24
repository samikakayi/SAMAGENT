"""Live check: the Ollama server SAM starts never outlives SAM (run by hand).

    .venv\\Scripts\\python.exe acceptance\\ollama_orphan.py [--port 11437]

A child Python process (standing in for SAM) starts ``OllamaServer`` on a
spare port with an EMPTY model folder (nothing is loaded, SAM v1's server on
11434 is never touched), prints the server's pid, and is then killed hard
(TerminateProcess: no clean-up code runs). The server must be gone within a
few seconds and the port closed: the Job Object with KILL_ON_JOB_CLOSE ends it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAM_AGENT = Path(r"C:\Users\samit\Desktop\SAM-Agent")

CHILD = r'''
import asyncio, json, sys, time
sys.path.insert(0, {root!r})
from sam.brain.local_server import OllamaServer

class Config:
    home = None
    data_dir = None
    log_dir = {logs!r}
    def get(self, key, default=None):
        return {{"llm.local.host": "127.0.0.1:{port}", "llm.local.ollama_exe": {exe!r},
                 "llm.local.models_dir": {models!r}}}.get(key, default)

async def main():
    server = OllamaServer(Config())
    ok = await server.ensure(wait_s=30)
    print(json.dumps({{"ok": ok, "pid": getattr(server._proc, "pid", None), "job": server._job is not None,
                      "error": server.last_error}}), flush=True)
    await asyncio.sleep(600)

asyncio.run(main())
'''


def alive(pid: int) -> bool:
    import ctypes

    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    code = ctypes.c_ulong()
    ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(handle)
    return code.value == 259


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=11437)
    args = parser.parse_args()
    from sam.brain.local_server import is_listening

    host = f"127.0.0.1:{args.port}"
    if is_listening(host):
        print(f"port {args.port} is in use; choose another")
        return 2
    exe = sorted((SAM_AGENT / "tools").glob("ollama*/ollama.exe"), reverse=True)
    if not exe:
        print("no ollama.exe under SAM-Agent/tools")
        return 2
    models = tempfile.mkdtemp(prefix="sam2-empty-models-", dir=ROOT / "work")
    logs = str(ROOT / "work" / "ollama-orphan-logs")
    code = CHILD.format(root=str(ROOT), logs=logs, port=args.port, exe=str(exe[0]), models=models)
    child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True,
                             creationflags=subprocess.CREATE_NO_WINDOW)
    line = child.stdout.readline() if child.stdout else ""
    info = json.loads(line or "{}")
    print("child (stands in for SAM):", child.pid, "server:", info)
    if not info.get("ok"):
        child.kill()
        return 1
    pid = int(info["pid"])
    print("server alive:", alive(pid), "listening:", is_listening(host))
    started = time.perf_counter()
    child.kill()                                   # TerminateProcess: no clean-up runs
    child.wait(10)
    while time.perf_counter() - started < 15 and (alive(pid) or is_listening(host)):
        time.sleep(0.25)
    gone = not alive(pid) and not is_listening(host)
    print(f"after SAM was killed: server alive={alive(pid)} listening={is_listening(host)} "
          f"({time.perf_counter() - started:.2f} s)")
    print("RESULT:", "PASS (no orphan)" if gone else "FAIL (orphaned server)")
    return 0 if gone else 1


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
