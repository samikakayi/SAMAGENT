"""run_python: static risk scan, the isolated runner (real subprocesses with
SAM's venv python), limits, and the tool through the registry."""

from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from conftest import FAKE_GROQ
from sam.hands.python_tool import child_env, execute, interpreter, scan

WINDOWS = os.name == "nt"


# -- scan -----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("code, risk", [
    ("import math\nsum(math.sqrt(i) for i in range(10))", "safe"),
    ("import numpy as np\nnp.mean(data['prices'])", "safe"),
    ("import statistics, json, datetime\nstatistics.mean([1, 2])", "safe"),
    ("open('out.txt', 'w').write('hi')", "safe"),
    ("from pathlib import Path\nPath('out.txt').write_text('x')", "safe"),
    ("import numpy as np\nfor i in range(2): np.save(f'arr_{i}.npy', [i])", "safe"),
    ("x = 'a/b'.split('/')\nx", "safe"),
    ("import os\nos.listdir('.')", "confirm"),
    ("import os\nos.environ.get('X')", "confirm"),
    ("import sys\nsys.path", "confirm"),
    ("import shutil\nshutil.rmtree('x')", "confirm"),
    ("import subprocess\nsubprocess.run(['cmd'])", "confirm"),
    ("import socket", "confirm"),
    ("import ctypes", "confirm"),
    ("import requests", "confirm"),
    ("import httpx", "confirm"),
    ("from urllib.request import urlopen", "confirm"),
    ("import some_unknown_package", "confirm"),
    ("eval('1+1')", "confirm"),
    ("exec('x = 1')", "confirm"),
    ("__import__('os')", "confirm"),
    ("().__class__.__base__.__subclasses__()", "confirm"),
    ("getattr(object, name)", "confirm"),
    ("open(r'C:\\Users\\x\\a.txt', 'w').write('hi')", "confirm"),
    ("open('../escape.txt', 'w')", "confirm"),
    ("from pathlib import Path\n(Path.home() / 'a.txt').write_text('x')", "confirm"),
    ("from pathlib import Path\np = Path(input())\np.read_text()", "confirm"),
    ("import pandas as pd\npd.read_csv('https://example.com/a.csv')", "confirm"),
    ("open('.env').read()", "blocked"),
    ("open(r'C:\\SAM\\data\\secrets.json').read()", "blocked"),
    ("import MetaTrader5 as mt5\nmt5.order_send({})", "blocked"),
    ("import win32crypt", "blocked"),
    ("x = 'Set-MpPreference -DisableRealtimeMonitoring $true'", "blocked"),
])
def test_scan_verdicts(code: str, risk: str) -> None:
    verdict = scan(code)
    assert verdict.risk == risk, verdict.reasons


def test_scan_reports_reasons_in_sorani_without_quoting_code() -> None:
    verdict = scan("import os, requests\nos.remove('x')")
    question = verdict.question_ckb()
    assert question.startswith("ئەم کۆدە پایتۆنە ") and question.endswith("جێبەجێی بکەم؟")
    assert "پەیوەندی بە ئینتەرنێتەوە دەکات" in question and "os.remove" not in question
    assert verdict.network


def test_scan_reads_are_marked_and_credential_reads_blocked_by_policy() -> None:
    class Policy:
        def classify_path(self, path: str, action: str) -> tuple[str, str | None]:
            return ("blocked", "keys live there") if "ssh" in path.lower() else ("safe", None)

    safe_read = scan("open(r'C:\\Users\\me\\Documents\\prices.csv').read()", policy=Policy())
    assert safe_read.risk == "safe" and safe_read.reads_files
    blocked = scan("open(r'C:\\Users\\me\\.ssh\\config').read()", policy=Policy())
    assert blocked.risk == "blocked"


def test_syntax_errors_are_reported_not_run() -> None:
    verdict = scan("def (")
    assert verdict.risk == "safe" and verdict.syntax_error.startswith("SyntaxError")


# -- runner -----------------------------------------------------------------------------------------------------------
def test_run_returns_stdout_last_value_files_and_data(tmp_path: Path) -> None:
    code = ("import math\nprint('سڵاو', round(math.pi, 2))\nopen('result.txt', 'w').write('ok')\n"
            "total = sum(data['prices'])\ntotal")
    result = execute(code, run_dir=tmp_path / "run", timeout_s=30, data={"prices": [1.5, 2.5]})
    assert result["exit_code"] == 0 and result["error"] is None
    assert result["stdout"].strip() == "سڵاو 3.14"
    assert result["result"] == "4.0" and result["result_type"] == "float"
    assert result["files"] == [{"name": "result.txt", "bytes": 2}]
    assert Path(result["run_dir"]).name == "run"


def test_errors_show_a_clean_traceback(tmp_path: Path) -> None:
    result = execute("def f():\n    return 1 / 0\nf()", run_dir=tmp_path / "err", timeout_s=30)
    assert result["error"] == "ZeroDivisionError: division by zero"
    assert 'File "<sam>", line 2, in f' in result["stderr"] and ".sam_runner" not in result["stderr"]


def test_isolated_mode_and_a_clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", FAKE_GROQ)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    env = child_env()
    assert "GROQ_API_KEY" not in env and "PYTHONPATH" not in env and env["MPLBACKEND"] == "Agg"
    code = "import os, sys\nprint(sys.flags.isolated, 'GROQ_API_KEY' in os.environ, os.getcwd())"
    result = execute(code, run_dir=tmp_path / "iso", timeout_s=30)
    isolated, has_key, cwd = result["stdout"].split()
    assert isolated == "1" and has_key == "False" and Path(cwd) == tmp_path / "iso"
    assert Path(interpreter()).name.lower() == "python.exe" or not WINDOWS


def test_output_is_capped_head_and_tail(tmp_path: Path) -> None:
    code = "print('START')\nfor i in range(20000): print('line', i)\nprint('END')"
    result = execute(code, run_dir=tmp_path / "big", timeout_s=30, max_output=500)
    assert result["stdout"].startswith("START") and result["stdout"].rstrip().endswith("END")
    assert "bytes cut" in result["stdout"] and len(result["stdout"]) < 3000


@pytest.mark.skipif(not WINDOWS, reason="job objects are Windows-only")
def test_timeout_kills_the_launcher_and_the_interpreter(tmp_path: Path) -> None:
    code = "import os, time\nopen('pid.txt', 'w').write(str(os.getpid()))\nwhile True: time.sleep(0.05)"
    started = time.perf_counter()
    result = execute(code, run_dir=tmp_path / "loop", timeout_s=2)
    assert result["timed_out"] and time.perf_counter() - started < 12
    pid = int((tmp_path / "loop" / "pid.txt").read_text())
    import ctypes

    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
    alive = False
    if handle:
        code_out = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code_out))
        alive = code_out.value == 259                                  # STILL_ACTIVE
        ctypes.windll.kernel32.CloseHandle(handle)
    assert not alive, "the real interpreter (child of the venv launcher) survived the timeout"


def test_cancel_event_stops_the_program(tmp_path: Path) -> None:
    cancel = threading.Event()
    threading.Timer(0.8, cancel.set).start()
    result = execute("while True: pass", run_dir=tmp_path / "cancel", timeout_s=60, cancel=cancel)
    assert result["cancelled"] and not result["timed_out"]


@pytest.mark.skipif(not WINDOWS, reason="job objects are Windows-only")
def test_memory_limit_ends_a_runaway_program(tmp_path: Path) -> None:
    code = "blocks = []\nwhile True: blocks.append(bytearray(50 * 1024 * 1024))"
    result = execute(code, run_dir=tmp_path / "mem", timeout_s=30, memory_mb=300)
    # the job's per-process cap makes the allocation fail inside Python
    assert not result["timed_out"]
    assert (result["error"] or "").startswith("MemoryError") or result["exit_code"] not in (0, None)


# -- the tool -----------------------------------------------------------------------------------------------------------
@pytest.fixture
def app(make_app: Any, tmp_path: Path) -> Any:
    from sam.hands import python_tool

    application = make_app()
    python_tool.register(application)      # sam.knowledge.register does the same in SAM
    application.config.set("hands.projects_dir", str(tmp_path / "SAM Projects"))
    return application


def asked_with(app: Any, answer: bool) -> list[str]:
    asked: list[str] = []

    async def fake(question: str, detail: str = "", **_kw: Any) -> bool:
        asked.append(question)
        return answer

    app.confirm.confirm = fake
    return asked


async def test_safe_code_runs_without_asking_in_its_own_folder(app: Any, tmp_path: Path) -> None:
    asked = asked_with(app, False)
    result = await app.tools.dispatch("run_python", {"code": "x = [3, 4]\nsum(i * i for i in x)"}, source="text")
    assert result["ok"] and not asked
    data = result["data"]
    assert data["result"] == "25" and "Result: 25" in result["summary"]
    run_dir = Path(data["run_dir"])
    assert run_dir.parent == tmp_path / "SAM Projects" / "python"


async def test_data_argument_arrives_as_the_data_variable(app: Any) -> None:
    result = await app.tools.dispatch("run_python", {"code": "max(data['closes']) - min(data['closes'])",
                                                     "data": '{"closes": [2650.5, 2661.0, 2642.25]}'})
    assert result["ok"] and result["data"]["result"] == "18.75"


async def test_risky_code_asks_and_a_no_means_nothing_runs(app: Any, tmp_path: Path) -> None:
    asked = asked_with(app, False)
    marker = tmp_path / "SAM Projects" / "python"
    result = await app.tools.dispatch("run_python", {"code": "import os\nos.listdir('.')"})
    assert not result["ok"] and result["data"]["declined"]
    assert asked and asked[0].startswith("ئەم کۆدە پایتۆنە")
    assert not marker.exists()
    asked_with(app, True)
    approved = await app.tools.dispatch("run_python", {"code": "import os\nlen(os.listdir('.')) >= 0"})
    assert approved["ok"] and approved["data"]["result"] == "True"


async def test_blocked_code_never_runs(app: Any) -> None:
    result = await app.tools.dispatch("run_python", {"code": "import MetaTrader5 as mt5\nmt5.order_send({})"})
    assert not result["ok"] and result["data"]["blocked"]
    assert "trading order" in result["summary"]


async def test_failures_timeouts_and_syntax_errors_are_honest(app: Any) -> None:
    failed = await app.tools.dispatch("run_python", {"code": "int('x')"})
    assert not failed["ok"] and "ValueError" in failed["summary"]
    syntax = await app.tools.dispatch("run_python", {"code": "def ("})
    assert not syntax["ok"] and "not run" in syntax["summary"]
    slow = await app.tools.dispatch("run_python", {"code": "while True: pass", "timeout_s": 1})
    assert not slow["ok"] and slow["data"]["timed_out"] and "1 s time limit" in slow["summary"]
    app.config.set("python.max_timeout_s", 2)
    clamped = await app.tools.dispatch("run_python", {"code": "1", "timeout_s": 500})
    assert clamped["data"]["timeout_s"] == 2


async def test_printed_keys_are_redacted_and_file_reads_are_untrusted(app: Any, tmp_path: Path) -> None:
    printed = await app.tools.dispatch("run_python", {"code": f"print('{FAKE_GROQ}')"})
    assert FAKE_GROQ not in str(printed)
    source = tmp_path / "prices.csv"
    source.write_text("close\n2650\n", encoding="utf-8")
    reads = await app.tools.dispatch("run_python", {"code": f"print(open(r'{source}').read())"})
    assert reads["ok"] and "untrusted" in reads["data"] and "2650" in reads["data"]["untrusted"]["stdout"]


async def test_stop_all_kills_a_running_program(app: Any) -> None:
    task = asyncio.ensure_future(app.tools.dispatch("run_python", {"code": "while True: pass", "timeout_s": 60}))
    for _ in range(100):
        await asyncio.sleep(0.05)
        if app.tools.running():
            break
    await asyncio.sleep(0.5)
    assert app.tools.cancel_all() == 1
    result = await asyncio.wait_for(task, 15)
    assert not result["ok"] and result["data"]["cancelled"]


# -- the OS sandbox for code that runs without asking (review 2026-09-25) --------------------------------------------
@pytest.mark.skipif(not WINDOWS, reason="Windows integrity levels")
def test_sandboxed_code_can_write_only_its_own_folder(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    code = ("open('inside.txt', 'w').write('ok')\n"
            f"try:\n    open(r'{outside}', 'w').write('x')\n    print('WROTE')\n"
            "except PermissionError:\n    print('blocked')\n")
    result = execute(code, run_dir=tmp_path / "run", timeout_s=30, sandbox=True)
    assert result["sandbox"] == "low_integrity", result
    assert result["stdout"].strip() == "blocked" and not outside.exists()
    assert (tmp_path / "run" / "inside.txt").read_text() == "ok"
    assert [f["name"] for f in result["files"]] == ["inside.txt"]          # .sam_* helpers are not listed


@pytest.mark.skipif(not WINDOWS, reason="Windows job objects")
def test_sandboxed_code_cannot_start_programs_or_use_the_clipboard(tmp_path: Path) -> None:
    code = ("import subprocess\n"
            "try:\n    subprocess.run(['cmd', '/c', 'echo hi'], capture_output=True, timeout=5)\n    print('STARTED')\n"
            "except OSError as exc:\n    print('blocked', type(exc).__name__)\n")
    result = execute(code, run_dir=tmp_path / "run", timeout_s=30, sandbox=True)
    assert result["stdout"].startswith("blocked"), result
    normal = execute(code, run_dir=tmp_path / "normal", timeout_s=30, sandbox=False)
    assert normal["stdout"].strip() == "STARTED" and normal["sandbox"] == "none"


@pytest.mark.skipif(not WINDOWS, reason="Windows integrity levels")
async def test_the_tool_sandboxes_unasked_code_and_not_approved_code(app: Any) -> None:
    safe = await app.tools.dispatch("run_python", {"code": "import statistics\nstatistics.mean([1, 2, 3])"})
    assert safe["ok"] and safe["data"]["result"] == "2" and safe["data"]["sandbox"] == "low_integrity"
    asked_with(app, True)
    approved = await app.tools.dispatch("run_python", {"code": "import os\nos.getcwd() != ''"})
    assert approved["ok"] and approved["data"]["sandbox"] == "none"


# Adversarial cases from a pass by someone other than the scan's author (2026-09-25):
# each reached the system, the network or eval through an allowlisted package.
@pytest.mark.parametrize("code", [
    "import numpy as np\nnp.ctypeslib.ctypes.windll.kernel32.Beep(750, 300)",
    "from numpy import ctypeslib",
    "import numpy.ctypeslib as c",
    "import numpy as np\nnp.DataSource().open('x')",
    "import pandas as pd\nu = 'ht' + 'tp://example.com/a.csv'\npd.read_csv(u)",
    "import pandas as pd\npd.read_html(data['url'])",
    "import sympy\nsympy.sympify(\"__import__('os')\")",
    "import numpy as np\nnp.load('a.npy', allow_pickle=True)",
    "import pandas as pd\npd.read_pickle('a.pkl')",
    "import pandas as pd\npd.read_clipboard()",
])
def test_scan_closes_the_allowlist_holes(code: str) -> None:
    assert scan(code).risk == "confirm", scan(code).reasons
