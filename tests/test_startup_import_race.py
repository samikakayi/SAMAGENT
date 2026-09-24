"""Startup must not leave numpy's first import to a race between threads.

Found live: the SAM started at 03:16:58 came up with a dead hands-free listener
("ImportError: cannot import name '__cpu_features__' from partially initialized
module 'numpy._core._multiarray_umath'", then "cannot load module more than
once per process" for the Whisper model) and a /api/trading/status that
answered 500 (MetaTrader5 imported without `shutdown`). A restart fixed it.

numpy was first imported on several threads at once. The wake warm thread
enters through `import numpy` (faster_whisper); MetaTrader5 and onnxruntime
enter through their C `import_array()`, which imports
`numpy._core._multiarray_umath` directly. Module locks are taken child first,
so the two entry points lock numpy's modules in opposite orders; the import
system breaks that deadlock by handing one thread a half-built module, and
numpy's core refuses to initialise a second time in the process. Every later
numpy user in that process is dead until a restart.

Measured in fresh processes: faster_whisper and MetaTrader5 started 0-0.4 s
apart failed 1 run in 40 with exactly the errors above; three MetaTrader5
imports beside faster_whisper and numpy failed 2 in 40, with the AttributeError
the trading route hit; the four entry points below, started together, failed
30 in 30. With numpy imported on the main thread first: 0 failures in 190.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Runs in a fresh interpreter: numpy can be imported for the first time only
# once per process, which is the whole bug.
STARTUP = r"""
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, sys.argv[1])
root = Path(sys.argv[2])
first = {}


class Watch:
    # Whoever looks numpy up first is the thread that loads it, whichever
    # submodule it asked for: a parent package is always found before its child.
    def find_spec(self, name, path=None, target=None):
        if name == "numpy" and not first:
            current = threading.current_thread()
            first.update(thread=current.name, main=current is threading.main_thread())
        return None


sys.meta_path.insert(0, Watch())

import sam_backend.config as config

config.is_elevated_windows_process = lambda: False  # as tests/conftest.py does
from sam_backend.app import create_app
from sam_backend.config import Settings

create_app(Settings(project_root=root, workspace_root=root / "workspace", data_dir=root / "data",
                    default_provider="ollama", default_model="fake"))
spec = getattr(sys.modules.get("numpy"), "__spec__", None)
after_create_app = {"loaded": spec is not None,
                    "initializing": bool(getattr(spec, "_initializing", False)), "first": first}

# What the 03:16:58 start amounted to, made certain rather than likely: the
# ways SAM's dependencies reach numpy, taken by several threads at once.
entry_points = ["numpy._core._multiarray_umath", "numpy", "numpy._core.multiarray", "numpy.linalg"]
barrier = threading.Barrier(len(entry_points))
errors = []


def first_import(name):
    barrier.wait()
    try:
        __import__(name)
    except BaseException as exc:
        errors.append(f"{name}: {type(exc).__name__}: {str(exc).strip().splitlines()[-1:]}")


threads = [threading.Thread(target=first_import, args=(name,)) for name in entry_points]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join(60)
try:
    import numpy

    numpy.zeros(3).sum()
except BaseException as exc:
    errors.append(f"afterwards: {type(exc).__name__}: {exc}")
print(json.dumps({"after_create_app": after_create_app, "race_errors": errors}))
"""


@pytest.fixture(scope="module")
def started(tmp_path_factory):
    root = tmp_path_factory.mktemp("startup-import-race")
    finished = subprocess.run(
        [sys.executable, "-c", STARTUP, str(PROJECT_ROOT), str(root)],
        capture_output=True, text=True, encoding="utf-8", timeout=180,
    )
    assert finished.returncode == 0, finished.stderr[-2000:]
    return json.loads(finished.stdout.strip().splitlines()[-1])


def test_create_app_has_imported_numpy_on_the_main_thread(started):
    """Before anything can start a thread: the lifespan, the wake listener, a request."""
    state = started["after_create_app"]
    assert state["loaded"], "create_app returned with numpy not yet imported"
    assert not state["initializing"]
    assert state["first"]["main"], f"numpy was first imported on {state['first']['thread']}"


def test_threads_reaching_numpy_by_different_routes_after_startup_all_succeed(started):
    assert started["race_errors"] == []
