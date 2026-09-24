"""acceptance/run_all.py: discovery, the result protocol, redaction, the table.
Runs tiny fake scripts in a temp folder (never the real live checks)."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from tests.conftest import FAKE_GROQ
from tests.launcher_helpers import ROOT

ACCEPTANCE = ROOT / "acceptance"


@pytest.fixture(scope="module")
def run_all():
    spec = importlib.util.spec_from_file_location("sam_run_all_under_test", ACCEPTANCE / "run_all.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module      # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


@pytest.fixture
def scripts(tmp_path) -> Path:
    folder = tmp_path / "acceptance"
    folder.mkdir()
    shutil.copy(ACCEPTANCE / "_common.py", folder / "_common.py")
    (folder / "demo_pass.py").write_text(
        "import os, sys\n"
        "from _common import Acceptance\n"
        "acc = Acceptance('demo_pass')\n"
        "with acc.check('home is passed') as c:\n"
        "    c.data['home'] = os.environ['SAM_HOME']\n"
        "with acc.check('sorani output') as c:\n"
        "    c.detail = 'زێڕ ئامادەیە'\n"
        "sys.exit(acc.finish())\n", encoding="utf-8")
    (folder / "demo_fail.py").write_text(
        "from _common import Acceptance\n"
        "import sys\n"
        "acc = Acceptance('demo_fail')\n"
        "with acc.check('leaky failure'):\n"
        f"    raise RuntimeError('provider refused key {FAKE_GROQ}')\n"
        "sys.exit(acc.finish())\n", encoding="utf-8")
    (folder / "demo_skip.py").write_text("import sys\nprint('TradingView is closed')\nsys.exit(77)\n", encoding="utf-8")
    (folder / "demo_liar.py").write_text(   # exit 0 but its own JSON says it failed
        "import json\nprint('working...')\nprint(json.dumps({'ok': False, 'summary': 'chart did not change'}, indent=1))\n",
        encoding="utf-8")
    (folder / "demo_plain.py").write_text("print('all good')\n", encoding="utf-8")
    (folder / "demo_slow.py").write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    (folder / "helpers.py").write_text("raise SystemExit('not a script')\n", encoding="utf-8")
    return folder


def test_discovery_skips_the_runner_helpers_and_plain_modules(run_all, scripts):
    names = [p.name for p in run_all.discover(scripts)]
    assert names == ["demo_fail.py", "demo_liar.py", "demo_pass.py", "demo_plain.py", "demo_skip.py", "demo_slow.py"]
    assert [p.name for p in run_all.discover(scripts, ["PASS", "skip"])] == ["demo_pass.py", "demo_skip.py"]
    assert "run_all.py" not in [p.name for p in run_all.discover(ACCEPTANCE)]


def test_result_protocol(run_all, scripts, tmp_path):
    home = tmp_path / "home"
    results = {name: run_all.run_script(scripts / f"{name}.py", home=home, timeout_s=60)
               for name in ("demo_pass", "demo_fail", "demo_skip", "demo_liar", "demo_plain")}

    assert results["demo_pass"].status == "pass"
    assert results["demo_pass"].checks[0]["data"]["home"] == str(home)
    assert results["demo_pass"].summary == "2/2 checks passed"
    assert results["demo_fail"].status == "fail" and "leaky failure" in results["demo_fail"].summary
    assert results["demo_skip"].status == "skip" and results["demo_skip"].exit_code == 77
    assert results["demo_liar"].status == "fail" and results["demo_liar"].summary == "chart did not change"
    assert results["demo_plain"].status == "pass"
    blob = json.dumps([r.__dict__ for r in results.values()], ensure_ascii=False)
    assert FAKE_GROQ not in blob and "[REDACTED]" in blob


def test_timeout(run_all, scripts, tmp_path):
    result = run_all.run_script(scripts / "demo_slow.py", home=tmp_path, timeout_s=1)
    assert result.status == "timeout"


def test_main_prints_a_table_writes_a_redacted_report_and_fails_on_failures(run_all, scripts, tmp_path, capsys):
    report = tmp_path / "report.json"
    code = run_all.main(["--dir", str(scripts), "--home", str(tmp_path / "home"), "--json", str(report),
                         "--only", "pass", "fail", "skip"])

    out = capsys.readouterr().out
    assert code == 1
    assert "demo_pass" in out and "PASS" in out and "FAIL" in out and "SKIP" in out
    assert "1 pass, 1 fail, 1 skip" in out
    saved = report.read_text(encoding="utf-8")
    assert FAKE_GROQ not in saved and json.loads(saved)["results"][0]["name"] == "demo_fail"


def test_main_passes_when_nothing_failed(run_all, scripts, tmp_path, capsys):
    code = run_all.main(["--dir", str(scripts), "--json", str(tmp_path / "r.json"), "--only", "demo_pass", "demo_skip"])
    assert code == 0


def test_list(run_all, scripts, capsys):
    assert run_all.main(["--dir", str(scripts), "--list"]) == 0
    assert "demo_pass.py" in capsys.readouterr().out


def test_nested_pretty_json_is_not_mistaken_for_the_result(run_all):
    text = 'log\n{\n "ok": true,\n "checks": [\n  {"name": "a", "ok": false}\n ]\n}'
    assert run_all._last_json_line(text) == {"ok": True, "checks": [{"name": "a", "ok": False}]}
    assert run_all._last_json_line('{"ok": false, "summary": "x"}\ntrailing words') == {"ok": False, "summary": "x"}
    assert run_all._last_json_line("no json here") is None


def test_every_real_acceptance_script_of_this_package_compiles():
    for script in sorted(ACCEPTANCE.glob("launcher_*.py")) + [ACCEPTANCE / "run_all.py", ACCEPTANCE / "_common.py"]:
        compile(script.read_text(encoding="utf-8"), str(script), "exec")
