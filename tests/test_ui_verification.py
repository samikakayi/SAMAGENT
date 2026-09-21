"""Browser verification of UI work, end to end through the real HTTP surface.

A run that touched UI files must open the page in a real (headless, isolated)
browser before it may call itself done. A console error fails validation and
feeds the same re-plan path a failing test suite does; a clean page passes;
and when no model can review the screenshot the run says so instead of
inventing a pass.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from sam_backend.app import create_app
from sam_backend.config import Settings
from sam_backend.models import AssistantTurn, ModelError, ToolCall


def _browser_available() -> bool:
    try:
        with sync_playwright() as runtime:
            for options in ({"channel": "msedge"}, {}):
                try:
                    runtime.chromium.launch(headless=True, **options).close()
                    return True
                except PlaywrightError:
                    continue
    except Exception:  # noqa: BLE001
        pass
    return False


pytestmark = pytest.mark.skipif(not _browser_available(), reason="no headless browser is available on this machine")

BROKEN_PAGE = "<html><body><h1>Dashboard</h1><script>throw new Error('boom')</script></body></html>"
CLEAN_PAGE = "<html><body><h1>Dashboard</h1></body></html>"


class UiAdapter:
    """Plans a UI edit, writes the page, and answers the visual review."""

    def __init__(self, *, pages: list[str], review: str | Exception | None):
        self.pages = list(pages)
        self.review = review
        self.review_calls = 0
        self.plans = 0

    async def list_models(self):
        return [{"id": "fake", "name": "fake", "provider": "ollama"}]

    async def complete(self, messages, tools, model):
        system = str(messages[0].get("content", "")) if messages else ""
        if "visual review stage" in system:
            self.review_calls += 1
            if isinstance(self.review, Exception):
                raise self.review
            return AssistantTurn(self.review or "")
        if "planning stage" in system or "re-planning stage" in system:
            self.plans += 1
            return AssistantTurn('```json\n{"steps":[{"text":"Write index.html","kind":"edit"}]}\n```')
        if self.pages:
            content = self.pages.pop(0)
            return AssistantTurn("Writing the page.", [ToolCall(f"call_{len(self.pages)}", "write_file", {"path": "index.html", "content": content})])
        return AssistantTurn("Nothing further.")


class Registry:
    def __init__(self, adapter):
        self.adapter = adapter

    def get(self, provider):
        return self.adapter


def make_client(tmp_path: Path, adapter) -> TestClient:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    settings = Settings(
        project_root=tmp_path, workspace_root=workspace, data_dir=tmp_path / "data",
        default_provider="ollama", default_model="fake", permission_mode="trusted",
    )
    return TestClient(create_app(settings, Registry(adapter)))


def run_to_end(client: TestClient, goal: str, *, timeout: float = 120.0) -> dict:
    """Drive a run to its end, approving overwrites the way a watching user would."""
    task_id = client.post("/api/tasks", json={"goal": goal}).json()["task_id"]
    deadline = time.time() + timeout
    task: dict = {}
    approved: set[str] = set()
    while time.time() < deadline:
        task = client.get(f"/api/tasks/{task_id}").json()["task"]
        if task["terminal"]:
            return task
        if task["state"] == "WAITING_FOR_APPROVAL":
            approval_id = next(
                event["detail"]["approval_id"] for event in reversed(task["events"]) if event["kind"] == "approval"
            )
            if approval_id not in approved:
                approved.add(approval_id)
                client.post(f"/api/approvals/{approval_id}/decision", json={"decision": "approved"})
        time.sleep(0.3)
    return task


def checks_of(task: dict) -> list[dict]:
    return [check for result in task["test_results"] for check in (result.get("checks") or [])]


def test_a_console_error_fails_validation_and_the_replan_fixes_it(tmp_path: Path):
    """The first page throws; the browser catches it; the run re-plans and writes a clean page."""
    adapter = UiAdapter(pages=[BROKEN_PAGE, CLEAN_PAGE], review='{"verdict":"pass","findings":[]}')
    with make_client(tmp_path, adapter) as client:
        task = run_to_end(client, "Build the dashboard page")

    assert task["state"] == "COMPLETED", task.get("summary")
    assert task["replans"] == 1, "the console error must have triggered exactly one re-plan"
    smokes = [check for check in checks_of(task) if check["kind"] == "ui_smoke"]
    assert [check["outcome"] for check in smokes] == ["FAILED", "PASSED"]
    assert any("boom" in failure for failure in smokes[0]["failures"])
    # The failure text the re-planner saw named the real problem.
    assert any("boom" in event["message"] or "boom" in json.dumps(event["detail"]) for event in task["events"])
    assert (tmp_path / "workspace" / "index.html").read_text(encoding="utf-8") == CLEAN_PAGE
    assert task["completion_status"] == "completed_verified"


def test_a_clean_page_passes_browser_and_visual_review(tmp_path: Path):
    adapter = UiAdapter(pages=[CLEAN_PAGE], review='{"verdict":"pass","findings":[]}')
    with make_client(tmp_path, adapter) as client:
        task = run_to_end(client, "Build the dashboard page")

    assert task["state"] == "COMPLETED" and task["completion_status"] == "completed_verified"
    assert task["replans"] == 0
    outcomes = {check["kind"]: check["outcome"] for check in checks_of(task)}
    assert outcomes["ui_smoke"] == "PASSED"
    assert outcomes["ui_review"] == "PASSED"
    assert adapter.review_calls == 1
    screenshots = list((tmp_path / "data" / "ui-review").glob("*.png"))
    assert screenshots, "a screenshot must have been captured for review"
    assert any(event["message"].startswith("Visual review passed") for event in task["events"])


def test_visual_findings_fail_validation_like_a_failing_test(tmp_path: Path):
    """The browser is clean but the reviewer sees a problem: that is still a failure."""
    reviews = iter([
        '{"verdict":"fail","findings":["The heading overlaps the navigation bar"]}',
        '{"verdict":"pass","findings":[]}',
    ])

    class ReviewingAdapter(UiAdapter):
        async def complete(self, messages, tools, model):
            if "visual review stage" in str(messages[0].get("content", "")):
                self.review_calls += 1
                return AssistantTurn(next(reviews))
            return await super().complete(messages, tools, model)

    adapter = ReviewingAdapter(pages=[CLEAN_PAGE, CLEAN_PAGE], review=None)
    with make_client(tmp_path, adapter) as client:
        task = run_to_end(client, "Build the dashboard page")

    assert task["state"] == "COMPLETED"
    assert task["replans"] == 1
    reviews_seen = [check for check in checks_of(task) if check["kind"] == "ui_review"]
    assert [check["outcome"] for check in reviews_seen] == ["FAILED", "PASSED"]
    assert "overlaps" in reviews_seen[0]["failures"][0]
    assert any("overlaps" in json.dumps(event["detail"]) for event in task["events"] if event["kind"] == "error")


@pytest.mark.parametrize(
    "review, reason_fragment",
    [
        (ModelError("no vision provider"), "no vision-capable model"),
        ('{"verdict":"no_image"}', "could not see the screenshot"),
        ("Sure! The page looks great to me.", "did not return a verdict"),
    ],
)
def test_an_unavailable_or_non_committal_reviewer_is_recorded_as_skipped(tmp_path: Path, review, reason_fragment: str):
    """A review that did not really happen must never read as a pass."""
    adapter = UiAdapter(pages=[CLEAN_PAGE], review=review)
    with make_client(tmp_path, adapter) as client:
        task = run_to_end(client, "Build the dashboard page")

    outcomes = {check["kind"]: check for check in checks_of(task)}
    assert outcomes["ui_smoke"]["outcome"] == "PASSED"
    assert outcomes["ui_review"]["outcome"] == "SKIPPED"
    assert reason_fragment in outcomes["ui_review"]["reason"]
    skip_events = [event for event in task["events"] if event["message"].startswith("Visual review skipped")]
    assert skip_events and reason_fragment in skip_events[0]["message"]
    # The browser smoke still counts, so the run is verified -- but honestly.
    assert task["state"] == "COMPLETED"
    assert "ui_review: SKIPPED" in task["summary"]


def test_non_ui_work_never_opens_a_browser(tmp_path: Path):
    class TextAdapter(UiAdapter):
        async def complete(self, messages, tools, model):
            system = str(messages[0].get("content", ""))
            if "planning stage" in system:
                return AssistantTurn('```json\n{"steps":[{"text":"Write notes","kind":"edit"}]}\n```')
            if self.pages:
                self.pages.pop(0)
                return AssistantTurn("ok", [ToolCall("c1", "write_file", {"path": "notes.txt", "content": "hello"})])
            return AssistantTurn("done")

    adapter = TextAdapter(pages=["x"], review=None)
    with make_client(tmp_path, adapter) as client:
        task = run_to_end(client, "Write notes")

    assert task["terminal"]
    assert not [check for check in checks_of(task) if check["kind"].startswith("ui_")]
    assert adapter.review_calls == 0


def test_an_unservable_ui_project_skips_the_browser_check_honestly(tmp_path: Path):
    """UI file touched, but nothing to serve: say so rather than fail or fake."""
    class CssAdapter(UiAdapter):
        async def complete(self, messages, tools, model):
            system = str(messages[0].get("content", ""))
            if "planning stage" in system:
                return AssistantTurn('```json\n{"steps":[{"text":"Write styles","kind":"edit"}]}\n```')
            if self.pages:
                self.pages.pop(0)
                return AssistantTurn("ok", [ToolCall("c1", "write_file", {"path": "theme.css", "content": "body{}"})])
            return AssistantTurn("done")

    adapter = CssAdapter(pages=["x"], review=None)
    with make_client(tmp_path, adapter) as client:
        task = run_to_end(client, "Add a theme")

    smoke = next(check for check in checks_of(task) if check["kind"] == "ui_smoke")
    assert smoke["outcome"] == "SKIPPED" and "index.html" in smoke["reason"]
    assert adapter.review_calls == 0
    assert task["state"] == "COMPLETED"


# -- dev server bring-up -----------------------------------------------------

def test_a_declared_dev_command_is_started_waited_for_and_stopped(tmp_path: Path):
    import json as json_module
    import shutil
    import urllib.request

    from sam_backend.project_map import ProjectScanner
    from sam_backend.verification import UiSmokeRunner, VerificationEngine

    if shutil.which("npm") is None:
        pytest.skip("npm is not installed")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "index.html").write_text(CLEAN_PAGE, encoding="utf-8")
    (workspace / "package.json").write_text(json_module.dumps({
        "name": "demo", "scripts": {"dev": "python -m http.server 3457 --bind 127.0.0.1"},
    }), encoding="utf-8")
    project_map = ProjectScanner().scan(workspace)
    assert project_map.commands["dev"] == "npm run dev"

    target = UiSmokeRunner(VerificationEngine()).bring_up(project_map, workspace)
    try:
        assert target.url == "http://127.0.0.1:3457/", target.reason
        assert target.started is True
        assert urllib.request.urlopen(target.url, timeout=3).status == 200
    finally:
        target.stop()
    time.sleep(1.0)
    with pytest.raises(Exception):
        urllib.request.urlopen("http://127.0.0.1:3457/", timeout=2)


def test_an_already_running_dev_server_is_reused_and_left_alone(tmp_path: Path):
    import json as json_module
    import subprocess
    import sys
    import urllib.request

    from sam_backend.project_map import ProjectScanner
    from sam_backend.verification import UiSmokeRunner, VerificationEngine

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "index.html").write_text(CLEAN_PAGE, encoding="utf-8")
    (workspace / "package.json").write_text(json_module.dumps({"scripts": {"dev": "vite --port 3458"}}), encoding="utf-8")
    # The user's own server, started outside SAM.
    users = subprocess.Popen(
        [sys.executable, "-m", "http.server", "3458", "--bind", "127.0.0.1", "--directory", str(workspace)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(40):
            try:
                urllib.request.urlopen("http://127.0.0.1:3458/", timeout=1)
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.25)
        target = UiSmokeRunner(VerificationEngine()).bring_up(ProjectScanner().scan(workspace), workspace)
        assert target.url == "http://127.0.0.1:3458/" and target.started is False
        target.stop()
        assert users.poll() is None, "a server SAM did not start must not be killed"
        assert urllib.request.urlopen(target.url, timeout=3).status == 200
    finally:
        users.kill()


def test_a_dev_command_that_never_answers_is_reported_not_hung(tmp_path: Path):
    import json as json_module

    from sam_backend.project_map import ProjectScanner
    from sam_backend.verification import UiSmokeRunner, VerificationEngine

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "package.json").write_text(json_module.dumps({"scripts": {"dev": "python -c \"import time; time.sleep(30)\""}}), encoding="utf-8")
    runner = UiSmokeRunner(VerificationEngine(), ready_timeout=3.0)

    started = time.monotonic()
    target = runner.bring_up(ProjectScanner().scan(workspace), workspace)

    assert target.url is None and "did not answer" in target.reason
    assert time.monotonic() - started < 15
    assert target.process is None, "the failed server was cleaned up"
