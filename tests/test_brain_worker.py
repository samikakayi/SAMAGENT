from __future__ import annotations

import asyncio
from pathlib import Path

from brain_helpers import CALLS, Reply, brain_app, collect, tool_names, tool_results

from sam.brain.llm import LLMError
from sam.brain.project_builder import FileBlockParser, missing_references, safe_relpath, strip_fences
from sam.brain.tools import ToolContext, ok, tool
from sam.brain.worker import SORANI_CANCELLED, SORANI_FAILED, SORANI_NO_MODEL
from sam.events import SpeakRequest, Transcript, WorkerProgress


def finish(ok_: bool, summary: str, evidence: str = "") -> Reply:
    return Reply(calls=[("finish_task", {"ok": ok_, "summary_ckb": summary, "evidence": evidence})])


async def test_worker_plans_acts_and_reports_in_sorani(make_app):
    app, backend = brain_app(make_app, [
        Reply(text="Plan: open Chrome, then check it.", calls=[("open_app", {"name": "Chrome"})]),
        Reply(calls=[("screen_look", {"window": "Chrome"})]),
        finish(True, "کرۆم کرایەوە و ئامادەیە.", "open_app ok, screen_look shows Chrome")])
    progress = collect(app.bus, WorkerProgress)
    speak = collect(app.bus, SpeakRequest)
    transcripts = collect(app.bus, Transcript)
    result = await app.worker.run("کرۆم بکەرەوە و دڵنیا بە کە کراوەتەوە")
    assert result["ok"] is True and result["summary_ckb"] == "کرۆم کرایەوە و ئامادەیە." and result["steps"] == 3
    assert [n for n, _ in CALLS] == ["open_app", "screen_look"]
    # Same tools as the conversation, minus delegate_task/stop_all, plus finish_task.
    offered = set(tool_names(backend.requests[0]))
    assert "finish_task" in offered and "open_app" in offered
    assert "delegate_task" not in offered and "stop_all" not in offered
    assert backend.models[0] == "openai/gpt-oss-120b"         # 'strong' ladder (omniroute/gemini unconfigured)
    assert "finish_task" in backend.requests[0].messages[0]["content"]
    assert progress[0].text_ckb == "دەستم پێکرد" and progress[-1].done and progress[-1].ok is True
    assert speak[-1].text_ckb == "کرۆم کرایەوە و ئامادەیە." and speak[-1].source == "worker"
    assert transcripts[-1].source == "worker"
    row = app.db.query_one("SELECT * FROM activity WHERE kind='worker'")
    assert row["ok"] == 1


async def test_claiming_success_after_a_failed_action_is_sent_back_once(make_app):
    app, backend = brain_app(make_app, [
        Reply(calls=[("open_app", {"name": "missing app"})]),
        finish(True, "کرایەوە."),                                   # dishonest: the action failed
        finish(False, "ببورە، ئەو بەرنامەیە دانەمەزراوە.")])
    result = await app.worker.run("بەرنامەکە بکەرەوە")
    assert result["ok"] is False and "دانەمەزراوە" in result["summary_ckb"]
    rejection = tool_results(backend.requests[2])[-1]
    assert rejection["ok"] is False and "failed" in rejection["summary"]


async def test_english_summary_is_rewritten_into_sorani(make_app):
    app, backend = brain_app(make_app, [finish(True, "Chrome is open."), "کرۆم کرایەوە."])
    result = await app.worker.run("open chrome")
    assert result["summary_ckb"] == "کرۆم کرایەوە."
    assert backend.models[-1] == "openai/gpt-oss-120b"          # 'sorani' ladder falls to groq 120b here


async def test_text_only_answer_is_nudged_then_accepted(make_app):
    app, backend = brain_app(make_app, ["I think it's done.", "کارەکە کرا."])
    result = await app.worker.run("something")
    assert result["summary_ckb"] == "کارەکە کرا." and len(backend.requests) == 2


async def test_worker_without_a_model_fails_honestly(make_app):
    app, _ = brain_app(make_app, default=LLMError("server", "down", provider="groq", model="x"))
    result = await app.worker.run("هەر شتێک")
    assert result["ok"] is False and result["summary_ckb"] == SORANI_NO_MODEL


async def test_out_of_steps(make_app):
    app, _ = brain_app(make_app, default=Reply(calls=[("screen_look", {})]))
    result = await app.worker.run("بێکۆتایی", max_steps=3)
    assert result["ok"] is False and result["steps"] == 3 and len(CALLS) == 3


async def test_stop_all_cancels_a_running_task_and_its_tool(make_app):
    app, _ = brain_app(make_app, default=Reply(calls=[("slow_job", {})]))
    progress = collect(app.bus, WorkerProgress)
    result = await app.tools.dispatch("delegate_task", {"goal": "کارێکی درێژ بکە"}, source="live")
    assert result["ok"] and result["data"]["task_id"]
    for _ in range(200):
        if CALLS:
            break
        await asyncio.sleep(0.01)
    assert CALLS == [("slow_job", {})]
    busy = await app.tools.dispatch("delegate_task", {"goal": "کارێکی تر"}, source="live")
    assert busy["ok"] is False and busy["data"]["running_task_id"] == result["data"]["task_id"]
    stopped = await app.stop_all()
    assert stopped["worker"] == 1
    for _ in range(200):
        if app.worker.busy() is None:
            break
        await asyncio.sleep(0.01)
    status = app.worker.status()[0]
    assert status["state"] == "cancelled" and status["summary_ckb"] == SORANI_CANCELLED
    assert progress[-1].done and progress[-1].ok is False


async def test_cancel_between_steps_when_run_directly(make_app):
    app, _ = brain_app(make_app)
    gate = asyncio.Event()

    @tool("wait_here", description="Waits for the test.")
    async def wait_here(ctx: ToolContext) -> dict:
        gate.set()
        await asyncio.sleep(0.05)
        return ok("waited")

    app.tools.add(wait_here, owner="test")
    app.llm.backends["groq"].default = Reply(calls=[("wait_here", {})])
    task = asyncio.create_task(app.worker.run("wait", task_id="t1"))
    await gate.wait()
    assert app.worker.cancel("t1") == 1
    result = await task
    assert result["cancelled"] is True


async def test_delegate_task_is_non_blocking_in_live(make_app):
    app, _ = brain_app(make_app)
    spec = app.tools.get("delegate_task")
    assert spec.blocking is False and spec.examples_ckb
    assert "goal" in spec.params["required"]


async def test_failed_summary_falls_back_to_a_sorani_template(make_app):
    app, _ = brain_app(make_app, [finish(False, ""), LLMError("server", "x", provider="groq", model="m")] +
                       [LLMError("server", "x", provider="groq", model="m")] * 3)
    result = await app.worker.run("x")
    assert result["summary_ckb"] == SORANI_FAILED


# --- project builder ------------------------------------------------------------------------

SITE = (
    "<<<FILE: index.html>>>\n<!doctype html>\n<html lang=\"ckb\" dir=\"rtl\"><head>"
    "<link rel=\"stylesheet\" href=\"style.css\"></head><body><h1>دوکانی سام</h1>"
    "<script src=\"js/app.js\"></script></body></html>\n<<<END FILE>>>\n"
    "<<<FILE: style.css>>>\n```css\nbody { font-family: Vazirmatn; }\n```\n<<<END FILE>>>\n"
    "<<<FILE: ../evil.txt>>>\nnope\n<<<END FILE>>>\n")


def test_file_block_parser_streams_and_rejects_unsafe_paths():
    parser = FileBlockParser()
    done = []
    for i in range(0, len(SITE), 7):                          # arbitrary stream cuts
        done += parser.feed(SITE[i:i + 7])
    assert [p for p, _ in done] == ["index.html", "style.css"]
    assert done[1][1] == "body { font-family: Vazirmatn; }\n"  # fence stripped
    assert parser.rejected == ["../evil.txt"]
    assert missing_references(dict(done)) == ["js/app.js"]


def test_safe_relpath():
    assert safe_relpath("css\\main.css") == "css/main.css"
    assert safe_relpath("./index.html") == "index.html"
    for bad in ("/etc/passwd", "C:/x.txt", "a/../../b", "", "..", "a/b/c/d/e/f.txt", "bad|name.txt"):
        assert safe_relpath(bad) is None
    assert strip_fences("```\nx\n```") == "x\n"


async def test_build_project_writes_files_and_repairs_missing_ones(make_app, tmp_path):
    app, backend = brain_app(make_app, [SITE, "<<<FILE: js/app.js>>>\nconsole.log('سڵاو');\n<<<END FILE>>>"])
    projects = tmp_path / "SAM Projects"
    app.config.set("hands.projects_dir", str(projects))
    steps: list[tuple[int, int, str]] = []
    result = await app.worker.build_project("ماڵپەڕێکی سادە بۆ دوکانەکەم", project_dir=projects / "shop",
                                            progress=lambda s, m, t: steps.append((s, m, t)))
    assert result["ok"] is True, result
    assert result["files"] == ["index.html", "js/app.js", "style.css"] and result["entry"] == "index.html"
    assert (projects / "shop" / "js" / "app.js").read_text(encoding="utf-8") == "console.log('سڵاو');\n"
    assert not (projects / "evil.txt").exists() and not (tmp_path / "evil.txt").exists()
    assert result["summary_ckb"] == "پرۆژەکە بە ٣ فایل دروست کرا."
    assert any("index.html" in text for _, _, text in steps)
    first = backend.requests[0]
    assert "<<<FILE:" in first.messages[0]["content"] and first.max_tokens >= 16000


async def test_build_project_goes_through_the_files_tool_when_present(make_app, tmp_path):
    written: list[str] = []

    @tool("files", description="Files.", params={"type": "object", "properties": {
        "action": {"type": "string"}, "path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["action", "path"]})
    async def files(ctx: ToolContext, action: str, path: str, content: str = "") -> dict:
        written.append(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content, encoding="utf-8")
        return ok("written")

    app, _ = brain_app(make_app, ["<<<FILE: main.py>>>\nprint('hi')\n"], tools=(files,))
    app.config.set("hands.projects_dir", str(tmp_path))
    result = await app.worker.build_project("a tiny script", project_dir=tmp_path / "tool", kind="python")
    assert result["ok"] and result["files"] == ["main.py"]            # missing END at the very end is fine
    assert written == [str(tmp_path / "tool" / "main.py")]


async def test_build_project_refuses_folders_outside_the_projects_dir_without_files_tool(make_app, tmp_path):
    app, _ = brain_app(make_app, [SITE])
    app.config.set("hands.projects_dir", str(tmp_path / "SAM Projects"))
    result = await app.worker.build_project("site", project_dir=tmp_path / "elsewhere")
    assert result["ok"] is False and not (tmp_path / "elsewhere").exists()


async def test_build_project_regenerates_a_file_cut_off_by_the_length_limit(make_app, tmp_path):
    cut = Reply(text="<<<FILE: index.html>>>\n<html><body>نیوە", finish_reason="length")
    app, _ = brain_app(make_app, [cut, "<<<FILE: index.html>>>\n<html><body>تەواو</body></html>\n<<<END FILE>>>"])
    app.config.set("hands.projects_dir", str(tmp_path))
    result = await app.worker.build_project("site", project_dir=tmp_path / "cut")
    assert result["ok"] and "تەواو" in (tmp_path / "cut" / "index.html").read_text(encoding="utf-8")
