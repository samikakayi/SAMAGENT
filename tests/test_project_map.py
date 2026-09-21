"""Project intelligence engine: scanning, caching, and secret hygiene."""

from __future__ import annotations

import json
import time
from pathlib import Path

from sam_backend.project_map import ProjectScanner


def _node_project(root: Path) -> Path:
    (root / "src").mkdir(parents=True)
    (root / "src" / "server.js").write_text(
        "const app = express();\napp.get('/api/users', handler);\napp.post('/api/login', handler);\n",
        encoding="utf-8",
    )
    (root / "package.json").write_text(
        json.dumps({
            "name": "demo",
            "main": "src/server.js",
            "scripts": {"test": "jest", "build": "webpack", "dev": "node src/server.js"},
            "dependencies": {"express": "^4.0.0"},
            "devDependencies": {"jest": "^29.0.0"},
        }),
        encoding="utf-8",
    )
    (root / "package-lock.json").write_text("{}", encoding="utf-8")
    tests = root / "__tests__"
    tests.mkdir()
    (tests / "server.test.js").write_text("test('x', () => {});", encoding="utf-8")
    return root


def test_a_node_project_is_understood_end_to_end(tmp_path: Path):
    _node_project(tmp_path)
    project_map = ProjectScanner().scan(tmp_path)

    assert project_map.languages.get("JavaScript") >= 2
    assert "npm" in project_map.package_managers
    assert "express" in project_map.dependencies["node"]
    assert project_map.commands["test"] == "npm run test"
    assert project_map.commands["build"] == "npm run build"
    assert "src/server.js" in project_map.entry_points
    assert "__tests__" in project_map.test_suites
    paths = {(route["method"], route["path"]) for route in project_map.api_routes}
    assert ("get", "/api/users") in paths
    assert ("post", "/api/login") in paths


def test_a_python_project_infers_pytest_without_a_package_json(tmp_path: Path):
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("fastapi>=0.116,<1\nhttpx==0.28.1\n# comment\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("@app.get('/health')\ndef health(): ...\n", encoding="utf-8")

    project_map = ProjectScanner().scan(tmp_path)

    assert project_map.commands["test"] == "python -m pytest"
    assert project_map.dependencies["python"] == ["fastapi", "httpx"]
    assert "app.py" in project_map.entry_points
    assert {"method": "get", "path": "/health", "file": "app.py"} in project_map.api_routes


def test_env_files_contribute_names_but_never_values(tmp_path: Path):
    """The map is summarised into model prompts, so a value must never enter it."""
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=sk-super-secret-value-0000\nSAM_PORT=8765\n", encoding="utf-8",
    )
    project_map = ProjectScanner().scan(tmp_path)

    assert project_map.env_var_names == ["OPENAI_API_KEY", "SAM_PORT"]
    serialized = json.dumps(project_map.as_dict()) + project_map.summary_text()
    assert "sk-super-secret-value-0000" not in serialized
    assert "8765" not in serialized.split("Env var names")[0]


def test_ignored_directories_are_never_walked(tmp_path: Path):
    (tmp_path / "keep.py").write_text("x = 1\n", encoding="utf-8")
    for noisy in ("node_modules", ".venv", "__pycache__", "dist"):
        directory = tmp_path / noisy
        directory.mkdir()
        (directory / "junk.py").write_text("y = 2\n", encoding="utf-8")

    project_map = ProjectScanner().scan(tmp_path)

    assert project_map.languages.get("Python") == 1
    assert project_map.file_count == 1


def test_an_unchanged_tree_reuses_the_cached_map(tmp_path: Path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    scanner = ProjectScanner(fresh_seconds=0.0)

    first = scanner.scan(tmp_path)
    second = scanner.scan(tmp_path)

    assert second is first


def test_a_new_file_invalidates_the_cache(tmp_path: Path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    scanner = ProjectScanner(fresh_seconds=0.0)
    first = scanner.scan(tmp_path)

    (tmp_path / "b.py").write_text("y = 2\n", encoding="utf-8")
    second = scanner.scan(tmp_path)

    assert second is not first
    assert second.file_count == 2
    assert second.fingerprint != first.fingerprint


def test_an_edited_file_invalidates_the_cache(tmp_path: Path):
    target = tmp_path / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    scanner = ProjectScanner(fresh_seconds=0.0)
    first = scanner.scan(tmp_path)

    time.sleep(0.01)
    target.write_text("x = 1\ny = 2\ny = 3\n", encoding="utf-8")
    second = scanner.scan(tmp_path)

    assert second.fingerprint != first.fingerprint


def test_a_fresh_map_is_reused_without_rewalking_the_tree(tmp_path: Path):
    """Inside the freshness window a repeat scan must not even fingerprint."""
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    scanner = ProjectScanner(fresh_seconds=60.0)
    first = scanner.scan(tmp_path)

    (tmp_path / "b.py").write_text("y = 2\n", encoding="utf-8")
    assert scanner.scan(tmp_path) is first

    assert scanner.scan(tmp_path, force=True).file_count == 2


def test_a_non_repository_reports_that_rather_than_guessing(tmp_path: Path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    assert ProjectScanner().scan(tmp_path).git == {"repository": False}
