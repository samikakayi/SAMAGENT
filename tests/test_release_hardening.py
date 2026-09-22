"""Release hardening: credential loading paths, error UX, Sorani routing, permissions."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from sam_backend.config import Settings
from sam_backend.db import Database
from sam_backend.schemas import SettingsUpdate
from sam_backend.secrets import SecretStore, resolve_credential

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SENTINEL = "sk-or-v1-HARDENING-SENTINEL-abcdefghij0123456789"
# A complete key-shaped literal, as opposed to a bare prefix appearing inside
# a detection regex, which is not a credential.
KEY_LITERAL = re.compile(r"sk-or-v1-[A-Za-z0-9_-]{12,}")


# --- Credential loading paths -------------------------------------------------


def test_the_process_environment_supplies_the_credential(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", SENTINEL)
    value, source = resolve_credential("openrouter_api_key", SecretStore(tmp_path))
    assert value == SENTINEL and source == "environment"


def test_the_secret_store_supplies_the_credential(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    store = SecretStore(tmp_path)
    store.set("openrouter_api_key", SENTINEL)
    value, source = resolve_credential("openrouter_api_key", store)
    assert value == SENTINEL and source == "secret_store"


def test_a_dotenv_file_reaches_settings(tmp_path: Path, monkeypatch):
    """A project-local .env is loaded into the process environment at startup."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    (tmp_path / ".env").write_text(f"OPENROUTER_API_KEY={SENTINEL}\n", encoding="utf-8")
    settings = Settings.from_env(project_root=tmp_path)
    assert settings.openrouter_api_key == SENTINEL


def test_the_environment_outranks_both_other_sources(tmp_path: Path, monkeypatch):
    """An operator's explicit environment must win over anything stored earlier."""
    store = SecretStore(tmp_path)
    store.set("openrouter_api_key", "sk-or-v1-FROM-THE-STORE-000000000000")
    monkeypatch.setenv("OPENROUTER_API_KEY", SENTINEL)
    value, source = resolve_credential("openrouter_api_key", store)
    assert value == SENTINEL and source == "environment"


def test_no_shipped_file_contains_a_credential():
    """No source, config, or frontend file may carry a key.

    Tests and acceptance scripts are excluded: they use obvious sentinels on
    purpose to prove containment, and a separate test asserts they stay obvious.
    """
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or any(
            part in {".venv", "work", "data", "__pycache__", ".git", ".pytest_cache", "tools", "tests"}
            for part in path.parts
        ):
            continue
        if path.suffix.lower() not in {".py", ".js", ".html", ".css", ".yaml", ".yml", ".json", ".md", ".ps1", ".txt"}:
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        # A detection pattern in the policy engine legitimately names the prefix,
        # so only a complete key-shaped literal counts as a leak.
        found = KEY_LITERAL.findall(content)
        assert not found, f"{path} contains a literal OpenRouter key"


def test_credentials_in_test_files_are_unmistakable_sentinels():
    """Anything key-shaped under tests/ or tools/ must read as a fixture.

    Only complete key-shaped literals are examined; a bare `sk-or-v1-` fragment
    used as a search needle by these very scanners is not a credential.
    """
    markers = ("SENTINEL", "TEST", "INVALID", "UNREACHABLE", "FROM-THE-STORE", "FROM-ENVIRONMENT")
    for folder in ("tests", "tools"):
        for path in (PROJECT_ROOT / folder).rglob("*.py"):
            for literal in KEY_LITERAL.findall(path.read_text(encoding="utf-8", errors="ignore")):
                assert any(marker in literal.upper() for marker in markers), (
                    f"{path.name} has a key-shaped literal that does not read as a fixture: {literal[:40]}"
                )


def test_the_litellm_config_reads_the_key_from_the_environment():
    config = (PROJECT_ROOT / "litellm-config.yaml").read_text(encoding="utf-8")
    assert "os.environ/OPENROUTER_API_KEY" in config
    assert "sk-or-" not in config


# --- Error messages -----------------------------------------------------------

REQUIRED_ERROR_CODES = {
    "WINDOW_NOT_FOUND", "TRADINGVIEW_NOT_FOREGROUND", "CALIBRATION_REQUIRED",
    "CALIBRATION_NOT_VERIFIED", "CALIBRATION_STALE_GEOMETRY", "CALIBRATION_DRIFTED",
    "CHART_PANNED", "PRICE_OFF_SCREEN", "ANCHOR_OFF_SCREEN", "CONSTRUCTION_OFF_SCREEN",
    "COMPUTER_CONTROL_DISABLED", "SCREEN_ACCESS_DISABLED", "TIME_CALIBRATION_REQUIRED",
    "DRAWING_NOT_VERIFIED", "OCR_UNAVAILABLE", "NO_REPLAY_SESSION", "UNKNOWN_TRIGGER",
}


def test_every_documented_error_code_is_actually_raised_somewhere():
    sources = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in (PROJECT_ROOT / "sam_backend").rglob("*.py")
    )
    missing = [code for code in REQUIRED_ERROR_CODES if code not in sources]
    assert not missing, f"error codes referenced but never produced: {missing}"


def test_failure_messages_tell_the_user_what_to_do(tmp_path: Path):
    """An error has to say what happened and what the user can do about it."""
    from sam_backend.cancellation import CancellationManager
    from sam_backend.db import Database
    from sam_backend.trading.service import TradingService

    settings = Settings(project_root=tmp_path, workspace_root=tmp_path / "w", data_dir=tmp_path / "d")
    settings.prepare()
    trading = TradingService(settings, Database(settings.database_path), CancellationManager())

    for result in (trading.draw_annotation("support", 1.0), trading.calibrate_chart(), trading.draw_analysis()):
        assert result.error, "a failure must carry a message"
        assert len(result.error) > 25, f"message is too terse to act on: {result.error!r}"
        assert "Traceback" not in result.error
        assert result.executed is False, "a refused action must not claim it executed"


def test_a_refusal_never_claims_to_be_verified(tmp_path: Path):
    from sam_backend.cancellation import CancellationManager
    from sam_backend.db import Database
    from sam_backend.trading.service import TradingService

    settings = Settings(project_root=tmp_path, workspace_root=tmp_path / "w", data_dir=tmp_path / "d")
    settings.prepare()
    trading = TradingService(settings, Database(settings.database_path), CancellationManager())
    for result in (trading.draw_annotation("support", 1.0), trading.verify_calibration()):
        assert result.verified is False


# --- Sorani intent routing ----------------------------------------------------

SORANI_COMMANDS = [
    "سام، ترەیدینگ ڤیو بکەرەوە.",
    "XAUUSD بە SnR شیکاریم بۆ بکە.",
    "لە 1H و 15m ببینە، لە 5m entry بدۆزەوە.",
    "ئێستا بە Wyckoff شیکاریم بۆ بکە.",
    "هەموو هێڵەکانی SAM بسڕەوە.",
    "ئەم setup ـە چاودێری بکە.",
    "وەستە.",
]


@pytest.mark.parametrize("command", SORANI_COMMANDS)
def test_sorani_text_commands_route_to_a_trading_intent(command, tmp_path: Path):
    from sam_backend.cancellation import CancellationManager
    from sam_backend.db import Database
    from sam_backend.trading.service import TradingService

    settings = Settings(project_root=tmp_path, workspace_root=tmp_path / "w", data_dir=tmp_path / "d")
    settings.prepare()
    trading = TradingService(settings, Database(settings.database_path), CancellationManager())
    routed = trading.route_natural_intent(command)
    # Either a trading skill claims it, or it falls through to the agent loop;
    # what must never happen is an exception on real user phrasing.
    assert routed is None or routed.status is not None


def test_sorani_stop_words_are_recognised_as_interruptions():
    from sam_backend.voice import BargeInController

    assert BargeInController.is_stop_command("وەستە") is True
    assert BargeInController.is_stop_command("وەستە، بچۆ 5 خولەکی") is True
    assert BargeInController.is_stop_command("XAUUSD شیکاریم بۆ بکە") is False


def test_sorani_speech_is_claimed_only_when_a_provider_backs_it():
    """The distinction has to be explicit so nobody assumes voice works.

    Sorani speech is real, but only through KurdishTTS. With no provider the
    honest answer is still "text yes, speech no".
    """
    from sam_backend.voice import language_support

    without = language_support("ckb-IQ")
    assert without["stt_supported"] is False
    assert "typed" in without["reason"].lower()

    with_provider = language_support("ckb-IQ", {"provider": "kurdishtts", "status": "CONNECTED"})
    assert with_provider["stt_supported"] is True


# --- Permission persistence ---------------------------------------------------


def test_dangerous_permissions_default_to_off(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("SAM_COMPUTER_CONTROL", raising=False)
    monkeypatch.delenv("SAM_SCREEN_ACCESS", raising=False)
    settings = Settings.from_env(project_root=tmp_path)
    assert settings.computer_control_enabled is False
    assert settings.screen_access_enabled is False


def test_permission_state_is_persisted_deliberately_not_by_accident(tmp_path: Path):
    settings = Settings(project_root=tmp_path, workspace_root=tmp_path / "w", data_dir=tmp_path / "d")
    settings.prepare()
    values = SettingsUpdate(computer_control_enabled=True, screen_access_enabled=True).provided()
    saved = Database(settings.database_path).update_settings(values)
    # The update schema is the one list of what persists, and it names these
    # by design, so the UI toggle survives a restart rather than resetting.
    assert saved["computer_control_enabled"] is True
    assert "computer_control_enabled" in SettingsUpdate.model_fields


def test_sam_refuses_to_run_elevated(tmp_path: Path, monkeypatch):
    import sam_backend.config as config

    settings = Settings(project_root=tmp_path, workspace_root=tmp_path / "w", data_dir=tmp_path / "d")
    monkeypatch.setattr(config, "is_elevated_windows_process", lambda: True)
    with pytest.raises(PermissionError, match="Administrator"):
        settings.prepare()


# --- Importing SAM is not starting SAM ----------------------------------------
#
# `sam_backend/app.py` used to build the application at module scope, and
# `sam_backend/__init__.py` imports it, so `import sam_backend` ran the whole
# composition root: DPI setup, Settings.prepare(), the privilege guard, the
# database and every service. On a Windows CI runner, which is elevated, that
# made the guard fire during test collection and no test could even be found.
#
# Three contracts, deliberately kept apart:
#   A. importing the package constructs nothing
#   B. asking for an application constructs one
#   C. asking for one on an elevated Windows process is still refused

IMPORT_WHILE_ELEVATED = """
import os, ctypes
if os.name == "nt":
    # What a GitHub Windows runner reports.
    ctypes.windll.shell32.IsUserAnAdmin = lambda: 1
import sam_backend
import sam_backend.app
print("imported", hasattr(sam_backend.app, "app"), callable(sam_backend.app.create_app))
"""


def test_importing_sam_backend_does_not_start_it(tmp_path: Path):
    """Import must survive on an elevated process, because import is not startup."""
    import subprocess
    import sys

    finished = subprocess.run(
        [sys.executable, "-c", IMPORT_WHILE_ELEVATED], cwd=PROJECT_ROOT,
        capture_output=True, text=True, encoding="utf-8",
    )

    assert finished.returncode == 0, f"importing SAM started it:\n{finished.stderr}"
    assert "Administrator" not in finished.stderr
    # No application object is left lying around at module scope, and the
    # factory is still exported for the callers that do want one.
    assert finished.stdout.strip() == "imported False True", finished.stdout


def test_asking_for_an_application_still_builds_one(tmp_path: Path):
    from sam_backend.app import create_app

    application = create_app(Settings(
        project_root=tmp_path, workspace_root=tmp_path / "w", data_dir=tmp_path / "d",
    ))

    assert application.state.settings is not None
    assert any(getattr(route, "path", "") == "/api/health" for route in application.routes)


def test_asking_for_an_application_while_elevated_is_still_refused(tmp_path: Path, monkeypatch):
    """The guard protects startup, and building the app is startup."""
    import sam_backend.config as config
    from sam_backend.app import create_app

    monkeypatch.setattr(config, "is_elevated_windows_process", lambda: True)

    with pytest.raises(PermissionError, match="Administrator"):
        create_app(Settings(
            project_root=tmp_path, workspace_root=tmp_path / "w", data_dir=tmp_path / "d",
        ))


def test_the_launch_command_names_something_that_exists():
    """`python -m sam_backend` must still resolve to a real application."""
    from sam_backend.app import create_app

    source = (PROJECT_ROOT / "sam_backend" / "__main__.py").read_text(encoding="utf-8")

    assert "sam_backend.app:create_app" in source, "the launcher names a target that no longer exists"
    assert "factory=True" in source, "uvicorn would treat the factory as an application"
    assert callable(create_app)


# --- Credential detection by shape --------------------------------------------


@pytest.mark.parametrize(
    "sample",
    [
        "sk-or-v1-" + "a" * 24,
        "sk-ant-api03-" + "b" * 40,
        "sk-proj-" + "c" * 24,
        "ghp_" + "d" * 36,
        "AKIAIOSFODNN7EXAMPLE",
        "xoxb-123456789012-abcdefghij",
        "AIza" + "e" * 35,
    ],
)
def test_a_bare_provider_key_is_refused_even_without_a_variable_name(sample, tmp_path: Path):
    """Matching only on a nearby variable name missed a key written on its own."""
    from sam_backend.policy import RiskPolicy

    settings = Settings(project_root=tmp_path, workspace_root=tmp_path / "w", data_dir=tmp_path / "d")
    decision = RiskPolicy(settings).evaluate("write_file", {"path": "notes.txt", "content": sample})
    assert decision.allowed is False
    assert decision.sensitive is True


@pytest.mark.parametrize(
    "sample",
    [
        "Support sits at 3400.50 on XAUUSD.",
        "Compare SnR, Wyckoff and ICT on the 15m chart.",
        "sk-or",
        "The RR is 2.5 with TP1 at 4460.",
        "AKIA",
    ],
)
def test_ordinary_trading_text_is_not_mistaken_for_a_credential(sample, tmp_path: Path):
    from sam_backend.policy import RiskPolicy

    settings = Settings(project_root=tmp_path, workspace_root=tmp_path / "w", data_dir=tmp_path / "d")
    assert RiskPolicy(settings).evaluate("write_file", {"path": "notes.txt", "content": sample}).allowed is True


def test_an_unapproved_write_cannot_escape_the_workspace(tmp_path: Path):
    from sam_backend.cancellation import CancellationManager
    from sam_backend.db import Database
    from sam_backend.tools import ToolRegistry

    settings = Settings(project_root=tmp_path, workspace_root=tmp_path / "w", data_dir=tmp_path / "d")
    settings.prepare()
    tools = ToolRegistry(settings, Database(settings.database_path), cancellation=CancellationManager())
    outcome = tools.execute("write_file", {"path": "../escaped.txt", "content": "x"}, approved=False)
    assert outcome.ok is False
    assert not (settings.workspace_root.parent / "escaped.txt").exists()


def test_a_protected_system_path_cannot_be_approved_at_all(tmp_path: Path):
    from sam_backend.policy import RiskPolicy

    settings = Settings(project_root=tmp_path, workspace_root=tmp_path / "w", data_dir=tmp_path / "d")
    decision = RiskPolicy(settings).evaluate("write_file", {"path": "C:/Windows/win.ini", "content": "x"})
    assert decision.allowed is False, "a system path must be denied, not merely gated"
