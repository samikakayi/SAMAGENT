from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import Settings


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(slots=True)
class PolicyDecision:
    allowed: bool
    approval_required: bool
    risk_level: RiskLevel
    reason: str
    sensitive: bool = False


_CREDENTIAL_NAMES = {
    ".env", ".env.local", ".env.production", "credentials", "credentials.json",
    "id_rsa", "id_ed25519", "known_hosts", ".npmrc", ".pypirc", ".netrc",
    "login data", "cookies", "wallet.dat", "secrets.json",
}
_CREDENTIAL_PARTS = {".ssh", ".aws", ".azure", ".gnupg", "keychain", "passwords", "credentials"}


def windows_path_violation(raw_path: str) -> str | None:
    """Reject Windows path forms that bypass ordinary containment checks."""
    if os.name != "nt":
        return None
    raw = raw_path.strip()
    lowered = raw.lower()
    if lowered.startswith(("\\\\?\\", "\\\\.\\", "\\??\\")):
        return "Windows device and extended path namespaces are blocked."
    drive = Path(raw).drive
    remainder = raw[len(drive):] if drive else raw
    if ":" in remainder:
        return "NTFS alternate data streams are blocked."
    reserved = re.compile(r"(?i)^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?$")
    for part in re.split(r"[\\/]", remainder):
        if not part or part in {".", ".."}:
            continue
        if part.endswith((" ", ".")) or reserved.match(part):
            return "Reserved Windows device names and trailing-dot/space paths are blocked."
    return None


class RiskPolicy:
    """Central policy for every tool call. Policy is evaluated again after approval."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.workspace = settings.workspace_root.resolve()

    def evaluate(self, tool_name: str, arguments: dict[str, Any]) -> PolicyDecision:
        if self.contains_embedded_secret(arguments):
            return PolicyDecision(
                False, False, RiskLevel.CRITICAL,
                "Tool arguments appear to contain a credential. SAM will not persist or execute embedded secrets; use the target application's own credential store.",
                sensitive=True,
            )
        if tool_name in {"list_files", "search_files", "read_file", "write_file", "replace_text", "delete_path"}:
            return self._file_policy(tool_name, arguments)
        if tool_name in {"market_snapshot", "analyze_market", "backtest_strategy", "list_entry_triggers"}:
            return PolicyDecision(True, False, RiskLevel.LOW, "Read-only deterministic market-data analysis; live broker order execution is not implemented.")
        if tool_name in {
            "get_tradingview_state", "list_processes", "process_info", "list_windows",
            "list_installed_apps", "list_theories", "list_sam_drawings",
        }:
            return PolicyDecision(True, False, RiskLevel.LOW, "Read-only local observation.")
        if tool_name == "set_chart_layer":
            return PolicyDecision(True, False, RiskLevel.LOW, "Updates SAM's own annotation-ownership records; the chart is not touched.")
        if tool_name in {"calibrate_chart", "verify_chart_calibration"}:
            return PolicyDecision(
                True, True, RiskLevel.MEDIUM,
                "Calibration screenshots the chart's price axis, so it needs the same approval as other screen reads.",
                sensitive=True,
            )
        if tool_name in {"draw_tradingview_level", "draw_tradingview_object", "draw_analysis_on_chart"}:
            return PolicyDecision(
                True, True, RiskLevel.MEDIUM,
                "Drawing moves the pointer and sends shortcuts to TradingView, which changes a desktop application's state.",
            )
        if tool_name == "clear_sam_drawings":
            return PolicyDecision(
                True, True, RiskLevel.HIGH,
                "Removing annotations clicks and deletes on the live chart; only SAM-owned drawings are ever targeted.",
            )
        if tool_name == "save_custom_theory":
            return PolicyDecision(True, False, RiskLevel.LOW, "Versioned local trading-memory update; existing versions are not overwritten.")
        if tool_name in {"focus_tradingview", "set_tradingview_symbol", "set_tradingview_timeframe"}:
            return PolicyDecision(True, True, RiskLevel.MEDIUM, "Changing a desktop application's state requires explicit approval when initiated by a model tool call.")
        if tool_name == "stop_process":
            return PolicyDecision(True, True, RiskLevel.HIGH, "Stopping a process can lose unsaved work and requires explicit approval.")
        if tool_name == "window_action":
            action = str(arguments.get("action", "")).lower()
            risk = RiskLevel.HIGH if action == "close" else RiskLevel.MEDIUM
            reason = "Closing a window can lose unsaved work." if action == "close" else "Changing a desktop window requires approval."
            return PolicyDecision(True, True, risk, reason)
        if tool_name == "read_clipboard":
            return PolicyDecision(True, True, RiskLevel.HIGH, "Clipboard text may contain credentials or private data and is omitted from model context.", sensitive=True)
        if tool_name in {"write_clipboard", "capture_screen", "inspect_ui_tree"}:
            sensitive = tool_name in {"capture_screen", "inspect_ui_tree"}
            return PolicyDecision(True, True, RiskLevel.HIGH, "Desktop/clipboard access may expose or alter private user state and requires approval.", sensitive=sensitive)
        if tool_name == "emergency_stop":
            return PolicyDecision(True, False, RiskLevel.LOW, "Emergency stop is always allowed.")
        if tool_name == "run_terminal":
            decision = self._command_policy(str(arguments.get("command", "")))
            raw_cwd = str(arguments.get("cwd", "."))
            cwd = Path(raw_cwd).expanduser()
            candidate = cwd.resolve(strict=False) if cwd.is_absolute() else (self.workspace / cwd).resolve(strict=False)
            if decision.allowed and not candidate.is_relative_to(self.workspace):
                return PolicyDecision(True, True, RiskLevel.HIGH, "Running a command outside the configured workspace requires explicit approval.")
            if decision.allowed and not decision.approval_required and self.settings.permission_mode != "trusted":
                return PolicyDecision(
                    True, True, RiskLevel.MEDIUM,
                    "Guarded and strict modes require approval before terminal execution; trusted mode still gates risky commands.",
                )
            return decision
        if tool_name == "run_python":
            return PolicyDecision(True, True, RiskLevel.HIGH, "Python can perform arbitrary computer actions; review the code before running it.")
        if tool_name == "launch_app":
            application = str(arguments.get("application", "")).strip()
            app_args = [str(item) for item in (arguments.get("arguments") or [])]
            if not application:
                return PolicyDecision(False, False, RiskLevel.HIGH, "Application name is empty.")
            command_like = " ".join([application, *app_args])
            app_name = Path(application).name.lower()
            if app_name in {"runas", "runas.exe"}:
                return PolicyDecision(False, False, RiskLevel.CRITICAL, "SAM never launches applications through UAC or runas.")
            if app_name in {
                "powershell", "powershell.exe", "pwsh", "pwsh.exe", "cmd", "cmd.exe",
                "wscript", "wscript.exe", "cscript", "cscript.exe", "mshta", "mshta.exe",
                "rundll32", "rundll32.exe",
            }:
                nested = self._command_policy(command_like)
                if not nested.allowed:
                    return nested
            return PolicyDecision(True, True, RiskLevel.HIGH, "Launching an application can change external state.")
        if tool_name == "browser_automate":
            return self._browser_policy(arguments)
        if tool_name == "open_url":
            url = str(arguments.get("url", ""))
            if not re.match(r"^https?://", url, re.I):
                return PolicyDecision(False, False, RiskLevel.HIGH, "Only http:// and https:// URLs are accepted.")
            return PolicyDecision(
                True, self.settings.permission_mode == "strict", RiskLevel.MEDIUM,
                "Strict mode requires approval before navigation." if self.settings.permission_mode == "strict"
                else "Navigating to an ordinary web page is allowed in the current mode.",
            )
        if tool_name in {"remember", "recall", "create_plan"}:
            return PolicyDecision(True, False, RiskLevel.LOW, "Local planning and memory action.")
        return PolicyDecision(False, False, RiskLevel.HIGH, f"Unknown tool: {tool_name}")

    def _browser_policy(self, arguments: dict[str, Any]) -> PolicyDecision:
        start_url = str(arguments.get("url", ""))
        urls = [start_url]
        actions = arguments.get("actions") or []
        if not isinstance(actions, list) or not actions:
            return PolicyDecision(False, False, RiskLevel.HIGH, "A browser workflow needs a non-empty actions list.")
        if len(actions) > 20:
            return PolicyDecision(False, False, RiskLevel.HIGH, "A browser workflow is limited to 20 actions.")

        interactive = False
        for action in actions:
            if not isinstance(action, dict):
                return PolicyDecision(False, False, RiskLevel.HIGH, "Every browser action must be a structured object.")
            action_type = str(action.get("type", "")).lower()
            if action_type not in {"goto", "click", "fill", "press", "wait_for", "extract_text", "screenshot"}:
                return PolicyDecision(False, False, RiskLevel.HIGH, f"Unsupported browser action: {action_type or '(empty)'}")
            if action_type == "goto":
                urls.append(str(action.get("url", "")))
            if action_type in {"click", "fill", "press", "screenshot"}:
                interactive = True

        for url in urls:
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
                return PolicyDecision(False, False, RiskLevel.HIGH, "Browser automation accepts only credential-free absolute http(s) URLs.")
            try:
                port = parsed.port or (443 if parsed.scheme == "https" else 80)
            except ValueError:
                return PolicyDecision(False, False, RiskLevel.HIGH, "The browser URL contains an invalid port.")
            if parsed.hostname in {"127.0.0.1", "localhost", "::1"} and port == self.settings.port:
                if interactive:
                    return PolicyDecision(False, False, RiskLevel.CRITICAL, "SAM cannot automate or click its own local control and approval interface.")

        if interactive:
            reason = "Clicking, typing, pressing keys, or writing a screenshot can change external or local state; review the complete workflow."
            risk = RiskLevel.HIGH
        else:
            reason = "Browser navigation and extraction require explicit single-use approval, even in a fresh isolated profile."
            risk = RiskLevel.MEDIUM
        return PolicyDecision(True, True, risk, reason)

    def _file_policy(self, tool_name: str, arguments: dict[str, Any]) -> PolicyDecision:
        raw_path = os.path.expandvars(str(arguments.get("path", ".")))
        violation = windows_path_violation(raw_path)
        if violation:
            return PolicyDecision(False, False, RiskLevel.CRITICAL, violation)
        path = Path(raw_path).expanduser()
        candidate = path.resolve() if path.is_absolute() else (self.workspace / path).resolve()
        if candidate == self.settings.data_dir or candidate.is_relative_to(self.settings.data_dir):
            return PolicyDecision(False, False, RiskLevel.CRITICAL, "SAM's private database and audit directory is not exposed as a model tool.")
        outside = not candidate.is_relative_to(self.workspace)
        sensitive = self.is_credential_path(candidate)
        modifies = tool_name in {"write_file", "replace_text", "delete_path"}
        if modifies and self.is_protected_system_path(candidate) and not self.settings.allow_unsafe_system_actions:
            return PolicyDecision(False, False, RiskLevel.CRITICAL, "Safe mode blocks direct changes to protected Windows and filesystem roots.")
        if sensitive:
            return PolicyDecision(
                True, True, RiskLevel.HIGH,
                "This path may contain credentials or private keys. Its contents are never written to the audit log.",
                sensitive=True,
            )
        if outside:
            action = "Deleting" if tool_name == "delete_path" else "Accessing or changing"
            return PolicyDecision(True, True, RiskLevel.HIGH, f"{action} a path outside SAM's workspace requires explicit approval.")
        if tool_name == "delete_path":
            return PolicyDecision(True, True, RiskLevel.HIGH, "Deletion requires explicit approval even inside the workspace.")
        if tool_name == "replace_text" or (tool_name == "write_file" and candidate.exists()):
            return PolicyDecision(True, True, RiskLevel.HIGH, "Overwriting an existing file requires explicit approval.")
        if tool_name == "write_file":
            return PolicyDecision(
                True, self.settings.permission_mode == "strict", RiskLevel.MEDIUM,
                "Strict mode requires approval before creating a workspace file."
                if self.settings.permission_mode == "strict" else "This creates a file inside SAM's workspace.",
            )
        return PolicyDecision(True, False, RiskLevel.LOW, "Read-only workspace action.")

    def _command_policy(self, command: str) -> PolicyDecision:
        normalized = " ".join(command.lower().split())
        if not normalized:
            return PolicyDecision(False, False, RiskLevel.LOW, "The command is empty.")

        hard_denied = [
            r"\bformat(?:\.com)?\b", r"\bdiskpart\b", r"\bclear-disk\b", r"\bremove-partition\b",
            r"rm\s+-rf\s+[/~](?:\s|$)", r"remove-item\s+(?:-[a-z]+\s+)*['\"]?[a-z]:\\\s*(?:-recurse|-r)\b",
            r"\bdd\s+.*\bof=/dev/", r"\bmanage-bde\b", r"\bclear-tpm\b", r"\bdisable-bitlocker\b",
            r"\bbcdedit\b", r"\bbootrec\b", r"\bcipher(?:\.exe)?\s+/w", r"\bvssadmin\s+delete\b",
            r"(?:^|\s)-(?:enc|encodedcommand)\b", r"\bfrombase64string\b", r"\binvoke-expression\b", r"(?:^|[;&|\s])iex(?:\s|\()",
            r"\bdownloadstring\b", r"\bmshta(?:\.exe)?\b", r"\brundll32(?:\.exe)?\b",
            r"\bmimikatz\b", r"\bsekurlsa\b", r"\blsass\b", r"\bprocdump(?:64)?(?:\.exe)?\b.*\s-ma\b",
            r"\breg\s+save\s+hk(?:lm|cu)\\sam\b", r"\bntdsutil\b", r"\bsecretsdump\b",
            r"\bstart-process\b.*-verb\s+runas", r"\brunas(?:\.exe)?\b", r"(?:^|[;&|\s])sudo\s", r"\bpkexec\b",
            r"\bset-mppreference\b.*disable", r"\bremove-mppreference\b", r"\bnetsh\b.*firewall.*(?:off|disable)",
            r"`", r"\[scriptblock\]::create", r"\badd-type\b.*(?:dllimport|reflection|unsafe)",
        ]
        if any(re.search(pattern, normalized, re.I) for pattern in hard_denied) and not self.settings.allow_unsafe_system_actions:
            return PolicyDecision(False, False, RiskLevel.CRITICAL, "Safe mode blocks elevation, obfuscated execution, credential dumping, security-control tampering, and destructive disk operations.")

        high_patterns = [
            r"\bremove-item\b", r"\bdel(?:ete)?\b", r"\brm\b", r"\brmdir\b", r"\bmove-item\b",
            r"\b(?:set|add|clear)-content\b", r"\bout-file\b", r"\bcopy-item\b", r"(?:^|\s)>{1,2}\s*\S",
            r"\breg(?:\.exe)?\s+(add|delete|import|restore)\b", r"\bset-itemproperty\b.*registry",
            r"\bwinget\s+(install|uninstall|upgrade)\b", r"\bchoco\s+(install|uninstall|upgrade)\b",
            r"\b(?:pip|uv\s+pip)\s+install\b", r"\bnpm\s+(install|uninstall|publish)\b", r"\bpnpm\s+(add|install|remove|publish)\b",
            r"\bshutdown\b", r"\brestart-computer\b", r"\bstop-computer\b", r"\bsc(?:\.exe)?\s+(create|delete|config|stop)\b",
            r"\bnet\s+user\b", r"\bnet\s+localgroup\b", r"\bset-executionpolicy\b", r"\bschtasks\b",
            r"\b(?:takeown|icacls|new-service|set-service)\b", r"\bnetsh\b.*firewall",
            r"\bcurl\b.*(?:-x\s+post|--data|--upload-file|-t\s|-o\s|--output)",
            r"\b(?:invoke-restmethod|invoke-webrequest)\b.*(?:-method\s+(post|put|patch|delete)|-outfile)",
            r"\bgit\s+(?:push|clean\b|reset\s+--hard)", r"\brobocopy\b.*\/(?:mir|purge)\b",
            r"\bgh\s+(pr\s+create|issue\s+create|release\s+create)\b",
            r"\bstart-process\b", r"\b(?:powershell|pwsh)(?:\.exe)?\b.*\s-(?:command|file)\b",
            r"\b(?:python|python3|py|node|ruby|perl)\b.*(?:\s-c\s|\s-e\s|\.(?:py|js|rb|pl)\b)", r"\bcmd(?:\.exe)?\s+/c\b",
        ]
        if any(re.search(pattern, normalized, re.I) for pattern in high_patterns):
            return PolicyDecision(True, True, RiskLevel.HIGH, "The command can delete data, install software, alter the system, elevate privileges, or affect an external service.")

        credential_patterns = [r"\bget-credential\b", r"credential\s+manager", r"\.ssh[\\/]", r"\.aws[\\/]", r"\bsecrets?\b"]
        if any(re.search(pattern, normalized, re.I) for pattern in credential_patterns):
            return PolicyDecision(True, True, RiskLevel.HIGH, "The command may access credentials or secrets.", sensitive=True)

        if re.search(
            r"(?i)(?:\b[a-z]:[\\/]|\\\\|\$(?:home\b|env:)|(?:^|[\s\"'=])\.\.[\\/])",
            command,
        ):
            return PolicyDecision(True, True, RiskLevel.HIGH, "The command references an absolute or profile path; review the exact target before execution.")

        return PolicyDecision(True, False, RiskLevel.MEDIUM, "Command execution is limited by timeout and output size.")

    @staticmethod
    def is_credential_path(path: Path) -> bool:
        lowered_parts = {part.lower() for part in path.parts}
        return path.name.lower() in _CREDENTIAL_NAMES or bool(lowered_parts & _CREDENTIAL_PARTS)

    @staticmethod
    def is_protected_system_path(path: Path) -> bool:
        candidate = path.resolve(strict=False)
        if candidate.parent == candidate or candidate == Path(candidate.anchor):
            return True
        roots = []
        for variable in ("SYSTEMROOT", "WINDIR", "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMDATA"):
            value = os.getenv(variable)
            if value:
                roots.append(Path(value).resolve(strict=False))
        return any(candidate == root or candidate.is_relative_to(root) for root in roots)

    @staticmethod
    def sanitize_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
        secret_keys = {"api_key", "apikey", "authorization", "password", "token", "secret", "cookie"}

        def clean(value: Any, key: str = "") -> Any:
            if key.lower().replace("-", "_") in secret_keys or any(part in key.lower() for part in ("password", "token", "secret", "api_key")):
                return "[REDACTED]"
            if isinstance(value, dict):
                cleaned = {str(k): clean(v, str(k)) for k, v in value.items()}
                if str(value.get("type", "")).lower() == "fill" and re.search(
                    r"(?i)(password|passwd|secret|token|api.?key|credit.?card|cvv)", str(value.get("selector", ""))
                ):
                    cleaned["value"] = "[REDACTED]"
                return cleaned
            if isinstance(value, list):
                return [clean(item) for item in value]
            text = str(value)
            return text[:2000] + ("..." if len(text) > 2000 else "") if isinstance(value, str) else value

        return clean(arguments)

    @staticmethod
    def contains_embedded_secret(arguments: dict[str, Any]) -> bool:
        secret_key = re.compile(r"(?i)(password|passwd|api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|private[_-]?key)")
        literal = re.compile(
            r"(?i)(?:bearer\s+[a-z0-9._~+/-]{16,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|"
            r"(?:password|passwd|api[_-]?key|access[_-]?token|client[_-]?secret)\s*[:=]\s*['\"]?[^\s'\"]{6,})"
        )
        # Provider keys are recognisable by shape alone. Matching only on a
        # nearby variable name misses a model that writes a bare key into a file,
        # which is exactly the case worth catching.
        key_shape = re.compile(
            r"(?:sk-or-v1-[A-Za-z0-9._-]{16,}"          # OpenRouter
            r"|sk-ant-[A-Za-z0-9._-]{16,}"              # Anthropic
            r"|sk-proj-[A-Za-z0-9._-]{16,}"             # OpenAI project
            r"|sk-[A-Za-z0-9]{32,}"                     # OpenAI classic
            r"|gh[pousr]_[A-Za-z0-9]{30,}"              # GitHub
            r"|AKIA[0-9A-Z]{16}"                        # AWS access key id
            r"|xox[baprs]-[A-Za-z0-9-]{10,}"            # Slack
            r"|AIza[0-9A-Za-z._-]{30,})"                # Google
        )

        def inspect(value: Any, key: str = "") -> bool:
            if secret_key.search(key) and value not in (None, "", "[REDACTED]"):
                return True
            if isinstance(value, dict):
                return any(inspect(item, str(item_key)) for item_key, item in value.items())
            if isinstance(value, list):
                return any(inspect(item) for item in value)
            if not isinstance(value, str):
                return False
            return bool(literal.search(value)) or bool(key_shape.search(value))

        return inspect(arguments)
