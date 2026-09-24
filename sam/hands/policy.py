"""Risk classification for PowerShell commands, file paths and URLs.

Ported from v1 ``policy.py`` (its hard-deny list, high-risk list, Windows
path-form checks and credential-path rules survived a security review there)
and mapped onto SAM 2's three tiers, decided by code and never by the model:

- ``safe``: read-only commands on an allowlist; reading/listing files;
  writing inside SAM's own folders (``hands.projects_dir``, ``workspace/``).
- ``confirm``: anything that changes the machine (spoken "بەڵێ" or a click).
- ``blocked``: credential dumping or reading key stores, disabling security
  tools, obfuscated/elevated execution, disk wiping, mass deletion, trading
  orders, device/ADS path tricks.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..textnorm import normalize_ckb
from . import _win

Verdict = tuple[str, str]  # (risk, reason)

# --- PowerShell ------------------------------------------------------------------
_BLOCK_RULES: tuple[tuple[str, str], ...] = (
    # obfuscation / elevation (v1 hard_denied)
    (r"(?:^|\s)-(?:e|en|enc|enco|encodedcommand|ec)\s", "encoded (hidden) commands are not allowed"),
    (r"\bfrombase64string\b|\binvoke-expression\b|(?:^|[;&|(\s])iex(?:\s|\(|$)|\bdownloadstring\b",
     "obfuscated or downloaded code execution is not allowed"),
    (r"\[scriptblock\]::create|\badd-type\b.*(?:dllimport|reflection|unsafe)|`", "hidden code tricks are not allowed"),
    (r"\bstart-process\b.*-verb\s+runas|\brunas(?:\.exe)?\b|(?:^|[;&|\s])sudo\s|\bpkexec\b",
     "SAM never runs anything as administrator"),
    (r"\bmshta(?:\.exe)?\b|\brundll32(?:\.exe)?\b|\bregsvr32(?:\.exe)?\b.*\s/i:", "script-host tricks are not allowed"),
    # disks / recovery
    (r"\bformat(?:\.com)?\s+[a-z]:|\bformat-volume\b|\bdiskpart\b|\bclear-disk\b|\bremove-partition\b|"
     r"\binitialize-disk\b|\bcipher(?:\.exe)?\s+/w|\bvssadmin\b.*\bdelete\b|\bwbadmin\b.*\bdelete\b|"
     r"\bshadowcopy\b.*\bdelete\b|\bbcdedit\b|\bbootrec\b|\bmanage-bde\b|\bdisable-bitlocker\b|\bclear-tpm\b|"
     r"\bdd\s+.*\bof=/dev/", "disk, boot and recovery changes are blocked"),
    # credential dumping / key stores
    (r"\bmimikatz\b|\bsekurlsa\b|\blsass\b|\bprocdump(?:64)?(?:\.exe)?\b.*\s-ma\b|\bcomsvcs\b|\bntdsutil\b|"
     r"\bsecretsdump\b|\blazagne\b|\breg(?:\.exe)?\s+(?:save|export)\s+hk(?:lm|ey_local_machine)\\(?:sam|system|security)\b|"
     r"\bget-storedcredential\b|protecteddata\]::unprotect|\bunprotect-cmsmessage\b|\bdpapi\b",
     "credential dumping is blocked"),
    (r"secrets\.json|sam2?\.sqlite3|\blogin data\b|\blocal state\b|\bcookies\b.*\b(?:chrome|edge|firefox)\b|"
     r"\b(?:chrome|edge|brave|firefox)\b.*\bcookies\b|\bkey[34]\.db\b|\blogins\.json\b|\bwallet\.dat\b|"
     r"(?:^|[\\/\s'\"])\.env(?![.\w])|[\\/]\.ssh[\\/]|\bid_(?:rsa|ed25519|ecdsa)\b|[\\/]\.aws[\\/]|[\\/]\.azure[\\/]|"
     r"[\\/]\.gnupg[\\/]", "reading stored keys, passwords or browser credentials is blocked"),
    # security tools
    (r"\bset-mppreference\b.*-disable|\badd-mppreference\b.*-exclusion|\bremove-mppreference\b|"
     r"\bnetsh\b.*firewall.*\b(?:off|disable)\b|\bnetsh\b.*\bstate\s+off\b|"
     r"\bset-netfirewallprofile\b.*-enabled\s+(?:\$?false|0)|"
     r"\b(?:sc(?:\.exe)?|stop-service|set-service)\b.*\b(?:windefend|wscsvc|mpssvc|sense|wdnissvc|securityhealthservice)\b|"
     r"\buninstall-windowsfeature\b.*defender|windows defender\\|\bdisable-computerrestore\b|\btamper|"
     # the same services stopped through ForEach-Object's member shorthand ("| % Stop")
     r"\b(?:windefend|wscsvc|mpssvc|sense|wdnissvc|securityhealthservice)\b.*"
     r"(?:(?:%|foreach-object)\s+(?:-membername\s+)?stop\b|\.stop\s*\()",
     "turning off security protection is blocked"),
    # trading orders
    (r"\border_send\b|\border_check\b|\bpositions?_close\b", "SAM never places or changes trading orders"),
    # critical system processes: stopping one crashes or logs off Windows
    # (v1 windows_control refused to stop protected processes as well)
    (r"\b(?:stop-process|spps|kill|taskkill)\b.*" + r"(?<![\w-])(?:csrss|wininit|winlogon|smss|services|lsass|svchost|dwm)"
     r"(?:\.exe)?(?![\w-])|(?<![\w-])(?:csrss|wininit|winlogon|smss|services|lsass|svchost|dwm)(?:\.exe)?(?![\w-])"
     r".*\|\s*(?:stop-process|spps)\b", "stopping a critical Windows process is blocked"),
)
_DELETE_VERB = r"\b(?:remove-item|ri|rm|del|erase|rd|rmdir)\b"
_ROOTISH = (r"(?:\s|^|['\"])(?:[a-z]:\\?\*?|[a-z]:\\users\\[^\\\s'\"]+\\?|~[\\/]?|\$home[\\/]?|"
            r"\$env:(?:userprofile|homedrive|systemroot|windir|programfiles|onedrive)[\\/]?|"
            r"[^\s'\"]*[\\/](?:desktop|documents|downloads|pictures|onedrive)[\\/]?\*?)(?:['\"]|\s|$)")
_MASS_DELETE = (
    re.compile(_DELETE_VERB + r".*(?:-r(?:ecurse)?\b|/s\b)" + r".*" + _ROOTISH, re.I),
    re.compile(_DELETE_VERB + r".*" + _ROOTISH + r".*(?:-r(?:ecurse)?\b|/s\b)", re.I),
    re.compile(_DELETE_VERB + r".*(?:-r(?:ecurse)?\b|/s\b).*[\\/]\*(?:\s|$|['\"])", re.I),
    re.compile(r"\b(?:get-childitem|gci|dir|ls)\b.*-r(?:ecurse)?\b.*\|\s*" + _DELETE_VERB, re.I),
    re.compile(r"\brm\s+-rf\s+[/~]", re.I),
)
# v1 high_patterns: need a confirmation.
_CONFIRM_RULES: tuple[tuple[str, str], ...] = (
    (r"\b(?:remove-item|ri|del|erase|rm|rmdir|rd|move-item|mi|mv|rename-item|ren|copy-item|cp|copy)\b",
     "the command deletes, moves or copies files"),
    (r"\b(?:set|add|clear)-content\b|\bout-file\b|\btee-object\b|\bnew-item\b|(?:^|[^<>=])>{1,2}\s*\S",
     "the command writes files"),
    (r"\breg(?:\.exe)?\s+(?:add|delete|import|restore)\b|\bset-itemproperty\b|\bnew-itemproperty\b|"
     r"\bremove-itemproperty\b", "the command changes the registry"),
    (r"\b(?:winget|choco|scoop)\s+(?:install|uninstall|upgrade|remove)\b|\b(?:pip|pip3|uv\s+pip)\s+install\b|"
     r"\bnpm\s+(?:install|i|uninstall|publish)\b|\bmsiexec\b", "the command installs or removes software"),
    (r"\bshutdown\b|\brestart-computer\b|\bstop-computer\b|\blogoff\b", "the command shuts down or restarts"),
    (r"\bstop-process\b|\bkill\b|\btaskkill\b|\bstop-service\b|\brestart-service\b|\bstart-service\b|"
     r"\bset-service\b|\bsc(?:\.exe)?\s+(?:create|delete|config|stop|start)\b", "the command stops or starts programs"),
    (r"\bnet\s+(?:user|localgroup)\b|\bset-executionpolicy\b|\bschtasks\b|\bregister-scheduledtask\b|"
     r"\btakeown\b|\bicacls\b|\bnew-service\b", "the command changes accounts, permissions or scheduled tasks"),
    (r"\binvoke-(?:webrequest|restmethod)\b|\biwr\b|\birm\b|\bcurl\b|\bwget\b|\bstart-bitstransfer\b",
     "the command talks to the internet"),
    (r"\bstart-process\b|\bsaps\b|\binvoke-item\b|\bii\b|(?:^|[;|]\s*)&\s|\bpython\b|\bnode\b|\bcmd(?:\.exe)?\s+/c\b",
     "the command runs another program"),
    (r"\bget-credential\b|\bcmdkey\b|\bvaultcmd\b|\bget-clipboard\b", "the command touches credentials or private data"),
    # Name lookups and connection tests reach other computers: a DNS name built
    # from file content is a way to send data out (repair review 2026-09-24).
    (r"\b(?:resolve-dnsname|test-netconnection|tnc|test-connection|nslookup|ping|tracert|pathping)\b",
     "the command contacts other computers on the network"),
    (r"\binvoke-cimmethod\b|\binvoke-wmimethod\b|-methodname\b", "the command calls a system method"),
)
READ_ONLY_CMDLETS = frozenset("""
get-childitem gci dir ls get-content gc cat type get-item gi get-itemproperty gp get-itempropertyvalue gpv
get-location gl pwd set-location sl cd chdir push-location pop-location get-date date get-process gps ps
get-service gsv get-computerinfo get-ciminstance gcim get-wmiobject gwmi get-volume get-psdrive gdr get-disk
get-physicaldisk get-partition get-netadapter get-netipaddress get-netipconfiguration get-nettcpconnection
get-netroute get-dnsclientserveraddress get-hotfix get-timezone get-culture get-uiculture get-host
get-command gcm get-help help get-alias gal get-module gmo get-appxpackage get-startapps get-winevent get-eventlog
get-filehash get-acl get-localuser get-localgroup get-mpcomputerstatus get-uptime get-member gm get-printer
get-pnpdevice get-windowsupdatelog get-counter get-random get-variable gv get-psreadlineoption
test-path resolve-path rvpa measure-object measure
select-object select where-object where ? sort-object sort group-object group format-table ft format-list fl
format-wide fw format-custom fc out-string out-host oh convertto-json convertfrom-json convertto-csv
convertfrom-csv convertto-html select-string sls write-output echo write-host write-information foreach-object %
foreach compare-object compare diff join-path split-path new-timespan get-unique gu
""".split())
# Native programs that only read (with the argument shapes that keep them read-only).
READ_ONLY_NATIVE: dict[str, str | None] = {
    "ipconfig": r"^(?:/all)?$", "systeminfo": None, "whoami": None, "hostname": None, "tasklist": None,
    "netstat": None, "getmac": None,
    "driverquery": None, "ver": None, "where": None, "route": r"^print\b", "arp": r"^-a\b",
    "winget": r"^(?:list|search|show|--version|-v)\b", "pip": r"^(?:list|show|freeze|--version)\b",
    "git": r"^(?:--version|status|log|branch)\b", "wmic": r"^\S+(?:\s+where\s+.+)?\s+(?:get|list)\b(?!.*\bcall\b)",
    "nvidia-smi": None, "chcp": r"^$", "vol": None, "tree": None, "powercfg": r"^/(?:list|l|query|q|a|availablesleepstates)\b",
}
_SAFE_METHODS = frozenset({"tostring", "tolower", "toupper", "trim", "trimstart", "trimend", "split", "contains",
                           "startswith", "endswith", "substring", "gettype", "replace", "padleft", "padright",
                           "indexof", "round", "floor", "ceiling", "abs", "max", "min", "sqrt", "pow", "now",
                           "toshortdatestring", "tolongdatestring", "totalseconds", "totalminutes", "totalhours",
                           "totaldays", "getenvironmentvariables"})
_SAFE_TYPES = frozenset({"math", "datetime", "system.math", "system.datetime", "environment", "system.environment",
                         "timespan", "convert", "int", "double", "string", "decimal"})


def _strip_literals(command: str) -> str:
    """Remove literal strings so their contents do not look like commands.
    Double-quoted strings with ``$`` (PowerShell expands ``$(...)`` inside)
    are kept and checked like code."""
    command = re.sub(r"'[^']*'", "''", command)
    return re.sub(r'"[^"$`]*"', '""', command)


def command_words(command: str) -> tuple[list[str], list[str]]:
    """(command names, problems) for each pipeline/statement segment."""
    code = _strip_literals(command)
    problems: list[str] = []
    words: list[str] = []
    for segment in re.split(r"\|\||&&|[|;{}()\n]|\$\(", code):
        segment = segment.strip()
        if not segment or segment in ("''", '""'):
            continue
        segment = re.sub(r"^\$[\w:]+\s*[-+*/]?=\s*", "", segment)
        if not segment:
            continue
        if re.match(r"^[&.]\s", segment) or segment in ("&", "."):
            problems.append("calls something through & or dot-sourcing")
            continue
        if segment[0] in "$0123456789'\"@-,=!<>+*/%" or segment.startswith("[") and "::" not in segment:
            continue  # an expression, not a command
        match = re.match(r"^\[([\w.]+)\]::(\w+)", segment)
        if match:
            if match.group(1).lower() not in _SAFE_TYPES:
                problems.append(f"calls .NET [{match.group(1)}]")
            continue
        match = re.match(r"^([A-Za-z_?%][\w\-.:\\/]*)", segment)
        if match:
            words.append(match.group(1).lower())
    for method in re.findall(r"\.\s*([A-Za-z_]\w*)\s*\(", code):
        if method.lower() not in _SAFE_METHODS:
            problems.append(f"calls the .{method}() method")
    for type_name, _ in re.findall(r"\[([\w.]+)\]::(\w+)", code):
        if type_name.lower() not in _SAFE_TYPES:
            problems.append(f"calls .NET [{type_name}]")
    return words, problems


# ForEach-Object's member shorthand calls a .NET method by NAME, without
# parentheses: "gci -Recurse | % Delete" deleted every file permanently while
# classified "Read-only command." (repair review ps_delete_proof.py).
_MEMBER_CALL = re.compile(r"(?:^|[|;(]\s*)(?:foreach-object|foreach|%)\s+(?:-membername\s+|-m\s+)?([a-z_]\w*)\b")
_DESTRUCTIVE_METHOD = re.compile(r"\.\s*(?:delete|moveto|copyto|remove|kill|terminate|stop|encrypt|replace)\s*\(")
_RECURSE = re.compile(r"(?:^|\s)-r(?:ecurse)?\b|(?:^|\s)/s\b")
_READ_FILE = re.compile(r"\b(?:get-content|gc|cat|type|get-item|gi|select-string|sls|import-csv|import-clixml|"
                        r"get-filehash|copy-item|cp|copy|format-hex|fhx)\b")
_BUILT_PATH = re.compile(r"\$|\+|\bjoin-path\b|-join\b|\[char\]|\bchild-path\b")
_SECRET_FILE_NAMES = ("*.env", ".env", ".env.local", "secrets.json", "sam2.sqlite3", "sam.sqlite3", "id_rsa",
                      "id_ed25519", "id_ecdsa", "login data", "cookies", "local state", "logins.json", "key4.db",
                      "key3.db", "wallet.dat", "credentials", ".git-credentials", ".netrc")


def _wildcard_hits_secret(code: str) -> bool:
    """A wildcard path whose pattern matches a key/credential file name
    ('.en*', 'secret?.json', 'Log?n Data')."""
    import fnmatch

    for token in re.findall(r"[^\s'\"|;,()]*[*?][^\s'\"|;,()]*", code):
        name = re.split(r"[\\/]", token)[-1].lower()
        if name and any(fnmatch.fnmatch(secret, name) for secret in _SECRET_FILE_NAMES):
            return True
    return False


def _member_call_verdict(lowered: str) -> Verdict | None:
    code = _strip_literals(lowered)
    recursive = bool(_RECURSE.search(code))
    for method in _MEMBER_CALL.findall(code):
        if method in _SAFE_METHODS or method in ("{",):
            continue
        if recursive:
            return "blocked", ("Blocked by SAM's safety rules: a recursive listing piped into a method call "
                               f"('{method}') is a mass change.")
        return "confirm", f"Needs the user's approval: the command calls the .{method} method on each item."
    if recursive and _DESTRUCTIVE_METHOD.search(code):
        return "blocked", "Blocked by SAM's safety rules: a recursive listing piped into delete/move calls."
    return None


def classify_powershell(command: str) -> Verdict:
    """(risk, reason) for one PowerShell command line."""
    text = " ".join(str(command or "").split())
    if not text:
        return "blocked", "The command is empty."
    lowered = text.lower()
    if "\x00" in text:
        return "blocked", "The command contains a null byte."
    for pattern, reason in _BLOCK_RULES:
        if re.search(pattern, lowered, re.I):
            return "blocked", f"Blocked by SAM's safety rules: {reason}."
    if any(p.search(lowered) for p in _MASS_DELETE):
        return "blocked", "Blocked by SAM's safety rules: mass deletion (recursive delete of a whole folder tree)."
    if _READ_FILE.search(lowered) and _wildcard_hits_secret(lowered):
        return "blocked", "Blocked by SAM's safety rules: the wildcard path could match a key or password file."
    member = _member_call_verdict(lowered)
    if member is not None:
        return member
    if _READ_FILE.search(_strip_literals(lowered)) and _BUILT_PATH.search(_strip_literals(lowered).split("|")[0]):
        # A path built from variables or pieces cannot be checked against the
        # key-file rules ('.' + 'env'), so a person decides.
        return "confirm", "Needs the user's approval: the file path is built from variables or pieces."
    # Discarding error output is not writing a file.
    code = re.sub(r"[1-6*]?>\s*\$null\b|2>&1|\|\s*out-null\b", " ", _strip_literals(lowered))
    words, problems = command_words(text)
    for pattern, reason in _CONFIRM_RULES:
        if re.search(pattern, code, re.I):
            return "confirm", f"Needs the user's approval: {reason}."
    if problems:
        return "confirm", f"Needs the user's approval: the command {problems[0]}."
    if not words:
        # Only expressions ($x = 5, [math]::Round(...)): method/static calls,
        # call operators and redirections were already checked above.
        return "safe", "Expression only."
    for word in words:
        base = word.removesuffix(".exe")
        if word in READ_ONLY_CMDLETS or base in READ_ONLY_CMDLETS:
            continue
        if base in READ_ONLY_NATIVE:
            rule = READ_ONLY_NATIVE[base]
            if rule is None:
                continue
            segment = re.search(re.escape(word) + r"\s*(.*?)(?:[|;]|$)", lowered)
            args = segment.group(1).strip() if segment else ""
            if re.search(rule, args, re.I):
                continue
            return "confirm", f"Needs the user's approval: '{(base + ' ' + args).strip()}' is not a read-only use."
        return "confirm", f"Needs the user's approval: '{word}' can change the computer."
    return "safe", "Read-only command."


# --- paths -------------------------------------------------------------------------
_CREDENTIAL_NAMES = frozenset({".env", ".env.local", ".env.production", "credentials", "credentials.json",
                               "id_rsa", "id_ed25519", "id_ecdsa", "known_hosts", ".npmrc", ".pypirc", ".netrc",
                               "login data", "cookies", "local state", "wallet.dat", "secrets.json", "key3.db",
                               "key4.db", "logins.json", ".git-credentials"})
_CREDENTIAL_PARTS = frozenset({".ssh", ".aws", ".azure", ".gnupg", "keychain", "passwords", "credentials"})
EXECUTABLE_SUFFIXES = frozenset({".exe", ".bat", ".cmd", ".ps1", ".vbs", ".vbe", ".js", ".jse", ".wsf", ".msi",
                                 ".msix", ".appx", ".scr", ".com", ".lnk", ".reg", ".hta", ".pif", ".cpl", ".jar"})
READ_ACTIONS = frozenset({"list", "read", "search", "reveal"})
MASS_DELETE_LIMIT = 200

FOLDER_ALIASES: dict[str, tuple[str, ...]] = {
    "desktop": ("desktop", "دێسکتۆپ", "دیسکتۆپ", "دێسکتۆب", "سەر مێز", "سەرمێز", "ڕووی مێز", "سەر دێسکتۆپ"),
    "documents": ("documents", "document", "docs", "my documents", "دۆکیومێنت", "دۆکیومێنتەکان", "بەڵگەنامەکان",
                  "بەڵگەنامە"),
    "downloads": ("downloads", "download", "داونلۆد", "داونلۆدەکان", "داگرتنەکان", "داگرتن", "داونلود"),
    "pictures": ("pictures", "picture", "photos", "images", "وێنەکان", "وێنە", "فۆتۆکان"),
    "music": ("music", "مۆسیقا", "گۆرانییەکان"),
    "videos": ("videos", "video", "ڤیدیۆکان", "ڤیدیۆ"),
    "projects": ("projects", "sam projects", "پرۆژەکان", "پرۆژە", "پرۆژەکانی سام"),
    "workspace": ("workspace",),
    "home": ("home", "~", "ماڵەوە"),
}


def windows_path_violation(raw_path: str) -> str | None:
    """Windows path forms that bypass ordinary containment checks (v1)."""
    raw = raw_path.strip()
    if "\x00" in raw:
        return "Paths with null bytes are blocked."
    lowered = raw.lower()
    if lowered.startswith(("\\\\?\\", "\\\\.\\", "\\??\\", "//?/", "//./")):
        return "Windows device and extended path namespaces are blocked."
    drive = raw[:2] if re.match(r"^[a-zA-Z]:", raw) else ""
    remainder = raw[len(drive):]
    if ":" in remainder:
        return "NTFS alternate data streams are blocked."
    reserved = re.compile(r"(?i)^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?$")
    for part in re.split(r"[\\/]", remainder):
        if not part or part in {".", ".."}:
            continue
        if part.endswith((" ", ".")) or reserved.match(part):
            return "Reserved Windows device names and trailing-dot/space paths are blocked."
    return None


class Policy:
    """Path + command + URL rules bound to this user's folders."""

    def __init__(self, *, home: Path | None = None, sam_home: Path | None = None, data_dir: Path | None = None,
                 projects_dir: Path | None = None, workspace_dir: Path | None = None,
                 folders: dict[str, Path] | None = None) -> None:
        self.home = Path(home or Path.home()).resolve()
        self.sam_home = Path(sam_home).resolve() if sam_home else None
        self.data_dir = Path(data_dir).resolve() if data_dir else None
        self.projects_dir = Path(projects_dir or self.home / "SAM Projects").resolve()
        self.workspace_dir = Path(workspace_dir).resolve() if workspace_dir else None
        self.folders = folders if folders is not None else self._default_folders()

    @classmethod
    def from_app(cls, app: Any) -> "Policy":
        config = app.config
        return cls(sam_home=config.home, data_dir=config.data_dir,
                   projects_dir=Path(str(config.get("hands.projects_dir") or Path.home() / "SAM Projects")),
                   workspace_dir=config.workspace_dir)

    def _default_folders(self) -> dict[str, Path]:
        folders: dict[str, Path] = {}
        for key, guid in _win.KNOWN_FOLDERS.items():
            found = _win.known_folder(guid)
            folders[key] = Path(found) if found else self.home / key.capitalize()
        folders["projects"] = self.projects_dir
        folders["home"] = self.home
        if self.workspace_dir is not None:
            folders["workspace"] = self.workspace_dir
        return folders

    # -- resolution -------------------------------------------------------------------
    def folder_for(self, word: str) -> Path | None:
        key = normalize_ckb(word.strip().strip("\"'"), strip_punct=False)
        for name, aliases in FOLDER_ALIASES.items():
            if key in (normalize_ckb(a) for a in aliases) and name in self.folders:
                return self.folders[name]
        return None

    def resolve(self, raw: str) -> Path:
        """Known-folder words ("Desktop/notes.txt", "دێسکتۆپ\\x"), ~, %VARS%,
        absolute paths; anything else is relative to the user's home."""
        text = os.path.expandvars(str(raw or "").strip().strip("\"'"))
        if text.startswith("~"):
            text = str(self.home) + text[1:]
        parts = re.split(r"[\\/]+", text, maxsplit=1)
        base = self.folder_for(parts[0]) if parts and parts[0] else None
        if base is not None:
            candidate = base / parts[1] if len(parts) > 1 and parts[1] else base
        else:
            path = Path(text)
            candidate = path if path.is_absolute() else self.home / path
        return candidate.resolve(strict=False)

    # -- predicates ------------------------------------------------------------------------
    def safe_zones(self) -> list[Path]:
        return [p for p in (self.projects_dir, self.workspace_dir) if p is not None]

    def in_safe_zone(self, path: Path) -> bool:
        return any(path == zone or path.is_relative_to(zone) for zone in self.safe_zones())

    def is_sam_private(self, path: Path) -> bool:
        if self.data_dir is not None and (path == self.data_dir or path.is_relative_to(self.data_dir)):
            return True
        return self.sam_home is not None and path.parent == self.sam_home and path.name.lower().startswith(".env")

    @staticmethod
    def is_credential_path(path: Path) -> bool:
        parts = {part.lower() for part in path.parts}
        return path.name.lower() in _CREDENTIAL_NAMES or bool(parts & _CREDENTIAL_PARTS)

    def is_protected_root(self, path: Path) -> bool:
        """Drive roots, the home folder and the known folders themselves."""
        if path.parent == path or path == Path(path.anchor):
            return True
        roots = {self.home, self.projects_dir, *self.folders.values()}
        if self.workspace_dir is not None:
            roots.add(self.workspace_dir)
        return path in roots

    @staticmethod
    def is_system_path(path: Path) -> bool:
        roots = []
        for variable in ("SYSTEMROOT", "WINDIR", "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMDATA"):
            value = os.getenv(variable)
            if value:
                roots.append(Path(value).resolve(strict=False))
        return path.parent == path or any(path == root or path.is_relative_to(root) for root in roots)

    @staticmethod
    def _count_entries(path: Path, limit: int) -> int:
        count = 0
        for _, dirs, files in os.walk(path):
            count += len(dirs) + len(files)
            if count > limit:
                break
        return count

    # -- classification ------------------------------------------------------------------------
    def classify_path(self, raw: str, action: str, *, dest: str | None = None, content: str | None = None) -> Verdict:
        action = (action or "").lower()
        for value in (raw, dest):
            if value:
                violation = windows_path_violation(str(value))
                if violation:
                    return "blocked", violation
        if action not in READ_ACTIONS and action != "open" and any(ch in str(raw) for ch in "*?"):
            return "blocked", "Wildcards are not allowed for changes: one file or folder at a time."
        path = self.resolve(raw)
        if self.is_sam_private(path):
            return "blocked", "SAM's own key store and database are never exposed to tools."
        if self.is_credential_path(path) and action != "list":
            return "blocked", "This path holds keys or passwords; SAM does not read or change it."
        if action in READ_ACTIONS:
            return "safe", "Reading/listing files."
        if action == "open":
            if path.suffix.lower() in EXECUTABLE_SUFFIXES:
                return "confirm", f"Opening {path.name} runs a program."
            return "safe", "Opening a file with its default app."
        if action == "delete":
            if self.is_protected_root(path) or self.is_system_path(path):
                return "blocked", "Deleting a drive, the home folder, a main user folder or a Windows folder is blocked."
            if path.is_dir() and self._count_entries(path, MASS_DELETE_LIMIT) > MASS_DELETE_LIMIT:
                return "blocked", f"The folder has more than {MASS_DELETE_LIMIT} items: mass deletion is blocked."
            return "confirm", f"Deleting {path.name} (it goes to the Recycle Bin)."
        if action in ("write", "append"):
            if content and contains_secret(content):
                return "blocked", "The content looks like an API key or password; SAM will not write it to a file."
            if self.is_system_path(path):
                return "blocked", "Changing files inside Windows or Program Files is blocked."
            if self.in_safe_zone(path):
                return "safe", "Writing inside SAM's project/workspace folder."
            return "confirm", f"Writing {path.name} outside SAM's project folder."
        if action in ("copy", "move", "rename"):
            if not dest:
                return "blocked", f"{action} needs a destination."
            target = self.resolve(self._dest_raw(raw, dest, action))
            if self.is_sam_private(target) or self.is_credential_path(target):
                return "blocked", "The destination is a key or credential location."
            if self.is_system_path(target) or (action != "copy" and (self.is_system_path(path) or
                                                                       self.is_protected_root(path))):
                return "blocked", "Moving system folders or main user folders is blocked."
            source_ok = action == "copy" or self.in_safe_zone(path)
            if source_ok and self.in_safe_zone(target):
                return "safe", f"{action.capitalize()} inside SAM's project/workspace folder."
            return "confirm", f"{action.capitalize()} {path.name} to {target}."
        return "blocked", f"Unknown file action '{action}'."

    @staticmethod
    def _dest_raw(raw: str, dest: str, action: str) -> str:
        """For rename, a bare new name stays in the same folder."""
        if action == "rename" and not re.search(r"[\\/]", dest) and not re.match(r"^[a-zA-Z]:", dest):
            source = str(raw).rstrip("\\/")
            head = re.split(r"[\\/]", source)
            return "/".join(head[:-1] + [dest]) if len(head) > 1 else dest
        return dest

    def dest_path(self, raw: str, dest: str, action: str) -> Path:
        return self.resolve(self._dest_raw(raw, dest, action))

    @staticmethod
    def classify_powershell(command: str) -> Verdict:
        return classify_powershell(command)

    @staticmethod
    def classify_url(url: str) -> Verdict:
        parsed = urlparse(str(url or "").strip())
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return "blocked", "Only http:// and https:// web addresses can be opened."
        if parsed.username or parsed.password:
            return "blocked", "Addresses with embedded user names or passwords are blocked."
        return "safe", "Opening a web page."


_SECRET_SHAPES = re.compile(
    r"(?:sk-or-v1-[A-Za-z0-9._-]{16,}|sk-ant-[A-Za-z0-9._-]{16,}|sk-proj-[A-Za-z0-9._-]{16,}|sk-[A-Za-z0-9]{32,}"
    r"|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z._-]{30,}"
    r"|gsk_[A-Za-z0-9]{20,}|AQ\.[A-Za-z0-9_-]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)")
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(?:password|passwd|api[_-]?key|access[_-]?token|client[_-]?secret|secret[_-]?key)\s*[:=]\s*['\"]?[^\s'\"]{8,}")


def contains_secret(text: str) -> bool:
    """True when text carries a provider key or a password assignment (v1
    ``contains_embedded_secret``: models sometimes write bare keys into files)."""
    return bool(_SECRET_SHAPES.search(text or "") or _SECRET_ASSIGNMENT.search(text or ""))


__all__ = ["FOLDER_ALIASES", "Policy", "READ_ONLY_CMDLETS", "classify_powershell", "command_words",
           "contains_secret", "windows_path_violation"]
