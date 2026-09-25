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

Full authority (setting ``safety.full_authority``, 2026-09-25): a ``confirm``
verdict is split by ``command_question`` / ``Policy.path_question`` into
``routine`` (ordinary: runs without a question while the user gives SAM full
authority) and a question that stays: permanent deletion, anything touching
more than ~20 files, registry / security / account / service changes,
passwords and credentials, sending data, shutdown, force-stopping programs,
removing software.
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
# PowerShell accepts any unambiguous prefix of a parameter name: -r, -Rec,
# -Recu ... all mean -Recurse, and -Depth (-Dep, -Dept) recurses too
# (acceptance review 2026-09-24).
_RECURSE_FLAG = r"(?:-r(?:e(?:c(?:u(?:r(?:s(?:e)?)?)?)?)?)?\b|-dep(?:t(?:h)?)?\b|/s\b)"
_ROOTISH = (r"(?:\s|^|['\"])(?:[a-z]:\\?\*?|[a-z]:\\users\\[^\\\s'\"]+\\?|~[\\/]?|\$home[\\/]?|"
            r"\$env:(?:userprofile|homedrive|systemroot|windir|programfiles|onedrive)[\\/]?|"
            r"[^\s'\"]*[\\/](?:desktop|documents|downloads|pictures|onedrive)[\\/]?\*?)(?:['\"]|\s|$)")
# Every file directly inside a drive, the home folder or a main user folder
# ("Remove-Item ~\Documents\*"): no -Recurse needed to empty the folder.
_ROOT_WILDCARD = (r"(?:[a-z]:\\|~[\\/]|\$home[\\/]|\$env:\w+[\\/]|[a-z]:\\users\\[^\\\s'\"]+\\|"
                  r"[^\s'\"]*[\\/](?:desktop|documents|downloads|pictures|onedrive)[\\/])\*(?:\.\*)?(?:['\"]|\s|$)")
_MASS_DELETE = (
    re.compile(_DELETE_VERB + r".*" + _RECURSE_FLAG + r".*" + _ROOTISH, re.I),
    re.compile(_DELETE_VERB + r".*" + _ROOTISH + r".*" + _RECURSE_FLAG, re.I),
    re.compile(_DELETE_VERB + r".*" + _RECURSE_FLAG + r".*[\\/]\*(?:\s|$|['\"])", re.I),
    re.compile(r"\b(?:get-childitem|gci|dir|ls)\b.*" + _RECURSE_FLAG + r".*\|\s*" + _DELETE_VERB, re.I),
    re.compile(r"\brm\s+-rf\s+[/~]", re.I),
    re.compile(_DELETE_VERB + r".*" + _ROOT_WILDCARD, re.I),
    # .NET recursive deletes: [IO.Directory]::Delete(path, $true), DirectoryInfo.Delete($true),
    # VisualBasic FileSystem.DeleteDirectory (no PowerShell parameter to look at).
    re.compile(r"\[(?:system\.)?io\.directory\]::delete\s*\([^)]*,\s*\$true|\.delete\s*\(\s*\$true\s*\)|"
               r"filesystem\]::deletedirectory\b", re.I),
)
# A recursive listing with a delete command anywhere later in the same
# statement ("gci -r | % { Remove-Item $_ }"); checked on literal-free code.
_LISTING_THEN_DELETE = re.compile(r"\b(?:get-childitem|gci|dir|ls)\b[^;\n]*" + _RECURSE_FLAG + r"[^;\n]*\|[^;\n]*"
                                  + _DELETE_VERB, re.I)
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
                           "totaldays", "getenvironmentvariables", "adddays", "addhours", "addminutes", "addseconds",
                           "addmonths", "addyears", "addmilliseconds", "tolowerinvariant", "toupperinvariant",
                           "compareto", "equals", "lastindexof", "tochararray", "toshorttimestring",
                           "tolongtimestring", "touniversaltime", "tolocaltime"})
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
# classified "Read-only command." (repair review ps_delete_proof.py). The name
# can also come quoted ('Delete'), through -Mem/-MemberName:Delete, from a
# variable or an expression (acceptance review ps_quoted_proof.py deleted
# files with all three), so every ForEach-Object argument that is not a
# script block counts as a member call (``foreach_members``).
_DESTRUCTIVE_METHOD = re.compile(r"(?:\.|::)\s*(?:delete|moveto|copyto|remove|kill|terminate|stop|encrypt|replace|"
                                 r"move|copy)\s*\(")
_RECURSE = re.compile(r"(?:^|\s)" + _RECURSE_FLAG)
# Properties that are commonly read with the shorthand ("Get-Process | % Name").
_SAFE_MEMBERS = frozenset({"name", "fullname", "basename", "extension", "length", "directoryname", "lastwritetime",
                           "creationtime", "lastaccesstime", "mode", "id", "processname", "path", "count",
                           "displayname", "status", "starttype", "cpu", "workingset", "ws", "mainwindowtitle",
                           "version", "source", "description", "value", "key", "psiscontainer", "attributes"})
_FE_COMMAND = re.compile(r"(^|\|\|?|;|&&|\(|\{|\n)\s*(foreach-object|foreach|%)(?=[\s('\"${@]|$)")
# Member names that change or remove what they are called on; with a listing
# of a drive / home / main user folder they are a mass change even without -Recurse.
_DESTRUCTIVE_MEMBERS = frozenset({"delete", "moveto", "copyto", "remove", "kill", "terminate", "stop", "encrypt",
                                  "replace", "move", "copy", "clear", "setaccesscontrol", "decrypt", "invoke"})
_FE_BLOCK_PARAMS = ("begin", "process", "end", "parallel", "remainingscripts")
_FE_VALUE_PARAMS = ("inputobject", "throttlelimit", "timeoutseconds", "erroraction", "warningaction",
                    "informationaction", "errorvariable", "warningvariable", "informationvariable", "outvariable",
                    "outbuffer", "pipelinevariable")
_FE_VALUE_ALIASES = frozenset({"ea", "wa", "infa", "ev", "wv", "iv", "ov", "ob", "pv"})
_FE_SWITCHES = ("asjob", "usenewrunspace", "confirm", "whatif", "verbose", "debug")


def _mask_literals(code: str) -> str:
    """Same-length copy with the insides of quoted strings blanked, so
    positions still match ``code``."""
    return re.sub(r"'[^']*'|\"[^\"]*\"", lambda m: m.group(0)[0] + "x" * (len(m.group(0)) - 2) + m.group(0)[-1],
                  code)


def _scan_args(code: str, start: int) -> list[tuple[str, str]]:
    """Top-level argument tokens of the command starting at ``start``:
    (kind, text) with kind block/string/param/other; stops at the end of the
    pipeline segment."""
    tokens: list[tuple[str, str]] = []
    i, n = start, len(code)
    while i < n:
        ch = code[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "|;)}\n" or code.startswith("&&", i):
            break
        if ch in "{(":
            close = "}" if ch == "{" else ")"
            depth, j, quote = 0, i, ""
            while j < n:
                c = code[j]
                if quote:
                    quote = "" if c == quote else quote
                elif c in "'\"":
                    quote = c
                elif c == ch:
                    depth += 1
                elif c == close:
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            tokens.append(("block" if ch == "{" else "other", code[i:j + 1]))
            i = j + 1
            continue
        if ch in "'\"":
            j = code.find(ch, i + 1)
            j = n if j < 0 else j
            tokens.append(("string", code[i + 1:j]))
            i = j + 1
            continue
        j = i
        while j < n and not code[j].isspace() and code[j] not in "|;)}{(\n":
            j += 1
        word = code[i:j]
        tokens.append(("param" if word.startswith("-") and len(word) > 1 else "other", word))
        i = j
    return tokens


def _prefix_of(name: str, options: tuple[str, ...]) -> bool:
    return bool(name) and any(option.startswith(name) for option in options)


def foreach_members(lowered: str) -> list[str]:
    """Every member name (or unknown value) that ForEach-Object / % /
    foreach would call in ``lowered``: all its arguments that are not
    script blocks or harmless parameters. The ``foreach (...)`` loop
    statement is not ForEach-Object and is skipped."""
    masked = _mask_literals(lowered)
    members: list[str] = []
    for match in _FE_COMMAND.finditer(masked):
        word, after = match.group(2), match.end()
        rest = lowered[after:].lstrip()
        if word == "foreach" and not match.group(1).startswith("|") and rest.startswith("("):
            continue  # foreach ($x in $list) { ... }: the loop statement
        tokens = _scan_args(lowered, after)
        k = 0
        while k < len(tokens):
            kind, text = tokens[k]
            if kind == "block":
                k += 1
                continue
            if kind != "param":
                members.append(text.strip().strip("'\"") or "(empty)")
                k += 1
                continue
            name, colon, attached = text[1:].partition(":")
            value = [("other", attached)] if colon else tokens[k + 1:k + 2]
            step = 1 if colon else 2
            if _prefix_of(name, _FE_SWITCHES) and not colon:
                k += 1
                continue
            if _prefix_of(name, _FE_BLOCK_PARAMS) and value and value[0][0] == "block":
                k += step
                continue
            if (len(name) >= 2 and _prefix_of(name, _FE_VALUE_PARAMS)) or name in _FE_VALUE_ALIASES:
                k += step
                continue
            # -MemberName / -Mem / -M / -ArgumentList, or anything unknown.
            members.append((value[0][1].strip().strip("'\"") if value else name) or name)
            k += step
    for match in re.finditer(r"\.foreach\s*\(\s*(?!\{)([^)]*)\)", lowered):
        members.append(match.group(1).strip().strip("'\"") or "(empty)")
    return members
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
    rootish = bool(re.search(_ROOTISH, lowered, re.I))
    for method in foreach_members(lowered):
        if method in _SAFE_METHODS or method in _SAFE_MEMBERS:
            continue
        if recursive or (rootish and method in _DESTRUCTIVE_MEMBERS):
            return "blocked", ("Blocked by SAM's safety rules: a folder listing piped into a method call "
                               f"('{method[:40]}') is a mass change.")
        return "confirm", f"Needs the user's approval: the command calls '{method[:40]}' on each item."
    if recursive and (_DESTRUCTIVE_METHOD.search(code) or _LISTING_THEN_DELETE.search(code)):
        return "blocked", "Blocked by SAM's safety rules: a recursive listing piped into delete/move calls."
    if recursive and _LISTING.search(code):
        # Any other method a recursive listing calls on its items ($_.psobject.Methods['Delete'].Invoke(),
        # $_.GetFiles()...) cannot be told apart from a deletion by its name alone.
        unsafe = [m for m in re.findall(r"\.\s*([a-z_]\w*)\s*\(", code) if m not in _SAFE_METHODS]
        if unsafe:
            return "blocked", ("Blocked by SAM's safety rules: a recursive listing calls "
                               f"'.{unsafe[0][:40]}()' on its items.")
    return None


_LISTING = re.compile(r"\b(?:get-childitem|gci|dir|ls|get-item|gi)\b")
# Under full authority these "confirm" commands still ask (see the module docstring).
_SERIOUS_PS: tuple[tuple[str, str], ...] = (
    (_DELETE_VERB + r"|\bclear-(?:content|item|recyclebin)\b", "it deletes files for good (not to the Recycle Bin)"),
    (r"\breg(?:\.exe)?\s+(?:add|delete|import|restore)\b|\b(?:set|new|remove|rename|clear)-itemproperty\b|"
     r"\bhk(?:lm|cu|cr|u|cc):|\bhkey_|\[(?:microsoft\.win32\.)?registry(?:key)?\]", "it changes the registry"),
    (r"\bset-executionpolicy\b|\bicacls\b|\btakeown\b|\bset-acl\b|\bnet\s+(?:user|localgroup|share)\b|"
     r"\b(?:new|set|remove|disable|enable)-local(?:user|group)\b|\bnetsh\b|\b\w+-netfirewall\w*\b|\b\w+-mppreference\b|"
     r"\bschtasks\b|\bregister-scheduledtask\b|\bnew-service\b|\bset-service\b|\bsc(?:\.exe)?\s+(?:create|delete|config)\b",
     "it changes security, accounts, permissions, services or scheduled tasks"),
    (r"\bget-credential\b|\bcmdkey\b|\bvaultcmd\b|\bget-clipboard\b|\bconvertto-securestring\b|\bpasswords?\b|"
     r"\bpasswd\b|\bsecrets?\b|\bcredentials?\b", "it touches passwords, credentials or private data"),
    (r"\bsend-mailmessage\b|\b(?:invoke-webrequest|invoke-restmethod|iwr|irm)\b[^|;]*-(?:method\s+['\"]?(?:post|put|"
     r"patch|delete)|body|infile|form)\b|\bcurl(?:\.exe)?\b[^|;]*\s(?:-d|--data\S*|-f|--form|-t|--upload-file|"
     r"-x\s*['\"]?(?:post|put|patch))\b|\bwget\b[^|;]*--post|\bstart-bitstransfer\b[^|;]*-transfertype\s+upload",
     "it sends data to another computer"),
    (r"\bshutdown\b|\brestart-computer\b|\bstop-computer\b|\blogoff\b", "it shuts down or restarts the computer"),
    (r"\bstop-process\b|\bspps\b|\bkill\b|\btaskkill\b|\bstop-service\b|\bsc(?:\.exe)?\s+stop\b",
     "it force-stops programs (unsaved work would be lost)"),
    (r"\b(?:winget|choco|scoop)\s+(?:uninstall|remove)\b|\bmsiexec\b[^|;]*\s/x\b|\bnpm\s+(?:uninstall|publish)\b|"
     r"\buninstall-\w+\b|\bremove-appxpackage\b", "it removes software or publishes a package"),
    (r"\binvoke-cimmethod\b|\binvoke-wmimethod\b|-methodname\b", "it calls a system method"),
    (r"(?:^|[\s;|(&])(?:remove|clear|disable|uninstall|unregister|reset|format|revoke|block|dismount|suspend|"
     r"lock|protect|unprotect)-[a-z]+\b", "it removes, clears or disables something"),
)
_SERIOUS_REASONS = ("on each item", "changes a property", "built from variables")


def command_question(command: str, reason: str = "") -> str | None:
    """For a command ``classify_powershell`` marks 'confirm': why it must still
    be asked while the user gives SAM full authority, or None (an ordinary
    command -- writing, copying, moving, renaming, starting a program,
    downloading, installing, a network probe -- that then runs without a question)."""
    if any(marker in (reason or "") for marker in _SERIOUS_REASONS):
        return reason                           # per-item method calls / property changes / computed paths
    text = " ".join(str(command or "").translate(_PS_CHARS).split()).lower()
    code = _strip_literals(text)
    if _DESTRUCTIVE_METHOD.search(code):
        return "it calls a delete/move/stop method"
    for pattern, why in _SERIOUS_PS:
        if re.search(pattern, code, re.I) or re.search(pattern, text, re.I):
            return why
    return None
# powershell -Command "..." / pwsh -c '...': the quoted inner command is classified too.
_NESTED_SHELL = re.compile(r"\b(?:powershell|pwsh)(?:\.exe)?\b[^|;]*?\s-(?:c|co|com|comm|comma|comman|command)\s+"
                           r"(?:'([^']*)'|\"([^\"]*)\"|(.+))", re.I)
_RANK = {"safe": 0, "confirm": 1, "blocked": 2}


def _nested_verdict(lowered: str, depth: int) -> Verdict | None:
    """The verdict of a PowerShell command inside ``powershell -c "..."``
    when it is stricter than 'confirm' (the outer call already asks)."""
    if depth >= 2:
        return None
    for match in _NESTED_SHELL.finditer(lowered):
        inner = next((g for g in match.groups() if g), "")
        if inner.strip():
            risk, reason = classify_powershell(inner, _depth=depth + 1)
            if risk == "blocked":
                return risk, reason
    return None


# PowerShell reads the en dash, em dash and horizontal bar as a parameter
# dash, and curly quotes as quotes: "Remove-Item ~\Documents –Recurse" really
# recurses (adversarial review 2026-09-24, ps_probe.py: classified 'confirm'
# instead of 'blocked'). Every rule below is written with ASCII, so the
# command is normalised first.
_PS_CHARS = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
                           "\u2015": "-", "\u2212": "-", "\ufe58": "-", "\ufe63": "-", "\uff0d": "-",
                           "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
                           "\u201c": '"', "\u201d": '"', "\u201e": '"'})
# "$_.Attributes = 'Hidden'" inside ForEach-Object changes every listed item
# without a method call (review: hid / set read-only on a scratch tree while
# classified "Read-only command."): an assignment to a member of a variable.
_MEMBER_ASSIGN = re.compile(r"\$[\w:?]+(?:\s*\[[^\]]*\])*\s*\.\s*\w+(?:\s*\.\s*\w+|\s*\[[^\]]*\])*\s*(?:[-+*/%]|\?\?)?=(?!=)")


def classify_powershell(command: str, *, _depth: int = 0) -> Verdict:
    """(risk, reason) for one PowerShell command line."""
    text = " ".join(str(command or "").translate(_PS_CHARS).split())
    if not text:
        return "blocked", "The command is empty."
    lowered = text.lower()
    if "\x00" in text:
        return "blocked", "The command contains a null byte."
    for pattern, reason in _BLOCK_RULES:
        if re.search(pattern, lowered, re.I):
            return "blocked", f"Blocked by SAM's safety rules: {reason}."
    if any(p.search(lowered) for p in _MASS_DELETE):
        return "blocked", "Blocked by SAM's safety rules: mass deletion (a whole folder or folder tree)."
    nested = _nested_verdict(lowered, _depth)
    if nested is not None:
        return nested
    if _READ_FILE.search(lowered) and _wildcard_hits_secret(lowered):
        return "blocked", "Blocked by SAM's safety rules: the wildcard path could match a key or password file."
    member = _member_call_verdict(lowered)
    if member is not None:
        return member
    if _MEMBER_ASSIGN.search(_strip_literals(lowered)):
        if _RECURSE.search(_strip_literals(lowered)) and _LISTING.search(lowered) \
                or re.search(_ROOTISH, lowered, re.I) and _LISTING.search(lowered):
            return "blocked", ("Blocked by SAM's safety rules: a folder listing that changes a property of "
                               "every item is a mass change.")
        return "confirm", "Needs the user's approval: the command changes a property of an object (file, process...)."
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
# Full authority: a delete or move of a folder with more items than this still asks.
AUTHORITY_ITEM_LIMIT = 20

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

    def path_question(self, raw: str, action: str, *, dest: str | None = None) -> str | None:
        """For a file action ``classify_path`` marks 'confirm': why it must still
        be asked while the user gives SAM full authority, or None (ordinary:
        writing, copying, renaming, moving, opening a program, deleting one file
        or a small folder to the Recycle Bin)."""
        action = (action or "").lower()
        path = self.resolve(raw)
        if action == "open" and path.suffix.lower() == ".reg":
            return "it imports settings into the registry"
        if action == "delete":
            if not _recycle_bin_drive(path):
                return "this drive has no Recycle Bin: the delete would be permanent"
            if path.is_dir() and self._count_entries(path, AUTHORITY_ITEM_LIMIT) > AUTHORITY_ITEM_LIMIT:
                return f"the folder has more than {AUTHORITY_ITEM_LIMIT} items"
        if action == "move" and path.is_dir() and self._count_entries(path, AUTHORITY_ITEM_LIMIT) > AUTHORITY_ITEM_LIMIT:
            return f"the folder has more than {AUTHORITY_ITEM_LIMIT} items"
        return None

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


def _recycle_bin_drive(path: Path) -> bool:
    """A local fixed drive (only those have a Recycle Bin: on USB sticks and
    network shares SHFileOperation's FOF_ALLOWUNDO deletes for good)."""
    anchor = path.anchor
    if not anchor or anchor.startswith(("\\\\", "//")):
        return False
    try:
        import ctypes

        return int(ctypes.windll.kernel32.GetDriveTypeW(anchor)) == 3        # DRIVE_FIXED
    except Exception:  # noqa: BLE001 - cannot tell: treat it as permanent
        return False


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


__all__ = ["FOLDER_ALIASES", "Policy", "READ_ONLY_CMDLETS", "classify_powershell", "command_question", "command_words",
           "contains_secret", "windows_path_violation", "AUTHORITY_ITEM_LIMIT"]
