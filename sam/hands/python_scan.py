"""Static risk scan for ``run_python`` (decided by code, never by the model).

The program is parsed (``ast``) and judged before anything runs:
- blocked: SAM's key store/.env, browser/SSH/wallet credential files, DPAPI
  decryption (win32crypt / CryptUnprotectData), trading orders
  (order_send/order_check: SAM never trades), disabling Windows security;
- confirm: imports outside a pure-computation allowlist (os, sys, shutil,
  subprocess, socket, ctypes, requests, httpx, urllib, ...), pathlib writes,
  open()/to_csv()/pathlib reads and writes on a path outside the run folder
  or on a computed path, web addresses, eval/exec/compile/__import__, and
  dunder attribute tricks (``().__class__.__base__.__subclasses__()`` reaches
  os without an import);
- safe: everything else (math, statistics, numpy, pandas on ``data``).
``Verdict.reads_files`` / ``network`` mark output that must be returned as
untrusted data. Literal read paths are checked with the hands path policy
(credential files -> blocked).

A static scan of Python is a heuristic, so "safe" code also runs in the OS
sandbox of python_sandbox.py (Low integrity, no child processes). Holes an
independent adversarial pass found in the allowlist (2026-09-25) are closed
here: ``numpy.ctypeslib`` (full ctypes without importing ctypes), ``from numpy
import ctypeslib``, numpy's URL fetcher ``DataSource``, pandas readers on a
computed path (a URL built at run time), pickle loads, and sympy's
``sympify``/``parse_expr`` (they call ``eval`` on strings the AST never sees).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any

SAFE_MODULES = frozenset({
    "math", "cmath", "statistics", "random", "decimal", "fractions", "numbers", "itertools", "functools",
    "operator", "collections", "heapq", "bisect", "array", "datetime", "time", "calendar", "zoneinfo", "re",
    "string", "textwrap", "unicodedata", "json", "csv", "dataclasses", "typing", "typing_extensions", "enum",
    "abc", "copy", "pprint", "reprlib", "hashlib", "hmac", "base64", "binascii", "struct", "uuid", "secrets",
    "difflib", "contextlib", "warnings", "traceback", "io", "zlib", "locale", "graphlib",
    "numpy", "pandas", "scipy", "statsmodels", "sklearn", "matplotlib", "PIL", "pathlib", "tzdata"})
# Attributes / imported names of allowlisted packages that reach the system, the
# network or eval (kind for REASON_CKB).
DANGEROUS_ATTRS = {"ctypeslib": "system", "ctypes": "system", "f2py": "system", "windll": "system",
                   "cdll": "system", "oledll": "system", "CDLL": "system", "WinDLL": "system", "system": "system",
                   "popen": "system", "startfile": "system", "read_clipboard": "system", "to_clipboard": "system",
                   "DataSource": "network", "_datasource": "network", "sympify": "code", "parse_expr": "code",
                   "lambdify": "code", "read_pickle": "code"}
BLOCKED_MODULES = {"win32crypt": "decrypts saved passwords/keys (DPAPI)", "MetaTrader5": "trading terminal access"}
# Patterns (matched on the lower-cased source) that make a program blocked outright.
BLOCKED_MARKERS: dict[str, str] = {
    r"secrets\.json": "SAM's key store", r"(?<![\w.])\.env\b": "a .env file with keys",
    r"cryptunprotectdata": "DPAPI decryption", r"login data": "browser passwords",
    r"logins\.json": "browser passwords", r"key[34]\.db": "browser passwords", r"local state": "browser key store",
    r"wallet\.dat": "a crypto wallet", r"\bid_(?:rsa|ed25519|ecdsa)\b": "SSH keys",
    r"\.git-credentials": "stored credentials", r"\border_send\b": "a trading order",
    r"\border_check\b": "a trading order", r"set-mppreference": "Windows Defender settings",
    r"disablerealtimemonitoring": "Windows Defender settings", r"\bvssadmin\b": "shadow-copy deletion",
    r"\bbcdedit\b": "boot settings", r"sam2?\.sqlite3": "SAM's database"}
ALLOWED_DUNDERS = frozenset({"__init__", "__name__", "__doc__", "__len__", "__str__", "__repr__", "__eq__",
                             "__lt__", "__iter__", "__next__", "__enter__", "__exit__", "__add__", "__version__",
                             "__main__", "__file__"})
DANGEROUS_NAMES = {"eval": "code", "exec": "code", "compile": "code", "__import__": "code", "globals": "internals",
                   "locals": "internals", "vars": "internals", "breakpoint": "internals",
                   "__builtins__": "internals", "setattr": "internals", "delattr": "internals"}
# pathlib calls whose second path cannot be judged from the receiver (str.replace
# looks the same, so with pathlib imported these always ask).
PATHLIB_WRITES = frozenset({"rename", "replace", "symlink_to", "hardlink_to"})
READ_FUNCS = frozenset({"read_csv", "read_excel", "read_json", "read_parquet", "read_table", "loadtxt",
                        "genfromtxt", "load", "read_html", "read_xml", "read_fwf", "read_feather", "read_orc",
                        "read_stata", "read_sas", "read_spss", "read_hdf", "fromfile", "memmap", "open_memmap",
                        "imread", "loadmat"})
WRITE_FUNCS = frozenset({"to_csv", "to_excel", "to_json", "to_parquet", "to_pickle", "savetxt", "save", "savez",
                         "savefig", "to_html", "to_hdf", "to_feather", "to_stata", "to_xml", "tofile", "imsave",
                         "savemat"})
_ABS_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\[^\\]|/[A-Za-z]|~[\\/]?|%[A-Za-z_]+%)")
_URL = re.compile(r"^(?:https?|ftp|wss?)://", re.I)

# Sorani phrases for the confirmation question (never quotes the code: the card shows it).
REASON_CKB = {
    "system": "دەستکاری سیستەم، فایل یان پرۆگرامەکان دەکات",
    "network": "پەیوەندی بە ئینتەرنێتەوە دەکات",
    "code": "کۆدێکی تر لە ناو خۆیدا جێبەجێ دەکات",
    "internals": "دەچێتە ناو بەشە شاراوەکانی پایتۆن",
    "outside": "فایل لە دەرەوەی بوخچەکەی خۆی دەخوێنێتەوە یان دەنووسێت",
    "module": "بەشێکی نەناسراو بەکاردەهێنێت",
}
NETWORK_MODULES = frozenset({"socket", "requests", "httpx", "urllib", "urllib3", "http", "ftplib", "smtplib",
                             "poplib", "imaplib", "telnetlib", "websockets", "aiohttp", "ssl", "webbrowser"})
SYSTEM_MODULES = frozenset({"os", "sys", "shutil", "subprocess", "ctypes", "multiprocessing", "winreg", "_winapi",
                            "msvcrt", "signal", "importlib", "runpy", "win32api", "win32con", "win32com",
                            "pythoncom", "pywintypes", "comtypes", "uiautomation", "pyautogui", "psutil",
                            "tempfile", "glob", "sqlite3", "pickle", "shelve", "marshal", "zipfile", "tarfile",
                            "threading", "concurrent", "asyncio", "mmap", "platform", "getpass", "sounddevice",
                            "mss", "code", "codeop", "gc", "inspect", "builtins", "sam"})


@dataclass
class Verdict:
    risk: str = "safe"                                       # safe | confirm | blocked
    reasons: list[str] = field(default_factory=list)         # English, for the model/activity log
    kinds: list[str] = field(default_factory=list)           # REASON_CKB keys
    reads_files: bool = False
    network: bool = False
    syntax_error: str | None = None

    def raise_to(self, risk: str, reason: str, kind: str = "") -> None:
        order = {"safe": 0, "confirm": 1, "blocked": 2}
        if order[risk] > order[self.risk]:
            self.risk = risk
        if reason not in self.reasons:
            self.reasons.append(reason)
        if kind and kind not in self.kinds:
            self.kinds.append(kind)

    def question_ckb(self) -> str:
        phrases = [REASON_CKB[k] for k in self.kinds if k in REASON_CKB] or [REASON_CKB["system"]]
        return "ئەم کۆدە پایتۆنە " + "، ".join(phrases[:3]) + ". جێبەجێی بکەم؟"


def _literal(node: ast.AST | None) -> str | None:
    """A string constant, or the fixed start of an f-string whose first part
    is text (``f"chart_{i}.png"`` -> "chart_": judged as a relative path)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr) and node.values:
        first = node.values[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str) and first.value:
            return first.value
    return None


def _path_outside(value: str) -> bool:
    text = value.strip()
    return bool(_ABS_PATH.match(text)) or ".." in re.split(r"[\\/]", text)


def _policy_read_blocked(path: str, policy: Any) -> str | None:
    if policy is None:
        return None
    try:
        verdict, reason = policy.classify_path(path, "read")
    except Exception:  # noqa: BLE001
        return None
    return reason if verdict == "blocked" else None


def scan(code: str, *, policy: Any = None) -> Verdict:
    """Static risk of ``code`` (see module doc). ``policy`` = hands Policy for
    judging literal read paths (credential files -> blocked)."""
    verdict = Verdict()
    lowered = code.lower()
    for pattern, what in BLOCKED_MARKERS.items():
        if re.search(pattern, lowered):
            verdict.raise_to("blocked", f"touches {what}")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        verdict.syntax_error = f"SyntaxError: {exc.msg} (line {exc.lineno})"
        return verdict
    imports: set[str] = set()
    read_literals: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                verdict.raise_to("confirm", "relative import", "module")
                continue
            names = [node.module or ""]
            if (node.module or "").split(".")[0] == "PIL" and any(a.name == "ImageGrab" for a in node.names):
                verdict.raise_to("confirm", "captures the screen (PIL.ImageGrab)", "system")
        else:
            continue
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:         # «from numpy import ctypeslib»
                if alias.name in DANGEROUS_ATTRS:
                    verdict.raise_to("confirm", f"imports {alias.name}", DANGEROUS_ATTRS[alias.name])
        for name in names:
            top = name.split(".")[0]
            imports.add(top)
            for part in name.split(".")[1:]:     # «import numpy.ctypeslib»
                if part in DANGEROUS_ATTRS:
                    verdict.raise_to("confirm", f"imports {name}", DANGEROUS_ATTRS[part])
            if top in BLOCKED_MODULES:
                verdict.raise_to("blocked", f"imports {top}: {BLOCKED_MODULES[top]}")
            elif top in NETWORK_MODULES:
                verdict.network = True
                verdict.raise_to("confirm", f"imports {top} (network)", "network")
            elif top in SYSTEM_MODULES:
                verdict.raise_to("confirm", f"imports {top} (system access)", "system")
            elif top not in SAFE_MODULES:
                verdict.raise_to("confirm", f"imports {top} (not on the computation allowlist)", "module")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in DANGEROUS_NAMES:
            verdict.raise_to("confirm", f"uses {node.id}()", DANGEROUS_NAMES[node.id])
        elif isinstance(node, ast.Attribute):
            attr = node.attr
            if attr.startswith("__") and attr.endswith("__") and attr not in ALLOWED_DUNDERS:
                verdict.raise_to("confirm", f"reaches Python internals ({attr})", "internals")
            if "pathlib" in imports and attr in PATHLIB_WRITES:
                verdict.raise_to("confirm", f"changes files with pathlib ({attr})", "system")
            if attr in DANGEROUS_ATTRS:
                verdict.raise_to("confirm", f"uses {attr}", DANGEROUS_ATTRS[attr])
        if isinstance(node, ast.Call):
            _scan_call(node, verdict, read_literals, policy, imports)
    for node in ast.walk(tree):
        value = _literal(node)
        if value is None or id(node) in read_literals:
            continue
        if _URL.match(value.strip()):
            verdict.network = True
            verdict.raise_to("confirm", "uses a web address", "network")
        elif len(value) < 260 and _path_outside(value):
            verdict.raise_to("confirm", "uses a path outside its own folder", "outside")
    return verdict


_PATH_CTORS = frozenset({"Path", "PurePath", "WindowsPath", "PureWindowsPath", "PosixPath"})
_PATH_KEEP = frozenset({"joinpath", "with_name", "with_suffix", "with_stem", "resolve", "absolute", "expanduser"})
_PATH_READS = frozenset({"read_text", "read_bytes"})
_PATH_WRITES = frozenset({"write_text", "write_bytes", "touch", "mkdir", "unlink", "rmdir", "chmod"})
_OPEN_MODULES = frozenset({"io", "codecs", "gzip", "bz2", "lzma"})


def _path_base(node: ast.AST) -> ast.AST | None:
    """The literal a pathlib expression starts from, or None when unknown:
    ``Path("out") / "a.txt"`` -> "out"; ``Path.home() / "x"`` -> "~";
    ``Path.cwd()``/``Path()`` -> "." (the run folder); ``.parent`` -> unknown."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _path_base(node.left)
    if isinstance(node, ast.Call):
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
        if name in _PATH_CTORS:
            return node.args[0] if node.args else ast.Constant(".")
        if name == "home":
            return ast.Constant("~")
        if name == "cwd":
            return ast.Constant(".")
        if name in _PATH_KEEP and isinstance(func, ast.Attribute):
            return _path_base(func.value)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node
    return None


def _scan_call(node: ast.Call, verdict: Verdict, read_literals: set[int], policy: Any, imports: set[str]) -> None:
    func = node.func
    name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
    for keyword in node.keywords:
        if keyword.arg == "allow_pickle" and not (isinstance(keyword.value, ast.Constant) and keyword.value.value is False):
            verdict.raise_to("confirm", "loads pickled data (runs code)", "code")
    if name in ("getattr", "hasattr") and len(node.args) >= 2:
        attr = _literal(node.args[1])
        if attr is None or (attr.startswith("__") and attr not in ALLOWED_DUNDERS):
            verdict.raise_to("confirm", "uses getattr with a computed or hidden name", "internals")
        return
    module_open = isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and         func.value.id in _OPEN_MODULES
    if name == "open" and (isinstance(func, ast.Name) or module_open):
        target = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg == "file"), None)
        mode_node = node.args[1] if len(node.args) > 1 else next((k.value for k in node.keywords if k.arg == "mode"),
                                                                  None)
        mode = "r" if mode_node is None else _literal(mode_node)
        _judge_path(target, writing=mode is None or any(c in mode for c in "wax+"), verdict=verdict,
                    read_literals=read_literals, policy=policy, how="open()")
        return
    if isinstance(func, ast.Attribute) and "pathlib" in imports and name in (_PATH_READS | _PATH_WRITES | {"open"}):
        # pathlib: the path is the receiver, not an argument.
        if name == "open":
            mode_node = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg == "mode"), None)
            mode = "r" if mode_node is None else _literal(mode_node)
            writing = mode is None or any(c in mode for c in "wax+")
        else:
            writing = name in _PATH_WRITES
        base = _path_base(func.value)
        if base is None:
            verdict.raise_to("confirm", f"{name}() on a computed path", "outside")
            verdict.reads_files = verdict.reads_files or not writing
            return
        _judge_path(base, writing=writing, verdict=verdict, read_literals=read_literals, policy=policy, how=name)
        return
    if isinstance(func, ast.Attribute) and (name in READ_FUNCS or name in WRITE_FUNCS):
        target = node.args[0] if node.args else next(
            (k.value for k in node.keywords if k.arg in ("path", "path_or_buf", "fname", "file", "filepath_or_buffer",
                                                         "io", "excel_writer")), None)
        if target is not None and _literal(target) is None:
            # a variable/expression path cannot be judged statically; for a reader
            # it may be a URL built at run time (pandas fetches URLs itself)
            if name in WRITE_FUNCS:
                verdict.raise_to("confirm", f"writes to a computed path ({name})", "outside")
            else:
                verdict.reads_files = True
                verdict.raise_to("confirm", f"reads a computed path or address ({name})", "outside")
            return
        _judge_path(target, writing=name in WRITE_FUNCS, verdict=verdict, read_literals=read_literals,
                    policy=policy, how=name)


def _judge_path(target: ast.AST | None, *, writing: bool, verdict: Verdict, read_literals: set[int], policy: Any,
                how: str) -> None:
    path = _literal(target)
    if target is not None:
        # This call judges the literal (and an f-string's parts): the generic
        # "path outside its folder" pass must not judge it again.
        read_literals.update(id(n) for n in ast.walk(target))
    if path is None:
        if target is not None:
            verdict.raise_to("confirm", f"{how} on a computed path", "outside")
            if not writing:
                verdict.reads_files = True
        return
    if not writing:
        verdict.reads_files = True
    if _URL.match(path.strip()):
        verdict.network = True
        verdict.raise_to("confirm", "reads from or sends to the network", "network")
        return
    if not _path_outside(path):
        return                                   # inside the run folder
    if writing:
        verdict.raise_to("confirm", f"{how} writes outside its own folder", "outside")
        return
    blocked = _policy_read_blocked(path, policy)
    if blocked:
        verdict.raise_to("blocked", f"reads a protected file: {blocked}")


__all__ = ["BLOCKED_MARKERS", "REASON_CKB", "SAFE_MODULES", "Verdict", "scan"]
