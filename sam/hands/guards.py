"""Risk classifiers for the hands tools that depend on what is on screen.

Risk is decided by code (docs/CONTRACTS.md 1.7). These classifiers are plain
synchronous functions of the tool arguments (the registry contract); the few
that need live state read the foreground window through
``Windows.foreground_sync`` (~2 ms).

Gaps closed after the repair review (2026-09-24, harmless proofs in
``review2-safety-launcher``):

- ``open_app`` passed its free-form ``args`` straight to the resolved program
  with risk "safe"; the Start-menu index resolves 'cmd', 'Windows PowerShell'
  and 'PowerShell 7' to the real shells, so one call ran a shell command with
  no question and none of the run_powershell rules. Now any arguments need a
  yes, shell/interpreter arguments go through ``classify_powershell`` and
  script hosts with arguments are blocked (``AppIndex.launch`` checks the
  resolved program again).
- Typing a command + Enter into a console, Windows Terminal or the Run box was
  "safe": it is now classified like run_powershell (Run box: always a yes).
- Trading apps (MetaTrader 5, TradingView): clicks and screen_act steps whose
  label names buy/sell/order/position/close/flatten/reverse/modify/lot, and
  the order hotkeys, are BLOCKED -- design section 1: "Blocked outright:
  trading orders". The review found 'Close Position', 'Close All Positions',
  'Reverse Position', 'Flatten', 'Modify Position' and F9 / Alt+B / Shift+B
  classified "safe".
"""

from __future__ import annotations

import re
from typing import Any

from ..textnorm import normalize_ckb

APP: dict[str, Any] = {}   # set by hands.tools.register_tools: classifiers find the app here

MESSAGING_PROCESSES = frozenset({"telegram.exe", "whatsapp.exe", "whatsapp.root.exe", "discord.exe", "slack.exe",
                                 "ms-teams.exe", "teams.exe", "outlook.exe", "olk.exe", "thunderbird.exe",
                                 "signal.exe", "messenger.exe", "viber.exe", "skype.exe"})
MESSAGING_TITLES = re.compile(r"(?i)gmail|outlook|whatsapp|telegram|messenger|facebook|instagram|discord|slack|"
                              r"linkedin|twitter|\bx\.com\b|inbox|mail")
RISKY_CHORDS = {(0x12, 0x73): "alt+f4", (0x11, 0x57): "ctrl+w", (0x10, 0x2E): "shift+delete",
                (0x11, 0x10, 0x2E): "ctrl+shift+delete", (0x5B, 0x4C): "win+l"}
CONSOLE_PROCESSES = frozenset({"cmd.exe", "powershell.exe", "pwsh.exe", "windowsterminal.exe", "wt.exe",
                               "conhost.exe", "openconsole.exe", "bash.exe", "wsl.exe", "mintty.exe",
                               "python.exe", "node.exe"})
CONSOLE_CLASSES = frozenset({"ConsoleWindowClass", "CASCADIA_HOSTING_WINDOW_CLASS", "mintty"})
RUN_TITLES = frozenset({"run", "اجرا", "تشغيل", "ڕاکردن"})
TRADING_PROCESSES = frozenset({"terminal64.exe", "terminal.exe", "tradingview.exe"})
TRADING_WORDS = re.compile(
    r"(?i)\b(?:buy|sell|orders?|positions?|close|flatten|reverse|modify|lots?|trade|one[ -]?click|"
    r"market execution|pending|take profit|stop loss)\b|"
    r"کڕین|بکڕە|بیکڕە|فرۆشتن|بفرۆشە|بیفرۆشە|پۆزیشن|ئۆردەر|داخستن|دابخە|لۆت|مامەڵە|خرید|فروش|معامله|سفارش")
# Programs whose arguments are code or commands.
SHELL_NAMES = frozenset({"cmd", "cmd.exe", "command prompt", "commandprompt", "powershell", "powershell.exe",
                         "windows powershell", "pwsh", "pwsh.exe", "powershell 7", "terminal", "windows terminal",
                         "wt", "wt.exe", "bash", "wsl", "git bash", "python", "python.exe", "py", "pythonw", "node",
                         "node.exe", "reg", "reg.exe", "regedit", "schtasks", "schtasks.exe", "msiexec", "certutil",
                         "bitsadmin", "conhost", "کۆماند پرۆمپت", "پاوەرشێڵ", "تێرمیناڵ"})
SCRIPT_HOSTS = frozenset({"wscript", "wscript.exe", "cscript", "cscript.exe", "mshta", "mshta.exe", "rundll32",
                          "rundll32.exe", "regsvr32", "regsvr32.exe", "hh", "hh.exe"})


def _order_chords() -> set[tuple[int, ...]]:
    # F9 (MT5 "New Order"), Alt/Shift(+Alt)+B/S (TradingView buy/sell panel shortcuts).
    F9, ALT, SHIFT, B, S = 0x78, 0x12, 0x10, 0x42, 0x53
    return {(F9,), (ALT, B), (ALT, S), (SHIFT, B), (SHIFT, S), (ALT, SHIFT, B), (ALT, SHIFT, S),
            (SHIFT, ALT, B), (SHIFT, ALT, S)}


ORDER_CHORDS = _order_chords()


def foreground(app: Any = None) -> Any:
    hands = getattr(app or APP.get("app"), "hands", None)
    if hands is None:
        return None
    try:
        return hands.windows.foreground_sync()
    except Exception:  # noqa: BLE001
        return None


def is_console(window: Any) -> bool:
    return window is not None and (window.process.lower() in CONSOLE_PROCESSES or window.cls in CONSOLE_CLASSES)


def is_run_box(window: Any) -> bool:
    return (window is not None and window.process.lower() == "explorer.exe" and window.cls == "#32770"
            and normalize_ckb(window.title).strip() in RUN_TITLES)


def is_trading(window: Any) -> bool:
    return window is not None and window.process.lower() in TRADING_PROCESSES


def trading_label(*labels: str) -> bool:
    return any(bool(label) and bool(TRADING_WORDS.search(str(label))) for label in labels)


def _messaging(window: Any) -> bool:
    return window is not None and (window.process.lower() in MESSAGING_PROCESSES
                                   or bool(MESSAGING_TITLES.search(window.title)))


def messaging_foreground(app: Any = None) -> bool:
    hands = getattr(app or APP.get("app"), "hands", None)
    if hands is None:
        return False
    try:
        window = hands.windows.foreground_sync()
    except Exception:  # noqa: BLE001 - cannot tell: assume it might be
        return True
    return _messaging(window)


def _shell_verdict(command: str) -> tuple[str, str | None]:
    from .policy import classify_powershell

    risk, reason = classify_powershell(command)
    if risk == "blocked":
        return "blocked", reason
    if risk == "confirm":
        return "confirm", "ئەم فەرمانە لە تێرمیناڵ جێبەجێ بکەم؟ فەرمانەکە لەسەر شاشەیە."
    return "safe", None


# -- tool classifiers ------------------------------------------------------------------
def open_app_risk(args: dict[str, Any]) -> tuple[str, str | None]:
    extra = str(args.get("args") or "").strip()
    if not extra:
        return "safe", None
    name = normalize_ckb(str(args.get("name") or ""), strip_punct=False).strip()
    base = name.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    if name in SCRIPT_HOSTS or base in SCRIPT_HOSTS:
        return "blocked", "Script hosts (wscript, mshta, rundll32...) are never started with arguments by SAM."
    if name in SHELL_NAMES or base in SHELL_NAMES:
        risk, _ = _shell_verdict(extra)
        if risk == "blocked":
            return "blocked", "Blocked by SAM's safety rules: the arguments are a command run_powershell blocks."
        return "confirm", "ئەم بەرنامەیە بە فەرمانێکەوە بکەمەوە؟ فەرمانەکە لەسەر شاشەیە."
    return "confirm", "ئەم بەرنامەیە بە ئەو ڕێکخستنانەوە بکەمەوە کە لەسەر شاشەن؟"


def type_risk(args: dict[str, Any], app: Any = None) -> tuple[str, str | None]:
    text = str(args.get("text", ""))
    enter = bool(args.get("press_enter")) or "\n" in text or "\r" in text
    window = foreground(app)
    if is_console(window):
        if not enter:
            return "safe", None
        return _shell_verdict(text.replace("\r", " ").replace("\n", " "))
    if enter and is_run_box(window):
        return "confirm", "ئەمە لە پەنجەرەی Run جێبەجێ بکەم؟ دەقەکە لەسەر شاشەیە."
    if enter and is_trading(window):
        return "confirm", "لە بەرنامەی ترەیدینگدا ئینتەر دابگرم؟ دەقەکە لەسەر شاشەیە."
    messaging = _messaging(window) if window is not None else messaging_foreground(app)
    if enter and messaging:
        # The message itself is on the confirmation card; it is never read aloud
        # (SAM's own question could otherwise approve itself through the mic).
        return "confirm", "ئەم نامەیە بنێرم؟ دەقەکەی لەسەر شاشەیە."
    return "safe", None


def keys_risk(args: dict[str, Any]) -> tuple[str, str | None]:
    from .input import parse_keys

    try:
        chords = parse_keys(str(args.get("keys", "")))
    except ValueError:
        return "safe", None  # the handler reports the bad key name
    window = foreground()
    for chord in chords:
        name = RISKY_CHORDS.get(tuple(chord))
        if name:
            return "confirm", f"کلیلەکانی {name} دابگرم؟ لەوانەیە شتێک دابخات یان بسڕێتەوە."
        if is_trading(window) and tuple(chord) in ORDER_CHORDS:
            return "blocked", "Blocked: SAM never uses order hotkeys in MetaTrader or TradingView."
        if chord in ([0x0D], [0x11, 0x0D]):
            if is_console(window) or is_run_box(window):
                return "confirm", "ئینتەر دابگرم؟ فەرمانێک جێبەجێ دەکات."
            if _messaging(window) if window is not None else messaging_foreground():  # noqa: SIM108
                return "confirm", "ئینتەر دابگرم؟ لەوانەیە نامەکە بنێردرێت."
    return "safe", None


def click_risk(args: dict[str, Any]) -> tuple[str, str | None]:
    from .vision import looks_dangerous

    if str(args.get("button", "")).startswith("scroll"):
        return "safe", None  # scrolling over a "Delete" button does not press it
    app = APP.get("app")
    hands = getattr(app, "hands", None)
    target = str(args.get("target", ""))
    label = hands.uia.label_of(target) if hands is not None else target
    window = foreground(app)
    wanted = str(args.get("window") or "")
    trading = is_trading(window) or bool(re.search(r"(?i)metatrader|mt5|tradingview|ترەیدینگ|مێتاترەیدەر", wanted))
    if trading and trading_label(label, target):
        return "blocked", "Blocked: SAM never clicks buy, sell, order or position controls (analysis and alerts only)."
    if looks_dangerous(label, target):
        return "confirm", f"کلیک لەسەر «{label}» بکەم؟"
    return "safe", None


__all__ = ["open_app_risk", "type_risk", "keys_risk", "click_risk", "messaging_foreground", "is_console",
           "is_trading", "trading_label", "is_run_box", "TRADING_PROCESSES", "SHELL_NAMES", "SCRIPT_HOSTS",
           "RISKY_CHORDS", "ORDER_CHORDS", "APP"]
