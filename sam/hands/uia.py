"""UI Automation: a short numbered list of the controls in a window, and
click/type by number or by (fuzzy) name.

v1 dumped up to 2000 controls with no positions and no way to act on them.
SAM 2 asks UIA once for the interactive, on-screen controls with their
properties cached (``FindAllBuildCache`` with an OR condition over the
clickable control types): measured on this PC 2026-09-24 at 140-450 ms for
Chrome (44), Explorer (64), VS Code (172), ChatGPT (127) and Snipping Tool
(23), where a child-by-child walk took up to 3 s on VS Code. Chromium/
Electron windows (TradingView) build their accessibility tree only when a
client first asks, so an almost-empty first answer is retried once.

All UIA COM objects live on ONE worker thread (STA, per-monitor-v2 DPI), the
thread that created them; rectangles are physical pixels.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass
from typing import Any

from ..textnorm import normalize_ckb
from . import _win

# UIA_*PropertyId / ControlTypeId / PatternId constants (UIAutomationClient.h).
P_NAME, P_TYPE, P_RECT, P_ENABLED, P_OFFSCREEN = 30005, 30003, 30001, 30010, 30022
P_PASSWORD, P_AUTOMATION_ID, P_FOCUSABLE, P_CLASS = 30019, 30011, 30009, 30012
INVOKE, SELECTION_ITEM, VALUE, TEXT, TOGGLE, EXPAND = 10000, 10010, 10002, 10014, 10015, 10005
TREE_SCOPE_DESCENDANTS = 4

ROLES = {
    50000: "button", 50002: "checkbox", 50003: "combobox", 50004: "edit", 50005: "link", 50007: "list item",
    50011: "menu item", 50013: "radio", 50015: "slider", 50016: "spinner", 50019: "tab", 50024: "tree item",
    50029: "data item", 50030: "document", 50031: "split button", 50034: "header", 50035: "header item",
}
INVOKABLE = {50000, 50005, 50011, 50031}
SELECTABLE = {50007, 50019, 50024, 50029, 50013}
TYPABLE = {50004, 50030, 50003}


@dataclass(frozen=True)
class Control:
    number: int
    name: str
    role: str
    rect: tuple[int, int, int, int]
    enabled: bool
    automation_id: str = ""
    password: bool = False

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["rect"] = list(self.rect)
        return data

    def line(self) -> str:
        state = "" if self.enabled else " (disabled)"
        return f"{self.number}. {self.role}: {self.name or '(no name)'}{state}"

    @property
    def center(self) -> tuple[int, int]:
        return ((self.rect[0] + self.rect[2]) // 2, (self.rect[1] + self.rect[3]) // 2)


class UiaBackend:
    """Real UIA through the ``uiautomation`` package's COM client. Every
    method runs on the UIA worker thread (the caller guarantees it)."""

    def __init__(self) -> None:
        self._client: Any = None
        self._cache_request: Any = None
        self._condition: Any = None

    def _ensure(self) -> Any:
        if self._client is None:
            # Import on the worker: the package sets process DPI awareness and
            # comtypes initialises the importing thread's COM apartment.
            import uiautomation

            client = uiautomation.uiautomation._AutomationClient.instance()
            ia = client.IUIAutomation
            request = ia.CreateCacheRequest()
            for prop in (P_NAME, P_TYPE, P_RECT, P_ENABLED, P_OFFSCREEN, P_PASSWORD, P_AUTOMATION_ID, P_FOCUSABLE):
                request.AddProperty(prop)
            condition = None
            for control_type in ROLES:
                single = ia.CreatePropertyCondition(P_TYPE, control_type)
                condition = single if condition is None else ia.CreateOrCondition(condition, single)
            self._condition = ia.CreateAndCondition(condition, ia.CreatePropertyCondition(P_OFFSCREEN, False))
            self._cache_request = request
            self._client = client
        return self._client

    def _pattern(self, element: Any, pattern_id: int) -> Any:
        client = self._ensure()
        core = client.UIAutomationCore
        interface = {INVOKE: core.IUIAutomationInvokePattern, SELECTION_ITEM: core.IUIAutomationSelectionItemPattern,
                     VALUE: core.IUIAutomationValuePattern, TEXT: core.IUIAutomationTextPattern,
                     TOGGLE: core.IUIAutomationTogglePattern, EXPAND: core.IUIAutomationExpandCollapsePattern}[pattern_id]
        raw = element.GetCurrentPattern(pattern_id)
        if not raw:
            return None
        try:
            return raw.QueryInterface(interface)
        except Exception:  # noqa: BLE001
            return None

    def snapshot(self, hwnd: int, max_controls: int, window_rect: tuple[int, int, int, int] | None) -> list[tuple[dict[str, Any], Any]]:
        client = self._ensure()
        ia = client.IUIAutomation
        root = ia.ElementFromHandle(hwnd)
        found: list[tuple[dict[str, Any], Any]] = []
        for attempt in (1, 2):
            elements = root.FindAllBuildCache(TREE_SCOPE_DESCENDANTS, self._condition, self._cache_request)
            count = elements.Length if elements else 0
            found = []
            for index in range(count):
                element = elements.GetElement(index)
                rect = element.CachedBoundingRectangle
                box = (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))
                if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                    continue
                if window_rect and (box[2] < window_rect[0] or box[0] > window_rect[2] or
                                    box[3] < window_rect[1] or box[1] > window_rect[3]):
                    continue
                control_type = int(element.CachedControlType)
                if control_type == 50030 and not element.CachedIsKeyboardFocusable:
                    continue
                info = {"name": str(element.CachedName or "").strip()[:120], "type": control_type,
                        "rect": box, "enabled": bool(element.CachedIsEnabled),
                        "automation_id": str(element.CachedAutomationId or "")[:80],
                        "password": bool(element.CachedIsPassword)}
                found.append((info, element))
                if len(found) >= max_controls:
                    break
            if len(found) >= 3 or attempt == 2:
                break
            time.sleep(0.35)  # Chromium builds its tree on the first request
        return found

    def invoke(self, element: Any, control_type: int) -> str | None:
        """Activate a control without moving the mouse; None if no pattern fits."""
        if control_type in INVOKABLE:
            pattern = self._pattern(element, INVOKE)
            if pattern is not None:
                pattern.Invoke()
                return "invoke"
        if control_type in SELECTABLE:
            pattern = self._pattern(element, SELECTION_ITEM)
            if pattern is not None:
                pattern.Select()
                return "select"
        if control_type == 50002:
            pattern = self._pattern(element, TOGGLE)
            if pattern is not None:
                pattern.Toggle()
                return "toggle"
        return None

    def focus(self, element: Any) -> None:
        element.SetFocus()

    def read_text(self, element: Any) -> str | None:
        pattern = self._pattern(element, TEXT)
        if pattern is not None:
            return str(pattern.DocumentRange.GetText(20000) or "")
        pattern = self._pattern(element, VALUE)
        if pattern is not None:
            return str(pattern.CurrentValue or "")
        return None

    def focused_text(self) -> tuple[str, str | None]:
        """(name, text) of the element that has keyboard focus."""
        client = self._ensure()
        element = client.IUIAutomation.GetFocusedElement()
        if not element:
            return "", None
        return str(element.CurrentName or ""), self.read_text(element)

    def password_rects(self, hwnd: int) -> list[tuple[int, int, int, int]]:
        client = self._ensure()
        ia = client.IUIAutomation
        root = ia.ElementFromHandle(hwnd)
        elements = root.FindAll(TREE_SCOPE_DESCENDANTS, ia.CreatePropertyCondition(P_PASSWORD, True))
        rects = []
        for index in range(elements.Length if elements else 0):
            rect = elements.GetElement(index).CurrentBoundingRectangle
            rects.append((int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)))
        return rects


class Uia:
    """Async facade: numbered snapshots of one window at a time."""

    def __init__(self, backend: Any = None, *, windows: Any = None, input_runner: Any = None,
                 budget_s: float = 4.0, max_controls: int = 80) -> None:
        self._backend = backend
        self.windows = windows
        self.input_runner = input_runner   # async callable(fn_name, *args) -> runs Input methods
        self.budget_s = budget_s
        self.max_controls = max_controls
        self.worker = _win.Worker("sam-uia", _win.com_sta_initializer)
        self.last_hwnd: int | None = None
        self.last_title = ""
        self.last_at = 0.0
        self.controls: list[Control] = []
        self._refs: list[Any] = []
        self._types: list[int] = []

    @property
    def backend(self) -> Any:
        if self._backend is None:
            self._backend = UiaBackend()
        return self._backend

    async def snapshot(self, hwnd: int | None = None, *, title: str = "",
                       window_rect: tuple[int, int, int, int] | None = None) -> list[Control]:
        """Numbered interactive controls of ``hwnd`` (default: foreground)."""
        if hwnd is None and self.windows is not None:
            window = await self.windows.foreground()
            if window is None:
                return []
            hwnd, title, window_rect = window.hwnd, window.title, window.rect
        started = time.perf_counter()
        # One FindAll call is not interruptible; the budget only stops SAM
        # from waiting on a hung app (the worker finishes the call later).
        items = await asyncio.wait_for(
            self.worker.run(self.backend.snapshot, int(hwnd or 0), self.max_controls, window_rect), self.budget_s)
        self.last_hwnd, self.last_title, self.last_at = hwnd, title, time.time()
        self.controls = [Control(number=i + 1, name=info["name"], role=ROLES.get(info["type"], "control"),
                                 rect=tuple(info["rect"]), enabled=info["enabled"],
                                 automation_id=info.get("automation_id", ""), password=info.get("password", False))
                         for i, (info, _) in enumerate(items)]
        self._refs = [ref for _, ref in items]
        self._types = [info["type"] for info, _ in items]
        self.last_ms = round((time.perf_counter() - started) * 1000)
        return self.controls

    def label_of(self, target: str | int) -> str:
        """Name of a numbered control from the last snapshot (for risk checks)."""
        text = str(target).strip()
        if text.isdigit():
            number = int(text)
            if 1 <= number <= len(self.controls):
                return self.controls[number - 1].name
        return text

    def find(self, target: str | int) -> Control | None:
        """A control of the last snapshot by number or fuzzy name."""
        from rapidfuzz import fuzz

        text = str(target).strip()
        if text.isdigit():
            number = int(text)
            return self.controls[number - 1] if 1 <= number <= len(self.controls) else None
        query = normalize_ckb(text, strip_punct=True)
        if not query:
            return None
        best: tuple[float, Control] | None = None
        for control in self.controls:
            name = normalize_ckb(control.name, strip_punct=True)
            if not name:
                continue
            score = 100.0 if name == query else float(fuzz.ratio(query, name))
            if query in name.split() or (len(query) >= 4 and query in name):
                score = max(score, 88.0 - min(10.0, len(name) - len(query)) * 0.5)
            if not control.enabled:
                score -= 5
            if best is None or score > best[0]:
                best = (score, control)
        return best[1] if best and best[0] >= 75.0 else None

    async def ensure_fresh(self, hwnd: int | None = None, max_age_s: float = 20.0) -> None:
        if hwnd is None and self.windows is not None:
            fg = await self.windows.foreground()
            hwnd = fg.hwnd if fg else None
        if not self.controls or (hwnd and hwnd != self.last_hwnd) or time.time() - self.last_at > max_age_s:
            await self.snapshot(hwnd)

    async def click(self, target: str | int, *, button: str = "left", double: bool = False) -> dict[str, Any]:
        """Click a control of the last snapshot. Uses the control's own
        Invoke/Select action for a plain left click (no mouse move), else a
        real mouse click at its centre."""
        control = self.find(target)
        if control is None:
            return {"ok": False, "summary": f"No control '{target}' in the last screen snapshot."}
        index = control.number - 1
        method = None
        if button == "left" and not double:
            try:
                method = await self.worker.run(self.backend.invoke, self._refs[index], self._types[index])
            except Exception:  # noqa: BLE001 - fall back to a real click
                method = None
        if method is None:
            if self.windows is not None and self.last_hwnd:
                await self.windows.focus(self.last_hwnd)
            x, y = control.center
            await self.input_runner("click", x, y, button=button, double=double)
            method = "mouse"
        return {"ok": True, "method": method, "control": control.as_dict(),
                "summary": f"Clicked {control.role} '{control.name or control.number}' ({method})."}

    async def scroll(self, target: str | int, clicks: int) -> dict[str, Any]:
        """Scroll the mouse wheel over a control of the last snapshot
        (``clicks`` > 0 = up, < 0 = down; 1 click = 120 wheel units). The
        wheel works in every toolkit, including Electron lists whose
        ScrollPattern is missing, so no UIA pattern is used."""
        control = self.find(target)
        if control is None:
            return {"ok": False, "summary": f"No control '{target}' in the last screen snapshot."}
        clicks = max(-25, min(25, int(clicks or 0))) or -3
        if self.windows is not None and self.last_hwnd:
            await self.windows.focus(self.last_hwnd)
        x, y = control.center
        await self.input_runner("scroll", x, y, clicks)
        direction = "up" if clicks > 0 else "down"
        return {"ok": True, "method": "wheel", "clicks": clicks, "control": control.as_dict(),
                "summary": f"Scrolled {direction} {abs(clicks)} steps over {control.role} "
                           f"'{control.name or control.number}'."}

    async def type(self, target: str | int, text: str, *, press_enter: bool = False) -> dict[str, Any]:
        """Focus a control and type into it (clipboard paste), then read it back."""
        control = self.find(target)
        if control is None:
            return {"ok": False, "summary": f"No control '{target}' in the last screen snapshot."}
        index = control.number - 1
        if self.windows is not None and self.last_hwnd:
            await self.windows.focus(self.last_hwnd)
        try:
            await self.worker.run(self.backend.focus, self._refs[index])
        except Exception:  # noqa: BLE001 - click it instead
            x, y = control.center
            await self.input_runner("click", x, y)
        typed = await self.input_runner("type_text", text, press_enter=press_enter)
        readback = await self.read(control.number)
        verified = readback is not None and normalize_ckb(text)[:40] in normalize_ckb(readback)
        return {"ok": True, "verified": verified, "control": control.as_dict(), "method": typed.get("method"),
                "summary": f"Typed into {control.role} '{control.name or control.number}'"
                           + (" (checked: the text is there)." if verified else ".")}

    async def read(self, target: str | int) -> str | None:
        control = self.find(target)
        if control is None:
            return None
        try:
            return await self.worker.run(self.backend.read_text, self._refs[control.number - 1])
        except Exception:  # noqa: BLE001
            return None

    async def focused_text(self) -> tuple[str, str | None]:
        try:
            return await self.worker.run(self.backend.focused_text)
        except Exception:  # noqa: BLE001
            return "", None

    async def password_rects(self, hwnd: int) -> list[tuple[int, int, int, int]]:
        """Password fields to blank before a screenshot leaves the PC. A UIA
        failure RAISES: ``Screen.capture_ex`` then blanks the whole capture
        (it used to return [] here, so a UIA timeout sent visible password
        fields -- repair review 2026-09-24)."""
        return await self.worker.run(self.backend.password_rects, int(hwnd))

    def describe(self, limit: int = 60) -> str:
        return "\n".join(c.line() for c in self.controls[:limit])


__all__ = ["Control", "ROLES", "Uia", "UiaBackend"]
