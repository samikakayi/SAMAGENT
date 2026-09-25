"""UI Automation facade with a fake backend: numbering, fuzzy names,
Invoke before mouse, typing with read-back verification."""

from __future__ import annotations

from typing import Any

from sam.hands.uia import Uia
from tests.hands_helpers import fake_windows, win


class FakeUiaBackend:
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.items = items
        self.invoked: list[str] = []
        self.focused: list[str] = []
        self.texts: dict[str, str] = {}
        self.invoke_result: str | None = "invoke"

    def snapshot(self, hwnd: int, max_controls: int, window_rect: Any) -> list[tuple[dict[str, Any], Any]]:
        return [(dict(item), item["name"]) for item in self.items[:max_controls]]

    def invoke(self, ref: str, control_type: int) -> str | None:
        self.invoked.append(ref)
        return self.invoke_result

    def focus(self, ref: str) -> None:
        self.focused.append(ref)

    def read_text(self, ref: str) -> str | None:
        return self.texts.get(ref)

    def focused_text(self) -> tuple[str, str | None]:
        return "", None

    def password_rects(self, hwnd: int) -> list[Any]:
        return [(10, 10, 100, 30)]


ITEMS = [
    {"name": "File", "type": 50011, "rect": (0, 0, 40, 20), "enabled": True},
    {"name": "Save as PDF", "type": 50000, "rect": (100, 100, 200, 130), "enabled": True},
    {"name": "Search", "type": 50004, "rect": (300, 10, 600, 40), "enabled": True},
    {"name": "Delete", "type": 50000, "rect": (700, 100, 780, 130), "enabled": False},
    {"name": "Text editor", "type": 50030, "rect": (0, 50, 900, 700), "enabled": True},
]


def make(invoke_result: str | None = "invoke") -> tuple[Uia, FakeUiaBackend, list[tuple[Any, ...]]]:
    windows, _ = fake_windows([win(9, "Doc - Notepad", "Notepad.exe", rect=(0, 0, 900, 700))], foreground=9)
    backend = FakeUiaBackend(ITEMS)
    backend.invoke_result = invoke_result
    inputs: list[tuple[Any, ...]] = []

    async def runner(method: str, *args: Any, **kwargs: Any) -> Any:
        inputs.append((method, *args, *sorted(kwargs.items())))
        return {"method": "paste"}
    return Uia(backend, windows=windows, input_runner=runner), backend, inputs


async def test_snapshot_numbers_controls_with_roles() -> None:
    uia, _, _ = make()
    controls = await uia.snapshot()
    assert [c.number for c in controls] == [1, 2, 3, 4, 5]
    assert controls[1].line() == "2. button: Save as PDF"
    assert controls[3].line() == "4. button: Delete (disabled)"
    assert uia.last_hwnd == 9
    assert "3. edit: Search" in uia.describe()


async def test_find_by_number_and_fuzzy_name() -> None:
    uia, _, _ = make()
    await uia.snapshot()
    assert uia.find("2").name == "Save as PDF"
    assert uia.find("save as pdf").number == 2
    assert uia.find("serch").name == "Search"      # misspelt
    assert uia.find("pdf").name == "Save as PDF"   # a word of the name
    assert uia.find("Upload") is None
    assert uia.find("99") is None
    assert uia.label_of("4") == "Delete" and uia.label_of("anything") == "anything"


async def test_click_prefers_invoke_then_falls_back_to_the_mouse() -> None:
    uia, backend, inputs = make()
    await uia.snapshot()
    result = await uia.click("Save as PDF")
    assert result["ok"] and result["method"] == "invoke" and inputs == []
    uia, backend, inputs = make(invoke_result=None)
    await uia.snapshot()
    result = await uia.click(2)
    assert result["method"] == "mouse"
    assert inputs == [("click", 150, 115, ("button", "left"), ("double", False))]
    result = await uia.click(2, button="right")
    assert inputs[-1] == ("click", 150, 115, ("button", "right"), ("double", False))


async def test_type_focuses_pastes_and_reads_back() -> None:
    uia, backend, inputs = make()
    await uia.snapshot()
    backend.texts["Text editor"] = "سەرەتا سڵاو لە سام"
    result = await uia.type("text editor", "سڵاو لە سام")
    assert result["ok"] and result["verified"]
    assert backend.focused == ["Text editor"]
    assert inputs == [("type_text", "سڵاو لە سام", ("press_enter", False))]
    result = await uia.type("text editor", "something else")
    assert result["ok"] and not result["verified"]


async def test_unknown_control_is_an_honest_failure() -> None:
    uia, _, _ = make()
    await uia.snapshot()
    assert not (await uia.click("Upload"))["ok"]
    assert not (await uia.type("Upload", "x"))["ok"]
    assert await uia.password_rects(9) == [(10, 10, 100, 30)]


async def test_scroll_turns_the_wheel_over_the_control() -> None:
    uia, _, inputs = make()
    await uia.snapshot()
    result = await uia.scroll("text editor", -5)
    assert result["ok"] and result["clicks"] == -5 and "down" in result["summary"]
    assert inputs == [("scroll", 450, 375, -5)]                 # the centre of the editor
    result = await uia.scroll(3, 400)                           # capped at 25 steps
    assert inputs[-1] == ("scroll", 450, 25, 25)
    assert not (await uia.scroll("Upload", 3))["ok"]
