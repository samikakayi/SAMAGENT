"""Vision: model coordinates -> physical screen pixels (DPI), the danger
word check, and the budgeted look-act loop with a fake model."""

from __future__ import annotations

import json
from typing import Any

import pytest

from sam.hands.vision import Vision, looks_dangerous, to_screen
from tests.conftest import FakeBackend
from tests.hands_helpers import FakeShot, win


def test_normalised_points_map_to_physical_pixels() -> None:
    # A window at (100, 50) captured at 1750x1050 physical px (175 % scaling:
    # 1000x600 logical) and sent downscaled to 1440x864.
    shot = FakeShot(100, 50, 1750, 1050, 1440, 864)
    assert to_screen(shot, x=0, y=0) == (100, 50)
    assert to_screen(shot, x=500, y=500) == (100 + 875, 50 + 525)
    assert to_screen(shot, x=999, y=999) == (100 + 1748, 50 + 1049)


def test_box_2d_uses_the_centre_in_y_x_order() -> None:
    shot = FakeShot(0, 0, 2880, 1800, 1440, 900)
    # Gemini box_2d is [ymin, xmin, ymax, xmax] on 0..1000
    assert to_screen(shot, box=[100, 200, 300, 400]) == (864, 360)


def test_pixel_answers_are_scaled_from_the_sent_image() -> None:
    # A model that ignores the 0-999 instruction answers in pixels of the
    # 1440x900 image it saw; the point is scaled back to 2880x1800.
    shot = FakeShot(0, 0, 2880, 1800, 1440, 900)
    assert to_screen(shot, x=1200, y=450) == (2400, 900)


def test_points_are_clamped_inside_the_window() -> None:
    shot = FakeShot(10, 20, 500, 400, 500, 400)
    x, y = to_screen(shot, x=-50, y=5000)
    assert 10 <= x < 510 and 20 <= y < 420
    with pytest.raises(ValueError):
        to_screen(shot)


@pytest.mark.parametrize("label, dangerous", [
    ("Delete", True), ("Send message", True), ("Buy", True), ("Sell 0.01", True), ("Pay now", True),
    ("بیسڕەوە", True), ("ناردن", True), ("کڕین", True), ("پارەدان", True),
    ("Save", False), ("Information", False), ("Sender name", False), ("Border colour", False), ("OK", False),
    ("پاشەکەوت", False),
])
def test_danger_words(label: str, dangerous: bool) -> None:
    assert looks_dangerous(label) is dangerous


class FakeScreen:
    def __init__(self) -> None:
        self.captures = 0

    async def capture_ex(self, hwnd: int | None = None, **_: Any) -> FakeShot:
        self.captures += 1
        return FakeShot(100, 50, 1750, 1050, 1440, 864)


class FakeWindows:
    def __init__(self) -> None:
        self.window = win(7, "Example - Google Chrome", "chrome.exe", rect=(100, 50, 1850, 1100))
        self.focused: list[int] = []

    async def find(self, query: Any) -> Any:
        return self.window

    async def foreground(self) -> Any:
        return self.window

    async def focus(self, hwnd: int) -> bool:
        self.focused.append(hwnd)
        return True


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.pos = (0, 0)
        self.user_moves_to: tuple[int, int] | None = None

    async def __call__(self, method: str, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((method, *args, *sorted(kwargs.items())))
        if method == "click":
            self.pos = (args[0], args[1])
        if method == "cursor":
            if self.user_moves_to is not None and len([c for c in self.calls if c[0] == "cursor"]) > 1:
                return self.user_moves_to
            return self.pos
        return {"method": "paste"}


def make_vision(make_app, answers: list[str]) -> tuple[Vision, FakeRunner, FakeBackend, Any]:
    backend = FakeBackend("groq", {"qwen/qwen3.8-27b": answers})
    app = make_app(backends={"groq": backend})
    app.load_packages(["sam.hands"])
    runner = FakeRunner()
    vision = Vision(app, screen=FakeScreen(), windows=FakeWindows(), input_runner=runner)
    return vision, runner, backend, app


def plan(**fields: Any) -> str:
    return json.dumps({"thought": "t", **fields})


async def test_act_clicks_mapped_coordinates_and_finishes(make_app) -> None:
    vision, runner, backend, app = make_vision(make_app, [
        plan(action="click", x=500, y=500, target_label="Search"),
        plan(action="type", text="gold price"),
        plan(action="done", summary="Searched."),
    ])
    result = await vision.act("search gold price", 7)
    assert result["ok"] and result["steps"] == 3 and result["summary"] == "Searched."
    clicks = [c for c in runner.calls if c[0] == "click"]
    assert clicks[0][1:3] == (975, 575)  # 100 + 0.5*1750, 50 + 0.5*1050
    assert ("type_text", "gold price") in [c[:2] for c in runner.calls]
    # every step sends ONE screenshot to the vision ladder
    request = backend.calls[0][1]
    assert request.has_images() and request.json_schema is not None
    assert app.db.usage_for("hands", "screen")["requests"] == 3


async def test_risky_step_needs_the_user(make_app) -> None:
    vision, runner, _, _ = make_vision(make_app, [plan(action="click", x=10, y=10, target_label="Send")])
    asked: list[str] = []

    async def deny(question: str, detail: str) -> bool:
        asked.append(question)
        return False
    result = await vision.act("send the message", 7, confirm=deny)
    assert not result["ok"] and result["declined"]
    assert asked and "Send" in asked[0]
    assert not [c for c in runner.calls if c[0] == "click"]


async def test_model_flagged_risky_step_is_confirmed(make_app) -> None:
    vision, runner, _, _ = make_vision(make_app, [plan(action="click", x=10, y=10, target_label="OK", risky=True),
                                                  plan(action="done", summary="ok")])

    async def approve(question: str, detail: str) -> bool:
        return True
    result = await vision.act("confirm dialog", 7, confirm=approve)
    assert result["ok"] and [c for c in runner.calls if c[0] == "click"]


async def test_daily_budget_stops_the_loop(make_app) -> None:
    vision, runner, backend, app = make_vision(make_app, [plan(action="wait")])
    app.config.set("hands.vision_daily_budget", 2)
    result = await vision.act("wait forever", 7, max_steps=12)
    assert not result["ok"] and "budget" in result["summary"]
    assert len(backend.calls) == 2


async def test_step_cap_and_user_takeover(make_app) -> None:
    vision, runner, _, app = make_vision(make_app, [plan(action="click", x=100, y=100, target_label="Next")])
    app.config.set("hands.screen_act_max_steps", 3)
    result = await vision.act("keep clicking", 7, max_steps=12)
    assert not result["ok"] and result["steps"] == 3 and "3-step limit" in result["summary"]
    vision, runner, _, _ = make_vision(make_app, [plan(action="click", x=100, y=100, target_label="Next")])
    runner.user_moves_to = (5, 5)
    result = await vision.act("keep clicking", 7)
    assert not result["ok"] and "moved the mouse" in result["summary"]


async def test_dangerous_keys_are_refused_inside_the_loop(make_app) -> None:
    vision, runner, _, _ = make_vision(make_app, [plan(action="press_keys", keys="alt+f4")])
    result = await vision.act("close it", 7)
    assert not result["ok"] and "not allowed" in result["summary"]


async def test_describe_counts_against_the_budget(make_app) -> None:
    vision, _, backend, app = make_vision(make_app, ["A price chart of gold."])
    answer = await vision.describe(b"\xff\xd8jpeg", "what is this?")
    assert answer == "A price chart of gold."
    assert app.db.usage_for("hands", "screen", "vision")["requests"] == 1


class FakeUia:
    async def snapshot(self, hwnd: int, **_: Any) -> list[Any]:
        from sam.hands.uia import Control

        return [Control(1, "OK", "button", (1500, 900, 1600, 960), True),
                Control(2, "Hidden", "button", (0, 0, 10, 10), False)]


class FakeOcr:
    async def read(self, hwnd: int | None = None, region: Any = None) -> list[dict[str, Any]]:
        return [{"text": "OK", "rect": (1520, 910, 1580, 950)},           # inside the OK button: not a new mark
                {"text": "Cancel", "rect": (1300, 910, 1420, 950)}]


async def test_act_prefers_numbered_marks_over_raw_coordinates(make_app) -> None:
    """Set-of-marks: the model names a numbered box and SAM clicks the box's
    exact centre (measured: raw 0-999 answers missed the button by 270 units)."""
    vision, runner, backend, _ = make_vision(make_app, [
        plan(action="click", mark=2, x=10, y=10, target_label="Cancel"),   # x/y are ignored when a mark is given
        plan(action="done", summary="Cancelled.")])
    vision.uia, vision.ocr = FakeUia(), FakeOcr()
    marks = await vision.marks(FakeWindows().window)
    assert [(m["n"], m["label"]) for m in marks] == [(1, "button 'OK'"), (2, "text 'Cancel'")]
    result = await vision.act("press cancel", 7)
    assert result["ok"], result
    assert [c[1:3] for c in runner.calls if c[0] == "click"] == [(1360, 930)]
    prompt = backend.calls[0][1].messages[1]["content"][0]["text"]
    assert "1: button 'OK'" in prompt and "2: text 'Cancel'" in prompt and "untrusted" in prompt
