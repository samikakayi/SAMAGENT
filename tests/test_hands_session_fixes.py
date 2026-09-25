"""Live session 2026-09-25: «کوڕە دەنگی بنەکەرە!» ("hey, be quiet") made the model
call system_control(mute) and SAM muted the WINDOWS master volume. The Windows
volume changes only when the user names the computer's sound; otherwise SAM
stops its own voice. Also: the TradingView STT spellings reach open_app."""

from __future__ import annotations

from typing import Any

import pytest

from sam.brain import conversation as conversation_mod, memory as memory_mod
from sam.hands.aliases import ALIASES


@pytest.fixture
def app(make_app, monkeypatch):
    from sam.hands import system

    app = make_app()
    memory_mod.register(app)
    conversation_mod.register(app)
    assert app.load_packages(["sam.hands"]) == {"sam.hands": "ok"}, app.failed
    state = {"level": 40, "muted": False}

    def fake_set(level: int | None = None, mute: bool | None = None) -> dict[str, Any]:
        if level is not None:
            state["level"] = level
        if mute is not None:
            state["muted"] = mute
        return dict(state)

    monkeypatch.setattr(system, "set_volume", fake_set)
    monkeypatch.setattr(system, "get_volume", lambda: dict(state))

    class Voice:
        stopped = 0

        async def stop_speaking(self) -> None:
            Voice.stopped += 1

    app.voice = Voice()
    app.volume_state = state
    return app


def said(app, text: str) -> None:
    cid = app.conversation.ensure_conversation("cascade")
    app.memory.add_turn(cid, "user", text, source="cascade")


@pytest.mark.parametrize("text", ["کوڕە دەنگی بنەکەرە!", "بێدەنگ بە", "دەنگ مەکە", "بەسە ئیتر", "shut up",
                                  "دەنگەکەت بکوژێنەوە"])
async def test_be_quiet_never_mutes_windows(app, text):
    said(app, text)
    result = await app.tools.dispatch("system_control", {"action": "mute"}, source="cascade")
    assert result["ok"] and result["data"]["own_voice"] is True and result["data"]["computer_volume_changed"] is False
    assert app.volume_state == {"level": 40, "muted": False} and app.voice.stopped >= 1
    zero = await app.tools.dispatch("system_control", {"action": "set_volume", "level": 0}, source="cascade")
    assert zero["data"]["own_voice"] is True and app.volume_state["level"] == 40


@pytest.mark.parametrize("text", ["دەنگی کۆمپیوتەر بکوژێنەوە", "دەنگی ویندۆز کپ بکە", "mute the volume",
                                  "دەنگەکە بکوژێنەوە"])
async def test_the_computer_volume_changes_when_the_user_names_it(app, text):
    said(app, text)
    result = await app.tools.dispatch("system_control", {"action": "mute"}, source="cascade")
    assert result["ok"] and app.volume_state["muted"] is True


async def test_raising_the_volume_and_workers_are_not_affected(app):
    said(app, "بێدەنگ بە")
    up = await app.tools.dispatch("system_control", {"action": "volume_up"}, source="cascade")
    assert app.volume_state["level"] == 50 and up["ok"]
    worker = await app.tools.dispatch("system_control", {"action": "mute"}, source="worker")
    assert worker["ok"] and app.volume_state["muted"] is True


def test_the_description_tells_the_model():
    from sam.hands.tools import system_control

    text = system_control.tool_spec.description
    assert "stop_speaking" in text and "never mute the computer" in text


@pytest.mark.parametrize("spoken", ["ترێیت ملیۆم", "ترێدینگ ڤیوو", "ترێدین ڤیو", "ترێیت ڤیو", "ترەیدین ڤیو"])
def test_tradingview_stt_spellings_are_aliases(spoken):
    from sam.textnorm import normalize_ckb

    tradingview = next(a for a in ALIASES if a.key == "tradingview")
    assert normalize_ckb(spoken) in {normalize_ckb(a) for a in tradingview.aliases}
