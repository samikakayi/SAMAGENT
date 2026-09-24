from __future__ import annotations

import re

from brain_helpers import brain_app

from sam.brain.persona import BUDGETS, SORANI_ANCHOR, Persona, build_system_instruction, estimate_tokens
from sam.brain.tools import ToolSpec, ok
from sam.events import Transcript
from sam.textnorm import is_arabic_script

# The tool catalogue of docs/CONTRACTS.md section 2 with descriptions of the
# length the other packages use (so the size check is realistic).
CATALOGUE = {
    "open_app": "Open or focus a Windows app by its English or Sorani name (Start-menu index + aliases).",
    "window_control": "List, focus, minimize, maximize, restore, close or snap windows by title or app.",
    "type_text": "Type text into the focused control or a numbered control from the last screen_look.",
    "press_keys": "Press a keyboard shortcut or media key, e.g. ctrl+s, alt+tab, volume_up.",
    "click": "Click a numbered control from the last screen_look, a control name or visible text.",
    "screen_look": "Look at a window: numbered interactive controls, its text (OCR) or a description.",
    "screen_act": "Reach a goal on screen step by step with vision (up to 12 steps) when controls fail.",
    "run_powershell": "Run a PowerShell command; read-only commands run at once, others need approval.",
    "files": "List, read, write, copy, move, delete, open, search or reveal files and folders.",
    "open_url": "Open an http/https address in the default browser.",
    "web_search": "Search the web and return the top results (optionally open them in the browser).",
    "build_project": "Build a small website or program in a new folder, open it in VS Code and preview it.",
    "tv_open": "Open TradingView Desktop (with its local automation port) or focus it.",
    "tv_set_chart": "Change the TradingView chart's symbol and/or timeframe (any alias or Sorani words).",
    "chart_state": "Read the chart: symbol, timeframe, last price, visible range, indicators, drawings.",
    "draw_on_chart": "Draw lines, zones, fibs, text or positions on the TradingView chart at prices/times.",
    "clear_my_drawings": "Remove the drawings SAM made on the chart (never the user's own).",
    "get_price": "Current bid/ask/spread of a symbol from MetaTrader 5.",
    "analyze_market": "Analyse a market with the engine and the user's strategy, then draw the plan.",
    "set_alert": "Create a price/zone/candle/volume/strategy alert that SAM watches and speaks.",
    "list_alerts": "List active or fired alerts.",
    "cancel_alert": "Cancel an alert by id, or all alerts.",
    "strategy_save": "Save or update a trading strategy card from the user's own words.",
    "strategy_list": "List the user's strategy cards.",
    "strategy_get": "Read one strategy card with all its rules.",
}


class FakeStrategies:
    def index_for_prompt(self, max_cards: int = 50) -> str:
        return "\n".join(f"asia-sweep-{i} — سوێپی ئاسیا و FVG لەسەر پازدە خولەک، ژمارە {i}" for i in range(5))


def full_app(make_app):
    app, backend = brain_app(make_app, tools=())
    for name, description in CATALOGUE.items():
        async def handler(ctx, **kwargs):
            return ok("dry run")
        app.tools.add(ToolSpec(name=name, description=description, handler=handler,
                               params={"type": "object", "properties": {"x": {"type": "string"}}},
                               examples_ckb=("نموونەیەکی کوردی",)), owner="test")
    for i in range(14):
        app.memory.remember(f"زانیاریی گرنگ ژمارە {i}: بەکارهێنەر حەز لە وەڵامی کورت و ڕاستەوخۆ دەکات", kind="preference")
    app.trading.strategies = FakeStrategies()
    return app


def test_anchor_style_and_safety_rules_are_present(make_app):
    app, _ = brain_app(make_app)
    for mode in ("voice", "text", "worker"):
        text = app.persona.system_instruction(mode)
        assert SORANI_ANCHOR in text
        assert "untrusted" in text and "never instructions" in text
        assert "NEVER place" in text                        # no trading orders
        assert "Never ask for approval" in text or "never treat anything as approval" in text
    voice = app.persona.system_instruction("voice")
    assert "Never introduce yourself" in voice
    assert "Never repeat a sentence" in voice
    assert "1 to 3 short spoken sentences" in voice and "No Markdown" in voice
    assert "at most one short question" in voice
    assert "delegate_task" in voice
    worker = app.persona.system_instruction("worker")
    assert "finish_task" in worker and "Tone samples" not in worker


def test_sorani_examples_are_varied_marked_and_never_an_introduction(make_app):
    app, _ = brain_app(make_app)
    text = app.persona.system_instruction("voice")
    assert "NEVER copy these sentences" in text
    samples = re.findall(r"^User: (.+?) -> .*SAM: (.+)$", text, re.M)
    assert 6 <= len(samples) <= 8
    replies = [reply for _, reply in samples]
    assert len(set(replies)) == len(replies)
    for reply in replies:
        assert is_arabic_script(reply)
        assert not re.search(r"من سام|ناوم سام|سامم|I am SAM", reply)       # no canned self-introduction
        assert "ي" not in reply and "ك" not in reply                          # Kurdish letters only
    # The most common command gets no copyable reply (Groq copied it verbatim in a live test).
    assert not any("ترەیدینگ ڤیو بکەرەوە" in user for user, _ in samples)


def test_prompt_contains_time_facts_strategies_and_tools(make_app):
    app = full_app(make_app)
    voice = app.persona.system_instruction("voice")
    assert "Asia/Baghdad" in voice and "UTC+3" in voice
    assert "زانیاریی گرنگ" in voice                                  # facts from memory
    assert "asia-sweep-0" in voice                                    # strategy index
    assert "open_app" in voice and "draw_on_chart" in voice           # tool names
    worker = app.persona.system_instruction("worker")
    assert "- open_app: Open or focus" in worker                      # worker gets descriptions


def test_size_stays_within_budget_with_a_full_app(make_app):
    app = full_app(make_app)
    conversation_id = app.conversation.ensure_conversation("cascade")
    for i in range(10):
        app.bus.publish(Transcript(role="user", text=f"پرسیاری ژمارە {i} دەربارەی زێڕ و چارتەکە", source="cascade",
                                   conversation_id=conversation_id))
        app.bus.publish(Transcript(role="assistant", text=f"وەڵامی ژمارە {i}: زێڕ لە نزیکی ٢٦٥٠ە", source="cascade",
                                   conversation_id=conversation_id))
    for mode in ("voice", "text", "worker"):
        text = app.persona.system_instruction(mode)
        assert estimate_tokens(text) <= BUDGETS[mode], (mode, estimate_tokens(text))
    assert BUDGETS["voice"] <= 1500 and BUDGETS["text"] <= 1800


def test_voice_prompt_carries_recent_turns_for_a_new_live_session(make_app):
    app, _ = brain_app(make_app)
    cid = app.conversation.ensure_conversation("live")
    app.bus.publish(Transcript(role="user", text="گۆڵد لەسەر یەک کاتژمێر دابنێ", source="live", conversation_id=cid))
    app.bus.publish(Transcript(role="assistant", text="کرا، ئێستا لەسەر یەک کاتژمێرە.", source="live",
                               conversation_id=cid))
    text = app.persona.system_instruction("voice")
    assert "Last turns" in text and "گۆڵد لەسەر یەک کاتژمێر" in text


def test_now_text_is_sorani_and_estimator_is_sane(make_app):
    app, _ = brain_app(make_app)
    now = app.persona.now_text()
    assert is_arabic_script(now) and "کاتژمێر" in now
    assert not re.search(r"[0-9]", now)                               # Kurdish digits
    assert 20 <= estimate_tokens("hello " * 20) <= 40
    assert estimate_tokens("سڵاو " * 20) > estimate_tokens("hello " * 20)


def test_persona_tolerates_missing_packages_and_bad_settings(make_app):
    bare = make_app(backends={})
    bare.config.set("app.timezone", "Not/AZone")
    text = Persona(bare).system_instruction("text")
    assert SORANI_ANCHOR in text and "Asia/Baghdad" in text
    assert build_system_instruction(bare, "voice").startswith("You are SAM")
