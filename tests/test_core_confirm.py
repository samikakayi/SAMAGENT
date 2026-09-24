from __future__ import annotations

import asyncio
import threading

import pytest

from sam.brain.confirm import ConfirmBroker, classify_answer
from sam.events import ConfirmRequest, ConfirmResult, EventBus


@pytest.mark.parametrize("text", ["بەڵێ", "بەلێ.", "ئا بەڵێ", "ئەرێ بیکە", "باشە بیکە", "بێ گومان", "yes",
                                  "Yeah, go ahead!", "OK", "بەڵێ‌", "بەڵی", "بەڵێ تکایە"])
def test_yes_answers(text):
    assert classify_answer(text) is True


@pytest.mark.parametrize("text", ["نا", "نەخێر", "مەیکە", "بوەستە", "no", "Don't", "cancel it", "بەڵێ نا",
                                  "نا مەیکە", "never mind", "maybe later",
                                  # repair review 2026-09-24 (confirm_probe.py): these approved before
                                  "چاوەڕێ بکە", "ئا نازانم", "باشە دواتر", "تەواو بەسە", "دواتر بیکە",
                                  "hmm okay I guess not", "بەسە", "لێگەڕێ", "پێویست ناکات"])
def test_no_answers(text):
    assert classify_answer(text) is False


@pytest.mark.parametrize("text", ["", "ئێستا کاتژمێر چەندە",
                                  "بەڵێ بەڵام پێش ئەوە کرۆم بکەرەوە و پاشان ترەیدینگ ڤیو بکەرەوە تکایە",
                                  # a new command, a filler or a bare "ok" in Sorani is not a yes
                                  "شیکاری گۆڵد بکە", "نرخی زێڕ بکە بە دۆلار", "ئا", "باشە", "تەواو", "ئەها",
                                  # SAM's own question heard back through the mic
                                  "ئەم نامەیە بنێرم؟ «باشە سبەی دێم»"])
def test_unclear_or_long_is_not_an_answer(text):
    assert classify_answer(text) is None


async def test_voice_yes_resolves_latest_and_publishes_result():
    bus = EventBus()
    seen: list = []
    bus.subscribe((ConfirmRequest, ConfirmResult), seen.append)
    broker = ConfirmBroker(bus, timeout_s=2)
    task = asyncio.create_task(broker.confirm("کرۆم دابخەم؟", tool_name="window_control"))
    await asyncio.sleep(0.01)
    assert broker.has_pending and broker.pending()[0]["question_ckb"] == "کرۆم دابخەم؟"
    assert broker.offer_transcript("ئەمڕۆ هەوا چۆنە؟") is False  # not an answer: not consumed
    assert broker.offer_transcript("بەڵێ") is True
    assert await task is True
    result = [e for e in seen if isinstance(e, ConfirmResult)][0]
    assert result.approved and result.via == "voice"
    assert broker.offer_transcript("بەڵێ") is False  # nothing pending any more


async def test_click_from_another_thread():
    broker = ConfirmBroker(EventBus(), timeout_s=2)
    task = asyncio.create_task(broker.confirm("بیسڕمەوە؟"))
    await asyncio.sleep(0.01)
    confirm_id = broker.pending()[0]["confirm_id"]
    thread = threading.Thread(target=broker.resolve, args=(confirm_id, True, "click"))
    thread.start()
    thread.join()
    assert await task is True


async def test_timeout_is_no():
    bus = EventBus()
    seen: list = []
    bus.subscribe(ConfirmResult, seen.append)
    broker = ConfirmBroker(bus, timeout_s=0.05)
    assert await broker.confirm("?") is False
    assert seen[0].via == "timeout" and seen[0].approved is False


async def test_cancel_all_answers_no():
    broker = ConfirmBroker(None, timeout_s=5)
    tasks = [asyncio.create_task(broker.confirm(f"q{i}")) for i in range(2)]
    await asyncio.sleep(0.01)
    assert broker.cancel_all() == 2
    assert await asyncio.gather(*tasks) == [False, False]
