from __future__ import annotations

from sam.brain.speech import MAX_CHUNK_CHARS, SentenceChunker, speakable, split_sentences


def feed_all(chunker: SentenceChunker, deltas: list[str]) -> list[str]:
    out: list[str] = []
    for delta in deltas:
        out.extend(chunker.feed(delta))
    return out + chunker.flush()


def test_first_sentence_is_emitted_before_the_stream_ends():
    chunker = SentenceChunker()
    assert chunker.feed("باشە، ئێستا") == []
    assert chunker.feed(" دەیکەمەوە.") == []          # boundary needs the next character
    assert chunker.feed(" ترەیدینگ") == ["باشە، ئێستا دەیکەمەوە."]
    assert chunker.feed(" ڤیو کرایەوە") == []
    assert chunker.flush() == ["ترەیدینگ ڤیو کرایەوە"]


def test_sorani_question_mark_and_newlines_split():
    chunks = split_sentences("سڵاو، باشم سوپاس. تۆ چۆنی؟ چی هەیە؟\nئەمە دێڕێکی ترە")
    assert chunks == ["سڵاو، باشم سوپاس.", "تۆ چۆنی؟", "چی هەیە؟", "ئەمە دێڕێکی ترە"]


def test_prices_and_decimals_are_not_cut():
    chunks = feed_all(SentenceChunker(), ["پشتگیری لە 2650", ".5 دایە و بەرگری لە ٢٧٠٠", ".٥ دایە. باشە"])
    assert chunks == ["پشتگیری لە 2650.5 دایە و بەرگری لە ٢٧٠٠.٥ دایە.", "باشە"]


def test_long_sentences_are_cut_at_commas_and_never_exceed_the_limit():
    long_text = "، ".join(["ئەمە بەشێکی درێژی ڕستەکەیە بۆ تاقیکردنەوە"] * 12) + "."
    chunks = split_sentences(long_text)
    assert len(chunks) > 1
    assert all(len(c) <= MAX_CHUNK_CHARS for c in chunks)
    assert chunks[0].endswith("،")
    assert "".join(c.replace(" ", "") for c in chunks) == long_text.replace(" ", "")


def test_speech_mode_strips_markdown_links_urls_and_emojis():
    text = "## ئەنجام\n- **پشتگیری**: ٢٦٥٠ 😀\n- [ماڵپەڕ](https://example.com) و https://x.y/z بینە.\n"
    chunks = split_sentences(text, speech=True)
    joined = " ".join(chunks)
    for bad in ("#", "*", "https", "😀", "](", "- "):
        assert bad not in joined
    assert "پشتگیری" in joined and "ماڵپەڕ" in joined


def test_code_blocks_are_not_spoken_even_across_chunks():
    chunker = SentenceChunker(speech=True)
    out = feed_all(chunker, ["فایلەکە ئامادەیە.\n```py", "thon\nprint('x')\n", "```\nتەواو بوو."])
    assert out == ["فایلەکە ئامادەیە.", "تەواو بوو."]


def test_text_mode_keeps_characters_and_skips_empty_pieces():
    chunks = split_sentences("**باشە.** ... \n\n  تەواو.")
    assert chunks == ["**باشە.**", "تەواو."]
    assert speakable("") == ""
