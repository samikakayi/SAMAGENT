"""Speakable text, the streaming Sorani sentence splitter, CER and script checks."""

from __future__ import annotations

from sam.voice.speech_text import SentenceSplitter, cer, sorani_script_ok, speakable_text, split_for_tts


def stream(text: str, step: int = 3, **kwargs):
    splitter = SentenceSplitter(**kwargs)
    pieces = []
    for i in range(0, len(text), step):
        pieces += splitter.feed(text[i:i + step])
    return pieces, splitter.flush()


def test_first_sentence_is_released_immediately_then_rest_is_packed():
    text = "باشە، ئێستا دەیکەمەوە. نرخی زێڕ ٢٦٥٠ دۆلارە! ترێندەکە بەرەو سەرەوەیە؟ هێڵی پشتگیری دەکێشم."
    early, rest = stream(text)
    assert early == ["باشە، ئێستا دەیکەمەوە."]
    assert rest == ["نرخی زێڕ ٢٦٥٠ دۆلارە! ترێندەکە بەرەو سەرەوەیە؟ هێڵی پشتگیری دەکێشم."]


def test_sorani_question_mark_and_exclamation_end_the_first_piece():
    assert stream("چی بکەم؟ ئامادەم.")[0] == ["چی بکەم؟"]
    assert stream("تەواو! دەستم پێکرد.")[0] == ["تەواو!"]


def test_arabic_comma_cuts_a_long_first_piece_but_not_a_short_one():
    long_clause = "ئەمڕۆ بازاڕی زێڕ زۆر جووڵەی هەبوو لە دانیشتنی لەندەن، " + "بەڵام ئێستا ئارامە"
    early, _ = stream(long_clause)
    assert early == ["ئەمڕۆ بازاڕی زێڕ زۆر جووڵەی هەبوو لە دانیشتنی لەندەن،"]
    early, rest = stream("باشە، دەیکەم")
    assert early == [] and rest == ["باشە، دەیکەم"]


def test_decimal_point_and_thousands_separator_are_not_boundaries():
    early, rest = stream("نرخەکە 2650.5 دۆلارە و بەرزترین 2,675 بوو. تەواو.", step=1)
    assert early == ["نرخەکە 2650.5 دۆلارە و بەرزترین 2,675 بوو."]
    assert rest == ["تەواو."]


def test_markdown_and_emoji_are_not_spoken():
    early, rest = stream("**باشە** 😀 ئەمە `code` ـە. - یەکەم\n- دووەم https://example.com/x")
    spoken = " ".join(early + rest)
    assert "*" not in spoken and "😀" not in spoken and "http" not in spoken and "`" not in spoken
    assert "باشە" in spoken and "یەکەم" in spoken and "دووەم" in spoken


def test_later_pieces_respect_max_chars_and_take_ready_releases_early():
    sentence = "ئەمە ڕستەیەکی تاقیکردنەوەیە بۆ دابەشکردن. "
    splitter = SentenceSplitter(max_chars=120, next_min_chars=100)
    first = splitter.feed(sentence)
    assert first == [sentence.strip()]
    assert splitter.feed(sentence) == []          # below next_min_chars: waits to pack
    assert splitter.take_ready() == [sentence.strip()]  # speaker starving: release now
    pieces = splitter.feed(sentence * 6) + splitter.flush()
    assert pieces and all(len(p) <= 120 for p in pieces)


def test_split_for_tts_caps_request_size():
    pieces = split_for_tts("سڵاو چۆنی. " * 120, 480)
    assert len(pieces) >= 3 and all(len(p) <= 480 for p in pieces)
    assert split_for_tts("سڵاو.") == ["سڵاو."]
    assert split_for_tts("  ") == []


def test_speakable_text_keeps_sorani_and_numbers():
    assert speakable_text("# سەردێڕ\nنرخ: ٢٦٥٠") == "سەردێڕ\nنرخ: ٢٦٥٠"


def test_cer_normalises_arabic_letter_variants_but_counts_sorani_letters():
    assert cer("کتێبەکە", "كتێبەکە") == 0.0          # Arabic kaf vs Kurdish keheh: same letter
    assert cer("سڵاو، چۆنی؟", "سڵاو چۆنی") == 0.0    # punctuation and spaces ignored
    assert cer("سڵاو", "سلاو") == 0.25              # ڵ vs ل is a real error in Sorani
    assert cer("", "") == 0.0 and cer("", "x") == 1.0


def test_sorani_script_check():
    assert sorani_script_ok("باشە، ئێستا دەیکەمەوە")
    assert not sorani_script_ok("باشه الان باز میکنم")     # Persian: Arabic script, no Sorani letters
    assert not sorani_script_ok("Baş e, ez ê niha vekim")  # Kurmanji: Latin
    assert not sorani_script_ok("")
