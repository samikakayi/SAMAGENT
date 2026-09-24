"""Knowledge library: text clean-up (presentation forms, visual order, fold),
chunking and question analysis."""

from __future__ import annotations

import unicodedata

from sam.knowledge.chunker import chunk_blocks
from sam.knowledge.extract import Block
from sam.knowledge.query import analyse, match_expression, stem
from sam.knowledge.textfix import clean_text, is_visual_order, reverse_line, script_of, search_form


def forms(*names: str) -> str:
    return "".join(unicodedata.lookup(f"ARABIC LETTER {n}") for n in names)


# -- clean_text -------------------------------------------------------------------------------------------------------
def test_presentation_forms_fold_back_to_kurdish_letters() -> None:
    # «سام» and «کتێب» as a Chromium PDF text layer returns them: positional forms
    raw = forms("SEEN INITIAL FORM", "ALEF FINAL FORM", "MEEM ISOLATED FORM") + " " + \
        forms("KEHEH INITIAL FORM") + "تێ" + forms("BEH FINAL FORM")
    text, flipped = clean_text(raw)
    assert text == "سام کتێب"
    assert not flipped


def test_visual_order_lines_are_reversed_but_numbers_and_latin_stay() -> None:
    logical = "ئەوە نرخی زێڕ 2650.5 دۆلارە بۆ XAUUSD"
    # the glyphs left to right, as a visual-order text layer lists them
    visual = "XAUUSD ۆب ەرالۆد 2650.5 ڕێز یخرن ەوەئ"
    assert reverse_line(visual) == logical
    page = "\n".join([visual, "ەوەئ ەیەتسارائ یناڕۆگ ەیەناشین", "Support and resistance"])
    assert is_visual_order(page)
    text, flipped = clean_text(page)
    assert flipped
    assert text.splitlines()[0] == logical
    assert text.splitlines()[1] == "نیشانەیە گۆڕانی ئاراستەیە ئەوە"
    assert text.splitlines()[2] == "Support and resistance"


def test_logical_order_is_left_alone_and_control_characters_go() -> None:
    text, flipped = clean_text("ئەوە نیشانەی گۆڕانی ئاراستەیە\x0e\u200b و ئەمە\tپشتگیرییە")
    assert not flipped
    assert text == "ئەوە نیشانەی گۆڕانی ئاراستەیە و ئەمە پشتگیرییە"


def test_english_hyphen_breaks_join_and_blank_lines_collapse() -> None:
    text, _ = clean_text("A strong trad-\ning plan\n\n\n\nNext  paragraph")
    assert text == "A strong trading plan\n\nNext paragraph"


def test_search_form_folds_ocr_and_keyboard_spellings_together() -> None:
    # Kurdish letters vs what ar-SA OCR / an Arabic keyboard produce
    assert search_form("شلەمەنی ڕاماڵین گۆڕان") == search_form("شلهمهني رامالين كوران")
    assert search_form("ستۆپ لۆس") == search_form("ستوپ لوس")
    assert search_form("Stop-Loss, 2650!") == "stop loss 2650"


def test_script_of() -> None:
    assert script_of("ستراتیژیی ڕاماڵینی شلەمەنی") == "ckb"
    assert script_of("Support and resistance levels") == "en"
    assert script_of("Gold زێڕ price نرخ") == "mixed"
    assert script_of("") == ""


# -- chunker ------------------------------------------------------------------------------------------------------------
def test_passages_stay_on_their_page_and_overlap_only_within_a_page() -> None:
    sentence = "This sentence about support levels is part of a long page of a trading book. "
    blocks = [Block(1, "Chapter 1", sentence * 30), Block(2, "Chapter 1", "Short page two."),
              Block(3, "Chapter 2", sentence * 3)]
    chunks = chunk_blocks(blocks, target=900, overlap=150)
    assert all(c.page_start == c.page_end for c in chunks)
    page1 = [c for c in chunks if c.page_start == 1]
    assert len(page1) >= 2 and all(len(c.text) < 1200 for c in page1)
    assert page1[1].text.startswith("… ")                     # overlap within the page
    page2 = [c for c in chunks if c.page_start == 2]
    assert [c.text for c in page2] == ["Short page two."]      # small page kept alone, no overlap carried in
    page3 = [c for c in chunks if c.page_start == 3]
    assert page3[0].section == "Chapter 2" and not page3[0].text.startswith("…")


def test_pageless_text_overlaps_across_sections_and_long_sentences_split() -> None:
    long_sentence = "word " * 600
    blocks = [Block(None, "Intro", "First paragraph. " * 20), Block(None, "Rules", long_sentence)]
    chunks = chunk_blocks(blocks, target=900, overlap=150, max_chars=1400)
    assert all(c.page_start is None for c in chunks)
    assert any(c.section == "Rules" for c in chunks)
    assert all(len(c.text) <= 1400 + 160 for c in chunks)
    assert chunks[1].text.startswith("… ")


# -- query analysis ------------------------------------------------------------------------------------------------------
def test_sorani_question_maps_to_english_trading_terms() -> None:
    concepts = analyse("پشتگیرییەکان و بەرگری چین؟")
    variants = {v for c in concepts for v in c.variants}
    assert "support" in variants and "resistance" in variants
    assert "و" not in [c.label for c in concepts]


def test_english_question_maps_to_sorani_and_drops_filler() -> None:
    concepts = analyse("What does my book say about the stop loss?")
    assert [c.label for c in concepts] == ["stop loss"]
    assert search_form("ستۆپ لۆس") in concepts[0].variants


def test_stems_strip_sorani_endings_and_english_inflections() -> None:
    assert stem("ستراتیژییەکانم") == "ستراتیژ"
    assert stem("مامەڵەکان") == "مامەڵ"
    assert stem("breakouts") == "breakout"
    assert stem("trading") == "trad"
    concepts = analyse("ئۆردەربلۆکەکان")
    assert concepts and concepts[0].glossary


def test_match_expression_is_quoted_and_drops_short_terms() -> None:
    assert match_expression(("stop loss", 'a"b"c', "ab")) == '"stop loss" OR "a""b""c"'
    assert match_expression(("ab",)) is None
