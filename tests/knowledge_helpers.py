"""Shared data for the knowledge-library tests: the synthetic trading corpus
(texts of tests/fixtures/knowledge/*.pdf, built by build_corpus.py) and the
retrieval question set with the pages that answer each question.

The scanned pages (notes p. 2-3) are images without a text layer: unit tests
use ``FakeOcr`` (returns these texts in page order); the live measurement
(acceptance/knowledge_retrieval.py) uses the real Windows OCR.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "knowledge"

SORANI_TITLE = "ستراتیژیی ڕاماڵینی شلەمەنی لە زێڕدا"
SORANI_PAGES = [
    ("بەشی یەکەم: ئاراستەی بازاڕ",
     "پێش هەموو شتێک ئاراستەی گشتی بازاڕ لەسەر چوارچێوەی کاتی چوار کاتژمێری دیاری بکە. ئەگەر نرخ لووتکەی بەرزتر "
     "و نزمترینی بەرزتر دروست بکات، ترێندەکە بەرزبوونەوەیە و تەنها بە دوای کڕیندا دەگەڕێین.\n"
     "کاتێک نرخ لە ژێر مامناوەندی جووڵاوی ٢٠٠ دایە، کڕین مەکە و چاوەڕێ بکە تا ترێندەکە ڕوون دەبێتەوە."),
    ("بەشی دووەم: ڕاماڵینی شلەمەنی و ئۆردەر بلۆک",
     "شلەمەنی لە سەرووی لووتکەی ڕۆژی پێشوو و لە ژێر نزمترینی دانیشتنی ئاسیا کۆدەبێتەوە. کاتێک نرخ نزمترینی ئاسیا "
     "دەبڕێت و بە خێرایی دەگەڕێتەوە ناو مەوداکە، ئەوە ڕاماڵینی شلەمەنییە.\n"
     "دوای ڕاماڵینەکە چاوەڕێی شکاندنی پێکهاتە بکە لەسەر چوارچێوەی کاتی پازدە خولەکی. دواتر کۆتا مۆمی دابەزین "
     "پێش جووڵە بەهێزەکە وەک ئۆردەر بلۆک دیاری بکە و چوونەژوورەوە لە ناو ئەو ئۆردەر بلۆکەدا بکە."),
    ("بەشی سێیەم: بەڕێوەبردنی مەترسی",
     "ستۆپ لۆس هەمیشە لە ژێر نزمترین خاڵی ڕاماڵینەکە دابنێ، نەک لە ناو ئۆردەر بلۆکەکە.\n"
     "لە هەر مامەڵەیەکدا زیاتر لە یەک لە سەدی سەرمایەکەت مەخە مەترسییەوە. ئامانجی یەکەم لووتکەی ڕۆژی پێشووە و "
     "ڕیسک بۆ ڕیوارد دەبێت لانیکەم یەک بۆ سێ بێت.\n"
     "دوای دوو مامەڵەی دۆڕاو لە یەک ڕۆژدا، بازرگانی ڕابگرە و ژۆرناڵەکەت بنووسە. لە کاتی هەواڵی گرنگی دۆلار، "
     "وەک NFP، نیو کاتژمێر پێش و دوای هەواڵەکە مامەڵە مەکە."),
]

ENGLISH_TITLE = "Price Action Essentials"
ENGLISH_PAGES = [
    ("Chapter 1: Support and Resistance",
     "Support is a price zone where buying pressure has repeatedly stopped a decline. Resistance is a zone where "
     "selling pressure has capped rallies. The more times a level is tested, the more significant it becomes, but "
     "each test also absorbs orders, so a level tested four or five times is more likely to break.\n"
     "When support breaks, it often becomes resistance on the retest. This change of role is called polarity."),
    ("Chapter 2: Candlestick Patterns",
     "A pin bar has a long wick and a small body; a bullish pin bar at support shows that lower prices were "
     "rejected. An engulfing candle fully covers the body of the previous candle.\n"
     "Inside bars signal consolidation before a breakout. Always read candles in context: a hammer in the middle "
     "of a range means little."),
    ("Chapter 3: Trend and Moving Averages",
     "An uptrend is a series of higher highs and higher lows. Draw a trendline connecting at least two swing lows. "
     "The 50 and 200 period moving averages act as dynamic support in a trend; a golden cross happens when the 50 "
     "crosses above the 200.\n"
     "RSI divergence, when price makes a higher high but RSI makes a lower high, warns that momentum is fading."),
    ("Chapter 4: Risk Management",
     "Risk no more than 1-2% of the account on a single trade. Place the stop loss beyond the structure that "
     "invalidates the idea, not at an arbitrary number of pips. Position size equals the account risk divided by "
     "the stop distance.\n"
     "Aim for a reward to risk ratio of at least 2:1, and keep a trading journal to review every setup. Fibonacci "
     "retracement levels of 61.8% and 78.6% mark the optimal trade entry zone during a pullback."),
]

NOTES_TITLE = "Trading notes week one"
NOTES_TEXT_PAGE = ("تێبینییەکانی بازرگانی",
                   "دانیشتنی لەندەن کاتژمێر دە بە کاتی عێراق دەست پێدەکات. باشترین کات بۆ مامەڵەی زێڕ "
                   "یەکەم کاتژمێرەکانی دانیشتنی لەندەن و نیویۆرکە.")
# Drawn into images (no text layer): notes pages 2 and 3.
SCANNED_PAGES = [
    ["Gold Scalping Checklist",
     "1. Mark the Asian session high and low.",
     "2. Wait for a liquidity sweep at the London open.",
     "3. Enter on the fair value gap after the break of structure.",
     "4. Close half at 1R and move the stop to breakeven."],
    ["تێبینیی گرنگ",
     "دوای ڕاماڵینی شلەمەنی چاوەڕێی گەڕانەوە بۆ ناو بۆشایی نرخ بکە.",
     "لە کاتی هەواڵدا مامەڵە مەکە."],
]

FILES = {"sorani": "sorani_strategy.pdf", "english": "price_action_essentials.pdf", "notes": "trading_notes.pdf"}

# (question, acceptable answers as {(document key, page)})
QUESTIONS: list[tuple[str, set[tuple[str, int]]]] = [
    ("ستۆپ لۆس لە کوێ دابنێم؟", {("sorani", 3), ("english", 4)}),
    ("ڕاماڵینی شلەمەنی چییە؟", {("sorani", 2)}),
    ("ئۆردەر بلۆک چۆن دیاری بکەم؟", {("sorani", 2)}),
    ("لە هەر مامەڵەیەکدا چەند لە سەد مەترسی بکەم؟", {("sorani", 3), ("english", 4)}),
    ("ئاراستەی بازاڕ لەسەر کام چوارچێوەی کات دیاری بکەم؟", {("sorani", 1)}),
    ("کەی بازرگانی ڕابگرم؟", {("sorani", 3)}),
    ("دانیشتنی لەندەن کەی دەست پێدەکات؟", {("notes", 1)}),
    ("پشتگیری چییە؟", {("english", 1)}),
    ("مۆمی ئینگەڵفینگ چییە؟", {("english", 2)}),
    ("دایڤێرجێنسی RSI", {("english", 3)}),
    ("ڤیبۆناچی بۆ چوونەژوورەوە", {("english", 4)}),
    ("هەواڵی NFP", {("sorani", 3)}),
    ("where should I put my stop loss", {("sorani", 3), ("english", 4)}),
    ("what is a liquidity sweep", {("sorani", 2), ("notes", 2)}),
    ("golden cross moving average", {("english", 3)}),
    ("engulfing candle", {("english", 2)}),
    ("support becomes resistance after it breaks", {("english", 1)}),
    ("position size formula", {("english", 4)}),
    ("Asian session high and low", {("notes", 2), ("sorani", 2)}),
    ("move the stop to breakeven", {("notes", 2)}),
    ("fair value gap entry", {("notes", 2), ("notes", 3)}),
    ("trendline higher lows", {("english", 3)}),
    ("how many times should a level be tested", {("english", 1)}),
    ("trading journal", {("sorani", 3), ("english", 4)}),
    ("بۆشایی نرخ دوای ڕاماڵین", {("notes", 3), ("sorani", 2)}),
    # Arabic-keyboard / speech-to-text spellings (ه for ە, و for ۆ, ی for ێ, ر for ڕ)
    ("شلهمهنی له کوی کودهبیتهوه", {("sorani", 2)}),
    ("ستوپ لوس", {("sorani", 3), ("english", 4)}),
    ("مامهڵهی دوراو", {("sorani", 3)}),
]


class FakeOcr:
    """OCR stand-in: returns the scanned pages' texts in the order asked."""

    def __init__(self, pages: list[list[str]] | None = None) -> None:
        self.pages = [list(p) for p in (pages if pages is not None else SCANNED_PAGES)]
        self.calls: list[tuple[int, int]] = []

    def __call__(self, image: Any) -> str:
        self.calls.append(tuple(image.size))
        index = min(len(self.calls) - 1, len(self.pages) - 1)
        return "\n".join(self.pages[index])


def hit_rate(library: Any, doc_keys: dict[int, str], *, k: int = 3,
             questions: list[tuple[str, set[tuple[str, int]]]] | None = None) -> dict[str, Any]:
    """Top-1 / top-k hit rates of ``library.search`` on the question set.
    ``doc_keys`` maps document ids to corpus keys (sorani/english/notes)."""
    rows = []
    top1 = topk = 0
    for question, expected in (questions or QUESTIONS):
        found = [(doc_keys.get(p["document_id"], "?"), p["page"]) for p in library.search(question, k)["passages"]]
        first = bool(found) and found[0] in expected
        anywhere = any(f in expected for f in found[:k])
        top1 += first
        topk += anywhere
        rows.append({"question": question, "found": found, "top1": first, "topk": anywhere})
    total = len(rows)
    return {"top1": top1 / total, "topk": topk / total, "n": total, "rows": rows}


def make_docx(path: Path, paragraphs: list[tuple[str, str]], *, title: str = "") -> Path:
    """A minimal real .docx: [(style, text)] with style '' | 'Heading1' |
    'Title' | 'PAGE' (a page break before the next paragraph)."""
    import zipfile

    ns = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    body = []
    for style, text in paragraphs:
        if style == "PAGE":
            body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
            continue
        ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
        safe = text.replace("&", "&amp;").replace("<", "&lt;")
        body.append(f'<w:p>{ppr}<w:r><w:t xml:space="preserve">{safe}</w:t></w:r></w:p>')
    document = f'<?xml version="1.0" encoding="UTF-8"?><w:document {ns}><w:body>{"".join(body)}</w:body></w:document>'
    styles = (f'<?xml version="1.0" encoding="UTF-8"?><w:styles {ns}>'
              '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/></w:style>'
              '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/></w:style></w:styles>')
    core = ('<?xml version="1.0" encoding="UTF-8"?><cp:coreProperties '
            'xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            f'xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>{title}</dc:title></cp:coreProperties>')
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types '
                         'xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>')
        archive.writestr("word/document.xml", document)
        archive.writestr("word/styles.xml", styles)
        archive.writestr("docProps/core.xml", core)
    return path


__all__ = ["ENGLISH_PAGES", "ENGLISH_TITLE", "FILES", "FIXTURES", "FakeOcr", "NOTES_TEXT_PAGE", "NOTES_TITLE",
           "QUESTIONS", "SCANNED_PAGES", "SORANI_PAGES", "SORANI_TITLE", "hit_rate", "make_docx"]
