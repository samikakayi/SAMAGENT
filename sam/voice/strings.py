"""User-facing Sorani strings of the voice package (Arabic script, Kurdish
letters ە ێ ۆ ڕ ڵ ی ک; English only for product names)."""

STT_UNCONFIGURED = "ناسینەوەی دەنگ ڕێکنەخراوە. تکایە لە ڕێکخستنەکاندا کلیلی KurdishTTS یان Gemini دابنێ."
STT_FAILED_SPOKEN = "ببورە، دەنگەکەتم بە باشی نەبیست. تکایە دووبارەی بکەرەوە."
STT_FAILED = "ناسینەوەی دەنگ سەرکەوتوو نەبوو."
TTS_UNCONFIGURED = "دەنگی قسەکردن ڕێکنەخراوە. تکایە لە ڕێکخستنەکاندا کلیلی Gemini یان KurdishTTS دابنێ."
TTS_FAILED = "نەمتوانی وەڵامەکە بە دەنگ بڵێم."
REPLY_FAILED = "وەڵامدانەوە سەرکەوتوو نەبوو."
MIC_FAILED = "مایکرۆفۆنەکە نەکرایەوە. تکایە ئامێری دەنگەکە بپشکنە."
LIVE_DEGRADED = "پەیوەندیی دەنگیی ڕاستەوخۆ خاو بوو، بە ڕێگای دووەم وەڵام دەدەمەوە."
LIVE_FAILED = "پەیوەندیی دەنگیی ڕاستەوخۆ دروست نەبوو، بە ڕێگای دووەم گوێت لێ دەگرم."
LIVE_NO_KEY = "بۆ دەنگی ڕاستەوخۆ کلیلی Gemini پێویستە؛ ئێستا بە ڕێگای دووەم کار دەکەم."
CONVERSATION_MISSING = "بەشی گفتوگۆ ئامادە نییە."
HOTKEY_FAILED = "کورتەڕێی {hotkey} تۆمار نەکرا؛ لەوانەیە بەرنامەیەکی تر بەکاری بهێنێت."
HOTKEY_FALLBACK = "کورتەڕێی {hotkey} لەلایەن بەرنامەیەکی ترەوە گیراوە، بۆیە {fallback} بەکاردێت. دەتوانیت لە ڕێکخستنەکاندا بیگۆڕیت."
SELFTEST_NO_KEY = "بۆ تاقیکردنەوەی دەنگ کلیلی Gemini پێویستە."

# Fixed Sorani sentences for the Live self-test (design 2.1): everyday speech,
# an app command with an English name, and a trading question.
SELFTEST_SENTENCES = (
    "سڵاو، ئەمڕۆ کەشوهەوا زۆر خۆشە.",
    "تکایە ترەیدینگ ڤیو بکەرەوە و نرخی زێڕ پیشانم بدە.",
    "کاتژمێر چەندە و بازاڕ کەی دادەخرێت؟",
)

__all__ = [name for name in dir() if name.isupper()]
