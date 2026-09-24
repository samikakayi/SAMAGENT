"""User-facing Sorani strings of the voice package (Arabic script, Kurdish
letters ە ێ ۆ ڕ ڵ ی ک; English only for product names)."""

STT_UNCONFIGURED = "ناسینەوەی دەنگ ڕێکنەخراوە. تکایە لە ڕێکخستنەکاندا کلیلی KurdishTTS یان Gemini دابنێ."
STT_FAILED_SPOKEN = "ببورە، دەنگەکەتم بە باشی نەبیست. تکایە دووبارەی بکەرەوە."
STT_FAILED = "ناسینەوەی دەنگ سەرکەوتوو نەبوو."
TTS_UNCONFIGURED = "دەنگی قسەکردن ڕێکنەخراوە. تکایە لە ڕێکخستنەکاندا کلیلی Gemini یان KurdishTTS دابنێ."
TTS_FAILED = "نەمتوانی وەڵامەکە بە دەنگ بڵێم."
TTS_RESTING = "دەنگی قسەکردن بۆ ماوەیەکی کەم بەردەست نییە؛ وەڵامەکە لەسەر شاشەکەیە."
REPLY_FAILED = "وەڵامدانەوە سەرکەوتوو نەبوو."
MIC_FAILED = "مایکرۆفۆنەکە نەکرایەوە. تکایە ئامێری دەنگەکە بپشکنە."
LIVE_DEGRADED = "پەیوەندیی دەنگیی ڕاستەوخۆ خاو بوو، بە ڕێگای دووەم وەڵام دەدەمەوە."
LIVE_FAILED = "پەیوەندیی دەنگیی ڕاستەوخۆ دروست نەبوو، بە ڕێگای دووەم گوێت لێ دەگرم."
LIVE_NO_KEY = "بۆ دەنگی ڕاستەوخۆ کلیلی Gemini پێویستە؛ ئێستا بە ڕێگای دووەم کار دەکەم."
CONVERSATION_MISSING = "بەشی گفتوگۆ ئامادە نییە."
HOTKEY_FAILED = "کورتەڕێی {hotkey} تۆمار نەکرا؛ لەوانەیە بەرنامەیەکی تر بەکاری بهێنێت."
HOTKEY_FALLBACK = "کورتەڕێی {hotkey} لەلایەن بەرنامەیەکی ترەوە گیراوە، بۆیە {fallback} بەکاردێت. دەتوانیت لە ڕێکخستنەکاندا بیگۆڕیت."
SELFTEST_NO_KEY = "بۆ تاقیکردنەوەی دەنگ کلیلی Gemini پێویستە."

# Listening windows, the near-field gate and "only my voice" (listening.py, gate.py, voiceprint.py).
LISTEN_CLOSED = "گوێگرتن داخرا — بۆ قسەکردن کرتە بکە"
LISTEN_CLOSED_ENROLL = "گوێگرتن داخرا — بۆ قسەکردن کرتە بکە. بۆ گفتوگۆی بێ کرتە، دەنگی خۆت بناسێنە."
LISTEN_NO_SPEECH = "هیچ قسەیەکم نەبیست، گوێگرتن داخرا"
IGNORED_NOT_YOU = "ئەو دەنگە هی تۆ نەبوو، گوێم پێ نەدا"
IGNORED_NO_NAME = "بۆ فەرمان سەرەتا بڵێ «سام»"
VOICEPRINT_UNAVAILABLE = "ناسینەوەی دەنگ ئامادە نییە — ئێستا هەموو دەنگێکی نزیک وەردەگیرێت"
# Quotas (quota.py). {time} is e.g. «کاتژمێر ١٠ی بەیانی».
GEMINI_VOICE_DAILY = "سنووری ئەمڕۆی دەنگی Gemini پڕە — دوای {time}"
GEMINI_VOICE_REST = "دەنگی Gemini کەمێک پشوو دەدات، KurdishTTS قسە دەکات"
MODELS_EXHAUSTED = "سنووری مۆدێلەکان پڕە — کەمێکی تر هەوڵ بدەرەوە"
MODELS_EXHAUSTED_DAILY = "سنووری ئەمڕۆ پڕە — دوای {time}"
KURDISH_STT_MONTH = "سنووری ئەم مانگەی ناسینەوەی دەنگی KurdishTTS پڕە"
# Voice enrollment (enroll.py, the Settings voice card).
ENROLL_SPOKEN = "باشە، ڕستەکانی سەر شاشەکە بە دەنگی ئاسایی خۆت بخوێنەوە."
ENROLL_TOO_SHORT = "کورت بوو، تکایە هەموو ڕستەکە بخوێنەوە."
ENROLL_TOO_QUIET = "دەنگەکە زۆر نزم بوو، کەمێک بەرزتر یان نزیکتر لە مایکەکە بخوێنەوە."
ENROLL_NO_SPEECH = "هیچ دەنگێکم نەبیست، تکایە دووبارە بخوێنەوە."
ENROLL_INCONSISTENT = "دەنگەکان وەک یەک نەبوون، تکایە لە شوێنێکی بێدەنگدا دووبارە هەوڵ بدەرەوە."
ENROLL_REPEAT_ONE = "ئەم ڕستەیە وەک ئەوانی تر نەبوو (لەوانەیە دەنگێکی تر تێکەڵ بووبێت)، تکایە جارێکی تر بیخوێنەوە."
ENROLL_SAVED = "دەنگت ناسرا. لەمەودوا تەنها گوێ لە دەنگی تۆ دەگرم."
ENROLL_DELETED = "دەنگی تۆ سڕایەوە."
ENROLL_DOWNLOADING = "مۆدێلی ناسینەوەی دەنگ دادەبەزێت ({percent}٪)…"
ENROLL_DOWNLOAD_FAILED = "دابەزاندنی مۆدێلی ناسینەوەی دەنگ سەرکەوتوو نەبوو. پەیوەندیی ئینتەرنێت بپشکنە."
ENROLL_UNAVAILABLE = "ناسینەوەی دەنگی تۆ لەم کۆمپیوتەرەدا ئامادە نییە (sherpa-onnx دانەمەزراوە)."
ENROLL_BUSY = "سام خەریکی قسەکردنە، کەمێک چاوەڕێ بکە."
# Short Sorani sentences the user reads for the voiceprint: everyday words,
# app names and trading words SAM hears most (4-5 are enough; ~3 s each).
ENROLL_SENTENCES = (
    "سڵاو سام، من خاوەنی ئەم کۆمپیوتەرەم.",
    "تکایە ترەیدینگ ڤیو بکەرەوە و چارتی زێڕ پیشان بدە.",
    "ئەمڕۆ بازاڕ چۆنە و ترێندەکە بەرەو کوێیە؟",
    "هێڵی پشتگیری و بەرگری لەسەر چارتەکە بکێشە.",
    "ئەگەر نرخ گەیشتە ئاستی بەرگری ئاگادارم بکەرەوە.",
)

# Fixed Sorani sentences for the Live self-test (design 2.1): everyday speech,
# an app command with an English name, and a trading question.
SELFTEST_SENTENCES = (
    "سڵاو، ئەمڕۆ کەشوهەوا زۆر خۆشە.",
    "تکایە ترەیدینگ ڤیو بکەرەوە و نرخی زێڕ پیشانم بدە.",
    "کاتژمێر چەندە و بازاڕ کەی دادەخرێت؟",
)

__all__ = [name for name in dir() if name.isupper()]
