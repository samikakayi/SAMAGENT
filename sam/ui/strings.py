"""Every user-visible UI string: Sorani (Arabic script) first, English second.

Rules (docs/CONTRACTS.md "Language"): Kurdish letters ە ێ ۆ ڕ ڵ ی ک, never the
Arabic ي/ك forms; numbers shown to the user use Eastern Arabic digits
(``ckb_digits``) the way Sorani text is normally written. English is only a
fallback / secondary line (setting ``ui.show_english``).

``tr(key)`` returns Sorani, ``en(key)`` the English text; ``tr`` falls back to
English and then to the key itself so a missing entry never crashes the UI.
"""

from __future__ import annotations

import math
from typing import Any

# key: (Sorani, English)
STRINGS: dict[str, tuple[str, str]] = {
    # --- app / island -----------------------------------------------------
    "app.name": ("سام", "SAM"),
    "state.idle": ("ئامادە", "Ready"),
    "state.sleeping": ("ئامادە", "Ready"),
    "state.listening": ("گوێ دەگرم", "Listening"),
    "state.thinking": ("بیردەکەمەوە", "Thinking"),
    "state.speaking": ("قسە دەکەم", "Speaking"),
    "state.working": ("کار دەکەم", "Working"),
    "state.error": ("هەڵە", "Error"),
    "state.muted": ("بێدەنگ", "Muted"),
    "island.voice_missing": ("بەشی دەنگ هێشتا ئامادە نییە.", "The voice engine is not available yet."),
    "island.tool_failed": ("{label} سەرکەوتوو نەبوو", "{label} failed"),
    "island.click_hint": ("کرتە بکە بۆ قسەکردن · دوو کرتە بۆ پانێڵ",
                          "Click to talk · double-click for the panel"),
    # --- menus / tray -------------------------------------------------------
    "menu.open_panel": ("کردنەوەی پانێڵ", "Open panel"),
    "menu.mute": ("بێدەنگ", "Mute"),
    "menu.settings": ("ڕێکخستنەکان", "Settings"),
    "menu.quit": ("داخستن", "Quit"),
    "menu.stop_all": ("ڕاگرتنی هەموو کارەکان", "Stop everything"),
    "menu.reset_position": ("گەڕانەوە بۆ ناوەڕاستی سەرەوە", "Move back to top centre"),
    "tray.open": ("کردنەوە", "Open"),
    "tray.restart": ("دەستپێکردنەوە", "Restart"),
    "tray.alert_title": ("سام · ئاگاداری", "SAM · Alert"),
    # --- confirmation card ------------------------------------------------------
    "confirm.title": ("دڵنیاکردنەوە", "Confirmation"),
    "confirm.yes": ("بەڵێ", "Yes"),
    "confirm.no": ("نەخێر", "No"),
    "confirm.hint": ("دەتوانیت بە دەنگیش بڵێیت «بەڵێ» یان «نەخێر».", "You can also say yes or no."),
    # --- panel navigation ---------------------------------------------------------
    "tab.chat": ("گفتوگۆ", "Chat"),
    "tab.strategies": ("ستراتیژییەکان", "Strategies"),
    "tab.monitor": ("چاودێری", "Monitoring"),
    "tab.activity": ("چالاکی", "Activity"),
    "tab.settings": ("ڕێکخستنەکان", "Settings"),
    "tab.chat.sub": ("بنووسە یان قسە بکە — هەمان مێشک وەڵام دەداتەوە.", "Type or talk — the same brain answers."),
    "tab.strategies.sub": ("ستراتیژی و تیۆرییەکانی ترەیدینگی تۆ.", "Your trading strategies and theories."),
    "tab.monitor.sub": ("ئاگادارکەرەوەی نرخ، ناوچە، مۆم و قەبارە.", "Price, zone, candle and volume alerts."),
    "tab.activity.sub": ("هەموو ئەو کارانەی سام کردوونی و خێرایی هەر قۆناغێک.",
                         "Everything SAM did and how long each stage took."),
    "tab.settings.sub": ("کلیلەکان، دەنگ، ترەیدینگ و پاراستنی نهێنی.", "Keys, voice, trading and privacy."),
    # --- chat -------------------------------------------------------------------------
    "chat.placeholder": ("پەیامێک بنووسە…", "Write a message…"),
    "chat.input_hint": ("Enter بۆ ناردن · Shift+Enter بۆ دێڕی نوێ", "Enter to send · Shift+Enter for a new line"),
    "chat.send": ("ناردن", "Send"),
    "chat.mic": ("قسەکردن", "Talk"),
    "chat.empty_title": ("چۆن یارمەتیت بدەم؟", "How can I help?"),
    "chat.empty_sub": ("بە کوردی بنووسە یان کرتە لە سام بکە و قسە بکە.",
                       "Write in Kurdish, or click SAM and talk."),
    "chat.you": ("تۆ", "You"),
    "chat.sam": ("سام", "SAM"),
    "chat.worker": ("ئەرکی چەند هەنگاوی", "Multi-step task"),
    "chat.voice": ("دەنگ", "voice"),
    "chat.typed": ("نووسین", "typed"),
    "chat.unavailable": ("بەشی گفتوگۆ هێشتا ئامادە نییە.", "The conversation module is not available yet."),
    "chat.failed": ("ناردن سەرکەوتوو نەبوو.", "Sending failed."),
    "chat.suggest.1": ("ترەیدینگ ڤیو بکەرەوە", "Open TradingView"),
    "chat.suggest.2": ("گۆڵد لەسەر ١٥ خولەک پیشان بدە", "Show gold on 15 minutes"),
    "chat.suggest.3": ("هێڵی پشتگیری و بەرگری بکێشە", "Draw support and resistance"),
    "chat.suggest.4": ("زێڕ شی بکەرەوە بە ستراتیژییەکەم", "Analyse gold with my strategy"),
    # --- strategies ---------------------------------------------------------------------
    "strat.add": ("ستراتیژیی نوێ", "New strategy"),
    "strat.add_hint": ("ستراتیژییەکەت لێرە بلکێنە یان بنووسە (کوردی یان ئینگلیزی). سام دەیکاتە کارتێکی ڕێکخراو.",
                       "Paste or write your strategy (Kurdish or English). SAM turns it into a structured card."),
    "strat.save": ("پاشەکەوتکردن", "Save"),
    "strat.cancel": ("هەڵوەشاندنەوە", "Cancel"),
    "strat.saving": ("سام ستراتیژییەکە ڕێک دەخات…", "SAM is structuring the strategy…"),
    "strat.save_failed": ("پاشەکەوتکردنی ستراتیژییەکە سەرکەوتوو نەبوو. هیچ شتێک نەگۆڕدرا.",
                          "Saving the strategy failed. Nothing was changed."),
    "strat.rules_count": ("{n} یاسا", "{n} rules"),
    "strat.activate": ("چالاککردن", "Activate"),
    "strat.archive": ("ئەرشیفکردن", "Archive"),
    "strat.draft": ("گەڕاندنەوە بۆ ڕەشنووس", "Back to draft"),
    "strat.filter.all": ("هەموو", "All"),
    "strat.status.active": ("چالاک", "Active"),
    "strat.status.draft": ("ڕەشنووس", "Draft"),
    "strat.status.archived": ("ئەرشیفکراو", "Archived"),
    "strat.rules": ("یاساکان", "Rules"),
    "strat.timeframes": ("کاتەکان", "Timeframes"),
    "strat.tf.bias": ("ئاراستە", "Bias"),
    "strat.tf.setup": ("ئامادەکاری", "Setup"),
    "strat.tf.entry": ("چوونەژوورەوە", "Entry"),
    "strat.markets": ("بازاڕەکان", "Markets"),
    "strat.sessions": ("کاتی بازاڕ", "Sessions"),
    "strat.risk": ("مەترسی", "Risk"),
    "strat.risk.max_risk_pct": ("زۆرترین مەترسی بۆ هەر مامەڵەیەک", "Max risk per trade"),
    "strat.risk.max_losses_per_day": ("زۆرترین دۆڕان لە ڕۆژێکدا", "Max losses per day"),
    "strat.kind.bias": ("ئاراستە", "bias"),
    "strat.kind.setup": ("ئامادەکاری", "setup"),
    "strat.kind.trigger": ("هاندەر", "trigger"),
    "strat.kind.entry": ("چوونەژوورەوە", "entry"),
    "strat.kind.stop": ("ستۆپ", "stop"),
    "strat.kind.target": ("ئامانج", "target"),
    "strat.kind.risk": ("مەترسی", "risk"),
    "strat.kind.filter": ("فلتەر", "filter"),
    "strat.kind.manage": ("بەڕێوەبردن", "manage"),
    "strat.source": ("دەقی ئەسڵی", "Original text"),
    "strat.versions": ("وەشانەکان", "Versions"),
    "strat.version": ("وەشانی", "version"),
    "strat.empty": ("هێشتا هیچ ستراتیژییەک نییە. یەکێک زیاد بکە یان بە دەنگ بیڵێ.",
                    "No strategies yet. Add one here or say it."),
    "strat.select": ("ستراتیژییەک هەڵبژێرە بۆ بینینی وردەکاری.", "Select a strategy to see its details."),
    "strat.unavailable": ("بەشی ستراتیژی هێشتا ئامادە نییە.", "The strategy store is not available yet."),
    "strat.predicate": ("پشکنینی خۆکار", "automatic check"),
    "strat.judged": ("بە چاو / مێشک هەڵدەسەنگێنرێت", "judged by vision / the model"),
    # --- monitor ---------------------------------------------------------------------------
    "mon.active": ("ئاگادارکەرەوە چالاکەکان", "Active alerts"),
    "mon.history": ("مێژوو", "History"),
    "mon.cancel": ("هەڵوەشاندنەوە", "Cancel"),
    "mon.cancel_all": ("هەڵوەشاندنەوەی هەمووی", "Cancel all"),
    "mon.empty": ("هیچ ئاگادارکەرەوەیەکی چالاک نییە. بڵێ: «ئەگەر زێڕ گەیشتە ئەو نرخە ئاگادارم بکەرەوە».",
                  "No active alerts. Say: \"alert me if gold reaches that price\"."),
    "mon.history_empty": ("هێشتا هیچ ئاگادارییەک نەبووە.", "No alerts have fired yet."),
    "mon.kind.price_cross": ("تێپەڕینی نرخ", "Price cross"),
    "mon.kind.zone_touch": ("گەیشتن بە ناوچە", "Zone touch"),
    "mon.kind.candle_close": ("داخستنی مۆم", "Candle close"),
    "mon.kind.volume_spike": ("بەرزبوونەوەی قەبارە", "Volume spike"),
    "mon.kind.strategy_state": ("دۆخی ستراتیژی", "Strategy state"),
    "mon.status.active": ("چالاک", "Active"),
    "mon.status.fired": ("ڕوویدا", "Fired"),
    "mon.status.cancelled": ("هەڵوەشایەوە", "Cancelled"),
    "mon.status.expired": ("بەسەرچوو", "Expired"),
    "mon.dir.up": ("سەرەوە", "up"),
    "mon.dir.down": ("خوارەوە", "down"),
    "mon.dir.any": ("هەر لایەک", "any"),
    "mon.repeat": ("دووبارە", "repeat"),
    "mon.volume": ("{k} هێندەی تێکڕای {n} مۆم", "{k}x the average of {n} bars"),
    # --- activity -----------------------------------------------------------------------------
    "act.recent": ("کارە تازەکان", "Recent actions"),
    "act.timings": ("خێرایی قۆناغەکان (٢٤ کاتژمێری ڕابردوو)", "Stage timings (last 24 h)"),
    "act.last_turn": ("دوایین نۆبە", "Last turn"),
    "act.stage": ("قۆناغ", "Stage"),
    "act.count": ("ژمارە", "Count"),
    "act.avg": ("تێکڕا", "Average"),
    "act.max": ("زۆرترین", "Max"),
    "act.empty": ("هێشتا هیچ کارێک تۆمار نەکراوە.", "Nothing recorded yet."),
    "act.refresh": ("نوێکردنەوە", "Refresh"),
    "act.running": ("لە کاردایە…", "running…"),
    "act.kind.tool": ("ئامراز", "Tool"),
    "act.kind.confirm": ("دڵنیاکردنەوە", "Confirmation"),
    "act.kind.alert": ("ئاگاداری", "Alert"),
    "act.kind.worker": ("ئەرک", "Task"),
    "act.kind.error": ("هەڵە", "Error"),
    "act.kind.system": ("سیستەم", "System"),
    "act.kind.voice": ("دەنگ", "Voice"),
    # --- settings --------------------------------------------------------------------------------
    "set.keys": ("کلیلەکانی API", "API keys"),
    "set.keys_sub": ("کلیلەکان بە DPAPI ی ویندۆز پارێزراون و هەرگیز پیشان نادرێنەوە — تەنها دۆخەکەیان دەبینیت.",
                     "Keys are protected with Windows DPAPI and never shown again — only their status."),
    "set.key.gemini_api_key": ("Gemini (گووگڵ)", "Gemini (Google)"),
    "set.key.groq_api_key": ("Groq", "Groq"),
    "set.key.openrouter_api_key": ("OpenRouter", "OpenRouter"),
    "set.key.kurdishtts_stt_api_key": ("KurdishTTS — دەنگ بۆ نووسین", "KurdishTTS — speech to text"),
    "set.key.kurdishtts_tts_api_key": ("KurdishTTS — نووسین بۆ دەنگ", "KurdishTTS — text to speech"),
    "set.key.get": ("وەرگرتنی کلیل", "Get a key"),
    "set.paste": ("کلیلەکە لێرە بلکێنە", "Paste the key here"),
    "set.save": ("پاشەکەوت", "Save"),
    "set.test": ("تاقیکردنەوە", "Test"),
    "set.saved": ("پاشەکەوت کرا", "Saved"),
    "set.saving": ("پاشەکەوت دەکرێت…", "Saving…"),
    "set.configured": ("دانراوە", "Configured"),
    "set.not_set": ("دانەنراوە", "Not set"),
    "set.from_env": ("لە .env ەوە", "from .env"),
    "set.testing": ("تاقی دەکرێتەوە…", "Testing…"),
    "set.test_ok": ("کار دەکات", "Works"),
    "set.test.auth_failed": ("کلیلەکە ڕەت کرایەوە", "Key rejected"),
    "set.test.rate_limited": ("سنووری بەکارهێنان تەواو بووە", "Rate limited"),
    "set.test.unreachable": ("پەیوەندی نەکرا", "Unreachable"),
    "set.test.unconfigured": ("کلیل دانەنراوە", "No key"),
    "set.test.error": ("هەڵەیەک ڕوویدا", "Error"),
    "set.test.presence": ("تەنها بوونی کلیلەکە پشکنرا", "Only presence checked"),
    "set.bad_key": ("ئەمە وەک کلیلێکی دروست نییە. دڵنیابە هەمووی لکاندووە.",
                    "That does not look like a valid key. Check the paste."),
    "set.save_failed": ("پاشەکەوتکردن سەرکەوتوو نەبوو.", "Saving failed."),
    "set.omniroute": ("OmniRoute (دەروازەی ناوخۆیی)", "OmniRoute (local gateway)"),
    "set.voice": ("دەنگ", "Voice"),
    "set.voice.engine": ("بزوێنەری دەنگ", "Voice engine"),
    "set.voice.auto": ("خۆکار", "Automatic"),
    # Plain words for a non-expert (review 2026-09-24: "STT + LLM + TTS", CER and TTFA meant nothing).
    "set.voice.live": ("دەنگی خێرا (گووگڵ)", "Fast voice (Gemini Live)"),
    "set.voice.cascade": ("دەنگی ئاسایی", "Regular voice (speech to text + model + speech)"),
    "set.voice.selftest_good": ("باشە — دەنگی خێرا کوردییەکەت باش تێدەگات.", "Good — the fast voice understands your Sorani."),
    "set.voice.selftest_weak": ("لاوازە — دەنگی خێرا کوردییەکەت باش تێناگات، بۆیە دەنگی ئاسایی بەکاردێت.",
                                "Weak — the fast voice does not understand Sorani well, so the regular voice is used."),
    "set.voice.selftest_unfinished": ("تاقیکردنەوەکە تەواو نەبوو؛ دواتر دووبارە دەکرێتەوە.",
                                      "The test did not finish; it will run again later."),
    "set.voice.selftest": ("تاقیکردنەوەی دەنگ", "Voice self-test"),
    "set.voice.selftest_none": ("هێشتا تاقی نەکراوەتەوە.", "Not tested yet."),
    "set.voice.selftest_note": ("تاقیکردنەوەکە چەند داواکارییەکی بەخۆڕایی Gemini بەکاردەهێنێت.",
                                "The self-test uses a few free Gemini requests."),
    "set.voice.name": ("ناوی دەنگ", "Voice"),
    "set.voice.hotkey": ("کورتەڕێگای کیبۆرد", "Hotkey"),
    "set.voice.timeout": ("ماوەی گفتوگۆ دوای بێدەنگی", "Conversation timeout"),
    "set.voice.seconds": ("چرکە", "s"),
    "set.voice.always": ("هەمیشە گوێ بگرە", "Always listen"),
    "set.voice.hotkey_bad": ("کورتەڕێگاکە دروست نییە (نموونە: ctrl+alt+space)", "Invalid hotkey (e.g. ctrl+alt+space)"),
    "set.voice.hotkey_taken": ("ئەم کورتەڕێگایە لەلایەن بەرنامەیەکی ترەوە گیراوە؛ یەکێکی تر هەڵبژێرە.",
                               "Another program already uses this hotkey; choose another one."),
    "set.trading": ("ترەیدینگ", "Trading"),
    "set.tv": ("ترەیدینگ ڤیو", "TradingView"),
    "set.tv.connect": ("پەیوەستکردن", "Connect"),
    "set.tv.note": ("سام بە شێوەیەکی نافەرمی و تەنها لەسەر ئەم کۆمپیوتەرە ترەیدینگ ڤیوی دیسکتۆپ کۆنترۆڵ دەکات "
                    "(خوێندنەوەی چارت و کێشانی هێڵ). هەرگیز فەرمانی کڕین و فرۆشتن نانێرێت. دەرگا ناوخۆییەکەی "
                    "ترەیدینگ ڤیو تا داخستنی ترەیدینگ ڤیو کراوە دەمێنێت، تەنانەت دوای داخستنی سام؛ ئەگەر "
                    "بەرنامەی نەناسراو لەسەر ئەم کۆمپیوتەرە هەیە، دوای کارەکەت ترەیدینگ ڤیو دابخە.",
                    "SAM automates your own TradingView Desktop locally and unofficially (reading charts, "
                    "drawing lines). It never places orders. TradingView's local DevTools port stays open until "
                    "TradingView itself is closed, also after SAM quits; close TradingView when you are done if "
                    "you do not trust every program on this PC."),
    "set.mt5": ("MetaTrader 5", "MetaTrader 5"),
    "set.mt5.refresh": ("پشکنین", "Check"),
    "set.privacy": ("پاراستنی نهێنی", "Privacy"),
    "set.privacy.note": ("لە پلانی بەخۆڕایی Gemini، لەوانەیە گووگڵ ئەو شتانەی دەینێریت بۆ باشترکردنی بەرهەمەکانی "
                         "بەکاربهێنێت. زانیاریی هەستیار وەک وشەی نهێنی مەڵێ و مەنووسە. کلیلەکان تەنها بۆ "
                         "خاوەنی خۆیان دەنێردرێن و لە هیچ لۆگێکدا نانووسرێن.",
                         "On Gemini's free tier Google may use what you send to improve its products. Do not say "
                         "or type sensitive data such as passwords. Keys are only sent to their own provider and "
                         "never logged."),
    "set.about": ("دەربارە", "About"),
    "set.home": ("فۆڵدەری داتا", "Data folder"),
    "set.logs": ("فۆڵدەری لۆگ", "Log folder"),
    # --- accessible names (UI Automation: screen readers, SAM itself, tests) ------------------------------
    # One unique name per control: before, the five key rows all read «پاشەکەوت» / «تاقیکردنەوە» and the
    # fields, voice list, hotkey, timeout and switch had no name at all (UIA walk 2026-09-24).
    "a11y.key.field": ("کلیلی {name}", "{name} key"),
    "a11y.key.save": ("پاشەکەوتکردنی کلیلی {name}", "Save the {name} key"),
    "a11y.key.test": ("تاقیکردنەوەی کلیلی {name}", "Test the {name} key"),
    "a11y.key.get": ("وەرگرتنی کلیلی {name}", "Get a {name} key"),
    "a11y.key.status": ("دۆخی کلیلی {name}", "{name} key status"),
    "a11y.omniroute.test": ("تاقیکردنەوەی OmniRoute", "Test OmniRoute"),
    "a11y.engine": ("بزوێنەری دەنگ: {choice}", "Voice engine: {choice}"),
    "a11y.selftest": ("دەستپێکردنی تاقیکردنەوەی دەنگ", "Run the voice self-test"),
    "a11y.voice_name": ("هەڵبژاردنی ناوی دەنگ", "Voice name"),
    "a11y.hotkey": ("کورتەڕێگای کیبۆرد بۆ گوێگرتن", "Listening hotkey"),
    "a11y.timeout": ("ماوەی گفتوگۆ دوای بێدەنگی بە چرکە", "Conversation timeout in seconds"),
    "a11y.always": ("هەمیشە گوێ بگرە", "Always listen"),
    "a11y.tv.connect": ("پەیوەستکردنی ترەیدینگ ڤیو", "Connect TradingView"),
    "a11y.mt5.check": ("پشکنینی مێتاتڕەیدەر", "Check MetaTrader 5"),
    # --- statuses -------------------------------------------------------------------------------------
    "status.ok": ("کار دەکات", "OK"),
    "status.degraded": ("لاوازە", "Degraded"),
    "status.down": ("کار ناکات", "Down"),
    "status.unconfigured": ("ڕێکنەخراوە", "Not configured"),
    "status.unknown": ("نەزانراو", "Unknown"),
    "status.unavailable": ("ئەم بەشە هێشتا ئامادە نییە.", "This part is not available yet."),
    "status.connected": ("پەیوەستە", "Connected"),
    "status.not_connected": ("پەیوەست نییە", "Not connected"),
    "comp.voice": ("دەنگ", "Voice"),
    "comp.omniroute": ("OmniRoute", "OmniRoute"),
    "comp.tradingview": ("ترەیدینگ ڤیو", "TradingView"),
    "comp.mt5": ("MT5", "MT5"),
    "tv.state.connected": ("پەیوەست کرا", "Connected"),
    "tv.state.started": ("کرایەوە و پەیوەست کرا", "Started and connected"),
    "tv.state.restarted": ("دووبارە کرایەوە و پەیوەست کرا", "Restarted and connected"),
    "tv.state.needs_restart": ("پێویستە ترەیدینگ ڤیو دووبارە بکرێتەوە", "TradingView needs a restart"),
    "tv.state.not_installed": ("ترەیدینگ ڤیو دانەمەزراوە", "TradingView is not installed"),
    "tv.state.failed": ("پەیوەستکردن سەرکەوتوو نەبوو", "Connection failed"),
    "mt5.offset": ("جیاوازیی کاتی بڕۆکەر", "broker time offset"),
    "mt5.hours": ("کاتژمێر", "h"),
    "common.error": ("هەڵەیەک ڕوویدا", "Something went wrong"),
    "common.loading": ("بارکردن…", "Loading…"),
}

# Tool names -> short Sorani activity labels (island caption, activity log).
# Note: "remember" is deliberately not "لەبیرکردن" (that means *to forget*).
TOOL_LABELS: dict[str, tuple[str, str]] = {
    "stop_all": ("ڕاگرتنی هەموو کارەکان", "Stopping everything"),
    "remember": ("هەڵگرتنی زانیاری", "Saving a fact"),
    "recall": ("گەڕان لە بیرەوەری", "Searching memory"),
    "forget": ("سڕینەوە لە بیرەوەری", "Removing a fact"),
    "fetch_page": ("خوێندنەوەی پەڕەی ماڵپەڕ", "Reading a web page"),
    "system_control": ("باری کۆمپیوتەر و دەنگ", "Computer status and volume"),
    "theory_info": ("زانیاریی تیۆری", "Theory information"),
    "delegate_task": ("ئەرکی چەند هەنگاوی", "Multi-step task"),
    "open_app": ("کردنەوەی بەرنامە", "Opening an app"),
    "window_control": ("کۆنترۆڵی پەنجەرە", "Window control"),
    "type_text": ("نووسین", "Typing"),
    "press_keys": ("داگرتنی دوگمە", "Pressing keys"),
    "click": ("کرتەکردن", "Clicking"),
    "screen_look": ("سەیرکردنی شاشە", "Looking at the screen"),
    "screen_act": ("کارکردن لەسەر شاشە", "Working on the screen"),
    "run_powershell": ("جێبەجێکردنی فەرمان", "Running a command"),
    "files": ("کارکردن لەگەڵ فایل", "Working with files"),
    "open_url": ("کردنەوەی لینک", "Opening a link"),
    "web_search": ("گەڕان لە ئینتەرنێت", "Searching the web"),
    "build_project": ("دروستکردنی پڕۆژە", "Building a project"),
    "tv_open": ("کردنەوەی ترەیدینگ ڤیو", "Opening TradingView"),
    "tv_set_chart": ("گۆڕینی چارت", "Changing the chart"),
    "chart_state": ("خوێندنەوەی چارت", "Reading the chart"),
    "draw_on_chart": ("کێشان لەسەر چارت", "Drawing on the chart"),
    "clear_my_drawings": ("سڕینەوەی هێڵەکانی سام", "Removing SAM's drawings"),
    "get_price": ("وەرگرتنی نرخ", "Getting the price"),
    "analyze_market": ("شیکردنەوەی بازاڕ", "Analysing the market"),
    "set_alert": ("دانانی ئاگادارکەرەوە", "Setting an alert"),
    "list_alerts": ("لیستی ئاگادارکەرەوەکان", "Listing alerts"),
    "cancel_alert": ("هەڵوەشاندنەوەی ئاگادارکەرەوە", "Cancelling an alert"),
    "strategy_save": ("پاشەکەوتکردنی ستراتیژی", "Saving a strategy"),
    "strategy_list": ("لیستی ستراتیژییەکان", "Listing strategies"),
    "strategy_get": ("خوێندنەوەی ستراتیژی", "Reading a strategy"),
}

# Timing stages (sam.timing stage names) -> labels for the activity page.
STAGE_LABELS: dict[str, tuple[str, str]] = {
    "end_of_speech": ("کۆتایی قسەکردن", "End of speech"),
    "stt": ("دەنگ بۆ نووسین", "Speech to text"),
    "llm_first_token": ("یەکەم وشەی وەڵام", "First LLM token"),
    "llm_total": ("تەواوی وەڵامی مێشک", "LLM total"),
    "tts_first_audio": ("یەکەم دەنگی TTS", "TTS first audio"),
    "first_audio": ("یەکەم دەنگ", "Time to first audio"),
    "first_answer": ("یەکەم ڕستەی وەڵام", "First answer text"),
    "first_answer_audio": ("یەکەم دەنگی وەڵام", "First answer audio"),
    "first_chunk": ("یەکەم پارچەی وەڵام", "First reply chunk"),
    "total": ("کۆی گشتی", "Total"),
    "live_connect": ("پەیوەندی Live", "Live connect"),
    "confirm_wait": ("چاوەڕێی دڵنیاکردنەوە", "Confirmation wait"),
    "worker_step": ("هەنگاوی ئەرک", "Worker step"),
    "analysis_engine": ("شیکاری — بزوێنەر", "Analysis engine"),
    "analysis_total": ("شیکاری — گشتی", "Analysis total"),
    "tv_cdp": ("ترەیدینگ ڤیو (CDP)", "TradingView CDP"),
    "mt5_fetch": ("هێنانی داتای MT5", "MT5 fetch"),
}

# Gemini prebuilt voices with the style word from the official speech docs
# (voice/speech-generation.txt, saved 2026-09-24).
GEMINI_VOICES: tuple[tuple[str, str], ...] = (
    ("Kore", "Firm"), ("Puck", "Upbeat"), ("Charon", "Informative"), ("Zephyr", "Bright"),
    ("Fenrir", "Excitable"), ("Leda", "Youthful"), ("Orus", "Firm"), ("Aoede", "Breezy"),
    ("Callirrhoe", "Easy-going"), ("Autonoe", "Bright"), ("Enceladus", "Breathy"), ("Iapetus", "Clear"),
    ("Umbriel", "Easy-going"), ("Algieba", "Smooth"), ("Despina", "Smooth"), ("Erinome", "Clear"),
    ("Algenib", "Gravelly"), ("Rasalgethi", "Informative"), ("Laomedeia", "Upbeat"), ("Achernar", "Soft"),
    ("Alnilam", "Firm"), ("Schedar", "Even"), ("Gacrux", "Mature"), ("Pulcherrima", "Forward"),
    ("Achird", "Friendly"), ("Zubenelgenubi", "Casual"), ("Vindemiatrix", "Gentle"), ("Sadachbia", "Lively"),
    ("Sadaltager", "Knowledgeable"), ("Sulafat", "Warm"),
)

_EASTERN_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def tr(key: str, **fmt: Any) -> str:
    """Sorani text for ``key`` (English, then the key itself, as fallbacks)."""
    pair = STRINGS.get(key)
    text = (pair[0] or pair[1]) if pair else key
    return text.format(**fmt) if fmt else text


def tr_or(key: str, fallback: str) -> str:
    """Sorani text for ``key`` or ``fallback`` when the key is unknown."""
    return tr(key) if key in STRINGS else fallback


def en(key: str, **fmt: Any) -> str:
    """English text for ``key`` (secondary line / tooltips)."""
    pair = STRINGS.get(key)
    text = pair[1] if pair else key
    return text.format(**fmt) if fmt else text


def state_word(state: str) -> str:
    """Island status word for a ``VoiceState.state`` (unknown -> ready)."""
    return tr(f"state.{state}") if f"state.{state}" in STRINGS else tr("state.idle")


def tool_label(name: str) -> str:
    pair = TOOL_LABELS.get(name)
    return pair[0] if pair else name


def learn_tool_labels(registry: Any) -> int:
    """Fill labels for tools added after this table was written from each
    tool's own ``description_ckb`` (the registry's Sorani one-liner), so the
    island never shows a raw English tool name. Returns how many were added."""
    added = 0
    try:
        specs = registry.specs()
    except Exception:  # noqa: BLE001 - a registry stand-in without specs()
        return 0
    for spec in specs:
        name = getattr(spec, "name", "")
        ckb = str(getattr(spec, "description_ckb", "") or "").strip()
        if name and name not in TOOL_LABELS and ckb:
            TOOL_LABELS[name] = (ckb, str(getattr(spec, "description", "") or name))
            added += 1
    return added


def stage_label(stage: str) -> str:
    """Sorani label for a timing stage ("tool:open_app" -> the tool's label)."""
    if stage.startswith("tool:"):
        return tool_label(stage[5:])
    pair = STAGE_LABELS.get(stage)
    return pair[0] if pair else stage


def ckb_digits(value: Any) -> str:
    """Render ASCII digits as Eastern Arabic digits (Sorani convention)."""
    return str(value).translate(_EASTERN_DIGITS)


def countdown_text(seconds_left: float) -> str:
    """"١٨" for 17.2 s left (never negative)."""
    return ckb_digits(max(0, int(math.ceil(seconds_left))))


def ms_text(ms: float | None) -> str:
    """Compact duration: 820 ms / 1.4 s / 2 m 5 s (Latin digits: technical values)."""
    if ms is None:
        return "—"
    # The leading LRM keeps "820 ms" in order inside Sorani (RTL) text; without it
    # the bidi algorithm shows "ms 820".
    if ms < 1000:
        return f"\u200e{ms:.0f} ms"
    seconds = ms / 1000.0
    if seconds < 60:
        return f"\u200e{seconds:.1f} s"
    return f"\u200e{int(seconds // 60)} m {int(seconds % 60)} s"


__all__ = ["STRINGS", "TOOL_LABELS", "STAGE_LABELS", "GEMINI_VOICES", "tr", "tr_or", "en", "state_word",
           "tool_label", "learn_tool_labels", "stage_label", "ckb_digits", "countdown_text", "ms_text"]
