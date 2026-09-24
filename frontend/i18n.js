(() => {
  "use strict";

  const CKB = {
    "brand.name": "ئەجێنتی سام",
    "brand.tagline": "یاریزانی ناوخۆیی زیرەکی دەستکرد",
    "nav.newChat": "گفتوگۆی نوێ",
    "nav.agent": "گفتوگۆ",
    "nav.desktop": "کۆنترۆڵی دێسکتۆپ",
    "nav.tasks": "ئەرک و پلان",
    "nav.market": "زانیاری بازاڕ",
    "nav.tradingview": "TradingView",
    "nav.mt5": "میتاترەیدەر ٥",
    "nav.files": "فایلەکان",
    "nav.tools": "ئامرازەکان",
    "nav.approvals": "پەسەندکردن",
    "nav.memory": "بیرگە",
    "nav.audit": "تۆماری چاودێری",
    "nav.permissions": "مۆڵەتەکان",
    "nav.settings": "ڕێکخستنەکان",
    "nav.section.agent": "ئەجێنت",
    "nav.section.trading": "ترەیدینگ",
    "nav.section.workspace": "وۆرکسپەیس",
    "nav.section.memory": "بیرگە و چاودێری",
    "status.connecting": "پەیوەندی دەکرێت…",
    "status.checking": "خزمەتگوزاری ناوخۆیی دەپشکنرێت",
    "status.ready": "سام ئامادەیە",
    "status.needsModel": "سام پێویستی بە مۆدێلی چاتە",
    "status.offline": "خزمەتگوزاری داخراوە",
    "status.startSam": "سام دەستپێبکە بۆ پەیوەندی",
    "status.pill.connecting": "پەیوەندی",
    "status.pill.offline": "داخراو",
    "status.pill.ready": "خزمەتگوزاری ناوخۆیی ئامادەیە",
    "status.noModel": "مۆدێل نییە",
    "status.noChatModel": "مۆدێلی چات بەردەست نییە",
    "mode.guarded": "دۆخی پارێزراو",
    "mode.strict": "دۆخی توند",
    "mode.trusted": "وۆرکسپەیسی متمانەپێکراو",
    "welcome.eyebrow": "ئەجێنتی ناوخۆیی  ·  شیکاری بازاڕ",
    "welcome.title": "لەسەر چی کار بکەین؟",
    "welcome.copy": "دەتوانم زێڕ و زیو شیکار بکەم، TradingView بەکاربهێنم، داتای میتاترەیدەر بخوێنمەوە، فایل بگۆڕم، فرمان جێبەجێ بکەم، و پلان دابنێم — پێش هەر کاری مەترسیدار پەسەندت دەوێت.",
    "suggest.analyze.title": "شیکاری تەواوی زێڕ",
    "suggest.analyze.sub": "هەموو تایمفرەیمەکان لە میتاترەیدەری ناوخۆیی",
    "suggest.analyze.prompt": "شیکاری تەواوی XAUUSD لە فیدی ناوخۆیی میتاترەیدەر ٥ بکە. تایمفرەیمەکان H1 و M15 و M5 و M1 بەکاربهێنە. پێکهاتە، پشتگیری/بەرگری، لیکویدیتی، و بڕیاری WAIT یان NO_TRADE یان ENTRY ڕوون بکەوە.",
    "suggest.pine.title": "TradingView بکەرەوە",
    "suggest.pine.sub": "زێڕ پیشان بدە و شیکاری بکە",
    "suggest.pine.prompt": "TradingView بکەرەوە و XAUUSD بە SnR شیکاریم بۆ بکە. 1H و 15m ببینە و لە 5m و 1m entry بدۆزەوە.",
    "suggest.workspace.title": "وۆرکسپەیس بپشکنە",
    "suggest.workspace.sub": "پێکهاتە و ئەوەی گرنگە ڕوون بکەوە",
    "suggest.workspace.prompt": "ئەم وۆرکسپەیسە بپشکنە و کورتەیەکی پڕۆژەکەم پێ بدە: پێکهاتە، API، ئەوەی کار دەکات، و ئەگەر کێشەیەک هەبێت.",
    "suggest.auto.title": "کارێک ئۆتۆماتیکی بکە",
    "suggest.auto.sub": "ئۆتۆماتیکی دێسکتۆپ بە پەسەندکردن",
    "suggest.auto.prompt": "یارمەتیم بدە کارێکی دووبارە ئۆتۆماتیکی بکەم. بپرسە ئامانجەکە چییە، پاشان پلانێکی پارێزراو دابنێ کە پێش هەر هەنگاوێک پەسەندم بدات.",
    "composer.placeholder": "لە سام بپرسە: شیکاری بازاڕ، فایل، ئۆتۆماتیک، یان ڕوونکردنەوە. Shift+Enter هێڵی نوێ.",
    "composer.hint": "Enter بنێرە  ·  Shift+Enter هێڵی نوێ",
    "composer.listen": "گوێدەگرێت…",
    "composer.transcribing": "ناسینەوە…",
    "composer.stop": "وەستە",
    "title.new": "گفتوگۆی نوێ",
    "workspace.loading": "وۆرکسپەیس بار دەکرێت…",
    "settings.title": "ڕێکخستنەکانی ئەجێنت",
    "settings.kicker": "ڕێکخستن",
    "settings.general": "گشتی",
    "settings.models": "مۆدێلەکان",
    "settings.voice": "دەنگ",
    "settings.safety": "پاراستن",
    "settings.privacy": "تایبەتمەندی",
    "settings.language": "زمانی ڕووکار",
    "settings.saved": "پاشەکەوت کرا",
    "settings.saveFailed": "پاشەکەوت نەکرا",
    "settings.saving": "پاشەکەوت دەکرێت…",
    "toast.offline": "سام داخراوە",
    "toast.offline.detail": "خزمەتگوزاری ناوخۆیی دەستپێبکە، پاشان پەیوەندییەکە نوێ بکەوە.",
    "toast.saved": "ڕێکخستنەکان پاشەکەوت کران",
    "toast.saved.detail": "سام لە ئێستاوە ئەم هەڵبژاردنانە بەکاردەهێنێت.",
    "market.brain": "مێشکی ترەیدینگ",
    "market.refresh": "نوێکردنەوە",
    "market.analyze": "شیکاری",
    "market.compare": "بەراورد",
    "market.idle": "چاوەڕوان",
    "tv.disclosure": "سام تەنها ئەو دۆخە دەڵێت کە دەتوانێت بیبینێت. گۆڕینی تایمفرەیم و کیشانەوە تا سەربەخۆ پشتڕاست نەکرێنەوە بە partial دەمێننەوە.",
    "drawing.blocked": "کیشانەوە تا calibrationی چارت پشتڕاست نەکرێتەوە داخراوە",
    "perm.orders": "داواکاری زیندووی بروکەر جێبەجێ ناکرێت؛ MT5 تەنها خوێندنەوەیە",
    "approve": "پەسەندکردن",
    "deny": "ڕەتکردنەوە",
    "voice.supported": "ناسینەوەی سۆرانی لە ڕێگەی KurdishTTS دەڕوات، نەک لە وێبگەڕ. دوای قسەکردن وەڵامەکە بە دەنگ دەبیستیت.",
    "voice.unsupported": "مایکرۆفۆن لەم وێبگەڕە بەردەست نییە. نووسین هەر دەتوانرێت.",
    "voice.tooShort": "قسەکە زۆر کورت بوو",
    "voice.tooShort.detail": "تکایە دووبارە بڵێوە.",
    "voice.noSpeech": "هیچ قسەیەک نەبیسترا",
    "voice.recognizeFailed": "ناسینەوە سەری نەگرت",
    "voice.speakFailed": "وەڵامەکە نەخوێندرایەوە",
    "handsFree.stopped": "گوێ لە {phrase} ناگیرێت: گوێگرەکە وەستاوە",
    "handsFree.modelFailed": "لەوانەیە {phrase} نەبیسترێت: مۆدێلی ناسینەوەی قسە هەڵەیەکی ڕاگەیاند",
    "handsFree.listenerError": "گوێگر",
    "handsFree.modelError": "مۆدێلی قسە",
    "handsFree.stoppedHint": "Start دابگرە تا دیسان گوێ بگرێت. ئەگەر دیسان وەستا، سام دووبارە دەستپێبکەرەوە.",
    "handsFree.modelHint": "ئەگەر ئەمە لانەچوو، سام دووبارە دەستپێبکەرەوە.",
  };

  const EN = {
    "brand.name": "SAM Agent",
    "brand.tagline": "Local desktop AI",
    "nav.newChat": "New conversation",
    "nav.agent": "Agent Chat",
    "nav.desktop": "Desktop Control",
    "nav.tasks": "Tasks & Plans",
    "nav.market": "Market Intelligence",
    "nav.tradingview": "TradingView",
    "nav.mt5": "MetaTrader 5",
    "nav.files": "Files",
    "nav.tools": "Tools",
    "nav.approvals": "Approvals",
    "nav.memory": "Memory",
    "nav.audit": "Audit Log",
    "nav.permissions": "Permissions",
    "nav.settings": "Settings",
    "nav.section.agent": "Agent",
    "nav.section.trading": "Trading",
    "nav.section.workspace": "Workspace",
    "nav.section.memory": "Memory & Audit",
    "status.connecting": "Connecting…",
    "status.checking": "Checking local service",
    "status.ready": "SAM is ready",
    "status.needsModel": "SAM needs a chat model",
    "status.offline": "Service offline",
    "status.startSam": "Start SAM to connect",
    "status.pill.connecting": "Connecting",
    "status.pill.offline": "Service offline",
    "status.pill.ready": "Local service ready",
    "status.noModel": "No model",
    "status.noChatModel": "No chat model available",
    "mode.guarded": "Guarded mode",
    "mode.strict": "Strict mode",
    "mode.trusted": "Trusted workspace",
    "welcome.eyebrow": "LOCAL AGENT · MARKET INTELLIGENCE",
    "welcome.title": "What should we work on?",
    "welcome.copy": "I can analyze gold and silver, drive TradingView, read MetaTrader data, edit files, run commands, and plan work — with approval before anything risky.",
    "suggest.analyze.title": "Full XAUUSD analysis",
    "suggest.analyze.sub": "All timeframes from local MetaTrader",
    "suggest.analyze.prompt": "Perform a complete XAUUSD analysis from the local MetaTrader 5 feed using H1, M15, M5, and M1. Report structure, support/resistance, liquidity, and a WAIT, NO_TRADE, or ENTRY decision.",
    "suggest.pine.title": "Open TradingView",
    "suggest.pine.sub": "Show gold and analyse it",
    "suggest.pine.prompt": "Open TradingView and analyse XAUUSD with SnR. Show 1H and 15m, then hunt an entry on 5m and 1m.",
    "suggest.workspace.title": "Explore workspace",
    "suggest.workspace.sub": "Map the project and explain what matters",
    "suggest.workspace.prompt": "Inspect this workspace and give me a concise project overview including architecture, APIs, current functionality, and any bugs.",
    "suggest.auto.title": "Automate a workflow",
    "suggest.auto.sub": "Desktop automation with approval",
    "suggest.auto.prompt": "Help me automate a repetitive workflow. Ask me for the target and outcome, then propose a safe plan requiring approval before each step.",
    "composer.placeholder": "Ask SAM to analyze markets, edit files, automate, or explain. Shift+Enter for a new line.",
    "composer.listen": "Listening…",
    "composer.transcribing": "Recognising…",
    "composer.stop": "Stop",
    "title.new": "New conversation",
    "settings.title": "Agent Settings",
    "settings.saved": "Saved",
    "settings.saveFailed": "Could not save",
    "settings.saving": "Saving…",
    "toast.offline": "SAM is offline",
    "toast.offline.detail": "Start the local service, then refresh the connection.",
    "toast.saved": "Settings saved",
    "toast.saved.detail": "SAM will use your updated preferences.",
    "market.brain": "Trading Brain",
    "market.refresh": "Refresh",
    "market.analyze": "Analyze",
    "market.compare": "Compare",
    "drawing.blocked": "Drawing stays blocked until chart calibration is verified",
    "voice.supported": "Sorani speech is recognised through KurdishTTS, not the browser. Spoken replies play in this page.",
    "voice.unsupported": "Voice input is not available in this browser. Typed chat still works.",
    "voice.tooShort": "That was too short to recognise",
    "voice.tooShort.detail": "Please say it again.",
    "voice.noSpeech": "No speech was heard",
    "voice.recognizeFailed": "Speech was not recognised",
    "voice.speakFailed": "The reply could not be spoken",
    "handsFree.stopped": "Not listening for {phrase}: the listener stopped",
    "handsFree.modelFailed": "{phrase} may not be heard: the speech model reported an error",
    "handsFree.listenerError": "Listener",
    "handsFree.modelError": "Speech model",
    "handsFree.stoppedHint": "Press Start to listen again. If it stops again, restart SAM.",
    "handsFree.modelHint": "If this does not clear, restart SAM.",
  };

  const I18N = {
    locale: "ckb-IQ",
    dictionaries: { "ckb-IQ": CKB, ckb: CKB, en: EN, "en-US": EN },

    resolve(locale) {
      const raw = String(locale || "").trim();
      if (!raw || raw === "auto") {
        const voice = document.querySelector("#settings-voice-language")?.value || "ckb-IQ";
        return voice.toLowerCase().startsWith("ckb") || voice.toLowerCase() === "ku" ? "ckb-IQ" : "en";
      }
      if (raw.toLowerCase().startsWith("ckb") || raw === "ku") return "ckb-IQ";
      return "en";
    },

    isRtl(locale) {
      return this.resolve(locale) === "ckb-IQ";
    },

    t(key, fallback) {
      const dict = this.dictionaries[this.locale] || CKB;
      if (dict[key]) return dict[key];
      if (CKB[key] && this.locale !== "en") return CKB[key];
      return fallback == null ? key : fallback;
    },

    apply(locale) {
      this.locale = this.resolve(locale);
      const rtl = this.isRtl(this.locale);
      document.documentElement.lang = rtl ? "ckb" : "en";
      document.documentElement.dir = rtl ? "rtl" : "ltr";
      document.documentElement.classList.toggle("locale-ckb", rtl);

      document.querySelectorAll("[data-i18n]").forEach((node) => {
        const value = this.t(node.getAttribute("data-i18n"), node.textContent);
        this._setText(node, value);
      });
      document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => {
        node.setAttribute("placeholder", this.t(node.getAttribute("data-i18n-placeholder"), node.getAttribute("placeholder") || ""));
      });
      document.querySelectorAll("[data-i18n-aria]").forEach((node) => {
        node.setAttribute("aria-label", this.t(node.getAttribute("data-i18n-aria"), node.getAttribute("aria-label") || ""));
      });
      document.querySelectorAll("[data-i18n-prompt]").forEach((node) => {
        node.setAttribute("data-prompt", this.t(node.getAttribute("data-i18n-prompt"), node.getAttribute("data-prompt") || ""));
      });
      document.querySelectorAll("[data-i18n-title]").forEach((node) => {
        const strong = node.querySelector("strong");
        const small = node.querySelector("small");
        if (strong) strong.textContent = this.t(`${node.getAttribute("data-i18n-title")}.title`, strong.textContent);
        if (small) small.textContent = this.t(`${node.getAttribute("data-i18n-title")}.sub`, small.textContent);
      });
    },

    _setText(node, value) {
      const texts = [...node.childNodes].filter((child) => child.nodeType === 3);
      const meaningful = texts.filter((child) => child.textContent.trim());
      if (meaningful.length) {
        meaningful[meaningful.length - 1].textContent = node.children.length ? ` ${value}` : value;
        return;
      }
      if (!node.children.length) {
        node.textContent = value;
        return;
      }
      node.appendChild(document.createTextNode(` ${value}`));
    },
  };

  window.SAM_I18N = I18N;
  window.t = (key, fallback) => I18N.t(key, fallback);
})();
