<div dir="rtl" lang="ckb">

# SAM 2 — یاریدەدەری دەنگیی سۆرانی و شیکەرەوەی ترەیدینگ

SAM 2 لە سەرەتاوە دووبارە دروست کراوەتەوە. بە کوردیی سۆرانی قسەت لەگەڵ دەکات، کۆمپیوتەرەکەت بۆ بەڕێوە دەبات،
چارتەکانی **TradingView Desktop** شی دەکاتەوە و هێڵیان لەسەر دەکێشێت، و ستراتیژی و تیۆرییەکانی ترەیدینگت لە بیر دەمێنێت.

## SAM 2 چی دەکات؟

- **قسەکردن:** بە سۆرانی قسەی لەگەڵ دەکەیت و بە دەنگ وەڵامت دەداتەوە. دەتوانیت بنووسیشیت.
- **کۆمپیوتەر:** بەرنامە دەکاتەوە (کرۆم، ئێج، ترەیدینگ ڤیو، تێلێگرام، MetaTrader 5، VS Code، نۆتپاد…)،
  پەنجەرەکان ڕێک دەخات، دەنووسێت، کرتە دەکات، فایل و وێب بەکاردەهێنێت، و دەتوانێت پرۆژەیەکی بچووک (وەک ماڵپەڕێک) دروست بکات.
- **ترەیدینگ:** چارتی TradingView شی دەکاتەوە، هێڵی پشتگیری و بەرگری، زۆن، چوونەژوورەوە و ستۆپ و ئامانج دەکێشێت و
  دەیانسڕێتەوە؛ نرخ، قەبارە (ڤۆلیوم) و مۆمەکان لە MetaTrader 5 چاودێری دەکات و ئاگادارت دەکاتەوە.
- **بیرەوەری:** ستراتیژی و تیۆرییەکانت، و ئەو شتانەی پێی دەڵێیت «لە بیرت بێت»، تەنها لەسەر ئەم کۆمپیوتەرە هەڵدەگرێت.
- **هەرگیز ئۆردەر ناکات:** نە کڕین، نە فرۆشتن، نە داخستنی پۆزیشن.

## چۆن دەستی پێ بکەم؟

- دوو کرتە لەسەر ئایکۆنی **SAM** لە دێسکتۆپ یان لە Start بکە. پیلێکی بچووک لە ناوەڕاستی سەرەوەی شاشە دەردەکەوێت و پانێڵەکەش دەکرێتەوە.
- SAM لەگەڵ چوونەژوورەوەی ویندۆز خۆی دەست پێ دەکات (تەنها پیلەکە دەردەکەوێت، بەبێ پانێڵ).
- ئەگەر SAM پێشتر کراوەتەوە، کرتەکردنەوە لەسەر ئایکۆنەکە تەنها پانێڵەکەی پیشان دەداتەوە؛ SAMی دووەم دروست نابێت.
- بۆ داخستن: کرتەی ڕاست لەسەر پیلەکە یان ئایکۆنی SAM لە تاسکبار (tray) ← **داخستن**.

## چۆن قسەی لەگەڵ بکەم؟

1. کورتەڕێگای گوێگرتن دابگرە (بە بنەڕەت **Ctrl+Alt+Space**؛ ئەگەر بەرنامەیەکی تر گرتبووی، SAM خۆی **Win+Alt+Space** بەکاردەهێنێت — کورتەڕێگای ڕاستەقینە لە **ڕێکخستنەکان** دەبینیت)، یان **کرتە** لەسەر پیلەکە بکە. دەنووسێت «گوێ دەگرم».
2. بە ئاسایی قسە بکە. SAM تا ماوەیەک گوێ دەگرێت، پاشان خۆی دەخەوێت؛ بۆ دەستپێکردنەوە دووبارە کرتە بکە.
3. دەتوانیت بنووسیشیت: دوو کرتە لەسەر پیلەکە ← **گفتوگۆ**.

نموونە:

- «ترەیدینگ ڤیو بکەرەوە»
- «گۆڵد لەسەر ١٥ خولەک پیشان بدە»
- «هێڵی پشتگیری و بەرگری بکێشە» — «هێڵەکانت بسڕەوە»
- «زێڕ شی بکەرەوە بە ستراتیژییەکەم»
- «ئەگەر زێڕ گەیشتە ٢٧٠٠ ئاگادارم بکەرەوە»
- «بوەستە» (هەموو کارەکان ڕادەگرێت)

کارە مەترسیدارەکان (سڕینەوە، ناردنی پەیام، داخستنی پەنجەرە…): SAM پێشتر پرسیارت لێ دەکات. بڵێ **«بەڵێ»**
یان کرتە لە **بەڵێ** بکە. ئەگەر لە ماوەی ٢٠ چرکەدا وەڵام نەدەیتەوە، کارەکە **ناکرێت**.

## کلیلی Gemini چۆن دابنێم؟ (بۆ دەنگی زیندوو و بینینی شاشە)

1. بڕۆ بۆ <https://aistudio.google.com/apikey> و بە هەژماری Google ـەکەت بچۆ ژوورەوە.
2. **Create API key** دابگرە و کلیلەکە کۆپی بکە (بە `AIza` یان `AQ.` دەست پێ دەکات).
3. لە SAM: کرتەی ڕاست لەسەر پیلەکە ← **ڕێکخستنەکان** ← بەشی **کلیلەکانی API**.
4. لە خانەی **Gemini (گووگڵ)** کلیلەکە بلکێنە ← **پاشەکەوت** ← پاشان **تاقیکردنەوە**. کە نووسی «کار دەکات»، تەواوە.

- کلیلەکە بە پاراستنی ویندۆز (DPAPI) تەنها لەسەر ئەم کۆمپیوتەرە هەڵدەگیرێت و هەرگیز پیشان نادرێتەوە.
- هەموو کلیلەکانی پێشووت (Groq، OpenRouter، KurdishTTS، OmniRoute) وەک خۆیان کار دەکەن؛ پێویست ناکات دووبارە دایانبنێیتەوە.
- بەبێ کلیلی Gemini ـیش SAM کار دەکات (دەنگی زنجیرەیی بە KurdishTTS و Groq/OmniRoute).

## تێبینیی پاراستنی نهێنی

- SAM تەنها **پلانی بەخۆڕایی** بەکاردەهێنێت و هیچ پارەدانێک چالاک ناکات.
- لە پلانی بەخۆڕایی Gemini، لەوانەیە Google ئەو شتانەی دەینێریت (دەنگ، نووسین، وێنەی شاشە) بۆ باشترکردنی
  بەرهەمەکانی بەکاربهێنێت و لەوانەیە مرۆڤ بیانبینێت. زانیاریی هەستیار وەک وشەی نهێنی و ژمارەی هەژمار مەڵێ و مەنووسە.
  SAM پێش ناردنی وێنەی شاشە، خانەی وشەی نهێنی و بەشی هەژماری MT5 دادەپۆشێت.
- کلیلەکان تەنها بۆ خاوەنی خۆیان دەنێردرێن و لە هیچ لۆگێکدا نانووسرێن.

## TradingView

- SAM لە ڕێگەی دەرگایەکی ناوخۆیی (`127.0.0.1:9222`) لەگەڵ TradingView Desktop ـی خۆت کار دەکات. ئەمە
  **ئۆتۆماتیکردنێکی نافەرمی و ناوخۆیی**ـە بۆ بەرنامەکەی خۆت، نەک API ـی فەرمیی TradingView.
- SAM تەنها ئەو هێڵانە دەسڕێتەوە کە خۆی کێشاونی؛ دەست لە هێڵەکانی تۆ نادات.
- ئەگەر TradingView بەبێ ئەو دەرگایە کراوەتەوە، SAM پێش دووبارە کردنەوەی پرسیارت لێ دەکات.
- ئەو دەرگایە تا داخستنی TradingView کراوە دەمێنێت، تەنانەت دوای داخستنی SAM. ماڵپەڕەکان ناتوانن
  بەکاری بهێنن (Host و Origin ڕەت دەکرێنەوە)، بەڵام هەر بەرنامەیەکی تری سەر ئەم کۆمپیوتەرە دەتوانێت. ئەگەر
  متمانەت بە هەموو بەرنامەکانی سەر کۆمپیوتەرەکەت نییە، دوای کارەکەت TradingView دابخە.

## سەلامەتیی ترەیدینگ

- SAM **هەرگیز** ئۆردەر ناکات، ناگۆڕێت و دایناخات — نە لە MetaTrader 5، نە لە TradingView. تەنها شیکردنەوە، کێشان و ئاگادارکردنەوە.
- وەڵامەکانی تەنها «چاوەڕێ بکە»، «ترەید نییە» یان «ستاپ» ـن و هەرگیز «ئێستا بیکڕە» نین.
- شیکردنەوەکان ئامۆژگاریی دارایی نین؛ بڕیار هەر بە دەستی خۆتە.

## داتاکانی SAMی کۆن (v1)

لە یەکەم دەستپێکردندا SAM 2 داتاکانی v1 دەخوێنێتەوە بەبێ ئەوەی فایلی v1 بگۆڕێت: ئەو ستراتیژییانەی تاقیکردنەوە
ئۆتۆماتیکییەکان دروستیان کردبوو وەک **ئەرشیف** دێن (چالاک نین)، و ستاپ و ژۆرناڵ و گفتوگۆکان لە ئەرشیفێکدا دەمێننەوە.
هیچ پەیامێک کە شتێکی وەک کلیلی تێدابێت ناهێنرێت.

## ئەگەر کێشە هەبوو

- لۆگی دەستپێکردن: `%LOCALAPPDATA%\SAM\sam.log` — لۆگی وردەکاری: `%LOCALAPPDATA%\SAM2\logs\sam2.log`. هیچ کلیلێک لە لۆگەکاندا نییە.
- لابردنی ئایکۆنەکان: `scripts\uninstall.ps1` (خودی SAM و داتاکانی دەست لێ نادرێن).

</div>

---

# SAM 2 — English

SAM 2 is a from-scratch rebuild of a Sorani-speaking desktop voice agent and trading analyst for one
Windows 11 PC. It talks naturally (Gemini Live native audio, or a cascade STT → text LLM → TTS), controls the
desktop, analyses and draws on **TradingView Desktop** charts, monitors MetaTrader 5 data, and remembers the
user's trading strategies. Free tiers only; SAM never places, modifies or closes orders.

Design: [`docs/DESIGN.md`](docs/DESIGN.md). Module interfaces: [`docs/CONTRACTS.md`](docs/CONTRACTS.md).

## Architecture

One process (Python 3.13): the PySide6 UI on the main thread, one asyncio core loop on its own thread
(`sam.bridge.CoreThread`; core modules never import Qt).

| package | role |
| --- | --- |
| `sam/app.py`, `config.py`, `secrets.py`, `db.py`, `events.py`, `timing.py` | core services: settings, DPAPI key store (v1-compatible `data/secrets.json`), SQLite `data/sam2.sqlite3` (WAL, FTS5 trigram), event bus, per-stage timings |
| `sam/brain/` | tool registry (one definition → Gemini + OpenAI schemas), confirmation broker (voice/click, 20 s → no), LLM ladders with 429 fallback, persona, conversation, memory, multi-step worker |
| `sam/voice/` | Live voice (Gemini Live) and the cascade fallback, hotkey, self-test |
| `sam/hands/` | apps (Start-menu index + Sorani aliases), windows, UI Automation, OCR, screen + vision, PowerShell/files policy, web |
| `sam/trading/` | TradingView CDP bridge (draw/clear own drawings), MT5 feed (read-only), analysis engine, strategy cards, monitor/alerts |
| `sam/ui/` | island pill (top centre), panel (chat, strategies, monitoring, activity, settings), tray |
| `sam/migrate_v1.py` | one-time, read-only import of SAM v1's database |
| `sam/omniroute.py` | starts the local OmniRoute gateway when installed and down (never stops it) |
| `SAM.pyw` | pythonw launcher: single instance, early OmniRoute start, logging without a console |

`SAM_HOME` is the folder holding `.env` and `data/` (keys, key store, database). Logs live outside it:
`%LOCALAPPDATA%\SAM2\logs\sam2.log` (app) and `%LOCALAPPDATA%\SAM\sam.log` (launcher).

## Install and run

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -DryRun   # show what would happen
powershell -ExecutionPolicy Bypass -File scripts\install.ps1           # venv (Python 3.13) + requirements + icon + shortcuts
powershell -ExecutionPolicy Bypass -File scripts\uninstall.ps1         # remove the shortcuts only
```

The installer is idempotent. It creates/repairs `.venv` with Python 3.13 (a venv made by another Python is
moved aside, never deleted), installs `requirements.txt`, writes `%LOCALAPPDATA%\SAM\sam.ico`, and makes
Desktop `SAM.lnk`, Start-menu `SAM.lnk` and Startup `SAM (background).lnk` →
`.venv\Scripts\pythonw.exe SAM.pyw --home <SAM_HOME> [--background]` (these replace v1's shortcuts of the
same names). `-SamHome`, `-NoAutostart`, `-SkipPip`, `-SkipCheck` are available.

`SAM.pyw` flags: `--background` (sign-in start: island only; sets `SAM_BACKGROUND=1`), `--home PATH`,
`--no-omniroute`, `--write-icon PATH`; anything else (`--no-ui`, `--console`, `--check`, `--after-pid N`)
is passed to `python -m sam`. A second launch asks the running SAM to show its panel and exits.
`--quit` asks a running SAM 2 to shut down cleanly (the same path as tray Quit: it stops listening, closes
the TradingView and MT5 connections and the database) and waits up to 30 s; exit code 0 = stopped.
Start-up shows the island first and starts the packages behind it (measured on this PC: island visible
0.5–0.8 s after launch, core ready 1.1–2.1 s, ~215 MB working set when idle).

## Development

```powershell
scripts\dev.ps1                 # run from source with SAM_HOME (default %USERPROFILE%\Desktop\SAM-Agent), log to the console
scripts\dev.ps1 -NoUi           # core only
scripts\dev.ps1 -Check          # load every package and print a status (booleans only, never key values)
scripts\dev.ps1 -Test           # pytest
scripts\dev.ps1 -Acceptance -Only launcher
```

Or directly: `.venv\Scripts\python.exe -m sam --console --home C:\Users\samit\Desktop\SAM-Agent`.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest --basetemp work\pytest-me -p no:cacheprovider
```

Tests never use the network, the speakers, the microphone, the real `SAM_HOME` or live apps (Qt runs
offscreen; keys are fakes with the real shapes). Per package: `tests/test_<package>_*.py`.

## Acceptance (live, on this PC)

```powershell
.venv\Scripts\python.exe acceptance\run_all.py --home C:\Users\samit\Desktop\SAM-Agent [--only launcher] [--list]
```

`run_all.py` runs every `acceptance/<module>_*.py` one at a time with `SAM_HOME` set and prints a table;
the redacted report goes to `work/acceptance/<timestamp>.json`. A script passes with exit code 0, is
skipped with 77, and may refine the result with a JSON object (`ok`, `skipped`, `summary`, `checks`) written
to `$SAM_ACCEPTANCE_OUT` or printed last; `acceptance/_common.py` implements this. Live scripts must clean up
(drawings, symbol/timeframe), never place orders, never play audio and never print a key.
The launcher's own checks: `launcher_env` (venv, imports, OmniRoute, installer dry run, current shortcuts),
`launcher_migrate_v1` (real v1 DB → temp SAM 2 DB), `launcher_startup` (island time, RAM, no console,
second-launch behaviour; uses a temp home unless `SAM_ACCEPTANCE_REAL_HOME=1`).

## Configuration

Keys are pasted in Settings (stored with DPAPI in `<SAM_HOME>\data\secrets.json`, the same file and names as
v1) or come from the process environment / `<SAM_HOME>\.env`, which is never copied into child processes.
See [`.env.example`](.env.example) for every variable name SAM 2 reads. Everything else is a setting in the
database (`settings` table; defaults in `sam/config.py`), editable in the Settings page.

## Privacy and safety

- Free tiers only; on Gemini's free tier Google may use submitted content to improve its products.
- Keys are only sent to their own provider and are redacted from logs, the database, the UI and tool results.
- Risky actions need a spoken "بەڵێ" or a click within 20 s (default no); trading orders, disabling security
  tools, credential exfiltration and mass deletion are blocked outright. Text read from screens, files and
  web pages is treated as data, never as instructions.
- TradingView support is an unofficial local automation of the user's own desktop app (Chrome DevTools
  protocol on 127.0.0.1 only); SAM removes only drawings it created. The DevTools port stays open until
  TradingView itself is closed (also after SAM quits): web pages cannot use it (bad Host headers and foreign
  Origins are refused, checked 2026-09-24), but any local program can, so close TradingView when you are done
  if you do not trust every program on this PC.
