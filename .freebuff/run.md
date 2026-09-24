# Running SAM 2

SAM 2 is a Windows desktop program (PySide6 UI + asyncio core in one Python
3.13 process). There is no web server, no frontend build and no Node tooling.

## A fresh checkout/worktree needs

1. A Python 3.13 virtualenv at `.venv/` with `requirements.txt`:
   ```powershell
   py -3.13 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
   (`scripts\install.ps1 -NoAutostart -SkipCheck` does the same and also
   writes the icon and shortcuts.)
2. Nothing else. Do NOT copy `.env` or `data\secrets.json` anywhere: SAM 2
   reads keys at runtime from `SAM_HOME` (the folder holding `.env` and
   `data\`). On this PC that is `C:\Users\samit\Desktop\SAM-Agent`.

## Running

```powershell
scripts\dev.ps1                  # UI, console log, SAM_HOME = Desktop\SAM-Agent
scripts\dev.ps1 -NoUi            # core only
scripts\dev.ps1 -Check           # load packages, print status (no key values)
.\.venv\Scripts\python.exe -m pytest --basetemp work\pytest-preview -p no:cacheprovider
```

Only one SAM 2 runs at a time (named mutex); a second start shows the running
one's panel. Logs: `%LOCALAPPDATA%\SAM2\logs\sam2.log`.
