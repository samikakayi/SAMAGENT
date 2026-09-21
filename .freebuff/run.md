# Running SAM-Agent for preview

SAM is a Python/FastAPI backend that also serves the static frontend
(`frontend/index.html`) directly — there is no separate npm/build step and no
Node tooling involved.

## Reproducing uncommitted artifacts

A fresh checkout/worktree needs:

1. A Python virtualenv at `.venv/` with dependencies from
   `sam_backend/requirements.txt` (plus `requirements-lock.txt` for exact
   pinned versions) installed. From the project root:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\pip.exe install -r requirements-lock.txt
   .\.venv\Scripts\pip.exe install -r sam_backend/requirements.txt
   ```
   Note: `python-multipart` is required (used by the `/api/voice/transcribe`
   upload endpoint) — it is listed in `sam_backend/requirements.txt`.
2. Copy `.env` from the main checkout (`C:\Users\samit\Desktop\SAM-Agent\.env`)
   into the worktree root. It holds local provider/model config — never
   invent values, copy the file itself. Adjust `SAM_PORT`/`SAM_HOST` in the
   copy if the default port is already taken in this worktree.
3. Optional local-voice packages (`faster-whisper`, `silero-vad`,
   `onnxruntime`, `piper-tts`, listed in `requirements-optional.txt`) are NOT
   required for the UI preview — browser voice and typed chat work without
   them; only fully-offline STT/TTS needs them.

## Running the server

Default port is `8765`. Start with the venv's Python directly (do not rely on
a `python`/`npm` shim being on PATH when detaching the process):

```powershell
C:\Users\samit\Desktop\SAM-Agent\.venv\Scripts\python.exe -m sam_backend --host 127.0.0.1 --port 8765
```

Health check: `GET http://127.0.0.1:8765/api/health` and the root `/` serves
the frontend (mounted via `StaticFiles` in `sam_backend/app.py`).

If 8765 is taken, pass a different `--port` (see `.\start.ps1 -Port 8877` in
the README for the PowerShell-wrapper equivalent).
