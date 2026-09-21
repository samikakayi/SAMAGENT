from __future__ import annotations

import uvicorn

from .config import Settings
from .dpi import ensure_dpi_awareness


def main() -> None:
    # Must happen before any window or screen coordinate is read.
    ensure_dpi_awareness()
    settings = Settings.from_env()
    # A computer-control API must not be exposed to the LAN. Reverse-proxy deployment is intentionally unsupported.
    host = "127.0.0.1"
    uvicorn.run("sam_backend.app:app", host=host, port=settings.port, reload=False, access_log=False)


if __name__ == "__main__":
    main()
