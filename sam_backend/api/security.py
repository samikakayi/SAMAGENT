"""What every response carries, and who is allowed to ask.

SAM binds to loopback and serves a local page, so the guard is about keeping
it that way: requests from anywhere but this machine are refused, and the
headers deny the page the capabilities it has no reason to want.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..config import Settings


def register_local_request_guard(application: FastAPI, settings: Settings) -> None:
    @application.middleware("http")
    async def local_request_guard(request: Request, call_next):
        host_header = request.headers.get("host", "").split(":", 1)[0].strip("[]").lower()
        client_host = (request.client.host if request.client else "").lower()
        allowed_hosts = {"127.0.0.1", "localhost", "::1", "testserver", "testclient"}
        if host_header and host_header not in allowed_hosts:
            return JSONResponse({"detail": "SAM only accepts loopback requests."}, status_code=403)
        if client_host and client_host not in allowed_hosts:
            return JSONResponse({"detail": "SAM only accepts local clients."}, status_code=403)
        origin = request.headers.get("origin")
        if origin and origin not in settings.cors_origins:
            return JSONResponse({"detail": "Origin is not allowed."}, status_code=403)
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Permissions-Policy", "microphone=(self), camera=(), geolocation=()")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; connect-src 'self'; media-src 'self' blob:; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
        )
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response
