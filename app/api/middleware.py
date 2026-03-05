"""
app/api/middleware.py
======================
ASGI middleware stack for the Crowd Analysis API.

Middleware (applied in order, outermost first)
----------------------------------------------
1. RequestLoggingMiddleware — structured access log per request.
2. ProcessTimeMiddleware    — injects ``X-Process-Time-Ms`` response header.
3. CorrelationIDMiddleware  — injects / propagates ``X-Request-ID`` header.

Usage
-----
    from app.api.middleware import register_middleware
    register_middleware(app)
"""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.middleware.cors import CORSMiddleware

logger = logging.getLogger("crowd_analysis.api.middleware")


# ---------------------------------------------------------------------------
# Process-time header
# ---------------------------------------------------------------------------

class ProcessTimeMiddleware(BaseHTTPMiddleware):
    """Appends ``X-Process-Time-Ms`` to every response."""

    async def dispatch(self, request: Request, call_next) -> Response:
        t0       = time.perf_counter()
        response = await call_next(request)
        ms       = (time.perf_counter() - t0) * 1000
        response.headers["X-Process-Time-Ms"] = f"{ms:.2f}"
        return response


# ---------------------------------------------------------------------------
# Correlation ID
# ---------------------------------------------------------------------------

class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """
    Propagate or generate a ``X-Request-ID`` header.

    If the client sends ``X-Request-ID`` it is forwarded; otherwise a new
    UUID is assigned.  The ID appears in both the request and response
    headers so it can be correlated across distributed logs.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        req_id   = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = req_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response


# ---------------------------------------------------------------------------
# Structured access logger
# ---------------------------------------------------------------------------

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Emit one structured log line per HTTP request.

    Skips WebSocket upgrades (logged separately by the WS manager) and
    health-check routes to reduce noise.
    """

    _SKIP_PATHS = {"/health", "/readiness", "/docs", "/redoc", "/openapi.json"}

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip noisy endpoints
        if request.url.path in self._SKIP_PATHS:
            return await call_next(request)

        # Skip WebSocket upgrades
        if request.headers.get("upgrade", "").lower() == "websocket":
            return await call_next(request)

        t0       = time.perf_counter()
        response = await call_next(request)
        ms       = (time.perf_counter() - t0) * 1000

        req_id  = getattr(request.state, "request_id", "-")
        client  = f"{request.client.host}:{request.client.port}" if request.client else "-"

        logger.info(
            "%s %s %d %.1fms rid=%s client=%s",
            request.method,
            request.url.path,
            response.status_code,
            ms,
            req_id,
            client,
        )
        return response


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------

def register_middleware(app: FastAPI, cors_origins: list[str] = None) -> None:
    """
    Attach all middleware to the FastAPI application.

    Parameters
    ----------
    app          : FastAPI instance.
    cors_origins : list of allowed CORS origins.  Defaults to ["*"] (dev).
    """
    origins = cors_origins or ["*"]

    # CORS — must be first so preflight requests are handled before auth
    app.add_middleware(
        CORSMiddleware,
        allow_origins     = origins,
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )

    # Process-time header
    app.add_middleware(ProcessTimeMiddleware)

    # Correlation ID
    app.add_middleware(CorrelationIDMiddleware)

    # Access logging (innermost — sees final response status)
    app.add_middleware(RequestLoggingMiddleware)