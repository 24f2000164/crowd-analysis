"""
app/api/dependencies.py
========================
FastAPI dependency injection factories.

All route handlers receive their dependencies (registry, settings, API-key
guard) through ``Depends()`` so they stay lean and independently testable.

Usage
-----
    from app.api.dependencies import get_registry, require_api_key

    @router.post("/streams/start")
    async def start(
        body:     StartStreamRequest,
        registry: StreamRegistry = Depends(get_registry),
        _:        None           = Depends(require_api_key),
    ): ...
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Depends, Header, HTTPException, Request, status

from app.api.websocket.manager import ConnectionManager
from app.api.stream_registry    import StreamRegistry

logger = logging.getLogger("crowd_analysis.api.dependencies")


# ---------------------------------------------------------------------------
# App-state accessors
# ---------------------------------------------------------------------------

def get_ws_manager(request: Request) -> ConnectionManager:
    """Return the singleton ConnectionManager from app state."""
    return request.app.state.ws_manager


def get_registry(request: Request) -> StreamRegistry:
    """Return the singleton StreamRegistry from app state."""
    return request.app.state.stream_registry


def get_settings(request: Request):
    """Return the application Settings object from app state."""
    return request.app.state.settings


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

async def require_api_key(
    request:   Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    """
    Validate the ``X-API-Key`` header when an API key is configured.

    If ``settings.api.api_key`` is ``None`` (development mode) this
    dependency is a no-op and every request passes through.

    Raises
    ------
    HTTP 401 when the key is missing.
    HTTP 403 when the key is wrong.
    """
    settings = request.app.state.settings
    expected = getattr(getattr(settings, "api", None), "api_key", None)

    if expected is None:
        return   # Auth disabled — dev mode

    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-API-Key header is required.",
        )

    if x_api_key != expected:
        logger.warning(
            "Invalid API key from %s", request.client.host if request.client else "unknown"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )


async def optional_api_key(
    request:   Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> Optional[str]:
    """
    Non-raising variant — returns the key or None.
    Used for WebSocket upgrade (can't raise HTTP errors mid-handshake).
    """
    settings = request.app.state.settings
    expected = getattr(getattr(settings, "api", None), "api_key", None)
    if expected is None:
        return None
    return x_api_key if x_api_key == expected else None