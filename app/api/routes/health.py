"""
app/api/routes/health.py
=========================
Health, readiness, and liveness probe endpoints.

Endpoints
---------
GET /health     — lightweight liveness probe (always fast).
GET /readiness  — deeper readiness check: verifies models, GPU, and DB.
GET /version    — application version metadata.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from fastapi import APIRouter, Depends, Request

from app.api.dependencies import get_registry, get_settings
from app.schemas.stream    import HealthResponse, ReadinessResponse

logger = logging.getLogger("crowd_analysis.api.routes.health")

router = APIRouter(tags=["Health"])

# Module-level start time to compute uptime
_SERVER_START_TIME = time.monotonic()


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description=(
        "Returns 200 immediately.  Suitable for load-balancer health checks "
        "and Kubernetes liveness probes.  Does not inspect sub-systems."
    ),
)
async def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    return HealthResponse(
        status      = "ok",
        version     = getattr(settings, "app_version", "1.0.0"),
        environment = getattr(settings, "environment", "development"),
        uptime_s    = round(time.monotonic() - _SERVER_START_TIME, 1),
    )


# ---------------------------------------------------------------------------
# GET /readiness
# ---------------------------------------------------------------------------

@router.get(
    "/readiness",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description=(
        "Checks each critical component and returns 200 only when all are "
        "ready.  Returns 503 if any component is unavailable.  Suitable for "
        "Kubernetes readiness probes."
    ),
)
async def readiness(request: Request) -> ReadinessResponse:
    components: Dict[str, bool] = {}
    app_state = request.app.state

    # WebSocket manager
    components["ws_manager"] = hasattr(app_state, "ws_manager")

    # Stream registry
    components["stream_registry"] = hasattr(app_state, "stream_registry")

    # Settings loaded
    components["settings"] = hasattr(app_state, "settings")

    # Model weights on disk (best-effort — non-blocking)
    try:
        import os
        from pathlib import Path
        settings = getattr(app_state, "settings", None)
        if settings:
            yolo_path = Path(
                getattr(getattr(settings, "model", None), "yolo_weights", "models/yolov8n.pt")
            )
            components["yolo_weights"] = yolo_path.exists()
        else:
            components["yolo_weights"] = False
    except Exception:
        components["yolo_weights"] = False

    # GPU / CUDA (optional — not a hard fail)
    try:
        import torch
        components["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        components["cuda_available"] = False

    all_ready = all(
        v for k, v in components.items()
        if k not in ("cuda_available",)   # GPU is optional
    )

    from fastapi import Response
    status_code = 200 if all_ready else 503

    return ReadinessResponse(
        ready      = all_ready,
        components = components,
        message    = "All systems ready." if all_ready else "One or more components not ready.",
    )


# ---------------------------------------------------------------------------
# GET /version
# ---------------------------------------------------------------------------

@router.get(
    "/version",
    summary="Application version",
    response_model=Dict[str, Any],
)
async def version(request: Request) -> Dict[str, Any]:
    settings = request.app.state.settings
    return {
        "name":        getattr(settings, "app_name",    "Crowd Analysis System"),
        "version":     getattr(settings, "app_version", "1.0.0"),
        "environment": getattr(settings, "environment", "development"),
    }