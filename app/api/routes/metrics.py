from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("crowd_analysis.api.routes.metrics")

router = APIRouter(tags=["Metrics"])


@router.get("/metrics", summary="Pipeline performance metrics")
async def get_performance_metrics(request: Request) -> JSONResponse:
    """
    Return rolling-average pipeline performance metrics.

    Response example::

        {
            "fps": 24.7,
            "detect_ms": 18.3,
            "track_ms": 12.1,
            "behavior_ms": 2.4,
            "annotate_ms": 1.8,
            "total_ms": 34.6,
            "frames_processed": 1450,
            "uptime_s": 58.2
        }
    """
    monitor = getattr(request.app.state, "performance_monitor", None)

    if monitor is None:
        try:
            from core.metrics.performance_monitor import PerformanceMonitor
            monitor = PerformanceMonitor()
            request.app.state.performance_monitor = monitor
        except Exception as exc:
            logger.error("PerformanceMonitor unavailable: %s", exc)
            return JSONResponse(
                status_code=503,
                content={"error": {"code": 503, "message": "Performance monitor not available."}},
            )

    try:
        return JSONResponse(content=monitor.get_metrics())
    except Exception as exc:
        logger.error("/metrics error: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": 500, "message": "Failed to retrieve metrics."}},
        )