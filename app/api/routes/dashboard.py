"""
app/api/routes/dashboard.py
=============================
FastAPI routes for the analytics dashboard backend.

Endpoints
---------
GET /dashboard/metrics  — per-stage pipeline performance (from PerformanceMonitor)
GET /dashboard/events   — recent abnormal events (from EventStore)
GET /dashboard/streams  — snapshot of all managed stream states

All endpoints read from shared singletons stored in ``app.state``.
They never write to the pipeline or start/stop streams.

Authentication follows the same ``X-API-Key`` pattern as the rest of the API
(enforced by the ``verify_api_key`` dependency from ``app.api.dependencies``).

Usage
-----
Mount this router in ``app/api/server.py``::

    from app.api.routes.dashboard import router as dashboard_router
    app.include_router(dashboard_router)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("crowd_analysis.api.routes.dashboard")

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------

def _get_event_store(request: Request):
    """Pull the EventStore singleton from app.state."""
    store = getattr(request.app.state, "event_store", None)
    if store is None:
        # Lazy initialise with the default DB path if not wired up yet
        try:
            from services.event_store import EventStore
            store = EventStore()
            request.app.state.event_store = store
            logger.warning(
                "EventStore was not pre-initialised in app.state — "
                "created lazily with default db path."
            )
        except Exception as exc:
            logger.error("Could not create EventStore: %s", exc)
            return None
    return store


def _get_performance_monitor(request: Request):
    """Pull the PerformanceMonitor singleton from app.state."""
    monitor = getattr(request.app.state, "performance_monitor", None)
    if monitor is None:
        try:
            from core.metrics.performance_monitor import PerformanceMonitor
            monitor = PerformanceMonitor()
            request.app.state.performance_monitor = monitor
            logger.warning(
                "PerformanceMonitor was not pre-initialised — created lazily."
            )
        except Exception as exc:
            logger.error("Could not create PerformanceMonitor: %s", exc)
            return None
    return monitor


def _get_stream_registry(request: Request):
    """Pull the StreamRegistry singleton from app.state."""
    return getattr(request.app.state, "stream_registry", None)


# ---------------------------------------------------------------------------
# GET /dashboard/metrics
# ---------------------------------------------------------------------------

@router.get(
    "/metrics",
    summary="Pipeline performance metrics",
    response_description=(
        "Rolling-average latency per pipeline stage and current FPS."
    ),
)
async def get_metrics(
    request: Request,
) -> JSONResponse:
    """
    Return rolling-average pipeline performance metrics.

    Response body
    -------------
    ::

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
    monitor = _get_performance_monitor(request)

    if monitor is None:
        return JSONResponse(
            status_code=503,
            content={"error": {"code": 503, "message": "PerformanceMonitor unavailable."}},
        )

    try:
        metrics = monitor.get_metrics()
    except Exception as exc:
        logger.error("get_metrics: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": 500, "message": "Failed to retrieve metrics."}},
        )

    return JSONResponse(content=metrics)


# ---------------------------------------------------------------------------
# GET /dashboard/events
# ---------------------------------------------------------------------------

@router.get(
    "/events",
    summary="Recent crowd behaviour events",
    response_description="List of most recently stored abnormal crowd events.",
)
async def get_events(
    request:   Request,
    limit:     int           = Query(default=50, ge=1, le=500,
                                     description="Maximum number of events to return."),
    stream_id: Optional[str] = Query(default=None,
                                     description="Filter events to a specific stream."),
    behavior:  Optional[str] = Query(default=None,
                                     description="Filter by behaviour label (e.g. 'panic')."),
) -> JSONResponse:
    """
    Return a list of recently stored abnormal crowd behaviour events.

    Events are returned in **newest-first** order.

    Response body
    -------------
    ::

        {
            "total": 12,
            "events": [
                {
                    "id": 42,
                    "timestamp": "2025-01-15T14:30:22.123456+00:00",
                    "stream_id": "abc123",
                    "frame_index": 820,
                    "behavior": "panic",
                    "confidence": 0.87,
                    "people_count": 34,
                    "density": 2.7e-05
                },
                ...
            ]
        }
    """
    store = _get_event_store(request)

    if store is None:
        return JSONResponse(
            status_code=503,
            content={"error": {"code": 503, "message": "EventStore unavailable."}},
        )

    try:
        events: List[Dict[str, Any]] = store.get_recent_events(
            limit=limit,
            stream_id=stream_id,
            behavior=behavior,
        )
    except Exception as exc:
        logger.error("get_events: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": 500, "message": "Failed to retrieve events."}},
        )

    return JSONResponse(content={"total": len(events), "events": events})


# ---------------------------------------------------------------------------
# GET /dashboard/streams
# ---------------------------------------------------------------------------

@router.get(
    "/streams",
    summary="Active stream snapshots",
    response_description="Metadata for all managed streams.",
)
async def get_streams(
    request: Request,
) -> JSONResponse:
    """
    Return a snapshot of all registered stream states.

    Each entry includes ``stream_id``, ``status``, ``source``,
    ``fps``, and how long the stream has been running.

    Response body
    -------------
    ::

        {
            "total": 2,
            "streams": [
                {
                    "stream_id": "abc123",
                    "status": "running",
                    "source": "0",
                    "fps": 24.7,
                    "started_at": "2025-01-15T14:25:00+00:00",
                    "ws_clients": 3
                },
                ...
            ]
        }
    """
    registry = _get_stream_registry(request)

    if registry is None:
        return JSONResponse(
            status_code=503,
            content={"error": {"code": 503, "message": "StreamRegistry unavailable."}},
        )

    try:
        # StreamRegistry stores entries as StreamRecord / dict — fetch all
        if hasattr(registry, "list_streams"):
            raw = registry.list_streams()
        elif hasattr(registry, "_streams"):
            raw = [_stream_to_dict(v) for v in registry._streams.values()]
        else:
            raw = []
    except Exception as exc:
        logger.error("get_streams: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": 500, "message": "Failed to retrieve streams."}},
        )

    return JSONResponse(content={"total": len(raw), "streams": raw})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stream_to_dict(record: Any) -> Dict[str, Any]:
    """
    Convert a StreamRecord (or any object with the expected attributes) to a
    plain dict for JSON serialisation.
    """
    if isinstance(record, dict):
        return record

    return {
        "stream_id":   getattr(record, "stream_id",  None),
        "status":      getattr(getattr(record, "status", None), "value",
                               str(getattr(record, "status", "unknown"))),
        "source":      getattr(record, "source",     None),
        "fps":         getattr(getattr(record, "pipeline", None), "fps", 0.0),
        "started_at":  str(getattr(record, "started_at",  "")),
        "ws_clients":  getattr(record, "ws_clients",  0),
    }