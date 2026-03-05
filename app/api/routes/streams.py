"""
app/api/routes/streams.py
==========================
REST endpoints for camera and stream lifecycle management.

Endpoints
---------
POST /streams/start    — start a new video pipeline for a camera source.
POST /streams/stop     — gracefully stop a running pipeline.
GET  /streams          — list all registered streams and their status.
GET  /streams/{id}     — get status of a specific stream.

These are the "control plane" routes.  The "data plane" (annotated frame
delivery) lives in the WebSocket endpoint at /ws/stream/{stream_id}.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.dependencies import get_registry, require_api_key
from app.api.stream_registry import StreamRegistry
from app.schemas.stream import (
    StartStreamRequest,
    StartStreamResponse,
    StopStreamRequest,
    StopStreamResponse,
    StreamInfo,
    StreamListResponse,
    StreamStatus,
)
from core.pipeline.video_pipeline import PipelineConfig, VideoPipeline

logger = logging.getLogger("crowd_analysis.api.routes.streams")

router = APIRouter(prefix="/streams", tags=["Streams"])


# ---------------------------------------------------------------------------
# POST /streams/start
# ---------------------------------------------------------------------------

@router.post(
    "/start",
    response_model=StartStreamResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start camera / video stream",
    description=(
        "Creates a new pipeline instance for the given video source and "
        "starts processing in the background.  Returns a ``stream_id`` that "
        "clients use to subscribe via WebSocket at "
        "``ws://<host>/ws/stream/{stream_id}``."
    ),
)
async def start_stream(
    body:     StartStreamRequest,
    request:  Request,
    registry: StreamRegistry = Depends(get_registry),
    _:        None           = Depends(require_api_key),
) -> StartStreamResponse:
    """Start a new video analysis stream."""

    logger.info(
        "start_stream — source=%r  %dx%d@%dfps  skip=%d",
        body.source, body.frame_width, body.frame_height,
        body.target_fps, body.frame_skip,
    )

    # Build pipeline configuration from request body
    pipeline_cfg = PipelineConfig(
        source          = body.source,
        frame_width     = body.frame_width,
        frame_height    = body.frame_height,
        target_fps      = body.target_fps,
        frame_skip      = body.frame_skip,
        show_heatmap    = body.show_heatmap,
        show_velocity   = body.show_velocity,
    )

    # Construct the full pipeline (detector + tracker + analyzer + renderer)
    try:
        pipeline = VideoPipeline.from_settings(config=pipeline_cfg)
    except Exception as exc:
        logger.exception("Failed to build pipeline for source=%r: %s", body.source, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline construction failed: {exc}",
        )

    # Register and start
    try:
        stream_id = await registry.start_stream(source=body.source, pipeline=pipeline)
    except Exception as exc:
        logger.exception("Failed to start stream: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stream start failed: {exc}",
        )

    # Build WebSocket URL from the incoming request's base URL
    base = str(request.base_url).rstrip("/")
    ws_url = base.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_url}/ws/stream/{stream_id}"

    logger.info("Stream %s started successfully — ws_url=%s", stream_id, ws_url)

    return StartStreamResponse(
        stream_id = stream_id,
        status    = StreamStatus.RUNNING,
        ws_url    = ws_url,
        message   = f"Stream {stream_id!r} started. Connect to {ws_url} to receive frames.",
    )


# ---------------------------------------------------------------------------
# POST /streams/stop
# ---------------------------------------------------------------------------

@router.post(
    "/stop",
    response_model=StopStreamResponse,
    summary="Stop a running stream",
)
async def stop_stream(
    body:     StopStreamRequest,
    registry: StreamRegistry = Depends(get_registry),
    _:        None           = Depends(require_api_key),
) -> StopStreamResponse:
    """Gracefully stop a running pipeline and release all resources."""

    logger.info("stop_stream — stream_id=%s", body.stream_id)

    if not registry.exists(body.stream_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stream {body.stream_id!r} not found.",
        )

    try:
        await registry.stop_stream(body.stream_id)
    except Exception as exc:
        logger.exception("Error stopping stream %s: %s", body.stream_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stop failed: {exc}",
        )

    return StopStreamResponse(
        stream_id = body.stream_id,
        status    = StreamStatus.STOPPED,
        message   = f"Stream {body.stream_id!r} stopped successfully.",
    )


# ---------------------------------------------------------------------------
# GET /streams
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=StreamListResponse,
    summary="List all streams",
)
async def list_streams(
    registry: StreamRegistry = Depends(get_registry),
) -> StreamListResponse:
    """Return status information for every registered stream."""
    streams = registry.list_streams()
    return StreamListResponse(streams=streams, total=len(streams))


# ---------------------------------------------------------------------------
# GET /streams/{stream_id}
# ---------------------------------------------------------------------------

@router.get(
    "/{stream_id}",
    response_model=StreamInfo,
    summary="Get stream status",
)
async def get_stream(
    stream_id: str,
    registry:  StreamRegistry = Depends(get_registry),
) -> StreamInfo:
    """Return the current status of a specific stream."""
    info = registry.get_stream_info(stream_id)
    if info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stream {stream_id!r} not found.",
        )
    return info