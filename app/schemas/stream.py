"""
app/schemas/stream.py
======================
Pydantic schemas for stream management endpoints.

All request and response bodies crossing the HTTP/WebSocket boundary are
validated here.  Using separate schema classes (rather than reusing internal
dataclasses) keeps the API contract stable even when internal types change.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Enumerations
# ============================================================================

class StreamStatus(str, Enum):
    """Lifecycle states of a managed stream."""
    IDLE      = "idle"
    STARTING  = "starting"
    RUNNING   = "running"
    STOPPING  = "stopping"
    STOPPED   = "stopped"
    ERROR     = "error"


class VideoSourceType(str, Enum):
    WEBCAM = "webcam"
    RTSP   = "rtsp"
    FILE   = "file"
    HTTP   = "http"


# ============================================================================
# Stream request / response
# ============================================================================

class StartStreamRequest(BaseModel):
    """Body for POST /streams/start."""

    source: str = Field(
        default="0",
        description=(
            "Video source.  Webcam index ('0'), RTSP URL "
            "(rtsp://...), or absolute file path."
        ),
        examples=["0", "rtsp://192.168.1.10:554/stream1", "/data/sample.mp4"],
    )
    source_type: VideoSourceType = Field(
        default=VideoSourceType.WEBCAM,
        description="Source type hint for the capture backend.",
    )
    frame_width:  int   = Field(default=1280, ge=320,  le=3840)
    frame_height: int   = Field(default=720,  ge=240,  le=2160)
    target_fps:   int   = Field(default=25,   ge=1,    le=120)
    frame_skip:   int   = Field(
        default=1, ge=1, le=10,
        description="Process every Nth frame (1 = every frame).",
    )
    show_heatmap:  bool = Field(default=True)
    show_velocity: bool = Field(default=True)

    @field_validator("source")
    @classmethod
    def source_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("source must not be empty.")
        return v.strip()


class StopStreamRequest(BaseModel):
    """Body for POST /streams/stop."""
    stream_id: str = Field(..., description="ID returned by /streams/start.")


class StreamInfo(BaseModel):
    """Single stream status record."""
    stream_id:    str
    source:       str
    status:       StreamStatus
    frame_index:  int
    fps:          float
    track_count:  int
    behavior:     str
    confidence:   float
    ws_clients:   int
    uptime_s:     float


class StreamListResponse(BaseModel):
    """Response for GET /streams."""
    streams: List[StreamInfo]
    total:   int


class StartStreamResponse(BaseModel):
    """Response for POST /streams/start."""
    stream_id:  str
    status:     StreamStatus
    ws_url:     str
    message:    str


class StopStreamResponse(BaseModel):
    """Response for POST /streams/stop."""
    stream_id: str
    status:    StreamStatus
    message:   str


# ============================================================================
# Health / readiness
# ============================================================================

class HealthResponse(BaseModel):
    """Response for GET /health."""
    status:      str
    version:     str
    environment: str
    uptime_s:    float


class ReadinessResponse(BaseModel):
    """Response for GET /readiness — reports individual component states."""
    ready:      bool
    components: Dict[str, bool]
    message:    str


# ============================================================================
# Analytics
# ============================================================================

class BehaviorEventRecord(BaseModel):
    """One persisted behavior event record."""
    id:          int
    stream_id:   str
    behavior:    str
    confidence:  float
    track_count: int
    frame_index: int
    timestamp:   str
    signals:     List[str]


class AnalyticsResponse(BaseModel):
    """Response for GET /analytics/events."""
    events:      List[BehaviorEventRecord]
    total:       int
    stream_id:   Optional[str]
    limit:       int
    offset:      int


# ============================================================================
# WebSocket message envelopes
# ============================================================================

class WSFrameMessage(BaseModel):
    """
    JSON metadata sent alongside each JPEG frame over the WebSocket.

    Clients receive two consecutive messages per frame:
      1. Text  — this JSON metadata envelope.
      2. Binary — raw JPEG bytes.
    """
    type:          str = "frame"
    stream_id:     str
    frame_index:   int
    fps:           float
    track_count:   int
    behavior:      str
    confidence:    float
    processing_ms: float
    tracks:        List[Dict[str, Any]]
    signals:       List[str]


class WSEventMessage(BaseModel):
    """
    JSON event message for detected anomalous behaviors.
    Sent whenever the behavior label is non-normal.
    """
    type:        str = "event"
    stream_id:   str
    behavior:    str
    confidence:  float
    frame_index: int
    signals:     List[str]
    track_labels: Dict[str, str]


class WSErrorMessage(BaseModel):
    """JSON error envelope pushed to WebSocket clients."""
    type:    str = "error"
    code:    int
    message: str


class WSStatusMessage(BaseModel):
    """Stream lifecycle status change message."""
    type:      str = "status"
    stream_id: str
    status:    StreamStatus
    message:   str