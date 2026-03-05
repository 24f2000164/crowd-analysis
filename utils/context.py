"""
src/utils/context.py
=====================
Context-aware LoggerAdapter classes for injecting structured fields into
every log record without cluttering call sites.

Adapters
--------
StreamContextAdapter   — injects ``stream_id``, ``frame_index``, and any
                         other per-stream fields into every record emitted
                         by a logger wrapped with this adapter.

PipelineStageAdapter   — extends StreamContextAdapter with the current
                         pipeline ``stage`` name (detect / track / behavior /
                         annotate).

RequestContextAdapter  — injects HTTP ``request_id``, ``method``, and
                         ``path`` for the API layer.

Thread-local context
--------------------
``ContextVar``-based context storage (``LogContext``) allows async
coroutines and threads to set their own context that is automatically
picked up by any logger in the same execution context.

Usage
-----
    from src.utils.context import StreamContextAdapter, LogContext

    logger = logging.getLogger("crowd_analysis.pipeline")

    # Adapter-based approach (explicit, preferred in classes)
    stream_log = StreamContextAdapter(logger, stream_id="a1b2", frame_index=0)
    stream_log.info("Frame processed")
    stream_log.set_frame(42)

    # ContextVar approach (implicit, useful in async routes)
    LogContext.set(stream_id="a1b2", request_id="req-123")
    logger.info("This record will include stream_id and request_id automatically")
    LogContext.clear()
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any, Dict, MutableMapping, Optional, Tuple


# ============================================================================
# Thread / Task local context store
# ============================================================================

_LOG_CONTEXT: ContextVar[Dict[str, Any]] = ContextVar(
    "log_context", default={}
)


class LogContext:
    """
    ContextVar-backed store for structured log fields.

    Safe for both threading and asyncio — each thread / coroutine has its
    own copy of the context dict.

    Usage
    -----
        LogContext.set(stream_id="a1b2", request_id="req-xyz")
        logger.info("message")   # automatically includes stream_id, request_id
        LogContext.clear()
    """

    @classmethod
    def set(cls, **kwargs: Any) -> None:
        """Merge key-value pairs into the current context."""
        ctx = dict(_LOG_CONTEXT.get())
        ctx.update(kwargs)
        _LOG_CONTEXT.set(ctx)

    @classmethod
    def get(cls) -> Dict[str, Any]:
        """Return a snapshot of the current context."""
        return dict(_LOG_CONTEXT.get())

    @classmethod
    def clear(cls) -> None:
        """Remove all fields from the current context."""
        _LOG_CONTEXT.set({})

    @classmethod
    def remove(cls, *keys: str) -> None:
        """Remove specific keys from the current context."""
        ctx = dict(_LOG_CONTEXT.get())
        for k in keys:
            ctx.pop(k, None)
        _LOG_CONTEXT.set(ctx)


# ============================================================================
# Base ContextAdapter — merges ContextVar store into every record
# ============================================================================

class _ContextInjectingAdapter(logging.LoggerAdapter):
    """
    Base adapter that merges the ``LogContext`` ContextVar store plus any
    adapter-level ``extra`` dict into every log record.
    """

    def process(
        self,
        msg:    str,
        kwargs: MutableMapping[str, Any],
    ) -> Tuple[str, MutableMapping[str, Any]]:
        # Start with the ContextVar store (lowest priority)
        merged = LogContext.get()

        # Overlay adapter-level extras (higher priority)
        if self.extra:
            merged.update(self.extra)

        # Overlay call-site extras (highest priority)
        call_extra = kwargs.get("extra") or {}
        merged.update(call_extra)

        kwargs["extra"] = merged
        return msg, kwargs


# ============================================================================
# Stream Context Adapter
# ============================================================================

class StreamContextAdapter(_ContextInjectingAdapter):
    """
    LoggerAdapter that injects stream-level context into every record.

    Fields injected
    ---------------
    stream_id   : str  — active stream identifier.
    frame_index : int  — current pipeline frame counter.

    Parameters
    ----------
    logger      : underlying logger.
    stream_id   : stream identifier.
    frame_index : initial frame counter (updated via ``set_frame``).
    extra       : additional static fields.

    Usage
    -----
        log = StreamContextAdapter(logger, stream_id="a1b2c3d4")
        log.info("Detection complete")          # → includes stream_id
        log.set_frame(42)
        log.debug("Tracking update")            # → includes stream_id, frame_index=42
    """

    def __init__(
        self,
        logger:      logging.Logger,
        stream_id:   str  = "",
        frame_index: int  = 0,
        **extra:     Any,
    ) -> None:
        base_extra = {"stream_id": stream_id, "frame_index": frame_index}
        base_extra.update(extra)
        super().__init__(logger, extra=base_extra)

    def set_frame(self, frame_index: int) -> None:
        """Update the frame_index field for subsequent log records."""
        self.extra["frame_index"] = frame_index

    def set_stream(self, stream_id: str) -> None:
        """Update the stream_id field."""
        self.extra["stream_id"] = stream_id


# ============================================================================
# Pipeline Stage Adapter
# ============================================================================

class PipelineStageAdapter(StreamContextAdapter):
    """
    Extends ``StreamContextAdapter`` with the current pipeline stage.

    Fields injected
    ---------------
    stream_id   : str  — active stream identifier.
    frame_index : int  — pipeline frame counter.
    stage       : str  — pipeline stage (detect / track / behavior / annotate).
    fps         : float — rolling pipeline FPS.

    Usage
    -----
        log = PipelineStageAdapter(logger, stream_id="a1b2", stage="detect")
        log.debug("YOLOv8 inference complete", extra={"detections": 7})
        log.set_stage("track")
        log.set_fps(24.8)
    """

    def __init__(
        self,
        logger:      logging.Logger,
        stream_id:   str   = "",
        frame_index: int   = 0,
        stage:       str   = "",
        fps:         float = 0.0,
        **extra:     Any,
    ) -> None:
        super().__init__(
            logger,
            stream_id=stream_id,
            frame_index=frame_index,
            stage=stage,
            fps=fps,
            **extra,
        )

    def set_stage(self, stage: str) -> None:
        """Switch the current pipeline stage label."""
        self.extra["stage"] = stage

    def set_fps(self, fps: float) -> None:
        """Update the rolling FPS field."""
        self.extra["fps"] = round(fps, 1)


# ============================================================================
# Request Context Adapter  (API layer)
# ============================================================================

class RequestContextAdapter(_ContextInjectingAdapter):
    """
    LoggerAdapter for HTTP request handling context.

    Fields injected
    ---------------
    request_id : str — X-Request-ID correlation identifier.
    method     : str — HTTP method (GET, POST, …).
    path       : str — URL path.
    client_ip  : str — remote client IP.

    Usage
    -----
        log = RequestContextAdapter(
            logger,
            request_id="rid-123",
            method="POST",
            path="/streams/start",
        )
        log.info("Stream started")
    """

    def __init__(
        self,
        logger:     logging.Logger,
        request_id: str = "",
        method:     str = "",
        path:       str = "",
        client_ip:  str = "",
        **extra:    Any,
    ) -> None:
        base = {
            "request_id": request_id,
            "method":     method,
            "path":       path,
            "client_ip":  client_ip,
        }
        base.update(extra)
        super().__init__(logger, extra=base)