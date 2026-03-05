"""
core/pipeline/frame_consumer.py
=================================
Annotated frame consumer: JPEG-encodes processed frames and dispatches them
to registered output sinks (WebSocket manager, output queue, file writer).

Responsibility
--------------
After the pipeline stages produce a ``PipelineFrame``, this module handles
everything that needs to happen *after* inference:

  1. JPEG encode the annotated frame (configurable quality).
  2. Package the frame + JSON metadata into a payload.
  3. Fan out to registered sinks:
       - WebSocket connection manager (push to all subscribers)
       - asyncio output queue (for FastAPI route polling)
       - Optional on-disk frame writer (debug / recording mode)

Design notes
------------
- ``FrameConsumer`` is intentionally sink-agnostic — it holds a list of
  callables rather than direct references to WebSocket objects so it can be
  tested with mock sinks.
- JPEG encoding is done synchronously because it is fast (<1 ms for 720p)
  and keeping it in the event loop avoids the overhead of executor dispatch
  for every frame.

Usage
-----
    from core.pipeline.frame_consumer import FrameConsumer
    from core.pipeline.video_pipeline import PipelineFrame

    consumer = FrameConsumer(jpeg_quality=85)
    consumer.add_queue_sink(output_queue)
    consumer.add_callback_sink(ws_manager.broadcast_bytes)

    async for pf in pipeline.run():
        await consumer.consume(pf)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, List, Optional

import cv2
import numpy as np

logger = logging.getLogger("crowd_analysis.pipeline.consumer")

# Type alias for async sink callbacks
AsyncSink = Callable[[bytes, dict], Awaitable[None]]


class FrameConsumer:
    """
    Fan-out consumer that encodes and dispatches ``PipelineFrame`` objects.

    Parameters
    ----------
    jpeg_quality  : JPEG encode quality (1–100).
    drop_on_slow  : when True, skip frame dispatch if any sink's queue is full
                    rather than blocking.  Prevents the pipeline from backing up
                    behind a slow WebSocket client.
    """

    def __init__(
        self,
        jpeg_quality: int  = 85,
        drop_on_slow: bool = True,
    ) -> None:
        self._quality     = jpeg_quality
        self._drop_slow   = drop_on_slow
        self._queue_sinks:    List[asyncio.Queue] = []
        self._callback_sinks: List[AsyncSink]     = []
        self._total_dispatched = 0
        self._total_dropped    = 0

    # ------------------------------------------------------------------
    # Sink registration
    # ------------------------------------------------------------------

    def add_queue_sink(self, q: asyncio.Queue) -> None:
        """Add an asyncio.Queue as an output sink."""
        self._queue_sinks.append(q)

    def add_callback_sink(self, cb: AsyncSink) -> None:
        """
        Add an async callback sink.

        The callback receives ``(jpeg_bytes: bytes, meta: dict)``.
        """
        self._callback_sinks.append(cb)

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    async def consume(self, pipeline_frame: Any) -> None:
        """
        Encode and dispatch a single ``PipelineFrame``.

        Parameters
        ----------
        pipeline_frame : PipelineFrame from the video pipeline.
        """
        # JPEG encode
        t0   = time.perf_counter()
        jpeg = self._encode_jpeg(pipeline_frame.annotated_frame)
        meta = pipeline_frame.to_meta_dict()
        encode_ms = (time.perf_counter() - t0) * 1000

        if not jpeg:
            logger.warning(
                "JPEG encode failed for frame %d — skipping dispatch.",
                pipeline_frame.frame_index,
            )
            return

        # Fan out to queue sinks
        for q in self._queue_sinks:
            await self._push_to_queue(q, (jpeg, meta))

        # Fan out to callback sinks
        for cb in self._callback_sinks:
            try:
                await cb(jpeg, meta)
            except Exception as exc:
                logger.warning("Callback sink error: %s", exc)

        self._total_dispatched += 1

        logger.debug(
            "Frame %d dispatched — jpeg=%d bytes  encode=%.1fms",
            pipeline_frame.frame_index,
            len(jpeg),
            encode_ms,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode_jpeg(self, frame: np.ndarray) -> bytes:
        """JPEG-encode a BGR ndarray. Returns empty bytes on failure."""
        try:
            ok, buf = cv2.imencode(
                ".jpg", frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self._quality],
            )
            return bytes(buf) if ok else b""
        except Exception as exc:
            logger.error("JPEG encode error: %s", exc)
            return b""

    async def _push_to_queue(
        self,
        q:    asyncio.Queue,
        item: Any,
    ) -> None:
        """Push an item to a queue, dropping if full and drop_on_slow=True."""
        if self._drop_slow and q.full():
            self._total_dropped += 1
            logger.debug("Queue sink full — dropping frame.")
            return
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            self._total_dropped += 1
            logger.debug("Queue sink full (race) — dropping frame.")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def total_dispatched(self) -> int:
        return self._total_dispatched

    @property
    def total_dropped(self) -> int:
        return self._total_dropped

    def stats(self) -> dict:
        return {
            "total_dispatched": self._total_dispatched,
            "total_dropped":    self._total_dropped,
            "queue_sinks":      len(self._queue_sinks),
            "callback_sinks":   len(self._callback_sinks),
        }
