"""
core/video/frame_buffer.py
===========================
Thread-safe ring buffer that decouples the high-frequency capture thread
from the lower-frequency processing pipeline.

Design
------
The capture thread writes frames unconditionally.  When the buffer is full
it silently drops the **oldest** frame (not the newest) so the processing
side always receives the most recent available image.  This drop-oldest
strategy keeps latency bounded under any CPU/GPU load condition.

The processing side calls ``get()`` which blocks with a configurable timeout
so the pipeline coroutine can yield control to the event loop without
busy-waiting.

Usage
-----
    buf = FrameBuffer(maxsize=32)

    # producer (capture thread)
    buf.put(frame)

    # consumer (pipeline coroutine, via executor)
    frame = buf.get(timeout=0.1)   # returns None on timeout
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("crowd_analysis.video.frame_buffer")


@dataclass(slots=True)
class BufferedFrame:
    """
    A single entry in the ring buffer.

    Attributes
    ----------
    frame       : np.ndarray — raw BGR frame (H, W, 3) uint8.
    frame_index : int        — 0-based monotonic counter from the capture thread.
    timestamp   : float      — time.perf_counter() at capture.
    """
    frame:       np.ndarray
    frame_index: int
    timestamp:   float


class FrameBuffer:
    """
    Thread-safe FIFO ring buffer with drop-oldest overflow policy.

    Parameters
    ----------
    maxsize : int
        Maximum number of frames held before the oldest is dropped.
        Keep this small (16–64) to bound latency; large values waste RAM
        and cause the pipeline to process stale frames.
    """

    def __init__(self, maxsize: int = 32) -> None:
        if maxsize < 2:
            raise ValueError("maxsize must be ≥ 2.")
        self._maxsize        = maxsize
        self._queue: queue.Queue[BufferedFrame] = queue.Queue(maxsize=maxsize)
        self._total_captured = 0
        self._total_dropped  = 0
        self._lock           = threading.Lock()

    # ------------------------------------------------------------------
    # Producer side (capture thread)
    # ------------------------------------------------------------------

    def put(self, frame: np.ndarray, frame_index: int) -> bool:
        """
        Insert a frame into the buffer.

        If the buffer is full the oldest frame is evicted first so this call
        never blocks.

        Parameters
        ----------
        frame       : BGR uint8 ndarray.
        frame_index : monotonic counter from the capture source.

        Returns
        -------
        bool — True if the frame was inserted cleanly, False if a drop occurred.
        """
        entry = BufferedFrame(
            frame=frame,
            frame_index=frame_index,
            timestamp=time.perf_counter(),
        )
        dropped = False

        with self._lock:
            if self._queue.full():
                try:
                    self._queue.get_nowait()   # evict oldest
                    self._total_dropped += 1
                    dropped = True
                except queue.Empty:
                    pass

            try:
                self._queue.put_nowait(entry)
                self._total_captured += 1
            except queue.Full:
                # Extremely rare race; just discard the new frame
                self._total_dropped += 1
                dropped = True

        if dropped:
            logger.debug(
                "Frame buffer full — dropped oldest frame. "
                "total_dropped=%d", self._total_dropped,
            )

        return not dropped

    # ------------------------------------------------------------------
    # Consumer side (pipeline)
    # ------------------------------------------------------------------

    def get(self, timeout: float = 0.05) -> Optional[BufferedFrame]:
        """
        Retrieve the oldest buffered frame.

        Parameters
        ----------
        timeout : float — maximum seconds to wait.  Returns ``None`` if the
                  buffer is empty after the timeout.

        Returns
        -------
        BufferedFrame | None
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> None:
        """Discard all buffered frames. Call on pipeline shutdown or reset."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        logger.debug("FrameBuffer drained.")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def qsize(self) -> int:
        """Current number of frames waiting in the buffer."""
        return self._queue.qsize()

    @property
    def is_empty(self) -> bool:
        return self._queue.empty()

    @property
    def total_captured(self) -> int:
        return self._total_captured

    @property
    def total_dropped(self) -> int:
        return self._total_dropped

    @property
    def drop_rate(self) -> float:
        """Fraction of frames dropped since creation."""
        total = self._total_captured + self._total_dropped
        return self._total_dropped / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "qsize":          self.qsize,
            "maxsize":        self._maxsize,
            "total_captured": self._total_captured,
            "total_dropped":  self._total_dropped,
            "drop_rate":      round(self.drop_rate, 4),
        }

    def __repr__(self) -> str:
        return (
            f"FrameBuffer(qsize={self.qsize}/{self._maxsize}, "
            f"dropped={self._total_dropped})"
        )