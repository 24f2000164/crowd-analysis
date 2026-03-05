"""
core/metrics/performance_monitor.py
=====================================
Lightweight per-stage latency and FPS monitor for the analysis pipeline.

Responsibility
--------------
Track wall-clock latency for each pipeline stage (detect, track, behavior)
and expose aggregate metrics as a plain dict consumable by the ``/metrics``
FastAPI endpoint.

Design
------
* A single ``PerformanceMonitor`` instance is created at app startup and
  stored in ``app.state``.
* Pipeline stages call ``record(stage, elapsed_s)`` after every frame.
* The ``/metrics`` endpoint calls ``get_metrics()`` to read the latest values.
* All operations are thread-safe (rolling deques under a lock).

Usage
-----
    from core.metrics.performance_monitor import PerformanceMonitor

    monitor = PerformanceMonitor()

    # inside pipeline — record one stage timing
    monitor.record("detect",   detect_elapsed_s)
    monitor.record("track",    track_elapsed_s)
    monitor.record("behavior", behavior_elapsed_s)

    # in the /metrics route
    return monitor.get_metrics()
    # {
    #     "fps": 24.7,
    #     "detect_ms":   18.3,
    #     "track_ms":    12.1,
    #     "behavior_ms": 2.4,
    #     "total_ms":    32.8,
    #     "frames_processed": 1450,
    # }
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict

logger = logging.getLogger("crowd_analysis.metrics.performance_monitor")

# Stages tracked by the monitor (must match pipeline stage names)
_TRACKED_STAGES = ("detect", "track", "behavior", "annotate")

# Rolling window size (number of recent samples to average)
_WINDOW_SIZE = 60


@dataclass
class _RollingBuffer:
    """Thread-safe rolling deque of elapsed-time samples."""
    maxlen: int = _WINDOW_SIZE
    _buf:   Deque[float] = field(default_factory=lambda: deque(maxlen=_WINDOW_SIZE))
    _lock:  threading.Lock = field(default_factory=threading.Lock)

    def record(self, elapsed_s: float) -> None:
        with self._lock:
            self._buf.append(elapsed_s)

    def avg_ms(self) -> float:
        with self._lock:
            if not self._buf:
                return 0.0
            return round(sum(self._buf) / len(self._buf) * 1_000, 2)

    def count(self) -> int:
        with self._lock:
            return len(self._buf)


class PerformanceMonitor:
    """
    Rolling-window per-stage latency monitor.

    Parameters
    ----------
    window_size : int
        Number of most-recent samples to include in rolling averages.
        Default 60 corresponds to ~2.4 s of history at 25 fps.
    """

    def __init__(self, window_size: int = _WINDOW_SIZE) -> None:
        self._window      = window_size
        self._stages:     Dict[str, _RollingBuffer] = {
            stage: _RollingBuffer(maxlen=window_size)
            for stage in _TRACKED_STAGES
        }
        self._frame_times: _RollingBuffer = _RollingBuffer(maxlen=window_size)
        self._total_frames: int  = 0
        self._start_time:   float = time.monotonic()
        self._lock          = threading.Lock()

        logger.info(
            "PerformanceMonitor initialised — window=%d stages=%s",
            window_size, list(_TRACKED_STAGES),
        )

    # ------------------------------------------------------------------
    # Write API (called by the pipeline)
    # ------------------------------------------------------------------

    def record(self, stage: str, elapsed_s: float) -> None:
        """
        Record the elapsed time for one stage of one frame.

        Parameters
        ----------
        stage     : one of ``"detect"``, ``"track"``, ``"behavior"``,
                    ``"annotate"``.  Unknown stages are silently ignored.
        elapsed_s : wall-clock seconds taken by that stage.
        """
        buf = self._stages.get(stage)
        if buf is not None:
            buf.record(elapsed_s)
        else:
            logger.debug("PerformanceMonitor: unknown stage '%s' ignored.", stage)

    def record_frame(self, elapsed_s: float) -> None:
        """
        Record total end-to-end frame processing time.

        Parameters
        ----------
        elapsed_s : total seconds from frame receipt to pipeline output.
        """
        self._frame_times.record(elapsed_s)
        with self._lock:
            self._total_frames += 1

    # ------------------------------------------------------------------
    # Read API (called by /metrics route)
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, object]:
        """
        Return the current rolling-average performance metrics.

        Returns
        -------
        dict with keys:
            ``fps``              — estimated frames per second.
            ``detect_ms``        — average YOLO detection latency (ms).
            ``track_ms``         — average DeepSORT tracking latency (ms).
            ``behavior_ms``      — average behavior analysis latency (ms).
            ``annotate_ms``      — average annotation latency (ms).
            ``total_ms``         — average end-to-end latency (ms).
            ``frames_processed`` — total frames processed since startup.
            ``uptime_s``         — seconds since the monitor was created.
        """
        detect_ms   = self._stages["detect"].avg_ms()
        track_ms    = self._stages["track"].avg_ms()
        behavior_ms = self._stages["behavior"].avg_ms()
        annotate_ms = self._stages["annotate"].avg_ms()

        # Total latency from per-frame wall-clock (preferred) or sum of stages
        total_avg = self._frame_times.avg_ms()
        if total_avg == 0.0:
            total_avg = round(detect_ms + track_ms + behavior_ms + annotate_ms, 2)

        fps = 0.0
        if total_avg > 0.0:
            fps = round(1_000 / total_avg, 1)

        with self._lock:
            frames_processed = self._total_frames
            uptime_s         = round(time.monotonic() - self._start_time, 1)

        return {
            "fps":               fps,
            "detect_ms":         detect_ms,
            "track_ms":          track_ms,
            "behavior_ms":       behavior_ms,
            "annotate_ms":       annotate_ms,
            "total_ms":          total_avg,
            "frames_processed":  frames_processed,
            "uptime_s":          uptime_s,
        }

    def reset(self) -> None:
        """Clear all recorded samples and reset the frame counter."""
        for buf in self._stages.values():
            with buf._lock:
                buf._buf.clear()
        with self._frame_times._lock:
            self._frame_times._buf.clear()
        with self._lock:
            self._total_frames = 0
            self._start_time   = time.monotonic()
        logger.info("PerformanceMonitor reset.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        """Current rolling average FPS (shortcut for the route layer)."""
        avg_ms = self._frame_times.avg_ms()
        return round(1_000 / avg_ms, 1) if avg_ms > 0 else 0.0

    @property
    def total_frames(self) -> int:
        """Total frames recorded since creation (or last reset)."""
        with self._lock:
            return self._total_frames

    def __repr__(self) -> str:
        m = self.get_metrics()
        return (
            f"PerformanceMonitor("
            f"fps={m['fps']}, "
            f"detect_ms={m['detect_ms']}, "
            f"track_ms={m['track_ms']}, "
            f"behavior_ms={m['behavior_ms']}, "
            f"frames={m['frames_processed']})"
        )