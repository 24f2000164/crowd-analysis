"""
core/video/capture.py
======================
Thread-based OpenCV VideoCapture wrapper.

Responsibility
--------------
Continuously read frames from a video source (RTSP, webcam, file, HTTP) in a
dedicated daemon thread and push them into a :class:`~core.video.frame_buffer.FrameBuffer`.
The pipeline processing thread never touches ``cv2.VideoCapture`` directly —
it only reads from the buffer.

Features
--------
- Automatic reconnection for RTSP/HTTP streams (configurable attempts + delay).
- Configurable target resolution (frame is resized on capture if needed).
- Clean start / stop lifecycle with a threading ``Event``.
- Exposes capture FPS via a rolling latency tracker.
- Emits structured log messages at every significant state change.

Usage
-----
    from core.video.capture       import VideoCapture
    from core.video.frame_buffer  import FrameBuffer

    buf     = FrameBuffer(maxsize=32)
    capture = VideoCapture(source="rtsp://cam/stream1", buffer=buf)
    capture.start()

    # … pipeline reads from buf …

    capture.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Union

import cv2
import numpy as np

from core.video.frame_buffer import FrameBuffer

logger = logging.getLogger("crowd_analysis.video.capture")


@dataclass
class CaptureConfig:
    """
    Configuration for a single capture source.

    Attributes
    ----------
    source               : RTSP URL, webcam index (int or "0"), or file path.
    target_width         : resize captured frames to this width (0 = no resize).
    target_height        : resize captured frames to this height (0 = no resize).
    target_fps           : desired read rate; 0 = as fast as possible.
    reconnect_attempts   : 0 = no retry.
    reconnect_delay_s    : seconds between reconnect attempts.
    buffer_api_preference: cv2 backend hint (e.g. cv2.CAP_FFMPEG).
    """
    source:                Union[str, int] = 0
    target_width:          int             = 1280
    target_height:         int             = 720
    target_fps:            int             = 25
    reconnect_attempts:    int             = 5
    reconnect_delay_s:     float           = 2.0
    buffer_api_preference: int             = cv2.CAP_ANY


class VideoCapture:
    """
    Thread-based video source reader.

    Parameters
    ----------
    source  : URL / index / path — forwarded to ``cv2.VideoCapture``.
    buffer  : shared FrameBuffer written by the capture thread.
    config  : CaptureConfig overrides.  ``source`` in config is ignored
              when ``source`` is passed as a positional arg.
    """

    def __init__(
        self,
        source: Union[str, int],
        buffer: FrameBuffer,
        config: Optional[CaptureConfig] = None,
    ) -> None:
        self._cfg    = config or CaptureConfig(source=source)
        self._cfg.source = source          # positional arg wins
        self._buffer = buffer

        self._cap:           Optional[cv2.VideoCapture] = None
        self._thread:        Optional[threading.Thread] = None
        self._stop_event     = threading.Event()
        self._frame_index    = 0
        self._reconnect_count = 0

        # Rolling FPS tracker
        self._latencies: Deque[float] = deque(maxlen=60)
        self._last_read_ts: float     = 0.0

        # Public state (read from any thread)
        self.is_running: bool           = False
        self.last_error: Optional[str]  = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "VideoCapture":
        """
        Open the video source and start the capture thread.

        Returns self for chaining::

            capture = VideoCapture("0", buf).start()
        """
        if self.is_running:
            logger.warning("VideoCapture already running — ignoring start().")
            return self

        self._stop_event.clear()
        self._cap = self._open_source()
        if self._cap is None:
            raise RuntimeError(
                f"Could not open video source: {self._cfg.source!r}"
            )

        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"capture-{self._cfg.source}",
            daemon=True,
        )
        self._thread.start()
        self.is_running = True
        logger.info(
            "VideoCapture started — source=%r  target=%dx%d@%dfps",
            self._cfg.source,
            self._cfg.target_width,
            self._cfg.target_height,
            self._cfg.target_fps,
        )
        return self

    def stop(self, timeout: float = 3.0) -> None:
        """
        Signal the capture thread to stop and wait for it to join.

        Parameters
        ----------
        timeout : maximum seconds to wait for the thread.
        """
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        if self._cap:
            self._cap.release()
            self._cap = None
        self.is_running = False
        logger.info(
            "VideoCapture stopped — total_frames=%d  reconnects=%d",
            self._frame_index,
            self._reconnect_count,
        )

    # ------------------------------------------------------------------
    # Capture loop (runs in daemon thread)
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Main read loop — runs until ``_stop_event`` is set."""
        frame_interval = 1.0 / max(self._cfg.target_fps, 1)

        while not self._stop_event.is_set():
            t0 = time.perf_counter()

            ret, raw_frame = self._read_frame()
         

             
            if not ret or raw_frame is None:

    # If the source is a video file, restart from beginning
                if isinstance(self._cfg.source, str):
                   logger.info(
                   "End of video reached — restarting video (index=%d)",
                   self._frame_index
                   )

                   self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                   continue

    # Otherwise try reconnecting (RTSP / webcam / http)
                logger.warning(
                 "Frame read failed (index=%d) — attempting reconnect.",
                self._frame_index,
                )

                if not self._reconnect():
                    logger.error(
                 "All reconnect attempts exhausted — stopping capture."
                   )
                    break

                continue

            # Optional resize
            frame = self._resize(raw_frame)

            # Push to buffer (drops oldest if full)
            self._buffer.put(frame, self._frame_index)
            self._frame_index += 1

            # FPS pacing — sleep the remainder of the frame interval
            elapsed = time.perf_counter() - t0
            self._latencies.append(elapsed)
            sleep_s = frame_interval - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

        self.is_running = False

    def _read_frame(self):
        """Wrap ``cap.read()`` with a None-guard."""
        if self._cap is None or not self._cap.isOpened():
            return False, None
        try:
            return self._cap.read()
        except Exception as exc:
            logger.error("cap.read() raised: %s", exc)
            return False, None

    # ------------------------------------------------------------------
    # Source management
    # ------------------------------------------------------------------

    def _open_source(self) -> Optional[cv2.VideoCapture]:
        """Open ``cv2.VideoCapture`` and apply resolution hints."""
        src = self._cfg.source
        # Coerce string integers like "0" to int so OpenCV uses the webcam API
        if isinstance(src, str) and src.isdigit():
            src = int(src)

        logger.debug("Opening video source: %r", src)
        cap = cv2.VideoCapture(src, self._cfg.buffer_api_preference)

        if not cap.isOpened():
            logger.error("cv2.VideoCapture could not open source: %r", src)
            return None

        # Request resolution (hint only — camera may ignore)
        if self._cfg.target_width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._cfg.target_width)
        if self._cfg.target_height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.target_height)
        if self._cfg.target_fps > 0:
            cap.set(cv2.CAP_PROP_FPS, self._cfg.target_fps)

        # Minimise internal OpenCV buffer to 1 frame to reduce latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        logger.info(
            "Source opened: %r  actual=%dx%d@%.1ffps",
            src, actual_w, actual_h, actual_fps,
        )
        return cap

    def _reconnect(self) -> bool:
        """
        Attempt to reopen the source up to ``reconnect_attempts`` times.

        Returns True if reconnection succeeded, False if all attempts failed.
        """
        max_attempts = self._cfg.reconnect_attempts
        if max_attempts == 0:
            return False

        for attempt in range(1, max_attempts + 1):
            logger.warning(
                "Reconnect attempt %d/%d for source %r …",
                attempt, max_attempts, self._cfg.source,
            )
            if self._cap:
                self._cap.release()
                self._cap = None

            time.sleep(self._cfg.reconnect_delay_s)
            cap = self._open_source()
            if cap is not None:
                self._cap = cap
                self._reconnect_count += 1
                logger.info("Reconnected to source %r.", self._cfg.source)
                return True

        return False

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame if target dimensions differ from actual."""
        th, tw = self._cfg.target_height, self._cfg.target_width
        if th <= 0 or tw <= 0:
            return frame
        h, w = frame.shape[:2]
        if h == th and w == tw:
            return frame
        return cv2.resize(frame, (tw, th), interpolation=cv2.INTER_LINEAR)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def capture_fps(self) -> float:
        """Rolling average capture FPS over the last 60 frames."""
        if not self._latencies:
            return 0.0
        avg = sum(self._latencies) / len(self._latencies)
        return round(1.0 / avg, 1) if avg > 0 else 0.0

    @property
    def frame_index(self) -> int:
        return self._frame_index

    def stats(self) -> dict:
        return {
            "source":          str(self._cfg.source),
            "frame_index":     self._frame_index,
            "capture_fps":     self.capture_fps,
            "reconnects":      self._reconnect_count,
            "buffer_stats":    self._buffer.stats(),
            "is_running":      self.is_running,
        }

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "VideoCapture":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    def __repr__(self) -> str:
        return (
            f"VideoCapture(source={self._cfg.source!r}, "
            f"running={self.is_running}, fps={self.capture_fps})"
        )