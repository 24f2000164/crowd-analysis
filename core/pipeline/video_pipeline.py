"""
core/pipeline/video_pipeline.py
=================================
Real-time video processing pipeline for the Crowd Behavior Analysis System.

Pipeline stages (executed per frame)
--------------------------------------
  Video Source  ──►  FrameBuffer  ──►  Frame Producer (async generator)
       │
       ▼
  [Stage 1]  Person Detection      — YOLOv8Detector.detect()
       │
  [Stage 2]  Multi-Object Tracking — DeepSORTTracker.update()
       │
  [Stage 3]  Behavior Analysis     — BehaviorAnalyzer.analyze()
       │
  [Stage 4]  Frame Annotation      — FrameRenderer.render()
       │
       ▼
  Annotated frame  ──►  WebSocket manager  /  output queue

Concurrency model
-----------------
The pipeline runs as a single ``asyncio`` coroutine (``run()``).
The heavy inference steps (detection, tracking) are dispatched to a
``ThreadPoolExecutor`` via ``loop.run_in_executor()`` so they never
block the event loop.  This allows the FastAPI / WebSocket layer to
remain responsive while GPU inference is executing.

Frame skipping
--------------
When the processing loop falls behind the capture rate, the ``FrameBuffer``
automatically drops the oldest frames (drop-oldest policy).  In addition,
the pipeline has an explicit ``frame_skip`` parameter: when set to N, every
Nth frame is processed and the rest are annotated with the last known result,
giving a speed-accuracy trade-off knob.

Error recovery
--------------
Every stage is wrapped in a try/except.  A stage failure returns the last
good result for that stage (or a safe default) so the pipeline never
terminates on a single bad frame.  After ``max_consecutive_errors``
consecutive failures the pipeline logs a critical error and exits gracefully.

Usage
-----
    from core.pipeline.video_pipeline import VideoPipeline, PipelineConfig

    cfg      = PipelineConfig(source="rtsp://cam/stream", frame_skip=2)
    pipeline = VideoPipeline.from_settings(config=cfg)

    async def main():
        async for annotated_frame, result in pipeline.run():
            # push annotated_frame to WebSocket clients
            # push result.to_dict() as JSON event
            pass

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional, Tuple

import numpy as np

from core.video.capture       import VideoCapture, CaptureConfig
from core.video.frame_buffer  import FrameBuffer, BufferedFrame
from core.detection.yolo_detector  import YOLOv8Detector, DetectionResult
from core.tracking.deepsort_tracker import DeepSORTTracker, TrackedPerson
from core.behavior.behavior_analyzer import BehaviorAnalyzer
from core.behavior.base_analyzer     import BehaviorLabel, BehaviorResult
from core.annotation.renderer        import FrameRenderer
from core.behavior.crowd_metrics     import CrowdMetricsAnalyzer

logger = logging.getLogger("crowd_analysis.pipeline")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PipelineConfig:
    """
    All runtime knobs for a single pipeline instance.

    Attributes
    ----------
    source               : video source (RTSP URL, webcam index, file path).
    frame_width          : capture / processing resolution width.
    frame_height         : capture / processing resolution height.
    target_fps           : desired pipeline throughput (frames / second).
    frame_buffer_size    : ring buffer capacity.
    frame_skip           : process every Nth frame; annotate skipped frames
                           with the last result (1 = process every frame).
    max_consecutive_errors : stop the pipeline after this many back-to-back
                             stage failures.
    output_jpeg_quality  : JPEG encode quality for WebSocket emission.
    reconnect_attempts   : RTSP reconnect attempts on stream failure.
    reconnect_delay_s    : seconds between reconnect attempts.
    executor_workers     : thread pool size for blocking inference calls.
    show_heatmap         : render density heatmap overlay.
    show_velocity        : render per-track velocity arrows.
    """
    source:                   str   = "0"
    frame_width:              int   = 1280
    frame_height:             int   = 720
    target_fps:               int   = 25
    frame_buffer_size:        int   = 32
    frame_skip:               int   = 1
    max_consecutive_errors:   int   = 10
    output_jpeg_quality:      int   = 85
    reconnect_attempts:       int   = 5
    reconnect_delay_s:        float = 2.0
    executor_workers:         int   = 2
    show_heatmap:             bool  = True
    show_velocity:            bool  = True


# ============================================================================
# Pipeline output
# ============================================================================

@dataclass
class PipelineFrame:
    """
    One fully-processed output unit from the pipeline.

    Attributes
    ----------
    annotated_frame : BGR uint8 ndarray ready for display or JPEG encoding.
    jpeg_bytes      : JPEG-encoded bytes for WebSocket streaming (lazy).
    tracks          : confirmed tracks this frame.
    behavior_result : behavior classification result.
    frame_index     : pipeline-level frame counter.
    capture_index   : source-level frame counter from the capture thread.
    processing_ms   : wall-clock time for all pipeline stages (ms).
    fps             : rolling pipeline FPS.
    was_skipped     : True when frame_skip caused inference to be omitted.
    """
    annotated_frame: np.ndarray
    jpeg_bytes:      Optional[bytes]
    tracks:          List[TrackedPerson]
    behavior_result: BehaviorResult
    frame_index:     int
    capture_index:   int
    processing_ms:   float
    fps:             float
    was_skipped:     bool = False

    def encode_jpeg(self, quality: int = 85) -> bytes:
        """Lazily encode the annotated frame as JPEG."""
        if self.jpeg_bytes is None:
            import cv2
            ok, buf = cv2.imencode(
                ".jpg", self.annotated_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), quality],
            )
            self.jpeg_bytes = bytes(buf) if ok else b""
        return self.jpeg_bytes

    def to_meta_dict(self) -> Dict:
        """Serialisable metadata (no image bytes) for JSON event streams."""
        # Compute crowd density from current track count and configured frame size
        try:
            density_info = CrowdMetricsAnalyzer.compute_density(
                people_count=len(self.tracks),
                frame_width=self.annotated_frame.shape[1],
                frame_height=self.annotated_frame.shape[0],
            )
        except Exception:
            density_info = {"density": 0.0, "level": "low_density"}

        return {
            "frame_index":    self.frame_index,
            "capture_index":  self.capture_index,
            "processing_ms":  round(self.processing_ms, 2),
            "fps":            round(self.fps, 1),
            "track_count":    len(self.tracks),
            "was_skipped":    self.was_skipped,
            "behavior":       self.behavior_result.to_dict(),
            "crowd_density":  density_info,
        }


# ============================================================================
# Performance tracker
# ============================================================================

class _PerfTracker:
    """Rolling FPS and per-stage latency tracker."""

    def __init__(self, window: int = 60) -> None:
        from collections import deque
        self._times: "deque[float]" = __import__("collections").deque(maxlen=window)
        self._stage: Dict[str, list] = {
            "detect":   [],
            "track":    [],
            "behavior": [],
            "annotate": [],
        }

    def record_frame(self, elapsed_s: float) -> None:
        self._times.append(elapsed_s)

    def record_stage(self, stage: str, elapsed_s: float) -> None:
        buf = self._stage.setdefault(stage, [])
        buf.append(elapsed_s)
        if len(buf) > 60:
            buf.pop(0)

    @property
    def fps(self) -> float:
        if not self._times:
            return 0.0
        avg = sum(self._times) / len(self._times)
        return round(1.0 / avg, 1) if avg > 0 else 0.0

    def stage_ms(self, stage: str) -> float:
        buf = self._stage.get(stage, [])
        if not buf:
            return 0.0
        return round(sum(buf[-10:]) / len(buf[-10:]) * 1000, 1)

    def summary(self) -> Dict:
        return {
            "fps":          self.fps,
            "detect_ms":    self.stage_ms("detect"),
            "track_ms":     self.stage_ms("track"),
            "behavior_ms":  self.stage_ms("behavior"),
            "annotate_ms":  self.stage_ms("annotate"),
        }


# ============================================================================
# Main Pipeline
# ============================================================================

class VideoPipeline:
    """
    Async real-time video processing pipeline.

    Wire all processing stages together and expose an async generator
    ``run()`` that yields ``PipelineFrame`` objects.

    Parameters
    ----------
    detector  : YOLOv8Detector
    tracker   : DeepSORTTracker
    analyzer  : BehaviorAnalyzer
    renderer  : FrameRenderer
    config    : PipelineConfig
    """

    def __init__(
        self,
        detector:  YOLOv8Detector,
        tracker:   DeepSORTTracker,
        analyzer:  BehaviorAnalyzer,
        renderer:  FrameRenderer,
        config:    PipelineConfig,
    ) -> None:
        self._detector  = detector
        self._tracker   = tracker
        self._analyzer  = analyzer
        self._renderer  = renderer
        self._cfg       = config

        # Shared infrastructure
        self._buffer    = FrameBuffer(maxsize=config.frame_buffer_size)
        self._executor  = ThreadPoolExecutor(
            max_workers=config.executor_workers,
            thread_name_prefix="pipeline-worker",
        )
        self._perf      = _PerfTracker()

        # Pipeline state
        self._frame_index:         int  = 0
        self._consecutive_errors:  int  = 0
        self._running:             bool = False

        # Last-good state (used when frame_skip omits inference)
        self._last_tracks:  List[TrackedPerson]  = []
        self._last_result:  Optional[BehaviorResult] = None
        self._last_heatmap: Optional[np.ndarray] = None

        self._capture: Optional[VideoCapture] = None

    # ------------------------------------------------------------------
    # Alternate constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(
        cls,
        config: Optional[PipelineConfig] = None,
    ) -> "VideoPipeline":
        """
        Construct a complete pipeline from ``config/settings.py``.

        An optional ``PipelineConfig`` can override individual fields.
        """
        cfg = config or PipelineConfig()

        try:
            from config.settings import get_settings
            s = get_settings()
            v = s.video

            if config is None:
                cfg = PipelineConfig(
                    source              = v.source,
                    frame_width         = v.frame_width,
                    frame_height        = v.frame_height,
                    target_fps          = v.target_fps,
                    frame_buffer_size   = v.frame_buffer_size,
                    output_jpeg_quality = v.output_jpeg_quality,
                    reconnect_attempts  = v.rtsp_reconnect_attempts,
                    reconnect_delay_s   = v.rtsp_reconnect_delay_s,
                )
        except Exception as exc:
            logger.warning(
                "Could not load settings (%s) — using PipelineConfig defaults.", exc
            )

        detector = YOLOv8Detector.from_settings()
        tracker  = DeepSORTTracker.from_settings()
        analyzer = BehaviorAnalyzer.from_settings()
        renderer = FrameRenderer(
            show_heatmap=cfg.show_heatmap,
            show_velocity=cfg.show_velocity,
        )
        analyzer.set_frame_shape(cfg.frame_height, cfg.frame_width)

        logger.info(
            "VideoPipeline constructed from settings — source=%r  %dx%d@%dfps",
            cfg.source, cfg.frame_width, cfg.frame_height, cfg.target_fps,
        )
        return cls(detector, tracker, analyzer, renderer, cfg)

    # ------------------------------------------------------------------
    # Primary async API
    # ------------------------------------------------------------------

    async def run(self) -> AsyncIterator[PipelineFrame]:
        """
        Main async generator.  Yields one ``PipelineFrame`` per processed frame.

        Starts the capture thread, then loops until:
          - the source is exhausted (file playback),
          - the capture thread dies without reconnecting, or
          - ``stop()`` is called from outside.

        Usage::

            async for pf in pipeline.run():
                await ws_manager.broadcast(pf.encode_jpeg())
        """
        self._running = True
        self._start_capture()

        loop = asyncio.get_running_loop()

        try:
            async for buffered_frame in self._frame_producer():
                if not self._running:
                    break

                t_frame_start = time.perf_counter()

                try:
                    pipeline_frame = await self._process_frame(
                        buffered_frame, loop
                    )
                except Exception as exc:
                    self._consecutive_errors += 1
                    logger.error(
                        "Pipeline error on frame %d: %s",
                        self._frame_index, exc, exc_info=True,
                    )
                    if self._consecutive_errors >= self._cfg.max_consecutive_errors:
                        logger.critical(
                            "Max consecutive errors (%d) reached — stopping pipeline.",
                            self._cfg.max_consecutive_errors,
                        )
                        break
                    continue
                else:
                    self._consecutive_errors = 0

                elapsed = time.perf_counter() - t_frame_start
                self._perf.record_frame(elapsed)
                self._frame_index += 1

                yield pipeline_frame

                if self._frame_index % 100 == 0:
                    logger.info(
                        "Pipeline perf [frame %d]: %s",
                        self._frame_index,
                        self._perf.summary(),
                    )

        finally:
            self._shutdown()

    def stop(self) -> None:
        """Signal the pipeline to stop after the current frame."""
        self._running = False
        logger.info("VideoPipeline stop requested.")

    # ------------------------------------------------------------------
    # Frame producer (async generator over the buffer)
    # ------------------------------------------------------------------

    async def _frame_producer(self) -> AsyncIterator[BufferedFrame]:
        """
        Async generator that yields ``BufferedFrame`` objects from the ring
        buffer without blocking the event loop.

        Polls the buffer in a thread executor so ``await`` yields control
        back to the event loop between polls.
        """
        loop     = asyncio.get_running_loop()
        empty_ms = 0

        while self._running:
            buffered = await loop.run_in_executor(
                self._executor,
                lambda: self._buffer.get(timeout=0.05),
            )

            if buffered is None:
                # Buffer temporarily empty — back-pressure
                empty_ms += 50
                if empty_ms >= 2000:
                    # Source is probably dead
                    if not self._capture or not self._capture.is_running:
                        logger.warning(
                            "Capture thread is not running — ending frame producer."
                        )
                        break
                continue

            empty_ms = 0
            yield buffered

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------

    async def _process_frame(
        self,
        buffered: BufferedFrame,
        loop:     asyncio.AbstractEventLoop,
    ) -> PipelineFrame:
        """
        Execute all pipeline stages for a single buffered frame.

        Detection and tracking run in the thread executor so they never
        block the event loop.
        """
        frame       = buffered.frame
        cap_index   = buffered.frame_index
        fidx        = self._frame_index
        should_infer = (fidx % max(self._cfg.frame_skip, 1) == 0)

        # ── Stage 1 — Detection (thread executor) ─────────────────────
        if should_infer:
            t0 = time.perf_counter()
            try:
                detections: List[DetectionResult] = await loop.run_in_executor(
                    self._executor,
                    lambda: self._detector.detect(frame),
                )
            except Exception as exc:
                logger.warning("Detection failed frame %d: %s", fidx, exc)
                detections = []
            self._perf.record_stage("detect", time.perf_counter() - t0)
        else:
            detections = []

        # ── Stage 2 — Tracking (thread executor) ──────────────────────
        if should_infer:
            t0 = time.perf_counter()
            try:
                tracks: List[TrackedPerson] = await loop.run_in_executor(
                    self._executor,
                    lambda: self._tracker.update(detections, frame),
                )
                self._last_tracks = tracks
            except Exception as exc:
                logger.warning("Tracking failed frame %d: %s", fidx, exc)
                tracks = self._last_tracks   # reuse last good
            self._perf.record_stage("track", time.perf_counter() - t0)
        else:
            tracks = self._last_tracks       # reuse last good

        # ── Stage 3 — Behavior Analysis ───────────────────────────────
        if should_infer:
            t0 = time.perf_counter()
            try:
                behavior_result: BehaviorResult = await loop.run_in_executor(
                    self._executor,
                    lambda: self._analyzer.analyze(tracks, frame_index=fidx),
                )
                self._last_result  = behavior_result
                self._last_heatmap = self._extract_heatmap(behavior_result)
            except Exception as exc:
                logger.warning("Behavior analysis failed frame %d: %s", fidx, exc)
                behavior_result = self._last_result or self._null_result(fidx)
            self._perf.record_stage("behavior", time.perf_counter() - t0)
        else:
            behavior_result = self._last_result or self._null_result(fidx)

        # ── Stage 4 — Annotation ──────────────────────────────────────
        t0 = time.perf_counter()
        try:
            annotated = self._renderer.render(
                frame=frame,
                tracks=tracks,
                behavior_result=behavior_result,
                heatmap=self._last_heatmap,
                fps=self._perf.fps,
                frame_index=fidx,
            )
        except Exception as exc:
            logger.warning("Annotation failed frame %d: %s", fidx, exc)
            annotated = frame.copy()
        self._perf.record_stage("annotate", time.perf_counter() - t0)

        # ── Build output ──────────────────────────────────────────────
        processing_ms = sum(
            self._perf.stage_ms(s) for s in ("detect", "track", "behavior", "annotate")
        )

        return PipelineFrame(
            annotated_frame = annotated,
            jpeg_bytes      = None,        # lazy-encode on demand
            tracks          = tracks,
            behavior_result = behavior_result,
            frame_index     = fidx,
            capture_index   = cap_index,
            processing_ms   = processing_ms,
            fps             = self._perf.fps,
            was_skipped     = not should_infer,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _start_capture(self) -> None:
        """Instantiate and start the VideoCapture thread."""
        cap_cfg = CaptureConfig(
            source             = self._cfg.source,
            target_width       = self._cfg.frame_width,
            target_height      = self._cfg.frame_height,
            target_fps         = self._cfg.target_fps,
            reconnect_attempts = self._cfg.reconnect_attempts,
            reconnect_delay_s  = self._cfg.reconnect_delay_s,
        )
        self._capture = VideoCapture(
            source=self._cfg.source,
            buffer=self._buffer,
            config=cap_cfg,
        )
        self._capture.start()
        logger.info("Capture thread started for source %r.", self._cfg.source)

    def _shutdown(self) -> None:
        """Gracefully release all resources."""
        logger.info("Pipeline shutdown initiated …")
        self._running = False

        if self._capture:
            self._capture.stop()

        self._buffer.drain()
        self._executor.shutdown(wait=False)
        self._detector.close()
        self._tracker.reset()
        self._analyzer.reset()

        logger.info(
            "Pipeline shut down. Total frames processed: %d  Final perf: %s",
            self._frame_index,
            self._perf.summary(),
        )

    def _null_result(self, frame_index: int) -> BehaviorResult:
        """Return a safe NORMAL result when no prior result exists."""
        from core.behavior.base_analyzer import FrameFeatures
        return BehaviorResult(
            label        = BehaviorLabel.NORMAL,
            confidence   = 0.0,
            frame_index  = frame_index,
            track_labels = {},
            features     = FrameFeatures.empty(frame_index),
            signals      = ["pipeline_init"],
            elapsed_ms   = 0.0,
        )

    @staticmethod
    def _extract_heatmap(
        result: BehaviorResult,
    ) -> Optional[np.ndarray]:
        """
        Pull the density heatmap out of the behavior result's features.

        The heatmap is stored in ``DensityResult`` inside the analyzer;
        for now we expose it via the crowd_density sub-module if available.
        Returns None if not present.
        """
        # The heatmap is not stored in BehaviorResult directly —
        # it lives in CrowdDensityAnalyzer.  We return None here and
        # let callers inject it from analyzer._density if needed.
        return None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        """Current rolling pipeline FPS."""
        return self._perf.fps

    @property
    def frame_index(self) -> int:
        """Total frames processed since pipeline start."""
        return self._frame_index

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def performance(self) -> Dict:
        """Per-stage latency summary."""
        return self._perf.summary()

    def __repr__(self) -> str:
        return (
            f"VideoPipeline(source={self._cfg.source!r}, "
            f"fps={self.fps}, frame={self._frame_index}, "
            f"running={self._running})"
        )
