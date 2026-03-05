"""
core/detection/yolo_detector.py
================================
Production-ready YOLOv8 person detection module.

Responsibilities
----------------
- Load and warm-up a YOLOv8 model on the best available compute device.
- Run per-frame inference, filtering results to the COCO "person" class only.
- Return a list of normalised DetectionResult objects containing bounding
  boxes in [x1, y1, x2, y2] absolute-pixel format and confidence scores.
- Expose performance metrics (FPS, latency) for Prometheus / logging.
- Handle all recoverable errors internally; re-raise only on unrecoverable
  initialisation failures so the pipeline can react appropriately.

Usage
-----
    from core.detection.yolo_detector import YOLOv8Detector, DetectionResult

    detector = YOLOv8Detector.from_settings()   # uses config/settings.py
    # — or —
    detector = YOLOv8Detector(weights="models/yolov8n.pt", device="cuda")

    results: list[DetectionResult] = detector.detect(frame)   # numpy BGR frame
    for r in results:
        print(r.bbox, r.confidence)

Output contract
---------------
    [
        {"bbox": [x1, y1, x2, y2], "confidence": 0.92},
        ...
    ]

    Available as:  [r.to_dict() for r in results]
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger("crowd_analysis.detection")

# ---------------------------------------------------------------------------
# COCO class index for "person"
# ---------------------------------------------------------------------------
_PERSON_CLASS_ID: int = 0


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class BoundingBox:
    """
    Axis-aligned bounding box in absolute pixel coordinates.

    Attributes
    ----------
    x1, y1 : float
        Top-left corner (origin = top-left of frame).
    x2, y2 : float
        Bottom-right corner.
    """
    x1: float
    y1: float
    x2: float
    y2: float

    # ------------------------------------------------------------------
    # Derived geometry — computed on demand, never stored
    # ------------------------------------------------------------------

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def centroid(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    def as_list(self) -> List[float]:
        """Return [x1, y1, x2, y2] — the canonical output format."""
        return [self.x1, self.y1, self.x2, self.y2]

    def as_tlwh(self) -> List[float]:
        """Return [top, left, width, height] — used by DeepSORT."""
        return [self.x1, self.y1, self.width, self.height]

    def clamp(self, frame_width: int, frame_height: int) -> "BoundingBox":
        """Clip coordinates to the frame boundary."""
        return BoundingBox(
            x1=max(0.0, min(self.x1, frame_width)),
            y1=max(0.0, min(self.y1, frame_height)),
            x2=max(0.0, min(self.x2, frame_width)),
            y2=max(0.0, min(self.y2, frame_height)),
        )


@dataclass(slots=True, frozen=True)
class DetectionResult:
    """
    A single person detection returned by the detector.

    Attributes
    ----------
    bbox        : BoundingBox  — absolute pixel coordinates [x1,y1,x2,y2].
    confidence  : float        — detector confidence in [0, 1].
    class_id    : int          — always 0 (COCO person) for this detector.
    frame_index : int          — 0-based frame counter set by the detector.
    """
    bbox:        BoundingBox
    confidence:  float
    class_id:    int  = _PERSON_CLASS_ID
    frame_index: int  = 0

    def to_dict(self) -> Dict[str, object]:
        """
        Serialise to the canonical output contract::

            {"bbox": [x1, y1, x2, y2], "confidence": 0.92}
        """
        return {
            "bbox":       self.bbox.as_list(),
            "confidence": round(float(self.confidence), 4),
        }

    def __repr__(self) -> str:
        b = self.bbox
        return (
            f"DetectionResult(conf={self.confidence:.2f}, "
            f"box=[{b.x1:.0f},{b.y1:.0f},{b.x2:.0f},{b.y2:.0f}])"
        )


# ---------------------------------------------------------------------------
# Performance tracker (rolling window statistics)
# ---------------------------------------------------------------------------

@dataclass
class _PerformanceTracker:
    """
    Maintains a rolling window of per-frame inference latencies and computes
    live FPS and average latency without storing unbounded history.
    """
    window_size: int = 60
    _latencies:  Deque[float] = field(default_factory=lambda: deque(maxlen=60))
    _lock:       Lock         = field(default_factory=Lock)

    def record(self, latency_s: float) -> None:
        with self._lock:
            self._latencies.append(latency_s)

    @property
    def avg_latency_ms(self) -> float:
        with self._lock:
            if not self._latencies:
                return 0.0
            return (sum(self._latencies) / len(self._latencies)) * 1_000

    @property
    def fps(self) -> float:
        avg = self.avg_latency_ms
        return round(1_000 / avg, 1) if avg > 0 else 0.0

    def summary(self) -> Dict[str, float]:
        return {"avg_latency_ms": round(self.avg_latency_ms, 2), "fps": self.fps}


# ---------------------------------------------------------------------------
# YOLOv8 Detector
# ---------------------------------------------------------------------------

class YOLOv8Detector:
    """
    YOLOv8-based person detector optimised for real-time video streams.

    The detector is intentionally *stateless* with respect to video frames —
    each call to :meth:`detect` is fully independent — so it is safe to call
    from multiple async coroutines as long as PyTorch model inference is not
    parallelised across threads on the same device.

    Parameters
    ----------
    weights : str | Path
        Path to a YOLOv8 ``.pt`` checkpoint, or a model identifier that
        ``ultralytics`` can resolve automatically (e.g. ``"yolov8n.pt"``).
    device : str
        PyTorch device string: ``"cpu"``, ``"cuda"``, ``"cuda:0"``,
        ``"mps"``.  Pass ``"auto"`` to let the detector choose the best
        available device.
    confidence_threshold : float
        Detections below this score are discarded before NMS.
    iou_threshold : float
        NMS IoU gate.
    input_size : int
        Square inference resolution (must be a multiple of 32).
    half_precision : bool
        Use FP16 inference.  Only effective on CUDA devices; silently
        downgraded to FP32 on CPU / MPS.
    max_detections : int
        Hard cap on boxes returned per frame.
    warmup_frames : int
        Number of dummy forward passes run during initialisation so the
        first real inference does not incur JIT compilation overhead.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        weights:              str | Path = "models/yolov8n.pt",
        device:               str        = "auto",
        confidence_threshold: float      = 0.45,
        iou_threshold:        float      = 0.45,
        input_size:           int        = 416,
        half_precision:       bool       = True,
        max_detections:       int        = 300,
        warmup_frames:        int        = 2,
    ) -> None:
        self._weights              = Path(weights)
        self._conf_threshold       = confidence_threshold
        self._iou_threshold        = iou_threshold
        self._input_size           = input_size
        self._max_detections       = max_detections
        self._warmup_frames        = warmup_frames
        self._frame_counter:  int  = 0
        self._perf                 = _PerformanceTracker()
        self._inference_lock       = Lock()

        # Resolve device
        self._device = self._resolve_device(device)

        # Determine whether FP16 is actually feasible on this device
        self._half = half_precision and self._device.type == "cuda"
        if half_precision and not self._half:
            logger.info(
                "FP16 requested but device is %s — falling back to FP32.",
                self._device,
            )

        # Load model — this is the only operation that can raise on init
        self._model = self._load_model()

        # Pre-allocate a reusable dummy tensor for warm-up
        self._warmup()

    # ------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls) -> "YOLOv8Detector":
        """
        Construct a detector from the application's centralised config.

        Reads ``config/settings.py`` via :func:`get_settings` so callers
        never need to repeat configuration values.
        """
        try:
            from config.settings import get_settings
            cfg = get_settings()
            det = cfg.detection
            return cls(
                weights=cfg.models.yolo_weights,
                device=cfg.device.value,
                confidence_threshold=det.confidence_threshold,
                iou_threshold=det.iou_threshold,
                input_size=det.input_image_size,
                half_precision=det.half_precision,
                max_detections=det.max_detections,
            )
        except Exception as exc:
            logger.error("Failed to load settings: %s.  Using defaults.", exc)
            return cls()

    # ------------------------------------------------------------------
    # Primary public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Run person detection on a single BGR video frame.

        Parameters
        ----------
        frame : np.ndarray
            A ``(H, W, 3)`` uint8 array in **BGR** colour order as returned
            by ``cv2.VideoCapture.read()``.

        Returns
        -------
        list[DetectionResult]
            Detections sorted by descending confidence.  Empty list when no
            persons are found or inference fails.

        Raises
        ------
        ValueError
            If ``frame`` is ``None`` or has an unexpected shape.
        """
        self._validate_frame(frame)

        t_start = time.perf_counter()

        try:
            results = self._run_inference(frame)
            detections = self._parse_results(results, frame.shape)
        except torch.cuda.OutOfMemoryError:
            logger.error(
                "CUDA OOM on frame %d — clearing cache and skipping frame.",
                self._frame_counter,
            )
            torch.cuda.empty_cache()
            return []
        except Exception as exc:
            logger.error(
                "Inference error on frame %d: %s",
                self._frame_counter,
                exc,
                exc_info=True,
            )
            return []
        finally:
            # Always increment counter so downstream consumers have a
            # consistent reference even on error frames
            self._frame_counter += 1

        latency = time.perf_counter() - t_start
        self._perf.record(latency)

        if self._frame_counter % 100 == 0:
            logger.debug(
                "Detector perf [frame %d]: %s | detections=%d",
                self._frame_counter,
                self._perf.summary(),
                len(detections),
            )

        return detections

    def detect_batch(
        self, frames: Sequence[np.ndarray]
    ) -> List[List[DetectionResult]]:
        """
        Detect persons in a batch of frames.

        Preferred over calling :meth:`detect` in a loop when processing
        pre-recorded footage or when the GPU has headroom for larger batches.

        Parameters
        ----------
        frames : sequence of np.ndarray
            BGR frames, all with the same shape.

        Returns
        -------
        list[list[DetectionResult]]
            One detection list per input frame, in the same order.
        """
        if not frames:
            return []

        for i, f in enumerate(frames):
            self._validate_frame(f, label=f"batch[{i}]")

        t_start = time.perf_counter()

        try:
            with self._inference_lock:
                results = self._model.predict(
                    source=list(frames),
                    imgsz=self._input_size,
                    conf=self._conf_threshold,
                    iou=self._iou_threshold,
                    classes=[_PERSON_CLASS_ID],
                    max_det=self._max_detections,
                    half=self._half,
                    device=self._device,
                    verbose=False,
                    stream=False,
                )
        except Exception as exc:
            logger.error("Batch inference error: %s", exc, exc_info=True)
            return [[] for _ in frames]

        output: List[List[DetectionResult]] = []
        for frame, result in zip(frames, results):
            output.append(self._parse_results([result], frame.shape))

        self._perf.record((time.perf_counter() - t_start) / len(frames))
        return output

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """The PyTorch device on which the model is loaded."""
        return self._device

    @property
    def is_half_precision(self) -> bool:
        """True when running FP16 inference."""
        return self._half

    @property
    def frame_count(self) -> int:
        """Total frames processed since initialisation."""
        return self._frame_counter

    @property
    def performance(self) -> Dict[str, float]:
        """Rolling-window performance statistics."""
        return self._perf.summary()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """
        Resolve a device string to a ``torch.device``.

        ``"auto"`` selects CUDA → MPS → CPU in priority order.
        """
        if device == "auto":
            if torch.cuda.is_available():
                resolved = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                resolved = "mps"
            else:
                resolved = "cpu"
            logger.info("Device auto-selected: %s", resolved)
            return torch.device(resolved)

        d = torch.device(device)
        if d.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable — falling back to CPU.")
            return torch.device("cpu")
        if d.type == "mps" and not (
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        ):
            logger.warning("MPS requested but unavailable — falling back to CPU.")
            return torch.device("cpu")

        logger.info("Using device: %s", d)
        return d

    def _load_model(self):
        """
        Load and configure the YOLOv8 model.

        Raises
        ------
        RuntimeError
            If the weights file is missing or the model cannot be loaded.
        """
        try:
            from ultralytics import YOLO  # deferred import — heavy module
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics package is not installed.  "
                "Run: pip install ultralytics"
            ) from exc

        weights_path = str(self._weights)

        if not Path(weights_path).exists():
            logger.warning(
                "Weights not found at '%s' — ultralytics will attempt to "
                "download them automatically.",
                weights_path,
            )

        logger.info("Loading YOLOv8 model from '%s' …", weights_path)
        try:
            model = YOLO(weights_path)
            model.to(self._device)
            if self._half:
                model.model.half()
            logger.info(
                "Model loaded successfully.  "
                "Device=%s  FP16=%s  InputSize=%d",
                self._device,
                self._half,
                self._input_size,
            )
            return model
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load YOLOv8 model from '{weights_path}': {exc}"
            ) from exc

    def _warmup(self) -> None:
        """
        Run dummy forward passes to trigger JIT compilation and CUDA kernel
        loading before real video frames arrive.
        """
        if self._warmup_frames <= 0:
            return

        logger.debug("Warming up model with %d dummy frame(s) …", self._warmup_frames)
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(self._warmup_frames):
            try:
                self._run_inference(dummy)
            except Exception as exc:
                logger.warning("Warm-up pass failed (non-fatal): %s", exc)
        logger.debug("Warm-up complete.")

    def _run_inference(self, frame: np.ndarray):
        """
        Execute a single forward pass through the model.

        Uses a threading lock so that a shared detector instance can safely
        be called from the asyncio event loop while a background thread may
        also reference it (e.g. a health-check endpoint).
        """
        with self._inference_lock:
            return self._model.predict(
                source=frame,
                imgsz=self._input_size,
                conf=self._conf_threshold,
                iou=self._iou_threshold,
                classes=[_PERSON_CLASS_ID],
                max_det=self._max_detections,
                half=self._half,
                device=self._device,
                verbose=False,      # silence ultralytics per-frame stdout logs
                stream=False,
            )

    def _parse_results(
        self,
        results,
        frame_shape: Tuple[int, ...],
    ) -> List[DetectionResult]:
        """
        Convert raw ultralytics ``Results`` objects into :class:`DetectionResult`.

        Parameters
        ----------
        results
            The list returned by ``model.predict()``.
        frame_shape
            ``(H, W, C)`` shape of the source frame, used to clamp boxes.

        Returns
        -------
        list[DetectionResult]
            Sorted by descending confidence score.
        """
        frame_h, frame_w = frame_shape[:2]
        detections: List[DetectionResult] = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            # ultralytics stores coords as xyxy tensors on the inference device
            xyxy_tensor  = boxes.xyxy.cpu().float()
            conf_tensor  = boxes.conf.cpu().float()
            class_tensor = boxes.cls.cpu().int()

            for xyxy, conf, cls_id in zip(xyxy_tensor, conf_tensor, class_tensor):
                # Redundant guard: model is already filtered to class 0, but
                # we re-check here to make the function self-contained.
                if int(cls_id) != _PERSON_CLASS_ID:
                    continue

                x1, y1, x2, y2 = xyxy.tolist()
                bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2).clamp(
                    frame_w, frame_h
                )

                # Skip degenerate boxes that collapsed after clamping
                if bbox.area <= 0:
                    continue

                detections.append(
                    DetectionResult(
                        bbox=bbox,
                        confidence=float(conf),
                        class_id=_PERSON_CLASS_ID,
                        frame_index=self._frame_counter,
                    )
                )

        # Return highest-confidence detections first
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    @staticmethod
    def _validate_frame(frame: Optional[np.ndarray], label: str = "frame") -> None:
        """
        Raise :exc:`ValueError` if the frame is unsuitable for inference.
        """
        if frame is None:
            raise ValueError(f"{label} is None.")
        if not isinstance(frame, np.ndarray):
            raise ValueError(
                f"{label} must be a numpy ndarray, got {type(frame).__name__}."
            )
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"{label} must have shape (H, W, 3), got {frame.shape}."
            )
        if frame.dtype != np.uint8:
            raise ValueError(
                f"{label} must be uint8, got {frame.dtype}."
            )

    # ------------------------------------------------------------------
    # Context-manager support — ensures the model is released on exit
    # ------------------------------------------------------------------

    def __enter__(self) -> "YOLOv8Detector":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        """Release model resources and clear the CUDA memory cache."""
        try:
            del self._model
            if self._device.type == "cuda":
                torch.cuda.empty_cache()
            logger.info("YOLOv8Detector closed and resources released.")
        except Exception as exc:
            logger.warning("Error during detector cleanup: %s", exc)

    def __repr__(self) -> str:
        return (
            f"YOLOv8Detector("
            f"weights='{self._weights}', "
            f"device='{self._device}', "
            f"conf={self._conf_threshold}, "
            f"iou={self._iou_threshold}, "
            f"fp16={self._half})"
        )
