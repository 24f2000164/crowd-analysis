"""
core/tracking/deepsort_tracker.py
===================================
Production-ready multi-object tracker built on the DeepSORT algorithm.

Algorithm overview
------------------
DeepSORT (Deep Simple Online and Realtime Tracking) extends the original SORT
tracker by adding an appearance descriptor (re-identification embedding) to
the Kalman-filter + Hungarian-algorithm assignment pipeline:

    1. Kalman Predict   — project each existing track forward one time-step.
    2. Appearance Match — compute cosine distance between stored embeddings
                          and new detection crops; gate with Mahalanobis dist.
    3. IoU Match        — fall-back for tracks / detections not yet matched.
    4. Update           — accepted matches update the Kalman state and gallery;
                          unmatched detections start new Tentative tracks;
                          unmatched tracks age toward deletion.

Responsibilities of this module
---------------------------------
- Accept a list of ``DetectionResult`` objects and a raw BGR frame per tick.
- Assign stable integer IDs that persist across occlusions and re-entries.
- Compute per-track velocity (pixels / frame) using an EMA-smoothed centroid
  history stored in a fixed-size rolling deque.
- Return a list of ``TrackedPerson`` objects (only Confirmed tracks) with the
  canonical output contract:

      [{"id": 3, "bbox": [x1,y1,x2,y2], "velocity": 4.71}, ...]

- Surface all performance metrics and a full track registry snapshot for the
  behaviour analysis and Prometheus layers.

Dependencies
------------
The module delegates the low-level Kalman filter and Hungarian assignment to
the ``deep_sort_realtime`` package (pip install deep-sort-realtime), which is
the most actively maintained pure-Python DeepSORT implementation.

Usage
-----
    from core.tracking.deepsort_tracker import DeepSORTTracker
    from core.detection.yolo_detector   import DetectionResult

    tracker = DeepSORTTracker.from_settings()

    detections: list[DetectionResult] = detector.detect(frame)
    tracks:     list[TrackedPerson]   = tracker.update(detections, frame)

    for t in tracks:
        print(t.id, t.bbox, t.velocity)
        print(t.to_dict())   # {"id": 3, "bbox": [...], "velocity": 4.71}
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from core.detection.yolo_detector import BoundingBox, DetectionResult
from core.tracking.track_state import TrackState

logger = logging.getLogger("crowd_analysis.tracking")


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

# Minimum bounding-box area (px²) — crops smaller than this produce
# uninformative embeddings and are forwarded as detection-only (no re-ID).
_MIN_CROP_AREA_PX2: int = 400

# EMA floor: don't apply smoothing until the track has this many history points.
_MIN_HISTORY_FOR_SMOOTHING: int = 3


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class VelocityVector:
    """
    Per-track velocity at a single instant.

    Attributes
    ----------
    vx : float  — horizontal component (pixels / frame), positive = right.
    vy : float  — vertical component (pixels / frame), positive = down.
    speed : float — Euclidean magnitude.
    direction_deg : float — compass-style angle in [0, 360).
        0° = up, 90° = right, 180° = down, 270° = left.
    """
    vx:            float
    vy:            float
    speed:         float
    direction_deg: float

    @classmethod
    def zero(cls) -> "VelocityVector":
        return cls(vx=0.0, vy=0.0, speed=0.0, direction_deg=0.0)

    @classmethod
    def from_delta(cls, dx: float, dy: float) -> "VelocityVector":
        speed = math.hypot(dx, dy)
        # atan2 measured from +y-axis (down), rotated to compass convention
        angle_rad = math.atan2(dx, -dy)
        direction = math.degrees(angle_rad) % 360.0
        return cls(vx=dx, vy=dy, speed=speed, direction_deg=direction)

    def to_dict(self) -> Dict[str, float]:
        return {
            "vx":            round(self.vx,            3),
            "vy":            round(self.vy,            3),
            "speed":         round(self.speed,         3),
            "direction_deg": round(self.direction_deg, 1),
        }


@dataclass(slots=True)
class TrackedPerson:
    """
    A single confirmed tracked person returned by :class:`DeepSORTTracker`.

    Attributes
    ----------
    id           : int           — stable track identifier (monotonically inc.).
    bbox         : BoundingBox   — current bounding box in absolute pixels.
    velocity     : VelocityVector — smoothed velocity at this frame.
    state        : TrackState    — lifecycle state (always Confirmed in output).
    age          : int           — total frames since track creation.
    hits         : int           — total frames with a matched detection.
    time_since_update : int      — frames since last successful match.
    frame_index  : int           — 0-based counter of the frame that produced
                                   this snapshot.
    """
    id:                 int
    bbox:               BoundingBox
    velocity:           VelocityVector
    state:              TrackState    = TrackState.Confirmed
    age:                int           = 0
    hits:               int           = 0
    time_since_update:  int           = 0
    frame_index:        int           = 0

    # ------------------------------------------------------------------
    # Output contract
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        """
        Return the canonical serialisable representation::

            {
                "id":       3,
                "bbox":     [x1, y1, x2, y2],
                "velocity": 4.71,            ← scalar speed (px/frame)
            }

        Extended metadata is available via ``to_full_dict()``.
        """
        return {
            "id":       self.id,
            "bbox":     self.bbox.as_list(),
            "velocity": round(self.velocity.speed, 4),
        }

    def to_full_dict(self) -> Dict[str, object]:
        """Full representation including velocity components and track metadata."""
        return {
            "id":                self.id,
            "bbox":              self.bbox.as_list(),
            "centroid":          list(self.bbox.centroid),
            "velocity":          self.velocity.to_dict(),
            "state":             self.state.name,
            "age":               self.age,
            "hits":              self.hits,
            "time_since_update": self.time_since_update,
            "frame_index":       self.frame_index,
        }

    def __repr__(self) -> str:
        b = self.bbox
        return (
            f"TrackedPerson(id={self.id}, "
            f"speed={self.velocity.speed:.1f}px/f, "
            f"box=[{b.x1:.0f},{b.y1:.0f},{b.x2:.0f},{b.y2:.0f}], "
            f"age={self.age})"
        )


# ---------------------------------------------------------------------------
# Internal per-track state managed by the tracker
# ---------------------------------------------------------------------------

@dataclass
class _TrackRecord:
    """
    Mutable bookkeeping entry held in the tracker's registry for each active
    track.  Never exposed outside this module.
    """
    track_id:        int
    centroid_history: Deque[Tuple[float, float]]
    smoothed_vx:     float = 0.0
    smoothed_vy:     float = 0.0
    last_state:      TrackState = TrackState.Tentative
    frame_created:   int = 0

    @classmethod
    def create(
        cls,
        track_id: int,
        initial_centroid: Tuple[float, float],
        history_length: int,
        frame_index: int,
    ) -> "_TrackRecord":
        history: Deque[Tuple[float, float]] = deque(maxlen=history_length)
        history.append(initial_centroid)
        return cls(
            track_id=track_id,
            centroid_history=history,
            frame_created=frame_index,
        )


# ---------------------------------------------------------------------------
# Velocity computer
# ---------------------------------------------------------------------------

class _VelocityComputer:
    """
    Stateless helper that computes EMA-smoothed velocity from a centroid deque.

    The EMA is applied to the (dx, dy) delta rather than the raw position so
    that brief detection jitter does not spike the reported speed.
    """

    def __init__(self, alpha: float = 0.3) -> None:
        """
        Parameters
        ----------
        alpha : float
            EMA smoothing factor.  0 = completely frozen (no update),
            1 = no smoothing (raw delta).  Typical range: 0.2–0.4.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self._alpha = alpha

    def compute(self, record: _TrackRecord) -> VelocityVector:
        """
        Derive a smoothed velocity vector from ``record.centroid_history``.

        Returns :meth:`VelocityVector.zero` when the history is too short to
        compute a meaningful delta.
        """
        history = record.centroid_history
        if len(history) < 2:
            return VelocityVector.zero()

        # Raw delta between the two most recent centroids
        cx_prev, cy_prev = history[-2]
        cx_curr, cy_curr = history[-1]
        raw_dx = cx_curr - cx_prev
        raw_dy = cy_curr - cy_prev

        # Apply EMA if we have prior smoothed values and sufficient history
        if len(history) >= _MIN_HISTORY_FOR_SMOOTHING:
            smooth_dx = self._alpha * raw_dx + (1 - self._alpha) * record.smoothed_vx
            smooth_dy = self._alpha * raw_dy + (1 - self._alpha) * record.smoothed_vy
        else:
            smooth_dx = raw_dx
            smooth_dy = raw_dy

        # Mutate the record's smoothed state in-place (caller owns the record)
        record.smoothed_vx = smooth_dx
        record.smoothed_vy = smooth_dy

        return VelocityVector.from_delta(smooth_dx, smooth_dy)


# ---------------------------------------------------------------------------
# DeepSORT Tracker
# ---------------------------------------------------------------------------

class DeepSORTTracker:
    """
    Real-time multi-person tracker based on the DeepSORT algorithm.

    The tracker wraps ``deep_sort_realtime.DeepSort`` and layers on top:

    * Stable per-track centroid history and EMA-smoothed velocity.
    * A full track registry (``_TrackRecord``) that outlives individual
      DeepSORT track objects.
    * Thread-safe update via an internal lock (safe to call from an asyncio
      thread-pool executor).
    * Structured ``TrackedPerson`` output that downstream stages can consume
      without importing DeepSORT internals.

    Parameters
    ----------
    max_age : int
        Frames a track can go without a match before deletion.
    min_hits : int
        Consecutive detections required before Tentative → Confirmed.
    iou_threshold : float
        IoU gate in the Hungarian fallback assignment.
    max_cosine_distance : float
        Appearance distance ceiling for re-identification.
    nn_budget : int | None
        Gallery size per track.  ``None`` = unlimited.
    trajectory_history_length : int
        Rolling centroid deque length used for velocity computation.
    velocity_alpha : float
        EMA smoothing factor for velocity.
    embedder_model : str
        Re-ID backbone passed to ``deep_sort_realtime``.
        Options: ``"mobilenet"`` (fast), ``"clip_RN50"`` (accurate).
    embedder_gpu : bool
        Run the re-ID encoder on GPU.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        max_age:                   int            = 30,
        min_hits:                  int            = 3,
        iou_threshold:             float          = 0.3,
        max_cosine_distance:       float          = 0.4,
        nn_budget:                 Optional[int]  = 100,
        trajectory_history_length: int            = 60,
        velocity_alpha:            float          = 0.3,
        embedder_model:            str            = "mobilenet",
        embedder_gpu:              bool           = False,
    ) -> None:
        self._max_age                   = max_age
        self._min_hits                  = min_hits
        self._iou_threshold             = iou_threshold
        self._max_cosine_distance       = max_cosine_distance
        self._nn_budget                 = nn_budget
        self._trajectory_history_length = trajectory_history_length
        self._embedder_model            = embedder_model
        self._embedder_gpu              = embedder_gpu

        self._frame_index: int = 0
        self._lock                     = Lock()

        # Per-track bookkeeping — keyed by integer track ID
        self._registry: Dict[int, _TrackRecord] = {}

        # Velocity computer (stateless except for config)
        self._velocity_computer = _VelocityComputer(alpha=velocity_alpha)

        # Performance counters
        self._total_update_time_s: float = 0.0
        self._update_count:        int   = 0

        # Initialise the underlying DeepSORT engine
        self._tracker = self._build_tracker()

        logger.info(
            "DeepSORTTracker initialised — max_age=%d  min_hits=%d  "
            "iou=%.2f  max_cos_dist=%.2f  nn_budget=%s  embedder=%s",
            max_age, min_hits, iou_threshold, max_cosine_distance,
            nn_budget, embedder_model,
        )

    # ------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls) -> "DeepSORTTracker":
        """
        Construct a tracker from the application's centralised config.

        Reads ``config/settings.py`` via :func:`get_settings`.
        """
        try:
            from config.settings import get_settings
            cfg = get_settings()
            t   = cfg.tracking
            return cls(
                max_age=t.max_age,
                min_hits=t.min_hits,
                iou_threshold=t.iou_threshold,
                max_cosine_distance=t.max_cosine_distance,
                nn_budget=t.nn_budget,
                trajectory_history_length=t.trajectory_history_length,
                velocity_alpha=t.velocity_smoothing_alpha,
                embedder_gpu=(cfg.device.value == "cuda"),
            )
        except Exception as exc:
            logger.error(
                "Failed to load tracker settings: %s — using defaults.", exc
            )
            return cls()

    # ------------------------------------------------------------------
    # Primary public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections: List[DetectionResult],
        frame: np.ndarray,
    ) -> List[TrackedPerson]:
        """
        Advance the tracker by one frame.

        Runs Kalman predict → appearance matching → IoU fallback matching →
        track state updates, then computes velocity for every confirmed track.

        Parameters
        ----------
        detections : list[DetectionResult]
            Detections from the current frame (may be empty).
        frame : np.ndarray
            The raw BGR frame corresponding to ``detections``.  Used by the
            re-ID encoder to extract appearance embeddings from detection crops.

        Returns
        -------
        list[TrackedPerson]
            All Confirmed tracks, sorted by ascending track ID.  Empty when
            no tracks are confirmed yet (typical during the first few frames).
        """
        if frame is None or frame.ndim != 3:
            logger.warning(
                "Frame %d is invalid — skipping tracker update.", self._frame_index
            )
            self._frame_index += 1
            return []

        t_start = time.perf_counter()

        with self._lock:
            try:
                raw_tracks = self._run_deepsort(detections, frame)
                confirmed  = self._build_output(raw_tracks, frame.shape)
            except Exception as exc:
                logger.error(
                    "Tracker update failed on frame %d: %s",
                    self._frame_index, exc, exc_info=True,
                )
                confirmed = []
            finally:
                self._frame_index += 1

        elapsed = time.perf_counter() - t_start
        self._total_update_time_s += elapsed
        self._update_count        += 1

        if self._frame_index % 100 == 0:
            logger.debug(
                "Tracker perf [frame %d]: avg_latency=%.1f ms  "
                "active_tracks=%d  confirmed=%d",
                self._frame_index,
                self.avg_update_latency_ms,
                len(self._registry),
                len(confirmed),
            )

        return confirmed

    def reset(self) -> None:
        """
        Reset all track state.

        Call between video clips or when switching camera feeds to prevent
        stale tracks from polluting a new scene.
        """
        with self._lock:
            self._tracker  = self._build_tracker()
            self._registry.clear()
            self._frame_index         = 0
            self._total_update_time_s = 0.0
            self._update_count        = 0
        logger.info("DeepSORTTracker reset.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def frame_index(self) -> int:
        """0-based index of the *next* frame to be processed."""
        return self._frame_index

    @property
    def active_track_count(self) -> int:
        """Total tracks in the registry (all states, including Lost)."""
        return len(self._registry)

    @property
    def avg_update_latency_ms(self) -> float:
        """Rolling average tracker update latency in milliseconds."""
        if self._update_count == 0:
            return 0.0
        return (self._total_update_time_s / self._update_count) * 1_000

    def get_track_history(self, track_id: int) -> List[Tuple[float, float]]:
        """
        Return the centroid trajectory for a given track ID.

        Parameters
        ----------
        track_id : int
            The integer track ID.

        Returns
        -------
        list[tuple[float, float]]
            Ordered centroid positions ``[(cx, cy), ...]`` oldest → newest.
            Empty list if the ID is unknown.
        """
        record = self._registry.get(track_id)
        return list(record.centroid_history) if record else []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_tracker(self):
        """
        Instantiate the underlying ``deep_sort_realtime.DeepSort`` engine.

        Raises
        ------
        RuntimeError
            If the package is not installed.
        """
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ImportError as exc:
            raise RuntimeError(
                "deep-sort-realtime is not installed.  "
                "Run: pip install deep-sort-realtime"
            ) from exc

         

        return DeepSort(
            max_age=self._max_age,
            n_init=self._min_hits,
            max_iou_distance=self._iou_threshold,   # ← correct parameter
            nms_max_overlap=1.0,                    # ← disable NMS overlap filtering
            max_cosine_distance=self._max_cosine_distance,
            nn_budget=self._nn_budget,
            embedder=self._embedder_model,
            half=False,
            bgr=True,
            embedder_gpu=self._embedder_gpu,
        )

    def _run_deepsort(
        self,
        detections: List[DetectionResult],
        frame: np.ndarray,
    ):
        """
        Convert ``DetectionResult`` objects to the format expected by
        ``deep_sort_realtime`` and run one update tick.

        ``deep_sort_realtime`` expects detections as::

            [ ([left, top, w, h], confidence, class_label), ... ]
        """
        ds_detections = []
        for det in detections:
            tlwh = det.bbox.as_tlwh()        # [x1, y1, w, h]
            # Reject degenerate crops that confuse the embedder
            if det.bbox.area < _MIN_CROP_AREA_PX2:
                continue
            ds_detections.append((tlwh, det.confidence, "person"))

        return self._tracker.update_tracks(ds_detections, frame=frame)

    def _build_output(
        self,
        raw_tracks,
        frame_shape: Tuple[int, ...],
    ) -> List[TrackedPerson]:
        """
        Convert raw DeepSORT track objects into ``TrackedPerson`` instances.

        Steps:
        1. Determine the lifecycle state of each raw track.
        2. Upsert the track's ``_TrackRecord`` in ``self._registry``.
        3. Append the latest centroid to the record's history deque.
        4. Compute EMA velocity via ``_VelocityComputer``.
        5. Prune deleted tracks from the registry.
        6. Return only Confirmed tracks.
        """
        frame_h, frame_w = frame_shape[:2]
        output: List[TrackedPerson] = []
        active_ids: set[int] = set()

        for raw in raw_tracks:
            track_id = int(raw.track_id)
            active_ids.add(track_id)

            # ── Map DeepSORT internal state → TrackState ───────────────
            state = self._map_state(raw)

            # ── Bounding box ───────────────────────────────────────────
            try:
                ltrb = raw.to_ltrb()          # [left, top, right, bottom]
            except Exception:
                logger.debug("Track %d: could not get bbox — skipping.", track_id)
                continue

            bbox = BoundingBox(
                x1=float(ltrb[0]),
                y1=float(ltrb[1]),
                x2=float(ltrb[2]),
                y2=float(ltrb[3]),
            ).clamp(frame_w, frame_h)

            if bbox.area <= 0:
                continue

            centroid = bbox.centroid

            # ── Registry upsert ────────────────────────────────────────
            if track_id not in self._registry:
                self._registry[track_id] = _TrackRecord.create(
                    track_id=track_id,
                    initial_centroid=centroid,
                    history_length=self._trajectory_history_length,
                    frame_index=self._frame_index,
                )
                logger.debug(
                    "New track registered: id=%d  frame=%d",
                    track_id, self._frame_index,
                )
            else:
                self._registry[track_id].centroid_history.append(centroid)

            record = self._registry[track_id]
            record.last_state = state

            # ── Velocity ───────────────────────────────────────────────
            velocity = self._velocity_computer.compute(record)

            # ── Only surface Confirmed tracks downstream ───────────────
            if state != TrackState.Confirmed:
                continue

            output.append(
                TrackedPerson(
                    id=track_id,
                    bbox=bbox,
                    velocity=velocity,
                    state=state,
                    age=getattr(raw, "age", 0),
                    hits=getattr(raw, "hits", 0),
                    time_since_update=getattr(raw, "time_since_update", 0),
                    frame_index=self._frame_index,
                )
            )

        # ── Prune deleted tracks from registry ─────────────────────────
        deleted_ids = [
            tid for tid, rec in self._registry.items()
            if rec.last_state == TrackState.Deleted
            and tid not in active_ids
        ]
        for tid in deleted_ids:
            del self._registry[tid]
            logger.debug("Track %d pruned from registry.", tid)

        return sorted(output, key=lambda t: t.id)

    @staticmethod
    def _map_state(raw_track) -> TrackState:
        state_val = getattr(raw_track, "state", 1)

    # Handle integer states (deep_sort_realtime uses IntEnum: 1=Tentative, 2=Confirmed, 3=Deleted)
        if isinstance(state_val, int):
            try:
                return TrackState(state_val)
            except ValueError:
                return TrackState.Tentative

    # Handle string states (some versions use strings)
        if hasattr(state_val, "name"):
             state_val = state_val.name

        mapping = {
           "Tentative": TrackState.Tentative,
            "Confirmed": TrackState.Confirmed,
             "Deleted":   TrackState.Deleted,
          }
        state = mapping.get(str(state_val), TrackState.Tentative)

    # Promote Confirmed → Lost when genuinely unmatched
        if state == TrackState.Confirmed and getattr(raw_track, "time_since_update", 0) > 1:
            state = TrackState.Lost

        return state

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "DeepSORTTracker":
        return self

    def __exit__(self, *_) -> None:
        self.reset()

    def __repr__(self) -> str:
        return (
            f"DeepSORTTracker("
            f"max_age={self._max_age}, "
            f"min_hits={self._min_hits}, "
            f"iou={self._iou_threshold}, "
            f"active_tracks={self.active_track_count})"
        )
