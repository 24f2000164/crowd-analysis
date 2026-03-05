"""
core/behavior/trajectory_features.py
======================================
Feature extraction module for ML-based crowd behavior prediction.

Consumes trajectory data from TrajectoryStore and DeepSORT tracks,
computes a 9-dimensional feature vector over a sliding window of frames,
and returns it ready for model inference or dataset export.

Features
--------
1.  velocity_mean          — average speed across all tracks in window
2.  velocity_variance      — variance of per-track mean speeds
3.  acceleration_mean      — mean absolute acceleration
4.  acceleration_spikes    — fraction of frames with abrupt speed jumps
5.  direction_entropy      — Shannon entropy of heading distribution (0=aligned)
6.  crowd_density          — persons per 10 000 px²
7.  density_change_rate    — Δdensity / Δframe (positive = crowd growing)
8.  trajectory_dispersion  — mean pairwise centroid distance (px)
9.  track_collision_rate   — fraction of frame-pairs where tracks are < threshold

Usage
-----
    extractor = TrajectoryFeatureExtractor(frame_shape=(720, 1280))
    extractor.update(tracks, frame_index=n)          # call every frame
    vec = extractor.compute_features()               # returns FeatureVector
    if vec:
        print(vec.to_dict())
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("crowd_analysis.behavior.trajectory_features")

# ── Constants ──────────────────────────────────────────────────────────────
_FEATURE_NAMES: List[str] = [
    "velocity_mean",
    "velocity_variance",
    "acceleration_mean",
    "acceleration_spikes",
    "direction_entropy",
    "crowd_density",
    "density_change_rate",
    "trajectory_dispersion",
    "track_collision_rate",
]

_ACCELERATION_SPIKE_THRESHOLD: float = 8.0   # px/frame²
_COLLISION_DISTANCE_PX: float         = 80.0  # px
_DIRECTION_BINS: int                  = 8      # compass octants for entropy


# ── Output dataclass ───────────────────────────────────────────────────────

@dataclass
class FeatureVector:
    """
    9-dimensional feature snapshot for one sliding window.

    All values are floats; safe to pass directly to sklearn/XGBoost.
    """
    velocity_mean:         float
    velocity_variance:     float
    acceleration_mean:     float
    acceleration_spikes:   float
    direction_entropy:     float
    crowd_density:         float
    density_change_rate:   float
    trajectory_dispersion: float
    track_collision_rate:  float

    # ── bookkeeping (not fed to model) ────────────────────────────────
    frame_window_start: int = 0
    frame_window_end:   int = 0
    track_count:        int = 0

    def to_array(self) -> np.ndarray:
        """Return feature values as a float32 numpy array (model input)."""
        return np.array([
            self.velocity_mean,
            self.velocity_variance,
            self.acceleration_mean,
            self.acceleration_spikes,
            self.direction_entropy,
            self.crowd_density,
            self.density_change_rate,
            self.trajectory_dispersion,
            self.track_collision_rate,
        ], dtype=np.float32)

    def to_dict(self) -> Dict[str, float]:
        """Return all 9 feature values as a plain dict."""
        return {k: float(getattr(self, k)) for k in _FEATURE_NAMES}

    @staticmethod
    def feature_names() -> List[str]:
        return list(_FEATURE_NAMES)

    def __repr__(self) -> str:
        vals = ", ".join(f"{k}={getattr(self, k):.3f}" for k in _FEATURE_NAMES)
        return f"FeatureVector({vals})"


# ── Per-frame snapshot stored in the rolling window ───────────────────────

@dataclass
class _FrameSnapshot:
    """Lightweight per-frame record held in the sliding window deque."""
    frame_index: int
    track_count: int
    centroids:   List[Tuple[float, float]]   # (cx, cy) per confirmed track
    speeds:      List[float]                  # px/frame per confirmed track
    directions:  List[float]                  # degrees per confirmed track
    accels:      List[float]                  # |Δspeed| per confirmed track


# ── Main extractor ─────────────────────────────────────────────────────────

class TrajectoryFeatureExtractor:
    """
    Stateful feature extractor that maintains a sliding window of frame
    snapshots and computes the 9 ML features on demand.

    Parameters
    ----------
    frame_shape    : (height, width) of the video frame in pixels.
    window_size    : number of consecutive frames to aggregate features over.
    min_tracks     : minimum confirmed tracks required to yield a valid vector.
    """

    def __init__(
        self,
        frame_shape: Tuple[int, int] = (720, 1280),
        window_size: int             = 30,
        min_tracks:  int             = 2,
    ) -> None:
        self._frame_h, self._frame_w = frame_shape
        self._window_size            = window_size
        self._min_tracks             = min_tracks

        self._window: Deque[_FrameSnapshot] = deque(maxlen=window_size)

        # per-track speed from the previous frame (for acceleration)
        self._prev_speeds: Dict[int, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        tracks: list,          # List[TrackedPerson] — avoids circular import
        frame_index: int = 0,
    ) -> None:
        """
        Ingest the confirmed tracks for one frame into the sliding window.

        Call this every pipeline frame, even when tracks is empty.
        """
        centroids:  List[Tuple[float, float]] = []
        speeds:     List[float]               = []
        directions: List[float]               = []
        accels:     List[float]               = []

        for t in tracks:
            cx, cy    = t.bbox.centroid
            speed     = t.velocity.speed
            direction = t.velocity.direction_deg
            prev_spd  = self._prev_speeds.get(t.id, speed)
            accel     = abs(speed - prev_spd)

            centroids.append((cx, cy))
            speeds.append(speed)
            directions.append(direction)
            accels.append(accel)

            self._prev_speeds[t.id] = speed

        # Evict tracks that are no longer active
        active_ids = {t.id for t in tracks}
        self._prev_speeds = {
            tid: s for tid, s in self._prev_speeds.items()
            if tid in active_ids
        }

        snapshot = _FrameSnapshot(
            frame_index=frame_index,
            track_count=len(tracks),
            centroids=centroids,
            speeds=speeds,
            directions=directions,
            accels=accels,
        )
        self._window.append(snapshot)

        logger.debug(
            "FeatureExtractor updated — frame=%d  tracks=%d  window=%d/%d",
            frame_index, len(tracks), len(self._window), self._window_size,
        )

    def compute_features(self) -> Optional[FeatureVector]:
        """
        Compute the 9-feature vector from the current sliding window.

        Returns None when the window is too short or has too few tracks.
        """
        if len(self._window) < max(2, self._window_size // 2):
            logger.debug("Window too short (%d) — skipping.", len(self._window))
            return None

        all_speeds:    List[float]                  = []
        all_accels:    List[float]                  = []
        all_directions: List[float]                 = []
        all_centroids: List[Tuple[float, float]]    = []
        spike_frames:  int                          = 0
        total_frames:  int                          = len(self._window)
        density_series: List[float]                 = []
        collision_pairs: int                        = 0
        total_pairs:    int                         = 0

        for snap in self._window:
            if snap.track_count < 1:
                density_series.append(0.0)
                continue

            all_speeds.extend(snap.speeds)
            all_accels.extend(snap.accels)
            all_directions.extend(snap.directions)
            all_centroids.extend(snap.centroids)

            # acceleration spike: any track exceeds threshold in this frame
            if any(a >= _ACCELERATION_SPIKE_THRESHOLD for a in snap.accels):
                spike_frames += 1

            # per-frame density
            density_series.append(self._density(snap.track_count))

            # collision pairs
            n = len(snap.centroids)
            for i in range(n):
                for j in range(i + 1, n):
                    dist = math.hypot(
                        snap.centroids[i][0] - snap.centroids[j][0],
                        snap.centroids[i][1] - snap.centroids[j][1],
                    )
                    total_pairs  += 1
                    if dist < _COLLISION_DISTANCE_PX:
                        collision_pairs += 1

        if not all_speeds:
            return None

        # ── compute each feature ──────────────────────────────────────
        vel_arr   = np.array(all_speeds, dtype=np.float32)
        vel_mean  = float(vel_arr.mean())
        vel_var   = float(vel_arr.var())

        acc_mean  = float(np.mean(all_accels)) if all_accels else 0.0
        acc_spikes = spike_frames / total_frames if total_frames > 0 else 0.0

        dir_entropy = self._direction_entropy(all_directions)

        crowd_density = float(np.mean(density_series)) if density_series else 0.0
        density_change = self._density_change_rate(density_series)

        traj_disp = self._trajectory_dispersion(all_centroids)

        collision_rate = (
            collision_pairs / total_pairs if total_pairs > 0 else 0.0
        )

        first_frame = self._window[0].frame_index
        last_frame  = self._window[-1].frame_index
        avg_tracks  = int(
            sum(s.track_count for s in self._window) / len(self._window)
        )

        vec = FeatureVector(
            velocity_mean         = vel_mean,
            velocity_variance     = vel_var,
            acceleration_mean     = acc_mean,
            acceleration_spikes   = acc_spikes,
            direction_entropy     = dir_entropy,
            crowd_density         = crowd_density,
            density_change_rate   = density_change,
            trajectory_dispersion = traj_disp,
            track_collision_rate  = collision_rate,
            frame_window_start    = first_frame,
            frame_window_end      = last_frame,
            track_count           = avg_tracks,
        )

        logger.debug("FeatureVector computed: %s", vec)
        return vec

    def reset(self) -> None:
        """Clear all state — call between scenes or camera switches."""
        self._window.clear()
        self._prev_speeds.clear()

    def set_frame_shape(self, height: int, width: int) -> None:
        self._frame_h = height
        self._frame_w = width

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _density(self, track_count: int) -> float:
        """Persons per 10 000 px²."""
        area = max(self._frame_h * self._frame_w, 1)
        return track_count / area * 10_000

    @staticmethod
    def _direction_entropy(directions: List[float]) -> float:
        """
        Shannon entropy of the heading distribution binned into octants.

        0.0 = everyone moving in the same direction.
        ~3.0 = maximum disorder (uniform distribution over 8 bins).
        """
        if not directions:
            return 0.0
        bins = np.zeros(_DIRECTION_BINS, dtype=np.float32)
        for d in directions:
            idx = int(d / (360.0 / _DIRECTION_BINS)) % _DIRECTION_BINS
            bins[idx] += 1.0
        total = bins.sum()
        if total == 0:
            return 0.0
        probs = bins / total
        entropy = -float(np.sum(probs[probs > 0] * np.log2(probs[probs > 0])))
        return entropy

    @staticmethod
    def _density_change_rate(series: List[float]) -> float:
        """
        Linear slope of the density time-series (persons/10k px² per frame).
        Positive = crowd growing, negative = dispersing.
        """
        n = len(series)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=np.float32)
        y = np.array(series, dtype=np.float32)
        # closed-form slope via least-squares
        xm, ym = x.mean(), y.mean()
        denom = float(((x - xm) ** 2).sum())
        if denom == 0:
            return 0.0
        return float(((x - xm) * (y - ym)).sum() / denom)

    @staticmethod
    def _trajectory_dispersion(
        centroids: List[Tuple[float, float]],
    ) -> float:
        """
        Mean pairwise Euclidean distance between all centroids (px).
        High value → crowd spreading out; low value → tightly clustered.
        """
        n = len(centroids)
        if n < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += math.hypot(
                    centroids[i][0] - centroids[j][0],
                    centroids[i][1] - centroids[j][1],
                )
                count += 1
        return total / count if count > 0 else 0.0