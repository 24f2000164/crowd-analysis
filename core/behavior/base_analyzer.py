"""
core/behavior/base_analyzer.py
================================
Shared types, enumerations, and the abstract base class for all behavior
analysis components in the crowd analysis pipeline.

All other modules in ``core/behavior/`` import from here — never from each
other — to prevent circular imports.

Exports
-------
    BehaviorLabel       — canonical behavior taxonomy (string enum)
    TrackFeatures       — per-track feature snapshot for one frame
    FrameFeatures       — crowd-level feature snapshot for one frame
    BehaviorResult      — final output of one analysis tick
    BehaviorThresholds  — all configurable numeric thresholds
    BaseBehaviorClassifier — ABC that every classifier must implement
"""

from __future__ import annotations

import math
import statistics
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, unique
from typing import Dict, List, Optional, Tuple


# ============================================================================
# Enumerations
# ============================================================================

@unique
class BehaviorLabel(str, Enum):
    """
    Canonical crowd behavior taxonomy.

    String enum so labels serialise directly to JSON without extra conversion::

        BehaviorLabel.CROWD_PANIC.value  →  "panic"
    """
    NORMAL            = "normal"
    RUNNING           = "running"
    CROWD_PANIC       = "panic"
    VIOLENCE          = "violence"
    SUSPICION         = "suspicion"
    CROWD_SURGE       = "crowd_surge"
    INSUFFICIENT_DATA = "insufficient_data"   # population too small to classify


# ============================================================================
# Per-track feature snapshot
# ============================================================================

@dataclass(slots=True)
class TrackFeatures:
    """
    Feature snapshot for a single tracked person in a single frame.

    All values use pixel / frame units so classifiers are independent of
    camera resolution and frame rate.

    Attributes
    ----------
    track_id         : stable integer assigned by DeepSORT.
    speed            : EMA-smoothed speed in px/frame.
    acceleration     : speed delta vs previous frame (+ = speeding up).
    direction_deg    : compass heading in [0, 360).  0° = up, 90° = right.
    direction_change : shortest angular delta vs previous frame in [0, 180].
    is_running       : True when speed ≥ run_threshold.
    is_anomalous     : True when z-score deviation ≥ zscore_threshold.
    centroid         : (cx, cy) in absolute pixels.
    """
    track_id:         int
    speed:            float
    acceleration:     float
    direction_deg:    float
    direction_change: float
    is_running:       bool
    is_anomalous:     bool
    centroid:         Tuple[float, float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "track_id":         self.track_id,
            "speed":            round(self.speed,            3),
            "acceleration":     round(self.acceleration,     3),
            "direction_deg":    round(self.direction_deg,    1),
            "direction_change": round(self.direction_change, 1),
            "is_running":       self.is_running,
            "is_anomalous":     self.is_anomalous,
            "centroid":         [round(self.centroid[0], 1),
                                 round(self.centroid[1], 1)],
        }


# ============================================================================
# Crowd-level feature snapshot
# ============================================================================

@dataclass(slots=True)
class FrameFeatures:
    """
    Aggregate crowd feature snapshot for a single frame.

    Produced by ``velocity_analyzer``, ``crowd_density``, and
    ``anomaly_detector`` and consumed by ``event_classifier``.

    Attributes
    ----------
    frame_index          : 0-based pipeline frame counter.
    track_count          : number of confirmed tracks this frame.
    track_features       : per-track feature list.
    mean_speed           : crowd mean speed in px/frame.
    std_speed            : crowd speed standard deviation.
    mean_direction_deg   : circular mean of all track headings.
    direction_dispersion : circular std of headings (°).  0 = aligned, 180 = chaos.
    crowd_density        : persons per 10 000 px².
    density_zones        : grid cell → person count mapping.
    running_count        : tracks with is_running == True.
    anomalous_count      : tracks with is_anomalous == True.
    running_fraction     : running_count / track_count.
    anomalous_fraction   : anomalous_count / track_count.
    proximity_pairs      : list of (id_a, id_b, distance_px) for close pairs.
    crowd_acceleration   : mean_speed − previous_frame_mean_speed.
    """
    frame_index:          int
    track_count:          int
    track_features:       List[TrackFeatures]
    mean_speed:           float
    std_speed:            float
    mean_direction_deg:   float
    direction_dispersion: float
    crowd_density:        float
    density_zones:        Dict[Tuple[int, int], int]
    running_count:        int
    anomalous_count:      int
    running_fraction:     float
    anomalous_fraction:   float
    proximity_pairs:      List[Tuple[int, int, float]]
    crowd_acceleration:   float

    def to_dict(self) -> Dict[str, object]:
        return {
            "frame_index":          self.frame_index,
            "track_count":          self.track_count,
            "mean_speed":           round(self.mean_speed,           3),
            "std_speed":            round(self.std_speed,            3),
            "mean_direction_deg":   round(self.mean_direction_deg,   1),
            "direction_dispersion": round(self.direction_dispersion, 1),
            "crowd_density":        round(self.crowd_density,        4),
            "running_fraction":     round(self.running_fraction,     3),
            "anomalous_fraction":   round(self.anomalous_fraction,   3),
            "crowd_acceleration":   round(self.crowd_acceleration,   3),
            "proximity_pairs":      len(self.proximity_pairs),
        }

    @staticmethod
    def empty(frame_index: int) -> "FrameFeatures":
        """Return a zero-valued FrameFeatures for frames with no tracks."""
        return FrameFeatures(
            frame_index=frame_index,
            track_count=0,
            track_features=[],
            mean_speed=0.0,
            std_speed=0.0,
            mean_direction_deg=0.0,
            direction_dispersion=0.0,
            crowd_density=0.0,
            density_zones={},
            running_count=0,
            anomalous_count=0,
            running_fraction=0.0,
            anomalous_fraction=0.0,
            proximity_pairs=[],
            crowd_acceleration=0.0,
        )


# ============================================================================
# Behavior result
# ============================================================================

@dataclass(slots=True)
class BehaviorResult:
    """
    Final output produced by :class:`~core.behavior.event_classifier.EventClassifier`
    for each frame.

    Output contracts
    ----------------
    Minimal (API / WebSocket)::

        result.to_dict()  →  {"behavior": "panic", "confidence": 0.87}

    Extended (logging / storage)::

        result.to_full_dict()  →  {all fields as JSON-safe dict}
    """
    label:        BehaviorLabel
    confidence:   float
    frame_index:  int
    track_labels: Dict[int, str]     # track_id → label for anomalous tracks
    features:     FrameFeatures
    signals:      List[str]          # human-readable triggered rule names
    elapsed_ms:   float = 0.0

    def to_dict(self) -> Dict[str, object]:
        """Canonical minimal output: ``{"behavior": "panic", "confidence": 0.87}``"""
        return {
            "behavior":   self.label.value,
            "confidence": round(self.confidence, 4),
        }

    def to_full_dict(self) -> Dict[str, object]:
        """Extended output including per-track labels, signals, and features."""
        return {
            "behavior":     self.label.value,
            "confidence":   round(self.confidence, 4),
            "frame_index":  self.frame_index,
            "signals":      self.signals,
            "track_labels": {str(k): v for k, v in self.track_labels.items()},
            "features":     self.features.to_dict(),
            "elapsed_ms":   round(self.elapsed_ms, 2),
        }

    def __repr__(self) -> str:
        return (
            f"BehaviorResult(label={self.label.value!r}, "
            f"conf={self.confidence:.2f}, "
            f"frame={self.frame_index}, "
            f"signals={self.signals})"
        )


# ============================================================================
# Thresholds configuration
# ============================================================================

@dataclass
class BehaviorThresholds:
    """
    All numeric thresholds consumed by the behaviour analysis pipeline.

    Defaults are calibrated for a 720p / 25 fps stream.  Load from
    ``config/settings.py`` via :meth:`~core.behavior.behavior_analyzer.BehaviorAnalyzer.from_settings`
    or override individual fields for testing.
    """
    # ── Velocity ──────────────────────────────────────────────────────
    run_threshold_px_per_frame:     float = 15.0
    min_track_age_frames:           int   = 5

    # ── Anomaly detection ─────────────────────────────────────────────
    zscore_threshold:               float = 2.5
    min_population_for_stats:       int   = 5

    # ── Density ───────────────────────────────────────────────────────
    density_zone_rows:              int   = 4
    density_zone_cols:              int   = 4
    density_alert_threshold:        int   = 10
    heatmap_bandwidth:              int   = 30

    # ── Panic ─────────────────────────────────────────────────────────
    panic_min_anomalous_fraction:   float = 0.40
    panic_min_running_fraction:     float = 0.25
    panic_direction_dispersion:     float = 60.0    # degrees

    # ── Violence ──────────────────────────────────────────────────────
    violence_proximity_px:          float = 80.0
    violence_min_relative_speed:    float = 10.0    # px/frame
    violence_min_pair_count:        int   = 1

    # ── Suspicion ─────────────────────────────────────────────────────
    suspicion_direction_change:     float = 90.0    # degrees per frame
    suspicion_erratic_frames:       int   = 5

    # ── Surge ─────────────────────────────────────────────────────────
    surge_velocity_multiplier:      float = 2.0
    surge_direction_dispersion_max: float = 30.0    # degrees

    # ── Running ───────────────────────────────────────────────────────
    running_fraction_threshold:     float = 0.30

    # ── Alert cool-down ───────────────────────────────────────────────
    alert_cooldown_s:               float = 5.0

    # ── Trajectory history ────────────────────────────────────────────
    history_length:                 int   = 60


# ============================================================================
# Abstract base classifier
# ============================================================================

class BaseBehaviorClassifier(ABC):
    """
    Contract that every behavior classifier must satisfy.

    Separating this ABC from the concrete implementations means
    :class:`~core.behavior.behavior_analyzer.BehaviorAnalyzer` depends only
    on this interface — making the rule-based and ML backends fully
    interchangeable at runtime via
    :meth:`~core.behavior.behavior_analyzer.BehaviorAnalyzer.swap_classifier`.
    """

    @abstractmethod
    def classify(
        self,
        features: FrameFeatures,
    ) -> Tuple[BehaviorLabel, float, List[str], Dict[int, str]]:
        """
        Classify crowd behavior from a pre-computed feature snapshot.

        Parameters
        ----------
        features : FrameFeatures

        Returns
        -------
        label         : BehaviorLabel — dominant classification.
        confidence    : float in [0, 1].
        signals       : list[str] — triggered rule / feature names.
        track_labels  : dict[int, str] — per-track label for anomalous tracks.
        """
