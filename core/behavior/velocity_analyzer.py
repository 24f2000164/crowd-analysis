"""
core/behavior/velocity_analyzer.py
=====================================
Per-track velocity, acceleration, and direction-change computation.

Responsibility
--------------
Given the current list of ``TrackedPerson`` objects and the previous frame's
speed / direction snapshots, compute a ``TrackFeatures`` object for every
confirmed track.  This module is **stateless** per call — all history is
owned by the :class:`~core.behavior.trajectory_store.TrajectoryStore` and
passed in by the orchestrating ``BehaviorAnalyzer``.

Algorithm
---------
1. Raw speed comes directly from ``TrackedPerson.velocity.speed`` (already
   EMA-smoothed by the tracker).
2. Acceleration = speed_now − speed_prev (uses ``prev_speeds`` dict).
3. Direction change = shortest angular arc between current and previous
   heading (wraps correctly at the 0°/360° boundary).
4. ``is_running`` flag is set when speed ≥ ``run_threshold_px_per_frame``.
5. ``is_anomalous`` flag requires z-score comparison against the crowd mean —
   see :mod:`~core.behavior.anomaly_detector` for that calculation.
   This module marks the flag as ``False``; the anomaly detector patches it.

Usage
-----
    from core.behavior.velocity_analyzer import VelocityAnalyzer

    analyzer = VelocityAnalyzer(thresholds)
    track_features = analyzer.compute(
        tracks=confirmed_tracks,
        prev_speeds={1: 5.0, 2: 8.0},
        prev_directions={1: 90.0, 2: 45.0},
    )
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Tuple

from core.tracking.deepsort_tracker import TrackedPerson
from core.behavior.base_analyzer import BehaviorThresholds, TrackFeatures

logger = logging.getLogger("crowd_analysis.behavior.velocity")


class VelocityAnalyzer:
    """
    Stateless per-track velocity feature extractor.

    Parameters
    ----------
    thresholds : BehaviorThresholds
        Provides ``run_threshold_px_per_frame`` and ``min_track_age_frames``.
    """

    def __init__(self, thresholds: BehaviorThresholds) -> None:
        self._t = thresholds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        tracks:          List[TrackedPerson],
        prev_speeds:     Dict[int, float],
        prev_directions: Dict[int, float],
    ) -> List[TrackFeatures]:
        """
        Compute velocity features for all confirmed tracks.

        The ``is_anomalous`` field is left ``False`` here and patched by
        :class:`~core.behavior.anomaly_detector.AnomalyDetector` afterwards,
        which requires the full crowd statistics derived from this output.

        Parameters
        ----------
        tracks          : confirmed TrackedPerson list from the current frame.
        prev_speeds     : {track_id: speed_px_per_frame} from the last frame.
        prev_directions : {track_id: direction_deg} from the last frame.

        Returns
        -------
        list[TrackFeatures]  — one entry per input track, same order.
        """
        results: List[TrackFeatures] = []

        for track in tracks:
            speed     = track.velocity.speed
            direction = track.velocity.direction_deg

            prev_spd = prev_speeds.get(track.id, speed)
            prev_dir = prev_directions.get(track.id, direction)

            acceleration   = speed - prev_spd
            direction_chg  = self._angular_diff(direction, prev_dir)
            is_running     = speed >= self._t.run_threshold_px_per_frame

            results.append(TrackFeatures(
                track_id=track.id,
                speed=speed,
                acceleration=acceleration,
                direction_deg=direction,
                direction_change=direction_chg,
                is_running=is_running,
                is_anomalous=False,       # patched by AnomalyDetector
                centroid=track.bbox.centroid,
            ))

        logger.debug(
            "VelocityAnalyzer: processed %d tracks, %d running.",
            len(results),
            sum(1 for tf in results if tf.is_running),
        )
        return results

    # ------------------------------------------------------------------
    # Crowd statistics helpers (used by anomaly_detector and event_classifier)
    # ------------------------------------------------------------------

    @staticmethod
    def crowd_speed_stats(
        track_features: List[TrackFeatures],
    ) -> Tuple[float, float]:
        """
        Compute crowd mean and standard deviation of speed.

        Returns
        -------
        (mean_speed, std_speed)  — both in px/frame.
        Returns (0.0, 0.0) for empty input.
        """
        if not track_features:
            return 0.0, 0.0
        speeds = [tf.speed for tf in track_features]
        mean   = sum(speeds) / len(speeds)
        if len(speeds) < 2:
            return mean, 0.0
        variance = sum((s - mean) ** 2 for s in speeds) / (len(speeds) - 1)
        return mean, math.sqrt(variance)

    @staticmethod
    def circular_direction_stats(
        track_features: List[TrackFeatures],
    ) -> Tuple[float, float]:
        """
        Compute circular mean and dispersion of track headings.

        Returns
        -------
        (mean_direction_deg, dispersion_deg)

        Dispersion uses the mean resultant length R:
          dispersion ≈ sqrt(−2 · ln(R)) · (180/π)
        R = 1 → fully aligned; R → 0 → fully random.
        """
        if not track_features:
            return 0.0, 0.0

        sin_sum = sum(math.sin(math.radians(tf.direction_deg)) for tf in track_features)
        cos_sum = sum(math.cos(math.radians(tf.direction_deg)) for tf in track_features)
        n = len(track_features)

        mean_sin = sin_sum / n
        mean_cos = cos_sum / n
        mean_rad = math.atan2(mean_sin, mean_cos)
        mean_deg = math.degrees(mean_rad) % 360.0

        R = min(math.hypot(mean_sin, mean_cos), 1.0 - 1e-9)
        dispersion_deg = math.degrees(math.sqrt(-2.0 * math.log(R)))

        return mean_deg, dispersion_deg

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _angular_diff(a_deg: float, b_deg: float) -> float:
        """
        Shortest angular distance between two compass angles.

        Returns a value in [0, 180] — always positive (magnitude only).
        """
        diff = abs(a_deg - b_deg) % 360.0
        return min(diff, 360.0 - diff)
