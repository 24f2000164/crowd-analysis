"""
core/behavior/anomaly_detector.py
===================================
Statistical anomaly detection for individual tracks within the crowd.

Responsibility
--------------
Given a list of ``TrackFeatures`` (output of ``VelocityAnalyzer``) and crowd
speed statistics, compute z-scores and patch the ``is_anomalous`` flag on
each ``TrackFeatures`` object **in-place**.

Why in-place patching?
    ``TrackFeatures`` is a ``slots`` dataclass — the anomaly flag is a field
    that is reserved for this module to set.  Patching in-place avoids
    rebuilding the entire list and keeps the object graph consistent for
    downstream consumers (``crowd_density``, ``event_classifier``).

Detection methods
-----------------
1. **Speed z-score** — deviation of a track's speed from the crowd mean.
   Flags the track when ``|z| ≥ zscore_threshold``.

2. **Direction outlier** — track heading deviates by more than
   ``direction_outlier_deg`` from the circular crowd mean.  Applied only
   when the crowd's direction dispersion is low (the crowd is moving
   coherently), so a single person going the wrong way stands out.

3. **Sudden acceleration** — instantaneous acceleration exceeds
   ``acceleration_threshold_px_per_frame``.

A track must trigger at least ``min_signals`` methods to be flagged
anomalous, preventing single-metric false positives.

Usage
-----
    from core.behavior.anomaly_detector import AnomalyDetector

    detector = AnomalyDetector(thresholds)
    anomaly_count = detector.detect(track_features, mean_speed, std_speed,
                                    mean_direction_deg, direction_dispersion)
    # track_features[i].is_anomalous is now set
"""

from __future__ import annotations

import logging
import math
from typing import List, Tuple

from core.behavior.base_analyzer import BehaviorThresholds, TrackFeatures

logger = logging.getLogger("crowd_analysis.behavior.anomaly")

# Minimum number of anomaly signals required to flag a track
_MIN_SIGNALS_TO_FLAG: int = 1

# Acceleration threshold (px/frame²) above which a track is flagged
_ACCELERATION_THRESHOLD: float = 8.0

# A track heading must deviate by this many degrees from the crowd mean
# while direction_dispersion < _COHERENT_CROWD_DISPERSION to be flagged
_DIRECTION_OUTLIER_DEG: float    = 120.0
_COHERENT_CROWD_DISPERSION: float = 40.0


class AnomalyDetector:
    """
    Patches ``is_anomalous`` on a ``TrackFeatures`` list using three
    complementary statistical methods.

    Parameters
    ----------
    thresholds : BehaviorThresholds
        Provides ``zscore_threshold`` and ``min_population_for_stats``.
    """

    def __init__(self, thresholds: BehaviorThresholds) -> None:
        self._t = thresholds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        track_features:      List[TrackFeatures],
        mean_speed:          float,
        std_speed:           float,
        mean_direction_deg:  float,
        direction_dispersion: float,
    ) -> int:
        """
        Patch ``is_anomalous`` on each ``TrackFeatures`` entry in-place.

        Parameters
        ----------
        track_features       : list produced by VelocityAnalyzer (mutated).
        mean_speed           : crowd mean speed in px/frame.
        std_speed            : crowd speed standard deviation.
        mean_direction_deg   : circular mean of crowd headings (degrees).
        direction_dispersion : circular std of crowd headings (degrees).

        Returns
        -------
        int — number of tracks flagged as anomalous.
        """
        n = len(track_features)
        if n < self._t.min_population_for_stats:
            logger.debug(
                "AnomalyDetector: population %d < min %d — skipping.",
                n, self._t.min_population_for_stats,
            )
            return 0

        anomalous_count = 0

        for tf in track_features:
            signals = 0

            # ── Method 1: speed z-score ────────────────────────────────
            if std_speed > 0:
                z = abs(tf.speed - mean_speed) / std_speed
                if z >= self._t.zscore_threshold:
                    signals += 1
                    logger.debug(
                        "Track %d: speed z-score=%.2f (threshold=%.1f).",
                        tf.track_id, z, self._t.zscore_threshold,
                    )

            # ── Method 2: direction outlier ────────────────────────────
            if direction_dispersion < _COHERENT_CROWD_DISPERSION:
                angular_gap = _angular_diff(tf.direction_deg, mean_direction_deg)
                if angular_gap >= _DIRECTION_OUTLIER_DEG:
                    signals += 1
                    logger.debug(
                        "Track %d: direction outlier %.1f° from crowd mean.",
                        tf.track_id, angular_gap,
                    )

            # ── Method 3: sudden acceleration ─────────────────────────
            if abs(tf.acceleration) >= _ACCELERATION_THRESHOLD:
                signals += 1
                logger.debug(
                    "Track %d: sudden acceleration %.2f px/f².",
                    tf.track_id, tf.acceleration,
                )

            # ── Flag if enough signals fired ───────────────────────────
            if signals >= _MIN_SIGNALS_TO_FLAG:
                # slots=True dataclass — use object.__setattr__ to mutate
                object.__setattr__(tf, "is_anomalous", True)
                anomalous_count += 1

        logger.debug(
            "AnomalyDetector: %d / %d tracks flagged anomalous.",
            anomalous_count, n,
        )
        return anomalous_count

    # ------------------------------------------------------------------
    # Batch anomaly summary
    # ------------------------------------------------------------------

    @staticmethod
    def anomaly_summary(
        track_features: List[TrackFeatures],
    ) -> Tuple[int, float]:
        """
        Count anomalous tracks and compute the anomalous fraction.

        Returns
        -------
        (anomalous_count, anomalous_fraction)
        """
        n = len(track_features)
        if n == 0:
            return 0, 0.0
        count = sum(1 for tf in track_features if tf.is_anomalous)
        return count, count / n


# ---------------------------------------------------------------------------
# Module-level geometry helper (reused by other modules via import)
# ---------------------------------------------------------------------------

def _angular_diff(a_deg: float, b_deg: float) -> float:
    """Shortest arc between two compass angles; returns value in [0, 180]."""
    diff = abs(a_deg - b_deg) % 360.0
    return min(diff, 360.0 - diff)
