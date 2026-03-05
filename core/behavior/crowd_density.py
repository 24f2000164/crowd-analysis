"""
core/behavior/crowd_density.py
================================
Crowd density analysis: zone-based person counting, Gaussian heatmap
generation, and close-contact proximity-pair detection.

Responsibility
--------------
Given a list of ``TrackFeatures``, compute:

1. **Zone density map** — the frame is partitioned into an R × C grid and
   each cell stores a person count.  Cells exceeding the alert threshold
   are surfaced as high-density zones.

2. **Overall density** — persons per 10 000 px² (resolution-independent).

3. **Proximity pairs** — all (id_a, id_b, distance_px) pairs whose
   centroids are within ``violence_proximity_px`` of each other.
   Used downstream by ``EventClassifier`` for violence detection.

4. **Heatmap array** — a float32 numpy array of shape (H, W) with a
   Gaussian kernel centred on each person centroid.  Ready for alpha-blend
   overlay in ``core/annotation/heatmap_overlay.py``.

Usage
-----
    from core.behavior.crowd_density import CrowdDensityAnalyzer

    density_analyzer = CrowdDensityAnalyzer(thresholds, frame_shape=(720,1280))
    result = density_analyzer.compute(track_features)

    print(result.overall_density)      # persons / 10 000 px²
    print(result.zone_counts)          # {(row, col): count}
    print(result.high_density_zones)   # zones above alert threshold
    print(result.proximity_pairs)      # [(id_a, id_b, dist_px), ...]
    heatmap = result.heatmap           # np.ndarray float32 (H, W)
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.behavior.base_analyzer import BehaviorThresholds, TrackFeatures

logger = logging.getLogger("crowd_analysis.behavior.density")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DensityResult:
    """
    Output of one :meth:`CrowdDensityAnalyzer.compute` call.

    Attributes
    ----------
    overall_density    : float — persons per 10 000 px² (frame-wide).
    zone_counts        : dict  — (row, col) → person count.
    high_density_zones : list  — (row, col) keys above the alert threshold.
    max_zone_count     : int   — highest single-zone person count.
    proximity_pairs    : list  — (id_a, id_b, distance_px) for close pairs.
    heatmap            : ndarray float32 (H, W) — Gaussian density map.
    """
    overall_density:    float
    zone_counts:        Dict[Tuple[int, int], int]
    high_density_zones: List[Tuple[int, int]]
    max_zone_count:     int
    proximity_pairs:    List[Tuple[int, int, float]]
    heatmap:            np.ndarray   # shape (H, W), float32

    def to_dict(self) -> Dict[str, object]:
        return {
            "overall_density":      round(self.overall_density, 4),
            "max_zone_count":       self.max_zone_count,
            "high_density_zones":   len(self.high_density_zones),
            "proximity_pairs":      len(self.proximity_pairs),
        }


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class CrowdDensityAnalyzer:
    """
    Zone-based crowd density and proximity analyzer.

    Parameters
    ----------
    thresholds  : BehaviorThresholds — provides zone dimensions, alert
                  threshold, violence_proximity_px, heatmap_bandwidth.
    frame_shape : (height, width) in pixels.  Update with
                  :meth:`set_frame_shape` if resolution changes.
    """

    def __init__(
        self,
        thresholds:  BehaviorThresholds,
        frame_shape: Tuple[int, int] = (720, 1280),
    ) -> None:
        self._t          = thresholds
        self._frame_h, self._frame_w = frame_shape

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, track_features: List[TrackFeatures]) -> DensityResult:
        """
        Compute density metrics and proximity pairs for one frame.

        Parameters
        ----------
        track_features : list of TrackFeatures (from VelocityAnalyzer).

        Returns
        -------
        DensityResult
        """
        if not track_features:
            return self._empty_result()

        zone_counts    = self._compute_zones(track_features)
        overall        = self._overall_density(len(track_features))
        high_zones     = [
            zone for zone, cnt in zone_counts.items()
            if cnt >= self._t.density_alert_threshold
        ]
        max_zone       = max(zone_counts.values(), default=0)
        pairs          = self._compute_proximity(track_features)
        heatmap        = self._build_heatmap(track_features)

        if high_zones:
            logger.warning(
                "High-density zones detected: %s (threshold=%d)",
                high_zones, self._t.density_alert_threshold,
            )

        return DensityResult(
            overall_density=overall,
            zone_counts=zone_counts,
            high_density_zones=high_zones,
            max_zone_count=max_zone,
            proximity_pairs=pairs,
            heatmap=heatmap,
        )

    def set_frame_shape(self, height: int, width: int) -> None:
        """Update frame dimensions at runtime (e.g. resolution change)."""
        self._frame_h = height
        self._frame_w = width

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_zones(
        self,
        track_features: List[TrackFeatures],
    ) -> Dict[Tuple[int, int], int]:
        """Partition the frame into an R × C grid and count persons per cell."""
        rows  = self._t.density_zone_rows
        cols  = self._t.density_zone_cols
        counts: Dict[Tuple[int, int], int] = defaultdict(int)

        for tf in track_features:
            cx, cy = tf.centroid
            col_idx = min(int(cx / self._frame_w * cols), cols - 1)
            row_idx = min(int(cy / self._frame_h * rows), rows - 1)
            counts[(row_idx, col_idx)] += 1

        return dict(counts)

    def _overall_density(self, person_count: int) -> float:
        """Persons per 10 000 px² (resolution-independent metric)."""
        frame_area = max(self._frame_h * self._frame_w, 1)
        return person_count / frame_area * 10_000

    def _compute_proximity(
        self,
        track_features: List[TrackFeatures],
    ) -> List[Tuple[int, int, float]]:
        """
        Find all person pairs within ``violence_proximity_px`` of each other.

        Complexity: O(n²) — suitable for n ≤ ~300 persons.
        For larger crowds, replace with a KD-tree (``scipy.spatial.cKDTree``).
        """
        threshold = self._t.violence_proximity_px
        pairs: List[Tuple[int, int, float]] = []
        n = len(track_features)

        for i in range(n):
            cx_i, cy_i = track_features[i].centroid
            for j in range(i + 1, n):
                cx_j, cy_j = track_features[j].centroid
                dist = math.hypot(cx_i - cx_j, cy_i - cy_j)
                if dist <= threshold:
                    pairs.append((
                        track_features[i].track_id,
                        track_features[j].track_id,
                        dist,
                    ))

        return pairs

    def _build_heatmap(
        self,
        track_features: List[TrackFeatures],
    ) -> np.ndarray:
        """
        Build a Gaussian kernel density map from person centroids.

        Each centroid contributes a 2-D isotropic Gaussian with sigma =
        ``heatmap_bandwidth`` pixels.  The result is normalised to [0, 1]
        and returned as a float32 array of shape (H, W).

        Uses a fast approximation: box-filter the centroid indicator image
        three times (≈ Gaussian via Central Limit Theorem) rather than
        evaluating the full kernel for every pixel.
        """
        h, w = self._frame_h, self._frame_w
        canvas = np.zeros((h, w), dtype=np.float32)

        for tf in track_features:
            cx, cy = tf.centroid
            xi = int(max(0, min(cx, w - 1)))
            yi = int(max(0, min(cy, h - 1)))
            canvas[yi, xi] += 1.0

        # Three-pass box filter approximates Gaussian blur
        bw = max(int(self._t.heatmap_bandwidth), 1)
        # Use a simple numpy-based box filter (avoids cv2 dependency here)
        from numpy.lib.stride_tricks import sliding_window_view
        try:
            import cv2
            for _ in range(3):
                canvas = cv2.GaussianBlur(canvas, (0, 0), bw)
        except ImportError:
            # Fallback: uniform box filter (less smooth but no dependency)
            from scipy.ndimage import uniform_filter
            kernel_size = max(bw * 3 | 1, 3)   # must be odd
            for _ in range(3):
                try:
                    from scipy.ndimage import uniform_filter
                    canvas = uniform_filter(canvas, size=kernel_size).astype(np.float32)
                except ImportError:
                    break

        max_val = canvas.max()
        if max_val > 0:
            canvas /= max_val

        return canvas

    def _empty_result(self) -> DensityResult:
        heatmap = np.zeros((self._frame_h, self._frame_w), dtype=np.float32)
        return DensityResult(
            overall_density=0.0,
            zone_counts={},
            high_density_zones=[],
            max_zone_count=0,
            proximity_pairs=[],
            heatmap=heatmap,
        )
