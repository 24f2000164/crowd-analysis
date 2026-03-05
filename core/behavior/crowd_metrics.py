"""
core/behavior/crowd_metrics.py
================================
Lightweight crowd density classifier.

Responsibility
--------------
Given a person count and frame dimensions, compute a raw density figure
(persons / px²) and classify it into one of four named bands:

    low_density       — density < 0.00001
    medium_density    — density < 0.00003
    high_density      — density < 0.00006
    critical_density  — density ≥ 0.00006

This module is intentionally stateless so it can be called from any pipeline
stage, background task, or test without side-effects.

Integration
-----------
The pipeline imports and calls ``CrowdMetricsAnalyzer.compute_density()``
inside ``PipelineFrame.to_meta_dict()`` — see
``core/pipeline/video_pipeline.py`` for the wiring.

Usage
-----
    from core.behavior.crowd_metrics import CrowdMetricsAnalyzer

    result = CrowdMetricsAnalyzer.compute_density(
        people_count=25, frame_width=1280, frame_height=720
    )
    # {"density": 2.7126e-05, "level": "medium_density"}
"""

from __future__ import annotations

import logging
from typing import Dict

logger = logging.getLogger("crowd_analysis.behavior.crowd_metrics")

# ---------------------------------------------------------------------------
# Density thresholds (persons / pixel²)
# ---------------------------------------------------------------------------
_LOW_DENSITY_THRESHOLD:      float = 0.00001
_MEDIUM_DENSITY_THRESHOLD:   float = 0.00003
_HIGH_DENSITY_THRESHOLD:     float = 0.00006


class CrowdMetricsAnalyzer:
    """
    Stateless crowd density classifier.

    All methods are class-level so no instantiation is required, though
    an instance can be created if the pipeline requires an object interface.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def compute_density(
        cls,
        people_count:  int,
        frame_width:   int,
        frame_height:  int,
    ) -> Dict[str, object]:
        """
        Compute crowd density and assign a density level.

        Parameters
        ----------
        people_count : int
            Number of detected persons in the frame.
        frame_width : int
            Frame width in pixels (must be > 0).
        frame_height : int
            Frame height in pixels (must be > 0).

        Returns
        -------
        dict
            ``{"density": float, "level": str}``

        Examples
        --------
        >>> CrowdMetricsAnalyzer.compute_density(25, 1280, 720)
        {"density": 2.7126e-05, "level": "medium_density"}
        """
        if frame_width <= 0 or frame_height <= 0:
            logger.warning(
                "compute_density: invalid frame dimensions %dx%d — returning zero.",
                frame_width, frame_height,
            )
            return {"density": 0.0, "level": "low_density"}

        if people_count < 0:
            logger.warning(
                "compute_density: negative people_count %d — clamping to 0.",
                people_count,
            )
            people_count = 0

        density = people_count / (frame_width * frame_height)
        level   = cls._classify_density(density)

        logger.debug(
            "Density: %.6f  level=%s  people=%d  frame=%dx%d",
            density, level, people_count, frame_width, frame_height,
        )

        return {"density": density, "level": level}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_density(density: float) -> str:
        """
        Map a raw density value to a named level string.

        Parameters
        ----------
        density : float  — persons / pixel²

        Returns
        -------
        str  — one of ``low_density``, ``medium_density``,
               ``high_density``, ``critical_density``.
        """
        if density < _LOW_DENSITY_THRESHOLD:
            return "low_density"
        if density < _MEDIUM_DENSITY_THRESHOLD:
            return "medium_density"
        if density < _HIGH_DENSITY_THRESHOLD:
            return "high_density"
        return "critical_density"