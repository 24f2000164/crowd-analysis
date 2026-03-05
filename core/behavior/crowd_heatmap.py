"""
core/behavior/crowd_heatmap.py
================================
Real-time crowd movement heatmap generator.

Responsibility
--------------
Given a list of tracked-person centroids and a source BGR frame, produce
a colourised heatmap overlay that visualises crowd density in real time.

Algorithm
---------
1. Extract centroid (cx, cy) of each confirmed tracked person.
2. Increment an accumulation grid at each centroid location.
3. Apply Gaussian blur to spread each centroid into a smooth kernel.
4. Normalise the result to [0, 1].
5. Map to a colourmap (COLORMAP_JET by default).
6. Alpha-blend the colourised heatmap over the source frame.

The same underlying heatmap array is returned alongside the blended frame
so callers can store or transmit it without re-generating it.

Usage
-----
    from core.behavior.crowd_heatmap import CrowdHeatmapGenerator

    generator = CrowdHeatmapGenerator(frame_height=720, frame_width=1280)

    # per-frame
    centroids = [(cx1, cy1), (cx2, cy2), ...]
    heatmap_frame, raw_heatmap = generator.overlay(frame, centroids)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("crowd_analysis.behavior.crowd_heatmap")


class CrowdHeatmapGenerator:
    """
    Gaussian-kernel crowd density heatmap with OpenCV colour overlay.

    Parameters
    ----------
    frame_height : int
        Frame height in pixels.
    frame_width : int
        Frame width in pixels.
    blur_sigma : int
        Standard deviation (pixels) of the Gaussian kernel applied to each
        centroid.  Larger values produce smoother, more spread-out heat blobs.
        Default 30 is calibrated for 720p footage at ~25 fps.
    alpha : float
        Heatmap opacity when blending over the source frame.
        0.0 = invisible, 1.0 = fully opaque.  Default 0.45.
    colormap : int
        OpenCV colourmap constant used to colour the normalised heatmap.
        Default ``cv2.COLORMAP_JET``.
    decay : float
        EMA decay factor applied to the accumulation canvas each frame so
        old heat cools off over time.  0.0 = no memory (single-frame),
        1.0 = no decay (unbounded accumulation).  Default 0.85.
    """

    def __init__(
        self,
        frame_height: int   = 720,
        frame_width:  int   = 1280,
        blur_sigma:   int   = 30,
        alpha:        float = 0.45,
        colormap:     int   = cv2.COLORMAP_JET,
        decay:        float = 0.85,
    ) -> None:
        if frame_height <= 0 or frame_width <= 0:
            raise ValueError(
                f"frame_height and frame_width must be positive, "
                f"got {frame_height}×{frame_width}."
            )
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"decay must be in [0, 1], got {decay}.")

        self._h        = frame_height
        self._w        = frame_width
        self._sigma    = max(int(blur_sigma), 1)
        self._alpha    = alpha
        self._colormap = colormap
        self._decay    = decay

        # Persistent accumulation canvas — carries heat across frames
        self._canvas: np.ndarray = np.zeros((frame_height, frame_width), dtype=np.float32)

        logger.info(
            "CrowdHeatmapGenerator init — %dx%d  sigma=%d  alpha=%.2f  decay=%.2f",
            frame_width, frame_height, blur_sigma, alpha, decay,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def overlay(
        self,
        frame:     np.ndarray,
        centroids: List[Tuple[float, float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a heatmap from centroid positions and blend it over ``frame``.

        Parameters
        ----------
        frame : np.ndarray
            Source BGR uint8 frame of shape (H, W, 3).
        centroids : list of (cx, cy) float tuples
            Tracked-person centroid positions in absolute pixels.

        Returns
        -------
        heatmap_frame : np.ndarray
            BGR uint8 frame with the heatmap blended in.
        raw_heatmap : np.ndarray
            Float32 array of shape (H, W) normalised to [0, 1].
            Useful for downstream storage or transmission.
        """
        self._validate_frame(frame)

        # 1. Apply temporal decay to the existing canvas
        self._canvas *= self._decay

        # 2. Increment at each centroid
        for cx, cy in centroids:
            xi = int(max(0.0, min(cx, self._w - 1)))
            yi = int(max(0.0, min(cy, self._h - 1)))
            self._canvas[yi, xi] += 1.0

        # 3. Gaussian blur — spreads each centroid into a smooth blob
        blurred = self._apply_blur(self._canvas.copy())

        # 4. Normalise to [0, 1]
        raw_heatmap = self._normalise(blurred)

        # 5. Colourise
        coloured = self._colourise(raw_heatmap)

        # 6. Alpha-blend over the source frame
        heatmap_frame = self._blend(frame, coloured)

        logger.debug(
            "Heatmap: %d centroids  canvas_max=%.3f  raw_max=%.3f",
            len(centroids),
            float(self._canvas.max()),
            float(raw_heatmap.max()),
        )

        return heatmap_frame, raw_heatmap

    def overlay_from_heatmap(
        self,
        frame:       np.ndarray,
        raw_heatmap: np.ndarray,
    ) -> np.ndarray:
        """
        Blend a pre-computed normalised heatmap over ``frame``.

        Use this when the heatmap has already been generated by the
        ``CrowdDensityAnalyzer`` (which also produces a ``float32`` heatmap)
        to avoid re-computing it.

        Parameters
        ----------
        frame : np.ndarray
            Source BGR uint8 frame.
        raw_heatmap : np.ndarray
            Float32 array of shape (H, W) with values in [0, 1].

        Returns
        -------
        np.ndarray — BGR uint8 frame with heatmap overlay.
        """
        self._validate_frame(frame)
        coloured = self._colourise(raw_heatmap)
        return self._blend(frame, coloured)

    def reset(self) -> None:
        """Clear the accumulation canvas (e.g. when switching streams)."""
        self._canvas[:] = 0.0
        logger.debug("CrowdHeatmapGenerator canvas reset.")

    def set_frame_shape(self, height: int, width: int) -> None:
        """Resize the internal canvas if the input resolution changes."""
        if height == self._h and width == self._w:
            return
        self._h      = height
        self._w      = width
        self._canvas = np.zeros((height, width), dtype=np.float32)
        logger.info(
            "CrowdHeatmapGenerator resized to %dx%d.", width, height
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_blur(self, canvas: np.ndarray) -> np.ndarray:
        """
        Apply three-pass Gaussian blur to approximate a wide isotropic kernel.

        Three passes of the same sigma are equivalent to one pass with
        sigma × √3, cheaply producing smoother blobs without a giant kernel.
        """
        ksize = 0   # auto-derived from sigma
        for _ in range(3):
            canvas = cv2.GaussianBlur(canvas, (ksize, ksize), self._sigma)
        return canvas

    @staticmethod
    def _normalise(canvas: np.ndarray) -> np.ndarray:
        """Normalise a float32 array to [0, 1]."""
        max_val = canvas.max()
        if max_val > 0:
            return (canvas / max_val).astype(np.float32)
        return canvas.astype(np.float32)

    def _colourise(self, normalised: np.ndarray) -> np.ndarray:
        """
        Map a [0, 1] float32 array to a BGR uint8 colourised image.
        """
        uint8_map = (normalised * 255).astype(np.uint8)
        return cv2.applyColorMap(uint8_map, self._colormap)

    def _blend(self, frame: np.ndarray, coloured: np.ndarray) -> np.ndarray:
        """
        Alpha-blend ``coloured`` over ``frame``.

        Where the heatmap is cold (near-black in JET = dark blue) the blend
        is transparent so the source frame is fully visible.  Hot zones
        (yellow/red) are rendered at full ``alpha`` opacity.
        """
        if frame.shape[:2] != coloured.shape[:2]:
            coloured = cv2.resize(
                coloured, (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        # Convert both to float32 for blending arithmetic
        f32_frame    = frame.astype(np.float32)
        f32_coloured = coloured.astype(np.float32)

        blended = cv2.addWeighted(
            f32_frame,    1.0 - self._alpha,
            f32_coloured, self._alpha,
            0,
        )
        return np.clip(blended, 0, 255).astype(np.uint8)

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        """Raise ValueError if the frame is not a valid BGR uint8 image."""
        if frame is None:
            raise ValueError("frame must not be None.")
        if not isinstance(frame, np.ndarray):
            raise ValueError(f"frame must be np.ndarray, got {type(frame).__name__}.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"frame must have shape (H, W, 3), got {frame.shape}."
            )
        if frame.dtype != np.uint8:
            raise ValueError(
                f"frame must be uint8, got {frame.dtype}."
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def frame_shape(self) -> Tuple[int, int]:
        """(height, width) of the internal canvas."""
        return self._h, self._w

    def __repr__(self) -> str:
        return (
            f"CrowdHeatmapGenerator("
            f"shape={self._w}×{self._h}, "
            f"sigma={self._sigma}, "
            f"alpha={self._alpha}, "
            f"decay={self._decay})"
        )