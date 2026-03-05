"""
core/annotation/renderer.py
=============================
OpenCV-based frame annotation renderer.

Responsibility
--------------
Draw all visual overlays on a BGR frame:
  - Bounding boxes colour-coded by behavior label
  - Track IDs and speed labels
  - Per-track velocity vectors (arrow)
  - Crowd behavior banner (top of frame)
  - Frame metadata HUD (FPS, track count, frame index)
  - Density heatmap alpha-blend (when provided)

All drawing is done **in-place** on a copy of the input frame so the
original buffer is never mutated.

Usage
-----
    from core.annotation.renderer import FrameRenderer

    renderer = FrameRenderer()
    annotated = renderer.render(
        frame, tracks, behavior_result, fps=25.0, frame_index=n
    )
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.tracking.deepsort_tracker import TrackedPerson
from core.behavior.base_analyzer    import BehaviorLabel, BehaviorResult

logger = logging.getLogger("crowd_analysis.annotation")


# ---------------------------------------------------------------------------
# Colour palette — keyed by BehaviorLabel value string
# ---------------------------------------------------------------------------
_LABEL_COLORS: Dict[str, Tuple[int, int, int]] = {
    BehaviorLabel.NORMAL.value:            (0,   200,  0),     # green
    BehaviorLabel.RUNNING.value:           (0,   165, 255),    # orange
    BehaviorLabel.CROWD_PANIC.value:       (0,   0,   255),    # red
    BehaviorLabel.VIOLENCE.value:          (0,   0,   180),    # dark red
    BehaviorLabel.SUSPICION.value:         (255, 165,  0),     # blue-orange
    BehaviorLabel.CROWD_SURGE.value:       (180,  0,  255),    # purple
    BehaviorLabel.INSUFFICIENT_DATA.value: (120, 120, 120),    # grey
}
_DEFAULT_COLOR: Tuple[int, int, int] = (200, 200, 200)

# Banner background colours for critical events
_BANNER_CRITICAL = {
    BehaviorLabel.CROWD_PANIC.value,
    BehaviorLabel.VIOLENCE.value,
    BehaviorLabel.CROWD_SURGE.value,
}


class FrameRenderer:
    """
    Stateless OpenCV frame annotation renderer.

    Parameters
    ----------
    font_scale     : cv2 font scale for all text.
    box_thickness  : bounding box line thickness in pixels.
    show_velocity  : draw velocity arrows on each track.
    show_heatmap   : alpha-blend the density heatmap when provided.
    heatmap_alpha  : blend weight [0, 1] for the heatmap overlay.
    show_hud       : draw the top-left HUD (FPS, track count).
    show_banner    : draw the behavior label banner.
    """

    def __init__(
        self,
        font_scale:    float = 0.55,
        box_thickness: int   = 2,
        show_velocity: bool  = True,
        show_heatmap:  bool  = True,
        heatmap_alpha: float = 0.35,
        show_hud:      bool  = True,
        show_banner:   bool  = True,
    ) -> None:
        self._font        = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale  = font_scale
        self._box_thick   = box_thickness
        self._show_vel    = show_velocity
        self._show_heat   = show_heatmap
        self._heat_alpha  = heatmap_alpha
        self._show_hud    = show_hud
        self._show_banner = show_banner

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        frame:            np.ndarray,
        tracks:           List[TrackedPerson],
        behavior_result:  Optional[BehaviorResult] = None,
        track_labels:     Optional[Dict[int, str]] = None,
        heatmap:          Optional[np.ndarray]     = None,
        fps:              float                    = 0.0,
        frame_index:      int                      = 0,
    ) -> np.ndarray:
        """
        Return an annotated copy of ``frame``.

        Parameters
        ----------
        frame           : BGR uint8 (H, W, 3) — source frame (not mutated).
        tracks          : confirmed TrackedPerson list.
        behavior_result : result from BehaviorAnalyzer.analyze().
        track_labels    : optional {track_id: label_str} overrides per track.
        heatmap         : float32 (H, W) density map, values in [0, 1].
        fps             : pipeline processing FPS for HUD.
        frame_index     : pipeline frame counter for HUD.

        Returns
        -------
        np.ndarray — annotated BGR frame, same shape as input.
        """
        canvas = frame.copy()

        # 1 — Heatmap underlay
        if self._show_heat and heatmap is not None:
            canvas = self._draw_heatmap(canvas, heatmap)

        # 2 — Per-track boxes, IDs, velocity
        label_map = track_labels or (
            behavior_result.track_labels if behavior_result else {}
        )
        crowd_label = (
            behavior_result.label.value
            if behavior_result else BehaviorLabel.NORMAL.value
        )

        for track in tracks:
            per_track_label = label_map.get(track.id, crowd_label)
            color = _LABEL_COLORS.get(per_track_label, _DEFAULT_COLOR)
            self._draw_track(canvas, track, per_track_label, color)

        # 3 — Behavior banner
        if self._show_banner and behavior_result is not None:
            self._draw_banner(canvas, behavior_result)

        # 4 — HUD overlay
        if self._show_hud:
            self._draw_hud(canvas, len(tracks), fps, frame_index)

        return canvas

    # ------------------------------------------------------------------
    # Private drawing helpers
    # ------------------------------------------------------------------

    def _draw_track(
        self,
        canvas:      np.ndarray,
        track:       TrackedPerson,
        label:       str,
        color:       Tuple[int, int, int],
    ) -> None:
        """Draw bounding box, track ID, speed, and optional velocity arrow."""
        b  = track.bbox
        x1 = int(b.x1); y1 = int(b.y1)
        x2 = int(b.x2); y2 = int(b.y2)

        # Bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, self._box_thick)

        # Label text: "ID:3 | 8.2px/f | running"
        spd  = track.velocity.speed
        text = f"ID:{track.id} {spd:.1f}px/f"
        if label not in (BehaviorLabel.NORMAL.value,):
            text += f" [{label}]"

        # Background pill for readability
        (tw, th), baseline = cv2.getTextSize(
            text, self._font, self._font_scale, 1
        )
        label_y = max(y1 - 6, th + 4)
        cv2.rectangle(
            canvas,
            (x1, label_y - th - baseline - 2),
            (x1 + tw + 4, label_y + baseline),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            canvas, text, (x1 + 2, label_y - baseline),
            self._font, self._font_scale, (255, 255, 255), 1, cv2.LINE_AA,
        )

        # Velocity arrow
        if self._show_vel and spd > 1.0:
            cx, cy = track.bbox.centroid
            scale  = min(spd * 2.0, 40.0)          # cap arrow length
            vx_n   = track.velocity.vx / max(spd, 1e-9)
            vy_n   = track.velocity.vy / max(spd, 1e-9)
            end_x  = int(cx + vx_n * scale)
            end_y  = int(cy + vy_n * scale)
            cv2.arrowedLine(
                canvas,
                (int(cx), int(cy)), (end_x, end_y),
                color, 2, tipLength=0.35,
            )

    def _draw_banner(
        self,
        canvas:  np.ndarray,
        result:  BehaviorResult,
    ) -> None:
        """Draw a full-width behavior label banner at the top of the frame."""
        h, w = canvas.shape[:2]
        label     = result.label.value.upper()
        conf_str  = f"{result.confidence:.0%}"
        text      = f"  BEHAVIOR: {label}  CONF: {conf_str}  "

        banner_h  = 36
        is_alert  = result.label.value in _BANNER_CRITICAL

        # Background
        bg_color  = (0, 0, 200) if is_alert else (30, 30, 30)
        alpha     = 0.82
        overlay   = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), bg_color, cv2.FILLED)
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

        # Flash border for critical events
        if is_alert:
            cv2.rectangle(canvas, (0, 0), (w, banner_h), (0, 0, 255), 3)

        # Text
        fs  = 0.65
        (tw, _), _ = cv2.getTextSize(text, self._font, fs, 2)
        tx  = (w - tw) // 2
        cv2.putText(
            canvas, text, (tx, 24),
            self._font, fs, (255, 255, 255), 2, cv2.LINE_AA,
        )

        # Signal strip (small, bottom of banner)
        if result.signals:
            sig_text = " | ".join(result.signals[:3])
            cv2.putText(
                canvas, sig_text, (8, banner_h + 16),
                self._font, 0.38, (180, 220, 255), 1, cv2.LINE_AA,
            )

    def _draw_hud(
        self,
        canvas:      np.ndarray,
        track_count: int,
        fps:         float,
        frame_index: int,
    ) -> None:
        """Draw the bottom-left HUD: FPS, track count, frame index."""
        h, w = canvas.shape[:2]
        lines = [
            f"FPS: {fps:.1f}",
            f"Tracks: {track_count}",
            f"Frame: {frame_index}",
        ]
        x, y  = 8, h - 10
        fs    = 0.45
        lh    = 18   # line height

        for i, line in enumerate(reversed(lines)):
            yy = y - i * lh
            # Drop shadow
            cv2.putText(canvas, line, (x + 1, yy + 1),
                        self._font, fs, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, line, (x, yy),
                        self._font, fs, (200, 255, 200), 1, cv2.LINE_AA)

    def _draw_heatmap(
        self,
        canvas:  np.ndarray,
        heatmap: np.ndarray,
    ) -> np.ndarray:
        """Alpha-blend a float32 density heatmap onto the canvas."""
        h, w = canvas.shape[:2]

        # Resize if heatmap dimensions differ from canvas
        if heatmap.shape[:2] != (h, w):
            heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

        # Map float [0,1] → uint8 BGR using COLORMAP_JET
        heat_uint8 = (heatmap * 255).clip(0, 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)

        # Only blend cells above a minimal density threshold to keep clean areas clear
        mask = (heatmap > 0.05).astype(np.float32)
        alpha_map = (mask * self._heat_alpha)[..., np.newaxis]

        blended = (
            canvas.astype(np.float32) * (1 - alpha_map) +
            heat_color.astype(np.float32) * alpha_map
        ).clip(0, 255).astype(np.uint8)

        return blended