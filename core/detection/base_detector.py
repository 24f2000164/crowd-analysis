"""
core/detection/base_detector.py
================================
Abstract base class that every detector implementation must satisfy.

This contract decouples the pipeline orchestrator from any specific detection
backend so that YOLOv8 can be swapped for another model without touching
anything outside the detection package.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np

from core.detection.yolo_detector import DetectionResult


class BaseDetector(ABC):
    """
    Minimal interface for a person detector.

    Subclasses must implement :meth:`detect`.  :meth:`detect_batch` has a
    default implementation that loops over frames, but subclasses should
    override it for GPU-batched inference.
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Detect persons in a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            Shape ``(H, W, 3)``, dtype ``uint8``, BGR colour order.

        Returns
        -------
        list[DetectionResult]
            Sorted by descending confidence.
        """

    def detect_batch(
        self, frames: Sequence[np.ndarray]
    ) -> List[List[DetectionResult]]:
        """
        Detect persons in a sequence of frames.

        Default implementation calls :meth:`detect` sequentially.
        Override for true batched inference.
        """
        return [self.detect(f) for f in frames]

    @abstractmethod
    def close(self) -> None:
        """Release model weights and device memory."""

    def __enter__(self) -> "BaseDetector":
        return self

    def __exit__(self, *_) -> None:
        self.close()
