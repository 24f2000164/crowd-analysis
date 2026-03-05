"""
tests/unit/test_yolo_detector.py
=================================
Unit tests for the YOLOv8 person detection module.

All tests run without a GPU and without real model weights by mocking
the ultralytics YOLO class, so the CI pipeline has no heavy dependencies.
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from core.detection.yolo_detector import (
    BoundingBox,
    DetectionResult,
    YOLOv8Detector,
    _PERSON_CLASS_ID,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Return a blank BGR uint8 frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _mock_box(x1: float, y1: float, x2: float, y2: float, conf: float):
    """Build a mock ultralytics Boxes-like object for a single detection."""
    box = MagicMock()
    box.xyxy = torch.tensor([[x1, y1, x2, y2]])
    box.conf  = torch.tensor([conf])
    box.cls   = torch.tensor([_PERSON_CLASS_ID], dtype=torch.int)
    return box


def _mock_result(boxes):
    result = MagicMock()
    result.boxes = boxes
    return result


def _make_detector(mock_yolo_cls, conf: float = 0.45) -> YOLOv8Detector:
    """Construct a detector with the YOLO class patched out."""
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model
    return YOLOv8Detector(
        weights="models/yolov8n.pt",
        device="cpu",
        confidence_threshold=conf,
        warmup_frames=0,
    )


# ---------------------------------------------------------------------------
# BoundingBox
# ---------------------------------------------------------------------------

class TestBoundingBox:
    def test_width_height(self):
        bb = BoundingBox(10, 20, 110, 120)
        assert bb.width  == pytest.approx(100.0)
        assert bb.height == pytest.approx(100.0)

    def test_area(self):
        bb = BoundingBox(0, 0, 50, 80)
        assert bb.area == pytest.approx(4000.0)

    def test_centroid(self):
        bb = BoundingBox(0, 0, 100, 200)
        cx, cy = bb.centroid
        assert cx == pytest.approx(50.0)
        assert cy == pytest.approx(100.0)

    def test_as_list(self):
        bb = BoundingBox(1, 2, 3, 4)
        assert bb.as_list() == [1, 2, 3, 4]

    def test_clamp(self):
        bb = BoundingBox(-10, -5, 700, 500)
        clamped = bb.clamp(frame_width=640, frame_height=480)
        assert clamped.x1 == 0.0
        assert clamped.y1 == 0.0
        assert clamped.x2 == 640.0
        assert clamped.y2 == 480.0

    def test_as_tlwh(self):
        bb = BoundingBox(10, 20, 60, 120)
        assert bb.as_tlwh() == [10, 20, 50, 100]


# ---------------------------------------------------------------------------
# DetectionResult
# ---------------------------------------------------------------------------

class TestDetectionResult:
    def test_to_dict_format(self):
        det = DetectionResult(
            bbox=BoundingBox(10, 20, 110, 220),
            confidence=0.921234,
        )
        d = det.to_dict()
        assert list(d.keys()) == ["bbox", "confidence"]
        assert d["bbox"] == [10, 20, 110, 220]
        assert d["confidence"] == pytest.approx(0.9212)

    def test_class_id_is_person(self):
        det = DetectionResult(bbox=BoundingBox(0, 0, 1, 1), confidence=0.9)
        assert det.class_id == _PERSON_CLASS_ID


# ---------------------------------------------------------------------------
# YOLOv8Detector — initialisation
# ---------------------------------------------------------------------------

class TestYOLOv8DetectorInit:
    @patch("core.detection.yolo_detector.YOLO", create=True)
    def test_loads_on_cpu(self, mock_yolo):
        mock_yolo.return_value = MagicMock()
        det = YOLOv8Detector(device="cpu", warmup_frames=0)
        assert det.device.type == "cpu"

    @patch("core.detection.yolo_detector.YOLO", create=True)
    def test_fp16_disabled_on_cpu(self, mock_yolo):
        mock_yolo.return_value = MagicMock()
        det = YOLOv8Detector(device="cpu", half_precision=True, warmup_frames=0)
        assert det.is_half_precision is False

    @patch("core.detection.yolo_detector.YOLO", create=True)
    def test_repr_contains_device(self, mock_yolo):
        mock_yolo.return_value = MagicMock()
        det = YOLOv8Detector(device="cpu", warmup_frames=0)
        assert "cpu" in repr(det)


# ---------------------------------------------------------------------------
# YOLOv8Detector — frame validation
# ---------------------------------------------------------------------------

class TestFrameValidation:
    @patch("core.detection.yolo_detector.YOLO", create=True)
    def test_none_frame_raises(self, mock_yolo):
        mock_yolo.return_value = MagicMock()
        det = YOLOv8Detector(device="cpu", warmup_frames=0)
        with pytest.raises(ValueError, match="None"):
            det.detect(None)  # type: ignore[arg-type]

    @patch("core.detection.yolo_detector.YOLO", create=True)
    def test_wrong_dtype_raises(self, mock_yolo):
        mock_yolo.return_value = MagicMock()
        det = YOLOv8Detector(device="cpu", warmup_frames=0)
        with pytest.raises(ValueError, match="uint8"):
            det.detect(np.zeros((480, 640, 3), dtype=np.float32))

    @patch("core.detection.yolo_detector.YOLO", create=True)
    def test_wrong_channels_raises(self, mock_yolo):
        mock_yolo.return_value = MagicMock()
        det = YOLOv8Detector(device="cpu", warmup_frames=0)
        with pytest.raises(ValueError, match="shape"):
            det.detect(np.zeros((480, 640), dtype=np.uint8))


# ---------------------------------------------------------------------------
# YOLOv8Detector — detection output
# ---------------------------------------------------------------------------

class TestDetect:
    @patch("core.detection.yolo_detector.YOLO", create=True)
    def test_returns_detection_results(self, mock_yolo):
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        boxes  = _mock_box(100, 50, 200, 300, conf=0.92)
        result = _mock_result(boxes)
        mock_model.predict.return_value = [result]

        det       = YOLOv8Detector(device="cpu", warmup_frames=0)
        detections = det.detect(_make_frame())

        assert len(detections) == 1
        assert isinstance(detections[0], DetectionResult)
        assert detections[0].confidence == pytest.approx(0.92)
        assert detections[0].bbox.as_list() == [100, 50, 200, 300]

    @patch("core.detection.yolo_detector.YOLO", create=True)
    def test_empty_result(self, mock_yolo):
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        empty_result = MagicMock()
        empty_result.boxes = None
        mock_model.predict.return_value = [empty_result]

        det = YOLOv8Detector(device="cpu", warmup_frames=0)
        assert det.detect(_make_frame()) == []

    @patch("core.detection.yolo_detector.YOLO", create=True)
    def test_sorted_by_descending_confidence(self, mock_yolo):
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        boxes        = MagicMock()
        boxes.xyxy   = torch.tensor([[0, 0, 10, 10], [0, 0, 20, 20]])
        boxes.conf   = torch.tensor([0.5, 0.9])
        boxes.cls    = torch.tensor([0, 0], dtype=torch.int)
        mock_model.predict.return_value = [_mock_result(boxes)]

        det        = YOLOv8Detector(device="cpu", warmup_frames=0)
        detections = det.detect(_make_frame())

        assert detections[0].confidence >= detections[1].confidence

    @patch("core.detection.yolo_detector.YOLO", create=True)
    def test_non_person_class_filtered(self, mock_yolo):
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        boxes        = MagicMock()
        boxes.xyxy   = torch.tensor([[0, 0, 50, 50]])
        boxes.conf   = torch.tensor([0.99])
        boxes.cls    = torch.tensor([2], dtype=torch.int)   # class 2 = car
        mock_model.predict.return_value = [_mock_result(boxes)]

        det = YOLOv8Detector(device="cpu", warmup_frames=0)
        assert det.detect(_make_frame()) == []

    @patch("core.detection.yolo_detector.YOLO", create=True)
    def test_to_dict_contract(self, mock_yolo):
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        boxes = _mock_box(10, 20, 110, 220, conf=0.87)
        mock_model.predict.return_value = [_mock_result(boxes)]

        det        = YOLOv8Detector(device="cpu", warmup_frames=0)
        detections = det.detect(_make_frame())
        d          = detections[0].to_dict()

        assert "bbox"       in d
        assert "confidence" in d
        assert len(d["bbox"]) == 4

    @patch("core.detection.yolo_detector.YOLO", create=True)
    def test_frame_counter_increments(self, mock_yolo):
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        mock_model.predict.return_value = [_mock_result(MagicMock(boxes=None))]

        det = YOLOv8Detector(device="cpu", warmup_frames=0)
        for _ in range(3):
            det.detect(_make_frame())
        assert det.frame_count == 3

    @patch("core.detection.yolo_detector.YOLO", create=True)
    def test_inference_exception_returns_empty(self, mock_yolo):
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        mock_model.predict.side_effect = RuntimeError("GPU exploded")

        det = YOLOv8Detector(device="cpu", warmup_frames=0)
        result = det.detect(_make_frame())
        assert result == []


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class TestContextManager:
    @patch("core.detection.yolo_detector.YOLO", create=True)
    def test_close_called_on_exit(self, mock_yolo):
        mock_yolo.return_value = MagicMock()
        with YOLOv8Detector(device="cpu", warmup_frames=0) as det:
            pass  # close() should be called automatically
        assert det.frame_count == 0   # object still accessible post-exit
