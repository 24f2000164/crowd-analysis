
import numpy as np

from core.detection.yolo_detector import YOLOv8Detector


def test_detector_initialization():

    detector = YOLOv8Detector(weights="models/yolov8n.pt")

    assert detector is not None


def test_detector_detect():

    detector = YOLOv8Detector(weights="models/yolov8n.pt")

    frame = np.zeros((640,480,3),dtype=np.uint8)

    detections = detector.detect(frame)

    assert isinstance(detections,list)

