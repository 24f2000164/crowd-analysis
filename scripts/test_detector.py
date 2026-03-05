import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.detection.yolo_detector import YOLOv8Detector
import cv2

detector = YOLOv8Detector(weights="models/yolov8n.pt")

cap = cv2.VideoCapture(r"C:\Users\kumar\Desktop\crowd_analysis\c.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)

    print("detections:", len(detections))

    for d in detections:
        x1,y1,x2,y2 = map(int,d.bbox.as_list())
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.imshow("test", frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()