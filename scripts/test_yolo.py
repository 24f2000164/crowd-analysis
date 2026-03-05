from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
print("Model loaded:", model)

cap = cv2.VideoCapture("C:/Users/kumar/Desktop/crowd_analysis/c.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for r in results:
        print("detections:", len(r.boxes))

    if cv2.waitKey(1) == 27:
        break
 