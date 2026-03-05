"""
Drop this script in your project root and run it:
    python debug_pipeline.py

It bypasses the WebSocket/FastAPI layer and runs detection + tracking
directly on a single frame so you can see exactly where 0 people comes from.
"""

import cv2
import numpy as np
import sys

# ── 1. Grab one frame from your video ─────────────────────────────────────
VIDEO_PATH = "c.mp4"   # change if needed

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if not ret or frame is None:
    print("ERROR: Could not read a frame from", VIDEO_PATH)
    sys.exit(1)

frame = cv2.resize(frame, (1280, 720))
print(f"[1] Frame grabbed: shape={frame.shape} dtype={frame.dtype}")

# ── 2. Run YOLO detection ──────────────────────────────────────────────────
from core.detection.yolo_detector import YOLOv8Detector

detector = YOLOv8Detector(
    weights="models/yolov8n.pt",
    device="cpu",
    confidence_threshold=0.25,   # lower threshold to see if anything at all fires
    warmup_frames=0,
)

detections = detector.detect(frame)
print(f"\n[2] YOLO detections (conf>=0.25): {len(detections)}")
for d in detections[:5]:
    print(f"    conf={d.confidence:.2f}  bbox={d.bbox.as_list()}  area={d.bbox.area:.0f}")

if not detections:
    print("    !! YOLO found nothing — check model path, confidence threshold,")
    print("       and that your video actually has people in early frames.")
    sys.exit(1)

# ── 3. Run tracker ─────────────────────────────────────────────────────────
from core.tracking.deepsort_tracker import DeepSORTTracker

tracker = DeepSORTTracker(
    max_age=30,
    min_hits=3,
    iou_threshold=0.3,
    max_cosine_distance=0.4,
)

# Feed same frame multiple times so tracks can reach Confirmed (needs min_hits=3)
print(f"\n[3] Feeding frame to tracker 5x (need {tracker._min_hits} hits to confirm)...")
tracks = []
for i in range(5):
    tracks = tracker.update(detections, frame)
    raw = tracker._tracker.update_tracks(
        [(d.bbox.as_tlwh(), d.confidence, "person") for d in detections],
        frame=frame
    )
    states = [getattr(r, "state", "?") for r in raw]
    tsu    = [getattr(r, "time_since_update", "?") for r in raw]
    print(f"    iter {i+1}: confirmed_out={len(tracks)}  raw_tracks={len(raw)}  states={states}  time_since_update={tsu}")

print(f"\n[4] Final confirmed tracks: {len(tracks)}")
for t in tracks:
    print(f"    id={t.id}  speed={t.velocity.speed:.2f}  bbox={t.bbox.as_list()}")

if not tracks:
    print("\n    !! Tracker returned 0 confirmed tracks.")
    print("    Check _map_state fix: time_since_update threshold should be > 1 not > 0")

# ── 4. Check _map_state directly ──────────────────────────────────────────
print("\n[5] Raw DeepSORT track states after 5 updates:")
raw_tracks = tracker._tracker.update_tracks(
    [(d.bbox.as_tlwh(), d.confidence, "person") for d in detections],
    frame=frame
)
for r in raw_tracks:
    sid   = getattr(r, "track_id", "?")
    state = getattr(r, "state", "?")
    tsu   = getattr(r, "time_since_update", "?")
    hits  = getattr(r, "hits", "?")
    print(f"    track_id={sid}  state={state}  time_since_update={tsu}  hits={hits}")