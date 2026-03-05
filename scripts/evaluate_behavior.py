#!/usr/bin/env python3
"""
scripts/evaluate_behavior.py
==============================
Offline batch evaluation of the crowd behaviour analysis pipeline.

What this script does
---------------------
1. Discovers every ``.mp4`` (and common video) file under ``data/test_videos/``.
2. For each video, runs the full stack:
       VideoCapture (OpenCV)  →  YOLOv8Detector  →  DeepSORTTracker
       →  BehaviorAnalyzer  →  per-frame records
3. Writes raw frame-level results to ``reports/evaluation_results.csv``.
4. Derives ground-truth labels from the file name when possible (see
   ``_label_from_filename``), then computes sklearn classification metrics.
5. Saves a JSON summary to ``reports/evaluation_summary.json``.

Running
-------
    python scripts/evaluate_behavior.py

    # evaluate only specific files
    python scripts/evaluate_behavior.py --videos data/test_videos/crowd_running.mp4

    # limit frames per video (useful for quick CI checks)
    python scripts/evaluate_behavior.py --max-frames 100

    # skip sklearn metrics (e.g. no ground-truth labels available)
    python scripts/evaluate_behavior.py --no-metrics

Ground-truth label inference
-----------------------------
The script derives a ground-truth label from the video filename using a simple
keyword mapping.  Override the mapping by adding ``--label-map`` JSON argument
or by editing ``_FILENAME_LABEL_MAP`` below.

Keyword matching is case-insensitive and matches substrings::

    "normal"       → normal
    "walk"         → normal
    "running"      → running
    "sprint"       → running
    "fight"        → violence
    "violence"     → violence
    "panic"        → panic
    "stampede"     → panic
    "surge"        → crowd_surge
    "suspicious"   → suspicion

If no keyword matches, the ground-truth label is ``None`` and that video is
excluded from accuracy / F1 computation (but raw results are still saved).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
)
logger = logging.getLogger("crowd_analysis.evaluate")

# ── Paths ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT  = Path(__file__).resolve().parent.parent
_VIDEO_DIR     = _PROJECT_ROOT / "data" / "test_videos"
_REPORTS_DIR   = _PROJECT_ROOT / "reports"
_CSV_PATH      = _REPORTS_DIR / "evaluation_results.csv"
_SUMMARY_PATH  = _REPORTS_DIR / "evaluation_summary.json"

# Supported video extensions
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

# ── Ground-truth keyword mapping ──────────────────────────────────────────────
_FILENAME_LABEL_MAP: Dict[str, str] = {
    "normal":     "normal",
    "walk":       "normal",
    "idle":       "normal",
    "running":    "running",
    "sprint":     "running",
    "run":        "running",
    "fight":      "violence",
    "violence":   "violence",
    "aggress":    "violence",
    "panic":      "panic",
    "stampede":   "panic",
    "chaos":      "panic",
    "surge":      "crowd_surge",
    "rush":       "crowd_surge",
    "suspect":    "suspicion",
    "suspicious": "suspicion",
    "abnormal":   "suspicion",
}


# ══════════════════════════════════════════════════════════════════════════════
# Ground-truth helpers
# ══════════════════════════════════════════════════════════════════════════════

def _label_from_filename(name: str) -> Optional[str]:
    """
    Infer a ground-truth label from a video filename using keyword matching.

    Parameters
    ----------
    name : str — filename (with or without extension).

    Returns
    -------
    str | None — matched label, or None if no keyword matched.
    """
    lower = name.lower()
    for keyword, label in _FILENAME_LABEL_MAP.items():
        if keyword in lower:
            return label
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Video processing
# ══════════════════════════════════════════════════════════════════════════════

def _process_video(
    video_path:  Path,
    detector:    object,
    tracker:     object,
    analyzer:    object,
    max_frames:  Optional[int],
) -> List[Dict]:
    """
    Run the full detection → tracking → behavior pipeline on a single video.

    Parameters
    ----------
    video_path  : Path to the video file.
    detector    : YOLOv8Detector instance.
    tracker     : DeepSORTTracker instance.
    analyzer    : BehaviorAnalyzer instance.
    max_frames  : Stop after this many frames (None = process all).

    Returns
    -------
    list of dicts, one per processed frame::

        {
            "video_name": str,
            "frame_index": int,
            "people_count": int,
            "predicted_behavior": str,
            "confidence": float,
        }
    """
    import cv2  # local import — avoids hard dep at module level

    records: List[Dict] = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Could not open video: %s", video_path)
        return records

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    logger.info(
        "Processing '%s'  —  %d frames @ %.1f fps",
        video_path.name, total_frames, video_fps,
    )

    # Reset tracker / analyzer state between videos
    if hasattr(tracker, "reset"):
        tracker.reset()
    if hasattr(analyzer, "reset"):
        analyzer.reset()

    frame_idx = 0
    t_video_start = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if max_frames is not None and frame_idx >= max_frames:
                break

            # Detection
            try:
                detections = detector.detect(frame)
            except Exception as exc:
                logger.warning("Detection failed frame %d: %s", frame_idx, exc)
                detections = []

            # Tracking
            try:
                tracks = tracker.update(detections, frame)
            except Exception as exc:
                logger.warning("Tracking failed frame %d: %s", frame_idx, exc)
                tracks = []

            # Behavior analysis
            try:
                result = analyzer.analyze(tracks, frame_index=frame_idx)
            except Exception as exc:
                logger.warning("Analysis failed frame %d: %s", frame_idx, exc)
                from core.behavior.base_analyzer import BehaviorLabel, BehaviorResult, FrameFeatures
                result = BehaviorResult(
                    label=BehaviorLabel.NORMAL, confidence=0.0,
                    frame_index=frame_idx, track_labels={},
                    features=FrameFeatures.empty(frame_idx), signals=[],
                )

            records.append({
                "video_name":         video_path.name,
                "frame_index":        frame_idx,
                "people_count":       len(tracks),
                "predicted_behavior": result.label.value,
                "confidence":         round(float(result.confidence), 4),
            })

            frame_idx += 1

    finally:
        cap.release()

    elapsed = time.perf_counter() - t_video_start
    logger.info(
        "  Done — %d frames in %.1f s (%.1f fps processing)",
        frame_idx, elapsed, frame_idx / max(elapsed, 0.001),
    )

    return records


# ══════════════════════════════════════════════════════════════════════════════
# Metrics computation
# ══════════════════════════════════════════════════════════════════════════════

def _compute_metrics(
    records:       List[Dict],
    gt_label_map:  Dict[str, Optional[str]],  # video_name → gt_label
) -> Dict:
    """
    Compute sklearn classification metrics for videos with a known GT label.

    The predicted label for a video is the **majority vote** across all its
    frames.  Videos without a ground-truth label are excluded.

    Parameters
    ----------
    records       : all per-frame records from ``_process_video``.
    gt_label_map  : mapping from video filename to ground-truth label.

    Returns
    -------
    dict with ``accuracy``, ``precision``, ``recall``, ``f1_score``,
    ``support``, ``class_report``, and ``per_video`` breakdown.
    """
    try:
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            f1_score,
            precision_score,
            recall_score,
        )
    except ImportError:
        logger.warning(
            "scikit-learn is not installed — skipping sklearn metrics. "
            "Run: pip install scikit-learn"
        )
        return {"error": "scikit-learn not installed"}

    from collections import Counter

    # Group predicted labels per video and take the majority vote
    video_preds: Dict[str, List[str]] = {}
    for r in records:
        video_preds.setdefault(r["video_name"], []).append(r["predicted_behavior"])

    y_true: List[str] = []
    y_pred: List[str] = []
    per_video: List[Dict] = []

    for video_name, preds in video_preds.items():
        gt = gt_label_map.get(video_name)
        if gt is None:
            logger.debug(
                "No ground-truth label for '%s' — skipping metrics.", video_name
            )
            continue

        majority_label = Counter(preds).most_common(1)[0][0]
        y_true.append(gt)
        y_pred.append(majority_label)

        per_video.append({
            "video":           video_name,
            "ground_truth":    gt,
            "predicted":       majority_label,
            "correct":         gt == majority_label,
            "frame_count":     len(preds),
            "label_breakdown": dict(Counter(preds)),
        })

    if not y_true:
        return {
            "note": "No videos with ground-truth labels found — no metrics computed.",
            "per_video": per_video,
        }

    labels_present = sorted(set(y_true + y_pred))
    avg = "weighted" if len(labels_present) > 2 else "binary"

    try:
        accuracy  = float(accuracy_score(y_true, y_pred))
        precision = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
        recall    = float(recall_score(y_true, y_pred,    average=avg, zero_division=0))
        f1        = float(f1_score(y_true, y_pred,        average=avg, zero_division=0))
        report    = classification_report(
            y_true, y_pred, labels=labels_present, zero_division=0
        )
    except Exception as exc:
        logger.error("sklearn metric computation failed: %s", exc)
        return {"error": str(exc), "per_video": per_video}

    logger.info(
        "Metrics (n=%d videos) — accuracy=%.3f  precision=%.3f  "
        "recall=%.3f  f1=%.3f",
        len(y_true), accuracy, precision, recall, f1,
    )
    logger.info("Classification report:\n%s", report)

    return {
        "accuracy":       round(accuracy,  4),
        "precision":      round(precision, 4),
        "recall":         round(recall,    4),
        "f1_score":       round(f1,        4),
        "support":        len(y_true),
        "class_report":   report,
        "per_video":      per_video,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CSV / JSON helpers
# ══════════════════════════════════════════════════════════════════════════════

def _write_csv(records: List[Dict], path: Path) -> None:
    """Write per-frame records to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["video_name", "frame_index", "people_count",
                  "predicted_behavior", "confidence"]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    logger.info("CSV written → %s  (%d rows)", path, len(records))


def _write_summary(summary: Dict, path: Path) -> None:
    """Write evaluation summary to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary written → %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch evaluation of the crowd behaviour pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--videos", nargs="*", metavar="PATH",
        help="Specific video files to evaluate (default: all in data/test_videos/).",
    )
    parser.add_argument(
        "--video-dir", default=str(_VIDEO_DIR), metavar="DIR",
        help="Directory to search for test videos.",
    )
    parser.add_argument(
        "--csv", default=str(_CSV_PATH), metavar="FILE",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--summary", default=str(_SUMMARY_PATH), metavar="FILE",
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, metavar="N",
        help="Limit frames per video (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--no-metrics", action="store_true",
        help="Skip sklearn metric computation.",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Compute device for YOLO inference (cpu / cuda / auto).",
    )
    parser.add_argument(
        "--conf-threshold", type=float, default=0.45,
        help="YOLO confidence threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Discover videos ───────────────────────────────────────────────────
    if args.videos:
        video_paths = [Path(p) for p in args.videos]
    else:
        video_dir = Path(args.video_dir)
        if not video_dir.exists():
            logger.error("Video directory does not exist: %s", video_dir)
            sys.exit(1)
        video_paths = [
            p for p in sorted(video_dir.iterdir())
            if p.suffix.lower() in _VIDEO_EXTENSIONS
        ]

    if not video_paths:
        logger.error(
            "No video files found in '%s'. "
            "Add .mp4 files to data/test_videos/ and re-run.",
            args.video_dir,
        )
        sys.exit(1)

    logger.info("Found %d video(s) to evaluate.", len(video_paths))

    # ── Build pipeline components ──────────────────────────────────────────
    logger.info("Initialising pipeline components …")

    try:
        from core.detection.yolo_detector    import YOLOv8Detector
        from core.tracking.deepsort_tracker  import DeepSORTTracker
        from core.behavior.behavior_analyzer import BehaviorAnalyzer

        detector = YOLOv8Detector(
            device=args.device,
            confidence_threshold=args.conf_threshold,
            warmup_frames=1,
        )
        tracker  = DeepSORTTracker()
        analyzer = BehaviorAnalyzer()
    except Exception as exc:
        logger.critical("Failed to initialise pipeline: %s", exc, exc_info=True)
        sys.exit(1)

    # ── Process each video ────────────────────────────────────────────────
    all_records:   List[Dict] = []
    gt_label_map:  Dict[str, Optional[str]] = {}

    t_total_start = time.perf_counter()

    for vpath in video_paths:
        if not vpath.exists():
            logger.warning("Video not found: %s — skipping.", vpath)
            continue

        gt_label_map[vpath.name] = _label_from_filename(vpath.stem)
        if gt_label_map[vpath.name]:
            logger.info(
                "  Ground-truth label for '%s': %s",
                vpath.name, gt_label_map[vpath.name],
            )
        else:
            logger.info(
                "  No ground-truth label matched for '%s' — excluded from metrics.",
                vpath.name,
            )

        records = _process_video(
            video_path=vpath,
            detector=detector,
            tracker=tracker,
            analyzer=analyzer,
            max_frames=args.max_frames,
        )
        all_records.extend(records)

    total_elapsed = time.perf_counter() - t_total_start
    logger.info(
        "All videos processed — %d total frames in %.1f s.",
        len(all_records), total_elapsed,
    )

    # ── Write CSV ─────────────────────────────────────────────────────────
    _write_csv(all_records, Path(args.csv))

    # ── Compute metrics ───────────────────────────────────────────────────
    if args.no_metrics:
        metrics = {"note": "Metrics skipped via --no-metrics flag."}
    else:
        metrics = _compute_metrics(all_records, gt_label_map)

    # ── Write summary JSON ────────────────────────────────────────────────
    summary = {
        "evaluated_videos":  len(video_paths),
        "total_frames":      len(all_records),
        "total_elapsed_s":   round(total_elapsed, 2),
        "csv_path":          str(Path(args.csv).resolve()),
        "metrics":           metrics,
    }
    _write_summary(summary, Path(args.summary))

    logger.info("Evaluation complete.")
    if "accuracy" in metrics:
        logger.info(
            "Overall accuracy=%.3f  precision=%.3f  recall=%.3f  F1=%.3f",
            metrics["accuracy"], metrics["precision"],
            metrics["recall"],   metrics["f1_score"],
        )


if __name__ == "__main__":
    main()