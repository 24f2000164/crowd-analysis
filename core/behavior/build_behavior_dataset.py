"""
scripts/build_behavior_dataset.py
===================================
Dataset builder for the ML-based crowd behavior model.

Reads stored trajectory logs (JSON-Lines format written by the pipeline),
recomputes feature vectors using TrajectoryFeatureExtractor, and exports
a CSV dataset ready for model training.

Expected input log format (one JSON object per line):
    {
        "frame_index": 42,
        "label": "panic",
        "tracks": [
            {"id": 1, "bbox": [x1,y1,x2,y2], "velocity": {"speed": 12.3, "direction_deg": 45.0}},
            ...
        ]
    }

Output CSV columns:
    frame_window, velocity_mean, velocity_variance, acceleration_mean,
    acceleration_spikes, direction_entropy, crowd_density,
    density_change_rate, trajectory_dispersion, track_collision_rate, label

Usage
-----
    python scripts/build_behavior_dataset.py \
        --input  data/trajectory_logs/          \
        --output data/crowd_behavior_dataset.csv \
        --window 30
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.behavior.trajectory_features import (
    FeatureVector,
    TrajectoryFeatureExtractor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
)
logger = logging.getLogger("build_behavior_dataset")

# ── CSV column order ───────────────────────────────────────────────────────
_CSV_COLUMNS = [
    "frame_window",
    "velocity_mean",
    "velocity_variance",
    "acceleration_mean",
    "acceleration_spikes",
    "direction_entropy",
    "crowd_density",
    "density_change_rate",
    "trajectory_dispersion",
    "track_collision_rate",
    "label",
]


# ── Lightweight proxy so the extractor can consume log data ───────────────

@dataclass
class _BBoxProxy:
    x1: float; y1: float; x2: float; y2: float

    @property
    def centroid(self):
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0


@dataclass
class _VelocityProxy:
    speed:         float
    direction_deg: float


@dataclass
class _TrackProxy:
    id:       int
    bbox:     _BBoxProxy
    velocity: _VelocityProxy


def _parse_tracks(raw_tracks: list) -> List[_TrackProxy]:
    """Convert raw log dicts into lightweight proxy objects."""
    proxies = []
    for t in raw_tracks:
        try:
            x1, y1, x2, y2 = t["bbox"]
            vel = t.get("velocity", {})
            proxies.append(_TrackProxy(
                id=int(t["id"]),
                bbox=_BBoxProxy(x1, y1, x2, y2),
                velocity=_VelocityProxy(
                    speed=float(vel.get("speed", 0.0)),
                    direction_deg=float(vel.get("direction_deg", 0.0)),
                ),
            ))
        except (KeyError, ValueError, TypeError) as exc:
            logger.debug("Skipping malformed track entry: %s", exc)
    return proxies


# ── Log file reader ───────────────────────────────────────────────────────

def _iter_log_lines(input_path: Path) -> Iterator[dict]:
    """
    Yield parsed JSON objects from all *.jsonl files under input_path.

    Accepts either a single file or a directory of files.
    """
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(input_path.glob("**/*.jsonl"))
        if not files:
            # also accept .json extension
            files = sorted(input_path.glob("**/*.json"))
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    logger.info("Found %d log file(s) under %s", len(files), input_path)

    for fpath in files:
        logger.info("Reading %s …", fpath.name)
        with fpath.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "%s line %d: JSON parse error — %s", fpath.name, lineno, exc
                    )


# ── Feature extraction over log stream ────────────────────────────────────

def build_dataset(
    input_path:  Path,
    output_path: Path,
    window_size: int   = 30,
    frame_shape: tuple = (720, 1280),
    min_tracks:  int   = 2,
) -> int:
    """
    Build and export the CSV dataset.

    Returns
    -------
    int — number of rows written.
    """
    extractor = TrajectoryFeatureExtractor(
        frame_shape=frame_shape,
        window_size=window_size,
        min_tracks=min_tracks,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    current_label: Optional[str] = None

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=_CSV_COLUMNS)
        writer.writeheader()

        for entry in _iter_log_lines(input_path):
            frame_index    = int(entry.get("frame_index", 0))
            label          = str(entry.get("label", "normal")).lower()
            raw_tracks     = entry.get("tracks", [])
            tracks         = _parse_tracks(raw_tracks)
            current_label  = label

            extractor.update(tracks, frame_index=frame_index)
            vec: Optional[FeatureVector] = extractor.compute_features()

            if vec is None:
                continue

            row = {
                "frame_window": f"{vec.frame_window_start}-{vec.frame_window_end}",
                **vec.to_dict(),
                "label": current_label,
            }
            writer.writerow(row)
            rows_written += 1

            if rows_written % 500 == 0:
                logger.info("Written %d rows …", rows_written)

    logger.info(
        "Dataset export complete — %d rows → %s", rows_written, output_path
    )
    return rows_written


# ── CLI ───────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a CSV behavior dataset from trajectory logs."
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("data/trajectory_logs"),
        help="Path to a .jsonl file or directory of .jsonl files.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/crowd_behavior_dataset.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--window", "-w",
        type=int,
        default=30,
        help="Sliding window size in frames (default: 30).",
    )
    parser.add_argument(
        "--height", type=int, default=720,  help="Frame height in pixels."
    )
    parser.add_argument(
        "--width",  type=int, default=1280, help="Frame width in pixels."
    )
    parser.add_argument(
        "--min-tracks", type=int, default=2,
        help="Minimum tracks per window to emit a row.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    n = build_dataset(
        input_path  = args.input,
        output_path = args.output,
        window_size = args.window,
        frame_shape = (args.height, args.width),
        min_tracks  = args.min_tracks,
    )
    if n == 0:
        logger.error("No rows were written — check your input logs.")
        sys.exit(1)
    logger.info("Done. %d samples exported to %s", n, args.output)