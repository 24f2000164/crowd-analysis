"""
core/behavior/trajectory_store.py
===================================
Per-track rolling centroid trajectory store.

Responsibility
--------------
Maintain a fixed-length deque of ``(cx, cy)`` centroid positions for every
active track so that velocity, acceleration, and direction-change calculations
always have access to historical position data without keeping unbounded
memory.

This store is the *single source of truth* for positional history within the
behavior pipeline.  The tracker (DeepSORT) owns track identity; this store
owns the behavioral motion history.

Lifecycle
---------
- Tracks are created lazily on first ``update()`` call with that ID.
- Tracks are pruned automatically when their ID is absent from the active
  set passed to ``prune()``.
- ``clear()`` resets all state (use between camera switches / scene changes).

Usage
-----
    from core.behavior.trajectory_store import TrajectoryStore

    store = TrajectoryStore(history_length=60)

    # Call once per frame with the list of confirmed track IDs + centroids
    store.update([(1, (320.0, 240.0)), (2, (500.0, 300.0))])
    store.prune(active_ids={1, 2})

    history = store.get(track_id=1)   # [(cx0,cy0), (cx1,cy1), ...]
    all_h   = store.get_all()         # {1: [...], 2: [...]}
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Deque, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("crowd_analysis.behavior.trajectory_store")


# ---------------------------------------------------------------------------
# Internal record
# ---------------------------------------------------------------------------

@dataclass
class _TrackHistory:
    """Mutable centroid history for a single track."""
    track_id:      int
    centroids:     Deque[Tuple[float, float]]
    frame_created: int = 0
    last_seen:     int = 0

    @classmethod
    def create(
        cls,
        track_id:      int,
        centroid:      Tuple[float, float],
        maxlen:        int,
        frame_index:   int,
    ) -> "_TrackHistory":
        d: Deque[Tuple[float, float]] = deque(maxlen=maxlen)
        d.append(centroid)
        return cls(
            track_id=track_id,
            centroids=d,
            frame_created=frame_index,
            last_seen=frame_index,
        )


# ---------------------------------------------------------------------------
# Public store
# ---------------------------------------------------------------------------

class TrajectoryStore:
    """
    Thread-safe rolling centroid history store.

    Parameters
    ----------
    history_length : int
        Maximum number of centroid positions retained per track.
        Oldest entries are silently dropped when the deque is full.
    """

    def __init__(self, history_length: int = 60) -> None:
        if history_length < 2:
            raise ValueError("history_length must be ≥ 2.")
        self._maxlen = history_length
        self._store:  Dict[int, _TrackHistory] = {}
        self._lock = Lock()
        self._frame_index: int = 0

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def update(
        self,
        id_centroid_pairs: List[Tuple[int, Tuple[float, float]]],
        frame_index: Optional[int] = None,
    ) -> None:
        """
        Append the latest centroid for each track.

        Parameters
        ----------
        id_centroid_pairs : list of (track_id, (cx, cy)).
        frame_index       : optional override; defaults to internal counter.
        """
        fidx = frame_index if frame_index is not None else self._frame_index

        with self._lock:
            for track_id, centroid in id_centroid_pairs:
                if track_id in self._store:
                    rec = self._store[track_id]
                    rec.centroids.append(centroid)
                    rec.last_seen = fidx
                else:
                    self._store[track_id] = _TrackHistory.create(
                        track_id=track_id,
                        centroid=centroid,
                        maxlen=self._maxlen,
                        frame_index=fidx,
                    )
                    logger.debug(
                        "Trajectory created for track %d at frame %d.",
                        track_id, fidx,
                    )

        self._frame_index = fidx + 1

    def prune(self, active_ids: Set[int]) -> None:
        """
        Remove history for tracks that are no longer active.

        Parameters
        ----------
        active_ids : set of track IDs present in the current frame.
        """
        with self._lock:
            dead = [tid for tid in self._store if tid not in active_ids]
            for tid in dead:
                del self._store[tid]
                logger.debug("Trajectory pruned for track %d.", tid)

    def clear(self) -> None:
        """Delete all history. Call between scenes or camera switches."""
        with self._lock:
            self._store.clear()
            self._frame_index = 0
        logger.info("TrajectoryStore cleared.")

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get(self, track_id: int) -> List[Tuple[float, float]]:
        """
        Return the centroid history for a single track.

        Parameters
        ----------
        track_id : int

        Returns
        -------
        list of (cx, cy), oldest first.  Empty list if ID is unknown.
        """
        with self._lock:
            rec = self._store.get(track_id)
            return list(rec.centroids) if rec else []

    def get_all(self) -> Dict[int, List[Tuple[float, float]]]:
        """
        Return centroid histories for all tracked persons.

        Returns
        -------
        dict mapping track_id → list of (cx, cy), oldest first.
        """
        with self._lock:
            return {tid: list(rec.centroids) for tid, rec in self._store.items()}

    def track_age(self, track_id: int) -> int:
        """Number of centroids stored for a given track (proxy for age in frames)."""
        with self._lock:
            rec = self._store.get(track_id)
            return len(rec.centroids) if rec else 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active_track_count(self) -> int:
        """Number of tracks currently in the store."""
        return len(self._store)

    @property
    def history_length(self) -> int:
        """Maximum deque length configured at construction."""
        return self._maxlen

    def __repr__(self) -> str:
        return (
            f"TrajectoryStore(tracks={self.active_track_count}, "
            f"maxlen={self._maxlen})"
        )
