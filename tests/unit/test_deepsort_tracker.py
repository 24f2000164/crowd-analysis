"""
tests/unit/test_deepsort_tracker.py
=====================================
Unit tests for the DeepSORT tracking module.

All tests mock ``deep_sort_realtime`` so the suite runs without the package
installed and without a GPU — making it safe to execute in CI.
"""

from __future__ import annotations

import math
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.detection.yolo_detector import BoundingBox, DetectionResult
from core.tracking.track_state import TrackState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_detection(
    x1: float = 100, y1: float = 50,
    x2: float = 200, y2: float = 300,
    conf: float = 0.85,
) -> DetectionResult:
    return DetectionResult(
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        confidence=conf,
    )


def _make_raw_track(
    track_id: int,
    ltrb: List[float],
    state: str = "Confirmed",
    age: int = 5,
    hits: int = 5,
    time_since_update: int = 0,
) -> MagicMock:
    """Build a mock object that mimics a deep_sort_realtime track."""
    t = MagicMock()
    t.track_id           = track_id
    t.state              = state
    t.age                = age
    t.hits               = hits
    t.time_since_update  = time_since_update
    t.to_ltrb.return_value = ltrb
    return t


def _make_tracker(**kwargs):
    """Construct a DeepSORTTracker with the real DeepSort engine mocked."""
    with patch("core.tracking.deepsort_tracker.DeepSort", create=True) as mock_ds:
        mock_ds.return_value = MagicMock()
        from core.tracking.deepsort_tracker import DeepSORTTracker
        tracker = DeepSORTTracker(**kwargs)
        # Replace the already-built internal tracker with a fresh mock
        tracker._tracker = MagicMock()
        return tracker


# ---------------------------------------------------------------------------
# VelocityVector
# ---------------------------------------------------------------------------

class TestVelocityVector:
    def test_zero(self):
        from core.tracking.deepsort_tracker import VelocityVector
        v = VelocityVector.zero()
        assert v.speed == 0.0
        assert v.vx    == 0.0
        assert v.vy    == 0.0

    def test_from_delta_speed(self):
        from core.tracking.deepsort_tracker import VelocityVector
        v = VelocityVector.from_delta(3.0, 4.0)
        assert v.speed == pytest.approx(5.0)

    def test_from_delta_direction_right(self):
        from core.tracking.deepsort_tracker import VelocityVector
        # Moving purely right: dx=1, dy=0 → compass 90°
        v = VelocityVector.from_delta(1.0, 0.0)
        assert v.direction_deg == pytest.approx(90.0)

    def test_from_delta_direction_down(self):
        from core.tracking.deepsort_tracker import VelocityVector
        # Moving purely down: dx=0, dy=1 → compass 180°
        v = VelocityVector.from_delta(0.0, 1.0)
        assert v.direction_deg == pytest.approx(180.0)

    def test_to_dict_keys(self):
        from core.tracking.deepsort_tracker import VelocityVector
        d = VelocityVector.from_delta(1.0, 2.0).to_dict()
        assert set(d.keys()) == {"vx", "vy", "speed", "direction_deg"}


# ---------------------------------------------------------------------------
# TrackedPerson
# ---------------------------------------------------------------------------

class TestTrackedPerson:
    def _make_person(self, track_id: int = 1) -> object:
        from core.tracking.deepsort_tracker import TrackedPerson, VelocityVector
        return TrackedPerson(
            id=track_id,
            bbox=BoundingBox(10, 20, 110, 220),
            velocity=VelocityVector.from_delta(3.0, 4.0),
        )

    def test_to_dict_contract(self):
        p = self._make_person()
        d = p.to_dict()
        assert set(d.keys()) == {"id", "bbox", "velocity"}
        assert d["id"]       == 1
        assert d["bbox"]     == [10, 20, 110, 220]
        assert d["velocity"] == pytest.approx(5.0)

    def test_to_full_dict_has_centroid(self):
        p = self._make_person()
        d = p.to_full_dict()
        assert "centroid" in d
        assert d["centroid"] == [60.0, 120.0]

    def test_repr_contains_id(self):
        p = self._make_person(track_id=42)
        assert "42" in repr(p)


# ---------------------------------------------------------------------------
# TrackState
# ---------------------------------------------------------------------------

class TestTrackState:
    def test_confirmed_is_output_ready(self):
        assert TrackState.Confirmed.is_output_ready is True

    def test_tentative_is_not_output_ready(self):
        assert TrackState.Tentative.is_output_ready is False

    def test_deleted_is_not_active(self):
        assert TrackState.Deleted.is_active is False

    def test_confirmed_is_active(self):
        assert TrackState.Confirmed.is_active is True


# ---------------------------------------------------------------------------
# DeepSORTTracker — initialisation
# ---------------------------------------------------------------------------

class TestDeepSORTTrackerInit:
    def test_default_construction(self):
        tracker = _make_tracker()
        assert tracker.frame_index       == 0
        assert tracker.active_track_count == 0

    def test_custom_params_stored(self):
        tracker = _make_tracker(max_age=50, min_hits=5)
        assert tracker._max_age   == 50
        assert tracker._min_hits  == 5

    def test_repr_contains_max_age(self):
        tracker = _make_tracker(max_age=20)
        assert "20" in repr(tracker)


# ---------------------------------------------------------------------------
# DeepSORTTracker — update()
# ---------------------------------------------------------------------------

class TestDeepSORTTrackerUpdate:
    def test_empty_detections_returns_empty(self):
        tracker = _make_tracker()
        tracker._tracker.update_tracks.return_value = []
        result = tracker.update([], _make_frame())
        assert result == []

    def test_confirmed_track_returned(self):
        tracker = _make_tracker()
        raw = _make_raw_track(1, [100, 50, 200, 300], state="Confirmed")
        tracker._tracker.update_tracks.return_value = [raw]

        result = tracker.update([_make_detection()], _make_frame())
        assert len(result) == 1
        assert result[0].id == 1

    def test_tentative_track_not_returned(self):
        tracker = _make_tracker()
        raw = _make_raw_track(2, [100, 50, 200, 300], state="Tentative")
        tracker._tracker.update_tracks.return_value = [raw]

        result = tracker.update([_make_detection()], _make_frame())
        assert result == []

    def test_to_dict_output_contract(self):
        tracker = _make_tracker()
        raw = _make_raw_track(3, [10, 20, 110, 220], state="Confirmed")
        tracker._tracker.update_tracks.return_value = [raw]

        tracks = tracker.update([_make_detection()], _make_frame())
        d = tracks[0].to_dict()
        assert list(d.keys()) == ["id", "bbox", "velocity"]
        assert d["id"]        == 3

    def test_frame_index_increments(self):
        tracker = _make_tracker()
        tracker._tracker.update_tracks.return_value = []
        tracker.update([], _make_frame())
        tracker.update([], _make_frame())
        assert tracker.frame_index == 2

    def test_invalid_frame_skipped(self):
        tracker = _make_tracker()
        result = tracker.update([], None)  # type: ignore[arg-type]
        assert result == []
        assert tracker.frame_index == 1   # still incremented

    def test_multiple_tracks_sorted_by_id(self):
        tracker = _make_tracker()
        raw_a = _make_raw_track(5, [10, 10, 60, 100], state="Confirmed")
        raw_b = _make_raw_track(2, [70, 10, 120, 100], state="Confirmed")
        tracker._tracker.update_tracks.return_value = [raw_a, raw_b]

        tracks = tracker.update([_make_detection()], _make_frame())
        assert [t.id for t in tracks] == [2, 5]

    def test_track_registered_in_registry(self):
        tracker = _make_tracker()
        raw = _make_raw_track(7, [10, 10, 60, 100], state="Confirmed")
        tracker._tracker.update_tracks.return_value = [raw]

        tracker.update([_make_detection()], _make_frame())
        assert 7 in tracker._registry

    def test_velocity_is_zero_first_frame(self):
        tracker = _make_tracker()
        raw = _make_raw_track(1, [100, 50, 200, 300], state="Confirmed")
        tracker._tracker.update_tracks.return_value = [raw]

        tracks = tracker.update([_make_detection()], _make_frame())
        # Only one centroid in history — velocity must be zero
        assert tracks[0].velocity.speed == pytest.approx(0.0)

    def test_velocity_non_zero_after_movement(self):
        tracker = _make_tracker()

        # Frame 1: bbox at x=[100,200]
        raw1 = _make_raw_track(1, [100, 50, 200, 300], state="Confirmed")
        tracker._tracker.update_tracks.return_value = [raw1]
        tracker.update([_make_detection()], _make_frame())

        # Frame 2: bbox shifted 50px right
        raw2 = _make_raw_track(1, [150, 50, 250, 300], state="Confirmed")
        tracker._tracker.update_tracks.return_value = [raw2]
        tracks = tracker.update([_make_detection()], _make_frame())

        assert tracks[0].velocity.speed > 0.0


# ---------------------------------------------------------------------------
# DeepSORTTracker — reset()
# ---------------------------------------------------------------------------

class TestDeepSORTTrackerReset:
    def test_registry_cleared_on_reset(self):
        tracker = _make_tracker()
        raw = _make_raw_track(1, [10, 10, 60, 100], state="Confirmed")
        tracker._tracker.update_tracks.return_value = [raw]
        tracker.update([_make_detection()], _make_frame())

        assert len(tracker._registry) > 0
        tracker.reset()
        assert len(tracker._registry) == 0
        assert tracker.frame_index    == 0

    def test_history_cleared_on_reset(self):
        tracker = _make_tracker()
        tracker.reset()
        assert tracker.get_track_history(99) == []


# ---------------------------------------------------------------------------
# DeepSORTTracker — get_track_history()
# ---------------------------------------------------------------------------

class TestTrackHistory:
    def test_unknown_id_returns_empty(self):
        tracker = _make_tracker()
        assert tracker.get_track_history(999) == []

    def test_history_grows_over_frames(self):
        tracker = _make_tracker()
        for i in range(5):
            raw = _make_raw_track(1, [100 + i * 10, 50, 200 + i * 10, 300],
                                   state="Confirmed")
            tracker._tracker.update_tracks.return_value = [raw]
            tracker.update([_make_detection()], _make_frame())

        history = tracker.get_track_history(1)
        assert len(history) == 5


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_reset_called_on_exit(self):
        tracker = _make_tracker()
        raw = _make_raw_track(1, [10, 10, 60, 100], state="Confirmed")
        tracker._tracker.update_tracks.return_value = [raw]
        tracker.update([_make_detection()], _make_frame())

        with tracker:
            pass  # __exit__ calls reset()

        assert tracker.frame_index == 0
