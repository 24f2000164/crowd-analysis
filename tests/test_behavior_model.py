"""
tests/test_behavior_model.py
==============================
Unit tests for the ML-based crowd behavior pipeline.

Covers:
  1. TrajectoryFeatureExtractor  — feature computation correctness
  2. MLBehaviorClassifier        — model prediction (mocked model)
  3. BehaviorAnalyzer integration — end-to-end with dummy tracks

Run with:
    pytest tests/test_behavior_model.py -v
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Allow import from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.behavior.trajectory_features import (
    FeatureVector,
    TrajectoryFeatureExtractor,
    _FEATURE_NAMES,
)


# ============================================================================
# Fixtures & helpers
# ============================================================================

@dataclass
class _MockBBox:
    x1: float; y1: float; x2: float; y2: float

    @property
    def centroid(self):
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0


@dataclass
class _MockVelocity:
    speed:         float
    direction_deg: float


@dataclass
class _MockTrack:
    id:       int
    bbox:     _MockBBox
    velocity: _MockVelocity


def _make_track(
    tid: int,
    cx: float = 320.0,
    cy: float = 240.0,
    speed: float = 5.0,
    direction: float = 90.0,
) -> _MockTrack:
    half_w, half_h = 30.0, 60.0
    return _MockTrack(
        id=tid,
        bbox=_MockBBox(cx - half_w, cy - half_h, cx + half_w, cy + half_h),
        velocity=_MockVelocity(speed=speed, direction_deg=direction),
    )


def _make_tracks(n: int = 5, base_speed: float = 5.0) -> List[_MockTrack]:
    return [
        _make_track(
            tid=i,
            cx=float(100 + i * 150),
            cy=float(200 + i * 50),
            speed=base_speed + i * 0.5,
            direction=float(i * 45),
        )
        for i in range(n)
    ]


# ============================================================================
# 1. TrajectoryFeatureExtractor
# ============================================================================

class TestTrajectoryFeatureExtractor:

    def test_returns_none_before_window_fills(self):
        ext = TrajectoryFeatureExtractor(window_size=10, min_tracks=2)
        tracks = _make_tracks(3)
        ext.update(tracks, frame_index=0)
        # Only 1 frame — below window_size // 2
        assert ext.compute_features() is None

    def test_returns_vector_after_sufficient_frames(self):
        ext = TrajectoryFeatureExtractor(window_size=10, min_tracks=2)
        tracks = _make_tracks(5)
        for i in range(6):
            ext.update(tracks, frame_index=i)
        vec = ext.compute_features()
        assert vec is not None
        assert isinstance(vec, FeatureVector)

    def test_feature_names_match(self):
        ext = TrajectoryFeatureExtractor(window_size=10)
        tracks = _make_tracks(5)
        for i in range(6):
            ext.update(tracks, frame_index=i)
        vec = ext.compute_features()
        assert vec is not None
        d = vec.to_dict()
        assert sorted(d.keys()) == sorted(_FEATURE_NAMES)

    def test_velocity_mean_is_positive(self):
        ext = TrajectoryFeatureExtractor(window_size=10)
        tracks = _make_tracks(5, base_speed=10.0)
        for i in range(6):
            ext.update(tracks, frame_index=i)
        vec = ext.compute_features()
        assert vec is not None
        assert vec.velocity_mean > 0.0

    def test_high_speed_variance_detected(self):
        ext = TrajectoryFeatureExtractor(window_size=10)
        for i in range(6):
            # Half tracks slow, half fast → high variance
            tracks = (
                _make_tracks(3, base_speed=2.0) +
                _make_tracks(3, base_speed=30.0)
            )
            # Re-assign unique IDs
            for j, t in enumerate(tracks):
                object.__setattr__(t, "id", j)
            ext.update(tracks, frame_index=i)
        vec = ext.compute_features()
        assert vec is not None
        assert vec.velocity_variance > 1.0

    def test_direction_entropy_max_for_uniform(self):
        """Tracks moving in all 8 directions → entropy near log2(8) = 3.0"""
        ext = TrajectoryFeatureExtractor(window_size=10)
        for i in range(6):
            tracks = [
                _make_track(j, speed=5.0, direction=float(j * 45))
                for j in range(8)
            ]
            ext.update(tracks, frame_index=i)
        vec = ext.compute_features()
        assert vec is not None
        assert vec.direction_entropy > 2.5   # near max

    def test_direction_entropy_low_for_aligned(self):
        """All tracks moving in same direction → entropy near 0"""
        ext = TrajectoryFeatureExtractor(window_size=10)
        for i in range(6):
            tracks = [
                _make_track(j, speed=5.0, direction=90.0)
                for j in range(8)
            ]
            ext.update(tracks, frame_index=i)
        vec = ext.compute_features()
        assert vec is not None
        assert vec.direction_entropy < 0.5

    def test_to_array_shape(self):
        ext = TrajectoryFeatureExtractor(window_size=10)
        tracks = _make_tracks(5)
        for i in range(6):
            ext.update(tracks, frame_index=i)
        vec = ext.compute_features()
        assert vec is not None
        arr = vec.to_array()
        assert arr.shape == (9,)
        assert arr.dtype == np.float32

    def test_reset_clears_state(self):
        ext = TrajectoryFeatureExtractor(window_size=10)
        tracks = _make_tracks(5)
        for i in range(10):
            ext.update(tracks, frame_index=i)
        assert ext.compute_features() is not None
        ext.reset()
        assert ext.compute_features() is None

    def test_collision_rate_increases_when_tracks_close(self):
        ext_spread  = TrajectoryFeatureExtractor(window_size=10)
        ext_crowded = TrajectoryFeatureExtractor(window_size=10)

        for i in range(6):
            spread_tracks = [
                _make_track(j, cx=float(j * 500), cy=300.0)
                for j in range(4)
            ]
            crowded_tracks = [
                _make_track(j, cx=300.0 + j * 5.0, cy=300.0)
                for j in range(4)
            ]
            ext_spread.update(spread_tracks, frame_index=i)
            ext_crowded.update(crowded_tracks, frame_index=i)

        v_spread  = ext_spread.compute_features()
        v_crowded = ext_crowded.compute_features()
        assert v_spread  is not None
        assert v_crowded is not None
        assert v_crowded.track_collision_rate >= v_spread.track_collision_rate

    def test_empty_tracks_updates_without_error(self):
        ext = TrajectoryFeatureExtractor(window_size=10)
        for i in range(6):
            ext.update([], frame_index=i)
        # Should not raise, but may return None due to insufficient tracks
        result = ext.compute_features()
        assert result is None  # empty window → no valid features


# ============================================================================
# 2. MLBehaviorClassifier (mocked model)
# ============================================================================

class TestMLBehaviorClassifier:

    def _make_mock_cache(self, pred_class: str = "normal", proba: float = 0.9):
        """Build a mock _ModelCache that returns a fixed prediction."""
        cache = MagicMock()
        cache.classes = ["crowd_surge", "normal", "panic", "suspicion", "violence"]
        n_classes = len(cache.classes)

        idx = cache.classes.index(pred_class) if pred_class in cache.classes else 1
        proba_arr = np.full(n_classes, (1.0 - proba) / (n_classes - 1))
        proba_arr[idx] = proba

        cache.scaler.transform.side_effect = lambda x: x
        cache.model.predict_proba.return_value = np.array([proba_arr])
        return cache

    def test_predict_returns_correct_behavior(self):
        from core.behavior.ml_behavior_classifier import MLBehaviorClassifier
        clf = MLBehaviorClassifier.__new__(MLBehaviorClassifier)
        clf._model_path   = Path("models/crowd_behavior_model.pkl")
        clf._scaler_path  = Path("models/crowd_behavior_scaler.pkl")
        clf._encoder_path = Path("models/crowd_behavior_label_encoder.pkl")
        clf._cache = self._make_mock_cache("panic", 0.91)

        vec = FeatureVector(
            velocity_mean=20.0, velocity_variance=50.0,
            acceleration_mean=5.0, acceleration_spikes=0.4,
            direction_entropy=2.8, crowd_density=0.05,
            density_change_rate=0.02, trajectory_dispersion=150.0,
            track_collision_rate=0.3,
        )
        result = clf.predict(vec)
        assert result["behavior"] == "panic"
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_returns_normal(self):
        from core.behavior.ml_behavior_classifier import MLBehaviorClassifier
        clf = MLBehaviorClassifier.__new__(MLBehaviorClassifier)
        clf._cache = self._make_mock_cache("normal", 0.95)
        clf._model_path   = Path("x")
        clf._scaler_path  = Path("x")
        clf._encoder_path = Path("x")

        vec = FeatureVector(
            velocity_mean=3.0, velocity_variance=1.0,
            acceleration_mean=0.5, acceleration_spikes=0.0,
            direction_entropy=1.0, crowd_density=0.01,
            density_change_rate=0.0, trajectory_dispersion=300.0,
            track_collision_rate=0.0,
        )
        result = clf.predict(vec)
        assert result["behavior"] == "normal"

    def test_predict_raw_returns_probabilities(self):
        from core.behavior.ml_behavior_classifier import MLBehaviorClassifier
        clf = MLBehaviorClassifier.__new__(MLBehaviorClassifier)
        clf._cache = self._make_mock_cache("crowd_surge", 0.88)
        clf._model_path   = Path("x")
        clf._scaler_path  = Path("x")
        clf._encoder_path = Path("x")

        arr = np.zeros(9, dtype=np.float32)
        label, conf, probs = clf.predict_raw(arr)
        assert label == "crowd_surge"
        assert abs(conf - 0.88) < 0.01
        assert sum(probs.values()) == pytest.approx(1.0, abs=0.01)

    def test_model_cache_invalidate(self):
        from core.behavior.ml_behavior_classifier import _ModelCache
        _ModelCache.invalidate()
        assert _ModelCache._instance is None


# ============================================================================
# 3. Integration: BehaviorAnalyzer with ML classifier
# ============================================================================

class TestBehaviorAnalyzerIntegration:

    def _make_analyzer_with_mock_clf(self):
        """Build a BehaviorAnalyzer wired to a mock ML classifier."""
        from core.behavior.behavior_analyzer import BehaviorAnalyzer
        from core.behavior.base_analyzer import BehaviorLabel

        mock_clf = MagicMock()
        mock_clf.classify.return_value = (
            BehaviorLabel.NORMAL, 0.95, ["ml:normal:0.95"], {}
        )

        analyzer = BehaviorAnalyzer(classifier=mock_clf, frame_shape=(720, 1280))
        return analyzer, mock_clf

    def test_analyze_returns_behavior_result(self):
        from core.behavior.base_analyzer import BehaviorResult, BehaviorLabel

        analyzer, _ = self._make_analyzer_with_mock_clf()
        tracks = _make_tracks(5)
        result = analyzer.analyze(tracks, frame_index=0)

        assert isinstance(result, BehaviorResult)
        assert result.label == BehaviorLabel.NORMAL
        assert 0.0 <= result.confidence <= 1.0

    def test_analyze_to_dict_format(self):
        analyzer, _ = self._make_analyzer_with_mock_clf()
        tracks = _make_tracks(5)
        result = analyzer.analyze(tracks, frame_index=0)
        d = result.to_dict()

        assert "behavior"   in d
        assert "confidence" in d
        assert isinstance(d["behavior"],   str)
        assert isinstance(d["confidence"], float)

    def test_analyze_calls_classifier(self):
        analyzer, mock_clf = self._make_analyzer_with_mock_clf()
        tracks = _make_tracks(5)
        analyzer.analyze(tracks, frame_index=0)
        mock_clf.classify.assert_called_once()

    def test_analyze_track_count_matches(self):
        analyzer, _ = self._make_analyzer_with_mock_clf()
        tracks = _make_tracks(7)
        result = analyzer.analyze(tracks, frame_index=0)
        assert result.features.track_count == 7

    def test_swap_classifier(self):
        analyzer, old_clf = self._make_analyzer_with_mock_clf()
        new_clf = MagicMock()
        from core.behavior.base_analyzer import BehaviorLabel
        new_clf.classify.return_value = (
            BehaviorLabel.CROWD_PANIC, 0.87, ["ml:panic:0.87"], {}
        )
        analyzer.swap_classifier(new_clf)
        tracks = _make_tracks(5)
        result = analyzer.analyze(tracks, frame_index=1)
        assert result.label == BehaviorLabel.CROWD_PANIC

    def test_reset_clears_history(self):
        analyzer, _ = self._make_analyzer_with_mock_clf()
        tracks = _make_tracks(5)
        for i in range(5):
            analyzer.analyze(tracks, frame_index=i)
        assert len(analyzer.history) == 5
        analyzer.reset()
        assert len(analyzer.history) == 0

    def test_empty_tracks_returns_insufficient_data(self):
        from core.behavior.base_analyzer import BehaviorLabel

        analyzer, mock_clf = self._make_analyzer_with_mock_clf()
        mock_clf.classify.return_value = (
            BehaviorLabel.INSUFFICIENT_DATA, 0.0, ["insufficient_data"], {}
        )
        result = analyzer.analyze([], frame_index=0)
        assert result.features.track_count == 0

    def test_websocket_output_format(self):
        """Verify the output dict is WebSocket-compatible."""
        analyzer, _ = self._make_analyzer_with_mock_clf()
        tracks = _make_tracks(5)
        result = analyzer.analyze(tracks, frame_index=0)
        ws_payload = result.to_dict()

        # Must be JSON-serialisable with exactly these two keys
        import json
        serialised = json.dumps(ws_payload)
        parsed = json.loads(serialised)
        assert set(parsed.keys()) == {"behavior", "confidence"}


# ============================================================================
# 4. Direction entropy edge cases
# ============================================================================

class TestDirectionEntropy:

    def test_empty_list(self):
        assert TrajectoryFeatureExtractor._direction_entropy([]) == 0.0

    def test_single_direction(self):
        e = TrajectoryFeatureExtractor._direction_entropy([90.0] * 10)
        assert e == pytest.approx(0.0, abs=0.01)

    def test_max_entropy(self):
        directions = [float(i * 45) for i in range(8)]
        e = TrajectoryFeatureExtractor._direction_entropy(directions)
        assert e == pytest.approx(math.log2(8), abs=0.01)