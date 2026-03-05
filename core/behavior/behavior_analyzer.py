"""
core/behavior/behavior_analyzer.py  (ML-integrated version)
=============================================================
Stateful facade that wires all behavior analysis sub-modules together,
now with an ML classifier replacing the rule-based backend.

Pipeline per frame
------------------
    VelocityAnalyzer.compute()       → List[TrackFeatures]
    AnomalyDetector.detect()         → patches is_anomalous in-place
    CrowdDensityAnalyzer.compute()   → DensityResult (zones, heatmap, pairs)
    _build_frame_features()          → FrameFeatures (consolidated snapshot)
    TrajectoryFeatureExtractor       → FeatureVector  (ML features)
    MLBehaviorClassifier.classify()  → (label, confidence, signals, track_labels)
    _apply_cooldown()                → suppresses repeated alerts
    BehaviorResult                   → returned to caller

Backward compatibility
----------------------
The public ``analyze()`` method signature and ``BehaviorResult.to_dict()``
output format are unchanged.  The WebSocket streamer sees no difference.

Usage
-----
    from core.behavior.behavior_analyzer import BehaviorAnalyzer

    analyzer = BehaviorAnalyzer.from_settings()
    result   = analyzer.analyze(tracks, frame_index=n)
    print(result.to_dict())   # {"behavior": "panic", "confidence": 0.87}
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

from core.tracking.deepsort_tracker import TrackedPerson
from core.behavior.base_analyzer import (
    BaseBehaviorClassifier,
    BehaviorLabel,
    BehaviorResult,
    BehaviorThresholds,
    FrameFeatures,
    TrackFeatures,
)
from core.behavior.velocity_analyzer import VelocityAnalyzer
from core.behavior.anomaly_detector  import AnomalyDetector
from core.behavior.crowd_density     import CrowdDensityAnalyzer
from core.behavior.trajectory_store  import TrajectoryStore
from core.behavior.trajectory_features import TrajectoryFeatureExtractor

logger = logging.getLogger("crowd_analysis.behavior")

# ── Model paths ────────────────────────────────────────────────────────────
_DEFAULT_MODEL_PATH   = Path("models/crowd_behavior_model.pkl")
_DEFAULT_SCALER_PATH  = Path("models/crowd_behavior_scaler.pkl")
_DEFAULT_ENCODER_PATH = Path("models/crowd_behavior_label_encoder.pkl")


def _build_classifier(
    thresholds:   BehaviorThresholds,
    model_path:   Path = _DEFAULT_MODEL_PATH,
    scaler_path:  Path = _DEFAULT_SCALER_PATH,
    encoder_path: Path = _DEFAULT_ENCODER_PATH,
) -> BaseBehaviorClassifier:
    """
    Return an ML classifier when the model artefacts exist, otherwise
    fall back gracefully to the rule-based classifier.
    """
    if model_path.exists() and scaler_path.exists():
        try:
            from core.behavior.ml_behavior_classifier import MLBehaviorClassifier
            clf = MLBehaviorClassifier(model_path, scaler_path, encoder_path)
            logger.info(
                "ML classifier loaded from '%s'.", model_path
            )
            return clf
        except Exception as exc:
            logger.warning(
                "Could not load ML classifier (%s) — falling back to rule-based.", exc
            )

    from core.behavior.event_classifier import RuleBasedClassifier
    logger.info(
        "Using rule-based classifier (ML model not found at '%s').", model_path
    )
    return RuleBasedClassifier(thresholds)


class BehaviorAnalyzer:
    """
    Stateful orchestrator for the crowd behavior analysis pipeline.

    Parameters
    ----------
    thresholds    : BehaviorThresholds
    classifier    : BaseBehaviorClassifier — ML (default) or rule-based backend.
    frame_shape   : (height, width) in pixels.
    model_path    : path to trained ML model (used when classifier is None).
    scaler_path   : path to fitted scaler.
    encoder_path  : path to label encoder.
    window_size   : sliding window for the feature extractor (frames).
    """

    def __init__(
        self,
        thresholds:   Optional[BehaviorThresholds]    = None,
        classifier:   Optional[BaseBehaviorClassifier] = None,
        frame_shape:  Tuple[int, int]                  = (720, 1280),
        model_path:   Path                             = _DEFAULT_MODEL_PATH,
        scaler_path:  Path                             = _DEFAULT_SCALER_PATH,
        encoder_path: Path                             = _DEFAULT_ENCODER_PATH,
        window_size:  int                              = 30,
    ) -> None:
        self._thresholds  = thresholds or BehaviorThresholds()
        self._frame_shape = frame_shape

        # Classifier: use provided or auto-detect ML vs rule-based
        self._classifier = classifier or _build_classifier(
            self._thresholds, model_path, scaler_path, encoder_path
        )

        self._velocity  = VelocityAnalyzer(self._thresholds)
        self._anomaly   = AnomalyDetector(self._thresholds)
        self._density   = CrowdDensityAnalyzer(self._thresholds, frame_shape)
        self._store     = TrajectoryStore(self._thresholds.history_length)
        self._features  = TrajectoryFeatureExtractor(
            frame_shape=frame_shape,
            window_size=window_size,
        )

        # Frame-to-frame state
        self._prev_speeds:     Dict[int, float] = {}
        self._prev_directions: Dict[int, float] = {}
        self._prev_mean_speed: float            = 0.0
        self._erratic_frames:  Dict[int, int]   = defaultdict(int)

        # Cool-down: label.value → last-fired monotonic timestamp
        self._last_alert_time: Dict[str, float] = {}

        self._history: Deque[BehaviorResult] = deque(
            maxlen=self._thresholds.history_length
        )

        logger.info(
            "BehaviorAnalyzer ready — classifier=%s  shape=%s",
            type(self._classifier).__name__, frame_shape,
        )

    # ------------------------------------------------------------------
    # Alternate constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(
        cls,
        classifier:   Optional[BaseBehaviorClassifier] = None,
        model_path:   Path                             = _DEFAULT_MODEL_PATH,
        scaler_path:  Path                             = _DEFAULT_SCALER_PATH,
        encoder_path: Path                             = _DEFAULT_ENCODER_PATH,
    ) -> "BehaviorAnalyzer":
        """Construct from ``config/settings.py`` with ML classifier."""
        try:
            from config.settings import get_settings
            cfg = get_settings()
            b, t, v = cfg.behavior, cfg.tracking, cfg.video

            thresholds = BehaviorThresholds(
                run_threshold_px_per_frame   = t.speed_run_threshold_px_per_frame,
                zscore_threshold             = t.anomaly_zscore_threshold,
                density_zone_rows            = b.density_zone_rows,
                density_zone_cols            = b.density_zone_cols,
                density_alert_threshold      = b.density_alert_threshold,
                panic_min_anomalous_fraction = b.panic_min_anomalous_fraction,
                violence_proximity_px        = b.fight_proximity_px,
                surge_velocity_multiplier    = b.surge_velocity_multiplier,
                alert_cooldown_s             = b.alert_cooldown_s,
                history_length               = t.trajectory_history_length,
            )
            frame_shape = (v.frame_height, v.frame_width)
        except Exception as exc:
            logger.error("settings load failed: %s — using defaults.", exc)
            thresholds, frame_shape = BehaviorThresholds(), (720, 1280)

        _clf = classifier or _build_classifier(
            thresholds, model_path, scaler_path, encoder_path
        )
        return cls(
            thresholds=thresholds,
            classifier=_clf,
            frame_shape=frame_shape,
            model_path=model_path,
            scaler_path=scaler_path,
            encoder_path=encoder_path,
        )

    # ------------------------------------------------------------------
    # Primary public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        tracks:      List[TrackedPerson],
        frame_index: int = 0,
    ) -> BehaviorResult:
        """
        Run the full pipeline for one frame.

        Returns
        -------
        BehaviorResult
            ``result.to_dict()`` → ``{"behavior": "panic", "confidence": 0.87}``
        """
        t_start = time.perf_counter()

        # 1. Velocity features
        track_features: List[TrackFeatures] = self._velocity.compute(
            tracks=tracks,
            prev_speeds=self._prev_speeds,
            prev_directions=self._prev_directions,
        )

        # 2. Crowd statistics
        mean_speed, std_speed = VelocityAnalyzer.crowd_speed_stats(track_features)
        mean_dir, dir_disp    = VelocityAnalyzer.circular_direction_stats(track_features)

        # 3. Anomaly detection (mutates is_anomalous in-place)
        anomalous_count = self._anomaly.detect(
            track_features=track_features,
            mean_speed=mean_speed,
            std_speed=std_speed,
            mean_direction_deg=mean_dir,
            direction_dispersion=dir_disp,
        )

        # 4. Density + proximity pairs
        density_result = self._density.compute(track_features)

        # 5. Trajectory store
        self._store.update(
            [(t.id, t.bbox.centroid) for t in tracks],
            frame_index=frame_index,
        )
        self._store.prune(active_ids={t.id for t in tracks})

        # 6. Update feature extractor (ML sliding window)
        self._features.update(tracks, frame_index=frame_index)

        # 7. Consolidated FrameFeatures
        n      = len(track_features)
        safe_n = max(n, 1)
        running_count = sum(1 for tf in track_features if tf.is_running)

        features = FrameFeatures(
            frame_index          = frame_index,
            track_count          = n,
            track_features       = track_features,
            mean_speed           = mean_speed,
            std_speed            = std_speed,
            mean_direction_deg   = mean_dir,
            direction_dispersion = dir_disp,
            crowd_density        = density_result.overall_density,
            density_zones        = density_result.zone_counts,
            running_count        = running_count,
            anomalous_count      = anomalous_count,
            running_fraction     = running_count / safe_n,
            anomalous_fraction   = anomalous_count / safe_n,
            proximity_pairs      = density_result.proximity_pairs,
            crowd_acceleration   = mean_speed - self._prev_mean_speed,
        )

        # 8. Classification (ML or rule-based depending on what loaded)
        try:
            label, confidence, signals, track_labels = self._classifier.classify(
                features
            )
        except Exception as exc:
            logger.error("Classifier error frame=%d: %s", frame_index, exc,
                         exc_info=True)
            label, confidence, signals, track_labels = (
                BehaviorLabel.NORMAL, 0.0, ["classifier_error"], {}
            )

        # 9. Cool-down suppression
        label, confidence, signals = self._apply_cooldown(label, confidence, signals)

        # 10. Persist frame-to-frame state
        self._update_state(tracks, features)

        # 11. Build result
        elapsed_ms = (time.perf_counter() - t_start) * 1_000
        result = BehaviorResult(
            label=label, confidence=confidence, frame_index=frame_index,
            track_labels=track_labels, features=features,
            signals=signals, elapsed_ms=elapsed_ms,
        )
        self._history.append(result)

        if label not in (BehaviorLabel.NORMAL, BehaviorLabel.INSUFFICIENT_DATA):
            logger.info(
                "⚠  frame=%d  behavior=%s  conf=%.2f  signals=%s",
                frame_index, label.value, confidence, signals,
            )
        else:
            logger.debug(
                "frame=%d  %s  conf=%.2f  tracks=%d  %.1fms",
                frame_index, label.value, confidence, n, elapsed_ms,
            )

        return result

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all cross-frame state. Call between scenes."""
        self._prev_speeds.clear()
        self._prev_directions.clear()
        self._prev_mean_speed = 0.0
        self._erratic_frames.clear()
        self._last_alert_time.clear()
        self._history.clear()
        self._store.clear()
        self._features.reset()
        logger.info("BehaviorAnalyzer reset.")

    def set_frame_shape(self, height: int, width: int) -> None:
        self._frame_shape = (height, width)
        self._density.set_frame_shape(height, width)
        self._features.set_frame_shape(height, width)

    def swap_classifier(self, classifier: BaseBehaviorClassifier) -> None:
        """Hot-swap rule-based ↔ ML classifier at runtime."""
        old = type(self._classifier).__name__
        self._classifier = classifier
        logger.info("Classifier swapped: %s → %s", old,
                    type(classifier).__name__)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def history(self) -> List[BehaviorResult]:
        return list(self._history)

    @property
    def last_result(self) -> Optional[BehaviorResult]:
        return self._history[-1] if self._history else None

    @property
    def trajectory_store(self) -> TrajectoryStore:
        return self._store

    @property
    def feature_extractor(self) -> TrajectoryFeatureExtractor:
        return self._features

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _apply_cooldown(
        self,
        label: BehaviorLabel,
        confidence: float,
        signals: List[str],
    ) -> Tuple[BehaviorLabel, float, List[str]]:
        if label in (BehaviorLabel.NORMAL, BehaviorLabel.INSUFFICIENT_DATA):
            return label, confidence, signals

        now        = time.monotonic()
        last_fired = self._last_alert_time.get(label.value, 0.0)

        if now - last_fired < self._thresholds.alert_cooldown_s:
            logger.debug("Cool-down active for '%s'.", label.value)
            return BehaviorLabel.NORMAL, 0.0, ["cooldown_suppressed"]

        self._last_alert_time[label.value] = now
        return label, confidence, signals

    def _update_state(
        self, tracks: List[TrackedPerson], features: FrameFeatures
    ) -> None:
        self._prev_mean_speed = features.mean_speed
        self._prev_speeds     = {t.id: t.velocity.speed         for t in tracks}
        self._prev_directions = {t.id: t.velocity.direction_deg for t in tracks}
        active_ids = {t.id for t in tracks}
        self._erratic_frames  = {
            tid: cnt for tid, cnt in self._erratic_frames.items()
            if tid in active_ids
        }