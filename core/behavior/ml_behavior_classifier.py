"""
core/behavior/ml_behavior_classifier.py
=========================================
ML-based crowd behavior inference module.

Loads a trained RandomForest/XGBoost model and StandardScaler from disk
(once, lazily), accepts a FeatureVector, and returns a BehaviorResult-
compatible dict:

    {"behavior": "panic", "confidence": 0.91}

Singleton model cache ensures the 100 MB model file is loaded at most once
per process regardless of how many times the module is imported.

Usage
-----
    from core.behavior.ml_behavior_classifier import MLBehaviorClassifier

    clf = MLBehaviorClassifier()                     # lazy load
    result = clf.predict(feature_vector)
    print(result)   # {"behavior": "panic", "confidence": 0.91}
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.behavior.base_analyzer import (
    BaseBehaviorClassifier,
    BehaviorLabel,
    BehaviorThresholds,
    FrameFeatures,
    TrackFeatures,
)
from core.behavior.trajectory_features import FeatureVector

logger = logging.getLogger("crowd_analysis.behavior.ml_classifier")

# ── Default model paths ────────────────────────────────────────────────────
_DEFAULT_MODEL_PATH   = Path("models/crowd_behavior_model.pkl")
_DEFAULT_SCALER_PATH  = Path("models/crowd_behavior_scaler.pkl")
_DEFAULT_ENCODER_PATH = Path("models/crowd_behavior_label_encoder.pkl")

# ── Label string → BehaviorLabel mapping ──────────────────────────────────
_LABEL_MAP: Dict[str, BehaviorLabel] = {
    "normal":       BehaviorLabel.NORMAL,
    "panic":        BehaviorLabel.CROWD_PANIC,
    "crowd_surge":  BehaviorLabel.CROWD_SURGE,
    "violence":     BehaviorLabel.VIOLENCE,
    "suspicion":    BehaviorLabel.SUSPICION,
    "running":      BehaviorLabel.RUNNING,
}


# ── Singleton cache ────────────────────────────────────────────────────────

class _ModelCache:
    """
    Thread-safe singleton that holds the loaded model, scaler, and encoder.

    Uses double-checked locking so multiple coroutines starting simultaneously
    do not each trigger a separate (expensive) load.
    """
    _instance: Optional["_ModelCache"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        model_path:   Path,
        scaler_path:  Path,
        encoder_path: Path,
    ) -> None:
        import joblib  # deferred — only needed once

        logger.info("Loading ML model from '%s' …", model_path)
        self.model   = joblib.load(model_path)
        self.scaler  = joblib.load(scaler_path)

        if encoder_path.exists():
            self.encoder = joblib.load(encoder_path)
            self.classes: List[str] = list(self.encoder.classes_)
        else:
            # Fallback: derive classes from model if encoder missing
            self.encoder = None
            self.classes = list(getattr(self.model, "classes_", []))

        logger.info(
            "ML model loaded — type=%s  classes=%s",
            type(self.model).__name__,
            self.classes,
        )

    @classmethod
    def get(
        cls,
        model_path:   Path = _DEFAULT_MODEL_PATH,
        scaler_path:  Path = _DEFAULT_SCALER_PATH,
        encoder_path: Path = _DEFAULT_ENCODER_PATH,
    ) -> "_ModelCache":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:   # double-checked
                    cls._instance = cls(model_path, scaler_path, encoder_path)
        return cls._instance

    @classmethod
    def invalidate(cls) -> None:
        """Force reload on the next call to get(). Useful for hot-swapping."""
        with cls._lock:
            cls._instance = None
        logger.info("MLModelCache invalidated — will reload on next inference.")


# ── Public classifier ──────────────────────────────────────────────────────

class MLBehaviorClassifier(BaseBehaviorClassifier):
    """
    Drop-in replacement for ``RuleBasedClassifier`` that uses a trained
    ML model for behavior prediction.

    Implements :class:`~core.behavior.base_analyzer.BaseBehaviorClassifier`
    so it can be hot-swapped into ``BehaviorAnalyzer`` via
    ``analyzer.swap_classifier(MLBehaviorClassifier())``.

    Parameters
    ----------
    model_path   : path to the trained model .pkl file.
    scaler_path  : path to the fitted scaler .pkl file.
    encoder_path : path to the LabelEncoder .pkl file.
    feature_extractor : optional pre-built extractor; created internally
                        if None.
    """

    def __init__(
        self,
        model_path:   Path = _DEFAULT_MODEL_PATH,
        scaler_path:  Path = _DEFAULT_SCALER_PATH,
        encoder_path: Path = _DEFAULT_ENCODER_PATH,
    ) -> None:
        self._model_path   = model_path
        self._scaler_path  = scaler_path
        self._encoder_path = encoder_path
        self._cache: Optional[_ModelCache] = None   # lazy

    # ------------------------------------------------------------------
    # BaseBehaviorClassifier interface
    # ------------------------------------------------------------------

    def classify(
        self,
        features: FrameFeatures,
    ) -> Tuple[BehaviorLabel, float, List[str], Dict[int, str]]:
        """
        Classify crowd behavior from a pre-computed FrameFeatures snapshot.

        This method is called by BehaviorAnalyzer every frame. It converts
        the FrameFeatures → FeatureVector → model prediction.

        Returns
        -------
        (label, confidence, signals, track_labels)
        """
        # Build a FeatureVector from FrameFeatures
        vec = self._frame_features_to_vector(features)

        if vec is None:
            return (
                BehaviorLabel.INSUFFICIENT_DATA,
                0.0,
                ["insufficient_data"],
                {},
            )

        label, confidence = self._predict_from_vector(vec)
        signals = [f"ml:{label.value}:{confidence:.2f}"]

        # Tag anomalous tracks with the crowd-level label
        track_labels: Dict[int, str] = {}
        if label not in (BehaviorLabel.NORMAL, BehaviorLabel.INSUFFICIENT_DATA):
            for tf in features.track_features:
                if tf.is_anomalous:
                    track_labels[tf.track_id] = label.value

        return label, confidence, signals, track_labels

    # ------------------------------------------------------------------
    # Direct feature-vector API (used by inference scripts / tests)
    # ------------------------------------------------------------------

    def predict(self, feature_vector: FeatureVector) -> Dict[str, object]:
        """
        Run inference on a FeatureVector.

        Returns
        -------
        {"behavior": "panic", "confidence": 0.91}
        """
        label, confidence = self._predict_from_vector(feature_vector)
        return {"behavior": label.value, "confidence": round(confidence, 4)}

    def predict_raw(
        self, feature_array: np.ndarray
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Run inference on a raw numpy array (shape (9,) or (1, 9)).

        Returns
        -------
        (label_str, confidence, class_probabilities)
        """
        cache = self._load_cache()
        x = feature_array.reshape(1, -1).astype(np.float32)
        x_scaled = cache.scaler.transform(x)

        proba = cache.model.predict_proba(x_scaled)[0]
        idx   = int(np.argmax(proba))

        raw_label = cache.classes[idx] if idx < len(cache.classes) else "normal"
        conf      = float(proba[idx])
        class_probs = {cls: round(float(p), 4)
                       for cls, p in zip(cache.classes, proba)}

        return raw_label, conf, class_probs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_cache(self) -> _ModelCache:
        if self._cache is None:
            self._cache = _ModelCache.get(
                self._model_path,
                self._scaler_path,
                self._encoder_path,
            )
        return self._cache

    def _predict_from_vector(
        self, vec: FeatureVector
    ) -> Tuple[BehaviorLabel, float]:
        try:
            raw_label, confidence, _ = self.predict_raw(vec.to_array())
        except Exception as exc:
            logger.error("ML inference error: %s", exc, exc_info=True)
            return BehaviorLabel.NORMAL, 0.0

        label = _LABEL_MAP.get(raw_label.lower(), BehaviorLabel.NORMAL)
        logger.debug(
            "ML prediction: raw=%r  label=%s  conf=%.3f",
            raw_label, label.value, confidence,
        )
        return label, confidence

    @staticmethod
    def _frame_features_to_vector(
        features: FrameFeatures,
    ) -> Optional[FeatureVector]:
        """
        Build a FeatureVector from a FrameFeatures snapshot.

        FrameFeatures already contains the pre-computed crowd statistics
        from VelocityAnalyzer and CrowdDensityAnalyzer, so we re-use them
        directly rather than recomputing from raw tracks.
        """
        if features.track_count == 0:
            return None

        tfs = features.track_features
        if not tfs:
            return None

        # Velocity
        speeds = [tf.speed for tf in tfs]
        vel_mean = float(np.mean(speeds))
        vel_var  = float(np.var(speeds))

        # Acceleration
        accels    = [abs(tf.acceleration) for tf in tfs]
        acc_mean  = float(np.mean(accels)) if accels else 0.0
        acc_spikes = float(
            sum(1 for a in accels if a >= 8.0) / max(len(accels), 1)
        )

        # Direction entropy (reuse static method from trajectory_features)
        from core.behavior.trajectory_features import TrajectoryFeatureExtractor
        directions = [tf.direction_deg for tf in tfs]
        dir_entropy = TrajectoryFeatureExtractor._direction_entropy(directions)

        # Density
        crowd_density = features.crowd_density

        # Dispersion: mean pairwise distance of centroids
        centroids = [tf.centroid for tf in tfs]
        traj_disp = TrajectoryFeatureExtractor._trajectory_dispersion(centroids)

        # Collision rate from proximity pairs
        n = features.track_count
        max_pairs = n * (n - 1) / 2 if n > 1 else 1
        collision_rate = len(features.proximity_pairs) / max_pairs

        return FeatureVector(
            velocity_mean         = vel_mean,
            velocity_variance     = vel_var,
            acceleration_mean     = acc_mean,
            acceleration_spikes   = acc_spikes,
            direction_entropy     = dir_entropy,
            crowd_density         = crowd_density,
            density_change_rate   = features.crowd_acceleration,
            trajectory_dispersion = traj_disp,
            track_collision_rate  = collision_rate,
            frame_window_start    = features.frame_index,
            frame_window_end      = features.frame_index,
            track_count           = features.track_count,
        )