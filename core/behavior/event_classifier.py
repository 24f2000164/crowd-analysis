"""
core/behavior/event_classifier.py
====================================
Rule-based crowd event classifier with an ML-ready plug-in interface.

Responsibility
--------------
Consume a fully populated ``FrameFeatures`` object (built by the pipeline
that runs ``VelocityAnalyzer`` → ``AnomalyDetector`` → ``CrowdDensityAnalyzer``)
and return a ``(BehaviorLabel, confidence, signals, track_labels)`` tuple.

Classifier hierarchy
--------------------

    BaseBehaviorClassifier   (base_analyzer.py — ABC)
        ├── RuleBasedClassifier   ← production default
        └── MLBehaviorClassifier  ← drop-in ML replacement

Rules (evaluated in priority order, highest severity first)
------------------------------------------------------------
    1. VIOLENCE    — proximity pair + high relative speed
    2. CROWD_PANIC — mass anomalous movement + direction chaos
    3. CROWD_SURGE — coordinated high-speed collective movement
    4. RUNNING     — large fraction sprinting
    5. SUSPICION   — isolated individual with erratic motion
    6. NORMAL      — default

Confidence model
----------------
Confidence is a **weighted evidence sum** rather than a hard 0/1 flag,
so the output is a continuous value in [0, 1] that mirrors what a trained
classifier would produce.  Each rule has a dedicated weight dictionary so
the numeric contribution of each signal is explicit and tunable via config.

Usage
-----
    from core.behavior.event_classifier import RuleBasedClassifier
    from core.behavior.base_analyzer    import BehaviorThresholds

    clf    = RuleBasedClassifier(BehaviorThresholds())
    label, confidence, signals, track_labels = clf.classify(frame_features)
"""

from __future__ import annotations

import logging
import math
import statistics
from typing import Dict, List, Optional, Tuple

from core.behavior.base_analyzer import (
    BaseBehaviorClassifier,
    BehaviorLabel,
    BehaviorThresholds,
    FrameFeatures,
    TrackFeatures,
)

logger = logging.getLogger("crowd_analysis.behavior.classifier")

# Shorthand for the classify return type
_ClassifyReturn = Tuple[Optional[BehaviorLabel], float, List[str], Dict[int, str]]


# ============================================================================
# Rule-Based Classifier
# ============================================================================

class RuleBasedClassifier(BaseBehaviorClassifier):
    """
    Priority-ordered rule-based crowd behavior classifier.

    Each rule method returns ``(label | None, confidence, signals, track_labels)``.
    Returning ``None`` means the rule did not fire; the next rule in priority
    order is tried.  The first rule that fires wins.

    Parameters
    ----------
    thresholds : BehaviorThresholds
    """

    # Signal weight tables — keyed by rule name + signal name
    _W_VIOLENCE: Dict[str, float] = {
        "proximity":         0.40,
        "relative_speed":    0.35,
        "high_acceleration": 0.25,
    }
    _W_PANIC: Dict[str, float] = {
        "anomalous_fraction": 0.35,
        "running_fraction":   0.30,
        "direction_chaos":    0.20,
        "high_density":       0.15,
    }
    _W_SURGE: Dict[str, float] = {
        "speed_multiplier":  0.50,
        "coordinated_dir":   0.30,
        "acceleration":      0.20,
    }
    _W_RUNNING: Dict[str, float] = {
        "fraction":          0.60,
        "speed_magnitude":   0.40,
    }
    _W_SUSPICION: Dict[str, float] = {
        "direction_change":  0.50,
        "isolated":          0.30,
        "speed_contrast":    0.20,
    }

    def __init__(self, thresholds: BehaviorThresholds) -> None:
        self._t = thresholds

    # ------------------------------------------------------------------
    # BaseBehaviorClassifier interface
    # ------------------------------------------------------------------

    def classify(
        self,
        features: FrameFeatures,
    ) -> Tuple[BehaviorLabel, float, List[str], Dict[int, str]]:
        """
        Classify crowd behavior.  Returns the first firing rule's result.

        Parameters
        ----------
        features : FrameFeatures — pre-computed frame feature snapshot.

        Returns
        -------
        (BehaviorLabel, confidence, signals, track_labels)
        """
        if features.track_count < self._t.min_population_for_stats:
            return (
                BehaviorLabel.INSUFFICIENT_DATA,
                0.0,
                [f"track_count={features.track_count} < min={self._t.min_population_for_stats}"],
                {},
            )

        for rule in (
            self._rule_violence,
            self._rule_panic,
            self._rule_surge,
            self._rule_running,
            self._rule_suspicion,
        ):
            label, conf, signals, track_labels = rule(features)
            if label is not None:
                logger.debug(
                    "Rule '%s' fired (conf=%.2f, signals=%s).",
                    label.value, conf, signals,
                )
                return label, min(conf, 1.0), signals, track_labels

        return BehaviorLabel.NORMAL, 1.0, ["all_checks_passed"], {}

    # ------------------------------------------------------------------
    # Rule 1 — Violence
    # ------------------------------------------------------------------

    def _rule_violence(self, f: FrameFeatures) -> _ClassifyReturn:
        """
        VIOLENCE: two or more persons in close proximity with high relative
        speed or sudden acceleration — indicative of a physical altercation.
        """
        if not f.proximity_pairs:
            return None, 0.0, [], {}

        t = self._t
        signals: List[str] = []
        track_labels: Dict[int, str] = {}
        evidence = 0.0
        speed_map = {tf.track_id: tf for tf in f.track_features}

        violent_pair_count = 0

        for id_a, id_b, dist in f.proximity_pairs:
            tf_a = speed_map.get(id_a)
            tf_b = speed_map.get(id_b)
            if tf_a is None or tf_b is None:
                continue

            rel_speed  = abs(tf_a.speed - tf_b.speed)
            high_accel = (
                abs(tf_a.acceleration) > 5.0 or
                abs(tf_b.acceleration) > 5.0
            )

            if rel_speed >= t.violence_min_relative_speed:
                violent_pair_count += 1
                signals.append(
                    f"violence_proximity(ids={id_a},{id_b},"
                    f"dist={dist:.0f}px,rel_spd={rel_speed:.1f})"
                )
                evidence += self._W_VIOLENCE["proximity"]
                evidence += self._W_VIOLENCE["relative_speed"] * min(
                    rel_speed / (t.violence_min_relative_speed * 2.0), 1.0
                )
                track_labels[id_a] = BehaviorLabel.VIOLENCE.value
                track_labels[id_b] = BehaviorLabel.VIOLENCE.value

            if high_accel:
                signals.append(
                    f"violence_acceleration(ids={id_a},{id_b})"
                )
                evidence += self._W_VIOLENCE["high_acceleration"]

        if violent_pair_count >= t.violence_min_pair_count:
            confidence = min(evidence / max(violent_pair_count, 1), 1.0)
            return BehaviorLabel.VIOLENCE, confidence, signals, track_labels

        return None, 0.0, [], {}

    # ------------------------------------------------------------------
    # Rule 2 — Crowd Panic
    # ------------------------------------------------------------------

    def _rule_panic(self, f: FrameFeatures) -> _ClassifyReturn:
        """
        CROWD_PANIC: large anomalous fraction + running fraction + directional
        chaos, amplified by high zone density.
        """
        t = self._t
        signals: List[str] = []
        evidence = 0.0

        track_labels = {
            tf.track_id: BehaviorLabel.CROWD_PANIC.value
            for tf in f.track_features if tf.is_anomalous
        }

        if f.anomalous_fraction >= t.panic_min_anomalous_fraction:
            signals.append(
                f"panic_anomalous({f.anomalous_fraction:.2f}>={t.panic_min_anomalous_fraction})"
            )
            evidence += self._W_PANIC["anomalous_fraction"] * f.anomalous_fraction

        if f.running_fraction >= t.panic_min_running_fraction:
            signals.append(
                f"panic_running({f.running_fraction:.2f}>={t.panic_min_running_fraction})"
            )
            evidence += self._W_PANIC["running_fraction"] * f.running_fraction

        if f.direction_dispersion >= t.panic_direction_dispersion:
            signals.append(
                f"panic_chaos(dispersion={f.direction_dispersion:.1f}°>={t.panic_direction_dispersion}°)"
            )
            evidence += self._W_PANIC["direction_chaos"] * min(
                f.direction_dispersion / 180.0, 1.0
            )

        max_zone = max(f.density_zones.values(), default=0)
        if max_zone >= t.density_alert_threshold:
            signals.append(f"panic_density(max_zone={max_zone})")
            evidence += self._W_PANIC["high_density"]

        primary_signals = [
            s for s in signals
            if s.startswith("panic_anomalous") or s.startswith("panic_running")
        ]
        if len(primary_signals) >= 1 and evidence >= 0.35:
            return BehaviorLabel.CROWD_PANIC, evidence, signals, track_labels

        return None, 0.0, [], {}

    # ------------------------------------------------------------------
    # Rule 3 — Crowd Surge
    # ------------------------------------------------------------------

    def _rule_surge(self, f: FrameFeatures) -> _ClassifyReturn:
        """
        CROWD_SURGE: entire crowd accelerating in a coordinated direction.
        High mean speed + low direction dispersion + positive crowd acceleration.
        """
        t = self._t
        signals: List[str] = []
        evidence = 0.0

        baseline = max(f.mean_speed / t.surge_velocity_multiplier, 1.0)
        if f.mean_speed >= baseline * t.surge_velocity_multiplier:
            signals.append(
                f"surge_speed({f.mean_speed:.1f}>={baseline * t.surge_velocity_multiplier:.1f})"
            )
            evidence += self._W_SURGE["speed_multiplier"]

        if f.direction_dispersion <= t.surge_direction_dispersion_max:
            signals.append(
                f"surge_coordinated(dispersion={f.direction_dispersion:.1f}°)"
            )
            evidence += self._W_SURGE["coordinated_dir"]

        if f.crowd_acceleration > 2.0:
            signals.append(
                f"surge_acceleration(Δ={f.crowd_acceleration:.1f})"
            )
            evidence += self._W_SURGE["acceleration"]

        if len(signals) >= 2 and evidence >= 0.50:
            track_labels = {
                tf.track_id: BehaviorLabel.CROWD_SURGE.value
                for tf in f.track_features
            }
            return BehaviorLabel.CROWD_SURGE, evidence, signals, track_labels

        return None, 0.0, [], {}

    # ------------------------------------------------------------------
    # Rule 4 — Running
    # ------------------------------------------------------------------

    def _rule_running(self, f: FrameFeatures) -> _ClassifyReturn:
        """
        RUNNING: a notable fraction of persons is sprinting without
        the directional chaos that would indicate panic.
        """
        t = self._t
        signals: List[str] = []
        evidence = 0.0

        if f.running_fraction >= t.running_fraction_threshold:
            signals.append(
                f"running_fraction({f.running_fraction:.2f}>={t.running_fraction_threshold})"
            )
            evidence += self._W_RUNNING["fraction"] * f.running_fraction

        if f.mean_speed >= t.run_threshold_px_per_frame:
            signals.append(
                f"running_mean_speed({f.mean_speed:.1f}>={t.run_threshold_px_per_frame})"
            )
            evidence += self._W_RUNNING["speed_magnitude"] * min(
                f.mean_speed / (t.run_threshold_px_per_frame * 2.0), 1.0
            )

        track_labels = {
            tf.track_id: BehaviorLabel.RUNNING.value
            for tf in f.track_features if tf.is_running
        }

        if signals:
            return BehaviorLabel.RUNNING, min(evidence, 1.0), signals, track_labels

        return None, 0.0, [], {}

    # ------------------------------------------------------------------
    # Rule 5 — Suspicion
    # ------------------------------------------------------------------

    def _rule_suspicion(self, f: FrameFeatures) -> _ClassifyReturn:
        """
        SUSPICION: isolated individual whose movement stands out from the
        otherwise normal crowd — erratic direction, contrary motion, or
        anomalous speed without a group panic context.
        """
        t = self._t
        signals: List[str] = []
        evidence = 0.0
        track_labels: Dict[int, str] = {}

        for tf in f.track_features:
            tf_ev = 0.0
            tf_sig: List[str] = []

            # Large sudden direction change
            if tf.direction_change >= t.suspicion_direction_change:
                tf_sig.append(
                    f"suspicion_dir_change(id={tf.track_id},Δ={tf.direction_change:.0f}°)"
                )
                tf_ev += self._W_SUSPICION["direction_change"] * min(
                    tf.direction_change / 180.0, 1.0
                )

            # Speed anomaly but not general running
            if f.std_speed > 0:
                z = abs(tf.speed - f.mean_speed) / f.std_speed
                if z >= t.zscore_threshold and not tf.is_running:
                    tf_sig.append(
                        f"suspicion_speed_outlier(id={tf.track_id},z={z:.1f})"
                    )
                    tf_ev += self._W_SUSPICION["speed_contrast"]

            # Moving contrary to crowd (>120° from mean direction) while
            # crowd is coherent (low dispersion)
            angular_gap = _angular_diff(tf.direction_deg, f.mean_direction_deg)
            if angular_gap >= 120.0 and f.direction_dispersion < 45.0:
                tf_sig.append(
                    f"suspicion_contrary(id={tf.track_id},gap={angular_gap:.0f}°)"
                )
                tf_ev += self._W_SUSPICION["isolated"]

            if tf_ev >= 0.30:
                signals.extend(tf_sig)
                evidence = max(evidence, tf_ev)
                track_labels[tf.track_id] = BehaviorLabel.SUSPICION.value

        if track_labels:
            return (
                BehaviorLabel.SUSPICION,
                min(evidence, 1.0),
                signals,
                track_labels,
            )

        return None, 0.0, [], {}


# ============================================================================
# ML Classifier stub
# ============================================================================

class MLBehaviorClassifier(BaseBehaviorClassifier):
    """
    ML-ready classifier implementing ``BaseBehaviorClassifier``.

    Until a trained model is provided this delegates to the rule-based
    fallback with a one-time warning.  Once a model is loaded via
    ``model_path``, it calls ``_ml_classify()`` instead.

    Feature vector (12 dimensions) documented in ``_features_to_vector()``.
    Train a classifier by logging ``FrameFeatures.to_dict()`` per frame and
    fitting on the ``_features_to_vector()`` output.

    Parameters
    ----------
    model_path : optional path to a joblib-serialised sklearn / ONNX model.
    fallback   : classifier to use until a model is loaded.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        fallback:   Optional[BaseBehaviorClassifier] = None,
    ) -> None:
        self._model    = None
        self._fallback = fallback
        self._warned   = False

        if model_path:
            self._load_model(model_path)

    def _load_model(self, path: str) -> None:
        try:
            import joblib
            self._model = joblib.load(path)
            logger.info("ML classifier loaded from '%s'.", path)
        except Exception as exc:
            logger.warning(
                "ML classifier load failed ('%s'): %s — using fallback.", path, exc
            )

    def classify(
        self,
        features: FrameFeatures,
    ) -> Tuple[BehaviorLabel, float, List[str], Dict[int, str]]:
        if self._model is not None:
            return self._ml_classify(features)

        if not self._warned:
            logger.warning(
                "MLBehaviorClassifier: no model loaded — delegating to fallback."
            )
            self._warned = True

        if self._fallback:
            return self._fallback.classify(features)

        return BehaviorLabel.NORMAL, 0.5, ["ml_no_model"], {}

    def _ml_classify(
        self,
        features: FrameFeatures,
    ) -> Tuple[BehaviorLabel, float, List[str], Dict[int, str]]:
        vec   = self._features_to_vector(features)
        proba = self._model.predict_proba([vec])[0]
        idx   = int(proba.argmax())
        conf  = float(proba[idx])
        labels = list(BehaviorLabel)
        label  = labels[idx] if idx < len(labels) else BehaviorLabel.NORMAL
        return label, conf, ["ml_inference"], {}

    @staticmethod
    def _features_to_vector(features: FrameFeatures) -> List[float]:
        """
        Flatten a FrameFeatures snapshot to a 12-dimensional numeric vector.

        Dimensions
        ----------
        [0]  track_count
        [1]  mean_speed
        [2]  std_speed
        [3]  direction_dispersion
        [4]  crowd_density
        [5]  running_fraction
        [6]  anomalous_fraction
        [7]  crowd_acceleration
        [8]  proximity_pair_count
        [9]  max_zone_density
        [10] mean_abs_acceleration
        [11] mean_direction_change
        """
        tfs = features.track_features
        mean_acc = (
            sum(abs(tf.acceleration) for tf in tfs) / len(tfs) if tfs else 0.0
        )
        mean_dir_chg = (
            sum(tf.direction_change for tf in tfs) / len(tfs) if tfs else 0.0
        )
        max_zone = max(features.density_zones.values(), default=0)

        return [
            float(features.track_count),
            features.mean_speed,
            features.std_speed,
            features.direction_dispersion,
            features.crowd_density,
            features.running_fraction,
            features.anomalous_fraction,
            features.crowd_acceleration,
            float(len(features.proximity_pairs)),
            float(max_zone),
            mean_acc,
            mean_dir_chg,
        ]


# ---------------------------------------------------------------------------
# Module-level geometry helper
# ---------------------------------------------------------------------------

def _angular_diff(a_deg: float, b_deg: float) -> float:
    """Shortest arc between two compass angles; returns value in [0, 180]."""
    diff = abs(a_deg - b_deg) % 360.0
    return min(diff, 360.0 - diff)