"""
scripts/train_behavior_model.py
=================================
Training pipeline for the ML-based crowd behavior classifier.

Reads the CSV dataset produced by build_behavior_dataset.py, trains a
RandomForestClassifier (and optionally XGBoostClassifier), evaluates both
models, and saves the best-performing one to disk.

Outputs
-------
    models/crowd_behavior_model.pkl   — trained classifier
    models/crowd_behavior_scaler.pkl  — fitted StandardScaler
    models/training_report.json       — metrics and class distribution

Usage
-----
    python scripts/train_behavior_model.py \
        --dataset data/crowd_behavior_dataset.csv \
        --model-dir models/ \
        --test-size 0.2 \
        --use-xgboost
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.behavior.trajectory_features import FeatureVector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
)
logger = logging.getLogger("train_behavior_model")

FEATURE_COLS = FeatureVector.feature_names()
LABEL_COL    = "label"

MODEL_PATH  = Path("models/crowd_behavior_model.pkl")
SCALER_PATH = Path("models/crowd_behavior_scaler.pkl")
ENCODER_PATH = Path("models/crowd_behavior_label_encoder.pkl")
REPORT_PATH = Path("models/training_report.json")


# ── Data loading ──────────────────────────────────────────────────────────

def load_dataset(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Load the CSV, encode labels, and return (X, y, encoder).

    Raises
    ------
    FileNotFoundError : if csv_path does not exist.
    ValueError        : if required columns are missing.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info("Loaded dataset: %d rows × %d cols", len(df), len(df.columns))

    # Validate columns
    missing = [c for c in FEATURE_COLS + [LABEL_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    # Drop rows with NaN in feature columns
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS + [LABEL_COL])
    if len(df) < before:
        logger.warning("Dropped %d rows with NaN values.", before - len(df))

    X = df[FEATURE_COLS].values.astype(np.float32)
    raw_labels = df[LABEL_COL].str.lower().values

    encoder = LabelEncoder()
    y = encoder.fit_transform(raw_labels)

    logger.info(
        "Classes: %s  |  Distribution: %s",
        list(encoder.classes_),
        dict(zip(*np.unique(raw_labels, return_counts=True))),
    )
    return X, y, encoder


# ── Training ──────────────────────────────────────────────────────────────

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train and return a RandomForestClassifier."""
    logger.info(
        "Training RandomForestClassifier (n_estimators=%d) …", n_estimators
    )
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    random_state: int = 42,
) -> Any:
    """Train and return an XGBoostClassifier (optional dependency)."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.warning("xgboost not installed — skipping XGBoost training.")
        return None

    logger.info("Training XGBoostClassifier …")
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        num_class=n_classes if n_classes > 2 else None,
        objective="multi:softprob" if n_classes > 2 else "binary:logistic",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


# ── Evaluation ────────────────────────────────────────────────────────────

def evaluate(
    clf,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    encoder: LabelEncoder,
    model_name: str,
) -> Dict[str, Any]:
    """
    Compute and log evaluation metrics.

    Returns a dict suitable for JSON serialisation.
    """
    y_pred = clf.predict(X_test)
    labels = list(encoder.classes_)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(
        y_test, y_pred, target_names=labels, zero_division=0
    )

    logger.info("── %s Evaluation ──────────────────────────────", model_name)
    logger.info("  Accuracy  : %.4f", acc)
    logger.info("  Precision : %.4f", prec)
    logger.info("  Recall    : %.4f", rec)
    logger.info("  F1-Score  : %.4f", f1)
    logger.info("\n%s", report)

    return {
        "model":              model_name,
        "accuracy":           round(acc, 4),
        "precision_weighted": round(prec, 4),
        "recall_weighted":    round(rec, 4),
        "f1_weighted":        round(f1, 4),
        "confusion_matrix":   cm,
        "classes":            labels,
    }


# ── Feature importance ────────────────────────────────────────────────────

def log_feature_importance(clf, feature_names: list) -> None:
    """Log feature importances if the model supports them."""
    if not hasattr(clf, "feature_importances_"):
        return
    importances = clf.feature_importances_
    ranked = sorted(
        zip(feature_names, importances), key=lambda x: x[1], reverse=True
    )
    logger.info("Feature importances:")
    for name, imp in ranked:
        logger.info("  %-30s %.4f", name, imp)


# ── Main training flow ────────────────────────────────────────────────────

def run_training(
    dataset_path: Path,
    model_dir:    Path,
    test_size:    float = 0.2,
    use_xgboost:  bool  = False,
    random_state: int   = 42,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────
    X, y, encoder = load_dataset(dataset_path)

    # ── Scale ─────────────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Split ─────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    logger.info(
        "Train: %d  Test: %d  (%.0f%% / %.0f%%)",
        len(X_train), len(X_test),
        (1 - test_size) * 100, test_size * 100,
    )

    # ── Train RF ──────────────────────────────────────────────────────
    rf_clf    = train_random_forest(X_train, y_train, random_state=random_state)
    rf_metrics = evaluate(rf_clf, X_test, y_test, encoder, "RandomForest")
    log_feature_importance(rf_clf, FEATURE_COLS)

    best_clf     = rf_clf
    best_metrics = rf_metrics
    all_reports  = [rf_metrics]

    # ── Optional XGBoost ──────────────────────────────────────────────
    if use_xgboost:
        xgb_clf = train_xgboost(
            X_train, y_train,
            n_classes=len(encoder.classes_),
            random_state=random_state,
        )
        if xgb_clf is not None:
            xgb_metrics = evaluate(xgb_clf, X_test, y_test, encoder, "XGBoost")
            all_reports.append(xgb_metrics)
            if xgb_metrics["f1_weighted"] > best_metrics["f1_weighted"]:
                logger.info(
                    "XGBoost outperforms RF (F1 %.4f > %.4f) — selecting XGBoost.",
                    xgb_metrics["f1_weighted"], best_metrics["f1_weighted"],
                )
                best_clf     = xgb_clf
                best_metrics = xgb_metrics

    # ── Save artefacts ────────────────────────────────────────────────
    joblib.dump(best_clf, model_dir / "crowd_behavior_model.pkl")
    joblib.dump(scaler,   model_dir / "crowd_behavior_scaler.pkl")
    joblib.dump(encoder,  model_dir / "crowd_behavior_label_encoder.pkl")
    logger.info("Model saved  → %s", model_dir / "crowd_behavior_model.pkl")
    logger.info("Scaler saved → %s", model_dir / "crowd_behavior_scaler.pkl")

    report = {
        "best_model":   best_metrics["model"],
        "best_f1":      best_metrics["f1_weighted"],
        "n_train":      int(len(X_train)),
        "n_test":       int(len(X_test)),
        "feature_cols": FEATURE_COLS,
        "classes":      list(encoder.classes_),
        "all_reports":  all_reports,
    }
    with (model_dir / "training_report.json").open("w") as f:
        json.dump(report, f, indent=2)
    logger.info(
        "Training complete — best model: %s  F1: %.4f",
        best_metrics["model"], best_metrics["f1_weighted"],
    )


# ── CLI ───────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the crowd behavior ML classifier."
    )
    p.add_argument(
        "--dataset", "-d",
        type=Path,
        default=Path("data/crowd_behavior_dataset.csv"),
        help="Path to the CSV dataset.",
    )
    p.add_argument(
        "--model-dir", "-m",
        type=Path,
        default=Path("models"),
        help="Directory to save model artefacts.",
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to reserve for testing (default: 0.2).",
    )
    p.add_argument(
        "--use-xgboost",
        action="store_true",
        help="Also train an XGBoost model and keep the better one.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_training(
        dataset_path = args.dataset,
        model_dir    = args.model_dir,
        test_size    = args.test_size,
        use_xgboost  = args.use_xgboost,
        random_state = args.seed,
    )