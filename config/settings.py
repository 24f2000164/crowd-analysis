"""
config/settings.py
==================
Centralised, validated configuration for the Real-Time Crowd Behavior
Analysis System.

All settings are loaded from environment variables (or a .env file) and
validated by Pydantic at application start-up.  No setting is ever read
directly from os.environ elsewhere in the codebase — import `get_settings()`
instead.

Usage
-----
    from config.settings import get_settings

    cfg = get_settings()
    print(cfg.detection.confidence_threshold)
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import AnyUrl, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class LogLevel(str, Enum):
    """Allowed Python logging levels."""
    DEBUG    = "DEBUG"
    INFO     = "INFO"
    WARNING  = "WARNING"
    ERROR    = "ERROR"
    CRITICAL = "CRITICAL"


class DeviceType(str, Enum):
    """Compute device used for PyTorch inference."""
    CPU  = "cpu"
    CUDA = "cuda"
    MPS  = "mps"   # Apple Silicon


class VideoSourceType(str, Enum):
    """Supported video input flavours."""
    RTSP    = "rtsp"
    WEBCAM  = "webcam"
    FILE    = "file"
    HTTP    = "http"


# ---------------------------------------------------------------------------
# Nested setting groups
# (Each group is its own BaseSettings-compatible model so it can also be
#  instantiated and tested in isolation.)
# ---------------------------------------------------------------------------

class VideoSettings(BaseSettings):
    """
    Video capture / streaming parameters.

    Environment prefix: VIDEO_
    """
    model_config = SettingsConfigDict(env_prefix="VIDEO_", extra="ignore")

    # Primary source — RTSP URL, integer webcam index, or file path
    source: str = Field(
        default="0",
        description=(
            "Video source.  Accepted formats: "
            "RTSP URL (rtsp://...), webcam index ('0'), "
            "absolute/relative file path ('/data/video.mp4')."
        ),
    )
    source_type: VideoSourceType = Field(
        default=VideoSourceType.WEBCAM,
        description="Categorises the source so the capture layer picks the right backend.",
    )

    # Capture
    frame_width:  int = Field(default=1280, ge=320,  le=3840, description="Capture width  in pixels.")
    frame_height: int = Field(default=720,  ge=240,  le=2160, description="Capture height in pixels.")
    target_fps:   int = Field(default=25,   ge=1,    le=120,  description="Target processing frame-rate.")

    # Ring-buffer — drops oldest frames when the pipeline falls behind
    frame_buffer_size: int = Field(
        default=32,
        ge=4,
        le=512,
        description="Maximum frames held in the inter-thread ring buffer.",
    )

    # RTSP-specific reconnection
    rtsp_reconnect_attempts: int   = Field(default=5,    ge=0,  description="0 = no retry.")
    rtsp_reconnect_delay_s:  float = Field(default=2.0,  ge=0.0, description="Seconds between reconnect attempts.")

    # Stream output
    output_jpeg_quality: int = Field(
        default=85,
        ge=10,
        le=100,
        description="JPEG quality for frames sent over WebSocket (1–100).",
    )

    @field_validator("source")
    @classmethod
    def source_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("VIDEO_SOURCE must not be empty.")
        return v.strip()


class DetectionSettings(BaseSettings):
    """
    YOLOv8 person-detection parameters.

    Environment prefix: DETECTION_
    """
    model_config = SettingsConfigDict(env_prefix="DETECTION_", extra="ignore")

    confidence_threshold: float = Field(
        default=0.45,
        ge=0.01,
        le=1.0,
        description="Minimum confidence to keep a YOLO detection.",
    )
    iou_threshold: float = Field(
        default=0.45,
        ge=0.01,
        le=1.0,
        description="IoU threshold used during Non-Maximum Suppression.",
    )
    # YOLO only returns detections for class indices listed here.
    # COCO class 0 = person.
    target_classes: List[int] = Field(
        default=[0],
        description="COCO class IDs to retain (default: [0] = person only).",
    )
    input_image_size: int = Field(
        default=640,
        ge=320,
        le=1280,
        description="Square input resolution fed to YOLO (must be a multiple of 32).",
    )
    half_precision: bool = Field(
        default=True,
        description="Use FP16 inference on CUDA devices for higher throughput.",
    )
    max_detections: int = Field(
        default=300,
        ge=1,
        le=1000,
        description="Hard cap on detections returned per frame.",
    )

    @field_validator("input_image_size")
    @classmethod
    def must_be_multiple_of_32(cls, v: int) -> int:
        if v % 32 != 0:
            raise ValueError(f"input_image_size ({v}) must be a multiple of 32.")
        return v


class TrackingSettings(BaseSettings):
    """
    DeepSORT multi-object tracking parameters.

    Environment prefix: TRACKING_
    """
    model_config = SettingsConfigDict(env_prefix="TRACKING_", extra="ignore")

    # Kalman filter / assignment
    max_age: int = Field(
        default=30,
        ge=1,
        description="Frames a track can be unmatched before deletion.",
    )
    min_hits: int = Field(
        default=3,
        ge=1,
        description="Consecutive detections required before a track is confirmed.",
    )
    iou_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="IoU threshold for the Hungarian assignment step.",
    )

    # Re-ID appearance metric
    max_cosine_distance: float = Field(
        default=0.4,
        ge=0.0,
        le=2.0,
        description="Cosine distance threshold for appearance-based re-identification.",
    )
    nn_budget: Optional[int] = Field(
        default=100,
        ge=1,
        description=(
            "Maximum size of the appearance descriptor gallery per track. "
            "None = unlimited."
        ),
    )

    # Trajectory history
    trajectory_history_length: int = Field(
        default=60,
        ge=5,
        le=600,
        description="Rolling window size (frames) for per-track trajectory storage.",
    )

    # Velocity / behaviour thresholds
    velocity_smoothing_alpha: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="EMA smoothing factor for velocity vectors (0 = no smoothing).",
    )
    speed_run_threshold_px_per_frame: float = Field(
        default=15.0,
        ge=1.0,
        description="Pixels-per-frame speed above which a track is labelled 'running'.",
    )
    anomaly_zscore_threshold: float = Field(
        default=2.5,
        ge=0.5,
        description="Z-score deviation from crowd mean velocity to flag as anomalous.",
    )


class ModelSettings(BaseSettings):
    """
    File-system paths to model weights.

    Environment prefix: MODEL_
    All paths are resolved to absolute Path objects at validation time.
    """
    model_config = SettingsConfigDict(env_prefix="MODEL_", extra="ignore")

    base_dir: Path = Field(
        default=Path("models"),
        description="Root directory that holds all weight files.",
    )

    # YOLOv8 weights
    yolo_weights: Path = Field(
        default=Path("models/yolov8n.pt"),
        description=(
            "Path to YOLOv8 weights file.  "
            "Options: yolov8n.pt / yolov8s.pt / yolov8m.pt / yolov8l.pt / yolov8x.pt "
            "— or a custom fine-tuned checkpoint."
        ),
    )

    # DeepSORT Re-ID encoder weights
    reid_weights: Path = Field(
        default=Path("models/osnet_x0_25_msmt17.pt"),
        description="Path to the OSNet re-identification encoder checkpoint.",
    )

    @field_validator("yolo_weights", "reid_weights", mode="before")
    @classmethod
    def coerce_to_path(cls, v: object) -> Path:
        return Path(str(v))

    @model_validator(mode="after")
    def weights_must_exist_if_not_placeholder(self) -> "ModelSettings":
        """
        Warn (rather than hard-fail) when weights are missing so the app can
        start in a 'download-pending' state and fetch weights on first use.
        """
        for attr in ("yolo_weights", "reid_weights"):
            p: Path = getattr(self, attr)
            if not p.exists():
                import warnings
                warnings.warn(
                    f"Model weight file not found: {p}.  "
                    "Run scripts/download_models.py before starting inference.",
                    stacklevel=2,
                )
        return self


class BehaviorSettings(BaseSettings):
    """
    Crowd behaviour analysis thresholds.

    Environment prefix: BEHAVIOR_
    """
    model_config = SettingsConfigDict(env_prefix="BEHAVIOR_", extra="ignore")

    # Density
    density_zone_rows: int = Field(default=4, ge=1, le=20, description="Grid rows for zone-based density map.")
    density_zone_cols: int = Field(default=4, ge=1, le=20, description="Grid cols for zone-based density map.")
    density_alert_threshold: int = Field(
        default=10,
        ge=1,
        description="Persons per zone above which a density alert fires.",
    )

    # Event classification
    panic_min_anomalous_fraction: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Fraction of confirmed tracks that must be anomalous to trigger PANIC.",
    )
    fight_proximity_px: float = Field(
        default=80.0,
        ge=10.0,
        description="Pixel distance within which two anomalous tracks are considered fighting.",
    )
    surge_velocity_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        description="Crowd mean speed must exceed this × baseline to trigger CROWD_SURGE.",
    )

    # Alerting cool-down (prevents repeated alerts for the same event)
    alert_cooldown_s: float = Field(
        default=5.0,
        ge=0.0,
        description="Minimum seconds between consecutive alerts of the same type.",
    )


class APISettings(BaseSettings):
    """
    FastAPI server and WebSocket settings.

    Environment prefix: API_
    """
    model_config = SettingsConfigDict(env_prefix="API_", extra="ignore")

    host: str  = Field(default="0.0.0.0", description="Bind address.")
    port: int  = Field(default=8000, ge=1024, le=65535, description="Bind port.")
    workers: int = Field(default=1, ge=1, description="Uvicorn worker count (use 1 for dev).")

    # WebSocket
    ws_ping_interval_s: float = Field(default=20.0, ge=1.0, description="WS keepalive ping interval.")
    ws_ping_timeout_s:  float = Field(default=10.0, ge=1.0, description="WS ping timeout before disconnect.")
    ws_max_queue:       int   = Field(default=32,   ge=1,   description="Max frames queued per WS connection.")

    # Security
    api_key: Optional[str] = Field(
        default=None,
        description="Static API key for WebSocket auth.  None = no auth (dev only).",
    )
    cors_origins: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins.  Lock down in production.",
    )

    # Metrics
    enable_prometheus: bool = Field(default=True, description="Expose /metrics endpoint.")


class StorageSettings(BaseSettings):
    """
    Event persistence layer.

    Environment prefix: STORAGE_
    """
    model_config = SettingsConfigDict(env_prefix="STORAGE_", extra="ignore")

    database_url: str = Field(
        default="sqlite+aiosqlite:///./crowd_events.db",
        description=(
            "SQLAlchemy async database URL.  "
            "SQLite for dev, postgresql+asyncpg://... for production."
        ),
    )
    event_retention_days: int = Field(
        default=30,
        ge=1,
        description="Behavioral events older than this are pruned automatically.",
    )
    echo_sql: bool = Field(default=False, description="Log all SQL statements (dev only).")


# ---------------------------------------------------------------------------
# Root settings object
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Top-level settings aggregator.

    Reads a .env file from the project root (if present) and delegates each
    domain to its own nested settings group.  All nested models also honour
    their own environment-variable prefixes, so both of the following work:

        DETECTION_CONFIDENCE_THRESHOLD=0.5   (via DetectionSettings prefix)
        VIDEO_SOURCE=rtsp://192.168.1.10/...  (via VideoSettings prefix)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Top-level settings
    # ------------------------------------------------------------------ #
    app_name:    str      = Field(default="Crowd Analysis System", description="Human-readable service name.")
    app_version: str      = Field(default="1.0.0",                 description="Semantic version string.")
    environment: str      = Field(default="development",           description="development | staging | production")
    debug:       bool     = Field(default=False,                   description="Enable debug mode (never True in prod).")
    log_level:   LogLevel = Field(default=LogLevel.INFO,           description="Root log level.")
    device:      DeviceType = Field(
        default=DeviceType.CPU,
        description="Inference device.  Override with DEVICE=cuda for GPU.",
    )

    # ------------------------------------------------------------------ #
    # Nested groups — each resolves its own env-prefix independently
    # ------------------------------------------------------------------ #
    video:     VideoSettings     = Field(default_factory=VideoSettings)
    detection: DetectionSettings = Field(default_factory=DetectionSettings)
    tracking:  TrackingSettings  = Field(default_factory=TrackingSettings)
    models:    ModelSettings     = Field(default_factory=ModelSettings)
    behavior:  BehaviorSettings  = Field(default_factory=BehaviorSettings)
    api:       APISettings       = Field(default_factory=APISettings)
    storage:   StorageSettings   = Field(default_factory=StorageSettings)

    # ------------------------------------------------------------------ #
    # Cross-cutting validators
    # ------------------------------------------------------------------ #
    @model_validator(mode="after")
    def production_safety_checks(self) -> "Settings":
        """Enforce stricter rules when environment == 'production'."""
        if self.environment == "production":
            if self.debug:
                raise ValueError("debug must be False in production.")
            if self.api.api_key is None:
                raise ValueError("API_API_KEY must be set in production.")
            if "*" in self.api.cors_origins:
                raise ValueError("CORS wildcard '*' is not allowed in production.")
        return self

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_gpu_available(self) -> bool:
        """True when the configured device is CUDA and torch reports it available."""
        import torch
        return self.device == DeviceType.CUDA and torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Singleton accessor — use this everywhere outside of tests
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the validated, cached Settings singleton.

    The result is cached so Pydantic only parses environment variables once
    per process.  In tests, call ``get_settings.cache_clear()`` after
    monkeypatching env vars to force a fresh parse.

    Returns
    -------
    Settings
        The fully validated application configuration.
    """
    return Settings()