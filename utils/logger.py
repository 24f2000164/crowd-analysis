"""
src/utils/logger.py
====================
Centralised logging module for the Real-Time Crowd Behavior Analysis System.

This is the single entry-point for all logging configuration and logger
creation throughout the project.  No other module should call
``logging.basicConfig()`` or ``logging.config.dictConfig()`` directly.

Architecture
------------

    ┌─────────────────────────────────────────────────────────────┐
    │                    logger.py  (this file)                   │
    │                                                             │
    │  setup_logging(config)  ──► dictConfig / programmatic       │
    │                              │                              │
    │              ┌───────────────┼──────────────────┐           │
    │              ▼               ▼                  ▼           │
    │         AsyncQueue      RotatingFile        Console         │
    │          Handler           Handler          Handler         │
    │              │               │                  │           │
    │              ▼               ▼                  ▼           │
    │         JSONFormatter   JSONFormatter   ConsoleFormatter    │
    │                                                             │
    │  get_logger(name)       ──► standard Python logger          │
    │  get_stream_logger(id)  ──► StreamContextAdapter            │
    │  get_pipeline_logger()  ──► PipelineStageAdapter            │
    │  get_request_logger()   ──► RequestContextAdapter           │
    └─────────────────────────────────────────────────────────────┘

Log levels per subsystem
------------------------
    crowd_analysis               INFO    (all sub-loggers inherit unless overridden)
    crowd_analysis.detection     INFO
    crowd_analysis.tracking      INFO
    crowd_analysis.behavior      INFO
    crowd_analysis.pipeline      DEBUG   (frame-level debug info)
    crowd_analysis.api           INFO
    crowd_analysis.api.websocket DEBUG
    ultralytics                  WARNING (suppress noisy training output)
    uvicorn                      INFO

Rotation policy
---------------
    logs/crowd_analysis.log        — 10 MiB per file, 5 backups (50 MiB total)
    logs/crowd_analysis_errors.log — 5 MiB per file, 3 backups (15 MiB total)
    logs/behavior_events.log       — 5 MiB per file, 10 backups (50 MiB total)
    logs/crowd_analysis_YYYY-MM-DD.log — midnight rotation, 30-day retention

Usage
-----
    # Application entry-point
    from src.utils.logger import setup_logging
    setup_logging()

    # Anywhere in the codebase
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Module loaded")

    # Pipeline with structured context
    from src.utils.logger import get_stream_logger
    log = get_stream_logger("a1b2c3d4")
    log.info("Frame processed", extra={"detections": 7})
    log.set_frame(42)
"""

from __future__ import annotations

import logging
import logging.config
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.formatters import ConsoleFormatter, JSONFormatter, PipelineFormatter
from src.utils.handlers   import (
    AsyncQueueHandler,
    BehaviorEventHandler,
    SafeRotatingFileHandler,
    SafeTimedRotatingFileHandler,
)
from src.utils.context import (
    LogContext,
    PipelineStageAdapter,
    RequestContextAdapter,
    StreamContextAdapter,
)


# ============================================================================
# Configuration dataclass
# ============================================================================

@dataclass
class LoggingConfig:
    """
    All knobs for the logging subsystem.

    Attributes
    ----------
    log_dir          : directory where log files are created.
    root_level       : minimum level for the root logger.
    app_level        : level for the ``crowd_analysis`` package logger.
    console_level    : level for the console (stdout) handler.
    file_level       : level for rotating file handlers.
    enable_json      : use JSON formatter on the file handler (production).
    enable_console   : emit log records to stdout.
    enable_file      : write log records to rotating files.
    enable_async     : wrap file handlers in AsyncQueueHandler (recommended
                       for production — decouples log I/O from inference loop).
    enable_timed     : also write to a timed (daily) log file.
    max_bytes        : rotating file max size in bytes.
    backup_count     : number of rotated files to keep.
    timed_backup_days: timed log retention days.
    service          : service name injected into every JSON record.
    environment      : environment tag (development / staging / production).
    extra_loggers    : ``{logger_name: level}`` overrides applied on top of
                       the built-in defaults.
    """
    log_dir:            Path  = Path("logs")
    root_level:         str   = "WARNING"
    app_level:          str   = "INFO"
    console_level:      str   = "DEBUG"
    file_level:         str   = "DEBUG"
    enable_json:        bool  = False        # True in production
    enable_console:     bool  = True
    enable_file:        bool  = True
    enable_async:       bool  = True
    enable_timed:       bool  = False
    max_bytes:          int   = 10 * 1024 * 1024   # 10 MiB
    backup_count:       int   = 5
    timed_backup_days:  int   = 30
    service:            str   = "crowd-analysis"
    environment:        str   = "development"
    extra_loggers:      Dict[str, str] = field(default_factory=dict)


# Default per-subsystem log levels
_DEFAULT_LOGGER_LEVELS: Dict[str, str] = {
    "crowd_analysis":                "INFO",
    "crowd_analysis.detection":      "INFO",
    "crowd_analysis.tracking":       "INFO",
    "crowd_analysis.behavior":       "INFO",
    "crowd_analysis.pipeline":       "DEBUG",
    "crowd_analysis.api":            "INFO",
    "crowd_analysis.api.websocket":  "DEBUG",
    "crowd_analysis.video":          "INFO",
    "crowd_analysis.annotation":     "INFO",
    # Third-party noise suppression
    "ultralytics":                   "WARNING",
    "uvicorn":                       "INFO",
    "uvicorn.error":                 "INFO",
    "uvicorn.access":                "WARNING",
    "httpx":                         "WARNING",
    "asyncio":                       "WARNING",
}


# ============================================================================
# Primary setup function
# ============================================================================

def setup_logging(
    config:    Optional[LoggingConfig] = None,
    yaml_path: Optional[Path]          = None,
    env_override: bool                 = True,
) -> None:
    """
    Configure the Python logging system for the entire application.

    Call this exactly **once** at application startup before any logger is
    used.  Subsequent calls are no-ops (guarded by ``_CONFIGURED`` flag).

    Priority order for configuration source
    ----------------------------------------
    1. ``yaml_path``   — load a ``dictConfig``-compatible YAML file.
    2. ``config``      — use the provided ``LoggingConfig`` dataclass.
    3. Environment     — fall back to ``LOG_LEVEL`` + ``LOG_FORMAT`` env vars.
    4. Defaults        — safe ``INFO`` console-only baseline.

    Parameters
    ----------
    config       : ``LoggingConfig`` instance (programmatic configuration).
    yaml_path    : path to a ``logging.yaml`` file (takes precedence over config).
    env_override : when True, the ``LOG_LEVEL`` environment variable overrides
                   ``config.app_level`` so operators can change verbosity
                   without redeploying.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    # ── 1. YAML config (highest precedence) ───────────────────────────
    if yaml_path and Path(yaml_path).exists():
        _configure_from_yaml(yaml_path)
        _CONFIGURED = True
        _post_configure(config or LoggingConfig())
        return

    # Try the default project path
    default_yaml = Path("config/logging.yaml")
    if default_yaml.exists():
        _configure_from_yaml(default_yaml)
        _CONFIGURED = True
        _post_configure(config or LoggingConfig())
        return

    # ── 2. Programmatic config ─────────────────────────────────────────
    cfg = config or LoggingConfig()

    # ── 3. Env var override ────────────────────────────────────────────
    if env_override:
        env_level = os.environ.get("LOG_LEVEL", "").upper()
        if env_level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            cfg.app_level = env_level

        env_format = os.environ.get("LOG_FORMAT", "").lower()
        if env_format == "json":
            cfg.enable_json = True
        elif env_format == "console":
            cfg.enable_json = False

        if os.environ.get("ENVIRONMENT", "").lower() == "production":
            cfg.enable_json = True
            cfg.environment = "production"

    # ── 4. Build handlers ──────────────────────────────────────────────
    handlers = _build_handlers(cfg)

    # ── 5. Configure root logger ───────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(cfg.root_level)
    # Remove any pre-existing handlers (e.g. from basicConfig auto-setup)
    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()
    for h in handlers["root"]:
        root.addHandler(h)

    # ── 6. Configure per-subsystem loggers ─────────────────────────────
    levels = {**_DEFAULT_LOGGER_LEVELS, **cfg.extra_loggers}
    for name, level in levels.items():
        log = logging.getLogger(name)
        log.setLevel(level)
        log.propagate = True   # All records bubble up to the root handler set

    # ── 7. Attach dedicated handlers to the app root logger ────────────
    app_logger = logging.getLogger("crowd_analysis")
    app_logger.setLevel(cfg.app_level)
    for h in handlers["app"]:
        app_logger.addHandler(h)
    app_logger.propagate = False   # Don't double-emit to the root console

    _CONFIGURED = True
    _post_configure(cfg)


# Guard against double-initialisation
_CONFIGURED: bool = False


def reset_logging() -> None:
    """Reset the initialisation guard. Use only in tests."""
    global _CONFIGURED
    _CONFIGURED = False


# ============================================================================
# Logger factory functions
# ============================================================================

def get_logger(name: str) -> logging.Logger:
    """
    Return a standard ``logging.Logger`` for ``name``.

    Follows the convention of the rest of the project:
    ``crowd_analysis.<subsystem>`` names are automatically configured.

    Parameters
    ----------
    name : typically ``__name__`` of the calling module.
    """
    return logging.getLogger(name)


def get_stream_logger(
    stream_id:   str,
    frame_index: int  = 0,
    name:        str  = "crowd_analysis.pipeline",
) -> StreamContextAdapter:
    """
    Return a ``StreamContextAdapter`` pre-configured with a stream context.

    Every record emitted by the returned adapter will include
    ``stream_id`` and ``frame_index`` as structured fields.

    Parameters
    ----------
    stream_id   : active stream identifier.
    frame_index : starting frame index.
    name        : underlying logger name.
    """
    logger = logging.getLogger(name)
    return StreamContextAdapter(logger, stream_id=stream_id, frame_index=frame_index)


def get_pipeline_logger(
    stream_id:   str   = "",
    stage:       str   = "",
    frame_index: int   = 0,
    fps:         float = 0.0,
    name:        str   = "crowd_analysis.pipeline",
) -> PipelineStageAdapter:
    """
    Return a ``PipelineStageAdapter`` for a specific pipeline stage.

    Parameters
    ----------
    stream_id   : active stream identifier.
    stage       : pipeline stage name (detect / track / behavior / annotate).
    frame_index : starting frame index.
    fps         : initial rolling FPS.
    name        : underlying logger name.
    """
    logger = logging.getLogger(name)
    return PipelineStageAdapter(
        logger,
        stream_id=stream_id,
        stage=stage,
        frame_index=frame_index,
        fps=fps,
    )


def get_request_logger(
    request_id: str = "",
    method:     str = "",
    path:       str = "",
    client_ip:  str = "",
    name:       str = "crowd_analysis.api",
) -> RequestContextAdapter:
    """
    Return a ``RequestContextAdapter`` scoped to an HTTP request.

    Parameters
    ----------
    request_id : X-Request-ID correlation identifier.
    method     : HTTP method.
    path       : URL path.
    client_ip  : remote client IP address.
    name       : underlying logger name.
    """
    logger = logging.getLogger(name)
    return RequestContextAdapter(
        logger,
        request_id=request_id,
        method=method,
        path=path,
        client_ip=client_ip,
    )


# ============================================================================
# Internal helpers
# ============================================================================

def _configure_from_yaml(yaml_path: Path) -> None:
    """Load ``dictConfig`` from a YAML file."""
    try:
        import yaml
        with yaml_path.open(encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        # Ensure the logs directory exists before dictConfig creates handlers
        _ensure_log_dirs(config_dict)

        logging.config.dictConfig(config_dict)

        logging.getLogger("crowd_analysis").info(
            "Logging configured from YAML: %s", yaml_path
        )
    except ImportError:
        sys.stderr.write(
            "[logging] PyYAML not installed — falling back to programmatic config.\n"
        )
        raise
    except Exception as exc:
        sys.stderr.write(
            f"[logging] Failed to load {yaml_path}: {exc} — falling back.\n"
        )
        raise


def _ensure_log_dirs(config_dict: dict) -> None:
    """
    Pre-create log directories referenced in a dictConfig dict so that
    file handlers don't fail on missing parents.
    """
    handlers = config_dict.get("handlers", {})
    for h_cfg in handlers.values():
        filename = h_cfg.get("filename")
        if filename:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)


def _build_handlers(cfg: LoggingConfig) -> Dict[str, List[logging.Handler]]:
    """
    Build all handlers from a ``LoggingConfig``.

    Returns a dict with two keys:
    - ``"root"`` : handlers attached to the root logger.
    - ``"app"``  : extra handlers attached only to ``crowd_analysis``.
    """
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # ── Formatter selection ────────────────────────────────────────────
    file_formatter = (
        JSONFormatter(service=cfg.service, environment=cfg.environment)
        if cfg.enable_json
        else ConsoleFormatter()
    )
    console_formatter = ConsoleFormatter()

    # ── Console handler ────────────────────────────────────────────────
    console_h: Optional[logging.Handler] = None
    if cfg.enable_console:
        console_h = logging.StreamHandler(sys.stdout)
        console_h.setLevel(cfg.console_level)
        console_h.setFormatter(console_formatter)

    # ── Rotating file handler (main log) ──────────────────────────────
    main_file_h: Optional[logging.Handler] = None
    error_file_h: Optional[logging.Handler] = None
    behavior_h:   Optional[logging.Handler] = None

    if cfg.enable_file:
        main_rotating = SafeRotatingFileHandler(
            filename     = cfg.log_dir / "crowd_analysis.log",
            max_bytes    = cfg.max_bytes,
            backup_count = cfg.backup_count,
        )
        main_rotating.setLevel(cfg.file_level)
        main_rotating.setFormatter(file_formatter)

        error_rotating = SafeRotatingFileHandler(
            filename     = cfg.log_dir / "crowd_analysis_errors.log",
            max_bytes    = cfg.max_bytes // 2,
            backup_count = max(cfg.backup_count - 2, 1),
        )
        error_rotating.setLevel(logging.ERROR)
        error_rotating.setFormatter(
            JSONFormatter(service=cfg.service, environment=cfg.environment)
        )

        behavior_handler = BehaviorEventHandler(
            filename     = cfg.log_dir / "behavior_events.log",
            max_bytes    = 5 * 1024 * 1024,
            backup_count = 10,
        )
        behavior_handler.setFormatter(
            JSONFormatter(service=cfg.service, environment=cfg.environment)
        )

        if cfg.enable_async:
            main_file_h  = AsyncQueueHandler([main_rotating])
            error_file_h = AsyncQueueHandler([error_rotating])
            behavior_h   = AsyncQueueHandler([behavior_handler])
        else:
            main_file_h  = main_rotating
            error_file_h = error_rotating
            behavior_h   = behavior_handler

    # ── Timed rotating handler (daily archives) ───────────────────────
    timed_h: Optional[logging.Handler] = None
    if cfg.enable_timed:
        timed = SafeTimedRotatingFileHandler(
            filename     = cfg.log_dir / "crowd_analysis.log",
            when         = "midnight",
            backup_count = cfg.timed_backup_days,
            utc          = True,
        )
        timed.setLevel(cfg.file_level)
        timed.setFormatter(file_formatter)
        timed_h = AsyncQueueHandler([timed]) if cfg.enable_async else timed

    # ── Assemble ───────────────────────────────────────────────────────
    root_handlers: List[logging.Handler] = []
    if console_h:
        root_handlers.append(console_h)

    app_handlers: List[logging.Handler] = []
    for h in (main_file_h, error_file_h, behavior_h, timed_h):
        if h is not None:
            app_handlers.append(h)

    return {"root": root_handlers, "app": app_handlers}


def _post_configure(cfg: LoggingConfig) -> None:
    """Emit a confirmation log and apply final tweaks after configuration."""
    logger = logging.getLogger("crowd_analysis")
    logger.info(
        "Logging initialised — level=%s  json=%s  async=%s  log_dir=%s",
        cfg.app_level, cfg.enable_json, cfg.enable_async, cfg.log_dir,
    )

