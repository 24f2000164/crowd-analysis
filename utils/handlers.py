"""
src/utils/handlers.py
======================
Custom logging handlers for the Crowd Analysis System.

Handlers
--------
SafeRotatingFileHandler    — extends ``RotatingFileHandler`` with:
                             • directory auto-creation
                             • permissions hardening (0o600)
                             • silent fallback to stderr on write failure
                             • UTF-8 encoding enforced

TimedRotatingFileHandler   — wraps stdlib ``TimedRotatingFileHandler`` with
                             the same safety improvements plus configurable
                             ``when`` / ``interval`` / ``backupCount``.

AsyncQueueHandler          — non-blocking handler that enqueues records and
                             forwards them to a ``QueueListener`` running in a
                             background thread.  Prevents log I/O from adding
                             latency to the inference pipeline.

BehaviorEventHandler       — specialist handler that fires only for
                             WARNING+ records from the behavior logger,
                             writes to a dedicated ``behavior_events.log``
                             for post-analysis / alerting systems.

Usage
-----
    from src.utils.handlers import SafeRotatingFileHandler, AsyncQueueHandler

    file_handler  = SafeRotatingFileHandler("logs/app.log", max_bytes=10_485_760)
    async_handler = AsyncQueueHandler([file_handler])
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import queue
import sys
from pathlib import Path
from typing import List, Optional


# ============================================================================
# Safe Rotating File Handler
# ============================================================================

class SafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """
    ``RotatingFileHandler`` with production safety improvements.

    Parameters
    ----------
    filename    : path to the log file.
    max_bytes   : rotate when the file exceeds this size.  Default 10 MiB.
    backup_count: number of rotated files to keep.  Default 5.
    encoding    : file encoding.  Default UTF-8.
    """

    def __init__(
        self,
        filename:     str | Path,
        max_bytes:    int = 10 * 1024 * 1024,   # 10 MiB
        backup_count: int = 5,
        encoding:     str = "utf-8",
        **kwargs,
    ) -> None:
        path = Path(filename)
        # Auto-create parent directories so callers never have to
        path.parent.mkdir(parents=True, exist_ok=True)

        super().__init__(
            filename     = str(path),
            maxBytes     = max_bytes,
            backupCount  = backup_count,
            encoding     = encoding,
            **kwargs,
        )

        # Harden file permissions to owner read/write only (Unix only)
        self._harden_permissions(str(path))

    # ------------------------------------------------------------------
    # Override emit() with silent-fallback on I/O failure
    # ------------------------------------------------------------------

    def emit(self, record: logging.LogRecord) -> None:
        try:
            super().emit(record)
        except Exception:
            # Never let a logging failure crash the application
            self._silent_fallback(record)

    def doRollover(self) -> None:
        try:
            super().doRollover()
        except Exception as exc:
            sys.stderr.write(f"[logging] Rollover failed: {exc}\n")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _harden_permissions(path: str) -> None:
        """Set file permissions to 0o600 (owner r/w only) on POSIX systems."""
        if os.name == "posix":
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass

    @staticmethod
    def _silent_fallback(record: logging.LogRecord) -> None:
        """Write the formatted record to stderr when the file handler fails."""
        try:
            sys.stderr.write(f"[LOGGING FALLBACK] {record.getMessage()}\n")
        except Exception:
            pass


# ============================================================================
# Safe Timed Rotating File Handler
# ============================================================================

class SafeTimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """
    ``TimedRotatingFileHandler`` with auto-directory creation and
    UTF-8 encoding enforcement.

    Parameters
    ----------
    filename     : path to the log file.
    when         : rotation interval unit.  Default ``"midnight"``.
    interval     : number of ``when`` units between rotations.  Default 1.
    backup_count : number of rotated files to keep.  Default 7.
    """

    def __init__(
        self,
        filename:     str | Path,
        when:         str = "midnight",
        interval:     int = 1,
        backup_count: int = 7,
        encoding:     str = "utf-8",
        utc:          bool = True,
        **kwargs,
    ) -> None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        super().__init__(
            filename    = str(path),
            when        = when,
            interval    = interval,
            backupCount = backup_count,
            encoding    = encoding,
            utc         = utc,
            **kwargs,
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            super().emit(record)
        except Exception:
            try:
                sys.stderr.write(f"[logging fallback] {record.getMessage()}\n")
            except Exception:
                pass


# ============================================================================
# Async Queue Handler  (non-blocking pipeline-safe logging)
# ============================================================================

class AsyncQueueHandler(logging.handlers.QueueHandler):
    """
    Non-blocking ``QueueHandler`` backed by a ``QueueListener``.

    Log records are put on a ``queue.Queue`` by the calling thread and
    forwarded to the real handlers (file, console) by a single background
    daemon thread.  This means the pipeline's inference loop never waits
    for file I/O.

    Parameters
    ----------
    handlers     : list of downstream handlers (file, console, etc.).
    queue_size   : internal queue depth.  -1 = unbounded.
    respect_handler_level : forward only records that pass each handler's level.

    Usage
    -----
        file_h  = SafeRotatingFileHandler("logs/app.log")
        async_h = AsyncQueueHandler([file_h])
        logging.getLogger().addHandler(async_h)

        # On shutdown:
        async_h.stop()
    """

    def __init__(
        self,
        handlers:               List[logging.Handler],
        queue_size:             int  = -1,
        respect_handler_level:  bool = True,
    ) -> None:
        log_queue: queue.Queue = queue.Queue(maxsize=queue_size if queue_size > 0 else 0)
        super().__init__(log_queue)

        self._listener = logging.handlers.QueueListener(
            log_queue,
            *handlers,
            respect_handler_level=respect_handler_level,
        )
        self._listener.start()

    def stop(self) -> None:
        """Stop the background listener thread.  Call at application shutdown."""
        self._listener.stop()

    def emit(self, record: logging.LogRecord) -> None:
        """Enqueue without blocking.  Drop record silently if queue is full."""
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            # Queue overflow — drop record rather than block the calling thread
            pass


# ============================================================================
# Behavior Event Handler  (specialist dedicated log)
# ============================================================================

class BehaviorEventHandler(SafeRotatingFileHandler):
    """
    Specialist rotating file handler that captures only behavior detection
    events (WARNING level and above from the behavior analysis logger).

    Writes to ``logs/behavior_events.log`` with a compact JSON format so
    the events can be ingested by external alerting / analytics pipelines.

    Parameters
    ----------
    filename     : path to the behavior events log.
    max_bytes    : rotate threshold.  Default 5 MiB.
    backup_count : rotated files to keep.  Default 10.
    """

    _BEHAVIOR_LOGGER_PREFIX = "crowd_analysis.behavior"

    def __init__(
        self,
        filename:     str | Path = "logs/behavior_events.log",
        max_bytes:    int        = 5 * 1024 * 1024,
        backup_count: int        = 10,
    ) -> None:
        super().__init__(
            filename     = filename,
            max_bytes    = max_bytes,
            backup_count = backup_count,
        )
        self.setLevel(logging.WARNING)

    def emit(self, record: logging.LogRecord) -> None:
        """Only emit records from the behavior analysis subsystem."""
        if record.name.startswith(self._BEHAVIOR_LOGGER_PREFIX):
            super().emit(record)