"""
src/utils/formatters.py
========================
Custom log formatters for the Crowd Analysis System.

Formatters
----------
JSONFormatter        — structured JSON output for log aggregators
                       (Loki, CloudWatch, Datadog, ELK).  Each log record
                       becomes a single JSON object on one line.

ConsoleFormatter     — colour-coded, human-readable output for terminals.
                       Uses ANSI escape codes with graceful fallback on
                       Windows / non-TTY streams.

PipelineFormatter    — JSON with extra pipeline-specific fields
                       (stream_id, frame_index, stage) injected from
                       LoggerAdapter extras.

Usage
-----
    from src.utils.formatters import JSONFormatter, ConsoleFormatter

    handler.setFormatter(JSONFormatter(service="crowd-analysis"))
    handler.setFormatter(ConsoleFormatter())
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional


# ============================================================================
# JSON Formatter
# ============================================================================

class JSONFormatter(logging.Formatter):
    """
    Emit each log record as a single-line JSON object.

    Standard fields in every record
    --------------------------------
    timestamp   : ISO-8601 UTC timestamp with milliseconds.
    level       : uppercased level name  (DEBUG, INFO, …).
    logger      : logger name.
    message     : formatted log message.
    module      : source file name without extension.
    function    : calling function name.
    line        : source line number.
    process     : process ID.
    thread      : thread name.
    service     : configurable service label (injected at formatter creation).

    Optional fields
    ---------------
    exc_info    : formatted traceback string (only present when an exception
                  is attached to the record).
    stack_info  : stack trace string (only when stack_info=True).
    extra.*     : any key-value pairs passed via LoggerAdapter or ``extra=``.

    Parameters
    ----------
    service        : service name tag injected into every record.
    environment    : environment tag (development / staging / production).
    extra_fields   : additional static key-value pairs merged into every record.
    """

    def __init__(
        self,
        service:      str              = "crowd-analysis",
        environment:  str              = "development",
        extra_fields: Dict[str, Any]   = None,
    ) -> None:
        super().__init__()
        self._service     = service
        self._environment = environment
        self._extra       = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        # Core fields
        payload: Dict[str, Any] = {
            "timestamp":   self._utc_timestamp(record.created),
            "level":       record.levelname,
            "logger":      record.name,
            "message":     record.getMessage(),
            "module":      record.module,
            "function":    record.funcName,
            "line":        record.lineno,
            "process":     record.process,
            "thread":      record.threadName,
            "service":     self._service,
            "environment": self._environment,
        }

        # Static extra fields (injected at formatter construction)
        payload.update(self._extra)

        # Dynamic extra fields (from LoggerAdapter / logger.info(..., extra={}) )
        for key, value in record.__dict__.items():
            if key not in _STDLIB_ATTRS and not key.startswith("_"):
                payload[key] = value

        # Exception
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.exc_text:
            payload["exc_text"] = record.exc_text

        # Stack info
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(payload, default=_json_default, ensure_ascii=False)

    @staticmethod
    def _utc_timestamp(created: float) -> str:
        dt = datetime.fromtimestamp(created, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


# ============================================================================
# Console Formatter  (colour-coded, human-readable)
# ============================================================================

# ANSI colour codes
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"

_LEVEL_COLOURS = {
    "DEBUG":    "\033[36m",    # cyan
    "INFO":     "\033[32m",    # green
    "WARNING":  "\033[33m",    # yellow
    "ERROR":    "\033[31m",    # red
    "CRITICAL": "\033[35m",    # magenta
}

# Logger-name colour map (deterministic hash → one of 6 colours)
_NAME_COLOURS = [
    "\033[94m",   # bright blue
    "\033[96m",   # bright cyan
    "\033[92m",   # bright green
    "\033[95m",   # bright magenta
    "\033[93m",   # bright yellow
    "\033[91m",   # bright red
]


class ConsoleFormatter(logging.Formatter):
    """
    Colour-coded, human-readable log formatter for terminal output.

    Format
    ------
    HH:MM:SS.mmm [LEVEL   ] logger.name:lineno — message

    Colours are applied only when the stream is a real TTY (i.e. not
    redirected to a file) or when ``force_colors=True``.

    Parameters
    ----------
    force_colors : override TTY detection and always emit ANSI codes.
    show_thread  : append thread name when it is not ``MainThread``.
    """

    _FMT_DATE  = "%H:%M:%S"
    _FMT_WIDTH = 9   # level column width: "CRITICAL " = 9

    def __init__(
        self,
        force_colors: bool = False,
        show_thread:  bool = True,
    ) -> None:
        super().__init__()
        self._use_colors = force_colors or (
            hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
        )
        self._show_thread = show_thread

    def format(self, record: logging.LogRecord) -> str:
        # Timestamp
        ts = self.formatTime(record, self._FMT_DATE)
        ms = int(record.msecs)
        ts = f"{ts}.{ms:03d}"

        # Level label
        level = f"{record.levelname:<8}"

        # Logger name (last two components for brevity)
        parts = record.name.split(".")
        name  = ".".join(parts[-2:]) if len(parts) > 1 else record.name

        # Thread (only non-main threads)
        thread_str = ""
        if self._show_thread and record.threadName not in ("MainThread", ""):
            thread_str = f" [{record.threadName}]"

        # Base line
        line = f"{ts} {level} {name}:{record.lineno}{thread_str} — {record.getMessage()}"

        # Apply colours
        if self._use_colors:
            lvl_color  = _LEVEL_COLOURS.get(record.levelname, "")
            name_color = _NAME_COLOURS[hash(record.name) % len(_NAME_COLOURS)]
            ts_str     = f"{_DIM}{ts}{_RESET}"
            lvl_str    = f"{_BOLD}{lvl_color}{level}{_RESET}"
            name_str   = f"{name_color}{name}:{record.lineno}{_RESET}"
            thread_col = f"{_DIM}{thread_str}{_RESET}" if thread_str else ""
            msg_color  = lvl_color if record.levelno >= logging.WARNING else ""
            msg_str    = f"{msg_color}{record.getMessage()}{_RESET if msg_color else ''}"
            line       = f"{ts_str} {lvl_str} {name_str}{thread_col} — {msg_str}"

        # Exception
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            exc = record.exc_text
            if self._use_colors:
                exc = f"{_LEVEL_COLOURS['ERROR']}{exc}{_RESET}"
            line = f"{line}\n{exc}"

        if record.stack_info:
            line = f"{line}\n{self.formatStack(record.stack_info)}"

        return line


# ============================================================================
# Pipeline Formatter  (JSON + pipeline-specific fields)
# ============================================================================

class PipelineFormatter(JSONFormatter):
    """
    JSON formatter extended with pipeline-specific context fields.

    Extra fields injected when present in the log record
    ------------------------------------------------------
    stream_id   : active stream identifier.
    frame_index : pipeline frame counter.
    stage       : pipeline stage name (detect / track / behavior / annotate).
    fps         : rolling pipeline FPS at the time of the log call.
    """

    _PIPELINE_KEYS = {"stream_id", "frame_index", "stage", "fps"}

    def format(self, record: logging.LogRecord) -> str:
        # Let the parent build the full JSON dict via format()
        raw = super().format(record)
        data = json.loads(raw)

        # Pull pipeline-specific extras to the top level for easy querying
        for key in self._PIPELINE_KEYS:
            if key in data:
                data[f"pipeline_{key}"] = data.pop(key)

        return json.dumps(data, default=_json_default, ensure_ascii=False)


# ============================================================================
# Helpers
# ============================================================================

def _json_default(obj: Any) -> Any:
    """Fallback JSON serialiser for non-standard types."""
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


# Stdlib LogRecord attributes that should NOT be forwarded as extra fields
_STDLIB_ATTRS = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "message", "module",
    "msecs", "msg", "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "thread", "threadName",
    "taskName",   # Python 3.12+
})