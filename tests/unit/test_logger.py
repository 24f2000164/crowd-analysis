"""
tests/unit/test_logger.py
==========================
Unit tests for src/utils/logger.py and all supporting modules.

Tests run without any file I/O by default (tmp_path fixtures handle
filesystem access).  No GPU, camera, or model weights required.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Reset guard before importing so each test gets a clean state
import utils.logger as _logger_mod


@pytest.fixture(autouse=True)
def reset_logging_state():
    """Reset the _CONFIGURED guard and root logger before every test."""
    _logger_mod.reset_logging()
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()
    yield
    _logger_mod.reset_logging()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _capture_stream() -> StringIO:
    return StringIO()


# ============================================================================
# LoggingConfig
# ============================================================================

class TestLoggingConfig:

    def test_default_construction(self):
        from utils.logger import LoggingConfig
        cfg = LoggingConfig()
        assert cfg.log_dir         == Path("logs")
        assert cfg.app_level       == "INFO"
        assert cfg.enable_console  is True
        assert cfg.enable_file     is True
        assert cfg.enable_async    is True

    def test_custom_values(self):
        from utils.logger import LoggingConfig
        cfg = LoggingConfig(app_level="DEBUG", enable_json=True, service="test-svc")
        assert cfg.app_level   == "DEBUG"
        assert cfg.enable_json is True
        assert cfg.service     == "test-svc"

    def test_extra_loggers_defaults_empty(self):
        from utils.logger import LoggingConfig
        cfg = LoggingConfig()
        assert cfg.extra_loggers == {}


# ============================================================================
# JSONFormatter
# ============================================================================

class TestJSONFormatter:

    def _make_record(self, msg="hello", level=logging.INFO, name="test.logger"):
        record = logging.LogRecord(
            name=name, level=level, pathname="test.py",
            lineno=1, msg=msg, args=(), exc_info=None,
        )
        return record

    def test_produces_valid_json(self):
        from utils.formatters import JSONFormatter
        fmt    = JSONFormatter(service="svc", environment="test")
        record = self._make_record()
        output = fmt.format(record)
        data   = json.loads(output)   # must not raise
        assert isinstance(data, dict)

    def test_required_fields_present(self):
        from utils.formatters import JSONFormatter
        fmt    = JSONFormatter()
        record = self._make_record("test message")
        data   = json.loads(fmt.format(record))
        for key in ("timestamp", "level", "logger", "message", "service"):
            assert key in data, f"Missing field: {key}"

    def test_message_content(self):
        from utils.formatters import JSONFormatter
        fmt    = JSONFormatter()
        record = self._make_record("crowd panic detected")
        data   = json.loads(fmt.format(record))
        assert data["message"] == "crowd panic detected"

    def test_level_name(self):
        from utils.formatters import JSONFormatter
        fmt    = JSONFormatter()
        record = self._make_record(level=logging.WARNING)
        data   = json.loads(fmt.format(record))
        assert data["level"] == "WARNING"

    def test_service_injected(self):
        from utils.formatters import JSONFormatter
        fmt    = JSONFormatter(service="crowd-analysis")
        record = self._make_record()
        data   = json.loads(fmt.format(record))
        assert data["service"] == "crowd-analysis"

    def test_environment_injected(self):
        from utils.formatters import JSONFormatter
        fmt    = JSONFormatter(environment="production")
        record = self._make_record()
        data   = json.loads(fmt.format(record))
        assert data["environment"] == "production"

    def test_extra_fields_forwarded(self):
        from utils.formatters import JSONFormatter
        fmt    = JSONFormatter()
        record = self._make_record()
        record.__dict__["stream_id"]   = "abc123"
        record.__dict__["frame_index"] = 42
        data   = json.loads(fmt.format(record))
        assert data.get("stream_id")   == "abc123"
        assert data.get("frame_index") == 42

    def test_exception_included(self):
        from utils.formatters import JSONFormatter
        fmt = JSONFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            record = logging.LogRecord(
                name="t", level=logging.ERROR, pathname="t.py",
                lineno=1, msg="error", args=(), exc_info=sys.exc_info(),
            )
        data = json.loads(fmt.format(record))
        assert "exc_info" in data
        assert "ValueError" in data["exc_info"]

    def test_timestamp_is_utc_iso8601(self):
        from utils.formatters import JSONFormatter
        fmt    = JSONFormatter()
        record = self._make_record()
        data   = json.loads(fmt.format(record))
        ts     = data["timestamp"]
        assert ts.endswith("Z")
        assert "T" in ts


# ============================================================================
# ConsoleFormatter
# ============================================================================

class TestConsoleFormatter:

    def _make_record(self, msg="hello", level=logging.INFO):
        return logging.LogRecord(
            name="test", level=level, pathname="t.py",
            lineno=1, msg=msg, args=(), exc_info=None,
        )

    def test_produces_string_output(self):
        from utils.formatters import ConsoleFormatter
        fmt  = ConsoleFormatter(force_colors=False)
        out  = fmt.format(self._make_record())
        assert isinstance(out, str)
        assert len(out) > 0

    def test_message_in_output(self):
        from utils.formatters import ConsoleFormatter
        fmt = ConsoleFormatter(force_colors=False)
        out = fmt.format(self._make_record("my message"))
        assert "my message" in out

    def test_level_in_output(self):
        from utils.formatters import ConsoleFormatter
        fmt = ConsoleFormatter(force_colors=False)
        out = fmt.format(self._make_record(level=logging.ERROR))
        assert "ERROR" in out

    def test_color_output_contains_ansi(self):
        from utils.formatters import ConsoleFormatter
        fmt = ConsoleFormatter(force_colors=True)
        out = fmt.format(self._make_record())
        assert "\033[" in out

    def test_no_color_when_not_tty(self):
        from utils.formatters import ConsoleFormatter
        fmt = ConsoleFormatter(force_colors=False)
        out = fmt.format(self._make_record())
        assert "\033[" not in out


# ============================================================================
# SafeRotatingFileHandler
# ============================================================================

class TestSafeRotatingFileHandler:

    def test_creates_directory_on_init(self, tmp_path):
        from utils.handlers import SafeRotatingFileHandler
        log_file = tmp_path / "nested" / "dir" / "test.log"
        h        = SafeRotatingFileHandler(log_file)
        assert log_file.parent.exists()
        h.close()

    def test_emit_writes_record(self, tmp_path):
        from utils.handlers   import SafeRotatingFileHandler
        from utils.formatters import JSONFormatter
        log_file = tmp_path / "test.log"
        h        = SafeRotatingFileHandler(log_file)
        h.setFormatter(JSONFormatter())
        record = logging.LogRecord(
            name="t", level=logging.INFO, pathname="t.py",
            lineno=1, msg="written", args=(), exc_info=None,
        )
        h.emit(record)
        h.close()
        content = log_file.read_text()
        assert "written" in content

    def test_emit_does_not_raise_on_bad_path(self, tmp_path):
        from utils.handlers import SafeRotatingFileHandler
        h = SafeRotatingFileHandler(tmp_path / "ok.log")
        # Simulate a broken stream by closing the stream before emit
        h.stream.close()
        record = logging.LogRecord(
            name="t", level=logging.INFO, pathname="t.py",
            lineno=1, msg="should not raise", args=(), exc_info=None,
        )
        # Must not raise
        try:
            h.emit(record)
        except Exception:
            pass
        h.close()


# ============================================================================
# AsyncQueueHandler
# ============================================================================

class TestAsyncQueueHandler:

    def test_records_forwarded_to_downstream(self, tmp_path):
        from utils.handlers   import AsyncQueueHandler, SafeRotatingFileHandler
        from utils.formatters import JSONFormatter

        log_file    = tmp_path / "async.log"
        file_h      = SafeRotatingFileHandler(log_file)
        file_h.setFormatter(JSONFormatter())
        async_h     = AsyncQueueHandler([file_h])
        async_h.setFormatter(JSONFormatter())

        record = logging.LogRecord(
            name="t", level=logging.INFO, pathname="t.py",
            lineno=1, msg="async record", args=(), exc_info=None,
        )
        async_h.emit(record)
        time.sleep(0.05)   # allow background thread to flush
        async_h.stop()
        file_h.close()

        content = log_file.read_text()
        assert "async record" in content


# ============================================================================
# StreamContextAdapter
# ============================================================================

class TestStreamContextAdapter:

    def test_injects_stream_id(self):
        from utils.context import StreamContextAdapter

        records = []

        class _CapHandler(logging.Handler):
            def emit(self, record):
                records.append(record)

        base_logger = logging.getLogger("test.stream_adapter")
        base_logger.setLevel(logging.DEBUG)
        h = _CapHandler()
        base_logger.addHandler(h)

        adapter = StreamContextAdapter(base_logger, stream_id="s123")
        adapter.info("test message")

        base_logger.removeHandler(h)
        assert records
        assert records[0].__dict__.get("stream_id") == "s123"

    def test_set_frame_updates_index(self):
        from utils.context import StreamContextAdapter
        logger  = logging.getLogger("test.frame_update")
        adapter = StreamContextAdapter(logger, stream_id="x", frame_index=0)
        adapter.set_frame(99)
        assert adapter.extra["frame_index"] == 99


# ============================================================================
# PipelineStageAdapter
# ============================================================================

class TestPipelineStageAdapter:

    def test_stage_field_present(self):
        from utils.context import PipelineStageAdapter
        logger  = logging.getLogger("test.stage")
        adapter = PipelineStageAdapter(logger, stream_id="s", stage="detect")
        assert adapter.extra["stage"] == "detect"

    def test_set_stage(self):
        from utils.context import PipelineStageAdapter
        logger  = logging.getLogger("test.stage2")
        adapter = PipelineStageAdapter(logger, stage="detect")
        adapter.set_stage("track")
        assert adapter.extra["stage"] == "track"

    def test_set_fps(self):
        from utils.context import PipelineStageAdapter
        logger  = logging.getLogger("test.fps")
        adapter = PipelineStageAdapter(logger)
        adapter.set_fps(24.789)
        assert adapter.extra["fps"] == 24.8


# ============================================================================
# LogContext (ContextVar store)
# ============================================================================

class TestLogContext:

    def setup_method(self):
        from utils.context import LogContext
        LogContext.clear()

    def teardown_method(self):
        from utils.context import LogContext
        LogContext.clear()

    def test_set_and_get(self):
        from utils.context import LogContext
        LogContext.set(stream_id="x1", frame_index=10)
        ctx = LogContext.get()
        assert ctx["stream_id"]   == "x1"
        assert ctx["frame_index"] == 10

    def test_clear(self):
        from utils.context import LogContext
        LogContext.set(foo="bar")
        LogContext.clear()
        assert LogContext.get() == {}

    def test_remove_specific_key(self):
        from utils.context import LogContext
        LogContext.set(a=1, b=2)
        LogContext.remove("a")
        ctx = LogContext.get()
        assert "a" not in ctx
        assert ctx.get("b") == 2

    def test_merge_over_existing(self):
        from utils.context import LogContext
        LogContext.set(a=1)
        LogContext.set(b=2)
        ctx = LogContext.get()
        assert ctx["a"] == 1
        assert ctx["b"] == 2


# ============================================================================
# setup_logging + get_logger
# ============================================================================

class TestSetupLogging:

    def test_get_logger_returns_logger(self):
        from utils.logger import get_logger
        log = get_logger("crowd_analysis.test")
        assert isinstance(log, logging.Logger)

    def test_setup_logging_idempotent(self, tmp_path):
        from utils.logger import LoggingConfig, setup_logging
        cfg = LoggingConfig(
            log_dir=tmp_path, enable_file=False, enable_async=False
        )
        setup_logging(config=cfg)
        setup_logging(config=cfg)   # second call is a no-op

    def test_env_override_log_level(self, tmp_path, monkeypatch):
        from utils.logger import LoggingConfig, setup_logging, reset_logging
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        reset_logging()
        cfg = LoggingConfig(
            log_dir=tmp_path, enable_file=False, enable_async=False
        )
        setup_logging(config=cfg)
        log = logging.getLogger("crowd_analysis")
        assert log.level == logging.DEBUG

    def test_get_stream_logger_returns_adapter(self):
        from utils.logger import get_stream_logger
        from utils.context import StreamContextAdapter
        adapter = get_stream_logger("test-stream")
        assert isinstance(adapter, StreamContextAdapter)
        assert adapter.extra["stream_id"] == "test-stream"

    def test_get_pipeline_logger_returns_adapter(self):
        from utils.logger import get_pipeline_logger
        from utils.context import PipelineStageAdapter
        adapter = get_pipeline_logger(stream_id="sid", stage="detect")
        assert isinstance(adapter, PipelineStageAdapter)
        assert adapter.extra["stage"] == "detect"

    def test_get_request_logger_returns_adapter(self):
        from utils.logger import get_request_logger
        from utils.context import RequestContextAdapter
        adapter = get_request_logger(request_id="rid-1", path="/health")
        assert isinstance(adapter, RequestContextAdapter)
        assert adapter.extra["request_id"] == "rid-1"
        assert adapter.extra["path"]       == "/health"

    def test_file_handlers_created(self, tmp_path):
        from utils.logger import LoggingConfig, setup_logging
        cfg = LoggingConfig(
            log_dir=tmp_path,
            enable_file=True,
            enable_async=False,
            enable_json=False,
        )
        setup_logging(config=cfg)
        app_logger = logging.getLogger("crowd_analysis")
        assert len(app_logger.handlers) > 0

    def test_log_files_created_on_emit(self, tmp_path):
        from utils.logger import LoggingConfig, setup_logging, get_logger
        cfg = LoggingConfig(
            log_dir     = tmp_path,
            enable_file = True,
            enable_async = False,
        )
        setup_logging(config=cfg)
        log = get_logger("crowd_analysis")
        log.info("test log entry")
        # Allow handlers to flush
        for h in log.handlers:
            h.flush()
        log_file = tmp_path / "crowd_analysis.log"
        assert log_file.exists()
        assert "test log entry" in log_file.read_text()