"""
src/utils/__init__.py
======================
Public API for the ``src.utils`` logging package.

Import the most commonly used names directly::

    from src.utils import setup_logging, get_logger, get_stream_logger
    from src.utils import LoggingConfig, LogContext
    from src.utils import StreamContextAdapter, PipelineStageAdapter
"""

from src.utils.logger import (
    LoggingConfig,
    get_logger,
    get_pipeline_logger,
    get_request_logger,
    get_stream_logger,
    reset_logging,
    setup_logging,
)
from src.utils.context import (
    LogContext,
    PipelineStageAdapter,
    RequestContextAdapter,
    StreamContextAdapter,
)
from src.utils.formatters import (
    ConsoleFormatter,
    JSONFormatter,
    PipelineFormatter,
)
from src.utils.handlers import (
    AsyncQueueHandler,
    BehaviorEventHandler,
    SafeRotatingFileHandler,
    SafeTimedRotatingFileHandler,
)

__all__ = [
    # Setup
    "setup_logging",
    "reset_logging",
    "LoggingConfig",
    # Logger factories
    "get_logger",
    "get_stream_logger",
    "get_pipeline_logger",
    "get_request_logger",
    # Context
    "LogContext",
    "StreamContextAdapter",
    "PipelineStageAdapter",
    "RequestContextAdapter",
    # Formatters
    "JSONFormatter",
    "ConsoleFormatter",
    "PipelineFormatter",
    # Handlers
    "SafeRotatingFileHandler",
    "SafeTimedRotatingFileHandler",
    "AsyncQueueHandler",
    "BehaviorEventHandler",
]