"""
core/pipeline/frame_producer.py
=================================
Async frame generator that bridges the capture thread and the pipeline loop.

Responsibility
--------------
Poll the ``FrameBuffer`` in a thread-pool executor and yield
``BufferedFrame`` objects to the async pipeline coroutine without blocking
the event loop.

This module is split from ``video_pipeline.py`` so it can be tested in
isolation and reused by alternative pipeline implementations (e.g. a batch
processor that reads from a file instead of a live stream).

Usage
-----
    from core.pipeline.frame_producer import FrameProducer

    producer = FrameProducer(buffer, max_empty_wait_s=2.0)

    async for buffered_frame in producer.frames():
        # buffered_frame.frame  — BGR ndarray
        # buffered_frame.frame_index
        # buffered_frame.timestamp
        pass
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Optional

from core.video.frame_buffer import FrameBuffer, BufferedFrame

logger = logging.getLogger("crowd_analysis.pipeline.producer")


class FrameProducer:
    """
    Async generator that yields frames from a :class:`FrameBuffer`.

    Parameters
    ----------
    buffer          : shared ring buffer written by the capture thread.
    executor        : thread pool for blocking ``buffer.get()`` calls.
    poll_timeout_s  : seconds each ``buffer.get()`` call waits.
    max_empty_wait_s: cumulative seconds of empty polls before yielding
                      ``None`` to signal a possibly-dead source.
    """

    def __init__(
        self,
        buffer:           FrameBuffer,
        executor:         Optional[ThreadPoolExecutor] = None,
        poll_timeout_s:   float = 0.05,
        max_empty_wait_s: float = 2.0,
    ) -> None:
        self._buffer          = buffer
        self._executor        = executor
        self._poll_timeout    = poll_timeout_s
        self._max_empty_wait  = max_empty_wait_s

    async def frames(
        self,
        stop_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[Optional[BufferedFrame]]:
        """
        Async generator of ``BufferedFrame`` objects.

        Yields ``None`` after ``max_empty_wait_s`` of continuous empty polls
        so callers can check whether the capture source is still alive.

        Parameters
        ----------
        stop_event : optional asyncio.Event; when set the generator exits.
        """
        loop      = asyncio.get_running_loop()
        empty_s   = 0.0

        while True:
            # Honour stop signal
            if stop_event and stop_event.is_set():
                logger.debug("FrameProducer stop_event set — exiting.")
                break

            # Poll buffer without blocking the event loop
            buffered: Optional[BufferedFrame] = await loop.run_in_executor(
                self._executor,
                lambda: self._buffer.get(timeout=self._poll_timeout),
            )

            if buffered is not None:
                empty_s = 0.0
                yield buffered
            else:
                empty_s += self._poll_timeout
                if empty_s >= self._max_empty_wait:
                    # Signal caller that source may be dead
                    yield None
                    empty_s = 0.0
                    logger.debug(
                        "FrameProducer: buffer empty for %.1fs.", self._max_empty_wait
                    )