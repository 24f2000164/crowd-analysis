"""
tests/integration/test_pipeline_flow.py
=========================================
Integration tests for the full video processing pipeline.

All tests use:
  - A synthetic frame generator (no real camera / RTSP source)
  - Mocked detector, tracker, and analyzer
  - A real FrameBuffer and FrameConsumer to verify the data flow

No GPU is required.  Tests run in < 5 seconds on any CI machine.
"""

from __future__ import annotations

import asyncio
import time
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.video.frame_buffer import FrameBuffer, BufferedFrame
from core.pipeline.frame_consumer import FrameConsumer
from core.pipeline.frame_producer import FrameProducer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _fill_buffer(buf: FrameBuffer, n: int = 5) -> None:
    for i in range(n):
        buf.put(_blank_frame(), frame_index=i)


# ---------------------------------------------------------------------------
# FrameBuffer
# ---------------------------------------------------------------------------

class TestFrameBuffer:

    def test_put_and_get(self):
        buf = FrameBuffer(maxsize=4)
        buf.put(_blank_frame(), 0)
        bf = buf.get(timeout=0.1)
        assert bf is not None
        assert bf.frame_index == 0

    def test_overflow_drops_oldest(self):
        buf = FrameBuffer(maxsize=2)
        buf.put(_blank_frame(), frame_index=0)
        buf.put(_blank_frame(), frame_index=1)
        # This push should evict frame 0
        buf.put(_blank_frame(), frame_index=2)
        bf = buf.get(timeout=0.05)
        # Oldest retained should now be frame 1
        assert bf.frame_index in (1, 2)

    def test_get_returns_none_on_timeout(self):
        buf = FrameBuffer(maxsize=4)
        bf = buf.get(timeout=0.05)
        assert bf is None

    def test_drop_rate_increases_on_overflow(self):
        buf = FrameBuffer(maxsize=1)
        buf.put(_blank_frame(), 0)
        buf.put(_blank_frame(), 1)   # should drop 0
        assert buf.total_dropped >= 1

    def test_drain_empties_buffer(self):
        buf = FrameBuffer(maxsize=8)
        _fill_buffer(buf, 5)
        buf.drain()
        assert buf.is_empty

    def test_stats_keys(self):
        buf = FrameBuffer(maxsize=4)
        s = buf.stats()
        for key in ("qsize", "maxsize", "total_captured", "total_dropped"):
            assert key in s

    def test_put_returns_true_when_not_full(self):
        buf = FrameBuffer(maxsize=8)
        result = buf.put(_blank_frame(), 0)
        assert result is True


# ---------------------------------------------------------------------------
# FrameProducer
# ---------------------------------------------------------------------------

class TestFrameProducer:

    @pytest.mark.asyncio
    async def test_yields_frames(self):
        buf = FrameBuffer(maxsize=8)
        _fill_buffer(buf, 3)

        producer = FrameProducer(buf, poll_timeout_s=0.02, max_empty_wait_s=0.1)
        stop     = asyncio.Event()
        frames   = []

        async for bf in producer.frames(stop_event=stop):
            if bf is not None:
                frames.append(bf)
            if len(frames) >= 3:
                stop.set()

        assert len(frames) == 3

    @pytest.mark.asyncio
    async def test_yields_none_when_empty_too_long(self):
        buf      = FrameBuffer(maxsize=4)
        producer = FrameProducer(buf, poll_timeout_s=0.02, max_empty_wait_s=0.04)
        stop     = asyncio.Event()
        got_none = False

        async for bf in producer.frames(stop_event=stop):
            if bf is None:
                got_none = True
                stop.set()

        assert got_none

    @pytest.mark.asyncio
    async def test_stop_event_exits_generator(self):
        buf  = FrameBuffer(maxsize=4)
        stop = asyncio.Event()
        stop.set()   # set immediately

        producer = FrameProducer(buf, poll_timeout_s=0.01)
        count    = 0
        async for _ in producer.frames(stop_event=stop):
            count += 1

        assert count == 0


# ---------------------------------------------------------------------------
# FrameConsumer
# ---------------------------------------------------------------------------

class TestFrameConsumer:

    def _make_pipeline_frame(self, idx: int = 0):
        """Build a minimal mock PipelineFrame."""
        pf = MagicMock()
        pf.annotated_frame = _blank_frame()
        pf.frame_index     = idx
        pf.to_meta_dict.return_value = {"frame_index": idx, "behavior": {"behavior": "normal"}}
        return pf

    @pytest.mark.asyncio
    async def test_pushes_to_queue_sink(self):
        consumer = FrameConsumer(jpeg_quality=50)
        q        = asyncio.Queue(maxsize=4)
        consumer.add_queue_sink(q)

        pf = self._make_pipeline_frame(0)
        await consumer.consume(pf)

        assert not q.empty()
        jpeg, meta = q.get_nowait()
        assert isinstance(jpeg, bytes)
        assert len(jpeg) > 0
        assert meta["frame_index"] == 0

    @pytest.mark.asyncio
    async def test_callback_sink_called(self):
        received = []

        async def sink(jpeg: bytes, meta: dict) -> None:
            received.append((jpeg, meta))

        consumer = FrameConsumer(jpeg_quality=50)
        consumer.add_callback_sink(sink)

        pf = self._make_pipeline_frame(5)
        await consumer.consume(pf)

        assert len(received) == 1
        assert received[0][1]["frame_index"] == 5

    @pytest.mark.asyncio
    async def test_drops_frame_when_queue_full_and_drop_on_slow(self):
        consumer = FrameConsumer(jpeg_quality=50, drop_on_slow=True)
        q        = asyncio.Queue(maxsize=1)
        q.put_nowait((b"existing", {}))   # fill queue
        consumer.add_queue_sink(q)

        pf = self._make_pipeline_frame(1)
        await consumer.consume(pf)

        assert consumer.total_dropped == 1

    @pytest.mark.asyncio
    async def test_multiple_frames_dispatched(self):
        consumer = FrameConsumer(jpeg_quality=50)
        q        = asyncio.Queue(maxsize=10)
        consumer.add_queue_sink(q)

        for i in range(5):
            await consumer.consume(self._make_pipeline_frame(i))

        assert consumer.total_dispatched == 5
        assert q.qsize() == 5

    def test_stats_keys(self):
        consumer = FrameConsumer()
        s = consumer.stats()
        for key in ("total_dispatched", "total_dropped",
                    "queue_sinks", "callback_sinks"):
            assert key in s


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------

class TestPipelineConfig:

    def test_default_construction(self):
        from core.pipeline.video_pipeline import PipelineConfig
        cfg = PipelineConfig()
        assert cfg.frame_skip          >= 1
        assert cfg.frame_buffer_size   >= 2
        assert cfg.max_consecutive_errors >= 1

    def test_custom_values(self):
        from core.pipeline.video_pipeline import PipelineConfig
        cfg = PipelineConfig(frame_skip=3, target_fps=30)
        assert cfg.frame_skip   == 3
        assert cfg.target_fps   == 30


# ---------------------------------------------------------------------------
# PipelineFrame
# ---------------------------------------------------------------------------

class TestPipelineFrame:

    def _make(self):
        from core.pipeline.video_pipeline import PipelineFrame
        from core.behavior.base_analyzer  import BehaviorLabel, BehaviorResult, FrameFeatures
        result = BehaviorResult(
            label        = BehaviorLabel.NORMAL,
            confidence   = 0.95,
            frame_index  = 0,
            track_labels = {},
            features     = FrameFeatures.empty(0),
            signals      = ["all_checks_passed"],
        )
        return PipelineFrame(
            annotated_frame = _blank_frame(),
            jpeg_bytes      = None,
            tracks          = [],
            behavior_result = result,
            frame_index     = 7,
            capture_index   = 7,
            processing_ms   = 12.5,
            fps             = 24.8,
        )

    def test_to_meta_dict_has_required_keys(self):
        pf = self._make()
        d  = pf.to_meta_dict()
        for key in ("frame_index", "fps", "track_count",
                    "processing_ms", "behavior"):
            assert key in d

    def test_encode_jpeg_returns_bytes(self):
        pf = self._make()
        raw = pf.encode_jpeg(quality=70)
        assert isinstance(raw, bytes)
        assert len(raw) > 0

    def test_encode_jpeg_cached(self):
        pf  = self._make()
        b1  = pf.encode_jpeg()
        b2  = pf.encode_jpeg()
        assert b1 is b2     # same object — cached


# ---------------------------------------------------------------------------
# End-to-end pipeline smoke test (mocked stages)
# ---------------------------------------------------------------------------

class TestPipelineEndToEnd:

    @pytest.mark.asyncio
    async def test_pipeline_yields_frames(self):
        """
        Smoke test: inject frames directly into the buffer, run one pipeline
        tick, and assert a PipelineFrame comes out.
        """
        from core.pipeline.video_pipeline import (
            VideoPipeline, PipelineConfig, PipelineFrame,
        )
        from core.behavior.base_analyzer import BehaviorLabel, BehaviorResult, FrameFeatures

        # Build a minimal pipeline with all stages mocked
        cfg = PipelineConfig(
            source="0",
            frame_skip=1,
            frame_buffer_size=8,
            max_consecutive_errors=3,
        )

        # Mock heavy components
        mock_detector = MagicMock()
        mock_detector.detect.return_value = []

        mock_tracker  = MagicMock()
        mock_tracker.update.return_value = []

        null_result = BehaviorResult(
            label=BehaviorLabel.NORMAL, confidence=1.0,
            frame_index=0, track_labels={},
            features=FrameFeatures.empty(0),
            signals=["test"],
        )
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = null_result
        mock_analyzer.set_frame_shape = MagicMock()

        mock_renderer = MagicMock()
        mock_renderer.render.return_value = _blank_frame()

        pipeline = VideoPipeline(
            detector=mock_detector,
            tracker=mock_tracker,
            analyzer=mock_analyzer,
            renderer=mock_renderer,
            config=cfg,
        )

        # Inject 3 frames into the buffer directly (bypass capture thread)
        for i in range(3):
            pipeline._buffer.put(_blank_frame(), frame_index=i)

        # Collect up to 3 output frames
        results: List[PipelineFrame] = []
        pipeline._running = True

        async for pf in pipeline._frame_producer():
            loop = asyncio.get_running_loop()
            out  = await pipeline._process_frame(pf, loop)
            results.append(out)
            pipeline._frame_index += 1
            if pipeline._frame_index >= 3:
                pipeline._running = False
                break

        assert len(results) == 3
        assert all(isinstance(r, PipelineFrame) for r in results)
        assert all(isinstance(r.annotated_frame, np.ndarray) for r in results)
