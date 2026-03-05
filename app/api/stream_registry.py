"""
app/api/stream_registry.py
============================
Registry that maps stream IDs to active pipeline instances.

Responsibility
--------------
One pipeline instance is created per /streams/start call.  This registry
owns the mapping stream_id → (pipeline, asyncio.Task, metadata).  It
handles the full lifecycle: create → start (asyncio Task) → stop → clean-up.

The registry is a singleton stored in ``app.state.stream_registry`` and
injected into route handlers via FastAPI ``Depends``.

Usage
-----
    from app.api.stream_registry import StreamRegistry

    registry = StreamRegistry(ws_manager)

    stream_id = await registry.start_stream(request, pipeline)
    await registry.stop_stream(stream_id)
    info = registry.get_stream_info(stream_id)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.schemas.stream import StreamStatus, StreamInfo
from app.api.websocket.manager import ConnectionManager
from core.pipeline.video_pipeline import VideoPipeline, PipelineFrame
from core.behavior.base_analyzer import BehaviorLabel

logger = logging.getLogger("crowd_analysis.api.stream_registry")


@dataclass
class _StreamRecord:
    """Internal bookkeeping for one active stream."""
    stream_id:     str
    source:        str
    pipeline:      VideoPipeline
    task:          Optional[asyncio.Task]
    status:        StreamStatus     = StreamStatus.IDLE
    started_at:    float            = field(default_factory=time.monotonic)
    last_frame_index: int           = 0
    last_fps:         float         = 0.0
    last_track_count: int           = 0
    last_behavior:    str           = BehaviorLabel.NORMAL.value
    last_confidence:  float         = 0.0
    error_message:    Optional[str] = None


class StreamRegistry:
    """
    Lifecycle manager for active pipeline instances.

    Parameters
    ----------
    ws_manager : ConnectionManager — shared WebSocket broadcast hub.
    """

    def __init__(self, ws_manager: ConnectionManager) -> None:
        self._manager  = ws_manager
        self._streams: Dict[str, _StreamRecord] = {}

    # ------------------------------------------------------------------
    # Stream management
    # ------------------------------------------------------------------

    async def start_stream(
        self,
        source:   str,
        pipeline: VideoPipeline,
    ) -> str:
        """
        Register a pipeline and start it as a background asyncio task.

        Parameters
        ----------
        source   : human-readable source string (RTSP URL, "0", path).
        pipeline : fully configured ``VideoPipeline`` instance.

        Returns
        -------
        str — unique stream_id.
        """
        stream_id = str(uuid.uuid4())[:8]   # short readable ID
        record    = _StreamRecord(
            stream_id=stream_id,
            source=source,
            pipeline=pipeline,
            task=None,
            status=StreamStatus.STARTING,
        )
        self._streams[stream_id] = record

        # Notify any early WebSocket subscribers
        await self._manager.send_status(stream_id, StreamStatus.STARTING, "Pipeline starting …")

        # Create the pipeline background task
        task = asyncio.create_task(
            self._run_pipeline(stream_id),
            name=f"pipeline-{stream_id}",
        )
        record.task    = task
        record.status  = StreamStatus.RUNNING

        logger.info("Stream %s started — source=%r", stream_id, source)
        return stream_id

    async def stop_stream(self, stream_id: str) -> None:
        """
        Gracefully stop a running stream.

        Parameters
        ----------
        stream_id : ID returned by ``start_stream``.

        Raises
        ------
        KeyError if stream_id is not found.
        """
        record = self._streams.get(stream_id)
        if record is None:
            raise KeyError(f"Stream {stream_id!r} not found.")

        record.status = StreamStatus.STOPPING
        await self._manager.send_status(stream_id, StreamStatus.STOPPING, "Pipeline stopping …")

        # Signal the pipeline coroutine to exit
        record.pipeline.stop()

        # Cancel the asyncio task with a short grace period
        if record.task and not record.task.done():
            record.task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.shield(record.task), timeout=5.0
                )
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        record.status = StreamStatus.STOPPED
        await self._manager.send_status(stream_id, StreamStatus.STOPPED, "Pipeline stopped.")
        logger.info("Stream %s stopped.", stream_id)

    async def stop_all(self) -> None:
        """Stop every active stream. Called on server shutdown."""
        for stream_id in list(self._streams):
            try:
                await self.stop_stream(stream_id)
            except Exception as exc:
                logger.warning("Error stopping stream %s: %s", stream_id, exc)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_stream_info(self, stream_id: str) -> Optional[StreamInfo]:
        """Return a ``StreamInfo`` snapshot, or None if not found."""
        record = self._streams.get(stream_id)
        if record is None:
            return None
        return self._record_to_info(record)

    def list_streams(self) -> List[StreamInfo]:
        """Return info for all registered streams."""
        return [self._record_to_info(r) for r in self._streams.values()]

    def exists(self, stream_id: str) -> bool:
        return stream_id in self._streams

    def is_running(self, stream_id: str) -> bool:
        record = self._streams.get(stream_id)
        return record is not None and record.status == StreamStatus.RUNNING

    # ------------------------------------------------------------------
    # Background pipeline coroutine
    # ------------------------------------------------------------------

    async def _run_pipeline(self, stream_id: str) -> None:
        """
        Consume the pipeline's async generator and broadcast each frame.

        Runs as an asyncio Task for the lifetime of the stream.
        """
        record = self._streams[stream_id]

        try:
            async for pipeline_frame in record.pipeline.run():
                # Update record stats
                self._update_record(record, pipeline_frame)

                # Encode JPEG
                jpeg = pipeline_frame.encode_jpeg()
                if not jpeg:
                    continue

                # Build metadata envelope
                meta = pipeline_frame.to_meta_dict()
                meta["stream_id"] = stream_id
                meta["type"]      = "frame"
                meta["tracks"]    = [t.to_dict() for t in pipeline_frame.tracks]
                meta["signals"]   = pipeline_frame.behavior_result.signals

                # Fan-out to all WebSocket subscribers
                await self._manager.broadcast_frame(stream_id, meta, jpeg)

                # Broadcast behavior events for anomalous labels
                br = pipeline_frame.behavior_result
                if br.label not in (
                    BehaviorLabel.NORMAL,
                    BehaviorLabel.INSUFFICIENT_DATA,
                ):
                    await self._manager.broadcast_event(stream_id, {
                        "type":         "event",
                        "stream_id":    stream_id,
                        "behavior":     br.label.value,
                        "confidence":   round(br.confidence, 4),
                        "frame_index":  br.frame_index,
                        "signals":      br.signals,
                        "track_labels": {str(k): v for k, v in br.track_labels.items()},
                    })

        except asyncio.CancelledError:
            logger.info("Pipeline task for stream %s cancelled.", stream_id)
        except Exception as exc:
            logger.exception("Pipeline error for stream %s: %s", stream_id, exc)
            record.status        = StreamStatus.ERROR
            record.error_message = str(exc)
            await self._manager.send_status(
                stream_id, StreamStatus.ERROR, f"Pipeline error: {exc}"
            )
        finally:
            if record.status not in (StreamStatus.STOPPED, StreamStatus.ERROR):
                record.status = StreamStatus.STOPPED

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _update_record(record: _StreamRecord, pf: PipelineFrame) -> None:
        record.last_frame_index = pf.frame_index
        record.last_fps         = pf.fps
        record.last_track_count = len(pf.tracks)
        record.last_behavior    = pf.behavior_result.label.value
        record.last_confidence  = pf.behavior_result.confidence

    def _record_to_info(self, record: _StreamRecord) -> StreamInfo:
        return StreamInfo(
            stream_id   = record.stream_id,
            source      = record.source,
            status      = record.status,
            frame_index = record.last_frame_index,
            fps         = round(record.last_fps, 1),
            track_count = record.last_track_count,
            behavior    = record.last_behavior,
            confidence  = round(record.last_confidence, 4),
            ws_clients  = self._manager.client_count(record.stream_id),
            uptime_s    = round(time.monotonic() - record.started_at, 1),
        )