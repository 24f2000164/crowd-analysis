"""
app/api/websocket/manager.py
==============================
WebSocket connection manager for multi-stream, multi-client real-time
frame broadcasting.

Architecture
------------
Clients subscribe to a specific ``stream_id``.  The manager maintains a
registry of ``stream_id → set[WebSocket]`` so that a single pipeline can
fan-out annotated JPEG frames to all subscribers of that stream without
any subscriber knowing about the others.

Message protocol
----------------
For each processed frame the pipeline calls ``broadcast_frame()``, which
sends two messages per connected client in order:

  1. Text frame   — JSON metadata envelope (WSFrameMessage).
  2. Binary frame — raw JPEG bytes.

For anomalous behavior events ``broadcast_event()`` sends a single JSON
text frame (WSEventMessage).

Error handling
--------------
Any ``WebSocketDisconnect`` or ``RuntimeError`` during a send silently
removes the offending client from the registry so one bad client never
blocks delivery to all others.

Thread safety
-------------
Subscriptions and broadcasts both run in the asyncio event loop so no
locking is required.  The manager is designed to be a singleton held
by the FastAPI app state.

Usage
-----
    from app.api.websocket.manager import ConnectionManager

    manager = ConnectionManager()

    @app.websocket("/stream/{stream_id}")
    async def ws_endpoint(ws: WebSocket, stream_id: str):
        await manager.connect(ws, stream_id)
        try:
            await ws.receive_text()   # keep alive until client disconnects
        finally:
            await manager.disconnect(ws, stream_id)
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from typing import Any, Dict, Optional, Set

from fastapi import WebSocket
from fastapi.websockets import WebSocketState
from starlette.websockets import WebSocketDisconnect

logger = logging.getLogger("crowd_analysis.api.websocket.manager")


class ConnectionManager:
    """
    Manages WebSocket connections grouped by stream ID.

    Parameters
    ----------
    max_queue_per_client : int
        Maximum JPEG frames queued per client.  When this limit is reached
        the oldest frame is dropped to prevent memory unboundedness with
        slow clients.
    """

    def __init__(self, max_queue_per_client: int = 32) -> None:
        # stream_id → set of connected WebSocket objects
        self._connections: Dict[str, Set[WebSocket]] = defaultdict(set)
        self._max_queue = max_queue_per_client
        # Per-client send queues (WebSocket → asyncio.Queue)
        self._queues: Dict[WebSocket, asyncio.Queue] = {}
        # Per-client sender tasks
        self._sender_tasks: Dict[WebSocket, asyncio.Task] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self, websocket: WebSocket, stream_id: str) -> None:
        """
        Accept a WebSocket connection and register it under ``stream_id``.

        Creates a dedicated send queue and background sender task for the
        client so broadcasts never block the pipeline loop.
        """
        await websocket.accept()
        self._connections[stream_id].add(websocket)

        q: asyncio.Queue = asyncio.Queue(maxsize=self._max_queue)
        self._queues[websocket] = q

        task = asyncio.create_task(
            self._sender_loop(websocket, q),
            name=f"ws-sender-{id(websocket)}",
        )
        self._sender_tasks[websocket] = task

        logger.info(
            "WebSocket client connected — stream=%s  total_clients=%d",
            stream_id,
            self.client_count(stream_id),
        )

    async def disconnect(self, websocket: WebSocket, stream_id: str) -> None:
        """
        Remove a client from the registry and cancel its sender task.
        """
        self._connections[stream_id].discard(websocket)

        # Cancel sender task
        task = self._sender_tasks.pop(websocket, None)
        if task and not task.done():
            task.cancel()

        # Drain and remove queue
        self._queues.pop(websocket, None)

        # Close socket if still open
        if websocket.client_state != WebSocketState.DISCONNECTED:
            try:
                await websocket.close()
            except Exception:
                pass

        # Clean up empty stream sets
        if not self._connections[stream_id]:
            del self._connections[stream_id]

        logger.info(
            "WebSocket client disconnected — stream=%s  remaining=%d",
            stream_id,
            self.client_count(stream_id),
        )

    # ------------------------------------------------------------------
    # Broadcast API
    # ------------------------------------------------------------------

    async def broadcast_frame(
        self,
        stream_id:  str,
        meta:       Dict[str, Any],
        jpeg_bytes: bytes,
    ) -> None:
        """
        Broadcast a processed frame to all clients of ``stream_id``.

        Enqueues ``(meta_json, jpeg_bytes)`` into each client's send queue.
        Drops the oldest entry when a queue is full (slow-client protection).

        Parameters
        ----------
        stream_id  : target stream group.
        meta       : dict that will be serialised to a JSON text message.
        jpeg_bytes : raw JPEG frame bytes sent as a binary message.
        """
        clients = set(self._connections.get(stream_id, set()))
        if not clients:
            return

        meta_json = json.dumps(meta)

        dead: Set[WebSocket] = set()
        for ws in clients:
            q = self._queues.get(ws)
            if q is None:
                dead.add(ws)
                continue
            if q.full():
                try:
                    q.get_nowait()   # evict oldest
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait((meta_json, jpeg_bytes))
            except asyncio.QueueFull:
                pass

        # Prune dead references
        for ws in dead:
            self._connections[stream_id].discard(ws)

    async def broadcast_event(
        self,
        stream_id: str,
        event:     Dict[str, Any],
    ) -> None:
        """
        Send a JSON event message to all clients of ``stream_id``.

        Used for anomaly alerts (panic, violence, etc.).
        """
        clients = set(self._connections.get(stream_id, set()))
        if not clients:
            return

        event_json = json.dumps(event)
        dead: Set[WebSocket] = set()

        for ws in clients:
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_text(event_json)
            except (WebSocketDisconnect, RuntimeError):
                dead.add(ws)
            except Exception as exc:
                logger.warning("Event broadcast error for stream=%s: %s", stream_id, exc)

        for ws in dead:
            self._connections[stream_id].discard(ws)

    async def send_status(
        self,
        stream_id: str,
        status:    str,
        message:   str = "",
    ) -> None:
        """Push a stream lifecycle status message to all subscribers."""
        await self.broadcast_event(
            stream_id,
            {"type": "status", "stream_id": stream_id,
             "status": status, "message": message},
        )

    async def send_error(
        self,
        websocket: WebSocket,
        code:      int,
        message:   str,
    ) -> None:
        """Send an error envelope to a single client."""
        try:
            await websocket.send_text(
                json.dumps({"type": "error", "code": code, "message": message})
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Background sender loop (one per client)
    # ------------------------------------------------------------------

    @staticmethod
    async def _sender_loop(
        websocket: WebSocket,
        queue:     asyncio.Queue,
    ) -> None:
        """
        Consume the per-client queue and forward messages over the wire.

        Running this as a dedicated task means the pipeline's broadcast call
        (which only enqueues) never awaits a slow network write.
        """
        try:
            while True:
                meta_json, jpeg_bytes = await queue.get()

                if websocket.client_state != WebSocketState.CONNECTED:
                    break

                try:
                    await websocket.send_text(meta_json)
                    await websocket.send_bytes(jpeg_bytes)
                except (WebSocketDisconnect, RuntimeError):
                    break
                except Exception as exc:
                    logger.debug("WebSocket send error: %s", exc)
                    break
        except asyncio.CancelledError:
            pass   # Normal shutdown

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def client_count(self, stream_id: str) -> int:
        """Number of active clients subscribed to a stream."""
        return len(self._connections.get(stream_id, set()))

    @property
    def total_clients(self) -> int:
        """Total connected clients across all streams."""
        return sum(len(s) for s in self._connections.values())

    @property
    def active_streams(self) -> list[str]:
        """Stream IDs that have at least one subscriber."""
        return [sid for sid, s in self._connections.items() if s]

    async def close_all(self) -> None:
        """Forcefully disconnect all clients. Call on server shutdown."""
        for stream_id, clients in list(self._connections.items()):
            for ws in list(clients):
                await self.disconnect(ws, stream_id)
        logger.info("All WebSocket connections closed.")