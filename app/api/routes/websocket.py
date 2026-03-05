"""
app/api/routes/websocket.py
============================
WebSocket endpoint for real-time annotated frame streaming.

Endpoint
--------
WS /ws/stream/{stream_id}

Protocol
--------
After the handshake the server continuously pushes two messages per frame:

    ┌──────────────────────────────────────────────────────────────────┐
    │  Message 1 (text)   — JSON metadata                              │
    │  {                                                               │
    │    "type":         "frame",                                      │
    │    "stream_id":    "a1b2c3d4",                                   │
    │    "frame_index":  142,                                          │
    │    "fps":          24.8,                                         │
    │    "track_count":  7,                                            │
    │    "behavior":     "normal",                                     │
    │    "confidence":   1.0,                                          │
    │    "processing_ms": 18.3,                                        │
    │    "tracks":       [{"id":1,"bbox":[...],"velocity":4.2}, ...],  │
    │    "signals":      ["all_checks_passed"]                         │
    │  }                                                               │
    │                                                                  │
    │  Message 2 (binary) — JPEG-encoded annotated frame bytes         │
    └──────────────────────────────────────────────────────────────────┘

On anomalous behavior a separate event message is sent:

    {
      "type":        "event",
      "stream_id":   "a1b2c3d4",
      "behavior":    "panic",
      "confidence":  0.87,
      "frame_index": 142,
      "signals":     ["panic_anomalous(0.45>=0.40)", ...],
      "track_labels": {"3": "panic", "7": "panic"}
    }

Client → Server messages (for keepalive / control):
    {"type": "ping"}   → server responds with {"type": "pong"}
    {"type": "stop"}   → server disconnects gracefully

Authentication
--------------
When API_KEY is configured the client must pass it as a query parameter:
    ws://<host>/ws/stream/{stream_id}?api_key=<key>

Connection errors
-----------------
If the stream_id does not exist the server sends an error JSON message
and closes with code 4004 before the pipeline push begins.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect, status
from starlette.websockets import WebSocketState

from app.api.dependencies import get_registry
from app.api.websocket.manager import ConnectionManager
from app.api.stream_registry    import StreamRegistry

logger = logging.getLogger("crowd_analysis.api.routes.websocket")

router = APIRouter(tags=["WebSocket"])


# ---------------------------------------------------------------------------
# WS /ws/stream/{stream_id}
# ---------------------------------------------------------------------------

@router.websocket("/ws/stream/{stream_id}")
async def ws_stream(
    websocket:  WebSocket,
    stream_id:  str,
     
    api_key:    Optional[str] = None,
) -> None:
    """
    WebSocket endpoint — streams annotated JPEG frames for ``stream_id``.

    Query Parameters
    ----------------
    api_key : str (optional) — required when API_KEY is configured server-side.
    """
    app_state = websocket.app.state
    manager:  ConnectionManager = app_state.ws_manager
    registry: StreamRegistry    = app_state.stream_registry
    settings                    = app_state.settings

    # ── Authentication ─────────────────────────────────────────────────
    expected_key = getattr(getattr(settings, "api", None), "api_key", None)
    if expected_key is not None:
        # api_key must arrive as query parameter (headers not available
        # after the WebSocket upgrade in most browsers)
        client_key = websocket.query_params.get("api_key")
        if client_key != expected_key:
            await websocket.accept()
            await manager.send_error(websocket, 4003, "Invalid or missing api_key.")
            await websocket.close(code=4003)
            logger.warning(
                "WS auth failure — stream=%s  client=%s",
                stream_id, websocket.client,
            )
            return

    # ── Stream existence check ─────────────────────────────────────────
    if not registry.exists(stream_id):
        await websocket.accept()
        await manager.send_error(websocket, 4004, f"Stream '{stream_id}' not found.")
        await websocket.close(code=4004)
        logger.warning("WS connect for unknown stream_id=%s", stream_id)
        return

    # ── Register client ────────────────────────────────────────────────
    await manager.connect(websocket, stream_id)
    logger.info(
        "WS client connected — stream=%s  client=%s  total=%d",
        stream_id, websocket.client, manager.client_count(stream_id),
    )

    # Send a welcome status message
    await manager.broadcast_event(stream_id, {
        "type":      "status",
        "stream_id": stream_id,
        "status":    "connected",
        "message":   f"Subscribed to stream '{stream_id}'. Frames incoming.",
    })

    # ── Client message loop ────────────────────────────────────────────
    # Keeps the connection alive and handles ping / stop commands from the client.
    try:
        while True:
            # Non-blocking receive with a timeout so the loop stays responsive
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(), timeout=30.0
                )
            except asyncio.TimeoutError:
                # No message from client — send a server-side keepalive ping
                if websocket.client_state == WebSocketState.CONNECTED:
                    try:
                        await websocket.send_text(
                            json.dumps({"type": "ping", "stream_id": stream_id})
                        )
                    except Exception:
                        break
                continue

            # Parse client command
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await manager.send_error(websocket, 4000, "Invalid JSON message.")
                continue

            msg_type = msg.get("type", "")

            if msg_type == "ping":
                await websocket.send_text(
                    json.dumps({"type": "pong", "stream_id": stream_id})
                )

            elif msg_type == "stop":
                logger.info(
                    "WS client requested stop — stream=%s", stream_id
                )
                break

            else:
                logger.debug(
                    "Unknown WS message type=%r from stream=%s", msg_type, stream_id
                )

    except WebSocketDisconnect:
        logger.info("WS client disconnected — stream=%s", stream_id)

    except Exception as exc:
        logger.error(
            "Unexpected WS error — stream=%s: %s", stream_id, exc, exc_info=True
        )

    finally:
        await manager.disconnect(websocket, stream_id)
        logger.info(
            "WS cleanup done — stream=%s  remaining_clients=%d",
            stream_id, manager.client_count(stream_id),
        )