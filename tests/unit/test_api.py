"""
tests/unit/test_api.py
========================
Unit and integration tests for the FastAPI server.

Uses ``httpx.AsyncClient`` with FastAPI's ``app`` in ASGI transport mode
so no real HTTP port is needed.  All pipeline components are mocked so
tests run without GPU, camera, or model weights.
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from app.api.server import create_app
from app.api.websocket.manager import ConnectionManager
from app.api.stream_registry    import StreamRegistry
from app.schemas.stream          import StreamStatus


# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def test_app():
    """
    Create a test FastAPI application with mocked heavy dependencies.
    The app lifespan still runs so app.state is fully populated.
    """
    app = create_app()

    # Override app.state singletons before each test
    ws_manager = ConnectionManager(max_queue_per_client=4)
    registry   = StreamRegistry(ws_manager=ws_manager)

    class _FakeSettings:
        app_name    = "Test App"
        app_version = "0.0.1"
        environment = "test"
        debug       = True
        class api:
            api_key      = None
            cors_origins = ["*"]
            ws_max_queue = 4
        class model:
            yolo_weights = "models/yolov8n.pt"

    app.state.settings        = _FakeSettings()
    app.state.ws_manager      = ws_manager
    app.state.stream_registry = registry
    app.state.start_time      = 0.0

    return app


@pytest_asyncio.fixture
async def client(test_app) -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(
        transport=ASGITransport(app=test_app),
        base_url="http://test",
    ) as ac:
        yield ac


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:

    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        r = await client.get("/health")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_health_body_structure(self, client):
        r   = await client.get("/health")
        body = r.json()
        for key in ("status", "version", "environment", "uptime_s"):
            assert key in body

    @pytest.mark.asyncio
    async def test_health_status_ok(self, client):
        r = await client.get("/health")
        assert r.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_readiness_returns_200_or_503(self, client):
        r = await client.get("/readiness")
        assert r.status_code in (200, 503)

    @pytest.mark.asyncio
    async def test_readiness_body_has_components(self, client):
        r = await client.get("/readiness")
        body = r.json()
        assert "ready"      in body
        assert "components" in body
        assert "message"    in body

    @pytest.mark.asyncio
    async def test_version_endpoint(self, client):
        r = await client.get("/version")
        assert r.status_code == 200
        assert "version" in r.json()

    @pytest.mark.asyncio
    async def test_root_returns_service_info(self, client):
        r = await client.get("/")
        assert r.status_code == 200
        assert "service" in r.json()


# ---------------------------------------------------------------------------
# /streams
# ---------------------------------------------------------------------------

class TestStreamsEndpoints:

    @pytest.mark.asyncio
    async def test_list_streams_empty(self, client):
        r = await client.get("/streams")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 0
        assert body["streams"] == []

    @pytest.mark.asyncio
    async def test_get_nonexistent_stream_404(self, client):
        r = await client.get("/streams/doesnotexist")
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_stop_nonexistent_stream_404(self, client):
        r = await client.post(
            "/streams/stop",
            json={"stream_id": "doesnotexist"},
        )
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_start_stream_creates_record(self, client, test_app):
        """
        Mock ``VideoPipeline.from_settings`` so no real camera or model
        is needed.
        """
        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=_empty_async_gen())
        mock_pipeline.stop = MagicMock()
        mock_pipeline.set_frame_shape = MagicMock()

        with patch(
            "app.api.routes.streams.VideoPipeline.from_settings",
            return_value=mock_pipeline,
        ):
            r = await client.post(
                "/streams/start",
                json={"source": "0", "target_fps": 10},
            )

        assert r.status_code == 201
        body = r.json()
        assert "stream_id" in body
        assert "ws_url"    in body
        assert body["status"] == StreamStatus.RUNNING

    @pytest.mark.asyncio
    async def test_start_then_list(self, client, test_app):
        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=_empty_async_gen())
        mock_pipeline.stop = MagicMock()

        with patch(
            "app.api.routes.streams.VideoPipeline.from_settings",
            return_value=mock_pipeline,
        ):
            start_r = await client.post(
                "/streams/start",
                json={"source": "test.mp4"},
            )

        stream_id = start_r.json()["stream_id"]

        list_r = await client.get("/streams")
        ids    = [s["stream_id"] for s in list_r.json()["streams"]]
        assert stream_id in ids

    @pytest.mark.asyncio
    async def test_start_then_stop(self, client, test_app):
        mock_pipeline = MagicMock()
        mock_pipeline.run = AsyncMock(return_value=_empty_async_gen())
        mock_pipeline.stop = MagicMock()

        with patch(
            "app.api.routes.streams.VideoPipeline.from_settings",
            return_value=mock_pipeline,
        ):
            start_r = await client.post(
                "/streams/start",
                json={"source": "0"},
            )

        stream_id = start_r.json()["stream_id"]
        stop_r    = await client.post(
            "/streams/stop",
            json={"stream_id": stream_id},
        )
        assert stop_r.status_code == 200
        assert stop_r.json()["status"] == StreamStatus.STOPPED


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

class TestExceptionHandlers:

    @pytest.mark.asyncio
    async def test_404_has_error_envelope(self, client):
        r = await client.get("/this/does/not/exist")
        assert r.status_code == 404
        body = r.json()
        assert "error" in body
        assert body["error"]["code"] == 404

    @pytest.mark.asyncio
    async def test_validation_error_422(self, client):
        # Missing required stream_id field
        r = await client.post("/streams/stop", json={})
        assert r.status_code == 422
        body = r.json()
        assert "error" in body
        assert body["error"]["code"] == 422

    @pytest.mark.asyncio
    async def test_validation_error_has_detail(self, client):
        r = await client.post("/streams/stop", json={})
        detail = r.json()["error"].get("detail", [])
        assert isinstance(detail, list)
        assert len(detail) >= 1


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

class TestMiddleware:

    @pytest.mark.asyncio
    async def test_process_time_header_present(self, client):
        r = await client.get("/health")
        assert "x-process-time-ms" in r.headers

    @pytest.mark.asyncio
    async def test_correlation_id_injected(self, client):
        r = await client.get("/health")
        assert "x-request-id" in r.headers

    @pytest.mark.asyncio
    async def test_custom_correlation_id_propagated(self, client):
        r = await client.get(
            "/health",
            headers={"X-Request-ID": "test-rid-123"},
        )
        assert r.headers.get("x-request-id") == "test-rid-123"


# ---------------------------------------------------------------------------
# ConnectionManager unit tests
# ---------------------------------------------------------------------------

class TestConnectionManager:

    def test_initial_state(self):
        mgr = ConnectionManager()
        assert mgr.total_clients == 0
        assert mgr.active_streams == []

    def test_client_count_zero_for_unknown_stream(self):
        mgr = ConnectionManager()
        assert mgr.client_count("nonexistent") == 0

    @pytest.mark.asyncio
    async def test_broadcast_does_nothing_with_no_clients(self):
        mgr = ConnectionManager()
        # Should not raise even with no connected clients
        await mgr.broadcast_frame("stream1", {"type": "frame"}, b"\xff\xd8\xff")
        await mgr.broadcast_event("stream1", {"type": "event"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _empty_async_gen():
    """Async generator that yields nothing — simulates an immediate-stop pipeline."""
    return
    yield   # makes this an async generator