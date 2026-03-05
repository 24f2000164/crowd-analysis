"""
app/api/server.py
==================
FastAPI application factory for the Real-Time Crowd Behavior Analysis System.

Architecture
------------
                       ┌─────────────────────────────────────┐
                       │           FastAPI App                │
                       │                                      │
     HTTP clients ───► │  /health          (GET)              │
                       │  /readiness       (GET)              │
                       │  /version         (GET)              │
                       │                                      │
                       │  /streams/start   (POST) ──────────► │──► StreamRegistry
                       │  /streams/stop    (POST) ──────────► │──► StreamRegistry
                       │  /streams         (GET)              │
                       │  /streams/{id}    (GET)              │
                       │                                      │
  WebSocket clients ───► /ws/stream/{id}  (WS) ─────────────► │──► ConnectionManager
                       └─────────────────────────────────────┘
                                    │
                           app.state singletons
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              ws_manager     stream_registry    settings
           (ConnectionManager) (StreamRegistry) (Settings)

Lifespan
--------
The ``lifespan`` async context manager (new in FastAPI 0.93 / Starlette 0.27)
replaces ``@app.on_event("startup")`` / ``@app.on_event("shutdown")``:

  startup  — configure logging, build singletons, store in app.state.
  shutdown — stop all streams, close all WebSocket connections.

Running the server
------------------
    # Development
    uvicorn app.api.server:create_app --factory --reload --port 8000

    # Production (from entrypoint)
    python -m app.api.server
"""

from __future__ import annotations

import logging
import logging.config
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.websocket.manager   import ConnectionManager
from app.api.stream_registry     import StreamRegistry
from app.api.middleware          import register_middleware
from app.api.exception_handlers  import (
    http_exception_handler,
    validation_exception_handler,
    unhandled_exception_handler,
)
from app.api.routes.health       import router as health_router
from app.api.routes.streams      import router as streams_router
from app.api.routes.websocket    import router as websocket_router
from app.api.routes.dashboard    import router as dashboard_router
from app.api.routes.metrics      import router as metrics_router

logger = logging.getLogger("crowd_analysis.api.server")

_SERVER_START_TIME = time.monotonic()


# ============================================================================
# Logging setup
# ============================================================================

def _configure_logging(debug: bool = False) -> None:
    """
    Configure structured logging from YAML if available, otherwise fall
    back to a sensible basicConfig.
    """
    yaml_path = Path("config/logging.yaml")
    if yaml_path.exists():
        try:
            import yaml
            with yaml_path.open() as f:
                cfg = yaml.safe_load(f)
            logging.config.dictConfig(cfg)
            return
        except Exception as exc:
            print(f"[WARNING] Could not load logging.yaml: {exc}")

    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level   = level,
        format  = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt = "%Y-%m-%dT%H:%M:%S",
    )


# ============================================================================
# Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan handler.

    Startup
    -------
    1. Configure logging.
    2. Load settings (non-fatal on failure — use defaults).
    3. Create ConnectionManager and StreamRegistry singletons.
    4. Store everything in ``app.state``.

    Shutdown
    --------
    1. Stop every active pipeline stream.
    2. Close every WebSocket connection.
    """

    # ── Startup ───────────────────────────────────────────────────────
    # 1. Load settings
    settings = _load_settings()
    _configure_logging(debug=getattr(settings, "debug", False))

    logger.info("=" * 60)
    logger.info(
        "Starting %s v%s [%s]",
        getattr(settings, "app_name",    "Crowd Analysis System"),
        getattr(settings, "app_version", "1.0.0"),
        getattr(settings, "environment", "development"),
    )
    logger.info("=" * 60)

    # 2. Build singletons
    ws_max_queue = getattr(
        getattr(settings, "api", None), "ws_max_queue", 32
    )
    ws_manager = ConnectionManager(max_queue_per_client=ws_max_queue)
    registry   = StreamRegistry(ws_manager=ws_manager)

    # Event store and performance monitor
    try:
        from services.event_store              import EventStore
        from core.metrics.performance_monitor  import PerformanceMonitor
        event_store         = EventStore()
        performance_monitor = PerformanceMonitor()
    except Exception as exc:
        logger.warning("Could not initialise EventStore / PerformanceMonitor: %s", exc)
        event_store         = None
        performance_monitor = None

    # 3. Attach to app state (accessible in every route via request.app.state)
    app.state.settings           = settings
    app.state.ws_manager         = ws_manager
    app.state.stream_registry    = registry
    app.state.start_time         = _SERVER_START_TIME
    app.state.event_store        = event_store
    app.state.performance_monitor = performance_monitor

    logger.info(
        "App state initialised — ws_manager=%s  registry=%s",
        type(ws_manager).__name__,
        type(registry).__name__,
    )

    yield   # ← application runs here

    # ── Shutdown ──────────────────────────────────────────────────────
    logger.info("Shutdown initiated …")

    # Stop all active pipelines
    try:
        await registry.stop_all()
        logger.info("All streams stopped.")
    except Exception as exc:
        logger.error("Error stopping streams: %s", exc)

    # Close all WebSocket connections
    try:
        await ws_manager.close_all()
        logger.info("All WebSocket connections closed.")
    except Exception as exc:
        logger.error("Error closing WebSocket connections: %s", exc)

    uptime = round(time.monotonic() - _SERVER_START_TIME, 1)
    logger.info("Server shutdown complete. Total uptime: %.1fs", uptime)


# ============================================================================
# Application factory
# ============================================================================

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns a fully wired ``FastAPI`` instance with:
    - Lifespan context (startup + shutdown)
    - CORS + logging + correlation-ID middleware
    - Global exception handlers
    - All routers mounted
    - OpenAPI docs (auto-disabled in production)

    Usage::

        # Direct import
        from app.api.server import create_app
        app = create_app()

        # uvicorn --factory
        uvicorn app.api.server:create_app --factory
    """
    settings    = _load_settings()
    environment = getattr(settings, "environment", "development")
    debug       = getattr(settings, "debug", False)

    # Disable interactive docs in production for security
    docs_url    = "/docs"    if environment != "production" else None
    redoc_url   = "/redoc"   if environment != "production" else None
    openapi_url = "/openapi.json" if environment != "production" else None

    app = FastAPI(
        title       = getattr(settings, "app_name",    "Crowd Analysis System"),
        version     = getattr(settings, "app_version", "1.0.0"),
        description = _API_DESCRIPTION,
        lifespan    = lifespan,
        debug       = debug,
        docs_url    = docs_url,
        redoc_url   = redoc_url,
        openapi_url = openapi_url,
    )

    # ── Middleware ─────────────────────────────────────────────────────
    cors_origins = getattr(
        getattr(settings, "api", None), "cors_origins", ["*"]
    )
    register_middleware(app, cors_origins=cors_origins)

    # ── Exception handlers ─────────────────────────────────────────────
    app.add_exception_handler(StarletteHTTPException,  http_exception_handler)
    app.add_exception_handler(RequestValidationError,  validation_exception_handler)
    app.add_exception_handler(Exception,               unhandled_exception_handler)

    # ── Routers ────────────────────────────────────────────────────────
    app.include_router(health_router)           # /health, /readiness, /version
    app.include_router(streams_router)          # /streams/start, /streams/stop …
    app.include_router(websocket_router)        # /ws/stream/{stream_id}
    app.include_router(dashboard_router)        # /dashboard/metrics|events|streams
    app.include_router(metrics_router)          # /metrics

    # ── Static files — dashboard ───────────────────────────────────────
    _dashboard_dir = Path(__file__).resolve().parent.parent / "dashboard"
    if _dashboard_dir.exists():
        from starlette.staticfiles import StaticFiles
        app.mount(
            "/dashboard",
            StaticFiles(directory=str(_dashboard_dir), html=True),
            name="dashboard",
        )
        logger.info("Dashboard UI mounted at /dashboard — dir=%s", _dashboard_dir)
    else:
        logger.debug(
            "Dashboard directory not found at %s — static UI not mounted.",
            _dashboard_dir,
        )

    # ── Root redirect ──────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        return JSONResponse({
            "service":   getattr(settings, "app_name", "Crowd Analysis System"),
            "version":   getattr(settings, "app_version", "1.0.0"),
            "docs":      docs_url or "disabled in production",
            "health":    "/health",
            "dashboard": "/dashboard",
            "metrics":   "/metrics",
        })

    logger.debug(
        "FastAPI app created — environment=%s  debug=%s  docs=%s",
        environment, debug, docs_url,
    )
    return app


# ============================================================================
# Helpers
# ============================================================================

def _load_settings():
    """
    Attempt to load application settings.  Returns a minimal namespace
    with safe defaults if ``config/settings.py`` is unavailable (e.g. in
    CI / test environments without all deps installed).
    """
    try:
        from config.settings import get_settings
        return get_settings()
    except Exception as exc:
        logger.warning("Could not load settings: %s — using inline defaults.", exc)

        # Lightweight fallback namespace
        class _Defaults:
            app_name    = "Crowd Analysis System"
            app_version = "1.0.0"
            environment = "development"
            debug       = False
            class api:
                host         = "0.0.0.0"
                port         = 8000
                api_key      = None
                cors_origins = ["*"]
                ws_max_queue = 32
            class model:
                yolo_weights = "models/yolov8n.pt"

        return _Defaults()


# ============================================================================
# OpenAPI description (shown in Swagger UI)
# ============================================================================

_API_DESCRIPTION = """
## Real-Time Crowd Behavior Analysis API

Stream annotated video frames from any camera source with live
crowd behavior classification.

### Quick Start

1. **Start a camera stream**
   ```
   POST /streams/start
   {"source": "0", "target_fps": 25}
   ```
   Response includes a `stream_id` and `ws_url`.

2. **Connect via WebSocket**
   ```
   ws://<host>/ws/stream/<stream_id>
   ```
   Receive alternating **text** (JSON metadata) + **binary** (JPEG) messages.

3. **Stop the stream**
   ```
   POST /streams/stop
   {"stream_id": "<stream_id>"}
   ```

### Behavior Labels

| Label | Description |
|---|---|
| `normal` | Baseline crowd movement |
| `running` | Significant fraction sprinting |
| `panic` | Mass anomalous chaotic movement |
| `violence` | Close-contact high-speed conflict |
| `suspicion` | Isolated erratic individual |
| `crowd_surge` | Coordinated high-speed collective surge |

### Authentication

Pass `X-API-Key: <key>` header on HTTP endpoints.  
Pass `?api_key=<key>` query parameter on WebSocket connections.  
Auth is disabled when `API_KEY` environment variable is not set.
"""


# ============================================================================
# Entrypoint
# ============================================================================

# Module-level app instance — used when running via:
#   uvicorn app.api.server:app
app = create_app()


if __name__ == "__main__":
    settings = _load_settings()
    api_cfg  = getattr(settings, "api", None)

    uvicorn.run(
        "app.api.server:app",
        host        = getattr(api_cfg, "host",    "0.0.0.0"),
        port        = getattr(api_cfg, "port",    8000),
        reload      = getattr(settings, "debug",  False),
        log_level   = "debug" if getattr(settings, "debug", False) else "info",
        ws_ping_interval = getattr(api_cfg, "ws_ping_interval_s", 20),
        ws_ping_timeout  = getattr(api_cfg, "ws_ping_timeout_s",  10),
        access_log  = False,   # handled by RequestLoggingMiddleware
    )