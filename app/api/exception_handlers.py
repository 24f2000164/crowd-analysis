"""
app/api/exception_handlers.py
===============================
Global exception handlers registered on the FastAPI application.

Handlers
--------
http_exception_handler     — wraps HTTPException into a consistent JSON envelope.
validation_exception_handler — wraps Pydantic RequestValidationError.
unhandled_exception_handler  — catch-all for unexpected 500 errors.

All responses follow the same envelope so API clients always parse the same
shape regardless of error type:

    {
        "error": {
            "code":    422,
            "type":    "validation_error",
            "message": "1 validation error for StartStreamRequest …",
            "detail":  [ … ]
        }
    }
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Dict

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("crowd_analysis.api.exceptions")


def _error_envelope(
    code:    int,
    type_:   str,
    message: str,
    detail:  Any = None,
) -> Dict:
    body: Dict = {"error": {"code": code, "type": type_, "message": message}}
    if detail is not None:
        body["error"]["detail"] = detail
    return body


# ---------------------------------------------------------------------------
# HTTP exceptions (4xx raised explicitly by routes)
# ---------------------------------------------------------------------------

async def http_exception_handler(
    request: Request,
    exc:     StarletteHTTPException,
) -> JSONResponse:
    logger.warning(
        "HTTP %d — %s %s — %s",
        exc.status_code,
        request.method,
        request.url.path,
        exc.detail,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_envelope(
            code    = exc.status_code,
            type_   = _status_to_type(exc.status_code),
            message = str(exc.detail),
        ),
    )


# ---------------------------------------------------------------------------
# Pydantic validation errors (422)
# ---------------------------------------------------------------------------

async def validation_exception_handler(
    request: Request,
    exc:     RequestValidationError,
) -> JSONResponse:
    errors = exc.errors()
    # Simplify each Pydantic error to a human-readable dict
    simplified = [
        {
            "field":   " → ".join(str(loc) for loc in err["loc"]),
            "message": err["msg"],
            "type":    err["type"],
        }
        for err in errors
    ]
    logger.warning(
        "Validation error — %s %s — %d field(s) invalid",
        request.method, request.url.path, len(errors),
    )
    return JSONResponse(
        status_code=422,
        content=_error_envelope(
            code    = 422,
            type_   = "validation_error",
            message = f"{len(errors)} validation error(s).",
            detail  = simplified,
        ),
    )


# ---------------------------------------------------------------------------
# Unhandled exceptions (500)
# ---------------------------------------------------------------------------

async def unhandled_exception_handler(
    request: Request,
    exc:     Exception,
) -> JSONResponse:
    tb = traceback.format_exc()
    logger.critical(
        "Unhandled exception — %s %s\n%s",
        request.method, request.url.path, tb,
    )
    return JSONResponse(
        status_code=500,
        content=_error_envelope(
            code    = 500,
            type_   = "internal_server_error",
            message = "An unexpected error occurred. Please check the server logs.",
        ),
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _status_to_type(code: int) -> str:
    return {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        405: "method_not_allowed",
        409: "conflict",
        422: "validation_error",
        429: "too_many_requests",
        500: "internal_server_error",
        502: "bad_gateway",
        503: "service_unavailable",
    }.get(code, "http_error")