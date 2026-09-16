"""RFC 9457 Problem Details (docs/11-api-spec.md §1).

Every error leaves the API in one shape, carrying a `correlation_id` that also
appears in the logs and the trace. That id is what a user quotes in a support
request, and it is what turns "it didn't work" into a single query.

**`detail` is user-facing and must stay that way.** No stack traces, no SQL, no
storage keys, no model names - an error message is an information-disclosure
channel that is easy to forget about because it only fires when something is
already going wrong. `unhandled_exception_handler` therefore never echoes the
exception: the id is the bridge to the detail, which stays in the logs.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

CONTENT_TYPE = "application/problem+json"
BASE_URI = "https://visiovox.app/errors/"

# Machine-readable codes by status. A client should branch on `code`, never on
# the prose in `title`, which is free to change.
_CODES: dict[int, tuple[str, str]] = {
    400: ("BAD_REQUEST", "Bad request"),
    401: ("UNAUTHENTICATED", "Authentication required"),
    403: ("FORBIDDEN", "Not permitted"),
    404: ("NOT_FOUND", "Not found"),
    409: ("CONFLICT", "Conflict"),
    413: ("PAYLOAD_TOO_LARGE", "File too large"),
    422: ("VALIDATION_FAILED", "Request validation failed"),
    429: ("RATE_LIMITED", "Too many requests"),
    500: ("INTERNAL_ERROR", "Something went wrong"),
    503: ("UNAVAILABLE", "Temporarily unavailable"),
}


def correlation_id(request: Request) -> str:
    value: str = getattr(request.state, "correlation_id", "")
    return value


def problem(
    request: Request,
    status_code: int,
    detail: str,
    *,
    code: str | None = None,
    title: str | None = None,
    headers: dict[str, str] | None = None,
    **extra: Any,
) -> JSONResponse:
    default_code, default_title = _CODES.get(status_code, ("ERROR", "Error"))
    resolved_code = code or default_code
    body: dict[str, Any] = {
        "type": f"{BASE_URI}{resolved_code.lower().replace('_', '-')}",
        "title": title or default_title,
        "status": status_code,
        "detail": detail,
        "instance": request.url.path,
        "code": resolved_code,
        "correlation_id": correlation_id(request),
    }
    body.update({k: v for k, v in extra.items() if v is not None})

    response_headers = dict(headers or {})
    # Retry-After is a header AND a field: the header is what clients and proxies
    # obey, the field is what a human reading the body can act on.
    if "Retry-After" in response_headers:
        body.setdefault("retry_after", int(response_headers["Retry-After"]))

    return JSONResponse(
        status_code=status_code,
        content=body,
        headers=response_headers,
        media_type=CONTENT_TYPE,
    )


async def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, HTTPException)
    detail = exc.detail if isinstance(exc.detail, str) else "Request failed"
    return problem(request, exc.status_code, detail, headers=dict(exc.headers or {}))


async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    # Field paths and messages are safe and genuinely useful; the raw input is
    # not echoed, because it may be exactly the thing the caller should not have
    # sent and should not see reflected.
    errors = [
        {"field": ".".join(str(p) for p in e.get("loc", ())[1:]), "message": e.get("msg", "")}
        for e in exc.errors()
    ]
    return problem(
        request,
        422,  # not the starlette constant: its name changed between versions
        "One or more fields are invalid.",
        errors=errors,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last resort: log everything, disclose nothing."""
    logger.exception(
        "unhandled exception",
        extra={"correlation_id": correlation_id(request), "path": request.url.path},
    )
    return problem(
        request,
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "Something went wrong on our side. Quote the correlation id if you contact support.",
    )
