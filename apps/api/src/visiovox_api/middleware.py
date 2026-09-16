"""Cross-cutting request handling: correlation ids, security headers, global limit.

Order matters and is set in `main`: correlation id first (everything else wants
to log with it), then the global per-IP limit (cheapest possible rejection for a
flood), then security headers on the way out.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from ulid import ULID

from .config import Settings
from .problems import problem
from .ratelimit import RULES, RateLimiter, client_ip

Next = Callable[[Request], Awaitable[Response]]

CORRELATION_HEADER = "X-Correlation-Id"


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Attach an id to every request, and echo it back.

    A client-supplied id is accepted so a trace can span the browser and the
    API, but it is length-capped: it lands in log lines, and an unbounded header
    that reaches the logs is a log-injection and log-volume problem.
    """

    async def dispatch(self, request: Request, call_next: Next) -> Response:
        supplied = request.headers.get(CORRELATION_HEADER, "")
        request.state.correlation_id = supplied[:64] if supplied else str(ULID())
        response = await call_next(request)
        response.headers[CORRELATION_HEADER] = request.state.correlation_id
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Security headers on every response (docs/11-api-spec.md §11)."""

    def __init__(self, app: object, settings: Settings) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._settings = settings

    async def dispatch(self, request: Request, call_next: Next) -> Response:
        response = await call_next(request)
        h = response.headers
        h.setdefault("X-Content-Type-Options", "nosniff")
        h.setdefault("X-Frame-Options", "DENY")
        h.setdefault("Referrer-Policy", "no-referrer")
        # This is the API, which renders nothing: the most restrictive policy
        # that exists is correct here. The web app's CSP is a separate thing and
        # has to be far more permissive to allow WebGL (docs/28 §W1).
        h.setdefault("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'")
        h.setdefault(
            "Permissions-Policy", "geolocation=(), microphone=(), camera=(), interest-cohort=()"
        )
        h.setdefault("Cross-Origin-Resource-Policy", "same-origin")
        # HSTS only over TLS: sending it on plain HTTP is meaningless, and in
        # local development it would pin localhost to https in the browser and
        # be genuinely annoying to undo.
        if self._settings.is_production:
            h.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains; preload"
            )
        return response


class GlobalRateLimitMiddleware(BaseHTTPMiddleware):
    """1000 requests/minute per IP (docs/11 §10), before any routing work.

    Deliberately the cheapest check in the stack: a flood should be rejected
    without a database round trip, without decoding a token, and without
    instantiating a route handler.
    """

    EXEMPT_PATHS = frozenset({"/healthz", "/readyz"})

    def __init__(self, app: object, settings: Settings, limiter: RateLimiter) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._settings = settings
        self._limiter = limiter

    async def dispatch(self, request: Request, call_next: Next) -> Response:
        # Health checks are polled by the platform, not by users; counting them
        # would let a monitoring agent exhaust the limit for everyone behind the
        # same address.
        if not self._settings.rate_limit_enabled or request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        decision = await self._limiter.check(RULES["global_ip"], client_ip(request, self._settings))
        if not decision.allowed:
            return problem(
                request,
                429,
                "Too many requests from this address. Slow down and try again shortly.",
                headers=decision.headers(),
            )

        response = await call_next(request)
        # Headers on every response, per docs/11 §1 - a client should be able to
        # back off before it is refused, not only after.
        for key, value in decision.headers().items():
            response.headers.setdefault(key, value)
        return response
