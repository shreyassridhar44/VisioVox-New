"""Shared async Redis client.

One pool per process. Creating a client per request exhausts file descriptors
under load - the failure arrives as connection errors from unrelated code, which
is a slow thing to diagnose.

Redis holds only disposable state here: rate-limit counters, job progress,
idempotency records. Losing it costs a window of limiting and some cached
progress, never correctness (docs/28 §D4). Anything that must survive a restart
belongs in Postgres.
"""

from __future__ import annotations

from typing import Any

from redis.asyncio import Redis

from .config import Settings

_client: Redis | None = None


def get_redis(settings: Settings) -> Redis:
    global _client
    if _client is None:
        _client = Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            # A hung Redis must not hold an API worker open: the limiter treats a
            # timeout as "unavailable" and falls back, which is only useful if
            # the timeout actually fires quickly.
            socket_timeout=settings.redis_timeout_seconds,
            socket_connect_timeout=settings.redis_timeout_seconds,
            health_check_interval=30,
        )
    return _client


async def close_redis() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def reset_redis_for_tests(client: Any = None) -> None:
    """Swap the module-level client. Tests only."""
    global _client
    _client = client
