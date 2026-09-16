"""Sliding-window rate limiting (docs/11-api-spec.md §10, docs/15-security.md §9).

Counters live in Redis because the limit must hold across API replicas; a
per-process counter is not a limit, it is a limit divided by the number of
workers.

**Rate limits are not the budget control.** A user inside every limit here can
still queue fifty hour-long videos. What actually bounds cost is the quota layer
(uploads/day, media-minutes/month, GPU-seconds) and the disk cut-out in
`diskspace`. Limits stop bursts and brute force; quotas stop expense.

The window is a **weighted two-bucket counter** rather than a fixed window or a
sorted-set log. A fixed window allows twice the limit across a boundary - 5
logins at 14:59 and 5 more at 15:00 defeats a 5-per-15-minutes rule entirely. A
sorted-set log is exact but stores one member per request, which is the most
expensive option for the endpoint most likely to be flooded. The weighted
counter costs two integers per key and is wrong by a bounded, small amount.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Protocol

from fastapi import HTTPException, Request, status

from .config import Settings


class RedisLike(Protocol):
    """The subset actually used, so tests can substitute a fake."""

    def pipeline(self) -> Any: ...
    def aclose(self) -> Awaitable[None]: ...


@dataclass(frozen=True)
class Rule:
    """One limit. `scope` names what the counter is keyed on, for the error message."""

    name: str
    limit: int
    window_seconds: int
    scope: str


# docs/11 §10, verbatim. Kept in one place so the spec and the code cannot drift.
RULES: dict[str, Rule] = {
    "auth_login": Rule("auth_login", 5, 15 * 60, "ip+email"),
    "auth_refresh": Rule("auth_refresh", 30, 60, "user"),
    "upload_init": Rule("upload_init", 10, 60 * 60, "user"),
    "exports": Rule("exports", 20, 60 * 60, "user"),
    "shared_token": Rule("shared_token", 60, 60, "token+ip"),
    "reads": Rule("reads", 300, 60, "user"),
    "global_ip": Rule("global_ip", 1000, 60, "ip"),
}


@dataclass(frozen=True)
class Decision:
    """The outcome of one check, and everything the response headers need."""

    allowed: bool
    limit: int
    remaining: int
    reset_seconds: int
    rule: Rule

    def headers(self) -> dict[str, str]:
        # RFC 9239-style names, as specified in docs/11 §1.
        h = {
            "RateLimit-Limit": str(self.limit),
            "RateLimit-Remaining": str(max(0, self.remaining)),
            "RateLimit-Reset": str(self.reset_seconds),
        }
        if not self.allowed:
            h["Retry-After"] = str(self.reset_seconds)
        return h


def client_ip(request: Request, settings: Settings) -> str:
    """The caller's address, as far as it can be trusted.

    Behind a tunnel or CDN every request arrives from the proxy, so
    `request.client.host` is the *proxy's* address and all users would share one
    bucket - a global limit that locks out everybody at once and a per-IP limit
    that means nothing. When a trusted proxy header is configured we read that
    instead.

    It is configuration rather than a default because a spoofable header is
    worse than a useless one: if anything can set `CF-Connecting-IP`, an
    attacker picks a new identity per request and per-IP limiting is over.
    Only enable it when a proxy that overwrites the header is genuinely in front.
    """
    header = settings.trusted_client_ip_header
    if header:
        value = request.headers.get(header)
        if value:
            # X-Forwarded-For style: first entry is the original client.
            return value.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RateLimiter:
    """Weighted two-bucket sliding window over Redis."""

    def __init__(self, redis: RedisLike, settings: Settings) -> None:
        self._redis = redis
        self._settings = settings

    async def check(self, rule: Rule, key: str, *, cost: int = 1) -> Decision:
        """Record a request against `key` and say whether it is allowed.

        `cost` lets one request consume several slots; unused for now, but it is
        how an expensive endpoint gets charged more than a cheap one without a
        second rule.
        """
        window = rule.window_seconds
        now = time.time()
        bucket = int(now // window)
        elapsed = now % window

        cur_key = f"rl:{rule.name}:{key}:{bucket}"
        prev_key = f"rl:{rule.name}:{key}:{bucket - 1}"

        try:
            pipe = self._redis.pipeline()
            pipe.incrby(cur_key, cost)
            # Two windows of TTL: the previous bucket must outlive its own window
            # because it is still being weighted into the current one.
            pipe.expire(cur_key, window * 2)
            pipe.get(prev_key)
            current_raw, _, previous_raw = await pipe.execute()
        except Exception:
            # Redis unavailable. Failing closed would take the whole API down
            # with the cache, which is a worse outcome than a gap in limiting -
            # and matches how the rest of the codebase treats Redis (progress
            # falls back to the database rather than erroring). Set
            # rate_limit_fail_open=false where the opposite trade is wanted.
            if self._settings.rate_limit_fail_open:
                return Decision(True, rule.limit, rule.limit, window, rule)
            raise

        current = int(current_raw or 0)
        previous = int(previous_raw or 0)

        # The previous bucket's contribution decays as the current one fills.
        weight = 1.0 - (elapsed / window)
        estimated = previous * weight + current

        remaining = int(rule.limit - estimated)
        reset = int(window - elapsed) or 1
        return Decision(estimated <= rule.limit, rule.limit, remaining, reset, rule)


def enforce(decision: Decision, detail: str | None = None) -> None:
    """Raise 429 when a decision refused, with the headers a client needs."""
    if decision.allowed:
        return
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=detail
        or (
            f"rate limit exceeded: {decision.rule.limit} per "
            f"{decision.rule.window_seconds}s per {decision.rule.scope}"
        ),
        headers=decision.headers(),
    )
