"""Idempotency keys for creating POSTs (docs/11-api-spec.md §1).

The problem this exists for: a client sends `POST /projects`, the connection
drops before the response arrives, the client retries. Without a key the retry
creates a second project. On an upload endpoint it creates a second multipart
upload, which is a second copy of a multi-gigabyte file's worth of reserved
storage - so this is a cost control as much as a correctness one.

Three states, and the middle one is the one people forget:

- **unseen** - first time; reserve the key and run the handler
- **in flight** - the original request has not finished yet. Returning 409 is
  correct and deliberate: replaying a half-finished operation is worse than
  telling the client to wait, and a client that retried this fast is usually a
  network timeout racing a slow handler
- **done** - replay the stored response verbatim

A key presented with a *different* body is a client bug - the same key naming
two different operations - and returns 422 rather than silently doing whichever
arrived first.

Stored in Redis on a 24-hour window. Losing it degrades to "a retry might
duplicate", which is where the system was before this module; no correctness
that must survive a restart depends on it (docs/28 §D4).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, status

from .config import Settings

WINDOW_SECONDS = 24 * 60 * 60
IN_FLIGHT = "in_flight"
DONE = "done"


@dataclass(frozen=True)
class Replay:
    """A stored response to return instead of running the handler again."""

    status_code: int
    body: dict[str, Any]


def _redis_key(user_id: str, endpoint: str, client_key: str) -> str:
    """Namespaced by user and endpoint.

    Without the user in the key, two people choosing the same key - and clients
    do choose predictable ones - would read each other's responses. That is a
    data leak, not merely a collision.
    """
    digest = hashlib.sha256(f"{user_id}|{endpoint}|{client_key}".encode()).hexdigest()
    return f"idem:{digest}"


def fingerprint(body: Any) -> str:
    """A stable hash of the request body, to detect a key reused for something else."""
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()[:32]


def validate_key(client_key: str | None, *, required: bool) -> str | None:
    if client_key is None:
        if required:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Idempotency-Key header is required on this endpoint.",
            )
        return None
    key = client_key.strip()
    # Bounded because it becomes part of a Redis key and is attacker-supplied.
    if not 8 <= len(key) <= 200:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Idempotency-Key must be between 8 and 200 characters.",
        )
    return key


async def begin(
    redis: Any,
    settings: Settings,
    *,
    user_id: str,
    endpoint: str,
    client_key: str,
    body_fingerprint: str,
) -> Replay | None:
    """Reserve the key, or hand back what the first attempt produced.

    `SET NX` is what makes the reservation atomic: two simultaneous retries
    cannot both win it, so exactly one runs the handler.
    """
    key = _redis_key(user_id, endpoint, client_key)
    reserved = {"state": IN_FLIGHT, "fingerprint": body_fingerprint}

    try:
        won = await redis.set(key, json.dumps(reserved), nx=True, ex=WINDOW_SECONDS)
        if won:
            return None
        raw = await redis.get(key)
    except Exception:
        # Redis down: fall through and run the handler. The degraded behaviour
        # is "a retry might duplicate", which is exactly where this endpoint was
        # before idempotency existed - not a new failure mode.
        if settings.rate_limit_fail_open:
            return None
        raise

    if raw is None:
        # Expired between SET and GET. Racing here is harmless: run the handler.
        return None

    record = json.loads(raw)
    if record.get("fingerprint") != body_fingerprint:
        raise HTTPException(
            status_code=422,
            detail=(
                "This Idempotency-Key was already used for a different request. "
                "Use a new key for a new operation."
            ),
        )

    if record.get("state") == IN_FLIGHT:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An identical request is still being processed. Retry shortly.",
            headers={"Retry-After": "2"},
        )

    return Replay(status_code=int(record["status_code"]), body=record["body"])


async def complete(
    redis: Any,
    *,
    user_id: str,
    endpoint: str,
    client_key: str,
    body_fingerprint: str,
    status_code: int,
    body: dict[str, Any],
) -> None:
    """Store the response so a retry replays it. Never raises."""
    key = _redis_key(user_id, endpoint, client_key)
    record = {
        "state": DONE,
        "fingerprint": body_fingerprint,
        "status_code": status_code,
        "body": body,
    }
    try:
        await redis.set(key, json.dumps(record, default=str), ex=WINDOW_SECONDS)
    except Exception:
        return


async def release(redis: Any, *, user_id: str, endpoint: str, client_key: str) -> None:
    """Drop a reservation whose handler failed.

    Without this, a handler that raises leaves the key parked in `in_flight` for
    24 hours and the client can never retry that operation - a transient error
    would become a day-long outage for one request.
    """
    try:
        await redis.delete(_redis_key(user_id, endpoint, client_key))
    except Exception:
        return
