"""Exponential backoff on repeated authentication failure (docs/15 §9).

The rate limit on `/auth/login` is keyed on ip+email, which bounds how fast one
attacker can guess one account's password. It does not bound a *slow* attacker:
five attempts per fifteen minutes is 480 guesses a day against a single account,
indefinitely, and a weak password does not survive that.

Backoff makes each failure cost more than the last, so a run of failures becomes
self-limiting regardless of how patiently it is paced. Counted per account,
because that is what is under attack - an attacker changes address freely, and a
victim cannot change the email being targeted.

The delay is capped. An uncapped doubling locks a real user out for days after a
bad afternoon, which converts a security control into a denial of service
against the person it is supposed to protect.

A successful login clears the counter, so the only people who ever feel this are
those who keep failing.
"""

from __future__ import annotations

import hashlib
from typing import Any

from fastapi import HTTPException, status

from .config import Settings

# Failures before backoff starts. The first few are ordinary human mistakes -
# a stale password manager entry, caps lock - and should cost nothing.
FREE_ATTEMPTS = 3
BASE_DELAY_SECONDS = 2
MAX_DELAY_SECONDS = 15 * 60
COUNTER_TTL_SECONDS = 60 * 60


def _key(email: str) -> str:
    """Hashed: the failure counters must not become a list of registered emails
    readable by anyone who can see the keyspace."""
    digest = hashlib.sha256(email.strip().lower().encode()).hexdigest()
    return f"authfail:{digest}"


def delay_for(failures: int) -> int:
    """Seconds a caller must wait after `failures` consecutive failures."""
    if failures <= FREE_ATTEMPTS:
        return 0
    over = failures - FREE_ATTEMPTS
    return int(min(BASE_DELAY_SECONDS * (2 ** (over - 1)), MAX_DELAY_SECONDS))


async def assert_not_backed_off(redis: Any, settings: Settings, email: str) -> None:
    """Refuse early when the account is inside its backoff window.

    Checked before the password is verified, so a locked-out attacker does not
    get to consume a bcrypt round per attempt - the hash comparison is the
    expensive part, and letting it run is its own small denial-of-service.
    """
    try:
        raw = await redis.get(_key(email))
    except Exception:
        if settings.rate_limit_fail_open:
            return
        raise

    failures = int(raw or 0)
    wait = delay_for(failures)
    if wait <= 0:
        return

    try:
        ttl = await redis.ttl(f"{_key(email)}:gate")
    except Exception:
        ttl = -1

    if ttl and ttl > 0:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Too many failed sign-in attempts for this account. Try again in {ttl} seconds."
            ),
            headers={"Retry-After": str(ttl)},
        )


async def record_failure(redis: Any, email: str) -> int:
    """Count a failure and open a gate for the resulting delay."""
    key = _key(email)
    try:
        failures = int(await redis.incr(key))
        await redis.expire(key, COUNTER_TTL_SECONDS)
        wait = delay_for(failures)
        if wait > 0:
            # A separate short-lived key holds the gate, so its TTL is the
            # remaining wait rather than the counter's much longer lifetime.
            await redis.set(f"{key}:gate", "1", ex=wait)
        return failures
    except Exception:
        return 0


async def clear(redis: Any, email: str) -> None:
    """A successful sign-in forgets the run of failures."""
    key = _key(email)
    try:
        await redis.delete(key, f"{key}:gate")
    except Exception:
        return
