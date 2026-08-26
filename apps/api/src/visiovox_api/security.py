"""Password hashing, token minting and refresh-token rotation primitives.

Argon2id for passwords: memory-hard, so a leaked hash cannot be attacked with
the GPU sitting in this very machine. bcrypt would be acceptable; SHA-family
hashing would not be, at any iteration count.

Refresh tokens are stored only as SHA-256. The raw token exists in the response
body and the client's cookie and nowhere else, so a database dump does not hand
over live sessions. SHA-256 rather than Argon2 here is deliberate: the token is
128 bits of CSPRNG output, not a guessable secret, so there is nothing for a
slow hash to buy, and refresh happens often enough that the cost would matter.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import hmac
import secrets
from dataclasses import dataclass
from typing import Any, Final

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

ALGORITHM: Final = "HS256"
TOKEN_BYTES: Final = 32  # 256 bits

_hasher = PasswordHasher()


# --------------------------------------------------------------------------
# passwords
# --------------------------------------------------------------------------


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    """Constant-ish time verify that does not leak whether the account exists.

    A missing hash (OIDC-only account, or no such user) still performs a dummy
    verification, so the response time does not distinguish "wrong password"
    from "no such user" — which is otherwise a free user-enumeration oracle.
    """
    if password_hash is None:
        # Still do the work, so response time does not distinguish this case.
        with contextlib.suppress(VerifyMismatchError, VerificationError, InvalidHashError):
            _hasher.verify(_DUMMY_HASH, password)
        return False
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


_DUMMY_HASH: Final = _hasher.hash("dummy-password-for-constant-time-comparison")


def needs_rehash(password_hash: str) -> bool:
    """True when the stored hash uses outdated parameters."""
    try:
        return _hasher.check_needs_rehash(password_hash)
    except InvalidHashError:
        return True


# --------------------------------------------------------------------------
# refresh tokens
# --------------------------------------------------------------------------


def new_refresh_token() -> str:
    return secrets.token_urlsafe(TOKEN_BYTES)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def tokens_equal(a: str, b: str) -> bool:
    return hmac.compare_digest(a, b)


# --------------------------------------------------------------------------
# access tokens
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AccessClaims:
    user_id: str
    session_id: str
    expires_at: dt.datetime


def mint_access_token(
    user_id: str, session_id: str, secret: str, ttl_seconds: int
) -> tuple[str, dt.datetime]:
    now = dt.datetime.now(dt.UTC)
    expires = now + dt.timedelta(seconds=ttl_seconds)
    payload: dict[str, Any] = {
        "sub": user_id,
        "sid": session_id,
        "iat": int(now.timestamp()),
        "exp": int(expires.timestamp()),
        "typ": "access",
    }
    return jwt.encode(payload, secret, algorithm=ALGORITHM), expires


class InvalidTokenError(Exception):
    """The access token is missing, malformed, expired or not an access token."""


def decode_access_token(token: str, secret: str) -> AccessClaims:
    try:
        payload = jwt.decode(token, secret, algorithms=[ALGORITHM])
    except jwt.PyJWTError as exc:
        raise InvalidTokenError(str(exc)) from exc

    # Reject a refresh token presented as an access token. Without this check a
    # long-lived credential would be accepted wherever a short-lived one is.
    if payload.get("typ") != "access":
        raise InvalidTokenError("not an access token")
    sub, sid, exp = payload.get("sub"), payload.get("sid"), payload.get("exp")
    if not isinstance(sub, str) or not isinstance(sid, str) or not isinstance(exp, int):
        raise InvalidTokenError("malformed claims")

    return AccessClaims(
        user_id=sub,
        session_id=sid,
        expires_at=dt.datetime.fromtimestamp(exp, tz=dt.UTC),
    )


def hash_ip(ip: str | None, secret: str) -> str | None:
    """Store a keyed hash of the client IP, never the address itself.

    Keyed so the stored value cannot be reversed by hashing the whole IPv4
    space, which takes seconds against an unkeyed digest.
    """
    if not ip:
        return None
    return hmac.new(secret.encode(), ip.encode(), hashlib.sha256).hexdigest()
