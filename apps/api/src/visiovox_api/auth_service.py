"""Authentication: registration, login, and refresh-token rotation.

The interesting part is rotation with family reuse detection (FR-ACC-04).

Every refresh mints a new token and marks the old one used. If a token that has
already been used is presented again, the only explanations are a stolen token
being replayed or a client bug — and we cannot tell which from the request. So
the entire family is revoked, which logs out the attacker and the legitimate
user together. That is the intended trade: a forced re-login is a small cost
next to an attacker holding a renewable session indefinitely.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .models import RefreshToken, Session, User, new_id
from .security import (
    hash_ip,
    hash_password,
    hash_token,
    mint_access_token,
    needs_rehash,
    new_refresh_token,
    verify_password,
)


class AuthError(Exception):
    """Authentication failed. Message is safe to return to the client."""


class TokenReuseError(AuthError):
    """A used refresh token was presented again — the family has been revoked."""


@dataclass(frozen=True)
class TokenPair:
    access_token: str
    access_expires_at: dt.datetime
    refresh_token: str
    refresh_expires_at: dt.datetime
    session_id: str
    user_id: str


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


async def register_user(
    session: AsyncSession, email: str, password: str, display_name: str | None = None
) -> User:
    """Create an account. Email is normalised so casing cannot fork an identity."""
    normalised = email.strip().lower()
    existing = await session.scalar(select(User).where(User.email == normalised))
    if existing is not None:
        # Deliberately the same message the caller shows for any failure, so the
        # endpoint does not become a registration oracle for existing accounts.
        raise AuthError("could not create account")

    user = User(
        email=normalised,
        password_hash=hash_password(password),
        display_name=display_name,
    )
    session.add(user)
    await session.flush()
    return user


async def _issue_pair(
    session: AsyncSession,
    user: User,
    auth_session: Session,
    family_id: str,
    parent_id: str | None,
    secret: str,
    access_ttl_seconds: int,
    refresh_ttl_days: int,
) -> TokenPair:
    access, access_expires = mint_access_token(user.id, auth_session.id, secret, access_ttl_seconds)
    raw_refresh = new_refresh_token()
    refresh_expires = _now() + dt.timedelta(days=refresh_ttl_days)

    record = RefreshToken(
        id=new_id("rft"),
        family_id=family_id,
        user_id=user.id,
        session_id=auth_session.id,
        token_hash=hash_token(raw_refresh),
        parent_id=parent_id,
        expires_at=refresh_expires,
    )
    session.add(record)
    await session.flush()

    return TokenPair(
        access_token=access,
        access_expires_at=access_expires,
        refresh_token=raw_refresh,
        refresh_expires_at=refresh_expires,
        session_id=auth_session.id,
        user_id=user.id,
    )


async def login(
    session: AsyncSession,
    email: str,
    password: str,
    *,
    secret: str,
    access_ttl_seconds: int,
    refresh_ttl_days: int,
    user_agent: str | None = None,
    ip: str | None = None,
) -> TokenPair:
    normalised = email.strip().lower()
    user = await session.scalar(
        select(User).where(User.email == normalised, User.deleted_at.is_(None))
    )

    # verify_password does the hashing work even when the user is absent, so
    # timing does not reveal whether the account exists.
    if not verify_password(password, user.password_hash if user else None) or user is None:
        raise AuthError("invalid email or password")

    if user.password_hash and needs_rehash(user.password_hash):
        user.password_hash = hash_password(password)

    auth_session = Session(
        id=new_id("ses"),
        user_id=user.id,
        user_agent=user_agent,
        ip_hash=hash_ip(ip, secret),
        expires_at=_now() + dt.timedelta(days=refresh_ttl_days),
    )
    session.add(auth_session)
    await session.flush()

    return await _issue_pair(
        session,
        user,
        auth_session,
        family_id=new_id("fam"),
        parent_id=None,
        secret=secret,
        access_ttl_seconds=access_ttl_seconds,
        refresh_ttl_days=refresh_ttl_days,
    )


async def refresh(
    session: AsyncSession,
    raw_token: str,
    *,
    secret: str,
    access_ttl_seconds: int,
    refresh_ttl_days: int,
) -> TokenPair:
    """Rotate a refresh token, detecting replay of an already-used one."""
    digest = hash_token(raw_token)
    record = await session.scalar(select(RefreshToken).where(RefreshToken.token_hash == digest))
    if record is None:
        raise AuthError("invalid refresh token")

    if record.used_at is not None:
        # Replay. We cannot distinguish theft from a buggy client, so assume the
        # worse case and revoke the lineage rather than the single token.
        await _revoke_family(session, record.family_id)
        raise TokenReuseError("refresh token reuse detected; session revoked")

    now = _now()
    if record.revoked_at is not None:
        raise AuthError("refresh token revoked")
    if record.expires_at <= now:
        raise AuthError("refresh token expired")

    auth_session = await session.get(Session, record.session_id)
    if auth_session is None or auth_session.revoked_at is not None:
        raise AuthError("session revoked")

    user = await session.get(User, record.user_id)
    if user is None or user.deleted_at is not None:
        raise AuthError("account unavailable")

    record.used_at = now
    auth_session.last_seen_at = now

    return await _issue_pair(
        session,
        user,
        auth_session,
        family_id=record.family_id,
        parent_id=record.id,
        secret=secret,
        access_ttl_seconds=access_ttl_seconds,
        refresh_ttl_days=refresh_ttl_days,
    )


async def _revoke_family(session: AsyncSession, family_id: str) -> None:
    now = _now()
    await session.execute(
        update(RefreshToken)
        .where(RefreshToken.family_id == family_id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=now)
    )
    session_ids = (
        await session.scalars(
            select(RefreshToken.session_id).where(RefreshToken.family_id == family_id)
        )
    ).all()
    if session_ids:
        await session.execute(
            update(Session)
            .where(Session.id.in_(set(session_ids)), Session.revoked_at.is_(None))
            .values(revoked_at=now)
        )


async def logout(session: AsyncSession, raw_token: str) -> None:
    """Revoke the whole family, so no sibling token survives the logout."""
    record = await session.scalar(
        select(RefreshToken).where(RefreshToken.token_hash == hash_token(raw_token))
    )
    if record is not None:
        await _revoke_family(session, record.family_id)
