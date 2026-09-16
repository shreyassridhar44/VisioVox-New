"""Share links (docs/28 §W7, §D3).

A share link is the one part of this product that is deliberately reachable
without an account, so it is also the part most worth being careful about.

- **Unguessable.** 256 bits from `secrets`, well past the 128 the plan asks for.
  The token is the entire authorisation, so it has to be infeasible to search.
- **Hashed at rest.** Only the SHA-256 is stored. A leaked database dump must
  not be a set of working links.
- **Revocation is durable and immediate.** It is a column, not a cache entry,
  and `is_live` is the single place that decides.
- **Never indexed.** `X-Robots-Tag: noindex` on every response, because a link
  someone shared with three people should not arrive in a search result.
- **Rate limited on token+IP** (docs/11 §10), which is what makes brute force
  pointless rather than merely expensive.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import secrets
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response, status
from sqlalchemy import select

from . import audit, problems
from .deps import CurrentUser, OwnedProject, SessionDep, SettingsDep
from .models import Project, ShareLink
from .ratelimit import RULES, RateLimiter, client_ip, enforce
from .redis_client import get_redis
from .schemas import (
    CreateShareRequest,
    ShareListResponse,
    SharePublicResponse,
    ShareResponse,
)
from .security import hash_password, verify_password

router = APIRouter(prefix="/v1", tags=["share"])

TOKEN_BYTES = 32  # 256 bits


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _response(row: ShareLink, token: str | None = None) -> ShareResponse:
    return ShareResponse(
        id=row.id,
        project_id=row.project_id,
        speaker_ordinal=row.speaker_ordinal,
        # Returned exactly once, at creation. There is no way to recover it
        # later, which is the point of storing only the hash.
        token=token,
        has_password=row.password_hash is not None,
        access_count=row.access_count,
        expires_at=row.expires_at,
        revoked_at=row.revoked_at,
        created_at=row.created_at,
    )


@router.post("/projects/{project_id}/shares", status_code=status.HTTP_201_CREATED)
async def create_share(
    body: CreateShareRequest,
    request: Request,
    project: OwnedProject,
    user: CurrentUser,
    session: SessionDep,
    settings: SettingsDep,
) -> ShareResponse:
    if project.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"This project is {project.status}; there is nothing to share yet.",
        )

    token = secrets.token_urlsafe(TOKEN_BYTES)
    row = ShareLink(
        project_id=project.id,
        user_id=user.id,
        token_hash=_hash_token(token),
        speaker_ordinal=body.speaker_ordinal,
        password_hash=hash_password(body.password) if body.password else None,
        expires_at=(
            dt.datetime.now(dt.UTC) + dt.timedelta(days=body.expires_in_days)
            if body.expires_in_days
            else None
        ),
    )
    session.add(row)

    await audit.record(
        session,
        "share.created",
        user_id=user.id,
        target_type="project",
        target_id=project.id,
        ip=client_ip(request, settings),
        ip_salt=settings.audit_ip_salt.get_secret_value(),
        correlation_id=problems.correlation_id(request),
        detail={"speaker": body.speaker_ordinal, "expires_in_days": body.expires_in_days},
    )
    await session.commit()
    return _response(row, token=token)


@router.get("/projects/{project_id}/shares")
async def list_shares(project: OwnedProject, session: SessionDep) -> ShareListResponse:
    rows = (
        await session.scalars(
            select(ShareLink)
            .where(ShareLink.project_id == project.id)
            .order_by(ShareLink.created_at.desc())
        )
    ).all()
    return ShareListResponse(items=[_response(r) for r in rows])


@router.delete("/projects/{project_id}/shares/{share_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_share(
    share_id: str,
    request: Request,
    project: OwnedProject,
    user: CurrentUser,
    session: SessionDep,
    settings: SettingsDep,
) -> None:
    """Revoke immediately and permanently.

    Kept as a row rather than deleted, so "this link was revoked on the 14th"
    stays answerable — which is what someone actually wants to know after
    sharing something they regret.
    """
    row = await session.scalar(
        select(ShareLink).where(ShareLink.id == share_id, ShareLink.project_id == project.id)
    )
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="share not found")

    if row.revoked_at is None:
        row.revoked_at = dt.datetime.now(dt.UTC)
        await audit.record(
            session,
            "share.revoked",
            user_id=user.id,
            target_type="share",
            target_id=row.id,
            ip=client_ip(request, settings),
            ip_salt=settings.audit_ip_salt.get_secret_value(),
            correlation_id=problems.correlation_id(request),
        )
        await session.commit()


@router.get("/shared/{token}")
async def open_share(
    token: str,
    request: Request,
    response: Response,
    session: SessionDep,
    settings: SettingsDep,
    password: str | None = None,
) -> SharePublicResponse:
    """Open a share. No account required; the token is the authorisation."""
    limiter = RateLimiter(get_redis(settings), settings)
    enforce(
        await limiter.check(RULES["shared_token"], f"{token[:16]}|{client_ip(request, settings)}")
    )

    # A link shared with three people should not turn up in a search result.
    response.headers["X-Robots-Tag"] = "noindex, nofollow"

    row = await session.scalar(select(ShareLink).where(ShareLink.token_hash == _hash_token(token)))
    # One message for missing, revoked and expired. Distinguishing them tells a
    # stranger whether a token ever existed, which is an oracle worth denying.
    if row is None or not row.is_live:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="This link is not available. It may have been revoked or expired.",
        )

    if row.password_hash is not None and (
        not password or not verify_password(password, row.password_hash)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="This link needs a password.",
        )

    project = await session.get(Project, row.project_id)
    if project is None or project.deleted_at is not None or project.manifest is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="This recording is no longer available."
        )

    row.access_count += 1
    row.last_accessed_at = dt.datetime.now(dt.UTC)
    await session.commit()

    manifest: dict[str, Any] = dict(project.manifest)
    # A share scoped to one speaker must not hand over the others' audio. The
    # manifest is the authorisation for the media, so it is filtered here rather
    # than hidden in the player.
    if row.speaker_ordinal is not None:
        speakers = manifest.get("speakers", [])
        if isinstance(speakers, list):
            manifest["speakers"] = [
                s
                for s in speakers
                if isinstance(s, dict) and s.get("ordinal") == row.speaker_ordinal
            ]

    return SharePublicResponse(
        title=project.title,
        duration_ms=project.duration_ms,
        speaker_count=len(manifest.get("speakers", []) or []),
        speaker_ordinal=row.speaker_ordinal,
        manifest=manifest,
    )
