"""Shared FastAPI dependencies: current user, and ownership-checked project lookup.

`owned_project` exists because invariant 4 says every artifact access is
ownership-checked server-side, and IDOR is the highest-impact vulnerability in
this system. Making it a dependency rather than a line of code in each handler
means a new endpoint gets the check by declaring its parameter, and forgetting
it is visible in the signature rather than buried in the body.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Path, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .config import Settings, get_settings
from .db import get_session
from .models import Project, User
from .models import Session as AuthSession
from .security import InvalidTokenError, decode_access_token

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def _bearer(request: Request) -> str:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


async def current_user(request: Request, session: SessionDep, settings: SettingsDep) -> User:
    token = _bearer(request)
    try:
        claims = decode_access_token(token, settings.auth_secret.get_secret_value())
    except InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    # A revoked session must stop working immediately, not when the access token
    # happens to expire. Reuse detection revokes sessions, so skipping this
    # check would leave a stolen token usable for its full TTL.
    auth_session = await session.get(AuthSession, claims.session_id)
    if auth_session is None or auth_session.revoked_at is not None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="session revoked")

    user = await session.get(User, claims.user_id)
    if user is None or user.deleted_at is not None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="account unavailable")
    return user


CurrentUser = Annotated[User, Depends(current_user)]


async def owned_project(
    project_id: Annotated[str, Path(pattern=r"^prj_[0-9A-HJKMNP-TV-Z]{26}$")],
    user: CurrentUser,
    session: SessionDep,
) -> Project:
    """Fetch a project, or 404 if it is not this user's.

    404 rather than 403 on purpose: 403 confirms the id exists, which turns the
    endpoint into an existence oracle for other people's projects.
    """
    project = await session.scalar(
        select(Project).where(
            Project.id == project_id,
            Project.user_id == user.id,
            Project.deleted_at.is_(None),
        )
    )
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="project not found")
    return project


OwnedProject = Annotated[Project, Depends(owned_project)]
