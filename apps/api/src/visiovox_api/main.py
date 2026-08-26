"""VisioVox control-plane API.

Routes are thin: validation via Pydantic, authorisation via dependencies, and
the actual work in service modules. That keeps the ownership check (invariant
4) visible in each handler's signature rather than buried in its body.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from . import auth_service
from .config import get_settings
from .db import dispose_engine
from .deps import CurrentUser, OwnedProject, SessionDep, SettingsDep
from .models import Job, JobStage, Project
from .routes_media import router as media_router
from .schemas import (
    CreateProjectRequest,
    ErrorResponse,
    JobResponse,
    JobStageResponse,
    LoginRequest,
    ProjectListResponse,
    ProjectResponse,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)

API_PREFIX = "/v1"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    await dispose_engine()


app = FastAPI(
    title="VisioVox API",
    version="0.1.0",
    description="Control plane for audio-visual target speaker extraction.",
    lifespan=lifespan,
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_settings.next_public_app_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

health = APIRouter(tags=["health"])
auth = APIRouter(prefix=f"{API_PREFIX}/auth", tags=["auth"])
projects = APIRouter(prefix=f"{API_PREFIX}/projects", tags=["projects"])


@health.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@health.get("/readyz")
async def readyz(session: SessionDep) -> dict[str, str]:
    """Readiness means the database answers, not merely that the process is up."""
    await session.execute(select(1))
    return {"status": "ready"}


# --------------------------------------------------------------------------
# auth
# --------------------------------------------------------------------------


def _token_response(pair: auth_service.TokenPair) -> TokenResponse:
    return TokenResponse(
        access_token=pair.access_token,
        expires_at=pair.access_expires_at,
        refresh_token=pair.refresh_token,
        refresh_expires_at=pair.refresh_expires_at,
    )


@auth.post("/register", status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterRequest, session: SessionDep, settings: SettingsDep
) -> TokenResponse:
    try:
        user = await auth_service.register_user(
            session, body.email, body.password, body.display_name
        )
    except auth_service.AuthError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    pair = await auth_service.login(
        session,
        body.email,
        body.password,
        secret=settings.auth_secret.get_secret_value(),
        access_ttl_seconds=settings.access_token_ttl_seconds,
        refresh_ttl_days=settings.refresh_token_ttl_days,
    )
    await session.commit()
    _ = user
    return _token_response(pair)


@auth.post("/login")
async def login(
    body: LoginRequest, request: Request, session: SessionDep, settings: SettingsDep
) -> TokenResponse:
    try:
        pair = await auth_service.login(
            session,
            body.email,
            body.password,
            secret=settings.auth_secret.get_secret_value(),
            access_ttl_seconds=settings.access_token_ttl_seconds,
            refresh_ttl_days=settings.refresh_token_ttl_days,
            user_agent=request.headers.get("user-agent"),
            ip=request.client.host if request.client else None,
        )
    except auth_service.AuthError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    await session.commit()
    return _token_response(pair)


@auth.post("/refresh")
async def refresh(
    body: RefreshRequest, session: SessionDep, settings: SettingsDep
) -> TokenResponse:
    try:
        pair = await auth_service.refresh(
            session,
            body.refresh_token,
            secret=settings.auth_secret.get_secret_value(),
            access_ttl_seconds=settings.access_token_ttl_seconds,
            refresh_ttl_days=settings.refresh_token_ttl_days,
        )
    except auth_service.TokenReuseError as exc:
        # Commit: the revocation must persist even though the request fails.
        await session.commit()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    except auth_service.AuthError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    await session.commit()
    return _token_response(pair)


@auth.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(body: RefreshRequest, session: SessionDep) -> None:
    await auth_service.logout(session, body.refresh_token)
    await session.commit()


@auth.get("/me")
async def me(user: CurrentUser) -> UserResponse:
    return UserResponse(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        plan=user.plan,
        retention_days=user.retention_days,
        allow_training_use=user.allow_training_use,
        persist_voiceprints=user.persist_voiceprints,
        created_at=user.created_at,
    )


# --------------------------------------------------------------------------
# projects
# --------------------------------------------------------------------------


def _project_response(p: Project) -> ProjectResponse:
    return ProjectResponse(
        id=p.id,
        title=p.title,
        status=p.status,
        duration_ms=p.duration_ms,
        has_video=p.has_video,
        speaker_count=p.speaker_count,
        overlap_ratio=p.overlap_ratio,
        difficulty=p.difficulty,
        warnings=list(p.warnings or []),
        created_at=p.created_at,
    )


@projects.post("", status_code=status.HTTP_201_CREATED)
async def create_project(
    body: CreateProjectRequest, user: CurrentUser, session: SessionDep
) -> ProjectResponse:
    if not body.rights_attested:
        # FR-UPL-08: refuse rather than record a false attestation.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="rights attestation is required",
        )
    project = Project(user_id=user.id, title=body.title, status="pending")
    session.add(project)
    await session.commit()
    return _project_response(project)


@projects.get("")
async def list_projects(
    user: CurrentUser,
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> ProjectListResponse:
    rows = (
        await session.scalars(
            select(Project)
            .where(Project.user_id == user.id, Project.deleted_at.is_(None))
            .order_by(Project.created_at.desc())
            .limit(limit)
        )
    ).all()
    return ProjectListResponse(items=[_project_response(p) for p in rows])


@projects.get("/{project_id}")
async def get_project(project: OwnedProject) -> ProjectResponse:
    return _project_response(project)


@projects.delete("/{project_id}", status_code=status.HTTP_202_ACCEPTED)
async def delete_project(project: OwnedProject, session: SessionDep) -> dict[str, str]:
    project.deleted_at = dt.datetime.now(dt.UTC)
    await session.commit()
    return {"status": "deletion_scheduled", "project_id": project.id}


@projects.get("/{project_id}/job")
async def get_job(project: OwnedProject, session: SessionDep) -> JobResponse:
    job = await session.scalar(select(Job).where(Job.project_id == project.id))
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no job yet")
    stages = (
        await session.scalars(
            select(JobStage).where(JobStage.job_id == job.id).order_by(JobStage.ordinal)
        )
    ).all()
    return JobResponse(
        id=job.id,
        project_id=job.project_id,
        status=job.status,
        progress=job.progress,
        pipeline_mode=job.pipeline_mode,
        error_code=job.error_code,
        stages=[
            JobStageResponse(
                stage=s.stage,
                status=s.status,
                ordinal=s.ordinal,
                duration_ms=s.duration_ms,
                warnings=list(s.warnings or []),
            )
            for s in stages
        ],
    )


app.include_router(health)
app.include_router(auth)
app.include_router(projects)
app.include_router(media_router)
