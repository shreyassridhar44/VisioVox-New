"""Exports and download (docs/28 §W6).

The player streams; this is the file people keep. Three things the API is
careful about:

- **Bytes never pass through here.** A download is a redirect to a presigned URL
  on the object store, which also means HTTP Range works and a 5 GB download
  resumes after a dropped connection (ADR-0012).
- **The same variant is never rendered twice.** The unique index on
  (project, speaker, kind, quality) makes a repeat request return the existing
  artifact. Re-encoding the same video to the same rung is pure waste on a
  single-GPU machine.
- **Only rungs the source can honestly serve.** A 720p upload does not offer
  1080p, because that is the same picture with more bytes and a worse name.
"""

from __future__ import annotations

import datetime as dt
import logging

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from sqlalchemy import select

from . import audit, problems, quotas
from .deps import CurrentUser, OwnedProject, SessionDep, SettingsDep
from .models import Export
from .ratelimit import RULES, RateLimiter, client_ip, enforce
from .redis_client import get_redis
from .schemas import (
    CreateExportRequest,
    ExportListResponse,
    ExportResponse,
    RenditionOption,
)
from .storage import ObjectStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/projects", tags=["exports"])

EXPORT_TTL_DAYS = 7


def _response(row: Export) -> ExportResponse:
    return ExportResponse(
        id=row.id,
        project_id=row.project_id,
        speaker_ordinal=row.speaker_ordinal,
        kind=row.kind,
        quality=row.quality,
        status=row.status,
        size_bytes=row.size_bytes,
        expires_at=row.expires_at,
        created_at=row.created_at,
    )


@router.get("/{project_id}/export-options")
async def export_options(project: OwnedProject) -> list[RenditionOption]:
    """What this project can actually produce.

    Derived from the source's real height, so the UI lists what exists rather
    than a fixed menu that sometimes lies.
    """
    from pipeline.s10_render import available_renditions

    if project.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"This project is {project.status}; there is nothing to export yet.",
        )

    options = [
        RenditionOption(name=r.name, height=r.height, kind="video")
        for r in available_renditions(project.height)
    ]
    # Audio is always available: it is the thing the product actually produces,
    # and a source with no video still has isolated speakers.
    options.append(RenditionOption(name="audio", height=None, kind="audio"))
    return options


@router.post("/{project_id}/exports", status_code=status.HTTP_202_ACCEPTED)
async def create_export(
    body: CreateExportRequest,
    request: Request,
    project: OwnedProject,
    user: CurrentUser,
    session: SessionDep,
    settings: SettingsDep,
) -> ExportResponse:
    limiter = RateLimiter(get_redis(settings), settings)
    enforce(await limiter.check(RULES["exports"], user.id))

    if project.status != "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"This project is {project.status}; there is nothing to export yet.",
        )

    from pipeline.s10_render import available_renditions

    if body.kind == "video":
        allowed = {r.name for r in available_renditions(project.height)}
        if body.quality not in allowed:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"This recording cannot be exported at {body.quality}. "
                    f"Available: {', '.join(sorted(allowed)) or 'none'}."
                ),
            )

    # Asking twice returns the same artifact rather than re-encoding it.
    existing = await session.scalar(
        select(Export).where(
            Export.project_id == project.id,
            Export.speaker_ordinal == body.speaker_ordinal,
            Export.kind == body.kind,
            Export.quality == body.quality,
        )
    )
    if existing is not None and existing.status in {"queued", "running", "ready"}:
        return _response(existing)

    row = existing or Export(
        project_id=project.id,
        user_id=user.id,
        speaker_ordinal=body.speaker_ordinal,
        kind=body.kind,
        quality=body.quality,
    )
    row.status = "queued"
    row.error_detail = None
    row.expires_at = dt.datetime.now(dt.UTC) + dt.timedelta(days=EXPORT_TTL_DAYS)
    session.add(row)

    await audit.record(
        session,
        "export.requested",
        user_id=user.id,
        target_type="project",
        target_id=project.id,
        ip=client_ip(request, settings),
        ip_salt=settings.audit_ip_salt.get_secret_value(),
        correlation_id=problems.correlation_id(request),
        detail={"kind": body.kind, "quality": body.quality},
    )
    await session.commit()

    _enqueue_export(row.id)
    return _response(row)


def _enqueue_export(export_id: str) -> None:
    """Hand the render to the GPU queue's worker.

    Swallowed on failure for the same reason as job enqueue: the row exists and
    the API must stay up when the broker does not.
    """
    try:
        from worker_gpu.tasks import enqueue_export

        enqueue_export(export_id)
    except Exception:
        logger.warning("could not enqueue export %s", export_id, exc_info=True)


@router.get("/{project_id}/exports")
async def list_exports(project: OwnedProject, session: SessionDep) -> ExportListResponse:
    rows = (
        await session.scalars(
            select(Export).where(Export.project_id == project.id).order_by(Export.created_at.desc())
        )
    ).all()
    return ExportListResponse(items=[_response(r) for r in rows])


@router.get("/{project_id}/exports/{export_id}/download")
async def download_export(
    export_id: str,
    project: OwnedProject,
    session: SessionDep,
    settings: SettingsDep,
) -> RedirectResponse:
    """Redirect to a short-lived presigned URL.

    A redirect rather than a proxied stream: the object store handles Range, so
    a large download resumes after a dropped connection, and the API never
    occupies a worker for the length of a transfer (ADR-0012).
    """
    row = await session.scalar(
        select(Export).where(Export.id == export_id, Export.project_id == project.id)
    )
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="export not found")

    if row.status != "ready" or row.storage_key is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"This export is {row.status}.",
        )

    if row.expires_at is not None and row.expires_at < dt.datetime.now(dt.UTC):
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="This export has expired. Request it again to re-render it.",
        )

    filename = f"{project.title[:60] or 'visiovox'}"
    if row.speaker_ordinal is not None:
        filename += f"-speaker-{row.speaker_ordinal}"
    if row.quality:
        filename += f"-{row.quality}"
    filename += ".m4a" if row.kind == "audio" else ".mp4"

    store = ObjectStore(settings)
    url = await store.presign_get(row.storage_key, download_as=filename)
    return RedirectResponse(url, status_code=status.HTTP_307_TEMPORARY_REDIRECT)


@router.get("/{project_id}/usage")
async def project_usage(
    project: OwnedProject, user: CurrentUser, session: SessionDep, settings: SettingsDep
) -> dict[str, dict[str, int]]:
    """Quota position, so a limit is visible before it is hit rather than after."""
    _ = project
    return await quotas.snapshot(session, user.id, settings)
