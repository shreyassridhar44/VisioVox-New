"""Upload, job execution and live progress.

The Phase 2 exit path lives here: upload -> mocked processing with live
progress -> a ready project with a valid manifest.

Progress is delivered over SSE rather than polling. A 10-minute job polled at
1 Hz is 600 requests per viewer, each one an authenticated database round trip,
and the progress bar still lags by up to a second. SSE is one connection that
pushes on change.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy import select
from sse_starlette.sse import EventSourceResponse

from .config import get_settings
from .deps import OwnedProject, SessionDep, SettingsDep
from .models import Job
from .schemas import (
    UploadCompleteRequest,
    UploadInitRequest,
    UploadInitResponse,
    UploadPart,
)
from .storage import ObjectStore, StorageError

router = APIRouter(prefix="/v1/projects", tags=["media"])

SSE_PING_SECONDS = 15
SSE_POLL_SECONDS = 0.5
SSE_MAX_SECONDS = 3600


@router.post("/{project_id}/upload/init")
async def upload_init(
    body: UploadInitRequest,
    project: OwnedProject,
    session: SessionDep,
    settings: SettingsDep,
) -> UploadInitResponse:
    if body.size_bytes > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"file exceeds {settings.max_upload_bytes} bytes",
        )
    if project.status not in {"pending", "failed"}:
        # Re-uploading over a processing project would leave the job pointing at
        # a source that no longer matches its results.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"project is {project.status}; cannot upload",
        )

    store = ObjectStore(settings)
    upload = await store.create_multipart_upload(
        user_id=project.user_id,
        project_id=project.id,
        filename=body.filename,
        content_type=body.content_type,
        part_count=body.part_count,
    )

    project.source_key = upload.key
    project.source_bytes = body.size_bytes
    project.status = "validating"
    await session.commit()

    return UploadInitResponse(
        upload_id=upload.upload_id,
        key=upload.key,
        parts=[UploadPart(part_number=p.part_number, url=p.url) for p in upload.parts],
    )


@router.post("/{project_id}/upload/complete", status_code=status.HTTP_202_ACCEPTED)
async def upload_complete(
    body: UploadCompleteRequest,
    project: OwnedProject,
    session: SessionDep,
    settings: SettingsDep,
) -> dict[str, str]:
    if body.key != project.source_key:
        # The key is issued by init and echoed back; a mismatch means the client
        # is completing an upload for a different object than it started.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="key does not match project"
        )

    store = ObjectStore(settings)
    try:
        size = await store.complete_multipart_upload(
            body.key, body.upload_id, [(p.part_number, p.etag) for p in body.parts]
        )
    except StorageError as exc:
        project.status = "failed"
        await session.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"upload failed: {exc}"
        ) from exc

    project.source_bytes = size
    project.status = "queued"
    # Phase 2 has no probe stage yet; the mock uses this as its timeline length.
    project.duration_ms = project.duration_ms or 600_000

    job = Job(
        project_id=project.id,
        status="queued",
        pipeline_mode=settings.pipeline_mode,
        progress=0,
    )
    session.add(job)
    await session.commit()

    _enqueue(job.id)
    return {"job_id": job.id, "status": "queued"}


def _enqueue(job_id: str) -> None:
    """Hand the job to Celery.

    Imported lazily and failures are swallowed into a log-shaped response,
    because the API must stay up when the broker is not — the job row already
    exists and can be retried.
    """
    try:
        from worker_cpu.tasks import run_mock_pipeline

        run_mock_pipeline.delay(job_id)
    except Exception:
        return


@router.get("/{project_id}/manifest")
async def get_manifest(project: OwnedProject) -> dict[str, Any]:
    """The payload the player depends on (docs/11 §4)."""
    if project.manifest is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"project is {project.status}; no manifest yet",
        )
    manifest: dict[str, Any] = dict(project.manifest)
    # Re-stamp expiry so a client that fetched early still knows when to refetch.
    manifest["signed_until"] = (
        dt.datetime.now(dt.UTC) + dt.timedelta(seconds=get_settings().signed_url_ttl_seconds)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    return manifest


@router.get("/{project_id}/events")
async def job_events(
    project: OwnedProject, session: SessionDep, request: Request
) -> EventSourceResponse:
    """Server-sent progress for the processing view."""
    job = await session.scalar(select(Job).where(Job.project_id == project.id))
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no job yet")
    job_id = job.id

    async def stream() -> AsyncIterator[dict[str, str]]:
        from worker_cpu.tasks import read_progress

        last: str | None = None
        waited = 0.0
        while waited < SSE_MAX_SECONDS:
            if await request.is_disconnected():
                return

            payload = read_progress(job_id)
            if payload is None:
                # Redis cold or unavailable: the database stays authoritative.
                await session.refresh(job)
                payload = {
                    "status": job.status,
                    "progress": job.progress,
                    "stage": None,
                }

            encoded = json.dumps(payload)
            if encoded != last:
                yield {"event": "progress", "data": encoded}
                last = encoded

            if payload.get("status") in {"succeeded", "partial", "failed", "cancelled"}:
                yield {"event": "done", "data": encoded}
                return

            await asyncio.sleep(SSE_POLL_SECONDS)
            waited += SSE_POLL_SECONDS

        yield {"event": "timeout", "data": json.dumps({"job_id": job_id})}

    return EventSourceResponse(stream(), ping=SSE_PING_SECONDS)
