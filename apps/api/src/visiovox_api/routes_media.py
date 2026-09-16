"""Upload, job execution and live progress.

The upload path is built for files far larger than a browser can hold in memory
and for connections that drop (docs/28 §W2):

- **Direct to storage.** Bytes never pass through the API; presigned part URLs
  go straight to the object store (ADR-0012).
- **Batched part URLs.** Fifty at a time rather than thousands up front, which
  also gives a resumed client a way to obtain fresh URLs for the parts it still
  needs instead of restarting.
- **Server-side part state.** A refreshed tab resumes; without this the part
  list lives only in the tab that began the upload.
- **Reserved headroom.** An init books the disk it will eventually need, so ten
  concurrent uploads cannot each be told yes because each one individually fits.

Progress is delivered over SSE rather than polling. A 10-minute job polled at
1 Hz is 600 authenticated database round trips per viewer, and the bar still lags
by up to a second.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy import select
from sse_starlette.sse import EventSourceResponse

from . import audit, problems, quotas, uploads
from .config import get_settings
from .deps import CurrentUser, OwnedProject, SessionDep, SettingsDep
from .models import Job, UploadSession
from .ratelimit import RULES, RateLimiter, client_ip, enforce
from .redis_client import get_redis
from .schemas import (
    LimitsResponse,
    UploadCompleteRequest,
    UploadInitRequest,
    UploadInitResponse,
    UploadPart,
    UploadPartsRequest,
    UploadPartsResponse,
    UploadStatusResponse,
)
from .storage import ObjectStore, StorageError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/projects", tags=["media"])
limits_router = APIRouter(prefix="/v1", tags=["limits"])

SSE_PING_SECONDS = 15
SSE_POLL_SECONDS = 0.5
SSE_MAX_SECONDS = 3600

_settings = get_settings()
_limiter = RateLimiter(get_redis(_settings), _settings)


# --------------------------------------------------------------------------
# limits
# --------------------------------------------------------------------------


@limits_router.get("/limits")
async def get_limits(
    user: CurrentUser, session: SessionDep, settings: SettingsDep
) -> LimitsResponse:
    """What this account may upload right now, and why.

    The upload screen shows these numbers before a file is chosen (docs/28 §D2).
    They move with free space and with other uploads in flight, so they are
    computed per request rather than read from configuration.
    """
    await uploads.expire_stale(session)
    await session.commit()

    return LimitsResponse(
        max_upload_bytes=await uploads.max_upload_bytes_now(session, settings),
        max_duration_seconds=settings.max_duration_seconds,
        max_speakers=settings.max_speakers,
        available_bytes=await uploads.available_bytes(session, settings),
        reserved_bytes=await uploads.reserved_bytes(session),
        quotas=await quotas.snapshot(session, user.id, settings),
    )


# --------------------------------------------------------------------------
# upload
# --------------------------------------------------------------------------


@router.post("/{project_id}/upload/init")
async def upload_init(
    body: UploadInitRequest,
    request: Request,
    project: OwnedProject,
    session: SessionDep,
    settings: SettingsDep,
) -> UploadInitResponse:
    enforce(await _limiter.check(RULES["upload_init"], project.user_id))

    if project.status not in {"pending", "failed"}:
        # Re-uploading over a processing project would leave the job pointing at
        # a source that no longer matches its results.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"This project is {project.status} and cannot accept a new upload.",
        )

    # Reclaim anything abandoned before deciding whether this one fits, so a
    # dead upload from yesterday does not refuse a live one today.
    await uploads.expire_stale(session)

    await quotas.assert_concurrency_available(session, project.user_id, settings)
    try:
        await uploads.assert_fits(session, settings, body.size_bytes)
    except uploads.UploadError as exc:
        await audit.record(
            session,
            audit.UPLOAD_REJECTED,
            user_id=project.user_id,
            target_type="project",
            target_id=project.id,
            outcome="denied",
            ip=client_ip(request, settings),
            ip_salt=settings.audit_ip_salt.get_secret_value(),
            correlation_id=problems.correlation_id(request),
            detail={"size_bytes": body.size_bytes},
        )
        await session.commit()
        raise HTTPException(
            status_code=413,  # not the starlette constant: its name changed between versions
            detail=str(exc),
        ) from exc

    # Charged here rather than at completion: an upload that is started and
    # abandoned still costs storage and a slot.
    await quotas.consume(session, project.user_id, quotas.UPLOADS, 1, settings)

    plan = uploads.plan_parts(body.size_bytes)
    store = ObjectStore(settings)
    created = await store.create_multipart_upload(
        user_id=project.user_id,
        project_id=project.id,
        filename=body.filename,
        content_type=body.content_type,
    )

    upload = UploadSession(
        user_id=project.user_id,
        project_id=project.id,
        storage_key=created.key,
        upload_id=created.upload_id,
        filename=body.filename,
        content_type=body.content_type,
        size_bytes=body.size_bytes,
        part_size_bytes=plan.part_size_bytes,
        part_count=plan.part_count,
        reserved_bytes=uploads.reservation_for(body.size_bytes, settings),
        completed_parts=[],
        status="active",
        expires_at=uploads.new_expiry(),
    )
    session.add(upload)

    project.source_key = created.key
    project.source_bytes = body.size_bytes
    project.status = "validating"

    await audit.record(
        session,
        audit.UPLOAD_INIT,
        user_id=project.user_id,
        target_type="project",
        target_id=project.id,
        ip=client_ip(request, settings),
        ip_salt=settings.audit_ip_salt.get_secret_value(),
        correlation_id=problems.correlation_id(request),
        detail={"size_bytes": body.size_bytes, "parts": plan.part_count},
    )
    await session.commit()

    first = uploads.batch_for(plan.part_count, after=0)
    presigned = await store.presign_parts(created.key, created.upload_id, first)

    return UploadInitResponse(
        upload_id=created.upload_id,
        key=created.key,
        part_size_bytes=plan.part_size_bytes,
        part_count=plan.part_count,
        parts=[UploadPart(part_number=p.part_number, url=p.url) for p in presigned],
    )


async def _load_session(
    session: SessionDep, project: OwnedProject, upload_id: str
) -> UploadSession:
    row = await session.scalar(
        select(UploadSession).where(
            UploadSession.upload_id == upload_id,
            UploadSession.project_id == project.id,
        )
    )
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="upload not found")
    return row


@router.post("/{project_id}/upload/{upload_id}/parts")
async def upload_parts(
    body: UploadPartsRequest,
    upload_id: str,
    project: OwnedProject,
    session: SessionDep,
    settings: SettingsDep,
) -> UploadPartsResponse:
    """Record confirmed parts and hand back the next batch of URLs.

    One call does both so a client uploading a large file makes one round trip
    per batch rather than two, and so progress is durable at batch granularity
    rather than only at the end.
    """
    upload = await _load_session(session, project, upload_id)
    if upload.status != "active":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"This upload is {upload.status}.",
        )

    if body.completed:
        upload.completed_parts = uploads.merge_parts(
            list(upload.completed_parts),
            [p.model_dump() for p in body.completed],
        )
    # Any activity means the client is alive; hold the reservation open.
    upload.expires_at = uploads.new_expiry()
    await session.commit()

    highest = max((int(p["part_number"]) for p in upload.completed_parts), default=0)
    wanted = uploads.batch_for(upload.part_count, after=max(highest, body.after))
    store = ObjectStore(settings)
    presigned = await store.presign_parts(upload.storage_key, upload.upload_id, wanted)

    return UploadPartsResponse(
        parts=[UploadPart(part_number=p.part_number, url=p.url) for p in presigned],
        completed_count=len(upload.completed_parts),
        part_count=upload.part_count,
    )


@router.get("/{project_id}/upload/{upload_id}")
async def upload_status(
    upload_id: str, project: OwnedProject, session: SessionDep
) -> UploadStatusResponse:
    """What a resuming client needs: which parts are already in the store.

    This is what makes a browser refresh cost nothing instead of restarting a
    multi-gigabyte transfer.
    """
    upload = await _load_session(session, project, upload_id)
    return UploadStatusResponse(
        upload_id=upload.upload_id,
        key=upload.storage_key,
        status=upload.status,
        size_bytes=upload.size_bytes,
        part_size_bytes=upload.part_size_bytes,
        part_count=upload.part_count,
        completed_parts=sorted(int(p["part_number"]) for p in upload.completed_parts),
        expires_at=upload.expires_at,
    )


@router.post("/{project_id}/upload/{upload_id}/abort", status_code=status.HTTP_204_NO_CONTENT)
async def upload_abort(
    upload_id: str, project: OwnedProject, session: SessionDep, settings: SettingsDep
) -> None:
    """Cancel and release the reservation.

    Aborting at the object store matters as much as the row: incomplete
    multipart uploads are billed storage that no bucket listing shows, which is
    a cost leak that stays invisible until the bill arrives.
    """
    upload = await _load_session(session, project, upload_id)
    if upload.status == "active":
        try:
            await ObjectStore(settings).abort_multipart_upload(upload.storage_key, upload.upload_id)
        except Exception:
            # Logged rather than silent: a failed abort leaves incomplete parts
            # that are billed storage no bucket listing shows. The row is still
            # released - holding a reservation for an upload the client has
            # walked away from helps nobody.
            logger.warning("abort_multipart_upload failed for %s", upload.upload_id, exc_info=True)
        upload.status = "aborted"
        upload.reserved_bytes = 0
        if project.status == "validating":
            project.status = "pending"
        await session.commit()


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

    upload = await _load_session(session, project, body.upload_id)
    parts = uploads.merge_parts(list(upload.completed_parts), [p.model_dump() for p in body.parts])
    if len(parts) != upload.part_count:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Expected {upload.part_count} parts but have {len(parts)}. "
                "Fetch the upload status and re-send the missing parts."
            ),
        )

    store = ObjectStore(settings)
    try:
        size = await store.complete_multipart_upload(
            body.key,
            body.upload_id,
            [(int(str(p["part_number"])), str(p["etag"])) for p in parts],
        )
    except StorageError as exc:
        project.status = "failed"
        upload.status = "aborted"
        upload.reserved_bytes = 0
        await session.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"upload failed: {exc}"
        ) from exc

    upload.status = "completed"
    # The bytes are real now and accounted for by the project, not by a booking.
    upload.reserved_bytes = 0
    upload.completed_parts = parts

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
    await audit.record(
        session,
        audit.UPLOAD_COMPLETED,
        user_id=project.user_id,
        target_type="project",
        target_id=project.id,
        detail={"size_bytes": size, "parts": len(parts)},
    )
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
