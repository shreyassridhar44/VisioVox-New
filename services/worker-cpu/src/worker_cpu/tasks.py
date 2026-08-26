"""Celery job execution and the job state machine (ADR-0007).

Celery over Temporal because this is a solo-maintained project and Celery is
boring technology with one fewer service to operate.

Progress is written to the database as each stage completes, and mirrored into
Redis so the SSE endpoint can stream without polling Postgres per client. The
database stays authoritative; Redis is a cache that may be cold, and the
endpoint falls back to reading the row.

Stage rows are keyed (job_id, stage, version) and a stage already recorded as
succeeded is skipped, so a retried job resumes instead of redoing work
(invariant 7). Partial failure marks the job `partial` rather than `failed` —
2 of 3 speakers beats nothing (invariant 8).
"""

from __future__ import annotations

import datetime as dt
import json
import time
from typing import Any

from celery import Celery
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from visiovox_api.config import get_settings
from visiovox_api.models import Job, JobStage, Project

from .mock_pipeline import build_manifest, plan_stages

settings = get_settings()

celery_app = Celery(
    "visiovox",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,  # a worker dying mid-stage must not lose the job
    worker_prefetch_multiplier=1,  # long tasks: do not hoard the queue
    task_track_started=True,
)

# Celery workers are synchronous, so this uses the sync driver rather than the
# app's async engine. Sharing an async pool across processes is not safe.
_sync_engine = create_engine(
    settings.database_url.replace("+asyncpg", "+psycopg2"), pool_pre_ping=True
)
_SyncSession = sessionmaker(bind=_sync_engine, expire_on_commit=False)

PROGRESS_KEY = "visiovox:job:{job_id}:progress"
PROGRESS_TTL_SECONDS = 3600


def _redis() -> Any:
    import redis

    return redis.Redis.from_url(settings.redis_url)


def _publish(job_id: str, payload: dict[str, Any]) -> None:
    """Best-effort progress mirror. Never fail a job because Redis is down."""
    try:
        client = _redis()
        client.setex(PROGRESS_KEY.format(job_id=job_id), PROGRESS_TTL_SECONDS, json.dumps(payload))
        client.publish(PROGRESS_KEY.format(job_id=job_id), json.dumps(payload))
    except Exception:
        return


def read_progress(job_id: str) -> dict[str, Any] | None:
    try:
        raw = _redis().get(PROGRESS_KEY.format(job_id=job_id))
    except Exception:
        return None
    if raw is None:
        return None
    decoded: dict[str, Any] = json.loads(raw)
    return decoded


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


@celery_app.task(name="visiovox.run_mock_pipeline", bind=True, max_retries=3)
def run_mock_pipeline(self: Any, job_id: str, speed: float = 0.02) -> dict[str, Any]:
    """Execute the mock pipeline for one job.

    `speed` scales the simulated stage durations; the default runs a 10-minute
    video in a few seconds so tests and local development are not gated on
    realtime, while preserving the *relative* stage weights that make the
    progress bar behave like the real thing.
    """
    with _SyncSession() as session:
        job = session.get(Job, job_id)
        if job is None:
            return {"status": "missing", "job_id": job_id}

        project = session.get(Project, job.project_id)
        if project is None:
            return {"status": "missing_project", "job_id": job_id}

        duration_ms = project.duration_ms or 600_000

        job.status = "running"
        job.started_at = job.started_at or _now()
        session.commit()
        _publish(job_id, {"status": "running", "progress": 0, "stage": None})

        degraded = False
        for plan in plan_stages(duration_ms):
            existing = session.scalar(
                select(JobStage).where(
                    JobStage.job_id == job_id,
                    JobStage.stage == plan.stage,
                    JobStage.version == "1.0.0",
                )
            )
            # Invariant 7: a completed stage is not redone on resume.
            if existing is not None and existing.status == "succeeded":
                continue

            row = existing or JobStage(
                job_id=job_id, stage=plan.stage, ordinal=plan.ordinal, version="1.0.0"
            )
            row.status = "running"
            # server_default only applies at INSERT, so a freshly constructed
            # row still has None here until it round-trips through the database.
            row.attempts = (row.attempts or 0) + 1
            row.started_at = _now()
            session.add(row)
            session.commit()

            time.sleep(max(0.0, plan.duration_ms / 1000.0 * speed))

            row.status = "succeeded"
            row.duration_ms = plan.duration_ms
            row.finished_at = _now()
            job.progress = plan.progress_after
            session.commit()

            _publish(
                job_id,
                {"status": "running", "progress": job.progress, "stage": plan.stage},
            )

        manifest = build_manifest(project.id, duration_ms, has_video=project.has_video)
        speakers = manifest.get("speakers", [])
        warnings = manifest.get("warnings", [])

        project.manifest = manifest
        project.speaker_count = len(speakers) if isinstance(speakers, list) else None
        project.overlap_ratio = manifest.get("overlap_ratio")  # type: ignore[assignment]
        project.difficulty = manifest.get("difficulty")  # type: ignore[assignment]
        project.warnings = list(warnings) if isinstance(warnings, list) else []
        project.status = "ready"

        job.status = "partial" if degraded else "succeeded"
        job.progress = 100
        job.finished_at = _now()
        session.commit()

        _publish(job_id, {"status": job.status, "progress": 100, "stage": "S9_package"})
        return {"status": job.status, "job_id": job_id, "speakers": project.speaker_count}
