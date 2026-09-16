"""GPU worker: the real pipeline behind a Celery task (docs/28 §W8, ADR-0007).

Runs on its own queue with concurrency 1. That is not tuning — the machine has
one A5000, and a second concurrent job halves throughput while pushing VRAM to
the 24 GB ceiling, which is how a job dies with a CUDA OOM two thirds of the way
through somebody's hour-long recording.

The task owns the parts of a job that are not the pipeline's business: fetching
the source from object storage, publishing the artifacts back, metering the GPU
time against the owner's quota, and leaving the job row in a state that tells
the truth about what happened.
"""

from __future__ import annotations

import datetime as dt
import logging
import shutil
from pathlib import Path
from typing import Any

from celery import Celery
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from visiovox_api.config import get_settings
from visiovox_api.models import Job, Project, UsageCounter, new_id

from .real_pipeline import PipelineError, run_pipeline
from .stagerunner import StageRunner, publish

logger = logging.getLogger(__name__)
settings = get_settings()

celery_app = Celery(
    "visiovox-gpu",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # A worker dying mid-stage must not lose the job; stage rows make the retry
    # resume rather than restart.
    task_acks_late=True,
    # One long job at a time, and no hoarding the queue while it runs.
    worker_prefetch_multiplier=1,
    worker_concurrency=1,
    task_default_queue="gpu",
)

# Celery workers are synchronous, so this uses the sync driver rather than the
# app's async engine. Sharing an async pool across processes is not safe.
_engine = create_engine(settings.database_url.replace("+asyncpg", "+psycopg2"), pool_pre_ping=True)
_Session = sessionmaker(bind=_engine, expire_on_commit=False)


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _meter_gpu_seconds(session: Any, user_id: str, seconds: float) -> None:
    """Charge GPU time to the owner.

    This, not rate limiting, is what bounds how much of the card one account can
    consume — a user inside every request limit can still queue fifty hour-long
    videos. Charged after the fact because the true cost is only known then;
    admission uses the running total from previous jobs.
    """
    from sqlalchemy.dialects.postgresql import insert

    if seconds <= 0:
        return
    key = _now().strftime("%Y-%m")
    amount = round(seconds)
    session.execute(
        insert(UsageCounter)
        .values(
            id=new_id("usc"),
            user_id=user_id,
            metric="gpu_seconds",
            period_key=key,
            value=amount,
        )
        .on_conflict_do_update(
            index_elements=[UsageCounter.user_id, UsageCounter.metric, UsageCounter.period_key],
            set_={"value": UsageCounter.value + amount},
        )
    )


def _publish_artifacts(artifact_dir: Path, project: Project, manifest: dict[str, Any]) -> None:
    """Move the packaged files where the player can fetch them.

    Copied onto the media volume under the project's own prefix rather than
    uploaded through the API: the bytes are already on the same host as the
    object store, and round-tripping them through a presign would double the
    I/O for no benefit (ADR-0012 keeps the API out of the data path either way).
    """
    destination = Path(settings.media_root) / "projects" / project.id
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(artifact_dir, destination)

    base = f"{settings.public_media_base_url.rstrip('/')}/projects/{project.id}/"
    _rewrite_urls(manifest, base)


def _rewrite_urls(node: Any, base: str) -> None:
    """Point every relative artifact name at the served base URL.

    The packager writes bare filenames so it does not have to know how the files
    will be served. This is the one place that decides.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if key in {
                "url",
                "vtt",
                "json",
                "peaks_url",
                "thumbnail_url",
                "audio_url",
            } and isinstance(value, str):
                if not value.startswith(("http://", "https://")):
                    node[key] = base + value.lstrip("/")
            else:
                _rewrite_urls(value, base)
    elif isinstance(node, list):
        for item in node:
            _rewrite_urls(item, base)


@celery_app.task(name="visiovox.run_real_pipeline", bind=True, max_retries=2)
def run_real_pipeline(self: Any, job_id: str) -> dict[str, Any]:
    """Execute the real pipeline for one job."""
    with _Session() as session:
        job = session.get(Job, job_id)
        if job is None:
            return {"status": "missing", "job_id": job_id}

        project = session.get(Project, job.project_id)
        if project is None:
            return {"status": "missing_project", "job_id": job_id}

        job.status = "running"
        job.started_at = job.started_at or _now()
        job.pipeline_mode = "real"
        session.commit()
        publish(settings.redis_url, job_id, {"status": "running", "progress": 0, "stage": None})

        runner = StageRunner(session, job, settings.redis_url)
        source = (
            Path(settings.media_root) / "minio" / settings.s3_bucket / (project.source_key or "")
        )

        if not source.is_file():
            job.status = "failed"
            job.error_code = "SOURCE_MISSING"
            job.error_detail = "the uploaded file could not be found"
            job.finished_at = _now()
            session.commit()
            publish(settings.redis_url, job_id, {"status": "failed", "progress": 0, "stage": None})
            return {"status": "failed", "job_id": job_id, "reason": "source_missing"}

        try:
            output = run_pipeline(
                runner=runner,
                source=source,
                settings=settings,
                project_id=project.id,
            )
        except PipelineError as exc:
            # A clean refusal: the recording cannot yield a result, and the
            # message is written to be shown to a person.
            job.status = "failed"
            job.error_code = "PIPELINE_REFUSED"
            job.error_detail = str(exc)[:500]
            job.finished_at = _now()
            project.status = "failed"
            project.warnings = list(runner.report.warnings)
            session.commit()
            publish(
                settings.redis_url,
                job_id,
                {"status": "failed", "progress": job.progress, "stage": None},
            )
            return {"status": "failed", "job_id": job_id, "reason": str(exc)}
        except Exception as exc:
            logger.exception("pipeline crashed for job %s", job_id)
            job.status = "failed"
            job.error_code = "PIPELINE_ERROR"
            # Never surfaced to a user; the correlation id is the bridge.
            job.error_detail = str(exc)[:500]
            job.finished_at = _now()
            project.status = "failed"
            session.commit()
            publish(
                settings.redis_url,
                job_id,
                {"status": "failed", "progress": job.progress, "stage": None},
            )
            raise

        _publish_artifacts(output.artifact_dir, project, output.manifest)
        shutil.rmtree(output.artifact_dir.parent, ignore_errors=True)

        project.manifest = output.manifest
        project.speaker_count = output.speaker_count
        project.overlap_ratio = output.overlap_ratio
        project.difficulty = output.difficulty
        project.warnings = output.warnings
        project.status = "ready"

        _meter_gpu_seconds(session, project.user_id, output.gpu_seconds)

        # Invariant 8: some speakers recovered is a partial success, not a
        # failure, and the distinction is visible to the user.
        runner.finish(
            status="partial" if output.warnings else "succeeded",
            stage="S9_package",
        )

        return {
            "status": job.status,
            "job_id": job_id,
            "speakers": output.speaker_count,
            "gpu_seconds": round(output.gpu_seconds, 1),
        }


def enqueue(job_id: str) -> None:
    """Hand a job to the GPU queue."""
    run_real_pipeline.apply_async(args=[job_id], queue="gpu")


def queue_depth() -> int:
    """How many jobs are waiting, for the 'you are Nth in line' display.

    Read from the broker rather than the database because the queue is the
    authoritative answer to "when will mine start"; a job row says queued from
    the moment it is created, including while a worker is already running it.
    """
    try:
        import redis

        client = redis.Redis.from_url(settings.celery_broker_url)
        # redis-py's stubs type this as `Awaitable[int] | int` because the same
        # class backs the async client. On the sync client it is always an int.
        depth = client.llen("gpu")
        return depth if isinstance(depth, int) else 0
    except Exception:
        return 0
