"""Stage bookkeeping shared by every pipeline mode (ADR-0007, invariants 7 & 8).

The mock worker and the real worker disagree about everything except how a job
is *recorded*: a stage row per attempt, keyed `(job_id, stage, version)`, a
progress publish on each transition, and a job that ends `partial` rather than
`failed` when some speakers came out.

Keeping that here means the two paths cannot drift in the way that matters —
a resumed real job and a resumed mock job behave identically, because it is the
same code deciding what to skip.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from visiovox_api.models import Job, JobStage

logger = logging.getLogger(__name__)

PROGRESS_KEY = "visiovox:job:{job_id}:progress"
PROGRESS_TTL_SECONDS = 3600
STAGE_VERSION = "1.0.0"


def now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def publish(redis_url: str, job_id: str, payload: dict[str, Any]) -> None:
    """Mirror progress into Redis. Never fails a job.

    The database stays authoritative; this is what lets the SSE endpoint stream
    without a per-client Postgres round trip, and the endpoint falls back to the
    row when the cache is cold.
    """
    try:
        import redis

        client = redis.Redis.from_url(redis_url)
        key = PROGRESS_KEY.format(job_id=job_id)
        encoded = json.dumps(payload)
        client.setex(key, PROGRESS_TTL_SECONDS, encoded)
        client.publish(key, encoded)
    except Exception:
        logger.debug("progress publish failed for %s", job_id, exc_info=True)


class StageFailedError(Exception):
    """A stage could not complete. Whether that fails the job is the caller's call."""


@dataclass
class StagePlan:
    """One stage, and the share of the bar it is worth."""

    stage: str
    ordinal: int
    progress_after: int


@dataclass
class RunReport:
    """What happened, for the job row and for metering."""

    degraded: bool = False
    warnings: list[str] = field(default_factory=list)
    gpu_seconds: float = 0.0
    stage_seconds: dict[str, float] = field(default_factory=dict)

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)
        self.degraded = True


class StageRunner:
    """Runs stages against one job, recording each and reporting progress."""

    def __init__(
        self,
        session: Session,
        job: Job,
        redis_url: str,
        *,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.session = session
        self.job = job
        self.redis_url = redis_url
        self.report = RunReport()
        self._on_progress = on_progress

    def _emit(self, payload: dict[str, Any]) -> None:
        publish(self.redis_url, self.job.id, payload)
        if self._on_progress:
            self._on_progress(payload)

    def already_done(self, stage: str) -> bool:
        """Invariant 7: a stage recorded as succeeded at this version is skipped.

        This is the whole resume story. A worker killed mid-job restarts at the
        stage it died in rather than re-decoding the video, which for a long
        recording is the difference between minutes and an hour.
        """
        row = self.session.scalar(
            select(JobStage).where(
                JobStage.job_id == self.job.id,
                JobStage.stage == stage,
                JobStage.version == STAGE_VERSION,
            )
        )
        return row is not None and row.status == "succeeded"

    @contextmanager
    def stage(self, plan: StagePlan, *, gpu: bool = False) -> Iterator[RunReport]:
        """Record one stage attempt around the work.

        Marks running, times it, and marks succeeded or failed. `gpu=True` adds
        the elapsed time to the GPU meter — which, not rate limiting, is what
        actually bounds how much of the card one account can consume.
        """
        row = self.session.scalar(
            select(JobStage).where(
                JobStage.job_id == self.job.id,
                JobStage.stage == plan.stage,
                JobStage.version == STAGE_VERSION,
            )
        )
        if row is None:
            row = JobStage(
                job_id=self.job.id,
                stage=plan.stage,
                ordinal=plan.ordinal,
                version=STAGE_VERSION,
            )
        row.status = "running"
        # server_default only applies at INSERT, so a freshly constructed row
        # still holds None here until it round-trips through the database.
        row.attempts = (row.attempts or 0) + 1
        row.started_at = now()
        self.session.add(row)
        self.session.commit()

        self._emit({"status": "running", "progress": self.job.progress, "stage": plan.stage})

        started = time.perf_counter()
        try:
            yield self.report
        except Exception as exc:
            elapsed = time.perf_counter() - started
            row.status = "failed"
            row.duration_ms = int(elapsed * 1000)
            row.finished_at = now()
            # Truncated, and never surfaced to a user: a stage failure message
            # can carry file paths and library internals.
            row.detail = str(exc)[:500]
            self.session.commit()
            logger.exception("stage %s failed for job %s", plan.stage, self.job.id)
            raise StageFailedError(plan.stage) from exc

        elapsed = time.perf_counter() - started
        self.report.stage_seconds[plan.stage] = elapsed
        if gpu:
            self.report.gpu_seconds += elapsed

        row.status = "degraded" if self.report.degraded else "succeeded"
        # A degraded stage still counts as done: invariant 8 says partial
        # results ship, and re-running it on resume would not improve them.
        if row.status == "degraded":
            row.status = "succeeded"
        row.duration_ms = int(elapsed * 1000)
        row.finished_at = now()
        row.warnings = list(self.report.warnings)

        self.job.progress = plan.progress_after
        self.session.commit()

        self._emit({"status": "running", "progress": self.job.progress, "stage": plan.stage})

    def finish(self, *, status: str, stage: str) -> None:
        self.job.status = status
        self.job.progress = 100
        self.job.finished_at = now()
        self.session.commit()
        self._emit({"status": status, "progress": 100, "stage": stage})
