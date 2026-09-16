"""Quotas: the control that actually bounds resource use (docs/15 §9).

Rate limiting and quotas are different tools for different problems, and
conflating them is how a service stays inside every documented limit while
consuming everything it has. A user who respects `10 uploads/hour` can still
queue fifty hour-long videos: the limiter counts *requests*, the quota counts
*work*.

Counters live in Postgres, not Redis. Enforcement has to be atomic across API
replicas and has to survive eviction - a quota that resets when the cache is
under memory pressure is not a quota. The increment is a single upsert, and
because it runs inside the caller's transaction, raising afterwards discards it:
a refused request consumes nothing.

**No per-plan branching yet.** Limits come from configuration and apply to
everyone, because whether plans exist at all is still open (docs/track-w
DECISIONS.md D6.1). Adding a plan dimension now would be structure built for a
decision that has not been made.

Since nothing is rented (docs/28 §D5), exceeding these does not cost money - it
costs the GPU being busy and the disk filling. The controls are the same; what
they protect is queue fairness and disk safety.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from .config import Settings
from .models import Job, Project, UsageCounter, new_id


class QuotaExceededError(Exception):
    """A limit was reached. Carries what a user needs to understand why."""

    def __init__(self, metric: str, used: int, limit: int, resets_in_seconds: int) -> None:
        self.metric = metric
        self.used = used
        self.limit = limit
        self.resets_in_seconds = resets_in_seconds
        super().__init__(f"{metric}: {used}/{limit}")


@dataclass(frozen=True)
class Metric:
    name: str
    period: str  # "day" | "month"
    settings_attr: str
    unit: str

    def limit(self, settings: Settings) -> int:
        value: int = getattr(settings, self.settings_attr)
        return value


UPLOADS = Metric("uploads", "day", "quota_uploads_per_day", "uploads")
MEDIA_SECONDS = Metric("media_seconds", "month", "quota_media_seconds_per_month", "seconds")
GPU_SECONDS = Metric("gpu_seconds", "month", "quota_gpu_seconds_per_month", "seconds")

METRICS = {m.name: m for m in (UPLOADS, MEDIA_SECONDS, GPU_SECONDS)}


def period_key(metric: Metric, now: dt.datetime | None = None) -> str:
    """The calendar bucket this metric counts in.

    A calendar bucket rather than a rolling window keeps the increment to one
    upsert, and makes the stored row directly readable when someone asks why
    they were refused. The cost is a reset that everyone shares - acceptable
    here, and it is what the published limits already describe.
    """
    moment = now or dt.datetime.now(dt.UTC)
    return moment.strftime("%Y-%m-%d") if metric.period == "day" else moment.strftime("%Y-%m")


def seconds_until_reset(metric: Metric, now: dt.datetime | None = None) -> int:
    moment = now or dt.datetime.now(dt.UTC)
    if metric.period == "day":
        nxt = (moment + dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        nxt = (moment.replace(day=1) + dt.timedelta(days=32)).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
    return max(1, int((nxt - moment).total_seconds()))


async def current_usage(session: AsyncSession, user_id: str, metric: Metric) -> int:
    value = await session.scalar(
        select(UsageCounter.value).where(
            UsageCounter.user_id == user_id,
            UsageCounter.metric == metric.name,
            UsageCounter.period_key == period_key(metric),
        )
    )
    return int(value or 0)


async def consume(
    session: AsyncSession,
    user_id: str,
    metric: Metric,
    amount: int,
    settings: Settings,
) -> int:
    """Charge `amount` against a quota, or raise `QuotaExceededError`.

    The upsert and the check are one statement plus one comparison inside the
    caller's transaction, so two concurrent requests cannot both read "under the
    limit" and both proceed - the second one's UPDATE serialises behind the
    first and sees the incremented value.

    Raising leaves the increment to be rolled back with the transaction, so a
    refused request is not billed for the attempt.
    """
    if amount <= 0:
        return await current_usage(session, user_id, metric)

    key = period_key(metric)
    stmt = (
        insert(UsageCounter)
        .values(id=new_id("usc"), user_id=user_id, metric=metric.name, period_key=key, value=amount)
        .on_conflict_do_update(
            index_elements=[UsageCounter.user_id, UsageCounter.metric, UsageCounter.period_key],
            set_={"value": UsageCounter.value + amount},
        )
        .returning(UsageCounter.value)
    )
    used = int(await session.scalar(stmt) or 0)

    limit = metric.limit(settings)
    if used > limit:
        raise QuotaExceededError(metric.name, used, limit, seconds_until_reset(metric))
    return used


async def assert_concurrency_available(
    session: AsyncSession, user_id: str, settings: Settings
) -> None:
    """One user must not be able to fill the queue.

    Counted from `jobs` rather than a counter, because the authoritative answer
    to "how many are running" is the jobs themselves. A separate counter would
    drift every time a worker died without decrementing it - and workers do die.
    """
    running = await session.scalar(
        select(func.count())
        .select_from(Job)
        .join(Project, Project.id == Job.project_id)
        .where(Project.user_id == user_id, Job.status.in_(("queued", "running")))
    )
    limit = settings.quota_concurrent_jobs
    if int(running or 0) >= limit:
        raise QuotaExceededError("concurrent_jobs", int(running or 0), limit, 60)


async def snapshot(
    session: AsyncSession, user_id: str, settings: Settings
) -> dict[str, dict[str, int]]:
    """Everything the account screen needs, so a user sees a limit approaching
    rather than discovering it at the moment of refusal."""
    out: dict[str, dict[str, int]] = {}
    for metric in METRICS.values():
        used = await current_usage(session, user_id, metric)
        out[metric.name] = {
            "used": used,
            "limit": metric.limit(settings),
            "resets_in_seconds": seconds_until_reset(metric),
        }
    return out
