"""Quotas and the audit log (docs/15 §9, §10).

Against a real Postgres, because the properties that matter are database
properties: the unique index that makes the upsert atomic, the rollback that
stops a refused request from being charged, and the FK that nulls `user_id` on
account deletion while keeping the audit row.

A mock would pass all of these while the schema was wrong.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from visiovox_api import audit, quotas
from visiovox_api.config import Settings, get_settings
from visiovox_api.models import AuditEvent, Job, Project, User
from visiovox_api.quotas import GPU_SECONDS, UPLOADS, QuotaExceededError

pytestmark = pytest.mark.integration

DB_URL = get_settings().database_url


async def _reachable() -> bool:
    try:
        engine = create_async_engine(DB_URL)
        async with engine.connect() as conn:
            await conn.execute(text("select 1"))
        await engine.dispose()
    except Exception:
        return False
    return True


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    if not await _reachable():
        pytest.skip("Postgres not reachable; run `make dev`")
    engine = create_async_engine(DB_URL)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as s:
        yield s
    await engine.dispose()


@pytest_asyncio.fixture
async def user(session: AsyncSession) -> User:
    u = User(email=f"quota-{uuid.uuid4().hex[:12]}@example.com")
    session.add(u)
    await session.commit()
    return u


def _settings(**overrides: object) -> Settings:
    return Settings(**overrides)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# counting
# --------------------------------------------------------------------------


async def test_usage_starts_at_zero(session: AsyncSession, user: User) -> None:
    assert await quotas.current_usage(session, user.id, UPLOADS) == 0


async def test_consume_accumulates(session: AsyncSession, user: User) -> None:
    s = _settings(quota_uploads_per_day=10)
    assert await quotas.consume(session, user.id, UPLOADS, 1, s) == 1
    assert await quotas.consume(session, user.id, UPLOADS, 2, s) == 3
    assert await quotas.current_usage(session, user.id, UPLOADS) == 3


async def test_consume_raises_at_the_limit(session: AsyncSession, user: User) -> None:
    s = _settings(quota_uploads_per_day=2)
    await quotas.consume(session, user.id, UPLOADS, 2, s)
    with pytest.raises(QuotaExceededError) as exc:
        await quotas.consume(session, user.id, UPLOADS, 1, s)
    assert exc.value.metric == "uploads"
    assert exc.value.limit == 2
    assert exc.value.resets_in_seconds > 0


async def test_a_refused_request_is_not_charged(session: AsyncSession, user: User) -> None:
    """The increment happens before the check, so the rollback is what makes a
    refusal free. If this regresses, a user who hits their limit once can never
    use the service again."""
    s = _settings(quota_uploads_per_day=1)
    # Bound before the rollback: rollback expires every ORM object in the
    # session, so reading user.id afterwards would trigger a lazy load.
    user_id = user.id
    await quotas.consume(session, user_id, UPLOADS, 1, s)
    await session.commit()

    with pytest.raises(QuotaExceededError):
        await quotas.consume(session, user_id, UPLOADS, 1, s)
    await session.rollback()

    assert await quotas.current_usage(session, user_id, UPLOADS) == 1


async def test_metrics_are_counted_separately(session: AsyncSession, user: User) -> None:
    s = _settings(quota_uploads_per_day=5, quota_gpu_seconds_per_month=100)
    await quotas.consume(session, user.id, UPLOADS, 5, s)
    assert await quotas.consume(session, user.id, GPU_SECONDS, 10, s) == 10


async def test_users_are_counted_separately(session: AsyncSession, user: User) -> None:
    s = _settings(quota_uploads_per_day=1)
    other = User(email=f"other-{uuid.uuid4().hex[:12]}@example.com")
    session.add(other)
    await session.flush()
    await quotas.consume(session, user.id, UPLOADS, 1, s)
    assert await quotas.consume(session, other.id, UPLOADS, 1, s) == 1


async def test_zero_amount_is_a_read(session: AsyncSession, user: User) -> None:
    s = _settings(quota_uploads_per_day=1)
    await quotas.consume(session, user.id, UPLOADS, 1, s)
    # Must not raise even though usage is already at the limit.
    assert await quotas.consume(session, user.id, UPLOADS, 0, s) == 1


# --------------------------------------------------------------------------
# periods
# --------------------------------------------------------------------------


def test_daily_and_monthly_buckets_differ() -> None:
    moment = dt.datetime(2026, 9, 16, 12, 0, tzinfo=dt.UTC)
    assert quotas.period_key(UPLOADS, moment) == "2026-09-16"
    assert quotas.period_key(GPU_SECONDS, moment) == "2026-09"


def test_daily_reset_is_at_midnight_utc() -> None:
    moment = dt.datetime(2026, 9, 16, 23, 0, tzinfo=dt.UTC)
    assert quotas.seconds_until_reset(UPLOADS, moment) == 3600


def test_monthly_reset_rolls_into_the_next_month() -> None:
    moment = dt.datetime(2026, 9, 30, 23, 0, tzinfo=dt.UTC)
    assert quotas.seconds_until_reset(GPU_SECONDS, moment) == 3600


def test_monthly_reset_handles_a_short_month() -> None:
    """February is where naive +30 days arithmetic goes wrong."""
    moment = dt.datetime(2026, 2, 28, 23, 0, tzinfo=dt.UTC)
    assert quotas.seconds_until_reset(GPU_SECONDS, moment) == 3600


# --------------------------------------------------------------------------
# concurrency
# --------------------------------------------------------------------------


async def test_concurrency_allows_up_to_the_limit(session: AsyncSession, user: User) -> None:
    s = _settings(quota_concurrent_jobs=2)
    await quotas.assert_concurrency_available(session, user.id, s)


async def test_concurrency_refuses_beyond_the_limit(session: AsyncSession, user: User) -> None:
    """One user must not be able to fill the queue."""
    s = _settings(quota_concurrent_jobs=1)
    project = Project(user_id=user.id, title="t", status="queued")
    session.add(project)
    await session.flush()
    session.add(Job(project_id=project.id, status="running"))
    await session.flush()

    with pytest.raises(QuotaExceededError) as exc:
        await quotas.assert_concurrency_available(session, user.id, s)
    assert exc.value.metric == "concurrent_jobs"


async def test_finished_jobs_do_not_count(session: AsyncSession, user: User) -> None:
    s = _settings(quota_concurrent_jobs=1)
    project = Project(user_id=user.id, title="t", status="ready")
    session.add(project)
    await session.flush()
    session.add(Job(project_id=project.id, status="succeeded"))
    await session.flush()
    await quotas.assert_concurrency_available(session, user.id, s)


async def test_another_users_jobs_do_not_count(session: AsyncSession, user: User) -> None:
    s = _settings(quota_concurrent_jobs=1)
    other = User(email=f"other-{uuid.uuid4().hex[:12]}@example.com")
    session.add(other)
    await session.flush()
    project = Project(user_id=other.id, title="t", status="queued")
    session.add(project)
    await session.flush()
    session.add(Job(project_id=project.id, status="running"))
    await session.flush()
    await quotas.assert_concurrency_available(session, user.id, s)


# --------------------------------------------------------------------------
# snapshot
# --------------------------------------------------------------------------


async def test_snapshot_reports_every_metric(session: AsyncSession, user: User) -> None:
    """So the account screen can show a limit approaching, rather than the user
    discovering it at the moment of refusal."""
    s = _settings(quota_uploads_per_day=7)
    await quotas.consume(session, user.id, UPLOADS, 3, s)
    snap = await quotas.snapshot(session, user.id, s)
    assert set(snap) == {"uploads", "media_seconds", "gpu_seconds"}
    assert snap["uploads"] == {
        "used": 3,
        "limit": 7,
        "resets_in_seconds": snap["uploads"]["resets_in_seconds"],
    }


# --------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------


async def test_audit_records_an_event(session: AsyncSession, user: User) -> None:
    await audit.record(
        session, audit.LOGIN_SUCCEEDED, user_id=user.id, ip="203.0.113.7", ip_salt="s"
    )
    await session.commit()
    row = await session.scalar(select(AuditEvent).where(AuditEvent.user_id == user.id).limit(1))
    assert row is not None
    assert row.action == audit.LOGIN_SUCCEEDED


async def test_audit_never_stores_a_raw_ip(session: AsyncSession, user: User) -> None:
    await audit.record(session, audit.LOGIN_FAILED, user_id=user.id, ip="203.0.113.7", ip_salt="s")
    await session.commit()
    row = await session.scalar(
        select(AuditEvent).where(
            AuditEvent.user_id == user.id, AuditEvent.action == audit.LOGIN_FAILED
        )
    )
    assert row is not None
    assert row.ip_hash is not None
    assert "203.0.113.7" not in row.ip_hash


def test_ip_hash_is_salted() -> None:
    """An unsalted hash of an IPv4 address is reversible by exhaustion, which
    would make 'hashed' decorative."""
    assert audit.hash_ip("1.2.3.4", "salt-a") != audit.hash_ip("1.2.3.4", "salt-b")


def test_ip_hash_is_stable_for_one_salt() -> None:
    assert audit.hash_ip("1.2.3.4", "s") == audit.hash_ip("1.2.3.4", "s")


def test_ip_hash_passes_through_none() -> None:
    assert audit.hash_ip(None, "s") is None


async def test_audit_truncates_the_user_agent(session: AsyncSession, user: User) -> None:
    """Attacker-controlled and headed for a text column."""
    await audit.record(session, audit.LOGIN_FAILED, user_id=user.id, user_agent="x" * 5000)
    await session.commit()
    rows = (
        await session.scalars(
            select(AuditEvent).where(AuditEvent.user_id == user.id).order_by(AuditEvent.created_at)
        )
    ).all()
    assert any(r.user_agent is not None and len(r.user_agent) == 400 for r in rows)


async def test_audit_failure_does_not_raise(session: AsyncSession) -> None:
    """An audit gap is bad; an outage because the audit table is unhappy is worse."""
    await audit.record(session, audit.LOGIN_FAILED, user_id="usr_does_not_exist")
    await session.rollback()
