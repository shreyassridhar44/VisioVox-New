"""Large-upload planning and reservations (docs/28 §W2, §D2).

Part planning is pure arithmetic and tested directly. Reservations are tested
against a real Postgres, because the property that matters is that concurrent
in-flight uploads are summed - ten 30 GB uploads must not all be admitted
because each one individually fits.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from visiovox_api import uploads
from visiovox_api.config import Settings, get_settings
from visiovox_api.models import Project, UploadSession, User
from visiovox_api.uploads import MAX_PARTS, MIN_PART_BYTES, UploadError

GIB = 1024**3
MIB = 1024**2


# --------------------------------------------------------------------------
# part planning — pure
# --------------------------------------------------------------------------


def test_small_file_is_a_single_part() -> None:
    plan = uploads.plan_parts(3 * MIB)
    assert plan.part_count == 1


def test_parts_never_go_below_the_s3_minimum() -> None:
    """Every part but the last must be >= 5 MiB or the store rejects it."""
    assert uploads.plan_parts(1).part_size_bytes >= MIN_PART_BYTES


@pytest.mark.parametrize("size_gib", [1, 5, 20, 50, 200, 500])
def test_part_count_stays_inside_the_ceiling(size_gib: int) -> None:
    """The reason part size scales instead of part count: at a fixed 16 MiB the
    10,000-part limit is reached at 160 GB."""
    plan = uploads.plan_parts(size_gib * GIB)
    assert 1 <= plan.part_count <= MAX_PARTS


@pytest.mark.parametrize("size_gib", [1, 5, 20, 50, 200, 500])
def test_parts_cover_the_whole_file(size_gib: int) -> None:
    size = size_gib * GIB
    plan = uploads.plan_parts(size)
    assert plan.part_size_bytes * plan.part_count >= size


def test_part_size_grows_for_very_large_files() -> None:
    small = uploads.plan_parts(10 * GIB).part_size_bytes
    huge = uploads.plan_parts(400 * GIB).part_size_bytes
    assert huge > small


def test_zero_size_is_refused() -> None:
    with pytest.raises(UploadError):
        uploads.plan_parts(0)


# --------------------------------------------------------------------------
# batching
# --------------------------------------------------------------------------


def test_first_batch_starts_at_part_one() -> None:
    assert uploads.batch_for(500, after=0)[0] == 1


def test_batch_is_capped() -> None:
    assert len(uploads.batch_for(500, after=0)) == uploads.PART_URL_BATCH


def test_batch_resumes_after_what_is_done() -> None:
    assert uploads.batch_for(500, after=120)[0] == 121


def test_batch_stops_at_the_last_part() -> None:
    assert uploads.batch_for(10, after=0) == list(range(1, 11))


def test_batch_is_empty_when_complete() -> None:
    assert uploads.batch_for(10, after=10) == []


# --------------------------------------------------------------------------
# merging parts
# --------------------------------------------------------------------------


def test_merge_sorts_by_part_number() -> None:
    merged = uploads.merge_parts(
        [{"part_number": 3, "etag": "c"}], [{"part_number": 1, "etag": "a"}]
    )
    assert [p["part_number"] for p in merged] == [1, 3]


def test_merge_lets_a_resent_part_win() -> None:
    """A resumed client re-sends parts it is unsure about; the newer etag is the
    one the object store kept."""
    merged = uploads.merge_parts(
        [{"part_number": 1, "etag": "old"}], [{"part_number": 1, "etag": "new"}]
    )
    assert merged == [{"part_number": 1, "etag": "new"}]


# --------------------------------------------------------------------------
# reservations — against the database
# --------------------------------------------------------------------------

pytest_plugins: list[str] = []
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
        await s.rollback()
    await engine.dispose()


async def _project(session: AsyncSession) -> Project:
    user = User(email=f"upl-{uuid.uuid4().hex[:12]}@example.com")
    session.add(user)
    await session.flush()
    project = Project(user_id=user.id, title="t", status="pending")
    session.add(project)
    await session.flush()
    return project


def _active(project: Project, reserved: int, hours: int = 12) -> UploadSession:
    return UploadSession(
        user_id=project.user_id,
        project_id=project.id,
        storage_key="k",
        upload_id=f"u-{uuid.uuid4().hex}",
        filename="f.mp4",
        size_bytes=reserved,
        part_size_bytes=16 * MIB,
        part_count=1,
        reserved_bytes=reserved,
        completed_parts=[],
        status="active",
        expires_at=dt.datetime.now(dt.UTC) + dt.timedelta(hours=hours),
    )


@pytest.mark.integration
async def test_active_uploads_are_summed(session: AsyncSession) -> None:
    project = await _project(session)
    before = await uploads.reserved_bytes(session)
    session.add(_active(project, 5 * GIB))
    session.add(_active(project, 3 * GIB))
    await session.flush()
    assert await uploads.reserved_bytes(session) == before + 8 * GIB


@pytest.mark.integration
async def test_expired_uploads_do_not_hold_space(session: AsyncSession) -> None:
    """A browser closed mid-upload must not keep its share of the disk."""
    project = await _project(session)
    before = await uploads.reserved_bytes(session)
    session.add(_active(project, 5 * GIB, hours=-1))
    await session.flush()
    assert await uploads.reserved_bytes(session) == before


@pytest.mark.integration
async def test_finished_uploads_do_not_hold_space(session: AsyncSession) -> None:
    project = await _project(session)
    before = await uploads.reserved_bytes(session)
    row = _active(project, 5 * GIB)
    row.status = "completed"
    row.reserved_bytes = 0
    session.add(row)
    await session.flush()
    assert await uploads.reserved_bytes(session) == before


@pytest.mark.integration
async def test_expire_stale_releases_the_reservation(session: AsyncSession) -> None:
    project = await _project(session)
    row = _active(project, 5 * GIB, hours=-1)
    session.add(row)
    await session.flush()

    assert await uploads.expire_stale(session) >= 1
    await session.flush()
    assert row.status == "expired"
    assert row.reserved_bytes == 0


@pytest.mark.integration
async def test_in_flight_uploads_reduce_the_advertised_limit(session: AsyncSession) -> None:
    """The property that stops ten concurrent uploads each being told yes."""
    # A high ceiling so the DISK is the binding constraint here; with the real
    # 50 GB backstop the cap binds first on this machine and the reservation
    # would be invisible.
    settings = Settings(media_root="/srv/media", disk_reserved_bytes=0, max_upload_bytes=500 * GIB)
    project = await _project(session)
    before = await uploads.max_upload_bytes_now(session, settings)

    session.add(_active(project, 20 * GIB))
    await session.flush()
    after = await uploads.max_upload_bytes_now(session, settings)

    assert after < before


@pytest.mark.integration
async def test_a_file_over_the_live_limit_is_refused_with_a_reason(
    session: AsyncSession,
) -> None:
    settings = Settings(media_root="/srv/media")
    with pytest.raises(UploadError) as exc:
        await uploads.assert_fits(session, settings, 900 * GIB)
    message = str(exc.value)
    # The number and the remedy both matter: "too large" alone tells the user
    # nothing about whether to trim, wait, or give up.
    assert "GB" in message
    assert "limit" in message.lower()


@pytest.mark.integration
async def test_a_reasonable_file_is_admitted(session: AsyncSession) -> None:
    settings = Settings(media_root="/srv/media")
    await uploads.assert_fits(session, settings, 1 * GIB)


def test_reservation_applies_the_peak_multiplier() -> None:
    """Accepting N bytes commits roughly multiplier * N, because source, working
    copy and outputs coexist on disk."""
    settings = Settings(upload_peak_multiplier=2.5)
    assert uploads.reservation_for(10 * GIB, settings) == int(25 * GIB)
