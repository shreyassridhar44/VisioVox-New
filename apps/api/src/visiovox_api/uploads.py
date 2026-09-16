"""Large resumable uploads (docs/28 §W2, §D2).

Three problems this solves that the Phase 2 upload path did not:

**The limit is computed, not configured.** `max_upload_bytes` was a constant. It
is now a backstop, and the number a user actually sees comes from live free space
on the media volume less what other in-flight uploads have already booked. A cap
that ignores the disk is a promise the machine cannot keep.

**Space is reserved, not merely counted.** Rate-limiting `/upload/init` is not
enough: each init creates a real multipart upload that occupies storage whether
or not it is ever completed. Ten concurrent 30 GB uploads must not all be told
yes because each one individually fits. The reservation is released on complete,
abort or expiry.

**Part URLs are issued in batches.** Presigning every part up front was fine at
2 GB and is not at 30 GB - it is thousands of URLs in one response, all with the
same short expiry, most of them expired before the client reaches them.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from . import diskspace
from .config import Settings
from .models import UploadSession

MIB = 1024 * 1024

# S3 requires >= 5 MiB for every part but the last, and at most 10,000 parts.
MIN_PART_BYTES = 5 * MIB
MAX_PARTS = 10_000

# 16 MiB balances two failure modes: small parts mean more round trips and more
# chances to fail, large parts mean more to re-send when one does. It also keeps
# a 160 GB file inside the part ceiling, which is well past anything admissible.
TARGET_PART_BYTES = 16 * MIB

# Presigned URLs handed out at a time.
PART_URL_BATCH = 50

# How long a started upload may sit before its reservation is reclaimed. Long
# enough for a slow connection to finish a large file, short enough that an
# abandoned upload does not hold space for a day.
SESSION_TTL_HOURS = 12


class UploadError(Exception):
    """The upload cannot proceed, with a reason meant for the user."""


@dataclass(frozen=True)
class PartPlan:
    part_size_bytes: int
    part_count: int


def plan_parts(size_bytes: int) -> PartPlan:
    """Choose a part size that satisfies S3's limits for this file.

    Scaling the part size up rather than accepting more parts is what keeps very
    large files legal: at a fixed 16 MiB, the 10,000-part ceiling is reached at
    160 GB, and past that the upload has to use bigger parts or fail.
    """
    if size_bytes <= 0:
        raise UploadError("file size must be greater than zero")

    part_size = max(TARGET_PART_BYTES, math.ceil(size_bytes / MAX_PARTS))
    part_size = max(MIN_PART_BYTES, math.ceil(part_size / MIB) * MIB)
    part_count = max(1, math.ceil(size_bytes / part_size))

    if part_count > MAX_PARTS:  # pragma: no cover - arithmetic makes this unreachable
        raise UploadError("file is too large to upload in one object")
    return PartPlan(part_size_bytes=part_size, part_count=part_count)


def batch_for(part_count: int, after: int) -> list[int]:
    """Part numbers to presign next, given what the client already holds."""
    start = max(1, after + 1)
    return list(range(start, min(part_count, start + PART_URL_BATCH - 1) + 1))


async def reserved_bytes(session: AsyncSession) -> int:
    """Space already booked by uploads that have not finished.

    Counted across all users: the disk is shared, and one user's in-flight
    upload is unavailable space for everyone.
    """
    total = await session.scalar(
        select(func.coalesce(func.sum(UploadSession.reserved_bytes), 0)).where(
            UploadSession.status == "active",
            UploadSession.expires_at > dt.datetime.now(dt.UTC),
        )
    )
    return int(total or 0)


async def available_bytes(session: AsyncSession, settings: Settings) -> int:
    """Usable headroom minus what in-flight uploads have booked."""
    headroom = diskspace.measure(settings)
    return max(0, headroom.usable_bytes - await reserved_bytes(session))


async def max_upload_bytes_now(session: AsyncSession, settings: Settings) -> int:
    """The number to show the user, right now.

    Divided by the peak multiplier because accepting N bytes commits roughly
    `multiplier * N` - source, working copy and outputs coexist on disk - and
    capped by the configured backstop against one absurd upload.
    """
    affordable = int(await available_bytes(session, settings) / settings.upload_peak_multiplier)
    return max(0, min(affordable, settings.max_upload_bytes))


def reservation_for(size_bytes: int, settings: Settings) -> int:
    return int(size_bytes * settings.upload_peak_multiplier)


async def assert_fits(session: AsyncSession, settings: Settings, size_bytes: int) -> None:
    """Refuse before creating anything, with a reason and a remedy.

    'Too large' with no number tells a user nothing about whether to trim the
    file, wait, or give up.
    """
    headroom = diskspace.measure(settings)
    if not diskspace.can_admit(settings, headroom):
        raise UploadError(
            "Storage is temporarily full and new uploads are paused. Please try again later."
        )

    limit = await max_upload_bytes_now(session, settings)
    if size_bytes > limit:
        gib = size_bytes / 1024**3
        allowed = limit / 1024**3
        raise UploadError(
            f"This file is {gib:.1f} GB and the current limit is {allowed:.1f} GB. "
            "The limit reflects free space and uploads already in progress, so it may rise shortly."
        )


async def expire_stale(session: AsyncSession) -> int:
    """Release reservations held by uploads that were abandoned.

    Without this, a browser closed mid-upload holds its share of the disk until
    someone notices. Returns how many were reclaimed.
    """
    stale = (
        await session.scalars(
            select(UploadSession).where(
                UploadSession.status == "active",
                UploadSession.expires_at <= dt.datetime.now(dt.UTC),
            )
        )
    ).all()
    for row in stale:
        row.status = "expired"
        row.reserved_bytes = 0
    return len(stale)


def new_expiry() -> dt.datetime:
    return dt.datetime.now(dt.UTC) + dt.timedelta(hours=SESSION_TTL_HOURS)


def merge_parts(
    existing: list[dict[str, object]], incoming: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Fold newly-confirmed parts into the recorded set, last write winning.

    A resumed upload re-sends parts it is unsure about, so the same part number
    legitimately arrives twice with different etags; the newer one is the one
    the object store will have kept.
    """
    by_number = {int(str(p["part_number"])): p for p in existing}
    for part in incoming:
        by_number[int(str(part["part_number"]))] = part
    return [by_number[n] for n in sorted(by_number)]
