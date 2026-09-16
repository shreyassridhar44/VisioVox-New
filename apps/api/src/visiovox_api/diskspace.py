"""Disk headroom for the media volume (docs/track-w/W0).

**Never ask `/` how much space there is.** The API runs inside a WSL2 distro whose
root filesystem is a dynamically-expanding vhdx. `statvfs("/")` reports that
vhdx's *virtual* maximum — on this workstation, 675 GB available on a host drive
with 5.6 GB actually left. A check against `/` therefore passes right up until
ffmpeg dies with ENOSPC halfway through someone's job, which is the worst
possible moment to discover it. This has already caused one bug in the project.

So every question about space is asked of the **configured media volume**, and
`Headroom` reports whether that volume is genuinely separate from `/` — if it is
not, the measurement is untrustworthy and admission control says so rather than
guessing.

The reserved floor exists because "free space" and "space we may use" are
different numbers. A job needs room for the source, a working copy and its
outputs simultaneously; running the volume to zero corrupts whatever is
mid-write, not merely the job that overshot.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import Settings


class DiskLayoutError(RuntimeError):
    """The media volume is missing, or is not where it was configured to be."""


@dataclass(frozen=True)
class Headroom:
    """A point-in-time measurement of the media volume.

    `usable_bytes` is what admission control may actually spend: free space less
    the reserved floor, never negative.
    """

    path: Path
    total_bytes: int
    free_bytes: int
    reserved_bytes: int
    shares_filesystem_with_root: bool

    @property
    def usable_bytes(self) -> int:
        return max(0, self.free_bytes - self.reserved_bytes)

    @property
    def is_exhausted(self) -> bool:
        return self.usable_bytes <= 0

    @property
    def is_trustworthy(self) -> bool:
        """False when the reading came from the vhdx root and means nothing."""
        return not self.shares_filesystem_with_root


def _device_id(path: Path) -> int:
    return path.stat().st_dev


def measure(settings: Settings) -> Headroom:
    """Measure the configured media volume.

    Raises rather than returning a zero reading when the directory is absent:
    a missing media root is a deployment error, and reporting it as "no space"
    would send the operator looking at the wrong problem.
    """
    media_root = Path(settings.media_root)
    if not media_root.is_dir():
        raise DiskLayoutError(
            f"media_root {media_root} does not exist; the media volume is not mounted"
        )

    usage = shutil.disk_usage(media_root)
    return Headroom(
        path=media_root,
        total_bytes=usage.total,
        free_bytes=usage.free,
        reserved_bytes=settings.disk_reserved_bytes,
        shares_filesystem_with_root=_device_id(media_root) == _device_id(Path("/")),
    )


def assert_dedicated_volume(headroom: Headroom) -> None:
    """Fail loudly when the media root is on the same filesystem as `/`.

    Called at startup where `require_dedicated_media_volume` is set. Local
    development and CI legitimately run without a separate volume, so this is
    configuration rather than an unconditional rule — but on the deployed
    workstation it must be on, because that is the exact machine whose `df` lies.
    """
    if not headroom.is_trustworthy:
        raise DiskLayoutError(
            f"media_root {headroom.path} shares a filesystem with '/'. Inside this "
            "distro that reading is the vhdx's virtual maximum, not real free "
            "space. Mount a dedicated media volume, or unset "
            "REQUIRE_DEDICATED_MEDIA_VOLUME if this is local development."
        )


def max_ingestible_bytes(settings: Settings, headroom: Headroom) -> int:
    """Largest single upload the volume can take right now.

    A job holds its source, a derived working copy and its outputs at the same
    time, so accepting N bytes commits roughly `peak_multiplier * N`. Dividing
    by that multiplier is what turns free space into an honest upload limit.

    The result is capped by `max_upload_bytes`, which is a backstop against one
    absurd upload rather than the number shown to users (docs/28 §D2).
    """
    if headroom.is_exhausted:
        return 0
    affordable = int(headroom.usable_bytes / settings.upload_peak_multiplier)
    return min(affordable, settings.max_upload_bytes)


def can_admit(settings: Settings, headroom: Headroom) -> bool:
    """Whether new work may be queued at all.

    This is the disk half of the admission cut-out that replaces the cloud
    budget cut-out in docs/15 §9 — nothing is rented, so the scarce resource is
    this volume rather than a bill (docs/28 §D5).
    """
    if not headroom.is_trustworthy and settings.require_dedicated_media_volume:
        return False
    return not headroom.is_exhausted
