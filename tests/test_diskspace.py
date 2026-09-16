"""Disk headroom and the admission cut-out (docs/track-w/W0).

The regression these tests exist for: measuring `/` inside the WSL distro returns
the vhdx's virtual maximum rather than real free space, so a naive check passes
on a full disk. `test_untrustworthy_reading_*` are the guard against that coming
back.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from visiovox_api.config import Settings
from visiovox_api.diskspace import (
    DiskLayoutError,
    Headroom,
    assert_dedicated_volume,
    can_admit,
    max_ingestible_bytes,
    measure,
)

GB = 1024**3


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {
        "media_root": str(tmp_path),
        "disk_reserved_bytes": 20 * GB,
        "upload_peak_multiplier": 2.5,
        "max_upload_bytes": 50 * GB,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _headroom(
    *,
    free: int,
    reserved: int = 20 * GB,
    shares_root: bool = False,
    path: Path | None = None,
) -> Headroom:
    return Headroom(
        path=path or Path("/srv/media"),
        total_bytes=250 * GB,
        free_bytes=free,
        reserved_bytes=reserved,
        shares_filesystem_with_root=shares_root,
    )


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------


def test_measure_reads_the_configured_volume(tmp_path: Path) -> None:
    headroom = measure(_settings(tmp_path))
    assert headroom.path == tmp_path
    assert headroom.total_bytes > 0


def test_missing_media_root_raises_rather_than_reporting_zero(tmp_path: Path) -> None:
    """A missing mount is a deployment fault, not a full disk.

    Reporting it as "no space" would send an operator to the wrong problem.
    """
    settings = _settings(tmp_path / "not-mounted")
    with pytest.raises(DiskLayoutError, match="not mounted"):
        measure(settings)


def test_trustworthiness_is_the_negation_of_sharing_root(tmp_path: Path) -> None:
    """The two flags must never disagree — admission control reads one and the
    startup guard reads the other."""
    headroom = measure(_settings(tmp_path))
    assert headroom.is_trustworthy is not headroom.shares_filesystem_with_root


# --------------------------------------------------------------------------
# the guard against the original bug
# --------------------------------------------------------------------------


def test_untrustworthy_reading_is_rejected_when_a_volume_is_required() -> None:
    with pytest.raises(DiskLayoutError, match="virtual maximum"):
        assert_dedicated_volume(_headroom(free=200 * GB, shares_root=True))


def test_untrustworthy_reading_blocks_admission_when_required() -> None:
    """Plenty of apparent space must not be enough if the reading is a lie."""
    settings = _settings(Path("/"), require_dedicated_media_volume=True)
    assert can_admit(settings, _headroom(free=600 * GB, shares_root=True)) is False


def test_untrustworthy_reading_is_tolerated_locally() -> None:
    """Local development and CI have no dedicated volume; that is legitimate."""
    settings = _settings(Path("/"), require_dedicated_media_volume=False)
    assert can_admit(settings, _headroom(free=600 * GB, shares_root=True)) is True


def test_dedicated_volume_passes_the_guard() -> None:
    assert_dedicated_volume(_headroom(free=200 * GB, shares_root=False))


# --------------------------------------------------------------------------
# usable space and the derived limit
# --------------------------------------------------------------------------


def test_reserved_floor_is_subtracted_from_usable() -> None:
    headroom = _headroom(free=100 * GB, reserved=20 * GB)
    assert headroom.usable_bytes == 80 * GB


def test_usable_never_goes_negative_below_the_floor() -> None:
    headroom = _headroom(free=5 * GB, reserved=20 * GB)
    assert headroom.usable_bytes == 0
    assert headroom.is_exhausted is True


def test_limit_divides_usable_space_by_the_peak_multiplier(tmp_path: Path) -> None:
    """A job holds source + working copy + outputs at once, so 80 GB free does
    not mean an 80 GB upload may be accepted."""
    settings = _settings(tmp_path, upload_peak_multiplier=2.5)
    limit = max_ingestible_bytes(settings, _headroom(free=100 * GB, reserved=20 * GB))
    assert limit == int(80 * GB / 2.5)


def test_limit_is_capped_by_the_hard_ceiling(tmp_path: Path) -> None:
    """A roomy volume still must not accept one absurd upload."""
    settings = _settings(tmp_path, max_upload_bytes=50 * GB, upload_peak_multiplier=1.0)
    limit = max_ingestible_bytes(settings, _headroom(free=4000 * GB, reserved=20 * GB))
    assert limit == 50 * GB


def test_exhausted_volume_offers_no_capacity(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    headroom = _headroom(free=1 * GB, reserved=20 * GB)
    assert max_ingestible_bytes(settings, headroom) == 0
    assert can_admit(settings, headroom) is False


def test_the_workstation_failure_case(tmp_path: Path) -> None:
    """The concrete situation on 2026-09-16: `df /` says 675 GB, the host drive
    has 5.6 GB. Measuring the real volume must refuse the job."""
    settings = _settings(tmp_path, disk_reserved_bytes=20 * GB)
    real = _headroom(free=int(5.6 * GB), reserved=20 * GB, shares_root=False)
    assert real.is_exhausted is True
    assert can_admit(settings, real) is False
    assert max_ingestible_bytes(settings, real) == 0
