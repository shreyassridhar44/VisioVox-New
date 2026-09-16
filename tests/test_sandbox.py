"""Confined media processing (ADR-0009, invariant 5).

Two layers are tested differently:

- **The flags are asserted directly.** A security posture nobody can see is one
  that quietly regresses, so `build_argv` is inspected rather than trusted.
- **The confinement is exercised for real** where Docker is available, because
  a flag being present in an argv list proves nothing about whether the network
  is actually unreachable.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from pipeline import sandbox
from pipeline.sandbox import (
    Limits,
    MediaFacts,
    UnsupportedMediaError,
    assert_admissible,
    build_argv,
    facts_from,
    sniff_container,
)

IMAGE = "visiovox/media-sandbox:1"


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    return sandbox.image_present(IMAGE)


requires_sandbox = pytest.mark.skipif(
    not _docker_ready(), reason="docker or the sandbox image is unavailable"
)


# --------------------------------------------------------------------------
# magic bytes
# --------------------------------------------------------------------------


def test_mp4_is_recognised(tmp_path: Path) -> None:
    f = tmp_path / "a.mp4"
    f.write_bytes(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 8)
    assert sniff_container(f) == "mp4"


def test_matroska_is_recognised(tmp_path: Path) -> None:
    f = tmp_path / "a.mkv"
    f.write_bytes(b"\x1a\x45\xdf\xa3" + b"\x00" * 12)
    assert sniff_container(f) == "matroska"


def test_an_extension_cannot_launder_a_non_media_file(tmp_path: Path) -> None:
    """The filename and Content-Type are attacker-chosen and never consulted."""
    f = tmp_path / "payload.mp4"
    f.write_bytes(b"#!/bin/sh\nrm -rf /\n")
    with pytest.raises(UnsupportedMediaError):
        sniff_container(f)


def test_an_empty_file_is_refused(tmp_path: Path) -> None:
    f = tmp_path / "empty.mp4"
    f.write_bytes(b"")
    with pytest.raises(UnsupportedMediaError):
        sniff_container(f)


# --------------------------------------------------------------------------
# the flags are the posture
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--network", "none"),
        ("--user", "65534:65534"),
        ("--cap-drop", "ALL"),
        ("--security-opt", "no-new-privileges"),
        ("--env-file", "/dev/null"),
    ],
)
def test_hardening_flags_are_present(tmp_path: Path, flag: str, value: str) -> None:
    argv = build_argv(IMAGE, ["ffprobe"], workdir=tmp_path, writable=False, limits=Limits())
    assert value == argv[argv.index(flag) + 1]


def test_rootfs_is_read_only(tmp_path: Path) -> None:
    argv = build_argv(IMAGE, ["ffprobe"], workdir=tmp_path, writable=False, limits=Limits())
    assert "--read-only" in argv


def test_memory_swap_matches_memory(tmp_path: Path) -> None:
    """Without this the container swaps indefinitely and the memory ceiling
    stops bounding a decompression bomb."""
    argv = build_argv(
        IMAGE, ["ffprobe"], workdir=tmp_path, writable=False, limits=Limits(memory="512m")
    )
    assert argv[argv.index("--memory") + 1] == argv[argv.index("--memory-swap") + 1] == "512m"


def test_pids_are_limited(tmp_path: Path) -> None:
    """A fork bomb is otherwise trivially available from inside any container."""
    argv = build_argv(IMAGE, ["ffprobe"], workdir=tmp_path, writable=False, limits=Limits())
    assert int(argv[argv.index("--pids-limit") + 1]) > 0


def test_mount_is_read_only_unless_writing(tmp_path: Path) -> None:
    ro = build_argv(IMAGE, ["x"], workdir=tmp_path, writable=False, limits=Limits())
    rw = build_argv(IMAGE, ["x"], workdir=tmp_path, writable=True, limits=Limits())
    assert any(a.endswith(",readonly") for a in ro)
    assert not any(a.endswith(",readonly") for a in rw)


def test_runtime_is_used_when_supplied(tmp_path: Path) -> None:
    """If gVisor is ever installed this is the only change needed."""
    argv = build_argv(
        IMAGE, ["x"], workdir=tmp_path, writable=False, limits=Limits(), runtime="runsc"
    )
    assert argv[argv.index("--runtime") + 1] == "runsc"


# --------------------------------------------------------------------------
# admission
# --------------------------------------------------------------------------


def _facts(**overrides: object) -> MediaFacts:
    base = {
        "duration_seconds": 60.0,
        "has_audio": True,
        "has_video": True,
        "width": 1920,
        "height": 1080,
        "stream_count": 2,
        "format_name": "mov,mp4",
    }
    base.update(overrides)
    return MediaFacts(**base)  # type: ignore[arg-type]


def test_a_normal_file_is_admitted() -> None:
    assert_admissible(_facts(), max_duration_seconds=3600)


def test_audio_is_required() -> None:
    with pytest.raises(UnsupportedMediaError, match="no audio"):
        assert_admissible(_facts(has_audio=False), max_duration_seconds=3600)


def test_zero_duration_is_refused() -> None:
    with pytest.raises(UnsupportedMediaError, match="truncated or corrupt"):
        assert_admissible(_facts(duration_seconds=0), max_duration_seconds=3600)


def test_over_length_is_refused_with_both_numbers() -> None:
    """Duration is what costs GPU time, so it is the real admission control."""
    with pytest.raises(UnsupportedMediaError) as exc:
        assert_admissible(_facts(duration_seconds=7200), max_duration_seconds=3600)
    assert "120 minutes" in str(exc.value)
    assert "60" in str(exc.value)


def test_a_complexity_bomb_is_refused() -> None:
    """Each declared stream costs a demuxer context before a frame is decoded."""
    with pytest.raises(UnsupportedMediaError, match="streams"):
        assert_admissible(_facts(stream_count=5000), max_duration_seconds=3600)


def test_a_decompression_bomb_is_refused() -> None:
    """A tiny file claiming 32000x32000 frames."""
    with pytest.raises(UnsupportedMediaError, match="frame size"):
        assert_admissible(_facts(width=32000, height=32000), max_duration_seconds=3600)


def test_facts_survive_missing_fields() -> None:
    facts = facts_from({"streams": [], "format": {}})
    assert facts.duration_seconds == 0.0
    assert facts.has_audio is False


def test_facts_survive_a_non_numeric_duration() -> None:
    """ffprobe reports 'N/A' for some containers."""
    facts = facts_from({"streams": [], "format": {"duration": "N/A"}})
    assert facts.duration_seconds == 0.0


# --------------------------------------------------------------------------
# real confinement
# --------------------------------------------------------------------------


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    """A genuine 2-second clip, built with the host's ffmpeg on trusted input."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg unavailable")
    out = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=10:duration=2",
            "-shortest",
            "-y",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


@requires_sandbox
@pytest.mark.integration
def test_probe_reads_real_media(clip: Path) -> None:
    facts = facts_from(sandbox.probe(clip, image=IMAGE))
    assert 1.5 < facts.duration_seconds < 2.5
    assert facts.has_audio and facts.has_video
    assert (facts.width, facts.height) == (320, 240)


@requires_sandbox
@pytest.mark.integration
def test_the_network_is_genuinely_unreachable(tmp_path: Path) -> None:
    """The flag list says --network none; this proves it."""
    tmp_path.chmod(0o755)
    proc = sandbox.run(
        ["ffprobe", "-v", "error", "http://example.com/x.mp4"],
        workdir=tmp_path,
        image=IMAGE,
    )
    assert proc.returncode != 0
    assert "resolve" in proc.stderr.lower() or "network" in proc.stderr.lower()


@requires_sandbox
@pytest.mark.integration
def test_the_mount_is_genuinely_read_only(tmp_path: Path) -> None:
    tmp_path.chmod(0o755)
    proc = sandbox.run(["sh", "-c", "echo x > /work/evil"], workdir=tmp_path, image=IMAGE)
    assert proc.returncode != 0
    assert not (tmp_path / "evil").exists()


@requires_sandbox
@pytest.mark.integration
def test_it_does_not_run_as_root(tmp_path: Path) -> None:
    tmp_path.chmod(0o755)
    proc = sandbox.run(["id", "-u"], workdir=tmp_path, image=IMAGE)
    assert proc.stdout.strip() == "65534"


@requires_sandbox
@pytest.mark.integration
def test_the_environment_carries_no_credentials(tmp_path: Path) -> None:
    """The control ADR-0009 calls decisive: a compromised ffmpeg gets a scratch
    directory and nothing else."""
    tmp_path.chmod(0o755)
    proc = sandbox.run(["env"], workdir=tmp_path, image=IMAGE)
    text = proc.stdout.lower()
    for secret in ("aws", "s3_", "secret", "token", "password", "key="):
        assert secret not in text, f"{secret!r} leaked into the sandbox"


@requires_sandbox
@pytest.mark.integration
def test_a_non_media_file_never_reaches_a_demuxer(tmp_path: Path) -> None:
    f = tmp_path / "payload.mp4"
    f.write_bytes(b"not media at all")
    with pytest.raises(UnsupportedMediaError):
        sandbox.probe(f, image=IMAGE)
