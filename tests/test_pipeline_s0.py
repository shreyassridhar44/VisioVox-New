"""S0 ingest and pipeline timing tests.

The timing assertions matter more than they look: invariant 2 exists because
float-seconds arithmetic accumulates error across a long timeline, and the
symptom (caption drift at minute 45) is far from the cause.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from pipeline.s0_ingest import IngestError, ingest, probe_media
from pipeline.types import (
    ANALYSIS_SAMPLE_RATE,
    Interval,
    StageResult,
    StageStatus,
    ms_to_samples,
    samples_to_ms,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mix_2spk.wav"

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not available",
)


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------


def test_sample_ms_roundtrip_is_stable() -> None:
    """Converting ms -> samples -> ms must not drift for realistic durations."""
    for ms in (0, 1, 33, 1000, 60_000, 3_600_000):
        assert samples_to_ms(ms_to_samples(ms)) == ms


def test_no_accumulated_drift_over_an_hour() -> None:
    """Summing 20 ms hops for an hour must land exactly, unlike float seconds."""
    total = sum(ms_to_samples(20) for _ in range(180_000))
    assert samples_to_ms(total) == 3_600_000


def test_samples_to_ms_rounds_half_up() -> None:
    # 8 samples @16 kHz = 0.5 ms exactly
    assert samples_to_ms(8) == 1
    assert samples_to_ms(7) == 0


def test_interval_rejects_inverted_span() -> None:
    with pytest.raises(ValueError, match="precedes start"):
        Interval(500, 100)


def test_interval_overlap_is_half_open() -> None:
    a = Interval(0, 16_000)
    b = Interval(16_000, 32_000)
    assert not a.overlaps(b), "touching intervals must not count as overlapping"
    assert a.intersection(b) == 0
    assert a.overlaps(Interval(15_999, 20_000))
    assert a.intersection(Interval(8_000, 24_000)) == 8_000


# --------------------------------------------------------------------------
# stage envelope
# --------------------------------------------------------------------------


def test_warning_tokens_must_be_machine_readable() -> None:
    r = StageResult(stage="S0_ingest", status=StageStatus.OK)
    r.warn("reference_audio_unavailable")
    assert r.warnings == ["reference_audio_unavailable"]
    r.warn("reference_audio_unavailable")
    assert len(r.warnings) == 1, "duplicate warnings must collapse"
    with pytest.raises(ValueError, match="snake_case"):
        r.warn("Reference audio unavailable!")


# --------------------------------------------------------------------------
# ingest
# --------------------------------------------------------------------------


@needs_ffmpeg
def test_probe_reads_the_fixture() -> None:
    probe = probe_media(FIXTURE)
    assert probe.has_audio
    assert not probe.has_video
    assert probe.duration_ms > 0
    assert probe.audio_sample_rate == ANALYSIS_SAMPLE_RATE


@needs_ffmpeg
def test_ingest_normalises_audio_only_input(tmp_path: Path) -> None:
    media, result = ingest(FIXTURE, tmp_path)
    assert result.status in (StageStatus.OK, StageStatus.DEGRADED)
    assert media.analysis_wav.exists()
    assert (tmp_path / "probe.json").exists()
    assert media.video_mp4 is None

    import soundfile as sf

    info = sf.info(media.analysis_wav)
    assert info.samplerate == ANALYSIS_SAMPLE_RATE
    assert info.channels == 1


@needs_ffmpeg
def test_ingest_is_idempotent(tmp_path: Path) -> None:
    """Invariant 7: a retried stage resumes rather than redoing the work."""
    media, _ = ingest(FIXTURE, tmp_path)
    first = media.analysis_wav.stat().st_mtime_ns
    media2, _ = ingest(FIXTURE, tmp_path)
    assert media2.analysis_wav.stat().st_mtime_ns == first


@needs_ffmpeg
def test_ingest_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="no such input"):
        ingest(tmp_path / "nope.mp4", tmp_path / "out")


@needs_ffmpeg
def test_ingest_rejects_audioless_input(tmp_path: Path) -> None:
    """FR-UPL-05: there is nothing to isolate without audio, so reject."""
    import subprocess

    silent_video = tmp_path / "novideo.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=128x128:rate=25",
            "-c:v",
            "libx264",
            str(silent_video),
        ],
        check=True,
        capture_output=True,
    )
    with pytest.raises(IngestError, match="no audio stream"):
        ingest(silent_video, tmp_path / "out")
