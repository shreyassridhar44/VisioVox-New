"""S0 — Ingest: demux, decode, normalise (docs/05 §2).

Produces the normalised media set every later stage reads. Two audio renders,
deliberately:

- `analysis.wav` at 16 kHz mono, resampled with soxr at high precision. The
  ffmpeg default resampler is fast and audibly fine but has poorer stopband
  rejection; aliasing here would be baked into every downstream measurement.
- `reference.wav` at 48 kHz, preserving fidelity for final packaging, because
  S9 must not encode from the 16 kHz analysis copy.

Sandboxing: docs/15 §4 requires every ffmpeg invocation on untrusted media to
run confined. This module always invokes ffmpeg through `_run`, which is the
single place that constraint is applied -- see `SANDBOX_NOTE`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from .types import (
    ANALYSIS_SAMPLE_RATE,
    MediaSet,
    Probe,
    StageResult,
    StageStatus,
)

STAGE = "S0_ingest"
VERSION = "1.0.0"

SANDBOX_NOTE = (
    "Phase 1 runs ffmpeg directly on curated AMI fixtures. Before any "
    "user-supplied media reaches this stage it must be wrapped per docs/15 §4 "
    "(invariant 5). Tracked as a Phase 2 gate."
)

_FFMPEG_TIMEOUT = 3600


class IngestError(RuntimeError):
    """Media that cannot be normalised. FR-UPL-05: reject, do not guess."""


def _run(cmd: list[str], timeout: int = _FFMPEG_TIMEOUT) -> subprocess.CompletedProcess[str]:
    """Single choke point for media-tool invocation.

    Everything goes through here so the sandbox wrapper has exactly one place
    to be added, rather than being sprinkled across call sites where a new one
    would quietly miss it.
    """
    return subprocess.run(  # noqa: S603 - argv built here, shell=False, no user strings
        cmd, check=False, capture_output=True, text=True, timeout=timeout
    )


def _require_tools() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise IngestError(f"{tool} not found on PATH")


def probe_media(src: Path) -> Probe:
    """Read stream metadata. The record is kept so a rejection can be explained."""
    _require_tools()
    proc = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(src),
        ],
        timeout=120,
    )
    if proc.returncode != 0:
        raise IngestError(f"ffprobe failed: {proc.stderr.strip()[:300]}")

    data = json.loads(proc.stdout)
    streams = data.get("streams", [])
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    video = next((s for s in streams if s.get("codec_type") == "video"), None)

    duration_s = float(data.get("format", {}).get("duration", 0.0))

    frame_rate: float | None = None
    if video is not None:
        raw = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/0"
        num, _, den = raw.partition("/")
        try:
            frame_rate = float(num) / float(den) if float(den) else None
        except (ValueError, ZeroDivisionError):
            frame_rate = None

    return Probe(
        duration_ms=round(duration_s * 1000),
        has_audio=audio is not None,
        has_video=video is not None,
        width=int(video["width"]) if video and "width" in video else None,
        height=int(video["height"]) if video and "height" in video else None,
        frame_rate=frame_rate,
        audio_channels=int(audio["channels"]) if audio and "channels" in audio else None,
        audio_sample_rate=int(audio["sample_rate"]) if audio and "sample_rate" in audio else None,
    )


def ingest(src: Path, out_dir: Path) -> tuple[MediaSet, StageResult]:
    """Normalise `src` into `out_dir`.

    Idempotent (invariant 7): existing outputs are reused, so a retried job
    resumes rather than re-decoding.
    """
    t0 = time.perf_counter()
    result = StageResult(stage=STAGE, status=StageStatus.OK)

    if not src.is_file():
        raise IngestError(f"no such input: {src}")
    out_dir.mkdir(parents=True, exist_ok=True)

    probe = probe_media(src)
    if not probe.has_audio:
        # FR-UPL-05 — there is nothing to isolate without audio.
        raise IngestError("input has no audio stream")
    if probe.duration_ms <= 0:
        raise IngestError("input reports zero duration")

    # Probe is a slots dataclass, so asdict rather than __dict__.
    probe_json = json.dumps(asdict(probe), indent=2)
    (out_dir / "probe.json").write_text(probe_json)

    analysis = out_dir / "analysis.wav"
    reference = out_dir / "reference.wav"
    video_out: Path | None = out_dir / "video.mp4" if probe.has_video else None

    if not analysis.exists():
        proc = _run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(src),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(ANALYSIS_SAMPLE_RATE),
                # soxr at high precision: the default resampler's stopband
                # rejection is not good enough to measure against.
                "-af",
                "aresample=resampler=soxr:precision=28",
                "-c:a",
                "pcm_s16le",
                str(analysis),
            ]
        )
        if proc.returncode != 0:
            raise IngestError(f"analysis render failed: {proc.stderr.strip()[:300]}")

    if not reference.exists():
        proc = _run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(src),
                "-vn",
                "-ar",
                "48000",
                "-c:a",
                "pcm_s24le",
                str(reference),
            ]
        )
        if proc.returncode != 0:
            # Non-fatal: analysis audio is what the pipeline needs; packaging
            # can fall back. Invariant 8 -- degrade, do not abort.
            result.status = StageStatus.DEGRADED
            result.warn("reference_audio_unavailable")
            reference = analysis

    if video_out is not None and not video_out.exists():
        cfr_args: list[str] = []
        if probe.frame_rate is None or abs(probe.frame_rate - 25.0) > 0.01:
            # Variable or non-25 fps: force CFR so frame index maps to time.
            cfr_args = ["-r", "25"]
            result.warn("video_resampled_to_25fps")
        proc = _run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(src),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-vf",
                "scale='min(1920,iw)':-2",
                *cfr_args,
                str(video_out),
            ]
        )
        if proc.returncode != 0:
            result.status = StageStatus.DEGRADED
            result.warn("video_normalisation_failed")
            video_out = None

    media = MediaSet(
        root=out_dir,
        analysis_wav=analysis,
        reference_wav=reference,
        video_mp4=video_out,
        probe=probe,
    )
    result.seconds = time.perf_counter() - t0
    result.detail = (
        f"{probe.duration_ms} ms, "
        f"{'video ' + str(probe.width) + 'x' + str(probe.height) if probe.has_video else 'audio-only'}"
    )
    return media, result
