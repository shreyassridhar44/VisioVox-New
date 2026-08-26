"""S9 — Packaging: loudness, encoding, captions, manifest (docs/05 §12).

Emits the artifact manifest that is the contract between the pipeline, the API
and the player (packages/contracts/schemas/manifest.schema.json).

Two invariants are enforced here rather than hoped for:

- **Invariant 1**: every audio track for a project has an identical sample
  count. A one-sample mismatch becomes accumulating A/V drift once the player
  puts all tracks on a single AudioContext clock, and it is essentially
  invisible until someone notices captions sliding at minute 40. Asserted, not
  assumed.
- **Invariant 2**: durations are integer milliseconds derived from sample
  counts, never float seconds.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .s7_transcribe import Transcript
from .s9_hls import choose_engine, package_hls
from .types import (
    ANALYSIS_SAMPLE_RATE,
    Modality,
    StageResult,
    StageStatus,
    samples_to_ms,
)

STAGE = "S9_package"
VERSION = "1.0.0"

MANIFEST_VERSION = "1.0"
TARGET_LUFS = -16.0  # EBU R128-ish, appropriate for speech playback


class PackagingError(RuntimeError):
    """Packaging cannot proceed without breaking a stated invariant."""


@dataclass
class SpeakerTrack:
    """One speaker's recovered audio plus everything the manifest needs."""

    speaker_id: str
    ordinal: int
    label: str
    faithful: np.ndarray
    modality: Modality
    speaking_ratio: float
    mean_confidence: float
    extraction_ok: bool
    transcript: Transcript | None = None
    natural: np.ndarray | None = None


def _lufs_normalise(audio: np.ndarray, target: float = TARGET_LUFS) -> np.ndarray:
    """Loudness-normalise to a common target so switching speakers is not a jump.

    Uses pyloudnorm when available. Peak-limits afterwards rather than allowing
    clipping, because a normalised-but-clipped track sounds worse than a quiet
    one and the player has no way to undo it.
    """
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(ANALYSIS_SAMPLE_RATE)
        loudness = meter.integrated_loudness(audio)
        if not np.isfinite(loudness):
            return audio
        out = pyln.normalize.loudness(audio, loudness, target)
    except Exception:
        return audio

    peak = float(np.abs(out).max())
    if peak > 0.99:
        out = out * (0.99 / peak)
    return np.asarray(out, dtype=np.float32)


def _encode_m4a(samples: np.ndarray, dest: Path, sample_rate: int) -> int:
    """Write AAC via ffmpeg and return the byte size."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise PackagingError("ffmpeg not on PATH")
    wav = dest.with_suffix(".tmp.wav")
    sf.write(wav, samples, sample_rate)
    proc = subprocess.run(  # noqa: S603 - fixed argv, shell=False
        [
            ffmpeg,
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(wav),
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(dest),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    wav.unlink(missing_ok=True)
    if proc.returncode != 0 or not dest.exists():
        raise PackagingError(f"aac encode failed: {proc.stderr.strip()[:200]}")
    return dest.stat().st_size


def _assert_equal_lengths(tracks: Sequence[SpeakerTrack]) -> int:
    """Invariant 1. Returns the common sample count."""
    lengths = {len(t.faithful) for t in tracks}
    for t in tracks:
        if t.natural is not None:
            lengths.add(len(t.natural))
    if len(lengths) > 1:
        raise PackagingError(
            f"tracks have differing sample counts {sorted(lengths)} — invariant 1. "
            "A one-sample mismatch becomes accumulating A/V drift in the player."
        )
    return lengths.pop() if lengths else 0


def package(
    project_id: str,
    tracks: Sequence[SpeakerTrack],
    out_dir: Path,
    *,
    has_video: bool,
    video_url: str | None = None,
    video_size: tuple[int, int] | None = None,
    overlap_ratio: float | None = None,
    signed_until: str,
    base_url: str = "",
) -> tuple[dict[str, Any], StageResult]:
    """Write artifacts and build the manifest."""
    t0 = time.perf_counter()
    result = StageResult(stage=STAGE, status=StageStatus.OK)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_samples = _assert_equal_lengths(tracks)
    duration_ms = samples_to_ms(n_samples)

    speakers: list[dict[str, Any]] = []
    for track in tracks:
        prefix = f"spk_{track.ordinal}"
        faithful = _lufs_normalise(track.faithful)
        faithful_path = out_dir / f"{prefix}_f.m4a"
        faithful_bytes = _encode_m4a(faithful, faithful_path, ANALYSIS_SAMPLE_RATE)

        audio: dict[str, Any] = {
            "faithful": {"url": f"{base_url}{faithful_path.name}", "bytes": faithful_bytes}
        }
        if track.natural is not None:
            natural_path = out_dir / f"{prefix}_n.m4a"
            natural_bytes = _encode_m4a(
                _lufs_normalise(track.natural), natural_path, ANALYSIS_SAMPLE_RATE
            )
            audio["natural"] = {
                "url": f"{base_url}{natural_path.name}",
                "bytes": natural_bytes,
            }

        captions: dict[str, Any] = {}
        if track.transcript is not None:
            vtt_path = out_dir / f"{prefix}.vtt"
            vtt_path.write_text(track.transcript.to_vtt(), encoding="utf-8")
            captions["vtt"] = f"{base_url}{vtt_path.name}"

            json_path = out_dir / f"{prefix}.json"
            json_path.write_text(
                json.dumps(
                    [
                        {
                            "text": s.text,
                            "start_ms": s.start_ms,
                            "end_ms": s.end_ms,
                            "words": [
                                {
                                    "text": w.text,
                                    "start_ms": w.start_ms,
                                    "end_ms": w.end_ms,
                                    "probability": round(w.probability, 4),
                                }
                                for w in s.words
                            ],
                        }
                        for s in track.transcript.segments
                    ],
                    indent=2,
                ),
                encoding="utf-8",
            )
            captions["json"] = f"{base_url}{json_path.name}"
        else:
            # Invariant 8: a speaker without captions is still a usable result.
            result.status = StageStatus.DEGRADED
            result.warn(f"speaker_{track.ordinal}_no_captions")
            captions["vtt"] = f"{base_url}{prefix}.vtt"
            (out_dir / f"{prefix}.vtt").write_text("WEBVTT\n\n", encoding="utf-8")

        if not track.extraction_ok:
            result.status = StageStatus.DEGRADED
            result.warn(f"speaker_{track.ordinal}_extraction_failed")
        if track.modality is Modality.AUDIO_ONLY:
            result.warn(f"speaker_{track.ordinal}_no_face_track")

        speakers.append(
            {
                "id": track.speaker_id,
                "ordinal": track.ordinal,
                "label": track.label,
                "color_token": f"spk-{track.ordinal}",
                "modality": track.modality.value,
                "speaking_ratio": round(track.speaking_ratio, 4),
                "mean_confidence": round(track.mean_confidence, 4),
                "extraction_ok": track.extraction_ok,
                "audio": audio,
                "captions": captions,
            }
        )

    manifest: dict[str, Any] = {
        "project_id": project_id,
        "manifest_version": MANIFEST_VERSION,
        "duration_ms": duration_ms,
        "has_video": has_video,
        "speakers": speakers,
        # docs/12 §2: the engine decision is made here, server-side, so the
        # duration and track-count thresholds are not duplicated in the client.
        "playback_hint": choose_engine(duration_ms, len(speakers)),
        "warnings": list(result.warnings),
        "signed_until": signed_until,
    }
    if has_video and video_url is not None and video_size is not None:
        manifest["video"] = {
            "url": video_url,
            "width": video_size[0],
            "height": video_size[1],
        }
    if overlap_ratio is not None:
        manifest["overlap_ratio"] = round(overlap_ratio, 4)
        manifest["difficulty"] = _difficulty(overlap_ratio, len(tracks))

    if manifest["playback_hint"] == "hls":
        try:
            hls = package_hls(
                out_dir,
                tracks=[(t.label, out_dir / f"spk_{t.ordinal}_f.m4a") for t in tracks],
                captions=[(t.label, f"spk_{t.ordinal}.vtt") for t in tracks],
                duration_ms=duration_ms,
            )
            manifest["hls"] = {"master": f"{base_url}{hls.master.name}"}
        except Exception as exc:
            # A failed HLS build should not lose the project: Web Audio still
            # works, it just costs more memory on a long recording.
            result.status = StageStatus.DEGRADED
            result.warn("hls_packaging_failed")
            manifest["playback_hint"] = "webaudio"
            result.detail = f"{type(exc).__name__}: {exc}"

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    result.seconds = time.perf_counter() - t0
    result.detail = f"{len(speakers)} speakers, {duration_ms} ms"
    return manifest, result


def _difficulty(overlap_ratio: float, n_speakers: int) -> str:
    """Coarse bucket surfaced to the user so expectations match reality."""
    if overlap_ratio < 0.10 and n_speakers <= 2:
        return "easy"
    if overlap_ratio < 0.25 and n_speakers <= 3:
        return "moderate"
    return "hard"
