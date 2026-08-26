"""Phase 1 end-to-end pipeline run (docs/21 Phase 1 exit criterion).

Chains S0 -> S2A -> S2B -> S3 -> S5 -> S7 -> S9 on real video and validates the
resulting manifest against the frozen contract. This is the run that proves the
plumbing holds end to end, not that the model is good.

ADR-0010 routing is applied here. The Phase 1 baseline measured that running a
separator across an entire timeline is *worse than doing nothing* -- and AMI is
~90% single-talker, so most of that damage is done to audio that never needed
separating. Each speaker track is therefore assembled as:

    single-talker region where they are the active speaker -> mixture verbatim
    overlap region                                          -> separated source
    everywhere else                                         -> silence

That is the design the docs already specify, and it is what makes the Tier 0
number a fair baseline rather than a strawman.

Usage:  uv run python scripts/run_pipeline.py [clip ...]
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from jsonschema import Draft202012Validator, FormatChecker

from pipeline.s0_ingest import ingest
from pipeline.s2a_audio import AudioAnalysis, analyse
from pipeline.s2b_video import analyse_video
from pipeline.s3_fuse import SpeakerRegistry, fuse
from pipeline.s5_separate import load_separator, overlap_add, separate_windows
from pipeline.s7_transcribe import load_transcriber, transcribe
from pipeline.s9_package import SpeakerTrack, package
from pipeline.types import (
    ANALYSIS_SAMPLE_RATE,
    Interval,
    StageResult,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = REPO_ROOT / "packages" / "contracts" / "schemas" / "manifest.schema.json"
CLIPS = Path.home() / "data" / "testvideos" / "clips"
WORK = Path.home() / "data" / "pipeline"

RATE = ANALYSIS_SAMPLE_RATE
MAX_VIDEO_FRAMES = 250  # face detection is CPU-bound here; 10 s is enough to bind


@dataclass
class RunReport:
    clip: str
    stages: list[StageResult]
    manifest_valid: bool
    errors: list[str]
    n_speakers: int
    passthrough_ratio: float


def _mask_from(intervals: list[Interval], n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for iv in intervals:
        mask[max(0, iv.start) : min(n, iv.end)] = True
    return mask


def build_speaker_audio(
    mixture: np.ndarray,
    separated: np.ndarray,
    analysis: AudioAnalysis,
    registry: SpeakerRegistry,
) -> dict[str, np.ndarray]:
    """Assemble per-speaker tracks under ADR-0010 routing.

    All returned tracks have exactly len(mixture) samples, which is what
    invariant 1 requires and what S9 will refuse to package without.
    """
    n = len(mixture)
    overlap_mask = _mask_from(analysis.overlap, n)
    tracks: dict[str, np.ndarray] = {}

    for i, binding in enumerate(registry.bindings):
        out = np.zeros(n, dtype=np.float32)

        own_turns = [t.interval for t in analysis.turns if t.speaker == binding.speaker]
        own_mask = _mask_from(own_turns, n)

        # 1. single-talker: hand back the original audio untouched
        passthrough = own_mask & ~overlap_mask
        out[passthrough] = mixture[passthrough]

        # 2. overlap: take a separated source, if one is available for them
        if i < separated.shape[0]:
            contested = own_mask & overlap_mask
            out[contested] = separated[i][contested]

        tracks[binding.speaker] = out

    return tracks


def run_clip(clip_dir: Path, separator: object, transcriber: object, hf_token: str) -> RunReport:
    name = clip_dir.name
    src = clip_dir / "input.mp4"
    stages: list[StageResult] = []
    print(f"[{name}]")

    # --- S0 ---
    media, r0 = ingest(src, WORK / name)
    stages.append(r0)
    print(f"  S0  {r0.status:8} {r0.detail}")

    mixture, _ = sf.read(media.analysis_wav, dtype="float32")
    mixture = np.asarray(mixture, dtype=np.float32)
    n = len(mixture)

    # --- S2A ---
    analysis, r2a = analyse(mixture, hf_token)
    stages.append(r2a)
    print(f"  S2A {r2a.status:8} {r2a.detail}")

    # --- S2B ---
    video = None
    if media.video_mp4 is not None:
        video, r2b = analyse_video(media.video_mp4, max_frames=MAX_VIDEO_FRAMES)
        stages.append(r2b)
        print(f"  S2B {r2b.status:8} {r2b.detail}")

    # --- S3 ---
    registry, r3 = fuse(analysis, video)
    stages.append(r3)
    print(f"  S3  {r3.status:8} {r3.detail}")

    # --- S5 ---
    sep = separate_windows(mixture, separator)  # type: ignore[arg-type]
    from pipeline.s5_separate import identity_assignment

    separated = overlap_add(sep, identity_assignment(sep), n)
    r5 = StageResult(stage="S5_separate", status=r0.status.OK)
    r5.detail = f"{sep.n_windows} windows x {sep.n_sources} sources"
    stages.append(r5)
    print(f"  S5  {r5.status:8} {r5.detail}")

    speaker_audio = build_speaker_audio(mixture, separated, analysis, registry)
    overlap_samples = sum(i.length for i in analysis.overlap)
    passthrough_ratio = 1.0 - (overlap_samples / n if n else 0.0)

    # --- S7 + S9 ---
    tracks: list[SpeakerTrack] = []
    for binding in registry.bindings:
        audio = speaker_audio[binding.speaker]
        transcript = None
        if np.abs(audio).max() > 1e-4:
            transcript, r7 = transcribe(audio, transcriber)  # type: ignore[arg-type]
            stages.append(r7)
        tracks.append(
            SpeakerTrack(
                speaker_id=f"spk_{_ulid()}",
                ordinal=binding.ordinal,
                label=binding.label,
                faithful=audio,
                modality=binding.modality,
                speaking_ratio=binding.speaking_ratio,
                mean_confidence=max(0.0, min(1.0, 0.5 + binding.agreement)),
                extraction_ok=bool(np.abs(audio).max() > 1e-4),
                transcript=transcript,
            )
        )

    signed_until = (dt.datetime.now(dt.UTC) + dt.timedelta(minutes=15)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    manifest, r9 = package(
        f"prj_{_ulid()}",
        tracks,
        clip_dir / "pipeline",
        has_video=media.probe.has_video,
        video_url="https://cdn.example/video.mp4?sig=demo" if media.probe.has_video else None,
        video_size=(media.probe.width or 0, media.probe.height or 0)
        if media.probe.has_video
        else None,
        overlap_ratio=analysis.overlap_ratio,
        signed_until=signed_until,
        base_url="https://cdn.example/",
    )
    stages.append(r9)
    print(f"  S9  {r9.status:8} {r9.detail}")

    validator = Draft202012Validator(json.loads(SCHEMA.read_text()), format_checker=FormatChecker())
    errors = [f"{e.json_path}: {e.message}" for e in validator.iter_errors(manifest)]
    print(f"  manifest {'VALID' if not errors else 'INVALID'}")
    for e in errors[:5]:
        print(f"    {e}")

    return RunReport(
        clip=name,
        stages=stages,
        manifest_valid=not errors,
        errors=errors,
        n_speakers=len(tracks),
        passthrough_ratio=passthrough_ratio,
    )


def _ulid() -> str:
    from ulid import ULID

    return str(ULID())


def main(argv: list[str]) -> int:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("HF_TOKEN not set; S2A diarization will degrade")

    dirs = [CLIPS / a for a in argv] if argv else sorted(d for d in CLIPS.iterdir() if d.is_dir())
    if not dirs:
        print(f"no clips under {CLIPS}")
        return 2

    WORK.mkdir(parents=True, exist_ok=True)
    print("loading models ...")
    separator = load_separator(cache=REPO_ROOT / "models" / "sepformer16k")
    transcriber = load_transcriber(cache=REPO_ROOT / "models" / "whisper")

    reports = [
        run_clip(d, separator, transcriber, token) for d in dirs if (d / "input.mp4").exists()
    ]
    if not reports:
        print("no clips had input.mp4")
        return 1

    ok = sum(1 for r in reports if r.manifest_valid)
    print(f"=== {ok}/{len(reports)} clips produced a schema-valid manifest ===")
    for r in reports:
        print(
            f"  {r.clip}: {r.n_speakers} speakers, "
            f"{r.passthrough_ratio:.0%} of timeline passed through (ADR-0010)"
        )
    return 0 if ok == len(reports) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
