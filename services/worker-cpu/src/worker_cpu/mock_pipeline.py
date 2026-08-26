"""Mock pipeline worker (docs/17 §2, docs/21 Phase 2).

Emits a schema-valid artifact manifest with realistic per-stage timing, without
a GPU. This is the component that takes the entire application off the ML
critical path — MEMORY.md calls it the highest-leverage decision in the
schedule — and it only works if what it emits is indistinguishable in *shape*
from what the real pipeline produces.

So the manifest here is built to the same frozen contract in
packages/contracts, and the same contract test runs against both. If the mock
drifts, CI fails rather than the integration in month five.

Stage durations are proportional to media length using the real measured
ratios from docs/05 §13, so progress feels like the real thing rather than a
uniform bar: video analysis really does dominate, and a UI tuned against a
fake uniform pipeline will feel wrong the first time it meets a real job.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import random
from collections.abc import Iterator
from dataclasses import dataclass

STAGES: tuple[tuple[str, int, float], ...] = (
    # (stage, ordinal, share of total runtime)
    ("S0_ingest", 0, 0.04),
    ("S1_enhance", 1, 0.08),
    ("S2A_audio", 2, 0.14),
    ("S2B_video", 3, 0.28),  # video analysis dominates — docs/05 §13
    ("S3_fuse", 4, 0.02),
    ("S4_enrol", 5, 0.03),
    ("S5_extract", 6, 0.17),
    ("S6_restore", 7, 0.05),
    ("S7_transcribe", 8, 0.11),
    ("S8_audit", 9, 0.04),
    ("S9_package", 10, 0.04),
)

# Roughly 0.9x realtime end to end on the reference GPU.
REALTIME_FACTOR = 0.9


@dataclass(frozen=True)
class StagePlan:
    stage: str
    ordinal: int
    duration_ms: int
    progress_after: int


def plan_stages(duration_ms: int) -> list[StagePlan]:
    """Per-stage durations and the progress value each one lands on."""
    total = max(1, int(duration_ms * REALTIME_FACTOR))
    plans: list[StagePlan] = []
    elapsed = 0
    for stage, ordinal, share in STAGES:
        d = max(1, int(total * share))
        elapsed += d
        plans.append(
            StagePlan(
                stage=stage,
                ordinal=ordinal,
                duration_ms=d,
                progress_after=min(100, round(100 * elapsed / total)),
            )
        )
    # Guarantee the last stage lands exactly on 100 rather than 99 from rounding.
    if plans:
        last = plans[-1]
        plans[-1] = StagePlan(last.stage, last.ordinal, last.duration_ms, 100)
    return plans


def iter_stages(duration_ms: int) -> Iterator[StagePlan]:
    yield from plan_stages(duration_ms)


def _rng(project_id: str) -> random.Random:
    """Deterministic per project, so a re-run of the same job looks identical.

    A mock that returns different speaker counts on every retry makes UI bugs
    unreproducible, which defeats the purpose of having one.
    """
    seed = int(hashlib.sha256(project_id.encode()).hexdigest()[:8], 16)
    # S311: this is fixture generation, not security. Determinism is the
    # entire point — a CSPRNG here would make mock jobs unreproducible.
    return random.Random(seed)  # noqa: S311


def build_manifest(
    project_id: str,
    duration_ms: int,
    *,
    has_video: bool = True,
    base_url: str = "https://cdn.local/mock/",
    signed_ttl_seconds: int = 900,
    speaker_count: int | None = None,
) -> dict[str, object]:
    """A schema-valid manifest with plausible values.

    Deliberately conforms to packages/contracts/schemas/manifest.schema.json —
    the same document the real S9 emits against.
    """
    rng = _rng(project_id)
    n = speaker_count if speaker_count is not None else rng.choice([2, 2, 3])
    overlap = round(rng.uniform(0.05, 0.28), 4)

    ordinals = list(range(1, n + 1))
    raw_shares = [rng.uniform(0.2, 1.0) for _ in ordinals]
    total_share = sum(raw_shares)

    speakers: list[dict[str, object]] = []
    warnings: list[str] = []
    for i, ordinal in enumerate(ordinals):
        # The last speaker in a 3-way mock has no face track, so the UI is
        # exercised against a mixed-modality project rather than the easy case.
        audiovisual = has_video and not (n >= 3 and ordinal == n)
        if not audiovisual and has_video:
            warnings.append(f"speaker_{ordinal}_no_face_track")

        speaker: dict[str, object] = {
            "id": f"spk_{_fake_ulid(rng)}",
            "ordinal": ordinal,
            "label": f"Speaker {ordinal}",
            "color_token": f"spk-{ordinal}",
            "modality": "audiovisual" if audiovisual else "audio_only",
            "speaking_ratio": round(raw_shares[i] / total_share, 4),
            "mean_confidence": round(rng.uniform(0.62, 0.94), 4),
            "extraction_ok": True,
            "audio": {
                "faithful": {
                    "url": f"{base_url}spk_{ordinal}_f.m4a",
                    "bytes": int(duration_ms * 16),
                },
                "natural": {
                    "url": f"{base_url}spk_{ordinal}_n.m4a",
                    "bytes": int(duration_ms * 16),
                },
            },
            "peaks_url": f"{base_url}spk_{ordinal}.peaks.json",
            "captions": {
                "vtt": f"{base_url}spk_{ordinal}.vtt",
                "json": f"{base_url}spk_{ordinal}.json",
            },
        }
        if audiovisual:
            speaker["thumbnail_url"] = f"{base_url}spk_{ordinal}.webp"
        speakers.append(speaker)

    signed_until = (dt.datetime.now(dt.UTC) + dt.timedelta(seconds=signed_ttl_seconds)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    manifest: dict[str, object] = {
        "project_id": project_id,
        "manifest_version": "1.0",
        "duration_ms": duration_ms,
        "has_video": has_video,
        "overlap_ratio": overlap,
        "difficulty": "easy" if overlap < 0.10 and n <= 2 else ("moderate" if n <= 3 else "hard"),
        "speakers": speakers,
        "mixed": {"audio_url": f"{base_url}mixed.m4a"},
        "playback_hint": "webaudio",
        "warnings": warnings,
        "signed_until": signed_until,
    }
    if has_video:
        manifest["video"] = {"url": f"{base_url}video.mp4", "width": 1920, "height": 1080}
    return manifest


_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _fake_ulid(rng: random.Random) -> str:
    return "".join(rng.choice(_CROCKFORD) for _ in range(26))
