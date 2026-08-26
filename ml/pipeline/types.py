"""Shared pipeline types.

Timing rule (invariant 2, docs/05 §2): every duration and offset in this
package is **samples at 16 kHz** or **integer milliseconds**. Float seconds are
converted only at the API boundary. Float-seconds arithmetic accumulates error
across a 60-minute timeline and is the classic source of caption drift, so the
types here make the unit explicit rather than relying on discipline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Self

ANALYSIS_SAMPLE_RATE = 16_000
FRAME_RATE = 25


def samples_to_ms(samples: int, rate: int = ANALYSIS_SAMPLE_RATE) -> int:
    """Convert samples to integer milliseconds, rounding half away from zero."""
    return (samples * 1000 + rate // 2) // rate


def ms_to_samples(ms: int, rate: int = ANALYSIS_SAMPLE_RATE) -> int:
    return (ms * rate + 500) // 1000


class StageStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"  # produced output, but with caveats worth surfacing
    FAILED = "failed"


class Modality(StrEnum):
    AUDIOVISUAL = "audiovisual"
    AUDIO_ONLY = "audio_only"
    VISUAL_ONLY = "visual_only"


@dataclass(frozen=True, slots=True)
class Interval:
    """A half-open span [start, end) in samples at 16 kHz."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"interval end {self.end} precedes start {self.start}")

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: Self) -> bool:
        return self.start < other.end and other.start < self.end

    def intersection(self, other: Self) -> int:
        return max(0, min(self.end, other.end) - max(self.start, other.start))

    def to_ms(self) -> tuple[int, int]:
        return samples_to_ms(self.start), samples_to_ms(self.end)


@dataclass(frozen=True, slots=True)
class Probe:
    """What ffprobe found. Kept as a record so validation failures are explainable."""

    duration_ms: int
    has_audio: bool
    has_video: bool
    width: int | None
    height: int | None
    frame_rate: float | None
    audio_channels: int | None
    audio_sample_rate: int | None


@dataclass(frozen=True, slots=True)
class MediaSet:
    """S0 output: the normalised inputs every later stage reads."""

    root: Path
    analysis_wav: Path
    reference_wav: Path
    video_mp4: Path | None
    probe: Probe

    @property
    def duration_samples(self) -> int:
        return ms_to_samples(self.probe.duration_ms)


@dataclass
class StageResult:
    """Uniform stage envelope.

    Invariant 8: partial failure yields partial results. A stage reports
    DEGRADED and carries on rather than aborting the job, and the warnings it
    emits become manifest warnings the UI can explain.
    """

    stage: str
    status: StageStatus
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)
    detail: str = ""

    def warn(self, token: str) -> None:
        """Warnings are snake_case tokens the UI maps to copy, never prose."""
        if not token.replace("_", "").isalnum() or token != token.lower():
            raise ValueError(f"warning must be lower snake_case token, got {token!r}")
        if token not in self.warnings:
            self.warnings.append(token)
