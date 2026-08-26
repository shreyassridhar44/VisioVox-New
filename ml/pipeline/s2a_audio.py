"""S2A — Audio analysis: VAD, diarization, overlap detection (docs/05 §4).

Produces the speaker registry the rest of the pipeline is organised around:
who spoke, when, and where two people spoke at once.

Overlap detection is not incidental. ADR-0010 routes single-talker regions
around the extractor entirely, because separating already-clean audio degrades
it — the Phase 1 baseline measured that as a real cost, not a theoretical one.
So the overlap map decides what work gets done at all.

Speaker embeddings are ephemeral (invariant 3, ADR-0008). This module returns
them in memory for S3 to consume; nothing here writes them to disk, and the
caller must not persist them unless the user opted in.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .types import (
    ANALYSIS_SAMPLE_RATE,
    Interval,
    StageResult,
    StageStatus,
)

STAGE = "S2A_audio"
VERSION = "1.0.0"

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
FRAME_SAMPLES = 160  # 10 ms


@dataclass
class SpeakerTurn:
    """One diarized turn. Sample-accurate, never float seconds (invariant 2)."""

    speaker: str
    interval: Interval


@dataclass
class AudioAnalysis:
    """S2A output."""

    speech: list[Interval]
    turns: list[SpeakerTurn]
    overlap: list[Interval]
    speakers: list[str]
    total_samples: int

    @property
    def overlap_ratio(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return sum(i.length for i in self.overlap) / self.total_samples

    def speaking_samples(self, speaker: str) -> int:
        return sum(t.interval.length for t in self.turns if t.speaker == speaker)

    def speaking_ratio(self, speaker: str) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.speaking_samples(speaker) / self.total_samples

    def single_talker(self) -> list[Interval]:
        """Speech regions with exactly one active speaker — ADR-0010 passthrough."""
        return _subtract(self.speech, self.overlap)


def _merge(intervals: list[Interval], gap: int = 0) -> list[Interval]:
    """Merge overlapping or near-touching intervals."""
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda i: i.start)
    out = [ordered[0]]
    for iv in ordered[1:]:
        last = out[-1]
        if iv.start <= last.end + gap:
            out[-1] = Interval(last.start, max(last.end, iv.end))
        else:
            out.append(iv)
    return out


def _subtract(base: list[Interval], holes: list[Interval]) -> list[Interval]:
    """base minus holes, both assumed sorted and non-overlapping after merge."""
    result: list[Interval] = []
    holes = _merge(holes)
    for iv in _merge(base):
        cursor = iv.start
        for h in holes:
            if h.end <= cursor or h.start >= iv.end:
                continue
            if h.start > cursor:
                result.append(Interval(cursor, min(h.start, iv.end)))
            cursor = max(cursor, h.end)
            if cursor >= iv.end:
                break
        if cursor < iv.end:
            result.append(Interval(cursor, iv.end))
    return [i for i in result if i.length > 0]


def find_overlap(turns: list[SpeakerTurn], min_samples: int = 1600) -> list[Interval]:
    """Regions where two or more distinct speakers are active simultaneously.

    Computed by sweeping turn boundaries rather than by pairwise intersection,
    so it is linear in the number of turns and correct for three or more
    concurrent speakers, not just two.

    Spans under `min_samples` (100 ms) are dropped: diarization boundaries are
    not sample-exact, and treating a 20 ms boundary wobble as overlap would
    route clean audio through the extractor for no reason.
    """
    events: list[tuple[int, int]] = []
    for t in turns:
        events.append((t.interval.start, 1))
        events.append((t.interval.end, -1))
    events.sort()

    out: list[Interval] = []
    active = 0
    region_start: int | None = None
    for position, delta in events:
        was_overlapping = active >= 2
        active += delta
        now_overlapping = active >= 2
        if not was_overlapping and now_overlapping:
            region_start = position
        elif was_overlapping and not now_overlapping and region_start is not None:
            if position - region_start >= min_samples:
                out.append(Interval(region_start, position))
            region_start = None
    return _merge(out)


def analyse(
    audio: np.ndarray,
    hf_token: str | None = None,
    device: str = "cuda",
) -> tuple[AudioAnalysis, StageResult]:
    """Run VAD and diarization over the analysis audio.

    Degrades rather than aborts (invariant 8): if diarization is unavailable the
    stage still returns VAD output, marked DEGRADED, and downstream can fall
    back to single-speaker handling.
    """
    import torch

    t0 = time.perf_counter()
    result = StageResult(stage=STAGE, status=StageStatus.OK)
    total = len(audio)

    # --- VAD (Silero, MIT, no gating) ---
    from silero_vad import get_speech_timestamps, load_silero_vad

    vad = load_silero_vad()
    stamps = get_speech_timestamps(torch.from_numpy(audio), vad, sampling_rate=ANALYSIS_SAMPLE_RATE)
    speech = _merge([Interval(int(s["start"]), int(s["end"])) for s in stamps])

    if not speech:
        result.status = StageStatus.DEGRADED
        result.warn("no_speech_detected")
        result.seconds = time.perf_counter() - t0
        return AudioAnalysis([], [], [], [], total), result

    # --- diarization (pyannote, gated) ---
    turns: list[SpeakerTurn] = []
    if not hf_token:
        result.status = StageStatus.DEGRADED
        result.warn("diarization_unavailable_no_token")
    else:
        try:
            turns = _diarize(audio, hf_token, device)
        except Exception as exc:
            result.status = StageStatus.DEGRADED
            result.warn("diarization_failed")
            result.detail = f"{type(exc).__name__}: {exc}"

    speakers = sorted({t.speaker for t in turns})
    overlap = find_overlap(turns)

    analysis = AudioAnalysis(
        speech=speech,
        turns=turns,
        overlap=overlap,
        speakers=speakers,
        total_samples=total,
    )

    if len(speakers) > 8:
        result.warn("too_many_speakers")

    result.seconds = time.perf_counter() - t0
    if not result.detail:
        result.detail = (
            f"{len(speakers)} speakers, {len(turns)} turns, overlap {analysis.overlap_ratio:.1%}"
        )
    return analysis, result


def _diarize(audio: np.ndarray, hf_token: str, device: str) -> list[SpeakerTurn]:
    """Run pyannote and normalise its output to sample-accurate turns."""
    import torch
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation

    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=hf_token)
    if pipeline is None:
        raise PermissionError(
            "pyannote returned no pipeline — accept the terms for "
            "pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0"
        )
    pipeline.to(torch.device(device))

    # Pass the waveform rather than a path: pyannote decodes files via
    # torchcodec, whose shared library must match the FFmpeg torch was built
    # against. We already have the samples.
    out = pipeline(
        {
            "waveform": torch.from_numpy(audio).unsqueeze(0),
            "sample_rate": ANALYSIS_SAMPLE_RATE,
        }
    )
    # pyannote 4.x wraps the annotation; 3.x returned it directly.
    annotation = getattr(out, "speaker_diarization", out)
    if not isinstance(annotation, Annotation):
        raise RuntimeError(f"unexpected diarization result: {type(annotation).__name__}")

    turns: list[SpeakerTurn] = []
    for segment, _, label in annotation.itertracks(yield_label=True):
        start = round(segment.start * ANALYSIS_SAMPLE_RATE)
        end = round(segment.end * ANALYSIS_SAMPLE_RATE)
        if end > start:
            turns.append(SpeakerTurn(speaker=str(label), interval=Interval(start, end)))
    return sorted(turns, key=lambda t: t.interval.start)
