"""S3 — Cross-modal fusion: bind voice clusters to face tracks (docs/05 §6).

Our algorithm, no weights. For every (voice cluster, face track) pair we score
how well the face was visibly active while that voice was speaking, then solve
the assignment globally with the Hungarian algorithm.

Global assignment rather than greedy per-speaker matching: greedy lets one
dominant speaker claim the best-correlated face and pushes everyone else onto
whatever remains, which is how two quiet participants end up swapped. The
mistake is not recoverable downstream because the extractor is conditioned on
the binding.

A pair is only accepted above `min_agreement`. Below that the speaker is
marked AUDIO_ONLY rather than bound to a doubtful face — a wrong face is worse
than no face, because it feeds the visual pathway a confident lie
(Novelty 2 reliability gating rests on this).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .s2a_audio import AudioAnalysis
from .s2b_video import VideoAnalysis
from .types import FRAME_RATE, Modality, StageResult, StageStatus

STAGE = "S3_fuse"
VERSION = "1.0.0"

MIN_AGREEMENT = 0.15
ACTIVITY_THRESHOLD = 0.25  # normalised motion score counted as "visibly active"


@dataclass
class SpeakerBinding:
    """One entry in the speaker registry."""

    speaker: str
    ordinal: int
    face_track_id: int | None
    modality: Modality
    agreement: float
    speaking_ratio: float

    @property
    def label(self) -> str:
        return f"Speaker {self.ordinal}"


@dataclass
class SpeakerRegistry:
    bindings: list[SpeakerBinding]

    def by_speaker(self, speaker: str) -> SpeakerBinding | None:
        return next((b for b in self.bindings if b.speaker == speaker), None)

    @property
    def audiovisual_count(self) -> int:
        return sum(1 for b in self.bindings if b.modality is Modality.AUDIOVISUAL)


def _speaking_frames(
    analysis: AudioAnalysis, speaker: str, n_frames: int, sample_rate: int = 16_000
) -> np.ndarray:
    """Boolean per-frame mask of when this voice cluster was speaking."""
    mask = np.zeros(n_frames, dtype=bool)
    for turn in analysis.turns:
        if turn.speaker != speaker:
            continue
        start = turn.interval.start * FRAME_RATE // sample_rate
        end = turn.interval.end * FRAME_RATE // sample_rate
        mask[max(0, start) : min(n_frames, end)] = True
    return mask


def agreement_score(voice: np.ndarray, activity: np.ndarray) -> float:
    """Balanced agreement between a voice mask and a face-activity trace.

    Not raw overlap: a face that moves constantly would score highly against
    every speaker. This is the mean of true-positive rate and true-negative
    rate, so a face must be active *while* the voice speaks and quiet while it
    does not. A constantly-moving face scores ~0.5 and loses to a real match.
    """
    visible = activity >= ACTIVITY_THRESHOLD
    speaking = voice
    silent = ~voice

    if speaking.sum() == 0 or silent.sum() == 0:
        return 0.0

    tpr = float((visible & speaking).sum()) / float(speaking.sum())
    tnr = float((~visible & silent).sum()) / float(silent.sum())
    return (tpr + tnr) / 2.0 - 0.5  # centre on 0 so chance agreement scores 0


def fuse(
    audio: AudioAnalysis,
    video: VideoAnalysis | None,
    min_agreement: float = MIN_AGREEMENT,
) -> tuple[SpeakerRegistry, StageResult]:
    """Build the speaker registry."""
    t0 = time.perf_counter()
    result = StageResult(stage=STAGE, status=StageStatus.OK)

    speakers = sorted(audio.speakers, key=lambda s: audio.speaking_samples(s), reverse=True)
    if not speakers:
        result.status = StageStatus.DEGRADED
        result.warn("no_speakers_to_bind")
        result.seconds = time.perf_counter() - t0
        return SpeakerRegistry([]), result

    tracks = video.tracks if video is not None else []
    if not tracks:
        result.status = StageStatus.DEGRADED
        result.warn("no_face_tracks_to_bind")
        audio_only: list[SpeakerBinding] = [
            SpeakerBinding(
                speaker=s,
                ordinal=i + 1,
                face_track_id=None,
                modality=Modality.AUDIO_ONLY,
                agreement=0.0,
                speaking_ratio=audio.speaking_ratio(s),
            )
            for i, s in enumerate(speakers)
        ]
        result.seconds = time.perf_counter() - t0
        result.detail = f"{len(audio_only)} speakers, audio-only"
        return SpeakerRegistry(audio_only), result

    n_frames = video.n_frames if video is not None else 0
    scores = np.zeros((len(speakers), len(tracks)))
    for i, speaker in enumerate(speakers):
        voice = _speaking_frames(audio, speaker, n_frames)
        for j, track in enumerate(tracks):
            activity = track.activity
            if activity is None or len(activity) != n_frames:
                continue
            scores[i, j] = agreement_score(voice, activity)

    # Hungarian maximises total agreement across all pairs at once.
    rows, cols = linear_sum_assignment(-scores)
    chosen = dict(zip(rows.tolist(), cols.tolist(), strict=True))

    bindings: list[SpeakerBinding] = []
    for i, speaker in enumerate(speakers):
        match_idx = chosen.get(i)
        score = float(scores[i, match_idx]) if match_idx is not None else 0.0
        if match_idx is None or score < min_agreement:
            # A wrong face is worse than no face: it feeds the visual pathway a
            # confident lie, which reliability gating cannot detect.
            bindings.append(
                SpeakerBinding(
                    speaker=speaker,
                    ordinal=i + 1,
                    face_track_id=None,
                    modality=Modality.AUDIO_ONLY,
                    agreement=score,
                    speaking_ratio=audio.speaking_ratio(speaker),
                )
            )
            result.warn(f"speaker_{i + 1}_no_face_track")
        else:
            bindings.append(
                SpeakerBinding(
                    speaker=speaker,
                    ordinal=i + 1,
                    face_track_id=tracks[match_idx].track_id,
                    modality=Modality.AUDIOVISUAL,
                    agreement=score,
                    speaking_ratio=audio.speaking_ratio(speaker),
                )
            )

    registry = SpeakerRegistry(bindings)
    if registry.audiovisual_count == 0:
        result.status = StageStatus.DEGRADED
        result.warn("all_speakers_audio_only")

    result.seconds = time.perf_counter() - t0
    result.detail = f"{len(bindings)} speakers, {registry.audiovisual_count} bound to faces"
    return registry, result
