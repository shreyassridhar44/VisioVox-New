"""S4 — Self-enrolment from diarization (SEAVE-SE, docs/04 §2).

Every TSE system in the research landscape assumes an externally supplied clean
enrolment recording. Real users have none, and that is the reason target
speaker extraction has essentially no consumer products despite a decade of
strong results. This stage mines the enrolment from the recording itself.

Two details carry the contribution:

**Purity weighting, not longest-segment.** The intuitive choice is the longest
clean stretch. It is worse: one long region captures a single prosodic context
— one sentence, one emotional register — and generalises poorly. A weighted
aggregate over several diverse regions lands closer to the speaker's true
centroid.

**The confidences are the interface to Contribution 2.** `audio_cue_confidence`
and `visual_cue_confidence` are not diagnostics to log; they are the routing
signal the reliability gate consumes. A cue that is wrong but confident is
worse than an absent one, so they are computed conservatively.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .s2a_audio import AudioAnalysis
from .types import ANALYSIS_SAMPLE_RATE, Interval, StageResult, StageStatus

STAGE = "S4_enrol"
VERSION = "1.0.0"

MIN_REGION_SAMPLES = int(1.5 * ANALYSIS_SAMPLE_RATE)  # 1.5 s
TARGET_AGGREGATE_SAMPLES = int(8.0 * ANALYSIS_SAMPLE_RATE)  # 8 s
MAX_REGIONS = 8


@dataclass(frozen=True)
class PurityWeights:
    """Term weights for the purity score (docs/04 §2)."""

    overlap_free: float = 0.35
    snr: float = 0.20
    speech_density: float = 0.20
    embedding_agreement: float = 0.25
    reverb_penalty: float = 0.15


@dataclass
class Candidate:
    """One single-talker region considered for enrolment."""

    interval: Interval
    overlap_free: float = 0.0
    snr_norm: float = 0.0
    speech_density: float = 0.0
    embedding_agreement: float = 0.0
    reverb: float = 0.0
    embedding: np.ndarray | None = None

    def purity(self, w: PurityWeights) -> float:
        score = (
            w.overlap_free * self.overlap_free
            + w.snr * self.snr_norm
            + w.speech_density * self.speech_density
            + w.embedding_agreement * self.embedding_agreement
            - w.reverb_penalty * self.reverb
        )
        return float(np.clip(score, 0.0, 1.0))


@dataclass
class Enrolment:
    """The cue set for one speaker."""

    speaker: str
    audio_embedding: np.ndarray | None
    audio_cue_confidence: float
    visual_regions: list[Interval] = field(default_factory=list)
    visual_cue_confidence: float = 0.0
    regions_used: list[Interval] = field(default_factory=list)
    aggregate_samples: int = 0

    @property
    def has_audio_cue(self) -> bool:
        return self.audio_embedding is not None and self.audio_cue_confidence > 0.0


def _normalise(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _snr_estimate(audio: np.ndarray) -> float:
    """Crude segmental SNR proxy in 0..1.

    Ratio of loud-frame energy to quiet-frame energy. Not calibrated dB — it
    only has to rank regions against each other, and a real SNR estimator would
    be a dependency for no extra ordering information.
    """
    if len(audio) < 320:
        return 0.0
    frames = audio[: len(audio) // 160 * 160].reshape(-1, 160)
    energy = (frames**2).mean(axis=1)
    if energy.size < 4:
        return 0.0
    loud = float(np.percentile(energy, 90))
    quiet = float(np.percentile(energy, 10))
    if quiet < 1e-12:
        return 1.0
    snr_db = 10 * np.log10((loud + 1e-12) / quiet)
    return float(np.clip(snr_db / 40.0, 0.0, 1.0))


def _reverb_proxy(audio: np.ndarray) -> float:
    """Decay-tail proxy in 0..1; higher means more reverberant.

    Measures how slowly energy falls after peaks. Reverberant enrolment audio
    smears the speaker's spectral signature, which is exactly what the
    embedding is supposed to capture.
    """
    if len(audio) < 1600:
        return 0.0
    frames = audio[: len(audio) // 160 * 160].reshape(-1, 160)
    energy = np.sqrt((frames**2).mean(axis=1) + 1e-12)
    if len(energy) < 8:
        return 0.0
    peaks = energy > np.percentile(energy, 75)
    idx = np.flatnonzero(peaks[:-3])
    if idx.size == 0:
        return 0.0
    decay = energy[idx + 3] / (energy[idx] + 1e-12)
    return float(np.clip(np.median(decay), 0.0, 1.0))


def find_candidates(
    analysis: AudioAnalysis, speaker: str, min_samples: int = MIN_REGION_SAMPLES
) -> list[Candidate]:
    """Single-talker regions belonging to this speaker, long enough to be useful."""
    own = [t.interval for t in analysis.turns if t.speaker == speaker]
    if not own:
        return []

    overlap = analysis.overlap
    out: list[Candidate] = []
    for iv in own:
        # subtract overlap from the turn, keeping the clean fragments
        cursor = iv.start
        pieces: list[Interval] = []
        for o in overlap:
            if o.end <= cursor or o.start >= iv.end:
                continue
            if o.start > cursor:
                pieces.append(Interval(cursor, min(o.start, iv.end)))
            cursor = max(cursor, o.end)
            if cursor >= iv.end:
                break
        if cursor < iv.end:
            pieces.append(Interval(cursor, iv.end))

        out.extend(Candidate(interval=p) for p in pieces if p.length >= min_samples)
    return out


def score_candidates(
    candidates: list[Candidate],
    audio: np.ndarray,
    embed: object | None = None,
    weights: PurityWeights | None = None,
) -> list[Candidate]:
    """Fill in the purity terms, and embeddings when an embedder is supplied."""
    w = weights or PurityWeights()
    if not candidates:
        return []

    vectors: list[np.ndarray] = []
    for c in candidates:
        seg = audio[c.interval.start : c.interval.end]
        c.overlap_free = 1.0  # candidates are already overlap-subtracted
        c.snr_norm = _snr_estimate(seg)
        c.reverb = _reverb_proxy(seg)
        c.speech_density = float(np.clip(np.mean(np.abs(seg) > 1e-3), 0.0, 1.0))

        if embed is not None:
            try:
                v = _normalise(np.asarray(embed(seg), dtype=np.float64).ravel())  # type: ignore[operator]
                c.embedding = v
                vectors.append(v)
            except Exception:
                c.embedding = None

    # Embedding agreement is measured against the centroid of the candidates
    # themselves, so a region that disagrees with the speaker's own consensus is
    # penalised — usually diarization having assigned it to the wrong person.
    if vectors:
        centroid = _normalise(np.mean(vectors, axis=0))
        for c in candidates:
            if c.embedding is not None:
                c.embedding_agreement = float(np.clip(np.dot(c.embedding, centroid), 0.0, 1.0))
    else:
        for c in candidates:
            c.embedding_agreement = 0.5  # neutral when no embedder is available

    return sorted(candidates, key=lambda c: c.purity(w), reverse=True)


def select_regions(
    scored: list[Candidate],
    target_samples: int = TARGET_AGGREGATE_SAMPLES,
    max_regions: int = MAX_REGIONS,
) -> list[Candidate]:
    """Take the purest regions until the aggregate duration target is met.

    Several diverse regions rather than one long one: a single stretch captures
    one prosodic context and generalises worse than a weighted aggregate.
    """
    chosen: list[Candidate] = []
    total = 0
    for c in scored:
        if len(chosen) >= max_regions:
            break
        chosen.append(c)
        total += c.interval.length
        if total >= target_samples:
            break
    return chosen


def aggregate_embedding(
    chosen: list[Candidate], weights: PurityWeights | None = None
) -> tuple[np.ndarray | None, float]:
    """Purity-weighted mean of L2-normalised embeddings, renormalised.

    Returns the cue and its confidence. Confidence reflects how much clean
    speech was found, how pure it was, and how consistent the regions are with
    each other — a cue built from one short disagreeing region should not be
    trusted at the same level as one from eight seconds of consistent audio.
    """
    w = weights or PurityWeights()
    usable = [c for c in chosen if c.embedding is not None]
    if not usable:
        return None, 0.0

    purities = np.array([c.purity(w) for c in usable], dtype=np.float64)
    if purities.sum() < 1e-9:
        purities = np.ones_like(purities)
    stacked = np.stack([c.embedding for c in usable])  # type: ignore[misc]
    cue = _normalise((purities[:, None] * stacked).sum(axis=0) / purities.sum())

    duration = sum(c.interval.length for c in usable)
    duration_factor = float(np.clip(duration / TARGET_AGGREGATE_SAMPLES, 0.0, 1.0))
    mean_purity = float(purities.mean())
    consistency = float(np.clip(np.mean([np.dot(v, cue) for v in stacked]), 0.0, 1.0))

    confidence = float(np.clip(duration_factor * mean_purity * consistency, 0.0, 1.0))
    return cue, confidence


def visual_confidence(
    face_size_px: float, frontality: float, sharpness: float, continuity: float
) -> float:
    """Confidence in the visual cue, 0..1.

    Blends the geometric mean with the worst factor, rather than averaging.
    A large, sharp, continuously-tracked face turned fully away is useless, and
    the aggregation has to say so: an arithmetic mean rates that 0.71, and a
    plain geometric mean still rates it 0.45 — both high enough to pass a gate
    it should fail. Folding in the minimum drops it to about 0.15 while leaving
    a genuinely good cue near 0.91.

    This is a routing signal for Contribution 2, not a diagnostic. A cue that is
    wrong but confident is worse than one that is absent, so the aggregation is
    deliberately pessimistic.
    """
    size = float(np.clip(face_size_px / 80.0, 0.0, 1.0))
    factors = np.array(
        [size, np.clip(frontality, 0, 1), np.clip(sharpness, 0, 1), np.clip(continuity, 0, 1)],
        dtype=np.float64,
    )
    if np.any(factors <= 1e-6):
        return 0.0
    geometric = float(np.exp(np.mean(np.log(factors))))
    worst = float(factors.min())
    return float(np.sqrt(geometric * worst))


def enrol(
    analysis: AudioAnalysis,
    audio: np.ndarray,
    speaker: str,
    embed: object | None = None,
    weights: PurityWeights | None = None,
) -> tuple[Enrolment, StageResult]:
    """Build the enrolment cue set for one speaker."""
    t0 = time.perf_counter()
    result = StageResult(stage=STAGE, status=StageStatus.OK)

    candidates = find_candidates(analysis, speaker)
    if not candidates:
        result.status = StageStatus.DEGRADED
        result.warn("no_clean_regions")
        result.seconds = time.perf_counter() - t0
        return Enrolment(speaker, None, 0.0), result

    scored = score_candidates(candidates, audio, embed, weights)
    chosen = select_regions(scored)
    cue, confidence = aggregate_embedding(chosen, weights)

    aggregate = sum(c.interval.length for c in chosen)
    if aggregate < MIN_REGION_SAMPLES:
        # docs/04 §2: under 1.5 s of pure speech, the audio cue is unreliable
        # and conditioning should fall back to visual only.
        result.status = StageStatus.DEGRADED
        result.warn("insufficient_clean_speech")
        confidence = 0.0
    elif aggregate < TARGET_AGGREGATE_SAMPLES:
        result.warn("short_enrolment")

    enrolment = Enrolment(
        speaker=speaker,
        audio_embedding=cue,
        audio_cue_confidence=confidence,
        regions_used=[c.interval for c in chosen],
        visual_regions=[c.interval for c in chosen],
        aggregate_samples=aggregate,
    )

    result.seconds = time.perf_counter() - t0
    result.detail = (
        f"{len(chosen)} regions, {aggregate / ANALYSIS_SAMPLE_RATE:.1f}s, "
        f"confidence {confidence:.2f}"
    )
    return enrolment, result
