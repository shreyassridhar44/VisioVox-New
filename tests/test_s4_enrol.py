"""S4 self-enrolment tests (SEAVE-SE, docs/04 §2).

The claim under test is that purity-weighted aggregation over several regions
beats naive longest-segment selection. That is Contribution 1's specific,
falsifiable part, so it gets a test constructed to be losable: the longest
region is genuinely clean and genuinely long, and it still loses because it
carries one prosodic context.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.s2a_audio import AudioAnalysis, SpeakerTurn
from pipeline.s4_enrol import (
    MIN_REGION_SAMPLES,
    Candidate,
    PurityWeights,
    aggregate_embedding,
    enrol,
    find_candidates,
    score_candidates,
    select_regions,
    visual_confidence,
)
from pipeline.types import ANALYSIS_SAMPLE_RATE, Interval, StageStatus

RATE = ANALYSIS_SAMPLE_RATE
DIM = 64


def _turn(speaker: str, a: float, b: float) -> SpeakerTurn:
    return SpeakerTurn(speaker, Interval(int(a * RATE), int(b * RATE)))


def _analysis(turns: list[SpeakerTurn], overlap: list[Interval], seconds: float) -> AudioAnalysis:
    return AudioAnalysis(
        speech=[Interval(0, int(seconds * RATE))],
        turns=turns,
        overlap=overlap,
        speakers=sorted({t.speaker for t in turns}),
        total_samples=int(seconds * RATE),
    )


def _speech(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    return (0.3 * env * np.sin(2 * np.pi * 200 * t) + 0.01 * rng.standard_normal(len(t))).astype(
        np.float32
    )


def _unit(v: np.ndarray) -> np.ndarray:
    return np.asarray(v / np.linalg.norm(v), dtype=np.float64)


# --------------------------------------------------------------------------
# candidate discovery
# --------------------------------------------------------------------------


def test_candidates_exclude_overlapped_speech() -> None:
    """Enrolment must come from single-talker audio; overlap contaminates it."""
    analysis = _analysis(
        [_turn("A", 0, 10)], overlap=[Interval(int(4 * RATE), int(6 * RATE))], seconds=10
    )
    cands = find_candidates(analysis, "A")
    assert len(cands) == 2
    assert [(c.interval.to_ms()) for c in cands] == [(0, 4000), (6000, 10000)]


def test_short_fragments_are_discarded() -> None:
    """Under 1.5 s carries too little of the speaker to be worth weighting in."""
    analysis = _analysis(
        [_turn("A", 0, 5)],
        overlap=[Interval(int(1.0 * RATE), int(4.8 * RATE))],
        seconds=5,
    )
    cands = find_candidates(analysis, "A")
    assert all(c.interval.length >= MIN_REGION_SAMPLES for c in cands)


def test_other_speakers_are_not_candidates() -> None:
    analysis = _analysis([_turn("A", 0, 5), _turn("B", 5, 10)], overlap=[], seconds=10)
    assert all(c.interval.start < 5 * RATE for c in find_candidates(analysis, "A"))


def test_no_turns_yields_no_candidates() -> None:
    analysis = _analysis([_turn("B", 0, 5)], overlap=[], seconds=5)
    assert find_candidates(analysis, "A") == []


# --------------------------------------------------------------------------
# the contribution: purity weighting versus longest-segment
# --------------------------------------------------------------------------


def test_purity_weighting_beats_longest_segment() -> None:
    """The claim, constructed so it could fail.

    A speaker's true centroid is the mean of their prosodic contexts. The
    longest region is clean and long but sits in ONE context, so it is offset
    from the centroid. Four shorter regions span the contexts. If weighted
    aggregation is not actually better, this test says so.
    """
    rng = np.random.default_rng(0)
    centroid = _unit(rng.standard_normal(DIM))

    # four context directions that average back to the centroid
    contexts = []
    for _ in range(4):
        offset = rng.standard_normal(DIM)
        offset -= offset.dot(centroid) * centroid  # orthogonal to the centroid
        contexts.append(_unit(centroid + 0.55 * _unit(offset)))

    # region 0 is the longest, and lives entirely in context 0
    lengths = [6.0, 2.0, 2.0, 2.0]
    embeddings = [contexts[0], contexts[1], contexts[2], contexts[3]]

    cands: list[Candidate] = []
    start = 0.0
    for length, emb in zip(lengths, embeddings, strict=True):
        c = Candidate(interval=Interval(int(start * RATE), int((start + length) * RATE)))
        c.overlap_free = 1.0
        c.snr_norm = 0.8
        c.speech_density = 0.9
        c.reverb = 0.1
        c.embedding = emb
        cands.append(c)
        start += length + 0.5

    all_vectors = np.stack([c.embedding for c in cands])  # type: ignore[misc]
    consensus = _unit(all_vectors.mean(axis=0))
    for c in cands:
        c.embedding_agreement = float(np.dot(c.embedding, consensus))  # type: ignore[arg-type]

    weighted, _ = aggregate_embedding(cands)
    assert weighted is not None

    longest = max(cands, key=lambda c: c.interval.length).embedding
    assert longest is not None

    sim_weighted = float(np.dot(weighted, centroid))
    sim_longest = float(np.dot(longest, centroid))

    assert sim_weighted > sim_longest, (
        f"weighted aggregate {sim_weighted:.4f} did not beat longest-segment {sim_longest:.4f}"
    )


def test_selection_spans_multiple_regions() -> None:
    """One long region is one prosodic context; diversity is the point."""
    cands = []
    for i in range(6):
        c = Candidate(interval=Interval(i * 2 * RATE, (i * 2 + 2) * RATE))
        c.overlap_free, c.snr_norm, c.speech_density = 1.0, 0.8, 0.9
        cands.append(c)
    chosen = select_regions(cands)
    assert len(chosen) >= 4, "selection collapsed onto too few regions"


def test_selection_stops_once_the_duration_target_is_met() -> None:
    cands = []
    for i in range(20):
        c = Candidate(interval=Interval(i * 3 * RATE, (i * 3 + 3) * RATE))
        c.overlap_free, c.snr_norm, c.speech_density = 1.0, 0.8, 0.9
        cands.append(c)
    chosen = select_regions(cands)
    assert sum(c.interval.length for c in chosen) >= 8 * RATE
    assert len(chosen) <= 8


# --------------------------------------------------------------------------
# purity scoring
# --------------------------------------------------------------------------


def test_reverberant_region_scores_lower() -> None:
    w = PurityWeights()
    clean = Candidate(Interval(0, 2 * RATE), 1.0, 0.8, 0.9, 0.9, reverb=0.0)
    reverberant = Candidate(Interval(0, 2 * RATE), 1.0, 0.8, 0.9, 0.9, reverb=0.9)
    assert clean.purity(w) > reverberant.purity(w)


def test_disagreeing_region_scores_lower() -> None:
    """A region inconsistent with the speaker's own consensus is usually
    diarization having assigned it to the wrong person."""
    w = PurityWeights()
    consistent = Candidate(Interval(0, 2 * RATE), 1.0, 0.8, 0.9, embedding_agreement=0.95)
    odd = Candidate(Interval(0, 2 * RATE), 1.0, 0.8, 0.9, embedding_agreement=0.10)
    assert consistent.purity(w) > odd.purity(w)


def test_scoring_ranks_by_purity() -> None:
    audio = _speech(12.0)
    cands = [
        Candidate(Interval(0, 3 * RATE)),
        Candidate(Interval(4 * RATE, 7 * RATE)),
    ]
    scored = score_candidates(cands, audio)
    w = PurityWeights()
    assert scored[0].purity(w) >= scored[1].purity(w)


# --------------------------------------------------------------------------
# confidence — the routing signal for Contribution 2
# --------------------------------------------------------------------------


def test_confidence_rises_with_more_clean_speech() -> None:
    rng = np.random.default_rng(1)
    base = _unit(rng.standard_normal(DIM))

    def build(n: int, seconds: float) -> float:
        cands = []
        for i in range(n):
            c = Candidate(Interval(int(i * seconds * RATE), int((i + 1) * seconds * RATE)))
            c.overlap_free, c.snr_norm, c.speech_density = 1.0, 0.9, 0.95
            c.embedding_agreement = 0.98
            c.embedding = base
            cands.append(c)
        return aggregate_embedding(cands)[1]

    assert build(1, 1.5) < build(4, 2.5)


def test_inconsistent_regions_lower_confidence() -> None:
    """A cue that is wrong but confident is worse than an absent one."""
    rng = np.random.default_rng(2)

    def build(consistent: bool) -> float:
        cands = []
        base = _unit(rng.standard_normal(DIM))
        for i in range(4):
            c = Candidate(Interval(i * 2 * RATE, (i + 1) * 2 * RATE))
            c.overlap_free, c.snr_norm, c.speech_density = 1.0, 0.9, 0.95
            c.embedding_agreement = 0.95
            c.embedding = base if consistent else _unit(rng.standard_normal(DIM))
            cands.append(c)
        return aggregate_embedding(cands)[1]

    assert build(consistent=False) < build(consistent=True)


def test_no_embeddings_gives_no_cue() -> None:
    cands = [Candidate(Interval(0, 2 * RATE))]
    cue, conf = aggregate_embedding(cands)
    assert cue is None and conf == 0.0


# --------------------------------------------------------------------------
# visual confidence
# --------------------------------------------------------------------------


def test_visual_confidence_is_sunk_by_one_bad_factor() -> None:
    """A large, sharp, continuously-tracked face turned fully away is useless.

    An arithmetic mean rates it 0.71 and a plain geometric mean 0.45 — both
    high enough to pass a gate it should fail. Blending in the worst factor
    drops it below 0.2 while leaving a good cue above 0.9."""
    good = visual_confidence(face_size_px=100, frontality=0.9, sharpness=0.9, continuity=0.9)
    turned_away = visual_confidence(
        face_size_px=100, frontality=0.05, sharpness=0.9, continuity=0.9
    )
    assert good > 0.85
    assert turned_away < 0.20
    arithmetic = (1.0 + 0.05 + 0.9 + 0.9) / 4
    geometric = (1.0 * 0.05 * 0.9 * 0.9) ** 0.25
    assert turned_away < geometric < arithmetic, (
        "should be stricter than both an arithmetic and a plain geometric mean"
    )


def test_visual_confidence_is_zero_when_a_factor_is_zero() -> None:
    assert visual_confidence(0, 0.9, 0.9, 0.9) == 0.0


def test_small_face_lowers_visual_confidence() -> None:
    assert visual_confidence(20, 0.9, 0.9, 0.9) < visual_confidence(100, 0.9, 0.9, 0.9)


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------


def test_enrol_produces_a_cue_from_clean_speech() -> None:
    rng = np.random.default_rng(3)
    analysis = _analysis([_turn("A", 0, 12)], overlap=[Interval(5 * RATE, 6 * RATE)], seconds=12)
    audio = _speech(12.0)

    def embed(_seg: np.ndarray) -> np.ndarray:
        return _unit(rng.standard_normal(DIM) * 0.1 + np.ones(DIM))

    enrolment, result = enrol(analysis, audio, "A", embed)
    assert result.status is StageStatus.OK
    assert enrolment.has_audio_cue
    assert enrolment.aggregate_samples >= MIN_REGION_SAMPLES
    assert 0.0 < enrolment.audio_cue_confidence <= 1.0


def test_enrol_degrades_when_speech_is_too_short() -> None:
    """docs/04 §2: under 1.5 s the audio cue is unreliable and conditioning
    should fall back to visual only."""
    analysis = _analysis([_turn("A", 0, 1.0)], overlap=[], seconds=2)
    enrolment, result = enrol(analysis, _speech(2.0), "A")
    assert result.status is StageStatus.DEGRADED
    assert enrolment.audio_cue_confidence == 0.0
    assert not enrolment.has_audio_cue


def test_enrol_degrades_with_no_regions_at_all() -> None:
    analysis = _analysis([_turn("B", 0, 5)], overlap=[], seconds=5)
    enrolment, result = enrol(analysis, _speech(5.0), "A")
    assert result.status is StageStatus.DEGRADED
    assert "no_clean_regions" in result.warnings
    assert enrolment.audio_embedding is None


def test_enrol_warns_on_short_but_usable_enrolment() -> None:
    analysis = _analysis([_turn("A", 0, 4)], overlap=[], seconds=4)
    _, result = enrol(analysis, _speech(4.0), "A")
    assert "short_enrolment" in result.warnings


@pytest.mark.parametrize("seconds", [3.0, 6.0, 12.0])
def test_enrol_is_stable_across_durations(seconds: float) -> None:
    analysis = _analysis([_turn("A", 0, seconds)], overlap=[], seconds=seconds)
    enrolment, result = enrol(analysis, _speech(seconds), "A")
    assert result.status in (StageStatus.OK, StageStatus.DEGRADED)
    assert 0.0 <= enrolment.audio_cue_confidence <= 1.0
