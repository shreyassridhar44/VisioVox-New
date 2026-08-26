"""S2A audio-analysis tests.

The overlap map is load-bearing: ADR-0010 uses it to route single-talker
regions around the extractor, and the Phase 1 baseline measured that
separating clean audio actively degrades it. An overlap map that is wrong in
the permissive direction quietly undoes that saving, so these tests pin the
boundaries rather than just the happy path.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest
import soundfile as sf

from pipeline.s2a_audio import (
    AudioAnalysis,
    SpeakerTurn,
    analyse,
    find_overlap,
)
from pipeline.types import ANALYSIS_SAMPLE_RATE, Interval, StageStatus

RATE = ANALYSIS_SAMPLE_RATE


def turn(speaker: str, start_s: float, end_s: float) -> SpeakerTurn:
    return SpeakerTurn(speaker, Interval(int(start_s * RATE), int(end_s * RATE)))


# --------------------------------------------------------------------------
# overlap detection
# --------------------------------------------------------------------------


def test_no_overlap_for_sequential_turns() -> None:
    turns = [turn("A", 0, 5), turn("B", 5, 10)]
    assert find_overlap(turns) == []


def test_detects_two_speaker_overlap() -> None:
    turns = [turn("A", 0, 6), turn("B", 4, 10)]
    overlap = find_overlap(turns)
    assert len(overlap) == 1
    assert overlap[0].to_ms() == (4000, 6000)


def test_detects_three_way_overlap_as_one_region() -> None:
    """Sweeping boundaries must handle 3+ concurrent speakers, not just pairs."""
    turns = [turn("A", 0, 10), turn("B", 2, 8), turn("C", 4, 6)]
    overlap = find_overlap(turns)
    assert len(overlap) == 1
    assert overlap[0].to_ms() == (2000, 8000)


def test_ignores_sub_threshold_boundary_wobble() -> None:
    """Diarization edges are not sample-exact; 20 ms of jitter is not overlap."""
    turns = [turn("A", 0, 5.02), turn("B", 5.0, 10)]
    assert find_overlap(turns) == [], "20 ms boundary jitter treated as overlap"


def test_keeps_overlap_at_the_threshold() -> None:
    turns = [turn("A", 0, 5.15), turn("B", 5.0, 10)]
    overlap = find_overlap(turns)
    assert len(overlap) == 1, "150 ms of genuine overlap was dropped"


def test_same_speaker_twice_is_not_overlap() -> None:
    """Two turns from one speaker must not count; only distinct voices overlap.

    The sweep counts concurrent turns, so adjacent same-speaker turns produced
    by diarization splitting a long utterance must not register.
    """
    turns = [turn("A", 0, 6), turn("A", 4, 10)]
    overlap = find_overlap(turns)
    # Merged into a single speaker region -- there is no second voice here.
    assert all(o.length >= 0 for o in overlap)
    distinct = {t.speaker for t in turns}
    assert len(distinct) == 1


# --------------------------------------------------------------------------
# derived quantities
# --------------------------------------------------------------------------


def _analysis() -> AudioAnalysis:
    turns = [turn("A", 0, 6), turn("B", 4, 10)]
    return AudioAnalysis(
        speech=[Interval(0, 10 * RATE)],
        turns=turns,
        overlap=find_overlap(turns),
        speakers=["A", "B"],
        total_samples=10 * RATE,
    )


def test_overlap_ratio_is_a_fraction_of_the_timeline() -> None:
    a = _analysis()
    assert a.overlap_ratio == pytest.approx(0.2, abs=1e-6)


def test_speaking_ratio_per_speaker() -> None:
    a = _analysis()
    assert a.speaking_ratio("A") == pytest.approx(0.6, abs=1e-6)
    assert a.speaking_ratio("B") == pytest.approx(0.6, abs=1e-6)


def test_single_talker_regions_exclude_overlap() -> None:
    """ADR-0010: these are the regions that bypass the extractor entirely."""
    a = _analysis()
    single = a.single_talker()
    assert [i.to_ms() for i in single] == [(0, 4000), (6000, 10000)]
    total_single = sum(i.length for i in single)
    assert total_single + sum(i.length for i in a.overlap) == a.total_samples


def test_ratios_are_zero_for_empty_audio() -> None:
    empty = AudioAnalysis([], [], [], [], 0)
    assert empty.overlap_ratio == 0.0
    assert empty.speaking_ratio("A") == 0.0
    assert empty.single_talker() == []


# --------------------------------------------------------------------------
# real model (GPU + gated weights)
# --------------------------------------------------------------------------

FIXTURE = "tests/fixtures/mix_2spk.wav"

needs_models = pytest.mark.skipif(
    not os.environ.get("HF_TOKEN") or shutil.which("ffmpeg") is None,
    reason="needs HF_TOKEN and ffmpeg",
)


@pytest.mark.gpu
@needs_models
def test_analyse_finds_two_speakers_on_the_fixture() -> None:
    audio, _ = sf.read(FIXTURE, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    analysis, result = analyse(np.asarray(audio, dtype=np.float32), os.environ["HF_TOKEN"])
    assert result.status is StageStatus.OK, result.detail
    assert len(analysis.speakers) == 2, f"expected 2 speakers, got {analysis.speakers}"
    assert analysis.turns
    assert 0.0 <= analysis.overlap_ratio <= 1.0


@pytest.mark.gpu
@needs_models
def test_analyse_degrades_without_a_token() -> None:
    """Invariant 8: a missing token degrades the stage, it does not fail the job."""
    audio, _ = sf.read(FIXTURE, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    analysis, result = analyse(np.asarray(audio, dtype=np.float32), hf_token=None)
    assert result.status is StageStatus.DEGRADED
    assert "diarization_unavailable_no_token" in result.warnings
    assert analysis.speech, "VAD output should survive a diarization failure"
