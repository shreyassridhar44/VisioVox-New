"""Forced-alignment tests (S7, docs/05 §10).

Most of these run on CPU without the acoustic model, because the parts most
likely to be wrong are the text handling and the degradation path, not the CTC
maths — which is torchaudio's and already tested upstream.

The degradation path gets particular attention. This stage improves accuracy;
it is not a correctness requirement, so any failure must return the original
transcript rather than lose it.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest
import soundfile as sf

from pipeline.s7_align import _normalise, align_transcript, timing_error_ms
from pipeline.s7_transcribe import Segment, Transcript, Word
from pipeline.types import StageStatus


def _transcript(words: list[tuple[str, int, int]]) -> Transcript:
    ws = tuple(Word(t, a, b, 0.9) for t, a, b in words)
    return Transcript(
        segments=[Segment(" ".join(w[0] for w in words), ws[0].start_ms, ws[-1].end_ms, ws, 0.01)],
        language="en",
        language_probability=0.99,
    )


# --------------------------------------------------------------------------
# text normalisation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Hello", "hello"),
        ("Hello,", "hello"),
        ("don't", "don't"),
        ("  spaced  ", "spaced"),
        ("...", ""),
        ("42", ""),
        ("'quoted'", "quoted"),
    ],
)
def test_normalisation_matches_the_ctc_vocabulary(raw: str, expected: str) -> None:
    """Punctuation and case are not in the vocabulary. Leaving them in makes
    the token lookup fail and drops the word from alignment entirely, which is
    worse than aligning a stripped form."""
    assert _normalise(raw) == expected


# --------------------------------------------------------------------------
# degradation
# --------------------------------------------------------------------------


class _BrokenAligner:
    """Stands in for a model that fails at inference."""

    device = "cpu"

    def __getattr__(self, name: str) -> object:
        raise RuntimeError("acoustic model unavailable")


def test_failure_returns_the_original_transcript() -> None:
    """Approximate timings beat no transcript. This stage is an improvement,
    not a requirement."""
    original = _transcript([("hello", 100, 500), ("there", 520, 900)])
    out, result = align_transcript(
        original,
        np.zeros(16_000, dtype=np.float32),
        _BrokenAligner(),  # type: ignore[arg-type]
    )
    assert result.status is StageStatus.DEGRADED
    assert "alignment_failed" in result.warnings
    assert out.segments[0].words == original.segments[0].words


def test_empty_transcript_degrades_cleanly() -> None:
    empty = Transcript(segments=[], language="en", language_probability=0.9)
    out, result = align_transcript(
        empty,
        np.zeros(16_000, dtype=np.float32),
        _BrokenAligner(),  # type: ignore[arg-type]
    )
    assert result.status is StageStatus.DEGRADED
    assert "nothing_to_align" in result.warnings
    assert out.segments == []


def test_unalignable_words_degrade_cleanly() -> None:
    """A transcript of only punctuation has nothing the vocabulary can match."""
    odd = _transcript([("...", 0, 100), ("!!", 100, 200)])
    _, result = align_transcript(
        odd,
        np.zeros(16_000, dtype=np.float32),
        _BrokenAligner(),  # type: ignore[arg-type]
    )
    assert result.status is StageStatus.DEGRADED
    assert "nothing_to_align" in result.warnings


# --------------------------------------------------------------------------
# timing error measurement
# --------------------------------------------------------------------------


def test_identical_transcripts_have_zero_error() -> None:
    t = _transcript([("hello", 100, 500), ("there", 520, 900)])
    assert timing_error_ms(t, t) == [0.0, 0.0]


def test_timing_error_reports_per_word_shift() -> None:
    a = _transcript([("hello", 100, 500), ("there", 520, 900)])
    b = _transcript([("hello", 180, 560), ("there", 600, 980)])
    assert timing_error_ms(a, b) == [80.0, 80.0]


def test_timing_error_skips_mismatched_words() -> None:
    """A different word at the same position means the transcripts are not
    comparable there, and pretending otherwise inflates the score."""
    a = _transcript([("hello", 100, 500), ("there", 520, 900)])
    b = _transcript([("hello", 100, 500), ("world", 520, 900)])
    assert timing_error_ms(a, b) == [0.0]


def test_timing_error_ignores_punctuation_differences() -> None:
    a = _transcript([("Hello,", 100, 500)])
    b = _transcript([("hello", 140, 540)])
    assert timing_error_ms(a, b) == [40.0]


# --------------------------------------------------------------------------
# the real model
# --------------------------------------------------------------------------

needs_gpu = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or not os.environ.get("HF_TOKEN"),
    reason="needs ffmpeg and model downloads",
)


@pytest.mark.gpu
@needs_gpu
def test_real_alignment_moves_timings_onto_the_audio() -> None:
    """The point of the stage: Whisper infers times from cross-attention and
    drifts; forced alignment finds the acoustic boundary."""
    from pipeline.s7_align import load_aligner

    audio, _ = sf.read("tests/fixtures/mix_2spk.wav", dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # deliberately wrong seed timings, evenly spaced
    seeded = _transcript([("i", 0, 250), ("noticed", 300, 550), ("how", 600, 850)])
    aligned, result = align_transcript(
        seeded, np.asarray(audio, dtype=np.float32), load_aligner("cuda")
    )

    assert result.status is StageStatus.OK, result.detail
    words = aligned.segments[0].words
    assert len(words) == 3
    starts = [w.start_ms for w in words]
    assert starts == sorted(starts), "aligned words went backwards in time"
    assert all(w.end_ms > w.start_ms for w in words)
    assert starts != [0, 300, 600], "alignment did not move the seeded timings"
