"""S5 extraction: routing, stitching and the audio-only fallback.

No GPU and no checkpoint. The interesting behaviour of this stage is not the
separator -- that is measured by scripts/eval_overlap.py against real audio --
but what happens around it: which windows reach the model at all, whether
untouched regions really are untouched, and whether the result is still the
right length when there is no face to look at. A stub extractor makes all of
that testable in milliseconds, and a stub is honest here because the stage
deliberately knows nothing about the model beyond `run_window`.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from pipeline import s5_extract
from pipeline.passthrough import FRAME_SAMPLES, Route
from pipeline.s5_extract import ExtractionResult, SeaveExtractor, extract_speaker

RATE = 16_000
EMB = np.zeros(192, dtype=np.float32)


class _StubExtractor:
    """Returns a recognisable marker instead of separating anything."""

    def __init__(self, has_video: bool = False) -> None:
        self._has_video = has_video
        self.windows_run = 0
        self.saw_video: list[bool] = []

    @property
    def has_video(self) -> bool:
        return self._has_video

    def run_window(
        self,
        mixture: np.ndarray,
        enrolment: np.ndarray,
        mouth: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        self.windows_run += 1
        self.saw_video.append(mouth is not None)
        # A tone unrelated to the input, so scale matching cannot turn it back
        # into the mixture and a passthrough region stays distinguishable.
        t = np.arange(len(mixture)) / RATE
        return (0.4 * np.sin(2 * np.pi * 997.0 * t)).astype(np.float32), 0.9


def _as_extractor(stub: _StubExtractor) -> SeaveExtractor:
    return cast(SeaveExtractor, stub)


def _mixture(frames: int, freq: float = 220.0) -> np.ndarray:
    t = np.arange(frames * FRAME_SAMPLES) / RATE
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------


def test_single_talker_audio_is_returned_untouched() -> None:
    """docs/05 SS8.1: passthrough is a fidelity decision, not only a cost one."""
    frames = 250
    mixture = _mixture(frames)
    stub = _StubExtractor()
    out = extract_speaker(
        mixture,
        EMB,
        _as_extractor(stub),
        target_active=np.ones(frames),
        others_active=np.zeros(frames),
    )
    assert stub.windows_run == 0, "a clean single-talker timeline must not reach the GPU"
    assert np.allclose(out.audio, mixture, atol=1e-5)
    assert out.processed_fraction == 0.0


def test_frames_where_the_target_is_silent_come_back_silent() -> None:
    frames = 250
    mixture = _mixture(frames)
    out = extract_speaker(
        mixture,
        EMB,
        _as_extractor(_StubExtractor()),
        target_active=np.zeros(frames),
        others_active=np.ones(frames),
    )
    assert np.max(np.abs(out.audio)) < 1e-6


def test_overlapped_frames_are_extracted_and_differ_from_the_mixture() -> None:
    frames = 250
    mixture = _mixture(frames)
    stub = _StubExtractor()
    out = extract_speaker(
        mixture,
        EMB,
        _as_extractor(stub),
        target_active=np.ones(frames),
        others_active=np.ones(frames),
    )
    assert stub.windows_run > 0
    assert out.processed_fraction == pytest.approx(1.0)
    # The interior, away from the crossfade ramps at either end.
    interior = slice(20 * FRAME_SAMPLES, 200 * FRAME_SAMPLES)
    assert not np.allclose(out.audio[interior], mixture[interior], atol=1e-3)


def test_only_windows_overlapping_an_extract_region_reach_the_model() -> None:
    """The 80-95% saving in docs/05 SS8.1 comes from skipping windows entirely."""
    frames = 500  # 20 s
    mixture = _mixture(frames)
    others = np.zeros(frames)
    others[:50] = 1.0  # a single 2 s overlap at the very start
    stub = _StubExtractor()
    extract_speaker(
        mixture,
        EMB,
        _as_extractor(stub),
        target_active=np.ones(frames),
        others_active=others,
    )
    # 20 s at a 2 s hop is ten windows; only the first couple touch the overlap.
    assert 0 < stub.windows_run <= 3


# --------------------------------------------------------------------------
# shape, length and the audio-only fallback
# --------------------------------------------------------------------------


def test_output_is_always_the_full_input_length() -> None:
    """S5 emits spk_k.faithful.wav at full length; the player assumes it."""
    for frames in (60, 137, 250, 313):
        mixture = _mixture(frames)
        out = extract_speaker(
            mixture,
            EMB,
            _as_extractor(_StubExtractor()),
            target_active=np.ones(frames),
            others_active=np.ones(frames),
        )
        assert len(out.audio) == len(mixture), f"length changed at {frames} frames"


def test_missing_video_falls_back_to_audio_only() -> None:
    """S2B finding no face must degrade, not fail (docs/05 SS8)."""
    frames = 200
    mixture = _mixture(frames)
    stub = _StubExtractor(has_video=True)
    out = extract_speaker(
        mixture,
        EMB,
        _as_extractor(stub),
        target_active=np.ones(frames),
        others_active=np.ones(frames),
        mouth=None,
    )
    assert len(out.audio) == len(mixture)
    assert not any(stub.saw_video)


def test_video_is_passed_through_when_both_rois_and_a_frontend_exist() -> None:
    frames = 200
    mixture = _mixture(frames)
    stub = _StubExtractor(has_video=True)
    extract_speaker(
        mixture,
        EMB,
        _as_extractor(stub),
        target_active=np.ones(frames),
        others_active=np.ones(frames),
        mouth=np.zeros((frames, 96, 96), dtype=np.uint8),
    )
    assert all(stub.saw_video)


def test_an_audio_only_checkpoint_ignores_offered_rois() -> None:
    """C1/C3/C4 checkpoints have no frontend; handing them ROIs must not crash."""
    frames = 200
    stub = _StubExtractor(has_video=False)
    extract_speaker(
        _mixture(frames),
        EMB,
        _as_extractor(stub),
        target_active=np.ones(frames),
        others_active=np.ones(frames),
        mouth=np.zeros((frames, 96, 96), dtype=np.uint8),
    )
    assert not any(stub.saw_video)


def test_short_roi_sequences_are_padded_rather_than_truncating_the_window() -> None:
    """Video often runs out before audio does; the tail still has to be extracted."""
    frames = 200
    stub = _StubExtractor(has_video=True)
    out = extract_speaker(
        _mixture(frames),
        EMB,
        _as_extractor(stub),
        target_active=np.ones(frames),
        others_active=np.ones(frames),
        mouth=np.zeros((frames - 40, 96, 96), dtype=np.uint8),
    )
    assert len(out.audio) == frames * FRAME_SAMPLES


# --------------------------------------------------------------------------
# scale matching
# --------------------------------------------------------------------------


def test_scale_matching_recovers_an_arbitrarily_scaled_estimate() -> None:
    """SI-SDR training leaves the output gain free; overlap-add does not."""
    t = np.arange(4000) / RATE
    target = (0.3 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)
    interferer = (0.3 * np.sin(2 * np.pi * 611.0 * t)).astype(np.float32)
    mixture = target + interferer
    for gain in (0.01, 1.0, 37.0):
        fitted = s5_extract._match_scale(target * gain, mixture)
        assert np.allclose(fitted, target, atol=1e-3), f"gain {gain} not recovered"


def test_scale_matching_survives_an_all_zero_estimate() -> None:
    zeros = np.zeros(1000, dtype=np.float32)
    assert np.array_equal(s5_extract._match_scale(zeros, _mixture(2)[:1000]), zeros)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def test_processed_fraction_reports_the_share_actually_extracted() -> None:
    route = np.array([Route.SILENCE.value] * 5 + [Route.EXTRACT.value] * 15, dtype=np.int8)
    result = ExtractionResult(audio=np.zeros(10), route=route)
    assert result.processed_fraction == pytest.approx(0.75)


def test_processed_fraction_of_an_empty_route_is_zero() -> None:
    result = ExtractionResult(audio=np.zeros(0), route=np.zeros(0, dtype=np.int8))
    assert result.processed_fraction == 0.0
