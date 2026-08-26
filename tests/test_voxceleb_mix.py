"""VoxCeleb2-Mix simulation tests.

Built on synthetic packed clips, so the mixing maths is checked without needing
the corpus. The properties tested are the ones a bug would hide inside a
training run for weeks: audio and video drifting out of alignment, the
target-to-interferer ratio not being what was asked for, and clip-avoidance
silently rescaling the target the model is asked to reproduce.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from training.voxceleb_mix import (
    FPS,
    RATE,
    SAMPLES_PER_FRAME,
    MixConfig,
    VoxCelebMixDataset,
)

MOUTH = 96


def _write_clip(
    root: Path, speaker: str, name: str, seconds: float, freq: float, seed: int
) -> None:
    frames = int(seconds * FPS)
    samples = frames * SAMPLES_PER_FRAME
    t = np.arange(samples) / RATE
    rng = np.random.default_rng(seed)
    audio = (0.3 * np.sin(2 * np.pi * freq * t) + 0.01 * rng.standard_normal(samples)).astype(
        np.float32
    )
    mouth = rng.integers(0, 255, (frames, MOUTH, MOUTH), dtype=np.uint8)
    d = root / speaker
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"{name}.npz", mouth=mouth, audio=audio)


@pytest.fixture
def packed(tmp_path: Path) -> Path:
    root = tmp_path / "packed"
    for i, freq in enumerate((180.0, 240.0, 300.0, 360.0)):
        for c in range(2):
            _write_clip(root, f"id{i:04d}", f"sess_{c}", seconds=6.0, freq=freq, seed=i * 10 + c)
    return root


# --------------------------------------------------------------------------
# structure and alignment
# --------------------------------------------------------------------------


def test_sample_shapes_are_consistent(packed: Path) -> None:
    ds = VoxCelebMixDataset(packed, MixConfig(chunk_seconds=4.0, seed=0))
    s = ds.sample(0)
    frames = int(4.0 * FPS)
    assert s.mouth.shape == (frames, MOUTH, MOUTH)
    assert len(s.mixture) == frames * SAMPLES_PER_FRAME
    assert len(s.target) == len(s.mixture)
    assert len(s.interferer) == len(s.mixture)
    assert len(s.active) == frames


def test_audio_and_video_stay_in_step(packed: Path) -> None:
    """640 samples per frame exactly; a rounding slip here becomes drift."""
    assert SAMPLES_PER_FRAME * FPS == RATE
    ds = VoxCelebMixDataset(packed, MixConfig(chunk_seconds=3.0, seed=1))
    s = ds.sample(0)
    assert len(s.target) == len(s.mouth) * SAMPLES_PER_FRAME


def test_mixture_is_exactly_target_plus_interferer(packed: Path) -> None:
    ds = VoxCelebMixDataset(packed, MixConfig(seed=2))
    s = ds.sample(0)
    assert np.allclose(s.mixture, s.target + s.interferer, atol=1e-6)


def test_short_clips_are_padded_not_dropped(tmp_path: Path) -> None:
    root = tmp_path / "packed"
    for i in range(2):
        _write_clip(root, f"id{i}", "short", seconds=1.5, freq=200.0 + i * 60, seed=i)
    ds = VoxCelebMixDataset(root, MixConfig(chunk_seconds=4.0, seed=3))
    s = ds.sample(0)
    assert len(s.mouth) == int(4.0 * FPS)


# --------------------------------------------------------------------------
# simulation realism
# --------------------------------------------------------------------------


def test_target_to_interferer_ratio_is_respected(packed: Path) -> None:
    """A fixed TIR must actually appear in the signal, not just be sampled."""
    for tir in (-5.0, 0.0, 5.0):
        ds = VoxCelebMixDataset(
            packed,
            MixConfig(seed=4, tir_db=(tir, tir), overlap_ratio=(1.0, 1.0), chunk_seconds=4.0),
        )
        s = ds.sample(0)
        t_rms = float(np.sqrt(np.mean(s.target**2)))
        i_rms = float(np.sqrt(np.mean(s.interferer**2)))
        measured = 20 * np.log10(t_rms / i_rms)
        assert abs(measured - tir) < 1.0, f"asked {tir} dB, measured {measured:.2f} dB"


def test_partial_overlap_leaves_the_interferer_silent_somewhere(packed: Path) -> None:
    """Real conversation is sparsely overlapped; full overlap is the Libri2Mix trap."""
    ds = VoxCelebMixDataset(packed, MixConfig(seed=5, overlap_ratio=(0.3, 0.3), chunk_seconds=4.0))
    s = ds.sample(0)
    silent = np.abs(s.interferer) < 1e-8
    assert silent.mean() > 0.5, "interferer occupied the whole chunk"


def test_full_overlap_covers_the_chunk(packed: Path) -> None:
    ds = VoxCelebMixDataset(packed, MixConfig(seed=6, overlap_ratio=(1.0, 1.0), chunk_seconds=4.0))
    s = ds.sample(0)
    assert (np.abs(s.interferer) < 1e-8).mean() < 0.2


def test_mixture_does_not_clip(packed: Path) -> None:
    ds = VoxCelebMixDataset(packed, MixConfig(seed=7, tir_db=(-15.0, -15.0)))
    for i in range(5):
        s = ds.sample(i)
        assert np.abs(s.mixture).max() <= 0.99 + 1e-6


def test_clip_avoidance_rescales_all_three_together(packed: Path) -> None:
    """Scaling only the mixture would change the target the model must match."""
    ds = VoxCelebMixDataset(packed, MixConfig(seed=8, tir_db=(-20.0, -20.0)))
    s = ds.sample(0)
    assert np.allclose(s.mixture, s.target + s.interferer, atol=1e-6)


# --------------------------------------------------------------------------
# speaker handling
# --------------------------------------------------------------------------


def test_target_and_interferer_are_different_speakers(packed: Path) -> None:
    ds = VoxCelebMixDataset(packed, MixConfig(seed=9))
    for i in range(20):
        s = ds.sample(i)
        assert s.target_speaker not in s.interferer_speakers


def test_speaker_filter_is_honoured(packed: Path) -> None:
    """Disjoint splits depend on this: an unlisted speaker must never appear."""
    allowed = ["id0000", "id0001"]
    ds = VoxCelebMixDataset(packed, MixConfig(seed=10), speakers=allowed)
    assert ds.speakers == allowed
    for i in range(20):
        s = ds.sample(i)
        assert s.target_speaker in allowed
        assert all(spk in allowed for spk in s.interferer_speakers)


def test_too_few_speakers_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "packed"
    _write_clip(root, "only", "a", seconds=4.0, freq=200.0, seed=0)
    with pytest.raises(ValueError, match="at least 2 speakers"):
        VoxCelebMixDataset(root, MixConfig())


def test_two_interferers_are_all_distinct(packed: Path) -> None:
    ds = VoxCelebMixDataset(packed, MixConfig(seed=11, n_interferers=2))
    s = ds.sample(0)
    assert len(set(s.interferer_speakers)) == 2
    assert s.target_speaker not in s.interferer_speakers


def test_sampling_is_reproducible(packed: Path) -> None:
    """Runs must be repeatable — every run logs a seed for exactly this reason."""
    a = VoxCelebMixDataset(packed, MixConfig(seed=42)).sample(3)
    b = VoxCelebMixDataset(packed, MixConfig(seed=42)).sample(3)
    assert np.array_equal(a.mixture, b.mixture)
    assert a.target_speaker == b.target_speaker


def test_activity_mask_marks_speech(packed: Path) -> None:
    ds = VoxCelebMixDataset(packed, MixConfig(seed=12))
    s = ds.sample(0)
    assert s.active.dtype == np.bool_
    assert s.active.any(), "a tonal target should register as active"
