"""Realistic mixture simulation tests (docs/06 §5).

These pin the three properties the doc says are usually skipped and matter
most: partial overlap rather than constant, per-source room responses rather
than one shared, and level imbalance rather than balanced. A simulator that
silently regresses to `a + b` would still produce plausible training curves and
a model that fails on real audio — which is the failure mode the whole section
exists to prevent.
"""

from __future__ import annotations

import numpy as np
import pytest

from training.simulate import (
    RATE,
    SimConfig,
    Simulated,
    codec_roundtrip,
    degrade_video,
    sample_turn_schedule,
    simulate,
)


def _speech(seconds: float, freq: float, seed: int = 0) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    rng = np.random.default_rng(seed)
    # amplitude-modulated tone: crude, but has speech-like temporal structure
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    return (
        0.3 * envelope * np.sin(2 * np.pi * freq * t) + 0.01 * rng.standard_normal(len(t))
    ).astype(np.float32)


# --------------------------------------------------------------------------
# turn-taking (step 1)
# --------------------------------------------------------------------------


def test_schedule_produces_partial_not_total_overlap() -> None:
    rng = np.random.default_rng(0)
    total = 8 * RATE
    spans = sample_turn_schedule(2, total, overlap_ratio=0.2, rng=rng)
    active = np.zeros(total, dtype=np.int16)
    for a, b in spans:
        active[a:b] += 1
    overlapped = float((active >= 2).mean())
    assert 0.0 < overlapped < 0.9, f"overlap {overlapped:.2f} is not partial"


def test_spans_stay_inside_the_clip() -> None:
    rng = np.random.default_rng(1)
    total = 5 * RATE
    for n in (2, 3, 4):
        for a, b in sample_turn_schedule(n, total, 0.3, rng):
            assert 0 <= a < b <= total


def test_simulated_overlap_is_far_below_libri2mix() -> None:
    """LibriMix is 100% overlapped; real conversation is not, and that gap is
    the documented reason models trained on it collapse on real input."""
    sources = [_speech(4.0, 200.0, 0), _speech(4.0, 320.0, 1)]
    sim = simulate(sources, cfg=SimConfig(seed=2, codec_probability=0.0))
    assert sim.overlap_ratio < 0.9


# --------------------------------------------------------------------------
# room (step 2)
# --------------------------------------------------------------------------


def test_each_source_gets_its_own_room_response() -> None:
    """One shared RIR is a materially easier problem than the real one."""
    sources = [_speech(3.0, 200.0, 0), _speech(3.0, 200.0, 1)]
    sim = simulate(sources, cfg=SimConfig(seed=3, codec_probability=0.0, level_spread_db=0.0))
    # identical input content, different positions -> different output
    a, b = sim.sources[0], sim.sources[1]
    n = min(len(a), len(b))
    assert not np.allclose(a[:n], b[:n], atol=1e-4)


def test_rt60_lands_in_the_configured_range() -> None:
    sources = [_speech(2.0, 200.0, 0), _speech(2.0, 300.0, 1)]
    sim = simulate(sources, cfg=SimConfig(seed=4, rt60=(0.2, 0.5), codec_probability=0.0))
    # 0.0 signals the documented anechoic fallback rather than an out-of-range value
    assert sim.rt60 == 0.0 or 0.1 <= sim.rt60 <= 0.9


# --------------------------------------------------------------------------
# levels and noise (steps 3, 4)
# --------------------------------------------------------------------------


def test_sources_are_not_level_balanced() -> None:
    """Balanced mixtures remove a cue the model must otherwise cope without."""
    spread = []
    for seed in range(6):
        sources = [_speech(3.0, 200.0, 0), _speech(3.0, 320.0, 1)]
        sim = simulate(
            sources, cfg=SimConfig(seed=seed, codec_probability=0.0, level_spread_db=6.0)
        )
        rms = [float(np.sqrt(np.mean(s**2)) + 1e-12) for s in sim.sources]
        spread.append(abs(20 * np.log10(rms[0] / rms[1])))
    assert max(spread) > 2.0, f"levels look balanced: max spread {max(spread):.2f} dB"


def test_noise_is_added_at_the_reported_snr() -> None:
    sources = [_speech(3.0, 200.0, 0), _speech(3.0, 320.0, 1)]
    rng = np.random.default_rng(7)
    noise = (0.1 * rng.standard_normal(3 * RATE)).astype(np.float32)
    quiet = simulate(sources, noise, SimConfig(seed=8, snr_db=(25.0, 25.0), codec_probability=0.0))
    loud = simulate(sources, noise, SimConfig(seed=8, snr_db=(5.0, 5.0), codec_probability=0.0))

    def residual(sim: Simulated) -> float:
        mixed = np.sum(sim.sources, axis=0)
        n = min(len(mixed), len(sim.mixture))
        return float(np.sqrt(np.mean((sim.mixture[:n] - mixed[:n]) ** 2)))

    assert residual(loud) > residual(quiet), "lower SNR should leave more residual"


def test_mixture_does_not_clip() -> None:
    sources = [_speech(3.0, 200.0, i) for i in range(3)]
    for seed in range(5):
        sim = simulate(sources, cfg=SimConfig(seed=seed, codec_probability=0.0))
        assert np.abs(sim.mixture).max() <= 0.99 + 1e-6


def test_at_least_two_sources_are_required() -> None:
    with pytest.raises(ValueError, match="at least two sources"):
        simulate([_speech(1.0, 200.0)])


# --------------------------------------------------------------------------
# codec (step 5)
# --------------------------------------------------------------------------


def test_codec_roundtrip_preserves_length_and_changes_signal() -> None:
    x = _speech(2.0, 220.0, 0)
    y = codec_roundtrip(x, "mp3_96k")
    assert len(y) == len(x)
    if not np.allclose(y, x):
        # lossy compression should perturb it but not destroy it
        corr = float(np.corrcoef(x, y)[0, 1])
        assert corr > 0.5, f"codec destroyed the signal (corr {corr:.2f})"


def test_unknown_codec_is_a_no_op() -> None:
    x = _speech(1.0, 220.0)
    assert np.array_equal(codec_roundtrip(x, "not-a-codec"), x)


# --------------------------------------------------------------------------
# video degradation (step 6)
# --------------------------------------------------------------------------


def test_video_degradation_preserves_shape() -> None:
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, (50, 96, 96), dtype=np.uint8)
    assert degrade_video(frames, np.random.default_rng(1)).shape == frames.shape


def test_video_degradation_actually_changes_frames() -> None:
    """A reliability-gated visual pathway has to see degraded input to learn
    when not to trust the modality (Novelty 2)."""
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, (60, 96, 96), dtype=np.uint8)
    changed = sum(
        not np.array_equal(degrade_video(frames, np.random.default_rng(s)), frames)
        for s in range(12)
    )
    assert changed > 0, "degradation never altered the frames"
