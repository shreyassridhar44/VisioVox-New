"""Overlap-add reconstruction and permutation-measurement tests.

The reconstruction test is required by the ML conventions: chunked inference is
worthless if stitching does not reproduce the input, and the failure mode
(periodic amplitude ripple at the hop rate) is subtle enough to survive casual
listening.

The permutation tests use synthetic estimates with a known answer, so the
measurement is validated independently of any model. Otherwise a bug in the
metric and a genuine finding look identical.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval.permutation import measure, si_sdr
from pipeline.s5_separate import (
    WindowedSeparation,
    identity_assignment,
    overlap_add,
)

RATE = 16_000


def _windowed(signal_per_source: np.ndarray, win: int, hop: int) -> WindowedSeparation:
    """Cut known sources into windows, preserving channel order."""
    n = signal_per_source.shape[1]
    starts = list(range(0, max(1, n - win + 1), hop))
    if starts[-1] + win < n:
        starts.append(n - win)
    est = []
    for s in starts:
        chunk = signal_per_source[:, s : s + win]
        if chunk.shape[1] < win:
            chunk = np.pad(chunk, ((0, 0), (0, win - chunk.shape[1])))
        est.append(chunk)
    return WindowedSeparation(
        estimates=np.stack(est),
        starts=np.asarray(starts, dtype=np.int64),
        window_samples=win,
        hop_samples=hop,
    )


# --------------------------------------------------------------------------
# overlap-add
# --------------------------------------------------------------------------


@pytest.mark.parametrize("hop_ratio", [0.5, 0.25])
def test_overlap_add_reconstructs_exactly(hop_ratio: float) -> None:
    rng = np.random.default_rng(0)
    total = 10 * RATE
    sources = rng.standard_normal((2, total)).astype(np.float32) * 0.1
    win = int(4.0 * RATE)
    hop = int(win * hop_ratio)

    sep = _windowed(sources, win, hop)
    out = overlap_add(sep, identity_assignment(sep), total)

    # Interior samples must reconstruct to numerical precision. Edges are
    # covered by a single tapered window, so they are excluded deliberately.
    a, b = win, total - win
    err = np.abs(out[:, a:b] - sources[:, a:b]).max()
    assert err < 1e-5, f"reconstruction error {err:.2e} exceeds tolerance"


def test_overlap_add_has_no_ripple_at_the_hop_rate() -> None:
    """A constant input must come back constant; ripple means bad normalisation."""
    total = 10 * RATE
    sources = np.ones((2, total), dtype=np.float32)
    win, hop = int(4.0 * RATE), int(2.0 * RATE)
    sep = _windowed(sources, win, hop)
    out = overlap_add(sep, identity_assignment(sep), total)
    interior = out[0, win : total - win]
    assert np.ptp(interior) < 1e-5, "amplitude ripple at the hop rate"


def test_overlap_add_applies_the_assignment() -> None:
    """Swapping the assignment must swap the reconstructed tracks."""
    total = 6 * RATE
    t = np.arange(total) / RATE
    sources = np.stack(
        [
            np.sin(2 * np.pi * 220 * t),
            np.sin(2 * np.pi * 440 * t),
        ]
    ).astype(np.float32)
    win, hop = int(2.0 * RATE), int(1.0 * RATE)
    sep = _windowed(sources, win, hop)

    swapped = identity_assignment(sep)[:, ::-1].copy()
    out = overlap_add(sep, swapped, total)
    a, b = win, total - win
    assert si_sdr(out[0, a:b], sources[1, a:b]) > 30
    assert si_sdr(out[1, a:b], sources[0, a:b]) > 30


# --------------------------------------------------------------------------
# si-sdr
# --------------------------------------------------------------------------


def test_si_sdr_is_scale_invariant() -> None:
    rng = np.random.default_rng(1)
    ref = rng.standard_normal(RATE).astype(np.float32)
    est = ref * 7.3
    assert si_sdr(est, ref) > 100


def test_si_sdr_penalises_the_wrong_source() -> None:
    rng = np.random.default_rng(2)
    a = rng.standard_normal(RATE).astype(np.float32)
    b = rng.standard_normal(RATE).astype(np.float32)
    assert si_sdr(a, a) > si_sdr(b, a) + 40


# --------------------------------------------------------------------------
# permutation measurement
# --------------------------------------------------------------------------


def _two_speaker_refs(total: int) -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.standard_normal((2, total)).astype(np.float32) * 0.1


def test_no_flips_when_channel_order_is_stable() -> None:
    total = 20 * RATE
    refs = _two_speaker_refs(total)
    win, hop = int(4.0 * RATE), int(2.0 * RATE)
    sep = _windowed(refs, win, hop)
    report = measure(sep.estimates, sep.starts, refs, win)
    assert report.n_scored > 5
    assert report.permutation_error_rate == 0.0
    assert "stable" in report.verdict()


def test_detects_every_flip_when_channels_alternate() -> None:
    """Swap channel order on alternate windows; the rate must approach 1.0."""
    total = 20 * RATE
    refs = _two_speaker_refs(total)
    win, hop = int(4.0 * RATE), int(2.0 * RATE)
    sep = _windowed(refs, win, hop)

    est = sep.estimates.copy()
    est[1::2] = est[1::2][:, ::-1]
    report = measure(est, sep.starts, refs, win)
    assert report.permutation_error_rate > 0.9, report
    assert "unusable" in report.verdict()


def test_silent_windows_are_skipped_not_counted() -> None:
    """A window with fewer than two active speakers says nothing about ordering."""
    total = 20 * RATE
    refs = _two_speaker_refs(total)
    refs[:, : 8 * RATE] = 0.0  # first 8 s silent
    win, hop = int(4.0 * RATE), int(2.0 * RATE)
    sep = _windowed(refs, win, hop)
    report = measure(sep.estimates, sep.starts, refs, win)
    assert report.skipped_silent >= 2
    assert report.n_scored < report.n_windows
