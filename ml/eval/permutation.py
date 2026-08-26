"""Permutation-error rate for blind separation (finding F1.1, ADR-0001).

The empirical test the plan asks for in week 3: does PIT-trained blind
separation hold speaker identity across inference windows over a full-length
recording?

Method. For each window we score every output channel against every reference
speaker with SI-SDR and take the best assignment (Hungarian). That gives the
*true* per-window mapping. The permutation-error rate is then the fraction of
consecutive window pairs where that mapping changes -- i.e. how often naive
stitching would swap speakers mid-recording.

Why SI-SDR for matching rather than correlation: correlation is scale
sensitive and rewards a loud channel regardless of content. SI-SDR is scale
invariant, so it compares what was separated rather than how loud it came out.

Interpretation is written down in advance so the result cannot be rationalised
after the fact:

    < 1%    identity is effectively stable; ADR-0001's premise is weak and the
            architecture decision should be revisited now, not in month 5
    1-10%   drift is real; TSE is justified
    > 10%   naive stitching is unusable, exactly as F1.1 predicts
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy.optimize import linear_sum_assignment

from pipeline.vad import FRAME_SAMPLES, speech_masks

EPS = 1e-8


def si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Scale-invariant SDR in dB. Returns -inf for a silent reference."""
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    ref_energy = float(np.dot(reference, reference))
    if ref_energy < EPS:
        return float("-inf")
    alpha = float(np.dot(estimate, reference)) / ref_energy
    target = alpha * reference
    noise = estimate - target
    noise_energy = float(np.dot(noise, noise))
    if noise_energy < EPS:
        return float("inf")
    ratio = float(np.dot(target, target)) / noise_energy
    return float(10.0 * np.log10(ratio))


@dataclass
class PermutationReport:
    n_windows: int
    n_scored: int
    n_flips: int
    permutation_error_rate: float
    mean_best_si_sdr: float
    skipped_silent: int

    def verdict(self) -> str:
        r = self.permutation_error_rate
        if r < 0.01:
            return "stable — ADR-0001's premise is weak; revisit the decision"
        if r < 0.10:
            return "drift present — TSE justified"
        return "naive stitching unusable — F1.1 confirmed"


def measure(
    estimates: np.ndarray,
    starts: np.ndarray,
    references: np.ndarray,
    window_samples: int,
    min_overlap_frames: int = 10,
) -> PermutationReport:
    """
    estimates:  (n_windows, n_sources, window_samples)
    starts:     (n_windows,) window offsets in samples
    references: (n_speakers, total_samples) ground-truth per speaker
    """
    n_windows, n_src, _ = estimates.shape
    n_ref = references.shape[0]
    k = min(n_src, n_ref)
    ref_masks = speech_masks(references)

    assignments: list[tuple[int, ...]] = []
    best_scores: list[float] = []
    skipped = 0

    for w in range(n_windows):
        s = int(starts[w])
        e = s + window_samples
        ref_win = references[:, s:e]
        if ref_win.shape[1] < window_samples:
            pad = window_samples - ref_win.shape[1]
            ref_win = np.pad(ref_win, ((0, 0), (0, pad)))

        # A window where two speakers are never simultaneously active carries
        # no information about ordering; scoring it would dilute the rate.
        # Activity uses the shared dominance VAD: an absolute dB floor marks
        # bleed and room tone as speech on hotter recordings, which made one
        # AMI meeting score every single window as overlapped.
        f0 = s // FRAME_SAMPLES
        f1 = f0 + window_samples // FRAME_SAMPLES
        if int((ref_masks[:, f0:f1].sum(axis=0) >= 2).sum()) < min_overlap_frames:
            skipped += 1
            assignments.append(())
            continue

        cost = np.zeros((n_src, n_ref))
        for c in range(n_src):
            for r in range(n_ref):
                score = si_sdr(estimates[w, c], ref_win[r])
                cost[c, r] = -score if np.isfinite(score) else 1e6
        rows, cols = linear_sum_assignment(cost)
        pairs = sorted(zip(cols.tolist(), rows.tolist(), strict=True))[:k]
        assignments.append(tuple(c for _, c in pairs))
        best_scores.append(float(-cost[rows, cols].mean()))

    scored = [a for a in assignments if a]
    flips = sum(1 for a, b in pairwise(scored) if a != b)
    comparisons = max(0, len(scored) - 1)

    return PermutationReport(
        n_windows=n_windows,
        n_scored=len(scored),
        n_flips=flips,
        permutation_error_rate=(flips / comparisons) if comparisons else 0.0,
        mean_best_si_sdr=float(np.mean(best_scores)) if best_scores else float("nan"),
        skipped_silent=skipped,
    )
