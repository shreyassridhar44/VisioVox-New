"""Separation metrics (docs/08 §2).

One implementation, used for every system including baselines. Numbers are
never quoted from papers, because published figures come from different
harnesses with different windowing, resampling and edge handling, and the
differences are comparable to the effects being measured.

**SIR is the metric that matches the requirement.** The stated goal is that
unselected speakers are inaudible, which is an interference property. SI-SDR
cannot express it: it sums interference and artifact into one number, so a
model can score well on SI-SDR while another person is clearly audible
underneath. Where the two disagree, SIR is the one that means what the product
promises.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-8


def _zero_mean(x: np.ndarray) -> np.ndarray:
    return x - x.mean()


def _project(estimate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Least-squares projection of `estimate` onto `reference`."""
    energy = float(np.dot(reference, reference))
    if energy < EPS:
        return np.zeros_like(reference)
    return (float(np.dot(estimate, reference)) / energy) * reference


def si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Scale-invariant SDR in dB."""
    estimate, reference = _zero_mean(estimate), _zero_mean(reference)
    if float(np.dot(reference, reference)) < EPS:
        return float("nan")
    target = _project(estimate, reference)
    noise = estimate - target
    noise_energy = float(np.dot(noise, noise))
    if noise_energy < EPS:
        return float("inf")
    return float(10 * np.log10(float(np.dot(target, target)) / noise_energy))


def si_sdri(estimate: np.ndarray, reference: np.ndarray, mixture: np.ndarray) -> float:
    """Improvement over doing nothing.

    Absolute SI-SDR depends on how much of the mixture the target occupies, so
    it is not comparable across items. Improvement over the unprocessed mixture
    is, which is why every target in docs/01 is stated as SI-SDRi.
    """
    return si_sdr(estimate, reference) - si_sdr(mixture, reference)


@dataclass(frozen=True)
class BssMetrics:
    """Decomposition of the estimate into target, interference and artifact."""

    sdr: float
    sir: float
    sar: float


def bss_decompose(
    estimate: np.ndarray, reference: np.ndarray, interferers: list[np.ndarray]
) -> BssMetrics:
    """BSS-eval style SDR / SIR / SAR.

    The estimate is projected onto the target, then onto the span of all
    sources. What lands on the other sources is interference; what lands
    outside the span entirely is artifact. Separating those two is the whole
    point -- leakage and distortion are different failures, and only one of
    them breaks the product's promise.
    """
    estimate = _zero_mean(estimate)
    reference = _zero_mean(reference)

    s_target = _project(estimate, reference)

    # Projection onto the subspace spanned by every source.
    sources = np.stack([reference, *[_zero_mean(i) for i in interferers]])
    gram = sources @ sources.T
    rhs = sources @ estimate
    try:
        coeffs = np.linalg.solve(gram + EPS * np.eye(len(sources)), rhs)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    s_all = coeffs @ sources

    e_interf = s_all - s_target
    e_artif = estimate - s_all

    def db(num: np.ndarray, den: np.ndarray) -> float:
        n, d = float(np.dot(num, num)), float(np.dot(den, den))
        if d < EPS:
            return float("inf")
        if n < EPS:
            return float("-inf")
        return float(10 * np.log10(n / d))

    return BssMetrics(
        sdr=db(s_target, e_interf + e_artif),
        sir=db(s_target, e_interf),
        sar=db(s_target + e_interf, e_artif),
    )


def silence_leakage_db(
    estimate: np.ndarray, target_active: np.ndarray, frame_samples: int = 640
) -> float:
    """Residual level where the target is silent, relative to where it speaks.

    This is the measurement that matches "the other speaker is inaudible" most
    directly. A model can hold a good SI-SDR while leaving the interferer
    audible in the target's pauses, and those pauses are exactly where a
    listener notices. More negative is better.
    """
    n = min(len(target_active), len(estimate) // frame_samples)
    if n == 0:
        return float("nan")
    frames = estimate[: n * frame_samples].reshape(n, frame_samples)
    energy = (frames**2).mean(axis=1)
    active = target_active[:n].astype(bool)
    if not active.any() or active.all():
        return float("nan")
    speech = float(energy[active].mean())
    silence = float(energy[~active].mean())
    if speech < EPS:
        return float("nan")
    return float(10 * np.log10((silence + EPS) / speech))


def pesq_wb(estimate: np.ndarray, reference: np.ndarray, rate: int = 16_000) -> float:
    """Wideband PESQ. Returns NaN when the clip is too short or degenerate."""
    try:
        from pesq import pesq as _pesq

        return float(_pesq(rate, reference.astype(np.float64), estimate.astype(np.float64), "wb"))
    except Exception:
        # PESQ raises on silent or very short input; a NaN row is more honest
        # than a fabricated score, and the harness drops NaNs when averaging.
        return float("nan")


def stoi(estimate: np.ndarray, reference: np.ndarray, rate: int = 16_000) -> float:
    """Extended STOI intelligibility, 0..1."""
    try:
        from pystoi import stoi as _stoi

        return float(_stoi(reference, estimate, rate, extended=True))
    except Exception:
        return float("nan")
