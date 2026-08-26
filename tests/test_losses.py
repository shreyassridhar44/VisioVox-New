"""SEAVE-SFO objective tests (docs/04 §4).

The decisive test is `test_objective_prefers_artifact_over_leakage`. The whole
contribution is the claim that SI-SDR cannot express "the other speaker must be
inaudible" and that this objective can. If the full loss does not actually rank
a slightly-distorted output above a leaky one, the objective does not do the
thing it was designed for and no amount of training will fix that.
"""

from __future__ import annotations

import pytest
import torch

from training.losses import (
    LossWeights,
    consistency_loss,
    mrstft_loss,
    seave_sfo_loss,
    si_sdr,
    silence_loss,
    sisdr_loss,
    suppression_loss,
)

RATE = 16_000
torch.manual_seed(0)


def _tone(freq: float, seconds: float = 1.0, batch: int = 2) -> torch.Tensor:
    t = torch.arange(int(seconds * RATE)) / RATE
    return (0.3 * torch.sin(2 * torch.pi * freq * t)).unsqueeze(0).repeat(batch, 1)


# --------------------------------------------------------------------------
# fidelity
# --------------------------------------------------------------------------


def test_si_sdr_rewards_a_perfect_estimate() -> None:
    x = _tone(220)
    assert si_sdr(x, x).mean() > 60


def test_si_sdr_is_scale_invariant() -> None:
    """Compared on an imperfect estimate: a perfect one sits at the numerical
    ceiling where any two values look equal for the wrong reason."""
    x = _tone(220)
    est = x + 0.1 * torch.randn_like(x)
    assert torch.allclose(si_sdr(est * 5.0, x), si_sdr(est, x), atol=0.01)


def test_sisdr_loss_falls_as_the_estimate_improves() -> None:
    target = _tone(220)
    noise = 0.1 * torch.randn_like(target)
    worse = sisdr_loss(target + noise, target)
    better = sisdr_loss(target + 0.1 * noise, target)
    assert better < worse


# --------------------------------------------------------------------------
# suppression — the term SI-SDR cannot provide
# --------------------------------------------------------------------------


def test_suppression_penalises_leakage() -> None:
    target, interferer = _tone(220), _tone(440)
    interferers = interferer.unsqueeze(1)

    clean = suppression_loss(target, interferers)
    leaky = suppression_loss(target + 0.5 * interferer, interferers)
    assert leaky > clean


def test_suppression_is_hinged_and_stops_pushing() -> None:
    """Unbounded suppression keeps going after the interferer is inaudible, and
    the only way left to reduce correlation is to damage the target."""
    target, interferer = _tone(220), _tone(440)
    interferers = interferer.unsqueeze(1)

    # orthogonal tones: already far below the hinge
    at_tau_10 = suppression_loss(target, interferers, tau_db=10.0)
    assert at_tau_10 == pytest.approx(0.0, abs=1e-5), "hinge did not release"

    # a very high threshold should engage the term again
    at_tau_100 = suppression_loss(target, interferers, tau_db=100.0)
    assert at_tau_100 > 0


def test_suppression_is_zero_with_no_interferers() -> None:
    target = _tone(220)
    empty = torch.zeros(target.shape[0], 0, target.shape[1])
    assert suppression_loss(target, empty) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# silence — where leakage is most audible
# --------------------------------------------------------------------------


def _with_pause(
    batch: int = 2, frames: int = 100, per_frame: int = 640
) -> tuple[torch.Tensor, torch.Tensor, int]:
    samples = frames * per_frame
    t = torch.arange(samples) / RATE
    target = (0.3 * torch.sin(2 * torch.pi * 220 * t)).unsqueeze(0).repeat(batch, 1)
    active = torch.zeros(batch, frames)
    active[:, :50] = 1.0
    target[:, 50 * per_frame :] = 0.0  # genuinely silent second half
    return target, active, per_frame


def test_silence_penalises_residual_in_pauses() -> None:
    target, active, per_frame = _with_pause()
    clean = silence_loss(target, active, target)

    leaking = target.clone()
    t = torch.arange(50 * per_frame) / RATE
    leaking[:, 50 * per_frame :] = 0.1 * torch.sin(2 * torch.pi * 440 * t)
    leaky = silence_loss(leaking, active, target)

    assert leaky > clean * 10, f"clean {clean:.2e} vs leaky {leaky:.2e}"


def test_silence_cannot_be_gamed_by_turning_everything_down() -> None:
    """An absolute energy penalty is satisfied by making the whole output
    quieter, which destroys the target. This one is relative, so it is not."""
    target, active, per_frame = _with_pause()
    leaking = target.clone()
    t = torch.arange(50 * per_frame) / RATE
    leaking[:, 50 * per_frame :] = 0.1 * torch.sin(2 * torch.pi * 440 * t)

    loud = silence_loss(leaking, active, target)
    quiet = silence_loss(leaking * 0.01, active, target)
    assert quiet == pytest.approx(loud, rel=0.05), "scaling changed the penalty"


def test_silence_is_zero_when_the_pause_is_truly_silent() -> None:
    target, active, _ = _with_pause()
    assert silence_loss(target, active, target) == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------
# consistency and texture
# --------------------------------------------------------------------------


def test_consistency_rewards_a_matching_identity() -> None:
    e = torch.nn.functional.normalize(torch.randn(4, 192), dim=-1)
    other = torch.nn.functional.normalize(torch.randn(4, 192), dim=-1)
    assert consistency_loss(e, e) < consistency_loss(e, other)
    assert consistency_loss(e, e) == pytest.approx(0.0, abs=1e-5)


def test_mrstft_rewards_spectral_agreement() -> None:
    target = _tone(220)
    assert mrstft_loss(target, target) < mrstft_loss(_tone(880), target)


# --------------------------------------------------------------------------
# the contribution
# --------------------------------------------------------------------------


def test_objective_prefers_artifact_over_leakage() -> None:
    """The claim, tested directly.

    Two outputs with matched error energy. One leaks the interferer; one has
    the interferer fully suppressed but carries unrelated noise instead. A
    listener strongly prefers the second — it is rough, not haunted by another
    person. SI-SDR alone must not distinguish them; SEAVE-SFO must.
    """
    target, interferer = _tone(220, 2.0), _tone(440, 2.0)
    noise = torch.randn_like(target)
    noise = noise / noise.norm() * interferer.norm()  # matched energy

    leaky = target + 0.4 * interferer
    distorted = target + 0.4 * noise
    interferers = interferer.unsqueeze(1)

    sisdr_gap = abs(float(sisdr_loss(leaky, target)) - float(sisdr_loss(distorted, target)))
    assert sisdr_gap < 3.0, f"SI-SDR alone already separates them by {sisdr_gap:.2f}"

    w = LossWeights()
    total_leaky = float(seave_sfo_loss(leaky, target, interferers, weights=w).total)
    total_distorted = float(seave_sfo_loss(distorted, target, interferers, weights=w).total)

    assert total_distorted < total_leaky, (
        "the objective does not prefer artifact over leakage: "
        f"distorted {total_distorted:.3f} vs leaky {total_leaky:.3f}"
    )


def test_breakdown_reports_every_term() -> None:
    """The ablation needs each term separately, not just the total."""
    target, interferer = _tone(220), _tone(440)
    active = torch.ones(target.shape[0], 25)
    emb = torch.nn.functional.normalize(torch.randn(target.shape[0], 192), dim=-1)

    out = seave_sfo_loss(
        target + 0.1 * interferer,
        target,
        interferer.unsqueeze(1),
        active,
        emb,
        emb,
    )
    floats = out.as_floats()
    assert set(floats) == {"total", "sisdr", "suppress", "consistency", "mrstft", "silence"}
    assert all(v == v for v in floats.values()), "a term produced NaN"


def test_missing_inputs_zero_their_term_rather_than_failing() -> None:
    """An ablation should be a weight change, not a different code path."""
    target = _tone(220)
    out = seave_sfo_loss(target + 0.05 * torch.randn_like(target), target)
    assert out.suppress == pytest.approx(0.0)
    assert out.consistency == pytest.approx(0.0)
    assert out.silence == pytest.approx(0.0)
    assert float(out.total) == pytest.approx(
        float(LossWeights().sisdr * out.sisdr + LossWeights().mrstft * out.mrstft), rel=1e-5
    )


def test_loss_is_differentiable() -> None:
    target, interferer = _tone(220), _tone(440)
    estimate = (target + 0.2 * interferer).requires_grad_(True)
    active = torch.ones(target.shape[0], 25)
    out = seave_sfo_loss(estimate, target, interferer.unsqueeze(1), active)
    out.total.backward()  # type: ignore[no-untyped-call]
    assert estimate.grad is not None
    assert torch.isfinite(estimate.grad).all(), "non-finite gradient"
