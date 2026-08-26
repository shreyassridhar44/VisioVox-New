"""SEAVE-SFO: the suppression-first training objective (docs/04 §4).

SI-SDR treats every error identically. Decomposed, the error is not
perceptually uniform:

    s_hat = s_target + e_interference + e_noise + e_artifact

Artifact sounds like poor audio quality — annoying. Interference sounds like
*another person talking* — disqualifying for a product whose stated
requirement is that the other speaker is almost inaudible. A system trading
1 dB of artifact for 4 dB less leakage is a large win that SI-SDR scores as a
small loss, so the distinction has to live in the objective rather than be left
to the metric.

    L = w_sisdr       * L_sisdr        fidelity to the target
      + w_suppress    * L_suppress     interferer energy in the output
      + w_consistency * L_consistency  output must sound like the enrolled speaker
      + w_mrstft      * L_mrstft       perceptual texture
      + w_silence     * L_silence      near-silence where the target is silent

L_silence carries more weight than its size suggests. Each speaker is silent
50-70% of a real conversation; that is the majority of the listening
experience and where leakage is most audible, because nothing masks it. It is
also where SI-SDR is weakest, having no target signal to reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812

EPS = 1e-8


@dataclass(frozen=True)
class LossWeights:
    """Defaults follow the ablation table in docs/04 §4."""

    sisdr: float = 1.0
    suppress: float = 0.5
    consistency: float = 0.2
    mrstft: float = 0.3
    silence: float = 0.5
    # Hinge threshold in dB. Suppression stops being pushed once the output
    # correlates with an interferer below this level, so the term cannot keep
    # driving toward over-suppression that eats the target.
    suppress_tau_db: float = 20.0


def _zero_mean(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=-1, keepdim=True)


def si_sdr(estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Scale-invariant SDR in dB, per item. Shapes are (batch, samples)."""
    estimate, reference = _zero_mean(estimate), _zero_mean(reference)
    dot = (estimate * reference).sum(dim=-1, keepdim=True)
    energy = (reference**2).sum(dim=-1, keepdim=True) + EPS
    target = dot / energy * reference
    noise = estimate - target
    ratio = (target**2).sum(dim=-1) / ((noise**2).sum(dim=-1) + EPS)
    return 10 * torch.log10(ratio + EPS)


def sisdr_loss(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -si_sdr(estimate, target).mean()


def suppression_loss(
    estimate: torch.Tensor, interferers: torch.Tensor, tau_db: float = 20.0
) -> torch.Tensor:
    """Hinged penalty on the output correlating with any interferer.

    interferers is (batch, n_interferers, samples).

    Hinged deliberately. An unbounded term keeps pushing correlation downward
    long after the interferer is inaudible, and the only remaining way to do
    that is to distort the target. The hinge releases below -tau dB.
    """
    if interferers.numel() == 0:
        return estimate.new_zeros(())

    batch, n_int, samples = interferers.shape
    flat_est = estimate.unsqueeze(1).expand(batch, n_int, samples).reshape(-1, samples)
    flat_int = interferers.reshape(-1, samples)

    # A high SI-SDR against an interferer means the output resembles it.
    leak_db = si_sdr(flat_est, flat_int)
    return F.relu(tau_db + leak_db).mean()


def silence_loss(
    estimate: torch.Tensor,
    active: torch.Tensor,
    target: torch.Tensor | None = None,
) -> torch.Tensor:
    """Energy where the target is silent, relative to where it speaks.

    active is a per-frame mask, (batch, frames).

    Relative rather than absolute, so the term cannot be satisfied by simply
    making the whole output quieter — which would score well against an
    absolute energy penalty while destroying the target.
    """
    batch, samples = estimate.shape
    frames = active.shape[-1]
    if frames == 0:
        return estimate.new_zeros(())

    per_frame = samples // frames
    if per_frame == 0:
        return estimate.new_zeros(())
    usable = per_frame * frames

    framed = estimate[:, :usable].reshape(batch, frames, per_frame)
    energy = (framed**2).mean(dim=-1)

    silent_mask = 1.0 - active
    silent_energy = (energy * silent_mask).sum(dim=-1) / (silent_mask.sum(dim=-1) + EPS)

    # Normalise against the ESTIMATE's own speech energy, not the target's.
    # Using the target leaves the term gameable: scaling the whole output down
    # shrinks the numerator while the denominator stays fixed, so a model can
    # win by going quiet and destroying the target. Dividing by the estimate's
    # own speech energy makes the ratio scale-invariant, which is the property
    # this term needs.
    est_speech = (energy * active).sum(dim=-1) / (active.sum(dim=-1) + EPS)

    return (silent_energy / (est_speech + EPS)).mean()


def consistency_loss(
    estimate_embedding: torch.Tensor, enrolment_embedding: torch.Tensor
) -> torch.Tensor:
    """One minus cosine similarity between output and enrolment embeddings.

    Attacks leakage at the identity level rather than the waveform level: an
    output carrying another person's voice moves away from the enrolled
    speaker even when the waveform error is small.
    """
    return (1.0 - F.cosine_similarity(estimate_embedding, enrolment_embedding, dim=-1)).mean()


def mrstft_loss(
    estimate: torch.Tensor,
    target: torch.Tensor,
    fft_sizes: tuple[int, ...] = (512, 1024, 2048),
    hop_ratio: float = 0.25,
) -> torch.Tensor:
    """Multi-resolution spectral convergence plus log-magnitude distance.

    SI-SDR is phase-sensitive and texture-blind; this is the term that moves
    perceived quality rather than the number.
    """
    total = estimate.new_zeros(())
    for n_fft in fft_sizes:
        hop = max(1, int(n_fft * hop_ratio))
        window = torch.hann_window(n_fft, device=estimate.device, dtype=estimate.dtype)
        est_mag = torch.stft(
            estimate, n_fft=n_fft, hop_length=hop, window=window, return_complex=True
        ).abs()
        ref_mag = torch.stft(
            target, n_fft=n_fft, hop_length=hop, window=window, return_complex=True
        ).abs()

        convergence = torch.norm(ref_mag - est_mag, p="fro") / (torch.norm(ref_mag, p="fro") + EPS)
        log_mag = F.l1_loss(torch.log(est_mag + EPS), torch.log(ref_mag + EPS))
        total = total + convergence + log_mag
    return total / len(fft_sizes)


@dataclass
class LossBreakdown:
    """Every term kept separately, because the ablation needs them."""

    total: torch.Tensor
    sisdr: torch.Tensor
    suppress: torch.Tensor
    consistency: torch.Tensor
    mrstft: torch.Tensor
    silence: torch.Tensor

    def as_floats(self) -> dict[str, float]:
        return {
            "total": float(self.total),
            "sisdr": float(self.sisdr),
            "suppress": float(self.suppress),
            "consistency": float(self.consistency),
            "mrstft": float(self.mrstft),
            "silence": float(self.silence),
        }


def seave_sfo_loss(
    estimate: torch.Tensor,
    target: torch.Tensor,
    interferers: torch.Tensor | None = None,
    active: torch.Tensor | None = None,
    estimate_embedding: torch.Tensor | None = None,
    enrolment_embedding: torch.Tensor | None = None,
    weights: LossWeights | None = None,
) -> LossBreakdown:
    """The full objective.

    An absent input zeroes its term rather than raising, so running an ablation
    is a change of weights and not a different code path.
    """
    w = weights or LossWeights()
    zero = estimate.new_zeros(())

    l_sisdr = sisdr_loss(estimate, target)
    l_suppress = (
        suppression_loss(estimate, interferers, w.suppress_tau_db)
        if interferers is not None and interferers.numel() > 0
        else zero
    )
    l_consistency = (
        consistency_loss(estimate_embedding, enrolment_embedding)
        if estimate_embedding is not None and enrolment_embedding is not None
        else zero
    )
    l_mrstft = mrstft_loss(estimate, target)
    l_silence = silence_loss(estimate, active, target) if active is not None else zero

    total = (
        w.sisdr * l_sisdr
        + w.suppress * l_suppress
        + w.consistency * l_consistency
        + w.mrstft * l_mrstft
        + w.silence * l_silence
    )
    return LossBreakdown(total, l_sisdr, l_suppress, l_consistency, l_mrstft, l_silence)
