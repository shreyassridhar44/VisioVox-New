"""Reliability-gated conditioning (SEAVE-RG, docs/04 §3).

AV separation literature evaluates on datasets where the face is always
visible, frontal and well-lit. Real uploads have head turns, occlusions, camera
cuts, and speakers who are never on camera. A fixed AV model fed a bad visual
stream does not merely lose the visual benefit — it is actively misled by it.

    alpha_t = sigmoid(g_a(audio_cue_conf))     per utterance
    beta_t  = sigmoid(g_v(visual_conf_t))      PER FRAME
    c_t     = alpha_t * (W_a @ e_spk) + beta_t * (W_v @ v_t) + b

**Per-frame beta is the point.** A speaker who turns away for two seconds
should lose visual conditioning for those two seconds only. Per-utterance
gating — what a hard modality switch gives you — throws away the whole visual
stream because part of it was bad, and that is most of the benefit.

**Modality dropout is what makes the gates work.** During training the visual
stream is randomly zeroed or corrupted and the audio cue randomly zeroed or
polluted. The gates learn to *recognise* corruption from the cue itself rather
than being told about it, which is what lets one set of weights serve
audio-visual, audio-only and visual-only inference with no dispatch logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DropoutConfig:
    """Corruption probabilities for training (docs/04 §3).

    Deliberately high. The gates only learn to distrust a modality if they see
    it fail often enough for the failure to matter to the loss.
    """

    drop_visual: float = 0.20
    drop_audio_cue: float = 0.15
    corrupt_visual: float = 0.25
    corrupt_audio_cue: float = 0.15
    # A sample where both cues are gone has no conditioning signal at all and
    # teaches nothing except to ignore conditioning, so it is excluded.
    allow_both_dropped: bool = False


class ReliabilityGate(nn.Module):
    """Maps a raw confidence to a gate value in 0..1.

    A learned scalar map rather than using the confidence directly: the
    confidences from S4 are calibrated for human interpretation, not for what
    the separator needs, and the useful operating point is discovered during
    training rather than assumed here.
    """

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        first = nn.Linear(1, hidden)
        # Kept as a named attribute, not just a Sequential index: gate
        # calibration needs to reach this layer, and indexing a Sequential
        # loses the type.
        self.output = nn.Linear(hidden, 1)
        self.net = nn.Sequential(first, nn.SiLU(), self.output)
        # Start near identity-ish and open: a gate initialised closed gives the
        # separator no conditioning signal early in training, and it learns to
        # ignore the cue before the gate ever opens.
        nn.init.zeros_(self.output.bias)
        nn.init.constant_(first.bias, 0.5)

    def forward(self, confidence: torch.Tensor) -> torch.Tensor:
        """confidence: (..., ) in 0..1 -> gate: same shape, in 0..1."""
        shape = confidence.shape
        flat = confidence.reshape(-1, 1)
        gate: torch.Tensor = torch.sigmoid(self.net(flat)).reshape(shape)
        return gate


class FiLM(nn.Module):
    """Feature-wise modulation of separator features by the conditioning vector."""

    def __init__(self, cond_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.to_scale = nn.Linear(cond_dim, feature_dim)
        self.to_shift = nn.Linear(cond_dim, feature_dim)
        # Identity at initialisation: scale 1, shift 0, so an untrained
        # conditioning path cannot destroy the separator's features.
        nn.init.zeros_(self.to_scale.weight)
        nn.init.ones_(self.to_scale.bias)
        nn.init.zeros_(self.to_shift.weight)
        nn.init.zeros_(self.to_shift.bias)

    def forward(self, features: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """features: (batch, feature_dim, time); cond: (batch, cond_dim, time)."""
        cond_t = cond.transpose(1, 2)  # (batch, time, cond_dim)
        scale = self.to_scale(cond_t).transpose(1, 2)
        shift = self.to_shift(cond_t).transpose(1, 2)
        out: torch.Tensor = features * scale + shift
        return out


class GatedCrossAttention(nn.Module):
    """Audio frames attend to the visual sequence, weighted by beta.

    The gate multiplies the attention *output*, not the scores. Scaling scores
    would only reshape which frames are attended to; scaling the output is what
    actually removes an unreliable visual contribution from the sum.
    """

    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self, audio: torch.Tensor, visual: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        """audio: (batch, time, dim); visual: (batch, vtime, dim); beta: (batch, time)."""
        attended, _ = self.attn(self.norm(audio), visual, visual, need_weights=False)
        out: torch.Tensor = audio + attended * beta.unsqueeze(-1)
        return out


class ReliabilityGatedConditioning(nn.Module):
    """Builds the per-frame conditioning vector from both cues."""

    def __init__(
        self,
        speaker_dim: int = 192,
        visual_dim: int = 512,
        cond_dim: int = 256,
    ) -> None:
        super().__init__()
        self.audio_gate = ReliabilityGate()
        self.visual_gate = ReliabilityGate()
        self.project_audio = nn.Linear(speaker_dim, cond_dim)
        self.project_visual = nn.Linear(visual_dim, cond_dim)
        self.bias = nn.Parameter(torch.zeros(cond_dim))
        self.cond_dim = cond_dim

    def forward(
        self,
        speaker_embedding: torch.Tensor | None,
        audio_confidence: torch.Tensor,
        visual_features: torch.Tensor | None,
        visual_confidence: torch.Tensor,
        frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (conditioning, alpha, beta).

        conditioning: (batch, cond_dim, frames)
        alpha: (batch, 1)      beta: (batch, frames)
        """
        batch = audio_confidence.shape[0]
        device = audio_confidence.device

        alpha = self.audio_gate(audio_confidence.reshape(batch, 1))
        beta = self.visual_gate(visual_confidence)  # (batch, frames)

        cond = self.bias.view(1, self.cond_dim, 1).expand(batch, self.cond_dim, frames).clone()

        if speaker_embedding is not None:
            a = self.project_audio(speaker_embedding)  # (batch, cond_dim)
            cond = cond + (alpha.unsqueeze(-1) * a.unsqueeze(-1))
        if visual_features is not None:
            v = self.project_visual(visual_features)  # (batch, frames, cond_dim)
            cond = cond + beta.unsqueeze(1) * v.transpose(1, 2)

        _ = device
        return cond, alpha, beta


def apply_modality_dropout(
    speaker_embedding: torch.Tensor | None,
    audio_confidence: torch.Tensor,
    visual_features: torch.Tensor | None,
    visual_confidence: torch.Tensor,
    config: DropoutConfig | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Randomly drop or corrupt each modality, per item in the batch.

    Confidences are lowered alongside the corruption. That pairing is the
    training signal: the gate learns that a low confidence accompanies a cue
    worth ignoring. Corrupting the features while leaving the confidence high
    would teach it the opposite.
    """
    cfg = config or DropoutConfig()
    batch = audio_confidence.shape[0]
    device = audio_confidence.device

    def draw(p: float) -> torch.Tensor:
        return torch.rand(batch, device=device, generator=generator) < p

    drop_v = draw(cfg.drop_visual)
    drop_a = draw(cfg.drop_audio_cue)
    if not cfg.allow_both_dropped:
        # Keep at least one cue: an item with neither teaches only that
        # conditioning can be ignored.
        both = drop_v & drop_a
        drop_v = drop_v & ~both

    corrupt_v = draw(cfg.corrupt_visual) & ~drop_v
    corrupt_a = draw(cfg.corrupt_audio_cue) & ~drop_a

    a_conf = audio_confidence.clone()
    v_conf = visual_confidence.clone()
    emb = None if speaker_embedding is None else speaker_embedding.clone()
    vis = None if visual_features is None else visual_features.clone()

    if emb is not None:
        emb[drop_a] = 0.0
        if corrupt_a.any():
            # Mix in another item's embedding: an interferer's identity is a
            # far more realistic corruption than noise, and it is what actually
            # happens when enrolment picks up the wrong speaker.
            perm = torch.randperm(batch, device=device, generator=generator)
            emb[corrupt_a] = 0.5 * emb[corrupt_a] + 0.5 * emb[perm][corrupt_a]
    a_conf[drop_a] = 0.0
    a_conf[corrupt_a] = a_conf[corrupt_a] * 0.4

    if vis is not None:
        vis[drop_v] = 0.0
        if corrupt_v.any():
            noise = torch.randn(
                vis[corrupt_v].shape, device=device, generator=generator, dtype=vis.dtype
            )
            vis[corrupt_v] = 0.5 * vis[corrupt_v] + 0.5 * noise
    v_conf[drop_v] = 0.0
    v_conf[corrupt_v] = v_conf[corrupt_v] * 0.4

    return emb, a_conf, vis, v_conf
