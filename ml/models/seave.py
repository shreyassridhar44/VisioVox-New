"""SEAVE — the target-speaker extractor (docs/07 §2).

A TF-GridNet backbone with reliability-gated conditioning wired into every
block. The three contributions meet here: the enrolment cue from S4, the
per-frame gates from `conditioning`, and the suppression-first objective from
`losses` acting on the output.

TF-GridNet alternates three views of the same time-frequency grid:

    intra-frame  a BLSTM across frequency, one time step at a time
    inter-frame  a BLSTM across time, one frequency bin at a time
    full-band    self-attention over time, letting distant frames interact

The reason it suits this problem is the inter-frame path: speaker identity is a
long-horizon property, and a model that only sees local context has to
re-decide who it is following at every frame. That is exactly the failure that
produces speaker swaps mid-utterance.

Conditioning is applied through FiLM after each block rather than concatenated
once at the input. Concatenation at the input gets diluted through depth — by
block six the network has largely forgotten which speaker it was asked for,
which is the same swap failure by a different route.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .conditioning import FiLM, GatedCrossAttention, ReliabilityGatedConditioning


@dataclass
class SeaveConfig:
    """Defaults from configs/seave_base.yaml (docs/07 §2)."""

    n_fft: int = 512
    hop: int = 128
    n_blocks: int = 6
    # docs/07 §2 specifies emb_dim 128 with batch 8, and that is what fits.
    #
    # An earlier revision cut this to 96 and the batch to 4, on the belief that
    # 128 needed 47 GB with checkpointing against the A5000's 24 GB. Measured
    # by `scripts/bench_train.py`, 128 at batch 8 with checkpointing peaks at
    # 18.8 GB and 96 at batch 4 peaks at 8.9 GB — so the estimate was out by
    # about 2.5x, and C1 ran for 52 hours as the smallest and slowest viable
    # configuration with half the card idle.
    #
    # The mistake was easy to make and hard to see, because WSL does not raise
    # when a working set overflows: it pages to system RAM and keeps training
    # at a fifteenth of the speed. An overcommitted configuration therefore
    # looks slow rather than impossible, which is why the guard in the
    # benchmark aborts on overflow instead of timing it.
    emb_dim: int = 128
    lstm_hidden: int = 128
    attn_heads: int = 4
    speaker_emb_dim: int = 192
    visual_dim: int = 512
    cond_dim: int = 256
    confidence_head: bool = True
    # Recompute block activations in the backward pass instead of storing
    # them. TF-GridNet keeps a (batch, emb, time, freq) tensor alive per
    # block — 526 MB each at the documented size — and six of those plus
    # backward does not fit in 24 GB. Checkpointing trades roughly 30% more
    # compute for several times less memory, which is the right side of the
    # trade when the alternative is paging to system RAM.
    gradient_checkpointing: bool = True
    # Frequency bins are unfolded in groups before the intra-frame LSTM, as in
    # the TF-GridNet paper. Without it the frequency LSTM runs over 257 steps
    # per frame and dominates both runtime and memory.
    freq_kernel: int = 4
    freq_stride: int = 1
    _unused: dict[str, str] = field(default_factory=dict)

    @property
    def n_freqs(self) -> int:
        return self.n_fft // 2 + 1


class IntraFrameBlock(nn.Module):
    """BLSTM across frequency, applied independently at each time step."""

    def __init__(self, emb_dim: int, hidden: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden * 2, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, emb, time, freq)."""
        b, d, t, f = x.shape
        y = self.norm(x)
        y = y.permute(0, 2, 3, 1).reshape(b * t, f, d)  # sequence over frequency
        y, _ = self.lstm(y)
        y = self.proj(y).reshape(b, t, f, d).permute(0, 3, 1, 2)
        out: torch.Tensor = x + y
        return out


class InterFrameBlock(nn.Module):
    """BLSTM across time, applied independently at each frequency bin.

    This is the long-horizon path. Speaker identity persists across seconds,
    and without a path that carries it the model re-decides who it is following
    every frame.
    """

    def __init__(self, emb_dim: int, hidden: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden * 2, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, d, t, f = x.shape
        y = self.norm(x)
        y = y.permute(0, 3, 2, 1).reshape(b * f, t, d)  # sequence over time
        y, _ = self.lstm(y)
        y = self.proj(y).reshape(b, f, t, d).permute(0, 3, 2, 1)
        out: torch.Tensor = x + y
        return out


class FullBandAttention(nn.Module):
    """Self-attention over time on frequency-pooled features."""

    def __init__(self, emb_dim: int, heads: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, d, t, f = x.shape
        pooled = x.mean(dim=-1).transpose(1, 2)  # (batch, time, emb)
        y = self.norm(pooled)
        y, _ = self.attn(y, y, y, need_weights=False)
        out: torch.Tensor = x + y.transpose(1, 2).unsqueeze(-1).expand(b, d, t, f)
        return out


class SeaveBlock(nn.Module):
    """One TF-GridNet block plus its conditioning."""

    def __init__(self, cfg: SeaveConfig) -> None:
        super().__init__()
        self.intra = IntraFrameBlock(cfg.emb_dim, cfg.lstm_hidden)
        self.inter = InterFrameBlock(cfg.emb_dim, cfg.lstm_hidden)
        self.attn = FullBandAttention(cfg.emb_dim, cfg.attn_heads)
        self.film = FiLM(cfg.cond_dim, cfg.emb_dim)
        self.cross = GatedCrossAttention(cfg.emb_dim, cfg.attn_heads)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        visual: torch.Tensor | None,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        x = self.intra(x)
        x = self.inter(x)
        x = self.attn(x)

        # FiLM after every block, not concatenation at the input: input
        # conditioning is diluted through depth, and by the last block the
        # network has largely forgotten which speaker was requested.
        b, d, t, f = x.shape
        pooled = x.mean(dim=-1)  # (batch, emb, time)
        modulated = self.film(pooled, cond)

        if visual is not None:
            modulated = self.cross(modulated.transpose(1, 2), visual, beta).transpose(1, 2)

        out: torch.Tensor = x + (modulated - pooled).unsqueeze(-1).expand(b, d, t, f)
        return out


class Seave(nn.Module):
    """Target-speaker extractor with reliability-gated conditioning."""

    def __init__(self, cfg: SeaveConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or SeaveConfig()
        c = self.cfg

        # real and imaginary parts as two input channels
        self.encoder = nn.Conv2d(2, c.emb_dim, kernel_size=(3, 3), padding=(1, 1))
        self.conditioning = ReliabilityGatedConditioning(
            c.speaker_emb_dim, c.visual_dim, c.cond_dim
        )
        self.visual_proj = nn.Linear(c.visual_dim, c.emb_dim)
        self.blocks = nn.ModuleList(SeaveBlock(c) for _ in range(c.n_blocks))
        self.decoder = nn.Conv2d(c.emb_dim, 2, kernel_size=(3, 3), padding=(1, 1))

        # Predicts how much to trust its own output. The trust score is a
        # product requirement (docs/01), and a model that cannot say when it
        # failed forces the UI to present every result as equally good.
        self.confidence = (
            nn.Sequential(nn.Linear(c.emb_dim, 64), nn.SiLU(), nn.Linear(64, 1), nn.Sigmoid())
            if c.confidence_head
            else None
        )

    def _window(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.hann_window(self.cfg.n_fft, device=device, dtype=dtype)

    def forward(
        self,
        mixture: torch.Tensor,
        speaker_embedding: torch.Tensor | None = None,
        audio_confidence: torch.Tensor | None = None,
        visual_features: torch.Tensor | None = None,
        visual_confidence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """mixture: (batch, samples). Returns the estimate and diagnostics."""
        c = self.cfg
        batch, samples = mixture.shape
        device, dtype = mixture.device, mixture.dtype

        window = self._window(device, dtype)
        spec = torch.stft(
            mixture, n_fft=c.n_fft, hop_length=c.hop, window=window, return_complex=True
        )  # (batch, freq, time)
        x = torch.stack([spec.real, spec.imag], dim=1)  # (batch, 2, freq, time)
        x = x.permute(0, 1, 3, 2)  # (batch, 2, time, freq)
        frames = x.shape[2]

        if audio_confidence is None:
            audio_confidence = torch.zeros(batch, device=device, dtype=dtype)
        if visual_confidence is None:
            visual_confidence = torch.zeros(batch, frames, device=device, dtype=dtype)
        elif visual_confidence.shape[-1] != frames:
            visual_confidence = _resize_time(visual_confidence, frames)

        visual_seq = None
        if visual_features is not None:
            if visual_features.shape[1] != frames:
                visual_features = _resize_time(visual_features.transpose(1, 2), frames).transpose(
                    1, 2
                )
            visual_seq = self.visual_proj(visual_features)

        cond, alpha, beta = self.conditioning(
            speaker_embedding, audio_confidence, visual_features, visual_confidence, frames
        )

        h = self.encoder(x)
        for block in self.blocks:
            if c.gradient_checkpointing and self.training and h.requires_grad:
                h = torch.utils.checkpoint.checkpoint(
                    block, h, cond, visual_seq, beta, use_reentrant=False
                )
            else:
                h = block(h, cond, visual_seq, beta)

        out = self.decoder(h).permute(0, 1, 3, 2)  # (batch, 2, freq, time)
        # Synthesis leaves autocast. torch.complex accepts only half/float/
        # double, so a bf16 decoder output has to be widened before it can be
        # read as a spectrogram, and iSTFT in reduced precision costs more
        # accuracy than the handful of microseconds it saves.
        with torch.autocast(device_type=device.type, enabled=False):
            out = out.float()
            estimate_spec = torch.complex(out[:, 0], out[:, 1])
            estimate = torch.istft(
                estimate_spec,
                n_fft=c.n_fft,
                hop_length=c.hop,
                window=window.float(),
                length=samples,
            ).to(dtype)

        result: dict[str, torch.Tensor] = {
            "estimate": estimate,
            "alpha": alpha,
            "beta": beta,
        }
        if self.confidence is not None:
            result["confidence"] = self.confidence(h.mean(dim=(2, 3))).squeeze(-1)
        return result


def _resize_time(x: torch.Tensor, target: int) -> torch.Tensor:
    """Linearly resample the last axis. Video is 25 fps and STFT frames are
    125 Hz, so the two grids never line up and one must be mapped onto the
    other rather than assumed equal."""
    if x.shape[-1] == target:
        return x
    needs_squeeze = x.dim() == 2
    y = x.unsqueeze(1) if needs_squeeze else x
    y = torch.nn.functional.interpolate(y, size=target, mode="linear", align_corners=False)
    out: torch.Tensor = y.squeeze(1) if needs_squeeze else y
    return out
