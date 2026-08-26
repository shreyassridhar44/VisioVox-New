"""Reliability-gated conditioning tests (SEAVE-RG, docs/04 §3).

The claim has three falsifiable parts and each gets a test:

- Per-frame gating keeps the usable part of a partly-bad visual stream, where a
  hard modality switch discards all of it.
- A degraded visual cue is down-weighted rather than trusted, so the model is
  not actively misled by bad video.
- One set of weights serves audio-visual, audio-only and visual-only inference,
  with no dispatch logic.
"""

from __future__ import annotations

import pytest
import torch

from models.conditioning import (
    DropoutConfig,
    FiLM,
    GatedCrossAttention,
    ReliabilityGate,
    ReliabilityGatedConditioning,
    apply_modality_dropout,
)

SPK, VIS, COND = 192, 512, 256
BATCH, FRAMES = 4, 100
torch.manual_seed(0)


def _inputs(
    batch: int = BATCH, frames: int = FRAMES
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.randn(batch, SPK),
        torch.rand(batch),
        torch.randn(batch, frames, VIS),
        torch.rand(batch, frames),
    )


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------


def test_gate_output_is_bounded() -> None:
    gate = ReliabilityGate()
    out = gate(torch.rand(8, 50))
    assert out.shape == (8, 50)
    assert (out >= 0).all() and (out <= 1).all()


def test_gate_starts_open() -> None:
    """A gate initialised closed gives the separator no signal early in
    training, and it learns to ignore the cue before the gate ever opens."""
    gate = ReliabilityGate()
    with torch.no_grad():
        assert float(gate(torch.full((16,), 0.8)).mean()) > 0.4


def test_gate_is_monotonic_after_training_on_a_ramp() -> None:
    """Higher confidence should mean more gate. Not guaranteed at init, so it
    is trained briefly on the relationship it is supposed to learn."""
    gate = ReliabilityGate()
    opt = torch.optim.Adam(gate.parameters(), lr=0.05)
    conf = torch.linspace(0, 1, 64)
    for _ in range(300):
        loss = torch.nn.functional.mse_loss(gate(conf), conf)
        opt.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
    out = gate(torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
    assert torch.all(out[1:] > out[:-1]), f"not monotonic: {out.tolist()}"


# --------------------------------------------------------------------------
# per-frame gating — the claim against a hard switch
# --------------------------------------------------------------------------


def test_per_frame_gating_keeps_the_usable_part_of_a_bad_stream() -> None:
    """A speaker turns away for 2 s of a 4 s clip.

    Per-frame beta should suppress visual conditioning for those frames only.
    A hard modality switch drops the whole stream because part of it was bad,
    which is most of the benefit thrown away.
    """
    model = ReliabilityGatedConditioning(SPK, VIS, COND)
    emb, a_conf, vis, _ = _inputs()

    v_conf = torch.full((BATCH, FRAMES), 0.9)
    v_conf[:, 25:75] = 0.0  # turned away for the middle half

    cond, _, beta = model(emb, a_conf, vis, v_conf, FRAMES)

    good = beta[:, :25].mean().detach()
    bad = beta[:, 25:75].mean().detach()
    assert bad < good, f"gate did not close on the bad frames ({bad:.3f} vs {good:.3f})"

    # visual contribution should survive outside the bad window
    audio_only, _, _ = model(emb, a_conf, None, v_conf, FRAMES)
    visible_delta = (cond[:, :, :25] - audio_only[:, :, :25]).abs().mean()
    hidden_delta = (cond[:, :, 25:75] - audio_only[:, :, 25:75]).abs().mean()
    assert visible_delta > hidden_delta, "visual contribution was not preserved where it was good"


def test_beta_is_per_frame_not_per_utterance() -> None:
    model = ReliabilityGatedConditioning(SPK, VIS, COND)
    emb, a_conf, vis, _ = _inputs()
    v_conf = torch.rand(BATCH, FRAMES)
    _, alpha, beta = model(emb, a_conf, vis, v_conf, FRAMES)
    assert alpha.shape == (BATCH, 1), "alpha should be per utterance"
    assert beta.shape == (BATCH, FRAMES), "beta must be per frame"
    assert float(beta.std().detach()) > 1e-4, "beta is constant across frames"


def test_zero_visual_confidence_removes_the_visual_contribution() -> None:
    """Not misled by video the model has been told is worthless."""
    model = ReliabilityGatedConditioning(SPK, VIS, COND)
    emb, a_conf, vis, _ = _inputs()

    with torch.no_grad():
        # force the gate closed so the test measures the wiring, not the init
        model.visual_gate.output.bias.fill_(-20.0)
    cond, _, beta = model(emb, a_conf, vis, torch.zeros(BATCH, FRAMES), FRAMES)
    audio_only, _, _ = model(emb, a_conf, None, torch.zeros(BATCH, FRAMES), FRAMES)

    assert float(beta.max().detach()) < 0.01
    assert torch.allclose(cond, audio_only, atol=1e-4)


# --------------------------------------------------------------------------
# one model, three modes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("has_audio", "has_visual"), [(True, True), (True, False), (False, True)])
def test_same_weights_serve_every_modality(has_audio: bool, has_visual: bool) -> None:
    """No model zoo, no dispatch logic — the point of continuous gating."""
    model = ReliabilityGatedConditioning(SPK, VIS, COND)
    emb, a_conf, vis, v_conf = _inputs()

    cond, alpha, beta = model(
        emb if has_audio else None,
        a_conf if has_audio else torch.zeros(BATCH),
        vis if has_visual else None,
        v_conf if has_visual else torch.zeros(BATCH, FRAMES),
        FRAMES,
    )
    assert cond.shape == (BATCH, COND, FRAMES)
    assert torch.isfinite(cond).all()
    assert alpha.shape == (BATCH, 1) and beta.shape == (BATCH, FRAMES)


def test_conditioning_is_differentiable() -> None:
    model = ReliabilityGatedConditioning(SPK, VIS, COND)
    emb, a_conf, vis, v_conf = _inputs()
    emb.requires_grad_(True)
    cond, _, _ = model(emb, a_conf, vis, v_conf, FRAMES)
    cond.sum().backward()
    assert emb.grad is not None and torch.isfinite(emb.grad).all()


# --------------------------------------------------------------------------
# FiLM and cross-attention
# --------------------------------------------------------------------------


def test_film_is_identity_at_initialisation() -> None:
    """An untrained conditioning path must not destroy the separator's features."""
    film = FiLM(COND, 128)
    features = torch.randn(BATCH, 128, FRAMES)
    cond = torch.randn(BATCH, COND, FRAMES)
    assert torch.allclose(film(features, cond), features, atol=1e-6)


def test_film_modulates_once_trained() -> None:
    film = FiLM(COND, 128)
    with torch.no_grad():
        film.to_shift.weight.normal_(0, 0.1)
    features = torch.randn(BATCH, 128, FRAMES)
    cond = torch.randn(BATCH, COND, FRAMES)
    assert not torch.allclose(film(features, cond), features, atol=1e-4)


def test_cross_attention_gate_removes_the_visual_contribution() -> None:
    """The gate multiplies the attention output, not the scores: scaling scores
    only reshapes what is attended to, it does not remove the contribution."""
    attn = GatedCrossAttention(64, heads=4)
    audio = torch.randn(BATCH, FRAMES, 64)
    visual = torch.randn(BATCH, 50, 64)

    off = attn(audio, visual, torch.zeros(BATCH, FRAMES))
    assert torch.allclose(off, audio, atol=1e-6)

    on = attn(audio, visual, torch.ones(BATCH, FRAMES))
    assert not torch.allclose(on, audio, atol=1e-4)


def test_cross_attention_gate_applies_per_frame() -> None:
    attn = GatedCrossAttention(64, heads=4)
    audio = torch.randn(BATCH, FRAMES, 64)
    visual = torch.randn(BATCH, 50, 64)
    beta = torch.zeros(BATCH, FRAMES)
    beta[:, :50] = 1.0

    out = attn(audio, visual, beta)
    assert not torch.allclose(out[:, :50], audio[:, :50], atol=1e-4)
    assert torch.allclose(out[:, 50:], audio[:, 50:], atol=1e-6)


# --------------------------------------------------------------------------
# modality dropout — what teaches the gates to distrust
# --------------------------------------------------------------------------


def test_dropout_zeroes_some_visual_streams() -> None:
    cfg = DropoutConfig(
        drop_visual=1.0, drop_audio_cue=0.0, corrupt_visual=0.0, corrupt_audio_cue=0.0
    )
    emb, a_conf, vis, v_conf = _inputs()
    _, _, vis2, v_conf2 = apply_modality_dropout(emb, a_conf, vis, v_conf, cfg)
    assert vis2 is not None
    assert float(vis2.abs().sum()) == 0.0
    assert float(v_conf2.abs().sum()) == 0.0


def test_dropout_lowers_confidence_alongside_corruption() -> None:
    """The pairing is the training signal. Corrupting features while leaving
    confidence high would teach the gate exactly the wrong association."""
    cfg = DropoutConfig(
        drop_visual=0.0, drop_audio_cue=0.0, corrupt_visual=1.0, corrupt_audio_cue=1.0
    )
    emb, a_conf, vis, v_conf = _inputs()
    emb2, a_conf2, vis2, v_conf2 = apply_modality_dropout(emb, a_conf, vis, v_conf, cfg)

    assert float(a_conf2.mean()) < float(a_conf.mean())
    assert float(v_conf2.mean()) < float(v_conf.mean())
    assert vis2 is not None and not torch.allclose(vis2, vis)
    assert emb2 is not None and not torch.allclose(emb2, emb)


def test_dropout_never_removes_both_cues() -> None:
    """An item with no conditioning at all teaches only that conditioning can
    be ignored."""
    cfg = DropoutConfig(drop_visual=1.0, drop_audio_cue=1.0)
    emb, a_conf, vis, v_conf = _inputs(batch=32)
    _, a_conf2, _, v_conf2 = apply_modality_dropout(emb, a_conf, vis, v_conf, cfg)
    # alpha is per utterance, beta per frame: reduce the visual side first
    both_gone = (a_conf2 == 0) & (v_conf2.sum(dim=1) == 0)
    assert not bool(both_gone.any()), "some items lost every cue"


def test_dropout_leaves_shapes_intact() -> None:
    emb, a_conf, vis, v_conf = _inputs()
    emb2, a_conf2, vis2, v_conf2 = apply_modality_dropout(emb, a_conf, vis, v_conf)
    assert emb2 is not None and emb2.shape == emb.shape
    assert vis2 is not None and vis2.shape == vis.shape
    assert a_conf2.shape == a_conf.shape and v_conf2.shape == v_conf.shape


def test_dropout_does_not_mutate_its_inputs() -> None:
    """Training loops reuse batches; in-place corruption would compound."""
    emb, a_conf, vis, v_conf = _inputs()
    before = emb.clone(), a_conf.clone(), vis.clone(), v_conf.clone()
    apply_modality_dropout(emb, a_conf, vis, v_conf, DropoutConfig(drop_visual=1.0))
    assert torch.equal(emb, before[0])
    assert torch.equal(a_conf, before[1])
    assert torch.equal(vis, before[2])
    assert torch.equal(v_conf, before[3])
