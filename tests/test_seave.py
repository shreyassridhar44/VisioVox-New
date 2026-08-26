"""SEAVE extractor tests.

Kept small and CPU-runnable so they gate every commit rather than only running
when a GPU is free. The memory behaviour that actually constrains the design is
measured separately in docs/27 §11; what is checked here is that the wiring is
right — shapes, rate alignment, gradient flow, and that the model degrades to
each single modality instead of requiring both.
"""

from __future__ import annotations

from typing import cast

import pytest
import torch

from models.conditioning import FiLM
from models.seave import Seave, SeaveConfig

# Small enough to run on CPU in seconds. The architecture is what is under
# test, not the capacity.
TINY = SeaveConfig(
    n_fft=128,
    hop=32,
    n_blocks=1,
    emb_dim=8,
    lstm_hidden=8,
    attn_heads=2,
    cond_dim=16,
    speaker_emb_dim=16,
    visual_dim=24,
)
BATCH, SAMPLES, VFRAMES = 2, 4096, 25


def _model() -> Seave:
    torch.manual_seed(0)
    return Seave(TINY)


def _cues(
    batch: int = BATCH, vframes: int = VFRAMES
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.randn(batch, TINY.speaker_emb_dim),
        torch.rand(batch),
        torch.randn(batch, vframes, TINY.visual_dim),
        torch.rand(batch, vframes),
    )


# --------------------------------------------------------------------------
# shapes and rates
# --------------------------------------------------------------------------


def test_output_length_matches_input() -> None:
    """STFT then iSTFT must round-trip to the same sample count, or every
    downstream length assumption breaks."""
    m = _model()
    emb, ac, vis, vc = _cues()
    out = m(torch.randn(BATCH, SAMPLES), emb, ac, vis, vc)
    assert out["estimate"].shape == (BATCH, SAMPLES)


@pytest.mark.parametrize("samples", [2048, 4096, 6000])
def test_various_input_lengths(samples: int) -> None:
    m = _model()
    emb, ac, vis, vc = _cues()
    out = m(torch.randn(BATCH, samples), emb, ac, vis, vc)
    assert out["estimate"].shape == (BATCH, samples)


def test_video_and_stft_rates_are_reconciled() -> None:
    """Video is 25 fps and STFT frames are far denser; the two grids never line
    up, so one has to be mapped onto the other rather than assumed equal."""
    m = _model()
    emb, ac, _, _ = _cues()
    for vframes in (10, 25, 100):
        vis = torch.randn(BATCH, vframes, TINY.visual_dim)
        vc = torch.rand(BATCH, vframes)
        out = m(torch.randn(BATCH, SAMPLES), emb, ac, vis, vc)
        assert out["estimate"].shape == (BATCH, SAMPLES)
        assert torch.isfinite(out["estimate"]).all()


def test_beta_is_reported_per_stft_frame() -> None:
    m = _model()
    emb, ac, vis, vc = _cues()
    out = m(torch.randn(BATCH, SAMPLES), emb, ac, vis, vc)
    expected_frames = SAMPLES // TINY.hop + 1
    assert out["beta"].shape == (BATCH, expected_frames)
    assert out["alpha"].shape == (BATCH, 1)


# --------------------------------------------------------------------------
# modality handling
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("audio", "visual"), [(True, True), (True, False), (False, True), (False, False)]
)
def test_runs_with_any_combination_of_cues(audio: bool, visual: bool) -> None:
    """Including neither: an unconditioned pass must still produce audio rather
    than raise, because that is what happens when enrolment fails entirely."""
    m = _model()
    emb, ac, vis, vc = _cues()
    out = m(
        torch.randn(BATCH, SAMPLES),
        emb if audio else None,
        ac if audio else None,
        vis if visual else None,
        vc if visual else None,
    )
    assert out["estimate"].shape == (BATCH, SAMPLES)
    assert torch.isfinite(out["estimate"]).all()


def test_conditioning_path_is_wired_and_will_learn() -> None:
    """FiLM initialises to identity, so at initialisation the speaker cue has
    exactly zero effect — deliberate, since an untrained conditioning path
    would otherwise corrupt the separator's features.

    That makes "does the cue change the output" the wrong question at init. The
    two that matter are whether FiLM receives gradient (so the path can ever
    become active) and whether the cue reaches the output once it is.
    """
    m = _model().train()
    mix = torch.randn(BATCH, SAMPLES)
    e1 = torch.randn(BATCH, TINY.speaker_emb_dim)
    _, _, vis, vc = _cues()

    m(mix, e1, torch.ones(BATCH), vis, vc)["estimate"].pow(2).mean().backward()
    film = cast(FiLM, m.blocks[0].film)
    grad = film.to_scale.weight.grad
    assert grad is not None and float(grad.norm()) > 0, "FiLM cannot learn; cue is dead"

    with torch.no_grad():
        film.to_scale.weight.normal_(0, 0.1)
    m.eval()
    e2 = torch.randn(BATCH, TINY.speaker_emb_dim)
    with torch.no_grad():
        a = m(mix, e1, torch.ones(BATCH), vis, vc)["estimate"]
        b = m(mix, e2, torch.ones(BATCH), vis, vc)["estimate"]
    assert not torch.allclose(a, b, atol=1e-6), "cue does not reach the output"


def test_cue_has_no_effect_at_initialisation() -> None:
    """Pinned deliberately: an identity-initialised FiLM is what keeps an
    untrained conditioning path from destroying the separator's features, and
    losing that property silently would be a regression."""
    m = _model().eval()
    mix = torch.randn(BATCH, SAMPLES)
    _, _, vis, vc = _cues()
    with torch.no_grad():
        a = m(mix, torch.randn(BATCH, TINY.speaker_emb_dim), torch.ones(BATCH), vis, vc)
        b = m(mix, torch.randn(BATCH, TINY.speaker_emb_dim), torch.ones(BATCH), vis, vc)
    assert torch.allclose(a["estimate"], b["estimate"], atol=1e-7)


# --------------------------------------------------------------------------
# confidence head
# --------------------------------------------------------------------------


def test_confidence_is_a_probability() -> None:
    m = _model()
    emb, ac, vis, vc = _cues()
    out = m(torch.randn(BATCH, SAMPLES), emb, ac, vis, vc)
    conf = out["confidence"]
    assert conf.shape == (BATCH,)
    assert (conf >= 0).all() and (conf <= 1).all()


def test_confidence_head_can_be_disabled() -> None:
    cfg = SeaveConfig(**{**TINY.__dict__, "confidence_head": False})
    m = Seave(cfg)
    emb, ac, vis, vc = _cues()
    out = m(torch.randn(BATCH, SAMPLES), emb, ac, vis, vc)
    assert "confidence" not in out


# --------------------------------------------------------------------------
# training mechanics
# --------------------------------------------------------------------------


def test_gradients_reach_every_parameter() -> None:
    """A parameter with no gradient is dead weight that will never train, and
    the symptom is a quietly worse model rather than an error."""
    m = _model().train()
    emb, ac, vis, vc = _cues()
    out = m(torch.randn(BATCH, SAMPLES), emb, ac, vis, vc)
    out["estimate"].pow(2).mean().backward()

    # The confidence head is excluded on purpose: it predicts output quality
    # and is supervised by its own term, not by reconstruction error. Expecting
    # a reconstruction loss to reach it would be asserting the wrong design.
    missing = [
        n
        for n, p in m.named_parameters()
        if p.requires_grad and p.grad is None and not n.startswith("confidence.")
    ]
    assert not missing, f"no gradient reached: {missing[:6]}"


def test_confidence_head_trains_from_its_own_signal() -> None:
    m = _model().train()
    emb, ac, vis, vc = _cues()
    out = m(torch.randn(BATCH, SAMPLES), emb, ac, vis, vc)
    out["confidence"].mean().backward()
    grads = [p.grad for n, p in m.named_parameters() if n.startswith("confidence.")]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_gradients_are_finite() -> None:
    m = _model().train()
    emb, ac, vis, vc = _cues()
    out = m(torch.randn(BATCH, SAMPLES), emb, ac, vis, vc)
    out["estimate"].pow(2).mean().backward()
    bad = [
        n
        for n, p in m.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    assert not bad, f"non-finite gradients in: {bad[:6]}"


def test_checkpointing_does_not_change_the_forward_result() -> None:
    """Checkpointing recomputes activations; it must not alter the output."""
    torch.manual_seed(3)
    on = Seave(SeaveConfig(**{**TINY.__dict__, "gradient_checkpointing": True})).eval()
    torch.manual_seed(3)
    off = Seave(SeaveConfig(**{**TINY.__dict__, "gradient_checkpointing": False})).eval()

    mix = torch.randn(BATCH, SAMPLES)
    emb, ac, vis, vc = _cues()
    with torch.no_grad():
        assert torch.allclose(
            on(mix, emb, ac, vis, vc)["estimate"],
            off(mix, emb, ac, vis, vc)["estimate"],
            atol=1e-5,
        )


def test_eval_mode_is_deterministic() -> None:
    m = _model().eval()
    mix = torch.randn(BATCH, SAMPLES)
    emb, ac, vis, vc = _cues()
    with torch.no_grad():
        a = m(mix, emb, ac, vis, vc)["estimate"]
        b = m(mix, emb, ac, vis, vc)["estimate"]
    assert torch.allclose(a, b)


def test_batch_items_are_independent() -> None:
    """A leak across the batch dimension inflates training metrics and cannot
    survive single-item inference."""
    m = _model().eval()
    emb, ac, vis, vc = _cues(batch=2)
    mix = torch.randn(2, SAMPLES)
    with torch.no_grad():
        both = m(mix, emb, ac, vis, vc)["estimate"]
        first = m(mix[:1], emb[:1], ac[:1], vis[:1], vc[:1])["estimate"]
    assert torch.allclose(both[:1], first, atol=1e-4)
