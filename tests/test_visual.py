"""Visual frontend (docs/05 §8.2).

The properties worth pinning are the ones whose violation is silent. A frontend
that quietly drops or adds frames still trains, still produces plausible loss
curves, and misaligns lips from speech by an amount nobody can see in a number.
"""

from __future__ import annotations

import pytest
import torch

from models.visual import VisualConfig, VisualFrontend, resample_to


@pytest.fixture(scope="module")
def frontend() -> VisualFrontend:
    return VisualFrontend(VisualConfig()).eval()


def test_frame_count_is_preserved(frontend: VisualFrontend) -> None:
    """The load-bearing property.

    Frame index maps to sample index exactly — 640 samples per frame at 25 fps
    — so losing or gaining a frame shifts lips against speech for the rest of
    the clip. Temporal stride is 1 and the 3D padding is chosen for this.
    """
    for frames in (25, 50, 150):
        out = frontend(torch.zeros(2, 1, frames, 96, 96))
        assert out.shape == (2, frames, 512), f"{frames} frames in, {out.shape} out"


def test_output_dimension_matches_seave(frontend: VisualFrontend) -> None:
    from models.seave import SeaveConfig

    assert frontend.cfg.out_dim == SeaveConfig().visual_dim


def test_rejects_wrongly_shaped_input(frontend: VisualFrontend) -> None:
    with pytest.raises(ValueError, match="batch, 1, frames"):
        frontend(torch.zeros(2, 150, 96, 96))


def test_output_depends_on_motion_not_only_appearance() -> None:
    """The reason for a 3D stem rather than per-frame encoding.

    Two clips with identical frames in a different order are indistinguishable
    to any per-frame encoder, and lip reading is largely about the order.
    """
    torch.manual_seed(0)
    net = VisualFrontend(VisualConfig()).eval()
    clip = torch.rand(1, 1, 25, 96, 96)
    reversed_clip = clip.flip(dims=[2])

    with torch.no_grad():
        a = net(clip)
        b = net(reversed_clip)
    # Compare as sets over time by sorting, so only ordering differs.
    assert not torch.allclose(a.sort(dim=1).values, b.sort(dim=1).values, atol=1e-4)


def test_gradients_reach_the_stem(frontend: VisualFrontend) -> None:
    net = VisualFrontend(VisualConfig()).train()
    out = net(torch.rand(1, 1, 25, 96, 96))
    out.mean().backward()
    stem_conv = net.stem[0]
    assert isinstance(stem_conv, torch.nn.Conv3d)
    assert stem_conv.weight.grad is not None
    assert float(stem_conv.weight.grad.abs().sum()) > 0


def test_resample_maps_video_rate_onto_stft_frames() -> None:
    """25 fps video against a 125 Hz STFT grid: they never line up."""
    features = torch.randn(2, 25, 512)
    assert resample_to(features, 125).shape == (2, 125, 512)
    assert resample_to(features, 25) is features  # no-op when already aligned


def test_resample_preserves_endpoints() -> None:
    """A shift at the boundary is a systematic lip-sync offset, not noise."""
    ramp = torch.linspace(0, 1, 25).view(1, 25, 1).expand(1, 25, 4).contiguous()
    out = resample_to(ramp, 125)
    assert out[0, 0, 0].item() == pytest.approx(0.0, abs=0.02)
    assert out[0, -1, 0].item() == pytest.approx(1.0, abs=0.02)


def test_parameter_count_is_resnet18_scale(frontend: VisualFrontend) -> None:
    """A trunk far off 11M means the widths or blocks are wrong."""
    total = sum(p.numel() for p in frontend.parameters())
    assert 8e6 < total < 20e6, f"{total / 1e6:.1f}M parameters"
