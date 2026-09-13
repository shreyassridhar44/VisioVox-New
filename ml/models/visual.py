"""Visual frontend: mouth ROIs to per-frame features (docs/05 §8.2).

The standard AVSR frontend, and standard for a reason. A 3D convolution over
the first few frames captures the short-horizon motion that distinguishes /b/
from /m/ — information that vanishes if each frame is encoded independently,
because the two look identical at any single instant. A 2D ResNet trunk then
encodes appearance per frame, and the sequence goes to the fusion path at
25 fps for upsampling to the STFT grid.

Written directly rather than pulled from torchvision. The trunk is sixty lines,
and the dependency cost is real: installing torchvision here upgraded torch and
broke its ABI. ImageNet weights would also be a poor fit — these are 96x96
grayscale lip crops, not natural colour images, and the first convolution would
have to be re-learned regardless.

The frontend is trainable but starts frozen (docs/05 §8.2: "pretrained, frozen
early training"). With no pretrained weights available the freeze is inverted:
the audio path already works and the visual path is random, so releasing it
immediately lets random visual gradients disturb a trained extractor. C2 warms
up with the frontend frozen only in the sense that its contribution is gated —
`beta_t` in the reliability-gated fusion starts near zero and the model learns
how much to trust it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class VisualConfig:
    """Defaults match the ROI format `prepare_voxceleb2.py` writes."""

    roi: int = 96  # 96x96 grayscale, landmark-aligned
    fps: int = 25
    out_dim: int = 512  # SeaveConfig.visual_dim
    stem_channels: int = 64
    #: Temporal extent of the 3D stem, in frames. Five frames is 200 ms, which
    #: covers a syllable — long enough for articulation, short enough that the
    #: output stays at frame rate.
    stem_frames: int = 5
    dropout: float = 0.1


class BasicBlock(nn.Module):
    """ResNet-18 residual block: two 3x3 convolutions and a shortcut."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        # Projection only when the shapes actually change, so the identity path
        # stays parameter-free wherever it can.
        self.short: nn.Module = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        activated: torch.Tensor = self.relu(out + self.short(x))
        return activated


class VisualFrontend(nn.Module):
    """(batch, 1, frames, H, W) uint8-scaled input -> (batch, frames, out_dim)."""

    def __init__(self, cfg: VisualConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or VisualConfig()
        c = self.cfg

        # 3D stem. Stride 1 in time and padding that preserves the frame count:
        # the output must stay aligned with audio frames, because the fusion
        # path maps frame index to sample index exactly (640 samples at 25 fps).
        pad_t = c.stem_frames // 2
        self.stem = nn.Sequential(
            nn.Conv3d(
                1,
                c.stem_channels,
                kernel_size=(c.stem_frames, 7, 7),
                stride=(1, 2, 2),
                padding=(pad_t, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(c.stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

        # ResNet-18 trunk, applied per frame.
        widths = (64, 128, 256, 512)
        strides = (1, 2, 2, 2)
        layers: list[nn.Module] = []
        in_ch = c.stem_channels
        for width, stride in zip(widths, strides, strict=True):
            layers.append(BasicBlock(in_ch, width, stride))
            layers.append(BasicBlock(width, width, 1))
            in_ch = width
        self.trunk = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(c.dropout)
        self.project = nn.Linear(widths[-1], c.out_dim)

    def forward(self, mouth: torch.Tensor) -> torch.Tensor:
        """mouth: (batch, 1, frames, H, W) in 0..1. Returns (batch, frames, out_dim)."""
        if mouth.dim() != 5:
            raise ValueError(f"expected (batch, 1, frames, H, W), got {tuple(mouth.shape)}")

        batch, _, frames = mouth.shape[:3]
        x = self.stem(mouth)  # (batch, C, frames, H', W')

        # Fold time into the batch so the 2D trunk sees one image per frame,
        # then unfold. Cheaper and simpler than a 3D trunk, and the temporal
        # modelling that matters already happened in the stem and happens again
        # in the backbone's inter-frame BLSTM.
        channels, height, width = x.shape[1], x.shape[3], x.shape[4]
        x = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        x = self.trunk(x)
        x = self.pool(x).flatten(1)
        x = self.project(self.dropout(x))
        out: torch.Tensor = x.reshape(batch, frames, self.cfg.out_dim)
        return out


def resample_to(features: torch.Tensor, frames: int) -> torch.Tensor:
    """Linearly resample a (batch, time, dim) sequence onto `frames` steps.

    Video is 25 fps and the STFT grid is 125 Hz, so the two never line up and
    one has to be mapped onto the other. Done here rather than in the model so
    the frontend's output stays at its native rate and the mapping is visible.
    """
    if features.shape[1] == frames:
        return features
    resampled: torch.Tensor = torch.nn.functional.interpolate(
        features.transpose(1, 2), size=frames, mode="linear", align_corners=False
    ).transpose(1, 2)
    return resampled


__all__ = ["BasicBlock", "VisualConfig", "VisualFrontend", "resample_to"]
