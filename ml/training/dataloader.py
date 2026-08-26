"""Torch dataloading for audio-visual mixtures (Phase 3 exit criterion).

docs/25 §5 targets >90% GPU utilisation, and on this hardware the dataloader is
what decides whether that happens. An A5000 consumes batches faster than one
Python process can build them, so the defaults here matter more than they look:
persistent workers (respawning per epoch costs seconds of idle GPU), deep
prefetching, and pinned memory for a non-blocking host-to-device copy.

The mixing itself stays in `voxceleb_mix`, which has no torch dependency, so
the simulation logic stays testable without a GPU. This module only adapts it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .voxceleb_mix import MixConfig, MixSample, VoxCelebMixDataset


@dataclass(frozen=True)
class LoaderConfig:
    batch_size: int = 8
    num_workers: int = 6
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last: bool = True


class TorchMixDataset(Dataset[dict[str, torch.Tensor]]):
    """Thin adapter over VoxCelebMixDataset.

    Sampling is keyed on the item index rather than a shared generator, so
    workers do not hand back identical mixtures — a silent way to lose most of
    the effective data variety while the loader still looks busy.
    """

    def __init__(self, inner: VoxCelebMixDataset) -> None:
        self.inner = inner

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        s: MixSample = self.inner.sample(index)
        return {
            "mixture": torch.from_numpy(s.mixture),
            "target": torch.from_numpy(s.target),
            "interferer": torch.from_numpy(s.interferer),
            # (frames, H, W) -> (1, frames, H, W), scaled to 0..1 here so the
            # model never has to remember the input was uint8.
            "mouth": torch.from_numpy(s.mouth.astype(np.float32) / 255.0).unsqueeze(0),
            "active": torch.from_numpy(s.active.astype(np.float32)),
        }


def collate(batch: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack fixed-length items.

    Chunks are fixed-length by construction, so a mismatch means the simulator
    produced something inconsistent. Failing here with the shapes named is far
    cheaper to debug than a broadcast error deep inside a model.
    """
    out: dict[str, torch.Tensor] = {}
    for k in batch[0]:
        shapes = {tuple(item[k].shape) for item in batch}
        if len(shapes) != 1:
            raise ValueError(f"inconsistent shapes for {k!r} in batch: {sorted(shapes)}")
        out[k] = torch.stack([item[k] for item in batch])
    return out


def build_loader(
    packed_root: str | Path,
    mix: MixConfig | None = None,
    loader: LoaderConfig | None = None,
    speakers: list[str] | None = None,
    length: int | None = None,
) -> DataLoader[dict[str, torch.Tensor]]:
    cfg = loader or LoaderConfig()
    inner = VoxCelebMixDataset(Path(packed_root), mix, speakers=speakers, length=length)

    # persistent_workers is invalid with num_workers=0, and prefetch_factor
    # must be omitted entirely rather than passed as None.
    kwargs: dict[str, object] = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
        "drop_last": cfg.drop_last,
        "collate_fn": collate,
        "shuffle": True,
    }
    if cfg.num_workers > 0:
        kwargs["prefetch_factor"] = cfg.prefetch_factor
        kwargs["persistent_workers"] = cfg.persistent_workers

    return DataLoader(TorchMixDataset(inner), **kwargs)  # type: ignore[arg-type]
