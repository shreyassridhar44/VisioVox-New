"""C3 dataset: realistic mixtures built on the fly (docs/06 §5, docs/07 §1).

C1 trained on Libri2Mix, where two clean recordings are summed with a fixed
offset and WHAM! noise added. Nothing in it has a room. Real uploads are
recorded in rooms, and reverberation is the degradation the Phase 1 baseline
report singled out as dominant — so a model that has never heard a reflection
is being asked to generalise across the one axis it has no experience of.

This wraps `training.simulate`, which does the physics: image-source RIRs from
sampled room dimensions and RT60, turn-taking rather than uniform overlap,
per-speaker level spread, additive noise at a sampled SNR, and an optional
lossy codec round-trip.

Two properties matter for the objective to stay meaningful:

- **The references are post-RIR.** `simulate` returns each source after its own
  room response and gain but before summing, so the mixture really is the sum
  of the references. Using the dry source as the target instead would ask the
  model to dereverberate and separate at once, and score it as if it had failed
  at separation when it merely left reverb in place. S1 is where dereverb
  belongs, not here.
- **Nothing is written to disk.** Mixtures are generated per sample, so the
  same clean clips give unlimited variety and the corpus stays the size of its
  sources rather than growing into terabytes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from training.librimix_data import FRAME_SAMPLES, RATE, Libri2MixDataset, MixItem
from training.simulate import SimConfig, simulate


@dataclass(frozen=True)
class RealisticConfig:
    chunk_seconds: float = 4.0
    seed: int | None = None
    sim: SimConfig = field(default_factory=SimConfig)
    #: Probability a sample is passed through unsimulated. Keeps some clean
    #: material in the mix so C3 does not forget what C1 learned — a model
    #: fine-tuned only on reverberant audio degrades on dry audio, and plenty
    #: of real uploads are close-mic'd.
    dry_fraction: float = 0.15


class RealisticMixDataset:
    """Libri2Mix sources, re-mixed through a simulated room.

    Sources come from an existing `Libri2MixDataset` rather than a new corpus:
    the clean single-speaker recordings are the same ones, and what changes is
    how they are combined. That also means the enrolment index, the leak-free
    sampling and the speaker bookkeeping are inherited rather than reimplemented.
    """

    def __init__(self, base: Libri2MixDataset, config: RealisticConfig | None = None) -> None:
        self.base = base
        self.cfg = config or RealisticConfig()

    def __len__(self) -> int:
        return len(self.base)

    @property
    def skipped_single_clip_speakers(self) -> int:
        return self.base.skipped_single_clip_speakers

    def sample(self, index: int) -> MixItem:
        item = self.base.sample(index)
        rng = random.Random((self.cfg.seed or 0) + index)

        if rng.random() < self.cfg.dry_fraction:
            return item

        # Seeded per index so a given sample is reproducible across runs and
        # across a resume — the room this item gets must not depend on when it
        # happened to be drawn.
        sim_cfg = SimConfig(
            overlap_ratio=self.cfg.sim.overlap_ratio,
            room_dim_low=self.cfg.sim.room_dim_low,
            room_dim_high=self.cfg.sim.room_dim_high,
            rt60=self.cfg.sim.rt60,
            level_spread_db=self.cfg.sim.level_spread_db,
            snr_db=self.cfg.sim.snr_db,
            codec_probability=self.cfg.sim.codec_probability,
            codecs=self.cfg.sim.codecs,
            seed=(self.cfg.seed or 0) + index,
        )
        result = simulate([item.target, item.interferer], cfg=sim_cfg)

        # Source 0 is the target: `simulate` preserves input order.
        target, interferer = result.sources[0], result.sources[1]
        frames = len(target) // FRAME_SAMPLES
        energy = (target[: frames * FRAME_SAMPLES].reshape(frames, FRAME_SAMPLES) ** 2).mean(axis=1)
        peak = float(energy.max()) if frames else 0.0
        # Recomputed rather than carried over: the room and the turn schedule
        # move where the target is actually speaking, and a stale mask would
        # tell the silence term to police the wrong samples.
        active = (energy > peak * 0.05) if peak > 0 else np.zeros(frames, dtype=bool)

        return MixItem(
            mixture=result.mixture,
            target=target,
            interferer=interferer,
            active=active.astype(np.float32),
            enrolment=item.enrolment,
            target_speaker=item.target_speaker,
            mixture_id=item.mixture_id,
        )

    def __getitem__(self, index: int) -> MixItem:
        return self.sample(index)


def build(
    split_dir: Path,
    enrolment_npz: Path,
    chunk_seconds: float = 4.0,
    seed: int | None = 0,
    dry_fraction: float = 0.15,
) -> RealisticMixDataset:
    """Convenience constructor mirroring `Libri2MixDataset`'s signature."""
    from training.librimix_data import LibriMixConfig

    base = Libri2MixDataset(
        split_dir, enrolment_npz, LibriMixConfig(chunk_seconds=chunk_seconds, seed=seed)
    )
    return RealisticMixDataset(
        base, RealisticConfig(chunk_seconds=chunk_seconds, seed=seed, dry_fraction=dry_fraction)
    )


__all__ = ["RATE", "RealisticConfig", "RealisticMixDataset", "build"]
