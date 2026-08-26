"""Libri2Mix dataset for target-speaker extraction (Phase 4b, C1).

Emits the same batch shape as `voxceleb_mix` so the trainer does not care which
corpus it is fed — C1 is audio-only, C2 adds the visual stream, and the loop
stays the same.

**The enrolment cue never comes from the clip being separated.** Sampling it
from the same utterance leaks the answer: the model learns to match acoustic
detail rather than speaker identity, scores far higher than it should, and
collapses on real input where no matched enrolment exists. A speaker with only
one clip in the split therefore cannot be a target at all, and is skipped rather
than quietly self-enrolled.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

RATE = 16_000
FRAME_SAMPLES = 640  # 25 fps grid, matching the audio-visual path


@dataclass(frozen=True)
class LibriMixConfig:
    chunk_seconds: float = 4.0
    seed: int | None = None
    # Where the mixture comes from. mix_both includes WHAM! noise; mix_clean
    # is speech only. C1 trains on mix_both because real uploads are noisy.
    mixture_type: str = "mix_both"


@dataclass
class MixItem:
    mixture: np.ndarray
    target: np.ndarray
    interferer: np.ndarray
    active: np.ndarray
    enrolment: np.ndarray
    target_speaker: str
    mixture_id: str


class Libri2MixDataset:
    """Indexable Libri2Mix TSE dataset.

    No torch dependency, like `voxceleb_mix`, so the sampling logic is testable
    without a GPU. `dataloader.TorchMixDataset` adapts it.
    """

    def __init__(
        self,
        split_dir: Path,
        enrolment_npz: Path,
        config: LibriMixConfig | None = None,
        length: int | None = None,
    ) -> None:
        self.dir = Path(split_dir)
        self.cfg = config or LibriMixConfig()
        self._rng = random.Random(self.cfg.seed)

        data = np.load(enrolment_npz, allow_pickle=False)
        ids = [str(x) for x in data["ids"]]
        speakers = [str(x) for x in data["speakers"]]
        self.embeddings = data["embeddings"].astype(np.float32)

        self.index_of: dict[str, int] = {k: i for i, k in enumerate(ids)}
        self.speaker_of: dict[str, str] = dict(zip(ids, speakers, strict=True))

        by_speaker: dict[str, list[str]] = defaultdict(list)
        for key, spk in zip(ids, speakers, strict=True):
            by_speaker[spk].append(key)
        self.by_speaker = dict(by_speaker)

        # A target needs at least one OTHER clip to enrol from.
        self.items: list[tuple[str, int]] = []
        for key in ids:
            mixture, slot = key.rsplit("|", 1)
            if len(self.by_speaker[self.speaker_of[key]]) >= 2:
                self.items.append((mixture, int(slot)))

        if not self.items:
            raise ValueError(
                f"no usable items in {split_dir}: every speaker has a single clip, "
                "so no enrolment can come from a different utterance"
            )
        self._length = length if length is not None else len(self.items)

    @property
    def skipped_single_clip_speakers(self) -> int:
        return sum(1 for v in self.by_speaker.values() if len(v) < 2)

    def __len__(self) -> int:
        return self._length

    def _chunk(self, arrays: list[np.ndarray], rng: random.Random) -> list[np.ndarray]:
        """Take the same random window from every array, padding a short clip."""
        want = int(self.cfg.chunk_seconds * RATE)
        n = min(len(a) for a in arrays)
        if n <= want:
            return [np.pad(a[:n], (0, want - n)).astype(np.float32) for a in arrays]
        start = rng.randint(0, n - want)
        return [a[start : start + want].astype(np.float32) for a in arrays]

    def _enrolment_for(self, speaker: str, exclude: str, rng: random.Random) -> np.ndarray:
        """An embedding from a different clip by the same speaker."""
        candidates = [k for k in self.by_speaker[speaker] if k != exclude]
        if not candidates:
            # Guarded at construction; reaching here means the index is
            # inconsistent, and self-enrolling would silently inflate results.
            raise ValueError(f"no other clip to enrol speaker {speaker}")
        vec: np.ndarray = self.embeddings[self.index_of[rng.choice(candidates)]]
        return vec

    def sample(self, index: int) -> MixItem:
        rng = random.Random((self.cfg.seed or 0) + index)
        mixture_id, slot = self.items[index % len(self.items)]

        mixture, _ = sf.read(self.dir / self.cfg.mixture_type / mixture_id, dtype="float32")
        target, _ = sf.read(self.dir / f"s{slot}" / mixture_id, dtype="float32")
        other_slot = 2 if slot == 1 else 1
        interferer, _ = sf.read(self.dir / f"s{other_slot}" / mixture_id, dtype="float32")

        mixture, target, interferer = self._chunk([mixture, target, interferer], rng)

        key = f"{mixture_id}|{slot}"
        speaker = self.speaker_of[key]
        enrolment = self._enrolment_for(speaker, key, rng)

        frames = len(target) // FRAME_SAMPLES
        energy = (target[: frames * FRAME_SAMPLES].reshape(frames, FRAME_SAMPLES) ** 2).mean(axis=1)
        peak = float(energy.max()) if frames else 0.0
        active = (energy > peak * 0.05) if peak > 0 else np.zeros(frames, dtype=bool)

        return MixItem(
            mixture=mixture,
            target=target,
            interferer=interferer,
            active=active.astype(np.float32),
            enrolment=enrolment,
            target_speaker=speaker,
            mixture_id=mixture_id,
        )

    def __getitem__(self, index: int) -> MixItem:
        return self.sample(index)


def to_batch_dict(item: MixItem) -> dict[str, np.ndarray]:
    """Shape the trainer expects, matching the VoxCeleb2 path."""
    return {
        "mixture": item.mixture,
        "target": item.target,
        "interferer": item.interferer,
        "active": item.active,
        "speaker_embedding": item.enrolment,
    }
