"""C4 dataset: real meetings (Phase 4e, ADR-0015).

Everything before this has been synthetic in some way. Libri2Mix sums two
audiobook recordings; C3 convolves them with simulated rooms. AMI is neither —
it is four people in a real room, recorded on real microphones, interrupting
each other. The per-participant headset channels are what make it usable as
supervision: the mixture decomposes into exactly those components, so SI-SDR
and SIR mean the same thing here as they did on Libri2Mix.

Two things differ from the earlier stages and both are deliberate.

**Enrolment comes from the same recording.** Each participant appears in one
session — that is what makes the split speaker-disjoint — so there is no second
recording to enrol from. The cue therefore comes from a different *time window*
of that participant's own microphone, and this class guarantees the enrolment
window never overlaps the chunk being separated. That is not a compromise: it
is Novelty 1, self-enrolment, which is how the product behaves when a user
uploads a video and provides no reference at all.

**The interferer is the sum of the other three.** Libri2Mix has one competing
speaker; a meeting has several, plus whatever the room contributes. Summing
them keeps the suppression term meaningful without pretending the problem is
two-speaker.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from training.librimix_data import FRAME_SAMPLES, RATE, MixItem


@dataclass(frozen=True)
class AmiConfig:
    chunk_seconds: float = 4.0
    seed: int | None = None


@dataclass(frozen=True)
class _Target:
    meeting: str
    index: int
    clip_dir: Path
    n_participants: int


class AmiMixDataset:
    """One item per (meeting, participant): that participant is the target."""

    def __init__(
        self,
        split_dir: Path,
        index_json: Path,
        enrolment_npz: Path,
        config: AmiConfig | None = None,
    ) -> None:
        self.cfg = config or AmiConfig()
        meetings = json.loads(index_json.read_text())

        data = np.load(enrolment_npz, allow_pickle=False)
        self.embeddings = data["embeddings"].astype(np.float32)
        self.n_windows = int(data["n_windows"])
        ids = [str(x) for x in data["ids"]]
        wins = [int(x) for x in data["windows"]]
        # key -> {window: row}, so a window can be chosen and looked up directly.
        self.by_key: dict[str, dict[int, int]] = {}
        for row, (key, w) in enumerate(zip(ids, wins, strict=True)):
            self.by_key.setdefault(key, {})[w] = row

        self.items: list[_Target] = []
        for meeting in meetings:
            clip_dir = split_dir / meeting["meeting"]
            n = len(meeting["participants"])
            for participant in meeting["participants"]:
                key = f"{meeting['meeting']}|{participant['index']}"
                # A participant needs at least two usable windows: one to be
                # separated and a different one to enrol from. With fewer, the
                # cue would have to come from the chunk under test, which leaks
                # the answer and inflates the score.
                if len(self.by_key.get(key, {})) >= 2:
                    self.items.append(
                        _Target(meeting["meeting"], participant["index"], clip_dir, n)
                    )

        if not self.items:
            raise ValueError(f"no usable AMI targets under {split_dir}")

    def __len__(self) -> int:
        return len(self.items)

    @property
    def skipped_single_clip_speakers(self) -> int:
        return 0

    def sample(self, index: int) -> MixItem:
        rng = random.Random((self.cfg.seed or 0) + index)
        target_spec = self.items[index % len(self.items)]
        key = f"{target_spec.meeting}|{target_spec.index}"

        want = int(self.cfg.chunk_seconds * RATE)
        target_full, _ = sf.read(
            str(target_spec.clip_dir / f"ref_spk{target_spec.index}.wav"), dtype="float32"
        )
        total = len(target_full)
        span = total // self.n_windows

        # Pick the enrolment window first, then take the chunk from anywhere
        # outside it. Doing it this way makes the disjointness a property of
        # the construction rather than something to be re-checked.
        available = sorted(self.by_key[key])
        enrol_window = rng.choice(available)
        forbidden = (enrol_window * span, (enrol_window + 1) * span)

        for _ in range(16):
            start = rng.randint(0, max(0, total - want))
            if start + want <= forbidden[0] or start >= forbidden[1]:
                break
        else:
            # Every draw landed in the enrolment window. Take the largest
            # region outside it rather than silently overlapping.
            start = 0 if forbidden[0] >= want else min(forbidden[1], max(0, total - want))

        stop = start + want

        def read(path: Path) -> np.ndarray:
            audio, _ = sf.read(str(path), dtype="float32")
            piece = audio[start:stop]
            if len(piece) < want:
                piece = np.pad(piece, (0, want - len(piece)))
            out: np.ndarray = piece.astype(np.float32)
            return out

        target = read(target_spec.clip_dir / f"ref_spk{target_spec.index}.wav")
        others = [
            read(target_spec.clip_dir / f"ref_spk{i}.wav")
            for i in range(target_spec.n_participants)
            if i != target_spec.index
        ]
        interferer = (
            np.sum(others, axis=0).astype(np.float32)
            if others
            else np.zeros(want, dtype=np.float32)
        )
        mixture = read(target_spec.clip_dir / "mixture.wav")

        enrolment = self.embeddings[self.by_key[key][enrol_window]]

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
            target_speaker=key,
            mixture_id=f"{target_spec.meeting}@{start}",
        )

    def __getitem__(self, index: int) -> MixItem:
        return self.sample(index)


__all__ = ["AmiConfig", "AmiMixDataset"]
