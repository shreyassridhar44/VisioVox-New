"""VoxCeleb2-Mix: on-the-fly audio-visual mixtures for AV-TSE training.

Mixtures are generated per sample rather than written to disk (docs/07 §3).
That keeps storage at tens of GB instead of terabytes and gives effectively
unlimited mixture variety from the same clips — the same target is seen against
a different interferer, at a different level, on every epoch.

Each item is one *target* speaker plus one or more interferers:

    mixture     (samples,)          what the model hears
    target      (samples,)          what it must recover
    interferer  (samples,)          the sum of the others, for L_suppress
    mouth       (frames, 96, 96)    the target's mouth ROI -- the conditioning
    active      (frames,)           whether the target is speaking

`interferer` is returned deliberately. The suppression-first objective
(docs/04 §4) needs the interference separately, because SI-SDR alone cannot
distinguish leakage from artifact — it sums both into one number, and a model
can score well on it while another speaker stays clearly audible. That
distinction is the entire point of the project, so the data has to carry it.

Speaker-disjointness is the caller's responsibility: pass a speaker list that
does not overlap between train and eval. `VoxCelebMixDataset.speakers` makes
that checkable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RATE = 16_000
FPS = 25
SAMPLES_PER_FRAME = RATE // FPS  # 640 — exact, so frame index maps to sample


@dataclass(frozen=True)
class MixConfig:
    """Simulation parameters.

    Defaults follow docs/06 §5: real conversation is sparsely overlapped and
    level-imbalanced, unlike Libri2Mix which is fully overlapped and balanced.
    Training on balanced full-overlap mixtures is a documented way to build a
    model that collapses on real input.
    """

    chunk_seconds: float = 4.0
    n_interferers: int = 1
    # Target-to-interferer ratio. Real recordings are not level-balanced; one
    # speaker is usually closer to the microphone.
    tir_db: tuple[float, float] = (-5.0, 5.0)
    # Fraction of the chunk where the interferer is present at all.
    overlap_ratio: tuple[float, float] = (0.2, 1.0)
    peak_ceiling: float = 0.99
    seed: int | None = None


@dataclass
class MixSample:
    mixture: np.ndarray
    target: np.ndarray
    interferer: np.ndarray
    mouth: np.ndarray
    active: np.ndarray
    target_speaker: str
    interferer_speakers: tuple[str, ...]


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2) + 1e-12))


def _scale_to_tir(target: np.ndarray, interferer: np.ndarray, tir_db: float) -> np.ndarray:
    """Rescale the interferer so target-to-interferer ratio is `tir_db`."""
    t, i = _rms(target), _rms(interferer)
    if i < 1e-9:
        return interferer
    desired = t / (10 ** (tir_db / 20))
    return interferer * (desired / i)


class VoxCelebMixDataset:
    """Indexable dataset of simulated audio-visual mixtures.

    Deliberately not a torch.utils.data.Dataset subclass: it has no torch
    dependency, so the mixing logic is testable without a GPU or a training
    loop. A thin torch wrapper can adapt it where one is needed.
    """

    def __init__(
        self,
        packed_root: Path,
        config: MixConfig | None = None,
        speakers: list[str] | None = None,
        length: int | None = None,
    ) -> None:
        self.root = Path(packed_root)
        self.config = config or MixConfig()
        self._rng = random.Random(self.config.seed)

        self.by_speaker: dict[str, list[Path]] = {}
        for spk_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            if speakers is not None and spk_dir.name not in speakers:
                continue
            clips = sorted(spk_dir.glob("*.npz"))
            if clips:
                self.by_speaker[spk_dir.name] = clips

        if len(self.by_speaker) < self.config.n_interferers + 1:
            raise ValueError(
                f"need at least {self.config.n_interferers + 1} speakers, "
                f"found {len(self.by_speaker)} under {self.root}"
            )
        self._length = (
            length if length is not None else sum(len(v) for v in self.by_speaker.values())
        )

    @property
    def speakers(self) -> list[str]:
        return sorted(self.by_speaker)

    def __len__(self) -> int:
        return self._length

    def _load(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        with np.load(path) as z:
            return z["audio"], z["mouth"]

    def _chunk(
        self, audio: np.ndarray, mouth: np.ndarray, frames: int, rng: random.Random
    ) -> tuple[np.ndarray, np.ndarray]:
        """Take aligned audio and video windows, padding a short clip."""
        if len(mouth) < frames:
            pad_f = frames - len(mouth)
            mouth = np.concatenate([mouth, np.zeros((pad_f, *mouth.shape[1:]), mouth.dtype)])
            audio = np.concatenate([audio, np.zeros(pad_f * SAMPLES_PER_FRAME, audio.dtype)])
            start_f = 0
        else:
            start_f = rng.randint(0, len(mouth) - frames)
        a = start_f * SAMPLES_PER_FRAME
        n = frames * SAMPLES_PER_FRAME
        audio_chunk = audio[a : a + n]
        if len(audio_chunk) < n:
            audio_chunk = np.pad(audio_chunk, (0, n - len(audio_chunk)))
        return audio_chunk.astype(np.float32), mouth[start_f : start_f + frames]

    def sample(self, index: int | None = None) -> MixSample:
        cfg = self.config
        rng = random.Random(self._rng.random() if index is None else (cfg.seed or 0) + index)

        frames = int(cfg.chunk_seconds * FPS)
        n_samples = frames * SAMPLES_PER_FRAME

        chosen = rng.sample(self.speakers, cfg.n_interferers + 1)
        target_spk, interferer_spks = chosen[0], tuple(chosen[1:])

        t_audio, t_mouth = self._load(rng.choice(self.by_speaker[target_spk]))
        target, mouth = self._chunk(t_audio, t_mouth, frames, rng)

        interferer = np.zeros(n_samples, dtype=np.float32)
        for spk in interferer_spks:
            i_audio, i_mouth = self._load(rng.choice(self.by_speaker[spk]))
            chunk, _ = self._chunk(i_audio, i_mouth, frames, rng)
            chunk = _scale_to_tir(target, chunk, rng.uniform(*cfg.tir_db))

            # Sparse overlap: the interferer occupies a contiguous span rather
            # than the whole chunk, because real conversation is not fully
            # overlapped and a model trained as if it were degrades on real
            # input (docs/06 §5).
            ratio = rng.uniform(*cfg.overlap_ratio)
            span = int(n_samples * ratio)
            if span < n_samples:
                start = rng.randint(0, n_samples - span)
                masked = np.zeros_like(chunk)
                masked[start : start + span] = chunk[start : start + span]
                chunk = masked
            interferer = interferer + chunk

        mixture = target + interferer
        peak = float(np.abs(mixture).max())
        if peak > cfg.peak_ceiling:
            # Scale all three together: rescaling only the mixture would change
            # the target the model is asked to reproduce.
            k = cfg.peak_ceiling / peak
            mixture, target, interferer = mixture * k, target * k, interferer * k

        # Per-frame target activity, for silence-region loss terms (docs/04 §4).
        frame_rms = np.sqrt((target.reshape(frames, SAMPLES_PER_FRAME) ** 2).mean(axis=1) + 1e-12)
        active = frame_rms > (frame_rms.max() * 0.05)

        return MixSample(
            mixture=mixture.astype(np.float32),
            target=target.astype(np.float32),
            interferer=interferer.astype(np.float32),
            mouth=mouth,
            active=active,
            target_speaker=target_spk,
            interferer_speakers=interferer_spks,
        )

    def __getitem__(self, index: int) -> MixSample:
        return self.sample(index)
