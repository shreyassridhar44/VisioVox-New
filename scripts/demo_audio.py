"""Render what the trained model actually sounds like.

A validation number says how much the interferer was attenuated on average. It
does not say whether the result is listenable, whether the target survived
intact, or what the residual sounds like — and those decide whether a
checkpoint is worth shipping. +5 dB of suppression with a mangled target is a
worse product than +4 dB of clean attenuation, and SI-SDRi cannot tell the two
apart.

For each example this writes the three signals side by side, plus one file
that plays them in sequence so the difference is audible without juggling
players:

    <name>_1_mixture.wav    what the model was given
    <name>_2_estimate.wav   what it returned
    <name>_3_target.wav     what it should have returned
    <name>_compare.wav      the three in order, separated by silence

Usage:
    uv run python scripts/demo_audio.py
    uv run python scripts/demo_audio.py --checkpoint ~/runs/c1/best.pt --items 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

from models.seave import Seave, SeaveConfig
from training.librimix_data import Libri2MixDataset, LibriMixConfig
from training.losses import si_sdr

ROOT = Path.home() / "data" / "Libri2Mix" / "Libri2Mix" / "wav16k" / "min"
ENROL = Path.home() / "data" / "Libri2Mix" / "enrolment"
RATE = 16_000


def normalise(audio: np.ndarray, peak: float = 0.89) -> np.ndarray:
    """Scale to a fixed peak so A/B comparison is not confounded by loudness.

    Level differences read as quality differences to the ear, and the point of
    these files is to compare content rather than gain.
    """
    largest = float(np.abs(audio).max())
    return audio if largest < 1e-9 else (audio * (peak / largest)).astype(np.float32)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=Path.home() / "runs" / "c1" / "best.pt")
    ap.add_argument("--out", type=Path, default=Path.home() / "runs" / "c1" / "audio")
    ap.add_argument("--items", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--gap", type=float, default=0.5)
    args = ap.parse_args(argv)

    if not args.checkpoint.exists():
        raise SystemExit(f"no checkpoint at {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = Seave(SeaveConfig())
    model.load_state_dict(checkpoint["model"])
    model.eval()
    trained = checkpoint["extra"].get("val_si_sdri")
    print(f"checkpoint step {checkpoint['step']:,}, dev SI-SDRi {trained:+.2f} dB\n")

    dataset = Libri2MixDataset(
        ROOT / "dev", ENROL / "dev.npz", LibriMixConfig(chunk_seconds=args.seconds, seed=0)
    )
    # The same held-out items the run validated on, so what you hear is drawn
    # from the set the reported number was computed over.
    rng = np.random.default_rng(0)
    picks = rng.choice(len(dataset), size=args.items, replace=False).tolist()

    args.out.mkdir(parents=True, exist_ok=True)
    silence = np.zeros(int(args.gap * RATE), dtype=np.float32)
    summary: list[tuple[str, float, float]] = []

    for n, index in enumerate(picks, start=1):
        item = dataset.sample(index)
        mixture = torch.from_numpy(item.mixture)[None]
        target = torch.from_numpy(item.target)[None]
        enrolment = torch.from_numpy(item.enrolment)[None]

        with torch.no_grad():
            estimate = model(mixture, enrolment, torch.ones(1))["estimate"]

        before = float(si_sdr(mixture, target)[0])
        after = float(si_sdr(estimate, target)[0])
        name = f"ex{n}_{item.target_speaker}"
        print(
            f"  {name}: mixture {before:+6.2f} dB -> estimate {after:+6.2f} dB  "
            f"(SI-SDRi {after - before:+.2f})"
        )

        parts = {
            "1_mixture": item.mixture,
            "2_estimate": estimate[0].numpy(),
            "3_target": item.target,
        }
        for suffix, audio in parts.items():
            sf.write(str(args.out / f"{name}_{suffix}.wav"), normalise(audio), RATE)

        joined = np.concatenate(
            [normalise(parts["1_mixture"]), silence,
             normalise(parts["2_estimate"]), silence,
             normalise(parts["3_target"])]
        )  # fmt: skip
        sf.write(str(args.out / f"{name}_compare.wav"), joined, RATE)
        summary.append((name, before, after))

    print(f"\nwrote {len(list(args.out.glob('*.wav')))} files to {args.out}")
    print("  each *_compare.wav plays: mixture, then model output, then ground truth")
    mean = float(np.mean([a - b for _, b, a in summary]))
    print(f"  mean SI-SDRi over these {len(summary)} examples: {mean:+.2f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
