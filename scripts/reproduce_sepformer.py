"""Reproduce published SepFormer numbers on Libri2Mix (Phase 3 exit criterion).

This validates the harness, not the model. If our measured SI-SDRi lands within
about 1 dB of SpeechBrain's published 20.6 dB under *matched* conditions, then
numbers this harness produces for our own systems can be trusted and compared
against the literature. If it does not, every later number is suspect and we
would not know it.

Matched conditions matter more than they sound. The published figure is for
`speechbrain/sepformer-libri2mix` on Libri2Mix **test / 8 kHz / min /
mix_clean**. Our training corpus is 16 kHz mix_both, and running the model on
that instead would produce a different number for perfectly good reasons —
which would then read as a harness failure. So the 8 kHz clean test split is
generated specifically for this comparison.

Separation output is permutation-free: the model emits two sources in arbitrary
order, so each item is scored under the best assignment, as the literature does.

Usage:
    uv run python scripts/reproduce_sepformer.py --limit 300
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

from eval.harness import bootstrap_ci
from eval.metrics import si_sdr

PUBLISHED_SI_SDRI_DB = 20.6
TOLERANCE_DB = 1.0
RATE = 8_000

ROOT = Path.home() / "data" / "Libri2Mix8k" / "Libri2Mix" / "wav8k" / "min" / "test"


def best_permutation_si_sdri(
    estimates: list[np.ndarray], references: list[np.ndarray], mixture: np.ndarray
) -> float:
    """Mean SI-SDRi over the best source assignment.

    Separation models have no notion of which output is which speaker, so
    scoring a fixed order would measure ordering luck rather than separation.
    """
    best = -np.inf
    for perm in itertools.permutations(range(len(references))):
        total = 0.0
        for ref_idx, est_idx in enumerate(perm):
            ref = references[ref_idx]
            n = min(len(ref), len(estimates[est_idx]), len(mixture))
            total += si_sdr(estimates[est_idx][:n], ref[:n]) - si_sdr(mixture[:n], ref[:n])
        best = max(best, total / len(references))
    return float(best)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=300, help="0 = the whole test set")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    mix_dir = ROOT / "mix_clean"
    if not mix_dir.is_dir():
        print(f"missing {mix_dir}; generate the 8k mix_clean test split first")
        return 2

    items = sorted(mix_dir.glob("*.wav"))
    if args.limit > 0:
        items = items[: args.limit]
    print(f"scoring {len(items)} items from {mix_dir}")

    import torch
    from speechbrain.inference.separation import SepformerSeparation

    model = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-libri2mix",
        savedir=str(Path.home() / "models" / "sepformer-libri2mix"),
        run_opts={"device": args.device},
    )

    scores: list[float] = []
    t0 = time.perf_counter()
    for i, mix_path in enumerate(items, start=1):
        mixture, rate = sf.read(mix_path, dtype="float32")
        if rate != RATE:
            print(f"unexpected rate {rate} in {mix_path.name}")
            return 1
        refs = [sf.read(ROOT / src / mix_path.name, dtype="float32")[0] for src in ("s1", "s2")]

        with torch.no_grad():
            est = model.separate_batch(torch.from_numpy(mixture).unsqueeze(0))
        estimates = [est[0, :, k].cpu().numpy() for k in range(est.shape[-1])]

        scores.append(best_permutation_si_sdri(estimates, refs, mixture))
        if i % 50 == 0:
            print(f"  {i}/{len(items)}  running mean {np.mean(scores):.2f} dB", flush=True)

    elapsed = time.perf_counter() - t0
    mean, lo, hi = bootstrap_ci(scores)
    delta = mean - PUBLISHED_SI_SDRI_DB
    passed = abs(delta) <= TOLERANCE_DB

    print(f"\n  items          {len(scores)}  ({elapsed:.0f}s)")
    print(f"  measured       {mean:.2f} dB SI-SDRi  95% CI [{lo:.2f}, {hi:.2f}]")
    print(f"  published      {PUBLISHED_SI_SDRI_DB:.2f} dB")
    print(f"  difference     {delta:+.2f} dB")
    print(f"\n  harness validation (within {TOLERANCE_DB:.0f} dB): {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
