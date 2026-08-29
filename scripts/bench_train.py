"""How many epochs fit in the compute budget, measured rather than assumed.

C1 reached +2.40 dB against a 13 dB gate, and the diagnosis was not a broken
model — swapping the enrolment cue moves the output by 11.7 dB, so extraction
works. It saw 11.5 epochs where docs/07 asks for ~60. The question that follows
is therefore about throughput, because throughput is what converts a fixed
GPU-hour budget into epochs.

The first run used batch 4 with grad_accum 4 and gradient checkpointing on,
which is the configuration `SeaveConfig` was trimmed to so it would fit 24 GB.
Both of those choices cost speed: four sequential forward/backward passes over
a tiny batch underuse the GPU, and checkpointing buys memory by recomputing the
forward pass. This sweep measures what each is actually worth, so the next run
is sized on numbers instead of on the comment that justified the last one.

Usage:
    uv run python scripts/bench_train.py
    uv run python scripts/bench_train.py --steps 12 --budget-hours 65
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

from models.seave import Seave, SeaveConfig
from training.librimix_data import Libri2MixDataset, LibriMixConfig, to_batch_dict
from training.losses import LossWeights
from training.trainer import TrainConfig, Trainer

ROOT = Path.home() / "data" / "Libri2Mix" / "Libri2Mix" / "wav16k" / "min"
ENROL = Path.home() / "data" / "Libri2Mix" / "enrolment"
# Nothing is saved during a benchmark; the trainer only wants somewhere to
# point at, so this stays under the user's own runs directory.
BENCH_OUT = Path.home() / "runs" / "bench"


@dataclass(frozen=True)
class Variant:
    emb_dim: int
    batch: int
    accum: int
    checkpointing: bool

    @property
    def effective(self) -> int:
        return self.batch * self.accum

    def __str__(self) -> str:
        ckpt = "ckpt" if self.checkpointing else "no-ckpt"
        return f"emb{self.emb_dim:3d}  b{self.batch:2d}x{self.accum}={self.effective:2d}  {ckpt:7s}"


@dataclass
class Result:
    variant: Variant
    seconds_per_step: float
    peak_gb: float
    failed: str | None = None


def collate(items: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.stack([i[k] for i in items])) for k in items[0]}


def measure(
    variant: Variant, dataset: Libri2MixDataset, steps: int, warmup: int, chunk: float
) -> Result:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = Seave(
            SeaveConfig(emb_dim=variant.emb_dim, gradient_checkpointing=variant.checkpointing)
        )
        trainer = Trainer(
            model,
            TrainConfig(
                steps=steps + warmup,
                grad_accum=variant.accum,
                device="cuda",
                modality_dropout=False,
                out_dir=BENCH_OUT,
            ),
            LossWeights(),
        )

        capacity = torch.cuda.get_device_properties(0).total_memory / 1024**3
        rng = np.random.default_rng(0)
        elapsed = 0.0
        for i in range(steps + warmup):
            picks = rng.integers(0, len(dataset), size=variant.effective).tolist()
            batches = [
                collate([to_batch_dict(dataset.sample(p)) for p in picks[j : j + variant.batch]])
                for j in range(0, variant.effective, variant.batch)
            ]
            # Warmup steps are excluded: the first pass pays for cuDNN
            # autotuning and allocator growth, which a long run amortises away.
            torch.cuda.synchronize()
            start = time.perf_counter()
            trainer.train_step(batches)
            torch.cuda.synchronize()
            if i >= warmup:
                elapsed += time.perf_counter() - start

            # Bail as soon as the working set exceeds the card. WSL does not
            # raise here — it pages the excess to system RAM and keeps going at
            # a fifteenth of the speed, which looks like a slow configuration
            # rather than an impossible one. Measured: b16 with checkpointing
            # wanted 35 GB and ran at 58 s/step against 3.8 s/step for a
            # configuration that fits. Timing that to completion tells us
            # nothing except how patient the benchmark is.
            live = torch.cuda.max_memory_allocated() / 1024**3
            if live > capacity:
                del trainer, model
                return Result(variant, float("nan"), live, failed=f"paged ({live:.0f} GB)")

        peak = torch.cuda.max_memory_allocated() / 1024**3
        del trainer, model
        return Result(variant, elapsed / steps, peak)
    except torch.cuda.OutOfMemoryError:
        return Result(variant, float("nan"), float("nan"), failed="OOM")
    finally:
        torch.cuda.empty_cache()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--chunk-seconds", type=float, default=4.0)
    ap.add_argument("--budget-hours", type=float, default=65.0)
    ap.add_argument("--items", type=int, default=27_800, help="train-set size, for epoch maths")
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"{name}, {total:.1f} GB\n")

    dataset = Libri2MixDataset(
        ROOT / "train-100",
        ENROL / "train-100.npz",
        LibriMixConfig(chunk_seconds=args.chunk_seconds),
    )

    variants = [
        # The configuration C1 actually ran, as the baseline to beat.
        Variant(96, 4, 4, True),
        # Same effective batch, fewer sequential passes.
        Variant(96, 8, 2, True),
        Variant(96, 16, 1, True),
        # What checkpointing costs, at each of those shapes.
        Variant(96, 8, 2, False),
        Variant(96, 16, 1, False),
        # The size docs/07 §2 actually specifies, which was trimmed to 96 to fit.
        Variant(128, 8, 2, True),
        Variant(128, 4, 4, True),
    ]

    results: list[Result] = []
    for variant in variants:
        print(f"  {variant} ... ", end="", flush=True)
        result = measure(variant, dataset, args.steps, args.warmup, args.chunk_seconds)
        results.append(result)
        if result.failed:
            print(result.failed)
        else:
            print(f"{result.seconds_per_step:6.2f} s/step   peak {result.peak_gb:5.1f} GB")

    print(f"\n  epochs reachable in {args.budget_hours:.0f} GPU-hours ({args.items:,} items):\n")
    print(f"  {'configuration':34s} {'samples/s':>10s} {'epochs':>8s} {'vs C1':>7s}")
    baseline = next((r for r in results if r.variant == variants[0] and not r.failed), None)
    for r in results:
        if r.failed:
            print(f"  {r.variant!s:34s} {r.failed:>10s}")
            continue
        rate = r.variant.effective / r.seconds_per_step
        epochs = rate * args.budget_hours * 3600 / args.items
        speedup = (
            f"{(baseline.seconds_per_step / baseline.variant.effective) / (r.seconds_per_step / r.variant.effective):.2f}x"
            if baseline
            else "-"
        )
        print(f"  {r.variant!s:34s} {rate:10.2f} {epochs:8.1f} {speedup:>7s}")

    print("\n  docs/07 §1 asks for ~60 epochs from a pretrained init, ~150 from scratch.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
