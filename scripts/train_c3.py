"""C3 — realistic simulation (Phase 4d, docs/07 §1).

C1 learned to separate two clean recordings summed together. Nothing it has
heard has a room in it, and the Phase 1 baseline report identified reverberation
as the dominant real-world degradation — so the one axis the model has no
experience of is the one every upload will have.

C3 keeps the same speakers and the same objective and changes only how the
mixture is built: image-source room responses at sampled RT60, turn-taking
instead of uniform overlap, per-speaker level spread, noise at a sampled SNR,
and a lossy codec round-trip half the time. The gate is >= 10 dB SI-SDRi on the
realistic set.

Initialised from C1 rather than from scratch, which is the whole point of a
curriculum: separation is already learned, and what is being added is
robustness to the room.

Validation runs on realistic dev mixtures generated with a fixed seed, so the
number is stable across runs and comparable between checkpoints. It is a
*different* and harder distribution than the C1 dev number — a drop on the
first validation is expected and is not a regression.

Usage:
    uv run python scripts/train_c3.py --init-from ~/runs/c1/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

from models.seave import Seave, SeaveConfig
from training.librimix_data import Libri2MixDataset, LibriMixConfig, MixItem, to_batch_dict
from training.losses import LossWeights, si_sdr
from training.realistic_data import RealisticConfig, RealisticMixDataset
from training.trainer import TrainConfig, Trainer

ROOT = Path.home() / "data" / "Libri2Mix" / "Libri2Mix" / "wav16k" / "min"
ENROL = Path.home() / "data" / "Libri2Mix" / "enrolment"
GATE_DB = 10.0


def collate(items: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.stack([i[k] for i in items])) for k in items[0]}


def make_batches(
    ds: RealisticMixDataset, indices: list[int], batch_size: int
) -> list[dict[str, torch.Tensor]]:
    out = []
    for start in range(0, len(indices), batch_size):
        group = indices[start : start + batch_size]
        if len(group) < batch_size:
            break
        out.append(collate([to_batch_dict(ds.sample(i)) for i in group]))
    return out


@torch.no_grad()
def validate(
    model: Seave,
    ds: RealisticMixDataset,
    n_items: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    scores: list[float] = []
    rng = np.random.default_rng(0)  # same items every time
    indices = rng.choice(len(ds), size=min(n_items, len(ds)), replace=False).tolist()

    for batch in make_batches(ds, indices, batch_size):
        mixture = batch["mixture"].to(device)
        target = batch["target"].to(device)
        emb = batch["speaker_embedding"].to(device)
        conf = torch.ones(mixture.shape[0], device=device)
        est = model(mixture, emb, conf)["estimate"]
        scores.extend((si_sdr(est, target) - si_sdr(mixture, target)).detach().cpu().tolist())

    model.train()
    finite = [s for s in scores if np.isfinite(s)]
    return {
        "si_sdri": float(np.mean(finite)) if finite else float("nan"),
        "n": float(len(finite)),
    }


def build_split(split: str, chunk: float, dry: float) -> RealisticMixDataset:
    base = Libri2MixDataset(
        ROOT / split, ENROL / f"{split}.npz", LibriMixConfig(chunk_seconds=chunk, seed=0)
    )
    return RealisticMixDataset(base, RealisticConfig(chunk_seconds=chunk, seed=0, dry_fraction=dry))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=8e-5, help="below the C1 peak; this is adaptation")
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-items", type=int, default=200)
    ap.add_argument("--chunk-seconds", type=float, default=4.0)
    ap.add_argument("--splits", default="train-100,train-360")
    ap.add_argument(
        "--dry-fraction",
        type=float,
        default=0.15,
        help="portion of samples left unsimulated, so dry audio is not forgotten",
    )
    ap.add_argument("--out", type=Path, default=Path.home() / "runs" / "c3")
    ap.add_argument("--init-from", type=Path, default=Path.home() / "runs" / "c1" / "best.pt")
    ap.add_argument("--resume", nargs="?", const="auto", default=None)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    missing = [s for s in splits if not (ENROL / f"{s}.npz").exists()]
    if missing:
        print(f"missing enrolment for {missing}")
        return 2

    # One RealisticMixDataset per split, addressed round-robin by index. Not
    # ConcatMixDataset, because the wrapper has to sit outside each split for
    # the per-index simulation seed to stay stable.
    parts = [build_split(s, args.chunk_seconds, args.dry_fraction) for s in splits]
    dev_ds = build_split("dev", args.chunk_seconds, dry=0.0)

    class RoundRobin:
        """Interleave the splits so every batch mixes both."""

        def __init__(self, datasets: list[RealisticMixDataset]) -> None:
            self.datasets = datasets
            self.total = sum(len(d) for d in datasets)

        def __len__(self) -> int:
            return self.total

        def sample(self, index: int) -> MixItem:
            which = index % len(self.datasets)
            return self.datasets[which].sample(index // len(self.datasets))

    train_ds = RoundRobin(parts)
    print(f"train {len(train_ds):,} items from {'+'.join(splits)}, dev {len(dev_ds):,} items")
    print(f"  realistic mixtures, {args.dry_fraction:.0%} left dry")

    model = Seave(SeaveConfig())
    trainer = Trainer(
        model,
        TrainConfig(
            steps=args.steps,
            lr=args.lr,
            grad_accum=args.grad_accum,
            warmup_steps=min(500, args.steps // 10),
            device=str(device),
            modality_dropout=False,
            out_dir=args.out,
        ),
        LossWeights(),
    )
    params = sum(p.numel() for p in model.parameters())
    print(
        f"device={device}  params={params / 1e6:.1f}M  batch={args.batch}x{args.grad_accum}  steps={args.steps}"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    best = -np.inf
    log: list[dict[str, float]] = []
    start_step = 0

    if args.resume is not None:
        path = args.out / "last.pt" if args.resume == "auto" else Path(args.resume)
        if not path.exists():
            print(f"missing {path}")
            return 2
        extra = trainer.load(path)
        start_step = trainer.step
        best = float(extra.get("val_si_sdri", -np.inf))
        if (args.out / "log.json").exists():
            log = json.loads((args.out / "log.json").read_text())
        print(f"resumed {path.name} at step {start_step}, best {best:+.2f} dB\n")
    elif args.init_from is not None:
        skipped = trainer.init_from(args.init_from)
        print(f"initialised from {args.init_from}, {len(skipped)} params left random\n")

    t0 = time.perf_counter()
    for step in range(start_step, args.steps):
        rng = np.random.default_rng([3, step])
        picks = rng.integers(0, len(train_ds), size=args.batch * args.grad_accum).tolist()
        batches = make_batches(train_ds, picks, args.batch)  # type: ignore[arg-type]
        result = trainer.train_step(batches)

        if step % 50 == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  step {step:6d}  loss {result.loss:8.3f}  sisdr {result.terms['sisdr']:7.2f}  "
                f"lr {result.lr:.2e}  {elapsed / (step - start_step + 1):.2f}s/step",
                flush=True,
            )

        if (step + 1) % args.val_every == 0 or step == args.steps - 1:
            metrics = validate(model, dev_ds, args.val_items, args.batch, device)
            log.append({"step": float(step), "val_si_sdri": metrics["si_sdri"], **result.terms})
            marker = ""
            if metrics["si_sdri"] > best:
                best = metrics["si_sdri"]
                trainer.save(args.out / "best.pt", {"val_si_sdri": best})
                marker = "  <- best"
            print(
                f"    val SI-SDRi {metrics['si_sdri']:+.2f} dB (n={int(metrics['n'])}){marker}",
                flush=True,
            )
            (args.out / "log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
            trainer.save(args.out / "last.pt", {"val_si_sdri": best})

    trainer.write_history(args.out / "history.json")
    print(f"\n  best realistic SI-SDRi {best:+.2f} dB")
    print(f"  C3 gate (>= {GATE_DB:.0f} dB): {'PASS' if best >= GATE_DB else 'not yet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
