"""C1 — audio-only target-speaker extraction on Libri2Mix (Phase 4b).

The gate is >= 13 dB SI-SDRi. This is the first stage where the numbers mean
something: C0 only proved the architecture can fit one batch, and this asks
whether it generalises across speakers it has and has not seen.

Validation runs on the dev split and reports SI-SDRi rather than loss, because
loss mixes five terms with arbitrary weights and cannot be compared against the
literature or against the gate.

The best checkpoint is chosen on validation SI-SDRi, never on training loss —
they diverge exactly when it matters, and picking on the wrong one is how a run
that looks fine ships a worse model than it had at step 4000.

Usage:
    uv run python scripts/train_c1.py --steps 20000
    uv run python scripts/train_c1.py --steps 200 --smoke
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
from training.librimix_data import (
    CLEAN,
    ConcatMixDataset,
    Libri2MixDataset,
    LibriMixConfig,
    to_batch_dict,
)
from training.losses import LossWeights, si_sdr
from training.trainer import TrainConfig, Trainer

ROOT = Path.home() / "data" / "Libri2Mix" / "Libri2Mix" / "wav16k" / "min"
ENROL = Path.home() / "data" / "Libri2Mix" / "enrolment"
GATE_DB = 13.0

# Either a single split or several addressed as one. The trainer only ever
# needs `len` and `sample`, so both satisfy it.
MixSource = Libri2MixDataset | ConcatMixDataset


def collate(items: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.stack([i[k] for i in items])) for k in items[0]}


def make_batches(
    ds: MixSource, indices: list[int], batch_size: int
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
    model: Seave, ds: MixSource, n_items: int, batch_size: int, device: torch.device
) -> dict[str, float]:
    """SI-SDRi on held-out data, with the improvement over the mixture."""
    model.eval()
    scores: list[float] = []
    rng = np.random.default_rng(0)  # same items every time, so runs are comparable
    indices = rng.choice(len(ds), size=min(n_items, len(ds)), replace=False).tolist()

    for batch in make_batches(ds, indices, batch_size):
        mixture = batch["mixture"].to(device)
        target = batch["target"].to(device)
        emb = batch["speaker_embedding"].to(device)
        conf = torch.ones(mixture.shape[0], device=device)
        est = model(mixture, emb, conf)["estimate"]
        improvement = si_sdr(est, target) - si_sdr(mixture, target)
        scores.extend(improvement.detach().cpu().tolist())

    model.train()
    finite = [s for s in scores if np.isfinite(s)]
    return {
        "si_sdri": float(np.mean(finite)) if finite else float("nan"),
        "n": float(len(finite)),
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=20000)
    # Batch 8 with two accumulation passes, not 4 with four. Same effective
    # batch of 16, 2.5x the throughput: four sequential passes over a quarter
    # of the card is the slowest way to reach 16 (scripts/bench_train.py).
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument(
        "--clean-fraction",
        type=float,
        default=0.35,
        help="portion of the run trained on noise-free mixtures before WHAM! is added",
    )
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-items", type=int, default=200)
    ap.add_argument("--chunk-seconds", type=float, default=4.0)
    # train-100 alone was overfit: +9.28 dB on seen data against +5.28 on
    # dev. train-360 is 3.7x the mixtures and already generated, so the
    # default trains on both.
    ap.add_argument("--splits", default="train-100,train-360")
    ap.add_argument("--out", type=Path, default=Path.home() / "runs" / "c1")
    ap.add_argument("--smoke", action="store_true", help="tiny model, for a wiring check")
    ap.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="resume from a checkpoint; bare --resume picks up <out>/last.pt",
    )
    ap.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="start from these weights with a fresh schedule (not a resume)",
    )
    ap.add_argument(
        "--init-partial",
        action="store_true",
        help="allow --init-from to leave unmatched parameters randomly initialised",
    )
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_npz, dev_npz = ENROL / "train-100.npz", ENROL / "dev.npz"
    for p in (train_npz, dev_npz):
        if not p.exists():
            print(f"missing {p}; run scripts/precompute_enrolment.py first")
            return 2

    # Two views of the same training set. The first half of the run sees the
    # sources summed with no noise; the rest sees the WHAM!-corrupted mixtures
    # the product actually has to cope with. Separating two voices *and*
    # denoising is a strictly harder problem than separating two voices, and
    # the first C1 run spent all 11.5 of its epochs on the harder one.
    #
    # Validation stays on mix_both throughout, so the reported number never
    # flatters itself: the curriculum changes what the model is shown, not
    # what it is judged on.
    noisy_cfg = LibriMixConfig(chunk_seconds=args.chunk_seconds, seed=0)
    clean_cfg = LibriMixConfig(chunk_seconds=args.chunk_seconds, seed=0, mixture_type=CLEAN)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    missing = [s for s in splits if not (ENROL / f"{s}.npz").exists()]
    if missing:
        print(f"missing enrolment for {missing}; run scripts/precompute_enrolment.py --split ...")
        return 2

    train_noisy = ConcatMixDataset(
        [Libri2MixDataset(ROOT / s, ENROL / f"{s}.npz", noisy_cfg) for s in splits]
    )
    train_clean = ConcatMixDataset(
        [Libri2MixDataset(ROOT / s, ENROL / f"{s}.npz", clean_cfg) for s in splits]
    )
    dev_ds = Libri2MixDataset(ROOT / "dev", dev_npz, noisy_cfg)
    clean_until = int(args.steps * args.clean_fraction)
    train_ds = train_clean if clean_until > 0 else train_noisy
    print(f"train {len(train_ds):,} items from {'+'.join(splits)}, dev {len(dev_ds):,} items")
    if clean_until > 0:
        print(f"  curriculum: noise-free until step {clean_until:,}, then mix_both")
    if train_ds.skipped_single_clip_speakers:
        print(f"  {train_ds.skipped_single_clip_speakers} speakers skipped (single clip)")

    model_cfg = SeaveConfig(emb_dim=32, lstm_hidden=48, n_blocks=2) if args.smoke else SeaveConfig()
    model = Seave(model_cfg)
    trainer = Trainer(
        model,
        TrainConfig(
            steps=args.steps,
            lr=args.lr,
            grad_accum=args.grad_accum,
            warmup_steps=min(500, args.steps // 10),
            device=str(device),
            modality_dropout=False,  # C1 is audio-only; there is no visual stream to drop
            out_dir=args.out,
        ),
        LossWeights(),
    )
    params = sum(p.numel() for p in model.parameters())
    print(
        f"device={device}  params={params / 1e6:.1f}M  "
        f"batch={args.batch}x{args.grad_accum}  steps={args.steps}\n"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    best = -np.inf
    log: list[dict[str, float]] = []
    start_step = 0

    if args.init_from is not None:
        if args.resume is not None:
            print("--init-from and --resume are mutually exclusive")
            return 2
        skipped = trainer.init_from(args.init_from, strict=not args.init_partial)
        print(f"initialised from {args.init_from.name}, {len(skipped)} params left random")
        print()

    if args.resume is not None:
        resume_path = args.out / "last.pt" if args.resume == "auto" else Path(args.resume)
        if not resume_path.exists():
            print(f"missing {resume_path}; nothing to resume from")
            return 2
        extra = trainer.load(resume_path)
        start_step = trainer.step
        best = float(extra.get("val_si_sdri", -np.inf))
        log_path = args.out / "log.json"
        if log_path.exists():
            log = json.loads(log_path.read_text())
        print(f"resumed {resume_path.name} at step {start_step}, best dev {best:+.2f} dB\n")

    t0 = time.perf_counter()

    for step in range(start_step, args.steps):
        # Seeded per step rather than from one long-lived generator, so a
        # resumed run draws the batches that step would have drawn instead of
        # replaying the first ones. A stream generator would need its position
        # checkpointed too, and getting that subtly wrong shows up as quiet
        # over-training on a prefix of the data rather than as a failure.
        rng = np.random.default_rng([1, step])
        stage = train_clean if step < clean_until else train_noisy
        if stage is not train_ds:
            train_ds = stage
            print(f"  -- switching to mix_both at step {step:,} --", flush=True)
        picks = rng.integers(0, len(train_ds), size=args.batch * args.grad_accum).tolist()
        batches = make_batches(train_ds, picks, args.batch)
        result = trainer.train_step(batches)

        if step % 50 == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  step {step:6d}  loss {result.loss:8.3f}  "
                f"sisdr {result.terms['sisdr']:7.2f}  "
                f"lr {result.lr:.2e}  {elapsed / (step - start_step + 1):.2f}s/step",
                flush=True,
            )

        if (step + 1) % args.val_every == 0 or step == args.steps - 1:
            metrics = validate(model, dev_ds, args.val_items, args.batch, device)
            entry = {"step": float(step), "val_si_sdri": metrics["si_sdri"], **result.terms}
            log.append(entry)
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
            # `last.pt` tracks the newest step, not the best score, and that is
            # the difference that makes it a resume point: a run whose
            # validation has not improved for 8000 steps still has 8000 steps
            # of progress to lose. 45 MB, overwritten in place.
            trainer.save(args.out / "last.pt", {"val_si_sdri": best})

    trainer.save(args.out / "last.pt", {"val_si_sdri": best})
    trainer.write_history(args.out / "history.json")

    print(f"\n  best dev SI-SDRi {best:+.2f} dB")
    print(f"  C1 gate (>= {GATE_DB:.0f} dB): {'PASS' if best >= GATE_DB else 'not yet'}")
    if args.smoke:
        print("  (smoke run: tiny model and few steps — the gate is not meaningful here)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
