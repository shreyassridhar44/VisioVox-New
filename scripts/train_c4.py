"""C4 — in-domain fine-tune on AMI (Phase 4e, docs/07 §1).

The last stage, and the one that decides whether any of this works on what
users actually upload. C3 taught the model simulated rooms; AMI is real ones,
recorded on real microphones, with four people interrupting each other.

docs/07 is explicit about the risk: "30 epochs on 18 sessions overfits fast.
Low LR, early stopping, heavy augmentation." This set is 80 targets across 20
meetings — small enough that the defaults here are deliberately conservative: a
low learning rate, a short run, frequent validation, and the best checkpoint
chosen on held-out AMI rather than on training loss.

Validation runs on **AMI-Eval**, which is the TS series — a different room and
different people from the ES and IS meetings trained on. So the number reported
is genuinely out-of-domain within AMI, not a re-read of the training set.

Usage:
    uv run python scripts/train_c4.py --init-from ~/runs/c3/best.pt
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
from training.ami_data import AmiConfig, AmiMixDataset
from training.librimix_data import to_batch_dict
from training.losses import LossWeights, si_sdr
from training.trainer import TrainConfig, Trainer

SETS = Path.home() / "data" / "ami" / "sets"
# NFR-ML-01: >= 14 dB on AMI-Eval, floor 11 dB. The floor is the gate here;
# the target is what the finished system is expected to reach after Phase 5.
GATE_DB = 11.0


def collate(items: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.stack([i[k] for i in items])) for k in items[0]}


def make_batches(
    ds: AmiMixDataset, indices: list[int], batch_size: int
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
    model: Seave, ds: AmiMixDataset, n_items: int, batch_size: int, device: torch.device
) -> dict[str, float]:
    model.eval()
    scores: list[float] = []
    rng = np.random.default_rng(0)
    # Sampled with replacement: AMI-Eval has 40 targets, fewer than the item
    # count we want, and each draw takes a different 4 s window.
    indices = rng.integers(0, len(ds), size=n_items).tolist()

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


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # Short by design. 80 targets will be memorised long before a C1-length run
    # would finish, and the best checkpoint is chosen on AMI-Eval anyway.
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5, help="low: this is adaptation, not training")
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--val-items", type=int, default=200)
    ap.add_argument("--chunk-seconds", type=float, default=4.0)
    ap.add_argument("--out", type=Path, default=Path.home() / "runs" / "c4")
    ap.add_argument("--init-from", type=Path, default=Path.home() / "runs" / "c3" / "best.pt")
    ap.add_argument("--resume", nargs="?", const="auto", default=None)
    args = ap.parse_args(argv)

    for name in ("train", "eval"):
        for path in (SETS / f"{name}.json", SETS / f"{name}-enrolment.npz"):
            if not path.exists():
                print(f"missing {path}; run build_ami_eval.py and precompute_ami_enrolment.py")
                return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = AmiConfig(chunk_seconds=args.chunk_seconds, seed=0)
    train_ds = AmiMixDataset(SETS / "train", SETS / "train.json", SETS / "train-enrolment.npz", cfg)
    dev_ds = AmiMixDataset(SETS / "eval", SETS / "eval.json", SETS / "eval-enrolment.npz", cfg)
    print(f"train {len(train_ds)} targets (ES+IS), eval {len(dev_ds)} targets (TS)")

    model = Seave(SeaveConfig())
    trainer = Trainer(
        model,
        TrainConfig(
            steps=args.steps,
            lr=args.lr,
            grad_accum=args.grad_accum,
            warmup_steps=min(200, args.steps // 10),
            device=str(device),
            modality_dropout=False,
            out_dir=args.out,
        ),
        LossWeights(),
    )
    print(
        f"device={device}  batch={args.batch}x{args.grad_accum}  steps={args.steps}  lr={args.lr}"
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
        # The pre-fine-tune number: how well C3 already does on real meetings,
        # before AMI is trained on at all. This is the honest measure of how
        # much the simulated rooms transferred, and it cannot be recovered
        # later once the weights have moved.
        baseline = validate(model, dev_ds, args.val_items, args.batch, device)
        best = baseline["si_sdri"]
        print(f"  before any AMI training: {baseline['si_sdri']:+.2f} dB on AMI-Eval\n")
        trainer.save(args.out / "best.pt", {"val_si_sdri": best})

    t0 = time.perf_counter()
    for step in range(start_step, args.steps):
        rng = np.random.default_rng([4, step])
        picks = rng.integers(0, len(train_ds), size=args.batch * args.grad_accum).tolist()
        result = trainer.train_step(make_batches(train_ds, picks, args.batch))

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
                f"    AMI-Eval SI-SDRi {metrics['si_sdri']:+.2f} dB "
                f"(n={int(metrics['n'])}){marker}",
                flush=True,
            )
            (args.out / "log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
            trainer.save(args.out / "last.pt", {"val_si_sdri": best})

    trainer.write_history(args.out / "history.json")
    print(f"\n  best AMI-Eval SI-SDRi {best:+.2f} dB")
    print(f"  NFR-ML-01 floor (>= {GATE_DB:.0f} dB): {'MET' if best >= GATE_DB else 'not met'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
