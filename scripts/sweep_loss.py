"""Loss-weight sweep for C1 (Phase 4b).

The finished C1 model is artefact-limited, not interference-limited. On the
Libri2Mix test split it scores SIR +20.45 dB against SAR +7.83 dB, with PESQ
1.47 and STOI 0.60 — the interferer is essentially gone and what caps SI-SDR is
damage to the target. `suppress_tau_db` is 20.0 and measured SIR is 20.45, so
the suppression hinge has already released; the term is not the thing still
pushing. That points at the balance between the reconstruction terms and the
rest, which is a weighting question and answerable in hours rather than days.

Each variant warm-starts from the finished checkpoint and fine-tunes briefly,
so the comparison is "what does changing this weight do to a trained model"
rather than "which weighting trains fastest from scratch" — a different and
less useful question.

Every variant sees identical data in an identical order (same seed, same
splits, same step count), so a difference between them is the weighting and
nothing else. The control repeats the shipped weights: without it, any
improvement could just be the extra fine-tuning.

Usage:
    uv run python scripts/sweep_loss.py
    uv run python scripts/sweep_loss.py --steps 2000 --only control,recon
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = Path.home() / "runs" / "sweep"
LOGS = Path.home() / "logs"

# Shipped weights, for reference: sisdr 1.0, suppress 0.5, consistency 0.2,
# mrstft 0.3, silence 0.5.
VARIANTS: dict[str, dict[str, float]] = {
    # Does fine-tuning alone move the number? Everything else is measured
    # against this, not against the pre-sweep checkpoint.
    "control": {},
    # Halve suppression. Cheapest test of whether the term is still costing
    # target quality even after its hinge has released.
    "suppress-half": {"w-suppress": 0.25},
    # Double the spectral reconstruction term. Multi-resolution STFT loss is
    # the one that most directly penalises the kind of damage PESQ and STOI
    # are seeing.
    "mrstft-double": {"w-mrstft": 0.6},
    # Halve the silence penalty. Measured leakage is -13.6 dB, comfortably
    # quiet, so the term may be buying silence at the cost of speech.
    "silence-half": {"w-silence": 0.25},
    # The combination the three above suggest, if each helps a little.
    "recon": {"w-suppress": 0.25, "w-mrstft": 0.6, "w-silence": 0.25},
}


def run_variant(name: str, weights: dict[str, float], args: argparse.Namespace) -> float | None:
    out = RUNS / name
    log = LOGS / f"sweep-{name}.log"
    cmd = [
        "uv", "run", "python", str(REPO / "scripts" / "train_c1.py"),
        "--steps", str(args.steps),
        "--init-from", str(args.init_from),
        "--clean-fraction", "0",
        "--lr", str(args.lr),
        "--val-every", str(args.val_every),
        "--val-items", "300",
        "--out", str(out),
    ]  # fmt: skip
    for key, value in weights.items():
        cmd += [f"--{key}", str(value)]

    print(f"  {name:16s} {weights or 'shipped weights'}")
    with log.open("w") as handle:
        subprocess.run(cmd, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, check=False)  # noqa: S603

    history = out / "log.json"
    if not history.exists():
        print(f"  {name:16s} FAILED — see {log}")
        return None
    best = max(float(e["val_si_sdri"]) for e in json.loads(history.read_text()))
    print(f"  {name:16s} best dev {best:+.2f} dB")
    return best


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=4e-5, help="fine-tune rate, below the C1 peak")
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--init-from", type=Path, default=Path.home() / "runs" / "c1" / "best.pt")
    ap.add_argument("--only", default=None, help="comma-separated subset of variant names")
    args = ap.parse_args(argv)

    if not args.init_from.exists():
        raise SystemExit(f"no checkpoint at {args.init_from}")

    chosen = (
        {k: VARIANTS[k] for k in (s.strip() for s in args.only.split(",")) if k in VARIANTS}
        if args.only
        else VARIANTS
    )
    RUNS.mkdir(parents=True, exist_ok=True)

    print(f"{len(chosen)} variants x {args.steps:,} steps from {args.init_from.name}\n")
    results: dict[str, float | None] = {}
    for name, weights in chosen.items():
        results[name] = run_variant(name, weights, args)

    print("\n  ranked:")
    ranked = sorted(((k, v) for k, v in results.items() if v is not None), key=lambda kv: -kv[1])
    control = results.get("control")
    for name, score in ranked:
        delta = f"  ({score - control:+.2f} vs control)" if control is not None else ""
        print(f"    {name:16s} {score:+.2f} dB{delta}")
    (RUNS / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
