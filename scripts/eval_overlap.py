"""Evaluate on the product's actual condition: 2-3 people talking at once.

Every corpus we have measures something adjacent to the product but not the
product. Libri2Mix is two speakers with no room. AMI has rooms and real
conversation but, measured properly, contains almost no genuine simultaneous
speech — across ten meetings exactly one 4-second window has two people both
clearly talking. Averaging over AMI therefore scores the model on passages
where there is nothing to separate, which is how several days of measurement
came back negative and misleading.

So this builds the condition directly: real recorded speech, a simulated room,
and an overlap ratio that is *set* and then *verified* rather than hoped for.
Items are generated deterministically from a seed, so the set is reproducible
without storing a single wav.

Three things it reports that a bare mean hides:

- **Measured overlap**, not requested overlap. If the schedule did not actually
  put two voices on top of each other, the item is discarded rather than
  scored — the mistake that made the AMI numbers meaningless.
- **Mixture SI-SDR**, so it is obvious when a "test" is already clean. Anything
  much above +5 dB is not a separation problem.
- **Results split by speaker count**, because two and three are different
  problems and the product allows both.

Usage:
    uv run python scripts/eval_overlap.py --checkpoint ~/runs/c3/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

from models.seave import Seave, SeaveConfig
from pipeline.passthrough import frame_activity
from training.librimix_data import Libri2MixDataset, LibriMixConfig
from training.losses import si_sdr
from training.simulate import SimConfig, simulate

ROOT = Path.home() / "data" / "Libri2Mix" / "Libri2Mix" / "wav16k" / "min"
ENROL = Path.home() / "data" / "Libri2Mix" / "enrolment"


def measured_overlap(sources: list[np.ndarray]) -> float:
    """Fraction of frames where at least two speakers are audible.

    Computed from the post-room sources, so it reflects what the mixture
    actually contains rather than what the turn schedule intended.
    """
    masks = [frame_activity(s) for s in sources]
    if not masks:
        return 0.0
    width = min(len(m) for m in masks)
    stack = np.stack([m[:width] for m in masks])
    return float((stack.sum(axis=0) >= 2).mean())


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=Path.home() / "runs" / "c3" / "best.pt")
    ap.add_argument("--speakers", default="2,3")
    ap.add_argument("--items", type=int, default=200)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument(
        "--min-overlap",
        type=float,
        default=0.25,
        help="discard items whose measured simultaneous speech is below this",
    )
    ap.add_argument("--overlap-range", default="0.3,0.7")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    lo, hi = (float(x) for x in args.overlap_range.split(","))
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = Seave(SeaveConfig()).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"checkpoint {args.checkpoint}, step {checkpoint['step']:,}")
    print(
        f"requested overlap {lo:.0%}-{hi:.0%}, discarding items measured below "
        f"{args.min_overlap:.0%}\n"
    )

    base = Libri2MixDataset(
        ROOT / "dev", ENROL / "dev.npz", LibriMixConfig(chunk_seconds=args.seconds, seed=0)
    )

    for n_spk in (int(c) for c in args.speakers.split(",")):
        scores: list[float] = []
        mixes: list[float] = []
        overlaps: list[float] = []
        discarded = 0
        rng = np.random.default_rng(args.seed)

        attempt = 0
        while len(scores) < args.items and attempt < args.items * 8:
            attempt += 1
            idx = int(rng.integers(0, len(base)))
            item = base.sample(idx)
            sources = [item.target, item.interferer]
            # A third voice comes from a different item, so it is a different
            # speaker rather than the same one twice.
            while len(sources) < n_spk:
                other = base.sample(int(rng.integers(0, len(base))))
                if other.target_speaker != item.target_speaker:
                    sources.append(other.target)

            sim = simulate(
                sources,
                cfg=SimConfig(overlap_ratio=(lo, hi), seed=args.seed * 100003 + attempt),
            )
            actual = measured_overlap(sim.sources)
            if actual < args.min_overlap:
                discarded += 1
                continue

            with torch.no_grad():
                mix_t = torch.from_numpy(sim.mixture)[None].to(args.device)
                tgt_t = torch.from_numpy(sim.sources[0])[None].to(args.device)
                emb = torch.from_numpy(item.enrolment)[None].to(args.device)
                est = model(mix_t, emb, torch.ones(1, device=args.device))["estimate"]
                before = float(si_sdr(mix_t, tgt_t)[0])
                after = float(si_sdr(est, tgt_t)[0])

            if np.isfinite(before) and np.isfinite(after):
                scores.append(after - before)
                mixes.append(before)
                overlaps.append(actual)

        if not scores:
            print(f"  {n_spk} speakers: no items met the overlap requirement")
            continue

        arr = np.array(scores)
        # Bootstrap rather than a bare mean: a 0.5 dB difference read off two
        # means is meaningless if the interval is +/-1.5 dB.
        boot = np.array(
            [np.mean(rng.choice(arr, size=arr.size, replace=True)) for _ in range(2000)]
        )
        print(
            f"  {n_spk} speakers  n={len(arr):3d} (discarded {discarded:3d})  "
            f"overlap {np.mean(overlaps):.0%}  mixture {np.mean(mixes):+6.2f} dB\n"
            f"      SI-SDRi {arr.mean():+6.2f} dB  "
            f"[{np.percentile(boot, 2.5):+.2f}, {np.percentile(boot, 97.5):+.2f}]  "
            f"median {np.median(arr):+6.2f}  p25 {np.percentile(arr, 25):+6.2f}"
        )

    print("\n  This is the product condition: real speech, simulated room, verified overlap.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
