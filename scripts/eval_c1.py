"""Evaluate a C1 checkpoint on the Libri2Mix test split (docs/08 §2).

Every number quoted during training came from `dev`, which is also the split
`best.pt` was selected on. That makes dev mildly optimistic — choosing the peak
of a noisy curve on the same data you then report is a small but real form of
selection. `test` has never been looked at, and the C1 gate is defined on it.

Routed through `ml.eval.harness` rather than a bespoke loop, for three reasons
the harness exists to enforce: per-item rows are written so the result can be
re-sliced and audited later, aggregates come back as mean +/- 95% bootstrap CI
rather than a bare mean, and SIR is reported alongside SI-SDRi — SI-SDR cannot
tell suppression from distortion, and NFR-ML-03 is written against SIR.

Usage:
    uv run python scripts/eval_c1.py
    uv run python scripts/eval_c1.py --checkpoint ~/runs/c1/best.pt --limit 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

from eval.harness import EvalItem, EvalSpeaker, evaluate, format_summary, summarise
from models.seave import Seave, SeaveConfig
from training.librimix_data import Libri2MixDataset, LibriMixConfig

ROOT = Path.home() / "data" / "Libri2Mix" / "Libri2Mix" / "wav16k" / "min"
ENROL = Path.home() / "data" / "Libri2Mix" / "enrolment"
GATE_DB = 13.0
#: Speaker id for the reference that exists only to make SIR computable.
INTERFERER = "__interferer__"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=Path.home() / "runs" / "c1" / "best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, default=Path.home() / "runs" / "c1" / "eval")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--heavy", action="store_true", help="also compute PESQ and STOI")
    args = ap.parse_args(argv)

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = Seave(SeaveConfig()).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"checkpoint {args.checkpoint.name}, step {checkpoint['step']:,}")
    print(f"  selected on dev at {checkpoint['extra'].get('val_si_sdri'):+.2f} dB SI-SDRi\n")

    dataset = Libri2MixDataset(
        ROOT / args.split,
        ENROL / f"{args.split}.npz",
        LibriMixConfig(chunk_seconds=args.seconds, seed=0),
    )
    # Deterministic order and a fixed seed, so a rerun reproduces exactly.
    rng = np.random.default_rng(0)
    count = min(args.limit, len(dataset))
    picks = sorted(rng.choice(len(dataset), size=count, replace=False).tolist())
    print(f"{args.split}: {len(dataset):,} items, evaluating {count:,}\n")

    def items() -> list[EvalItem]:
        out: list[EvalItem] = []
        for index in picks:
            sample = dataset.sample(index)
            out.append(
                EvalItem(
                    item_id=f"{sample.mixture_id}|{sample.target_speaker}",
                    mixture=sample.mixture,
                    # The interferer is listed as a second speaker so the
                    # harness can decompose the output into signal, inter-
                    # ference and artefact, which is what gives SIR. Without
                    # another reference in the item there is nothing to measure
                    # interference against and SIR comes back NaN — and SIR is
                    # the metric NFR-ML-03 is written on, because SI-SDR cannot
                    # tell suppression from distortion.
                    speakers=[
                        EvalSpeaker(
                            speaker_id=sample.target_speaker,
                            reference=sample.target,
                            active=sample.active,
                        ),
                        EvalSpeaker(speaker_id=INTERFERER, reference=sample.interferer),
                    ],
                    overlap_ratio=float(np.mean(sample.active)),
                    extra={"enrolment": sample.enrolment},
                )
            )
        return out

    @torch.no_grad()
    def system(item: EvalItem) -> list[np.ndarray]:
        mixture = torch.from_numpy(item.mixture)[None].to(args.device)
        enrolment = torch.from_numpy(item.extra["enrolment"])[None].to(args.device)
        conf = torch.ones(1, device=args.device)
        estimate = model(mixture, enrolment, conf)["estimate"]
        extracted = estimate[0].float().cpu().numpy()
        # The harness wants one estimate per listed speaker. C1 only extracts
        # the target, so the interferer's "estimate" is what is left of the
        # mixture. That row is dropped before summarising — it exists to give
        # the target row an interference reference, not to be reported.
        return [extracted, item.mixture - extracted]

    frame = evaluate(system, items(), args.out, system_name="seave-c1", heavy_metrics=args.heavy)
    targets = frame[frame["speaker"] != INTERFERER]
    metrics = ("si_sdri", "sir", "si_sdr") + (("stoi", "pesq") if args.heavy else ())
    summary = summarise(targets, metrics=metrics)
    print(format_summary(summary, metrics=metrics))

    achieved = float(targets["si_sdri"].mean())
    print(
        f"\n  C1 gate (>= {GATE_DB:.0f} dB SI-SDRi on {args.split}): "
        f"{'PASS' if achieved >= GATE_DB else 'not met'} at {achieved:+.2f} dB"
    )
    print(f"  per-item rows: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
