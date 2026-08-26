"""C0 smoke: overfit one batch (Phase 4a exit gate, docs/21).

The cheapest test that distinguishes "the model is training" from "the model is
wired up wrong". A network that cannot drive a single fixed batch toward zero
loss has a defect — a detached gradient, a broken mask, an output that does not
depend on its input — and no amount of data or training time will fix it. It is
worth spending two minutes on before spending two weeks.

What passing means is narrow and worth stating: the architecture can represent
and reach a solution. It says nothing about generalisation, which is what the
Libri2Mix run in 4b measures.

Usage:
    uv run python scripts/c0_overfit.py --steps 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

from models.seave import Seave, SeaveConfig
from training.losses import LossWeights, si_sdr
from training.trainer import TrainConfig, Trainer

RATE = 16_000


def synthetic_batch(
    batch: int, seconds: float, device: torch.device, seed: int = 0
) -> dict[str, torch.Tensor]:
    """A fixed batch with genuine structure to learn.

    Target and interferer are harmonic stacks at different fundamentals with
    different envelopes, so separating them is a real task rather than one
    solvable by copying the input. Each item gets its own speaker embedding so
    the conditioning path has something to key on.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = int(seconds * RATE)
    t = torch.arange(n) / RATE

    def voice(f0: float, rate: float, phase: float) -> torch.Tensor:
        sig = sum((1.0 / k) * torch.sin(2 * torch.pi * f0 * k * t + phase) for k in (1, 2, 3, 4))
        env = 0.5 + 0.5 * torch.sin(2 * torch.pi * rate * t + phase)
        return (0.25 * env * sig).float()

    targets, interferers = [], []
    for i in range(batch):
        targets.append(voice(110 + 15 * i, 2.5, 0.3 * i))
        interferers.append(voice(230 + 21 * i, 3.7, 1.1 * i))

    target = torch.stack(targets)
    interferer = torch.stack(interferers)
    mixture = target + interferer

    frames = n // 640
    energy = (target[:, : frames * 640].reshape(batch, frames, 640) ** 2).mean(-1)
    active = (energy > energy.max(dim=1, keepdim=True).values * 0.05).float()

    return {
        "mixture": mixture.to(device),
        "target": target.to(device),
        "interferer": interferer.to(device),
        "active": active.to(device),
        "speaker_embedding": torch.randn(batch, 192, generator=g).to(device),
        "audio_confidence": torch.ones(batch).to(device),
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--min-improvement-db", type=float, default=10.0)
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    # Smaller than the training config on purpose: this checks the wiring, and
    # a smaller model reaches the answer faster without changing what is tested.
    model = Seave(SeaveConfig(emb_dim=48, lstm_hidden=64, n_blocks=2))
    cfg = TrainConfig(
        steps=args.steps,
        lr=3e-4,
        grad_accum=1,
        warmup_steps=20,
        device=str(device),
        modality_dropout=False,  # a fixed batch cannot be learned through dropout
        amp=False,  # bf16 noise masks small losses at this scale
    )
    # No suppression or silence terms here: with one batch they would be
    # optimised as easily as anything else and would not add signal to the
    # wiring question this test asks.
    trainer = Trainer(model, cfg, LossWeights(suppress=0.5, silence=0.5))

    batch = synthetic_batch(args.batch, args.seconds, device)
    params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  params={params / 1e6:.2f}M  steps={args.steps}")

    with torch.no_grad():
        baseline = float(si_sdr(batch["mixture"], batch["target"]).mean())
    print(f"baseline SI-SDR (mixture vs target): {baseline:+.2f} dB\n")

    first_loss = None
    for step in range(args.steps):
        result = trainer.train_step([batch])
        if first_loss is None:
            first_loss = result.loss
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                est = model(
                    batch["mixture"], batch["speaker_embedding"], batch["audio_confidence"]
                )["estimate"]
                sdr = float(si_sdr(est, batch["target"]).mean())
            model.train()
            terms = result.terms
            print(
                f"  step {step:4d}  loss {result.loss:8.3f}  "
                f"sisdr {terms['sisdr']:7.2f}  mrstft {terms['mrstft']:6.3f}  "
                f"| est SI-SDR {sdr:+6.2f} dB"
            )

    model.eval()
    with torch.no_grad():
        est = model(batch["mixture"], batch["speaker_embedding"], batch["audio_confidence"])[
            "estimate"
        ]
        final = float(si_sdr(est, batch["target"]).mean())

    improvement = final - baseline
    last_loss = trainer.history[-1].loss
    assert first_loss is not None

    print(f"\n  loss           {first_loss:.3f} -> {last_loss:.3f}")
    print(f"  SI-SDR         {baseline:+.2f} -> {final:+.2f} dB  ({improvement:+.2f} dB)")
    grads = [float(np.mean([h.grad_norm for h in trainer.history[-20:]]))]
    print(f"  grad norm      {grads[0]:.3f} (last 20 steps)")

    passed = improvement >= args.min_improvement_db and last_loss < first_loss
    print(
        f"\n  C0 gate (>= {args.min_improvement_db:.0f} dB improvement): "
        f"{'PASS' if passed else 'FAIL'}"
    )
    if not passed:
        print("  a model that cannot overfit one batch has a wiring defect, not a data problem")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
