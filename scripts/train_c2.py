"""C2 — add the visual pathway (Phase 4c, docs/07 §1).

The project's premise, and the last untrained part of the architecture. C1 and
C3 conditioned on a voice embedding alone; SEAVE was always built to take mouth
ROIs as well, through the reliability-gated fusion and the per-block cross
attention. This is the stage that trains them.

The gate is >= +1.5 dB over the audio-only model on same-gender pairs, which is
the right place to look: same-gender overlap is where a voice embedding is
weakest and where seeing which mouth is moving should help most. A gain
averaged over all pairs would hide that.

**Modality dropout is on.** Without it the model learns to lean on the video
and collapses when a face is missing or turned away — and docs/05 §8 requires
the pipeline to degrade to audio-only rather than fail. Dropout during training
is what makes `beta_t` meaningful at inference instead of decorative.

Initialised from C3, so the audio path already separates and the visual path is
learning to add to a working model rather than co-adapting from noise.

Honest about the data: 118 speakers, which is thin. C1 overfit 251. Expect this
to demonstrate the visual cue works and measure what it is worth, not to be
production-strength.

Usage:
    uv run python scripts/train_c2.py --init-from ~/runs/c3/best.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

from models.seave import Seave, SeaveConfig
from models.visual import VisualFrontend
from training.losses import LossWeights, si_sdr
from training.trainer import TrainConfig, Trainer
from training.voxceleb_mix import MixConfig, VoxCelebMixDataset

PACKED = Path.home() / "data" / "voxceleb2" / "packed"
GATE_DB = 1.5  # over the audio-only baseline, on same-gender pairs


class EnroledMixDataset:
    """VoxCeleb2 mixtures with an enrolment embedding attached.

    The embedding always comes from a *different* clip by the target speaker.
    Taking it from the clip being separated leaks the answer — the model learns
    to match acoustic detail rather than identity and collapses on real input.
    """

    def __init__(self, inner: VoxCelebMixDataset, enrolment_npz: Path, seed: int = 0) -> None:
        self.inner = inner
        data = np.load(enrolment_npz, allow_pickle=False)
        self.embeddings = data["embeddings"].astype(np.float32)
        ids = [str(x) for x in data["ids"]]
        labels = [str(x) for x in data["speakers"]]
        self.rows_by_speaker: dict[str, list[int]] = {}
        for row, speaker in enumerate(labels):
            self.rows_by_speaker.setdefault(speaker, []).append(row)
        self.row_of = {clip_id: row for row, clip_id in enumerate(ids)}
        self.seed = seed

    def __len__(self) -> int:
        return len(self.inner)

    def sample(self, index: int) -> dict[str, np.ndarray]:
        s = self.inner.sample(index)
        # Seeded per index, not cryptographic: the same item must draw the
        # same enrolment on every run, including after a resume.
        rng = random.Random(self.seed + index)  # noqa: S311
        rows = self.rows_by_speaker.get(s.target_speaker, [])
        if not rows:
            raise KeyError(f"no enrolment for {s.target_speaker}")
        vector = self.embeddings[rng.choice(rows)]
        return {
            "mixture": s.mixture,
            "target": s.target,
            "interferer": s.interferer,
            "active": s.active.astype(np.float32),
            "speaker_embedding": vector,
            "mouth": (s.mouth.astype(np.float32) / 255.0),
        }


def collate(items: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    out = {k: torch.from_numpy(np.stack([i[k] for i in items])) for k in items[0]}
    # (batch, frames, H, W) -> (batch, 1, frames, H, W) for the 3D stem.
    out["mouth"] = out["mouth"].unsqueeze(1)
    return out


def make_batches(
    ds: EnroledMixDataset, indices: list[int], batch_size: int
) -> list[dict[str, torch.Tensor]]:
    out = []
    for start in range(0, len(indices), batch_size):
        group = indices[start : start + batch_size]
        if len(group) < batch_size:
            break
        out.append(collate([ds.sample(i) for i in group]))
    return out


class AudioVisualSeave(torch.nn.Module):
    """SEAVE with the visual frontend attached.

    Kept as a wrapper rather than folded into `Seave` so every audio-only
    checkpoint stays loadable by the audio-only class, and so C1/C3 evaluation
    keeps working untouched.
    """

    def __init__(self, cfg: SeaveConfig | None = None) -> None:
        super().__init__()
        self.seave = Seave(cfg)
        self.visual = VisualFrontend()

    def forward(
        self,
        mixture: torch.Tensor,
        speaker: torch.Tensor,
        audio_conf: torch.Tensor,
        mouth: torch.Tensor | None = None,
        visual_conf: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Seave resizes the visual sequence onto the STFT grid itself, so the
        # frontend's native 25 fps output goes straight through.
        features = self.visual(mouth) if mouth is not None else None
        out: dict[str, torch.Tensor] = self.seave(
            mixture, speaker, audio_conf, features, visual_conf
        )
        return out


@torch.no_grad()
def validate(
    model: AudioVisualSeave,
    ds: EnroledMixDataset,
    n_items: int,
    batch_size: int,
    device: torch.device,
    *,
    with_video: bool,
) -> float:
    model.eval()
    scores: list[float] = []
    rng = np.random.default_rng(0)
    indices = rng.choice(len(ds), size=min(n_items, len(ds)), replace=False).tolist()
    for batch in make_batches(ds, indices, batch_size):
        mixture = batch["mixture"].to(device)
        target = batch["target"].to(device)
        emb = batch["speaker_embedding"].to(device)
        conf = torch.ones(mixture.shape[0], device=device)
        mouth = batch["mouth"].to(device) if with_video else None
        est = model(mixture, emb, conf, mouth)["estimate"]
        scores.extend((si_sdr(est, target) - si_sdr(mixture, target)).cpu().tolist())
    model.train()
    finite = [s for s in scores if np.isfinite(s)]
    return float(np.mean(finite)) if finite else float("nan")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=4, help="video is memory-hungry")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-items", type=int, default=120)
    ap.add_argument("--chunk-seconds", type=float, default=4.0)
    ap.add_argument("--holdout", type=int, default=18, help="speakers reserved for validation")
    ap.add_argument("--out", type=Path, default=Path.home() / "runs" / "c2")
    ap.add_argument("--init-from", type=Path, default=Path.home() / "runs" / "c3" / "best.pt")
    ap.add_argument("--resume", nargs="?", const="auto", default=None)
    args = ap.parse_args(argv)

    enrol = PACKED / "test-enrolment.npz"
    if not enrol.exists():
        print(f"missing {enrol}; run scripts/precompute_vox_enrolment.py")
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    speakers = sorted(p.name for p in (PACKED / "test").iterdir() if p.is_dir())
    # Held out by speaker, not by clip: validating on unseen clips of seen
    # speakers measures memorisation of voices, which is not the question.
    val_speakers = speakers[: args.holdout]
    train_speakers = speakers[args.holdout :]

    cfg = MixConfig(chunk_seconds=args.chunk_seconds, seed=0)
    train_ds = EnroledMixDataset(
        VoxCelebMixDataset(PACKED / "test", cfg, speakers=train_speakers), enrol, seed=0
    )
    dev_ds = EnroledMixDataset(
        VoxCelebMixDataset(PACKED / "test", cfg, speakers=val_speakers), enrol, seed=1
    )
    print(f"train {len(train_ds):,} items / {len(train_speakers)} speakers")
    print(f"dev   {len(dev_ds):,} items / {len(val_speakers)} speakers (disjoint)")

    model = AudioVisualSeave(SeaveConfig()).to(device)
    trainer = Trainer(
        model.seave,
        TrainConfig(
            steps=args.steps,
            lr=args.lr,
            grad_accum=args.grad_accum,
            warmup_steps=min(500, args.steps // 10),
            device=str(device),
            modality_dropout=True,  # so the model still works with no face
            out_dir=args.out,
        ),
        LossWeights(),
    )
    # The trainer optimises the audio model; the frontend needs to be in the
    # same optimiser or it never learns.
    trainer.opt.add_param_group({"params": model.visual.parameters(), "lr": args.lr})
    trainer.visual_encoder = model.visual
    audio_params = sum(p.numel() for p in model.seave.parameters())
    visual_params = sum(p.numel() for p in model.visual.parameters())
    print(
        f"device={device}  audio {audio_params / 1e6:.1f}M + visual {visual_params / 1e6:.1f}M  "
        f"batch={args.batch}x{args.grad_accum}  steps={args.steps}"
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
        state = torch.load(path, map_location=device, weights_only=False)
        if "visual" in state:
            model.visual.load_state_dict(state["visual"])
        print(f"resumed {path.name} at step {start_step}, best {best:+.2f} dB\n")
    elif args.init_from is not None:
        skipped = trainer.init_from(args.init_from)
        print(f"initialised audio path from {args.init_from}, {len(skipped)} left random")
        audio_only = validate(model, dev_ds, args.val_items, args.batch, device, with_video=False)
        print(f"  audio-only baseline on this dev set: {audio_only:+.2f} dB\n")

    def save(path: Path, extra: dict[str, float]) -> None:
        trainer.save(path, extra)
        state = torch.load(path, map_location="cpu", weights_only=False)
        state["visual"] = model.visual.state_dict()
        torch.save(state, path)

    t0 = time.perf_counter()
    for step in range(start_step, args.steps):
        rng = np.random.default_rng([2, step])
        picks = rng.integers(0, len(train_ds), size=args.batch * args.grad_accum).tolist()
        batches = make_batches(train_ds, picks, args.batch)
        result = trainer.train_step(batches)

        if step % 50 == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  step {step:6d}  loss {result.loss:8.3f}  sisdr {result.terms['sisdr']:7.2f}  "
                f"lr {result.lr:.2e}  {elapsed / (step - start_step + 1):.2f}s/step",
                flush=True,
            )

        if (step + 1) % args.val_every == 0 or step == args.steps - 1:
            av = validate(model, dev_ds, args.val_items, args.batch, device, with_video=True)
            ao = validate(model, dev_ds, args.val_items, args.batch, device, with_video=False)
            log.append({"step": float(step), "val_si_sdri": av, "audio_only": ao})
            marker = ""
            if av > best:
                best = av
                save(args.out / "best.pt", {"val_si_sdri": best})
                marker = "  <- best"
            print(
                f"    AV {av:+.2f} dB   audio-only {ao:+.2f} dB   "
                f"visual worth {av - ao:+.2f} dB{marker}",
                flush=True,
            )
            (args.out / "log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
            save(args.out / "last.pt", {"val_si_sdri": best})

    trainer.write_history(args.out / "history.json")
    print(f"\n  best audio-visual SI-SDRi {best:+.2f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
