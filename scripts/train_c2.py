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
from typing import Protocol

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

from models.conditioning import DropoutConfig
from models.seave import Seave, SeaveConfig
from models.visual import VisualFrontend
from training.librimix_data import Libri2MixDataset, LibriMixConfig, MixItem, to_batch_dict
from training.losses import LossWeights, si_sdr
from training.realistic_data import RealisticConfig, RealisticMixDataset
from training.trainer import TrainConfig, Trainer
from training.voxceleb_mix import MixConfig, VoxCelebMixDataset

PACKED = Path.home() / "data" / "voxceleb2" / "packed"
LIBRI = Path.home() / "data" / "Libri2Mix" / "Libri2Mix" / "wav16k" / "min"
LIBRI_ENROL = Path.home() / "data" / "Libri2Mix" / "enrolment"
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
    # Absent for LibriMix micro-batches, which have no video at all; the
    # trainer reads the missing key as "no visual cue" and runs the audio-only
    # path, which is exactly the behaviour the blend is meant to exercise.
    if "mouth" in out:
        out["mouth"] = out["mouth"].unsqueeze(1)
    return out


class MixSource(Protocol):
    """What the LibriMix side of the blend has to provide.

    A protocol rather than a concrete type because the training side is a
    round-robin over several splits while the dev side is a single one, and
    both are equally valid sources of a micro-batch.
    """

    def __len__(self) -> int: ...
    def sample(self, index: int) -> MixItem: ...


class RoundRobinMix:
    """Several room-simulated splits addressed as one, by index.

    ConcatMixDataset joins splits *inside* the Libri2Mix layer, but the room
    simulation wraps each split from the outside, so the joining has to happen
    out here instead -- the same shape train_c3.py uses.
    """

    def __init__(self, datasets: list[RealisticMixDataset]) -> None:
        self.datasets = datasets
        self._len = sum(len(d) for d in datasets)

    def __len__(self) -> int:
        return self._len

    def sample(self, index: int) -> MixItem:
        d = self.datasets[index % len(self.datasets)]
        return d.sample(index // len(self.datasets) % len(d))


def build_librimix(split: str, chunk: float, dry: float = 0.15) -> RealisticMixDataset:
    """Room-simulated LibriMix, exactly as C3 built it.

    Same construction rather than a variation on it: the point of blending this
    corpus back in is to hold the distribution C3 learned, and a subtly
    different simulation would defeat that.
    """
    base = Libri2MixDataset(
        LIBRI / split, LIBRI_ENROL / f"{split}.npz", LibriMixConfig(chunk_seconds=chunk, seed=0)
    )
    return RealisticMixDataset(base, RealisticConfig(chunk_seconds=chunk, seed=0, dry_fraction=dry))


def make_libri_batches(
    ds: MixSource, indices: list[int], batch_size: int
) -> list[dict[str, torch.Tensor]]:
    """Audio-only micro-batches — no `mouth` key, so the visual path stays off.

    Blending happens at micro-batch granularity rather than within a batch.
    Mixing corpora inside one batch would need ragged collation for the ROI
    tensor, and there is no benefit to pay for it: gradients are accumulated
    across the whole step either way, so the optimiser sees both domains in
    every update regardless of how they are grouped.
    """
    out = []
    for start in range(0, len(indices), batch_size):
        group = indices[start : start + batch_size]
        if len(group) < batch_size:
            break
        out.append(collate([to_batch_dict(ds.sample(i)) for i in group]))
    return out


@torch.no_grad()
def validate_product(
    model: AudioVisualSeave,
    ds: MixSource,
    n_items: int,
    batch_size: int,
    device: torch.device,
) -> float:
    """SI-SDRi on room-simulated LibriMix — the product-domain proxy.

    This column exists because its absence cost two days. C2 v1 and v2 both
    scored above +9.7 dB on VoxCeleb2 while collapsing to +2.49 and -1.61 dB
    on the product condition, and nothing in the training log hinted at it:
    every number on screen was going up. A held-out set from the domain the
    model is *for* is the only thing that catches a corpus-specific win.
    """
    model.eval()
    scores: list[float] = []
    rng = np.random.default_rng(0)
    indices = rng.choice(len(ds), size=min(n_items, len(ds)), replace=False).tolist()
    for batch in make_libri_batches(ds, indices, batch_size):
        mixture = batch["mixture"].to(device)
        target = batch["target"].to(device)
        emb = batch["speaker_embedding"].to(device)
        conf = torch.ones(mixture.shape[0], device=device)
        est = model(mixture, emb, conf, None)["estimate"]
        scores.extend((si_sdr(est, target) - si_sdr(mixture, target)).cpu().tolist())
    model.train()
    finite = [s for s in scores if np.isfinite(s)]
    return float(np.mean(finite)) if finite else float("nan")


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
        speaker: torch.Tensor | None,
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
    with_speaker: bool = True,
) -> float:
    """Mean SI-SDRi over a fixed dev subset, with either cue optionally withheld.

    Withholding the speaker embedding is the direct measurement this stage was
    missing. `AV` minus `audio-only` is a difference between two strong numbers
    and stayed inside noise all through C2 v1; `visual-only` asks the question
    on its own terms -- given no idea what the target sounds like, can the model
    find them from the mouth? A near-zero answer there means the video is not
    being used, whatever the AV column says.
    """
    model.eval()
    scores: list[float] = []
    rng = np.random.default_rng(0)
    indices = rng.choice(len(ds), size=min(n_items, len(ds)), replace=False).tolist()
    for batch in make_batches(ds, indices, batch_size):
        mixture = batch["mixture"].to(device)
        target = batch["target"].to(device)
        emb = batch["speaker_embedding"].to(device) if with_speaker else None
        # The gate is trained to read the confidence, so withholding a cue means
        # declaring it absent as well as zeroing it.
        conf = torch.full((mixture.shape[0],), float(with_speaker), device=device)
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
    # Step B: the audio cue is withheld far more often than the 0.15 the shared
    # default uses. C2 v1 could satisfy its loss from the voice embedding alone
    # on 85% of items, so it did, and the visual path never had to work. At 0.5
    # half of all items offer no voice cue at all.
    ap.add_argument("--drop-audio-cue", type=float, default=0.5)
    # Raised alongside it to keep the audio-only fallback trained. The mutual
    # exclusion in apply_modality_dropout drops visual only when audio survives,
    # so the effective rate is this times (1 - drop_audio_cue): 0.40 * 0.5 = the
    # 0.20 the fallback had before.
    ap.add_argument("--drop-visual", type=float, default=0.40)
    # Step C.
    ap.add_argument("--confusable-prob", type=float, default=0.5)
    ap.add_argument(
        "--tir", type=float, nargs=2, default=(-8.0, 3.0),
        help="target-to-interferer ratio range; wider and lower than the default (-5, 5)",
    )  # fmt: skip
    # The fix for C2 v2. Training on VoxCeleb2 alone cost +8.25 -> -1.61 dB on
    # the product condition, monotonically with the number of steps taken; the
    # corpus has no room simulation, so the model unlearned one. Half the
    # micro-batches now come from the corpus C3 was trained on.
    ap.add_argument("--librimix-ratio", type=float, default=0.5)
    ap.add_argument("--librimix-splits", default="train-100,train-360")
    ap.add_argument("--out", type=Path, default=Path.home() / "runs" / "c2")
    ap.add_argument("--init-from", type=Path, default=Path.home() / "runs" / "c3" / "best.pt")
    ap.add_argument(
        "--init-visual", type=Path, default=Path.home() / "runs" / "sync" / "visual.pt",
        help="frontend from scripts/pretrain_sync.py; 'none' to start random",
    )  # fmt: skip
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

    partners: dict[str, list[str]] = {}
    pairs_path = PACKED / "test-confusable.json"
    if args.confusable_prob > 0:
        if not pairs_path.exists():
            print(f"missing {pairs_path}; run scripts/build_confusable_pairs.py")
            return 2
        partners = json.loads(pairs_path.read_text())

    cfg = MixConfig(
        chunk_seconds=args.chunk_seconds,
        seed=0,
        tir_db=(float(args.tir[0]), float(args.tir[1])),
        confusable_prob=args.confusable_prob,
    )
    train_ds = EnroledMixDataset(
        VoxCelebMixDataset(PACKED / "test", cfg, speakers=train_speakers, partners=partners),
        enrol,
        seed=0,
    )
    # The dev set keeps the same confusable bias: a held-out set that is easier
    # than the training set reports a number the product will not reproduce.
    dev_ds = EnroledMixDataset(
        VoxCelebMixDataset(PACKED / "test", cfg, speakers=val_speakers, partners=partners),
        enrol,
        seed=1,
    )
    print(f"train {len(train_ds):,} items / {len(train_speakers)} speakers")
    print(f"dev   {len(dev_ds):,} items / {len(val_speakers)} speakers (disjoint)")

    libri_train: MixSource | None = None
    libri_dev: RealisticMixDataset | None = None
    if args.librimix_ratio > 0:
        splits = [x.strip() for x in args.librimix_splits.split(",") if x.strip()]
        missing = [x for x in [*splits, "dev"] if not (LIBRI_ENROL / f"{x}.npz").exists()]
        if missing:
            print(f"missing LibriMix enrolment for {missing}")
            return 2
        parts = [build_librimix(x, args.chunk_seconds) for x in splits]

        libri_train = RoundRobinMix(parts)
        libri_dev = build_librimix("dev", args.chunk_seconds)
        print(f"libri {len(libri_train):,} items from {splits} + dev {len(libri_dev):,}")

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
        DropoutConfig(
            drop_visual=args.drop_visual,
            drop_audio_cue=args.drop_audio_cue,
        ),
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
    # The effective rates, not the drawn ones: apply_modality_dropout suppresses
    # a visual drop whenever the audio cue was also drawn, so the configured
    # numbers are not the ones the model sees.
    p_audio_only = args.drop_visual * (1 - args.drop_audio_cue)
    print(
        f"cues: visual-only {args.drop_audio_cue:.0%}  audio-only {p_audio_only:.0%}  "
        f"both {1 - args.drop_audio_cue - p_audio_only:.0%}   "
        f"tir={cfg.tir_db}  confusable={args.confusable_prob:.0%}"
    )
    n_libri = round(args.grad_accum * args.librimix_ratio)
    n_vox = args.grad_accum - n_libri
    print(f"corpora per step: {n_vox} VoxCeleb2 micro-batches + {n_libri} room-simulated LibriMix")

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
        # Step A: the frontend arrives knowing what a moving mouth sounds like,
        # rather than learning it from a separation gradient the audio path has
        # already satisfied. Loaded after init_from because that call only
        # touches the audio model.
        if str(args.init_visual).lower() != "none":
            if not args.init_visual.exists():
                print(f"missing {args.init_visual}; run scripts/pretrain_sync.py")
                return 2
            sync = torch.load(args.init_visual, map_location=device, weights_only=False)
            model.visual.load_state_dict(sync["visual"])
            print(
                f"initialised visual frontend from {args.init_visual} "
                f"(sync acc {sync.get('acc', float('nan')):.3f} at step {sync.get('step', -1)})"
            )
        base_ao = validate(model, dev_ds, args.val_items, args.batch, device, with_video=False)
        base_vo = validate(
            model, dev_ds, args.val_items, args.batch, device, with_video=True, with_speaker=False
        )
        print(
            f"  baselines on this dev set: audio-only {base_ao:+.2f} dB, "
            f"visual-only {base_vo:+.2f} dB"
        )
        print()

    def save(path: Path, extra: dict[str, float]) -> None:
        trainer.save(path, extra)
        state = torch.load(path, map_location="cpu", weights_only=False)
        state["visual"] = model.visual.state_dict()
        torch.save(state, path)

    t0 = time.perf_counter()
    for step in range(start_step, args.steps):
        rng = np.random.default_rng([2, step])
        picks = rng.integers(0, len(train_ds), size=args.batch * n_vox).tolist()
        batches = make_batches(train_ds, picks, args.batch)
        if libri_train is not None and n_libri > 0:
            lpicks = rng.integers(0, len(libri_train), size=args.batch * n_libri).tolist()
            batches += make_libri_batches(libri_train, lpicks, args.batch)
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
            vo = validate(
                model, dev_ds, args.val_items, args.batch, device,
                with_video=True, with_speaker=False,
            )  # fmt: skip
            product = (
                validate_product(model, libri_dev, args.val_items, args.batch, device)
                if libri_dev is not None
                else float("nan")
            )
            log.append(
                {
                    "step": float(step),
                    "val_si_sdri": av,
                    "audio_only": ao,
                    "visual_only": vo,
                    "product": product,
                }
            )
            # Selection is on the *weaker* of the two domains, not on VoxCeleb2
            # alone. C2 v2 was chosen by its VoxCeleb2 score and turned out to
            # be the worst product-condition model of the four trained; a
            # max-min criterion cannot pick a checkpoint that has collapsed on
            # either side, which is exactly the failure to rule out here.
            score = av if np.isnan(product) else min(av, product)
            marker = ""
            if score > best:
                best = score
                save(args.out / "best.pt", {"val_si_sdri": av, "product": product})
                marker = "  <- best"
            print(
                f"    AV {av:+.2f} dB   audio-only {ao:+.2f} dB   visual-only {vo:+.2f} dB   "
                f"visual worth {av - ao:+.2f} dB   product {product:+.2f} dB{marker}",
                flush=True,
            )
            (args.out / "log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
            save(args.out / "last.pt", {"val_si_sdri": best})

    trainer.write_history(args.out / "history.json")
    print(f"\n  best min(VoxCeleb2 AV, product) {best:+.2f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
