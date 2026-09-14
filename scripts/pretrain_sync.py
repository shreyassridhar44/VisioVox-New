"""Pretrain the visual frontend on audio-visual synchronisation (Phase 4c, step A).

C2 trained the frontend from random weights against a separation loss and the
video ended up worth +0.08 dB — noise. Two reasons, and this addresses the
first: 11.4M random parameters cannot learn what a moving mouth means from a
gradient that is already satisfied by the audio path.

The fix needs no labels and no new data. The packed VoxCeleb2 clips are 55
hours of mouth and audio that are, by construction, perfectly synchronised. So
the frontend is asked a question it can only answer by understanding
articulation: *does this mouth belong to this sound, at this instant?*

Formulated as InfoNCE between per-frame visual and audio embeddings. For a
frame t, the matching audio frame is the positive and every other frame is a
negative — both other frames of the same clip (temporal negatives, which force
articulation rather than identity) and frames of other clips in the batch
(content negatives). A model that only learned "this speaker looks like this
voice" would beat the cross-clip negatives and fail the within-clip ones, which
is why both are in the same denominator.

Only the visual frontend is kept. The audio encoder exists to supply a training
signal and is discarded.

Usage:
    uv run python scripts/pretrain_sync.py --steps 12000
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
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

from models.visual import VisualFrontend, resample_to

PACKED = Path.home() / "data" / "voxceleb2" / "packed" / "test"
RATE = 16_000
FPS = 25
SAMPLES_PER_FRAME = RATE // FPS  # 640, exact


class AudioEncoder(nn.Module):
    """Waveform to one embedding per video frame.

    Strided 1D convolutions rather than a spectrogram front-end: the stack
    reduces 640 samples to a single step, so the output lands on the video
    grid exactly and no resampling is needed on the audio side. Deliberately
    small — it is scaffolding, thrown away after pretraining, and a large
    audio encoder would simply solve the task on its own and teach the visual
    side nothing.
    """

    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        # 640 = 5 * 4 * 4 * 2 * 2 * 2, matching the strides below.
        channels = [1, 64, 128, 256, 256, 512]
        strides = [5, 4, 4, 2, 2]
        layers: list[nn.Module] = []
        for i, stride in enumerate(strides):
            layers += [
                nn.Conv1d(channels[i], channels[i + 1], kernel_size=stride * 2 + 1,
                          stride=stride, padding=stride),
                nn.BatchNorm1d(channels[i + 1]),
                nn.ReLU(inplace=True),
            ]  # fmt: skip
        self.net = nn.Sequential(*layers)
        self.project = nn.Linear(channels[-1], dim)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.net(audio.unsqueeze(1))  # (batch, C, frames)
        out: torch.Tensor = self.project(x.transpose(1, 2))
        return out


def info_nce(visual: torch.Tensor, audio: torch.Tensor, temperature: float) -> torch.Tensor:
    """Contrastive loss over every (clip, frame) position in the batch.

    Both sequences are flattened so a frame competes against every other frame
    present — inside its own clip and inside every other. Symmetric, because a
    one-directional loss lets one tower collapse while the other compensates.
    """
    v = nn.functional.normalize(visual.reshape(-1, visual.shape[-1]), dim=-1)
    a = nn.functional.normalize(audio.reshape(-1, audio.shape[-1]), dim=-1)
    logits = v @ a.t() / temperature
    labels = torch.arange(v.shape[0], device=v.device)
    return 0.5 * (
        nn.functional.cross_entropy(logits, labels)
        + nn.functional.cross_entropy(logits.t(), labels)
    )


@torch.no_grad()
def sync_accuracy(visual: torch.Tensor, audio: torch.Tensor) -> float:
    """Fraction of frames whose nearest audio embedding is the correct one.

    More legible than the loss: chance is 1/(batch*frames), so anything well
    above that means the frontend has learned something about articulation.
    """
    v = nn.functional.normalize(visual.reshape(-1, visual.shape[-1]), dim=-1)
    a = nn.functional.normalize(audio.reshape(-1, audio.shape[-1]), dim=-1)
    predicted = (v @ a.t()).argmax(dim=1)
    truth = torch.arange(v.shape[0], device=v.device)
    return float((predicted == truth).float().mean())


class SyncClips:
    """Random aligned (mouth, audio) windows from the packed clips."""

    def __init__(self, root: Path, frames: int, speakers: list[str] | None = None) -> None:
        self.frames = frames
        self.clips = sorted(
            p for p in root.glob("*/*.npz") if speakers is None or p.parent.name in speakers
        )
        if not self.clips:
            raise ValueError(f"no packed clips under {root}")

    def __len__(self) -> int:
        return len(self.clips)

    def sample(self, index: int) -> tuple[np.ndarray, np.ndarray] | None:
        # Seeded per index so a clip always yields the same window; not
        # cryptographic, and reproducibility is the point.
        rng = random.Random(index)  # noqa: S311
        data = np.load(self.clips[index % len(self.clips)])
        mouth, audio = data["mouth"], data["audio"]
        available = min(len(mouth), len(audio) // SAMPLES_PER_FRAME)
        if available < self.frames:
            return None
        start = rng.randint(0, available - self.frames)
        clip_mouth = mouth[start : start + self.frames].astype(np.float32) / 255.0
        a0 = start * SAMPLES_PER_FRAME
        clip_audio = audio[a0 : a0 + self.frames * SAMPLES_PER_FRAME].astype(np.float32)
        # A silent window has no articulation to align to and would teach the
        # frontend to match noise.
        if float(np.abs(clip_audio).max()) < 1e-3:
            return None
        return clip_mouth, clip_audio


def batch_of(
    ds: SyncClips, indices: list[int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor] | None:
    mouths, audios = [], []
    for i in indices:
        item = ds.sample(i)
        if item is None:
            continue
        mouths.append(item[0])
        audios.append(item[1])
    if len(mouths) < 2:
        return None
    mouth = torch.from_numpy(np.stack(mouths)).unsqueeze(1).to(device)
    audio = torch.from_numpy(np.stack(audios)).to(device)
    return mouth, audio


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--frames", type=int, default=25, help="1 s of video per item")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--holdout", type=int, default=18)
    ap.add_argument("--out", type=Path, default=Path.home() / "runs" / "sync")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    speakers = sorted(p.name for p in PACKED.iterdir() if p.is_dir())
    # Same speaker holdout as C2, so a frontend pretrained here has still never
    # seen the speakers C2 validates on.
    train_ds = SyncClips(PACKED, args.frames, speakers[args.holdout :])
    dev_ds = SyncClips(PACKED, args.frames, speakers[: args.holdout])
    print(f"train {len(train_ds):,} clips / {len(speakers) - args.holdout} speakers")
    print(f"dev   {len(dev_ds):,} clips / {args.holdout} speakers")

    visual = VisualFrontend().to(device)
    audio_net = AudioEncoder().to(device)
    opt = torch.optim.AdamW(
        [*visual.parameters(), *audio_net.parameters()], lr=args.lr, weight_decay=1e-2
    )
    chance = 1.0 / (args.batch * args.frames)
    print(f"device={device}  visual {sum(p.numel() for p in visual.parameters()) / 1e6:.1f}M")
    print(f"chance accuracy {chance:.4f}\n")

    args.out.mkdir(parents=True, exist_ok=True)
    log: list[dict[str, float]] = []
    best = 0.0
    t0 = time.perf_counter()

    for step in range(args.steps):
        rng = np.random.default_rng([9, step])
        picks = rng.integers(0, len(train_ds), size=args.batch).tolist()
        got = batch_of(train_ds, picks, device)
        if got is None:
            continue
        mouth, audio = got

        visual.train()
        audio_net.train()
        v = visual(mouth)
        a = audio_net(audio)
        a = resample_to(a, v.shape[1])
        loss = info_nce(v, a, args.temperature)

        opt.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_([*visual.parameters(), *audio_net.parameters()], 5.0)
        opt.step()

        if step % 100 == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  step {step:6d}  loss {float(loss):7.4f}  "
                f"train acc {sync_accuracy(v, a):.3f}  {elapsed / (step + 1):.2f}s/step",
                flush=True,
            )

        if (step + 1) % args.val_every == 0 or step == args.steps - 1:
            visual.eval()
            audio_net.eval()
            accs = []
            vrng = np.random.default_rng(0)
            with torch.no_grad():
                for _ in range(8):
                    vpicks = vrng.integers(0, len(dev_ds), size=args.batch).tolist()
                    vgot = batch_of(dev_ds, vpicks, device)
                    if vgot is None:
                        continue
                    vv = visual(vgot[0])
                    va = resample_to(audio_net(vgot[1]), vv.shape[1])
                    accs.append(sync_accuracy(vv, va))
            acc = float(np.mean(accs)) if accs else 0.0
            log.append({"step": float(step), "sync_acc": acc})
            marker = ""
            if acc > best:
                best = acc
                torch.save({"visual": visual.state_dict(), "step": step, "acc": acc},
                           args.out / "visual.pt")  # fmt: skip
                marker = "  <- best"
            print(
                f"    dev sync accuracy {acc:.3f}  ({acc / chance:.0f}x chance){marker}",
                flush=True,
            )
            (args.out / "log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"\n  best dev sync accuracy {best:.3f} ({best / chance:.0f}x chance)")
    print(f"  frontend written to {args.out / 'visual.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
