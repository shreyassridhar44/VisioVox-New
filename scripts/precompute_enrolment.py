"""Precompute speaker embeddings for Libri2Mix enrolment (Phase 4b).

Target-speaker extraction needs a cue saying *which* speaker to keep. In C1 that
cue is an ECAPA embedding, and where it comes from decides whether the resulting
number means anything.

**The enrolment must come from a different utterance than the one being
separated.** Deriving it from the clip under test leaks the answer: the model
learns to match acoustic detail rather than speaker identity, scores far above
what it should, and collapses on real input where no such matched enrolment
exists. This script builds a per-utterance index so the dataset can sample an
enrolment from elsewhere in the same speaker's recordings.

Embeddings are computed once and cached. ECAPA on 28,000 clips is minutes of
GPU time, but doing it inside the dataloader would pay that cost every epoch and
starve the GPU that the training step needs.

Usage:
    uv run python scripts/precompute_enrolment.py --split train-100
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

RATE = 16_000
ROOT = Path.home() / "data" / "Libri2Mix" / "Libri2Mix" / "wav16k" / "min"
OUT = Path.home() / "data" / "Libri2Mix" / "enrolment"


def speaker_of(utterance_id: str) -> str:
    """LibriSpeech ids are <speaker>-<chapter>-<utterance>."""
    return utterance_id.split("-")[0]


def parse_mixture(name: str) -> tuple[str, str]:
    """A Libri2Mix filename is <utt1>_<utt2>.wav."""
    stem = name[:-4] if name.endswith(".wav") else name
    a, b = stem.split("_", 1)
    return a, b


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="train-100")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    split_dir = ROOT / args.split
    if not (split_dir / "mix_both").is_dir():
        print(f"no split at {split_dir}")
        return 2

    mixtures = sorted(p.name for p in (split_dir / "mix_both").iterdir())
    if args.limit:
        mixtures = mixtures[: args.limit]

    # Each mixture contributes two clean sources, one per speaker slot.
    jobs: list[tuple[str, str, int]] = []  # (mixture, utterance id, source index)
    for m in mixtures:
        u1, u2 = parse_mixture(m)
        jobs.append((m, u1, 1))
        jobs.append((m, u2, 2))
    print(f"{len(mixtures)} mixtures -> {len(jobs)} source clips")

    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(Path.home() / "models" / "ecapa"),
        run_opts={"device": args.device},
    )

    ids: list[str] = []
    speakers: list[str] = []
    vectors: list[np.ndarray] = []

    t0 = time.perf_counter()
    for start in range(0, len(jobs), args.batch):
        chunk = jobs[start : start + args.batch]
        waves, lengths = [], []
        for mixture, _utt, idx in chunk:
            path = split_dir / f"s{idx}" / mixture
            audio, _ = sf.read(path, dtype="float32")
            waves.append(torch.from_numpy(audio))
            lengths.append(len(audio))

        longest = max(lengths)
        padded = torch.zeros(len(waves), longest)
        for i, w in enumerate(waves):
            padded[i, : len(w)] = w
        rel = torch.tensor([n / longest for n in lengths])

        with torch.no_grad():
            emb = encoder.encode_batch(padded.to(args.device), rel.to(args.device))
        emb = emb.squeeze(1).cpu().numpy()
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9

        for (mixture, utt, idx), v in zip(chunk, emb, strict=True):
            ids.append(f"{mixture}|{idx}")
            speakers.append(speaker_of(utt))
            vectors.append(v.astype(np.float32))

        done = start + len(chunk)
        if done % (args.batch * 40) == 0:
            rate = done / (time.perf_counter() - t0)
            print(f"  {done}/{len(jobs)}  {rate:.0f} clips/s", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{args.split}.npz"
    np.savez(
        dest,
        ids=np.array(ids),
        speakers=np.array(speakers),
        embeddings=np.stack(vectors),
    )

    per_speaker: defaultdict[str, int] = defaultdict(int)
    for s in speakers:
        per_speaker[s] += 1
    singletons = sum(1 for v in per_speaker.values() if v < 2)
    print(f"\nwrote {dest}  ({len(ids)} clips, {len(per_speaker)} speakers)")
    print(
        f"  clips per speaker: min {min(per_speaker.values())}, "
        f"median {int(np.median(list(per_speaker.values())))}, max {max(per_speaker.values())}"
    )
    if singletons:
        # A speaker with one clip cannot supply an enrolment from elsewhere, so
        # the dataset must skip them rather than fall back to the same clip.
        print(f"  {singletons} speakers have a single clip and cannot be enrolled")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
