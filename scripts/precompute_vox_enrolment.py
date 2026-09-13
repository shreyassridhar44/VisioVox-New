"""Speaker embeddings for the packed VoxCeleb2 clips (Phase 4c).

C2 conditions on two cues at once: an ECAPA embedding of the target's voice and
their mouth ROIs. The packed `.npz` files carry mouth and audio but no
embedding, so this builds the index the way `precompute_enrolment.py` does for
Libri2Mix — one vector per utterance, so the dataset can draw an enrolment from
a *different* clip by the same speaker.

That last point is the whole reason this is a separate pass. Deriving the cue
from the clip being separated leaks the answer: the model learns to match
acoustic detail instead of speaker identity, scores far above what it should,
and collapses on real input where no matched enrolment exists.

Usage:
    uv run python scripts/precompute_vox_enrolment.py --split test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PACKED = Path.home() / "data" / "voxceleb2" / "packed"
RATE = 16_000


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    split_dir = PACKED / args.split
    if not split_dir.is_dir():
        print(f"no packed split at {split_dir}; run scripts/prepare_voxceleb2.py first")
        return 2

    clips = sorted(split_dir.glob("*/*.npz"))
    if args.limit:
        clips = clips[: args.limit]
    speakers = sorted({p.parent.name for p in clips})
    print(f"{len(clips):,} clips across {len(speakers)} speakers")

    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(Path.home() / "models" / "ecapa"),
        run_opts={"device": args.device},
    )

    ids: list[str] = []
    labels: list[str] = []
    vectors: list[np.ndarray] = []
    t0 = time.perf_counter()

    for start in range(0, len(clips), args.batch):
        chunk = clips[start : start + args.batch]
        waves = []
        keep = []
        for path in chunk:
            audio = np.load(path)["audio"].astype(np.float32)
            # A clip with no energy gives an embedding of the noise floor, not
            # of the speaker; excluded rather than stored.
            if float(np.abs(audio).max()) < 1e-3:
                continue
            waves.append(torch.from_numpy(audio))
            keep.append(path)
        if not waves:
            continue

        longest = max(len(w) for w in waves)
        padded = torch.zeros(len(waves), longest)
        rel = torch.zeros(len(waves))
        for i, wave in enumerate(waves):
            padded[i, : len(wave)] = wave
            rel[i] = len(wave) / longest

        with torch.no_grad():
            emb = encoder.encode_batch(padded.to(args.device), rel.to(args.device))
        emb = emb.squeeze(1).cpu().numpy().astype(np.float32)

        for path, vector in zip(keep, emb, strict=True):
            norm = float(np.linalg.norm(vector))
            vectors.append(vector / norm if norm > 1e-9 else vector)
            ids.append(f"{path.parent.name}/{path.stem}")
            labels.append(path.parent.name)

        done = start + len(chunk)
        if done % (args.batch * 40) == 0:
            rate = done / (time.perf_counter() - t0)
            print(f"  {done}/{len(clips)}  {rate:.0f} clips/s", flush=True)

    out = PACKED / f"{args.split}-enrolment.npz"
    np.savez(out, ids=np.array(ids), speakers=np.array(labels), embeddings=np.stack(vectors))
    per_speaker = np.bincount(np.unique(labels, return_inverse=True)[1])
    print(f"\nwrote {out}")
    print(f"  {len(vectors):,} embeddings, {len(set(labels))} speakers")
    print(
        f"  clips per speaker: min {per_speaker.min()}, "
        f"median {int(np.median(per_speaker))}, max {per_speaker.max()}"
    )
    usable = int((per_speaker >= 2).sum())
    print(f"  {usable} speakers have >=2 clips and can therefore be enrolled leak-free")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
