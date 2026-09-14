"""Which speakers sound alike (Phase 4c, step C).

C2 v1 drew its interferer uniformly at random, and a uniform draw over 118
speakers is usually a pair that differs in pitch, gender and timbre. An ECAPA
embedding separates those trivially, so the voice cue alone settled the task
and the visual pathway was left with nothing to contribute -- +0.08 dB.

The fix is to make the interferer hard to tell apart *by voice*, so the only
remaining evidence about which speaker is which is which mouth is moving. This
builds the map used to do that: for every speaker, the speakers whose mean
ECAPA embedding is closest.

Cosine similarity over mean embeddings is a proxy for "same gender, similar
pitch and timbre" rather than a gender label -- VoxCeleb2's metadata is not
packed alongside the clips, and the proxy is arguably the better target anyway,
since it ranks by the confusability the separator actually experiences rather
than by a binary attribute.

Usage:
    uv run python scripts/build_confusable_pairs.py --split test --top-k 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PACKED = Path.home() / "data" / "voxceleb2" / "packed"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test")
    ap.add_argument("--top-k", type=int, default=10, help="confusable partners per speaker")
    args = ap.parse_args(argv)

    enrol = PACKED / f"{args.split}-enrolment.npz"
    if not enrol.exists():
        print(f"missing {enrol}; run scripts/precompute_vox_enrolment.py")
        return 2

    data = np.load(enrol, allow_pickle=False)
    embeddings = data["embeddings"].astype(np.float32)
    labels = [str(x) for x in data["speakers"]]

    speakers = sorted(set(labels))
    # One vector per speaker: the mean of their utterance embeddings, which is
    # the standard way to get an identity centroid out of per-utterance vectors.
    centroids = np.stack(
        [embeddings[[i for i, s in enumerate(labels) if s == spk]].mean(axis=0) for spk in speakers]
    )
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9

    similarity = centroids @ centroids.T
    np.fill_diagonal(similarity, -np.inf)  # a speaker is not their own interferer

    top_k = min(args.top_k, len(speakers) - 1)
    partners: dict[str, list[str]] = {}
    scores: list[float] = []
    for i, spk in enumerate(speakers):
        order = np.argsort(-similarity[i])[:top_k]
        partners[spk] = [speakers[j] for j in order]
        scores.append(float(similarity[i][order].mean()))

    out = PACKED / f"{args.split}-confusable.json"
    out.write_text(json.dumps(partners, indent=2), encoding="utf-8")

    off_diagonal = similarity[np.isfinite(similarity)]
    print(f"{len(speakers)} speakers, top-{top_k} partners each -> {out}")
    print(f"  mean similarity, random pair     {off_diagonal.mean():+.3f}")
    print(f"  mean similarity, chosen partners {np.mean(scores):+.3f}")
    print(
        f"  a partner is therefore {np.mean(scores) - off_diagonal.mean():+.3f} closer than chance"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
