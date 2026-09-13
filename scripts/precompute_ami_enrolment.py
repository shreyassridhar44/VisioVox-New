"""Speaker embeddings for AMI participants, one per time window (Phase 4e).

C4 needs an enrolment cue for each AMI participant, and AMI cannot supply it
the way Libri2Mix does. There, a speaker appears in many separate utterances, so
the cue can come from a different recording entirely. Here each participant
appears in exactly one 120-second clip — one session per group, which is what
makes the split speaker-disjoint — so there is no second recording to enrol from.

The answer is the one the product itself uses. Novelty 1 is **self-enrolment**:
in production nobody uploads a reference recording, so S4 derives the cue from
the uploaded audio. C4 therefore enrols from a *different time window* of the
same participant's own microphone, and the dataset guarantees the enrolment
window never overlaps the chunk being separated. That is not a weaker
substitute for a clean enrolment — it is the deployment condition, trained on
directly.

Windows are non-overlapping and fixed, so the disjointness is checkable rather
than probabilistic, and one embedding per window keeps the whole index tiny.

Usage:
    uv run python scripts/precompute_ami_enrolment.py --split train
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

RATE = 16_000
SETS = Path.home() / "data" / "ami" / "sets"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="train", choices=("train", "eval", "holdout"))
    ap.add_argument("--windows", type=int, default=4, help="segments per participant")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)

    index_path = SETS / f"{args.split}.json"
    if not index_path.exists():
        print(f"missing {index_path}; run scripts/build_ami_eval.py --split {args.split}")
        return 2
    meetings = json.loads(index_path.read_text())

    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(Path.home() / "models" / "ecapa"),
        run_opts={"device": args.device},
    )

    ids: list[str] = []
    speakers: list[str] = []
    windows: list[int] = []
    vectors: list[np.ndarray] = []

    for meeting in meetings:
        clip_dir = SETS / args.split / meeting["meeting"]
        for participant in meeting["participants"]:
            audio, rate = sf.read(str(clip_dir / participant["reference_audio"]), dtype="float32")
            if rate != RATE:
                print(f"  {meeting['meeting']}: expected {RATE} Hz, got {rate}")
                return 2

            span = len(audio) // args.windows
            for w in range(args.windows):
                segment = audio[w * span : (w + 1) * span]
                # A window a participant is silent through would give an
                # embedding of the room, not of them. Skipped rather than
                # stored, and the dataset falls back to another window.
                if float(np.abs(segment).max()) < 1e-3:
                    continue
                with torch.no_grad():
                    emb = encoder.encode_batch(torch.from_numpy(segment)[None].to(args.device))
                vec = emb.squeeze().cpu().numpy().astype(np.float32)
                vectors.append(vec / (np.linalg.norm(vec) + 1e-9))
                ids.append(f"{meeting['meeting']}|{participant['index']}")
                speakers.append(str(participant["global_name"]))
                windows.append(w)

        print(f"  {meeting['meeting']}: {len(meeting['participants'])} participants")

    out = SETS / f"{args.split}-enrolment.npz"
    np.savez(
        out,
        ids=np.array(ids),
        speakers=np.array(speakers),
        windows=np.array(windows),
        embeddings=np.stack(vectors),
        n_windows=np.array(args.windows),
    )
    print(f"\nwrote {out}")
    print(f"  {len(vectors)} embeddings, {len(set(ids))} participants, {args.windows} windows each")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
