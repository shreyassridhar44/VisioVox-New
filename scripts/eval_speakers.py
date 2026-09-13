"""How well does the model isolate a speaker, as a function of how many are talking?

Everything measured so far has been at one difficulty. Libri2Mix is always two
speakers at roughly equal level. AMI is always four headsets summed, with the
target sitting about 14 dB below the mixture. The product is neither: users
upload two or three people on one camera microphone.

That distinction is not cosmetic. C3 scores +10.92 dB on simulated two-speaker
rooms and -3.43 dB on four-speaker AMI, and from two points it is impossible to
tell whether the model fails on *real* audio or fails on *four speakers*. Those
have completely different remedies, so this sweeps the speaker count over the
same real recordings and holds everything else fixed.

Two things it fixes that the first C4 attempt got wrong:

- **Windows are chosen where the target actually speaks.** Roughly a quarter of
  random 4 s windows of an AMI participant are near-silent, and SI-SDR against
  near-silence is arbitrarily negative. Averaging those in measures nothing but
  the silence rate.
- **Audio is read once and cached.** Re-reading five 120 s files per sample cost
  63 s/step in the first attempt.

Enrolment is self-enrolment from a non-overlapping window of the target's own
microphone, which is what the product does when nobody supplies a reference.

Usage:
    uv run python scripts/eval_speakers.py --checkpoint ~/runs/c3/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

from models.seave import Seave, SeaveConfig
from pipeline.passthrough import apply, balance, decide, frame_activity
from training.losses import si_sdr

SETS = Path.home() / "data" / "ami" / "sets"
RATE = 16_000
FRAME = 640


def activity(audio: np.ndarray) -> np.ndarray:
    frames = len(audio) // FRAME
    if frames == 0:
        return np.zeros(0, dtype=np.float32)
    energy = (audio[: frames * FRAME].reshape(frames, FRAME) ** 2).mean(axis=1)
    peak = float(energy.max())
    return (energy > peak * 0.05).astype(np.float32) if peak > 0 else np.zeros(frames, np.float32)


def active_windows(audio: np.ndarray, want: int, min_active: float) -> list[int]:
    """Window starts where the target is speaking for at least `min_active`."""
    act = activity(audio)
    per_window = want // FRAME
    starts: list[int] = []
    for start_frame in range(0, max(1, len(act) - per_window), per_window // 2):
        window = act[start_frame : start_frame + per_window]
        if window.size and float(window.mean()) >= min_active:
            starts.append(start_frame * FRAME)
    return starts


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=Path.home() / "runs" / "c3" / "best.pt")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--speakers", default="2,3,4")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--min-active", type=float, default=0.35)
    ap.add_argument(
        "--min-overlap",
        type=float,
        default=0.30,
        help="each interferer must also be speaking this much of the window",
    )
    ap.add_argument("--per-count", type=int, default=150)
    ap.add_argument("--no-balance", dest="balance", action="store_false")
    ap.add_argument("--router", action="store_true", help="apply F11 passthrough routing")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = Seave(SeaveConfig()).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"checkpoint {args.checkpoint}, step {checkpoint['step']:,}\n")

    meetings = json.loads((SETS / f"{args.split}.json").read_text())
    enrol = np.load(SETS / f"{args.split}-enrolment.npz", allow_pickle=False)
    embeddings = enrol["embeddings"].astype(np.float32)
    n_windows = int(enrol["n_windows"])
    rows = {
        f"{k}|{w}": i for i, (k, w) in enumerate(zip(enrol["ids"], enrol["windows"], strict=True))
    }

    # Read every reference once. The whole split is well under a gigabyte, and
    # the first C4 attempt spent 90% of its time re-reading these files.
    cache: dict[str, list[np.ndarray]] = {}
    for meeting in meetings:
        clip = SETS / args.split / meeting["meeting"]
        raw = [
            sf.read(str(clip / p["reference_audio"]), dtype="float32")[0]
            for p in meeting["participants"]
        ]
        # AMI headset gains vary enormously between participants, so the raw
        # sum is nothing like a camera mic. Balanced on speech-active frames,
        # every speaker arrives at a comparable level — which is the condition
        # the product actually faces.
        cache[meeting["meeting"]] = balance(raw) if args.balance else raw
    print(f"cached {len(cache)} meetings\n")

    want = int(args.seconds * RATE)
    counts = [int(c) for c in args.speakers.split(",")]
    summary: dict[int, list[float]] = {}

    for n_spk in counts:
        scores: list[float] = []
        mix_levels: list[float] = []
        rng = np.random.default_rng(0)

        for meeting in meetings:
            refs = cache[meeting["meeting"]]
            full_act = [activity(r) for r in refs]
            if len(refs) < n_spk:
                continue
            for target_idx in range(len(refs)):
                target_full = refs[target_idx]
                starts = active_windows(target_full, want, args.min_active)
                if not starts:
                    continue
                # Interferers: the other participants, picked at random but
                # deterministically, so every speaker count sees comparable
                # material rather than a different easy/hard subset.
                others = [i for i in range(len(refs)) if i != target_idx]
                chosen = rng.permutation(others)[: n_spk - 1].tolist()

                # Require the interferers to be talking too. Selecting on the
                # target alone picks windows where nobody else is speaking —
                # the mixture is then already 14 dB clean and there is nothing
                # to separate, which measures passthrough damage rather than
                # isolation. "Two speakers talking simultaneously" has to mean
                # simultaneously.
                # Activity must be measured against each speaker's own peak
                # over the WHOLE clip, then sliced. Recomputing it per window
                # renormalises to that window's peak, so a silent stretch looks
                # fully active and the overlap requirement selects nothing.
                per_window = want // FRAME
                overlapping = [
                    s
                    for s in starts
                    if all(
                        float(full_act[o][s // FRAME : s // FRAME + per_window].mean())
                        >= args.min_overlap
                        for o in chosen
                    )
                ]
                if not overlapping:
                    continue

                start = int(rng.choice(overlapping))
                key = f"{meeting['meeting']}|{target_idx}"
                window_of_start = min(n_windows - 1, start // max(1, len(target_full) // n_windows))
                enrol_window = next(
                    (w for w in range(n_windows) if w != window_of_start and f"{key}|{w}" in rows),
                    None,
                )
                if enrol_window is None:
                    continue

                target = target_full[start : start + want]
                if len(target) < want:
                    continue
                mixture = target.copy()
                for other in chosen:
                    piece = refs[other][start : start + want]
                    mixture = mixture + np.pad(piece, (0, want - len(piece)))

                with torch.no_grad():
                    mix_t = torch.from_numpy(mixture.astype(np.float32))[None].to(args.device)
                    tgt_t = torch.from_numpy(target.astype(np.float32))[None].to(args.device)
                    emb = torch.from_numpy(embeddings[rows[f"{key}|{enrol_window}"]])[None]
                    est = model(mix_t, emb.to(args.device), torch.ones(1, device=args.device))
                    out = est["estimate"][0].float().cpu().numpy()
                    if args.router:
                        others_act = np.maximum.reduce(
                            [frame_activity(refs[o][start : start + want]) for o in chosen]
                        )
                        route = decide(frame_activity(target), others_act)
                        out = apply(mixture.astype(np.float32), out, route)
                    out_t = torch.from_numpy(out)[None].to(args.device)
                    improvement = float((si_sdr(out_t, tgt_t) - si_sdr(mix_t, tgt_t))[0])
                    mix_levels.append(float(si_sdr(mix_t, tgt_t)[0]))
                if np.isfinite(improvement):
                    scores.append(improvement)
                if len(scores) >= args.per_count:
                    break
            if len(scores) >= args.per_count:
                break

        summary[n_spk] = scores
        if scores:
            arr = np.array(scores)
            print(
                f"  {n_spk} speakers  n={len(arr):3d}   "
                f"mixture {np.mean(mix_levels):+6.2f} dB   "
                f"SI-SDRi mean {arr.mean():+6.2f}  median {np.median(arr):+6.2f}  "
                f"p25 {np.percentile(arr, 25):+6.2f}"
            )
        else:
            print(f"  {n_spk} speakers  no usable windows")

    print("\n  The product case is 2-3 speakers. Four is AMI's full meeting and is harder.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
