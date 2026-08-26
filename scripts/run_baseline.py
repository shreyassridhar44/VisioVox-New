"""Phase 1 Tier 0 baseline run (docs/21 Phase 1, docs/25 section 4).

For each AMI test clip: ingest, run blind separation in windows, measure the
permutation-error rate against the per-speaker headset references, and compare
two ways of stitching the windows back together.

The naive-vs-oracle comparison is the point. Both use identical separator
output; they differ only in how windows are assigned to speakers:

    naive   trust the channel order the model emits (what a real system gets)
    oracle  best per-window assignment, using the
            references as an answer key             (unreachable upper bound)

The gap between them is the cost of permutation ambiguity alone, isolated from
separation quality. If the gap is small, F1.1 is overstated and ADR-0001 should
be revisited. If it is large, TSE is justified on evidence rather than argument.

Usage:  uv run python scripts/run_baseline.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.optimize import linear_sum_assignment

from eval.permutation import measure, si_sdr
from pipeline.s0_ingest import ingest
from pipeline.s5_separate import (
    Separator,
    identity_assignment,
    load_separator,
    overlap_add,
    separate_windows,
)
from pipeline.vad import speech_masks

REPO_ROOT = Path(__file__).resolve().parents[1]
CLIPS = Path.home() / "data" / "testvideos" / "clips"
WORK = Path.home() / "data" / "baseline"
REPORT_JSON = WORK / "baseline_report.json"

RATE = 16_000


@dataclass
class ClipResult:
    clip: str
    duration_s: float
    n_windows: int
    n_reference_speakers: int
    permutation_error_rate: float
    permutation_flips: int
    windows_scored: int
    windows_skipped_silent: int
    si_sdr_mixture_db: float
    si_sdr_naive_db: float
    si_sdr_oracle_db: float
    si_sdri_naive_db: float
    si_sdri_oracle_db: float
    si_sdr_gap_db: float
    seconds_elapsed: float
    verdict: str


def load_refs(clip_dir: Path) -> np.ndarray:
    refs = sorted(clip_dir.glob("ref_spk*.wav"))
    if not refs:
        raise FileNotFoundError(f"no references in {clip_dir}")
    loaded = [sf.read(p, dtype="float32")[0] for p in refs]
    n = min(len(x) for x in loaded)
    return np.stack([x[:n] for x in loaded])


def _cost_matrix(estimates_w: np.ndarray, ref_win: np.ndarray) -> np.ndarray:
    n_src, n_ref = estimates_w.shape[0], ref_win.shape[0]
    cost = np.zeros((n_src, n_ref))
    for c in range(n_src):
        for r in range(n_ref):
            sc = si_sdr(estimates_w[c], ref_win[r])
            cost[c, r] = -sc if np.isfinite(sc) else 1e6
    return cost


def select_targets(references: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k speakers who talk most, fixed for the whole clip.

    The separator emits fewer channels than an AMI meeting has participants,
    so some speakers cannot be recovered at all. Choosing the target set once,
    globally, keeps output track k meaning the same person in every window --
    which is the property naive stitching is supposed to provide and fails to.
    Choosing it per window would quietly grade the model on a moving target.
    """
    masks = speech_masks(references)
    activity = masks.mean(axis=1)
    return np.argsort(activity)[::-1][:k].copy()


def oracle_assignment(
    estimates: np.ndarray, starts: np.ndarray, targets: np.ndarray, win: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-window channel->track assignment chosen with the references as key.

    Not achievable at inference time -- that is the point. It bounds what any
    perfect stitcher could do with this separator output.

    Returns assignment[w, k] = channel index feeding output track k, so it is
    a permutation of the channels for every window. An earlier version indexed
    the output by reference id and dropped matches beyond the channel count,
    leaving both tracks pointing at channel 0 and scoring worse than naive.
    """
    n_windows, n_src, _ = estimates.shape
    out = np.tile(np.arange(n_src), (n_windows, 1))
    gains = np.ones((n_windows, n_src), dtype=np.float64)
    for w in range(n_windows):
        s = int(starts[w])
        ref_win = targets[:, s : s + win]
        if ref_win.shape[1] < win:
            ref_win = np.pad(ref_win, ((0, 0), (0, win - ref_win.shape[1])))
        rows, cols = linear_sum_assignment(_cost_matrix(estimates[w], ref_win))
        for channel, track in zip(rows.tolist(), cols.tolist(), strict=True):
            out[w, track] = channel
            # Least-squares projection onto the reference. This corrects scale
            # AND polarity, which the separator does not keep consistent across
            # independent windows; without it, correctly identified windows
            # still cancel each other in the overlap region.
            est = estimates[w, channel]
            denom = float(np.dot(est, est))
            gains[w, track] = float(np.dot(est, ref_win[track])) / denom if denom > 1e-12 else 0.0
    return out, gains


def score_tracks(tracks: np.ndarray, targets: np.ndarray) -> float:
    """Mean SI-SDR of track k against target speaker k.

    Deliberately positional, not Hungarian. Re-matching here would let each
    stitching strategy pick whichever reference flatters it, and would hide
    exactly the identity errors this baseline exists to expose.
    """
    n = min(tracks.shape[1], targets.shape[1])
    k = min(tracks.shape[0], targets.shape[0])
    scores = [si_sdr(tracks[i, :n], targets[i, :n]) for i in range(k)]
    finite = [s for s in scores if np.isfinite(s)]
    return float(np.mean(finite)) if finite else float("nan")


def run_clip(clip_dir: Path, model: Separator) -> ClipResult | None:
    name = clip_dir.name
    src = clip_dir / "input.mp4"
    if not src.exists():
        print(f"  {name}: no input.mp4, skipping")
        return None

    t0 = time.perf_counter()
    media, _ = ingest(src, WORK / name)
    mixture, _ = sf.read(media.analysis_wav, dtype="float32")
    references = load_refs(clip_dir)

    n = min(len(mixture), references.shape[1])
    mixture, references = mixture[:n], references[:, :n]

    sep = separate_windows(mixture, model)
    win = sep.window_samples

    target_idx = select_targets(references, sep.n_sources)
    targets = references[target_idx]

    report = measure(sep.estimates, sep.starts, references, win)

    naive = overlap_add(sep, identity_assignment(sep), n)
    oracle_asg, oracle_gains = oracle_assignment(sep.estimates, sep.starts, targets, win)
    oracle = overlap_add(sep, oracle_asg, n, gains=oracle_gains)

    naive_db = score_tracks(naive, targets)
    oracle_db = score_tracks(oracle, targets)
    # SI-SDRi is improvement over doing nothing: feeding the mixture itself
    # as every estimate. Absolute SI-SDR is not comparable across clips
    # because it depends on how much of the mixture each speaker occupies,
    # and docs/25 section 4 states the Tier 0 expectation as SI-SDRi.
    mixture_tracks = np.tile(mixture[None, :], (naive.shape[0], 1))
    mixture_db = score_tracks(mixture_tracks, targets)

    out = clip_dir / "baseline"
    out.mkdir(exist_ok=True)
    for i in range(naive.shape[0]):
        sf.write(out / f"naive_track{i}.wav", naive[i], RATE)
        sf.write(out / f"oracle_track{i}.wav", oracle[i], RATE)

    return ClipResult(
        clip=name,
        duration_s=round(n / RATE, 2),
        n_windows=sep.n_windows,
        n_reference_speakers=int(references.shape[0]),
        permutation_error_rate=round(report.permutation_error_rate, 4),
        permutation_flips=report.n_flips,
        windows_scored=report.n_scored,
        windows_skipped_silent=report.skipped_silent,
        si_sdr_mixture_db=round(mixture_db, 2),
        si_sdr_naive_db=round(naive_db, 2),
        si_sdr_oracle_db=round(oracle_db, 2),
        si_sdri_naive_db=round(naive_db - mixture_db, 2),
        si_sdri_oracle_db=round(oracle_db - mixture_db, 2),
        si_sdr_gap_db=round(oracle_db - naive_db, 2),
        seconds_elapsed=round(time.perf_counter() - t0, 1),
        verdict=report.verdict(),
    )


def main() -> int:
    if not CLIPS.exists():
        print(f"no clips at {CLIPS}; run scripts/fetch_testvideos.py first")
        return 2
    dirs = sorted(d for d in CLIPS.iterdir() if d.is_dir())
    if not dirs:
        print("no clip directories found")
        return 2

    WORK.mkdir(parents=True, exist_ok=True)
    print("loading separator (16 kHz checkpoint)")
    model = load_separator(cache=REPO_ROOT / "models" / "sepformer16k")

    results: list[ClipResult] = []
    for d in dirs:
        print(f"[{d.name}]")
        r = run_clip(d, model)
        if r is None:
            continue
        results.append(r)
        print(f"  windows            {r.n_windows} ({r.windows_scored} scored)")
        print(f"  permutation error  {r.permutation_error_rate:.1%} ({r.permutation_flips} flips)")
        print(f"  SI-SDR mixture     {r.si_sdr_mixture_db:6.2f} dB  (do nothing)")
        print(
            f"  SI-SDR naive       {r.si_sdr_naive_db:6.2f} dB  -> SI-SDRi {r.si_sdri_naive_db:+.2f}"
        )
        print(
            f"  SI-SDR oracle      {r.si_sdr_oracle_db:6.2f} dB  -> SI-SDRi {r.si_sdri_oracle_db:+.2f}"
        )
        print(f"  cost of ambiguity  {r.si_sdr_gap_db:6.2f} dB")
        print(f"  verdict            {r.verdict}")

    if not results:
        print("no clips produced results")
        return 1

    REPORT_JSON.write_text(json.dumps([asdict(r) for r in results], indent=2))
    mean_per = float(np.mean([r.permutation_error_rate for r in results]))
    mean_gap = float(np.mean([r.si_sdr_gap_db for r in results]))
    print(f"=== {len(results)} clips ===")
    print(f"mean permutation-error rate  {mean_per:.1%}")
    print(f"mean cost of ambiguity       {mean_gap:.2f} dB")
    print(f"report: {REPORT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
