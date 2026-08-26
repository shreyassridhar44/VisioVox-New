"""Permutation-error rate on FULL-LENGTH blind separation output.

docs/21 Phase 1 exits on the permutation-error rate measured on full-length
output, and docs/02 F1.1 frames the problem in terms of a 6-minute video being
~72 windows. A 90 s clip yields only a handful of scorable windows once
single-talker regions are excluded, which is far too few to conclude anything
-- so this runs the whole meeting.

Only windows with at least two genuinely active speakers are scored. AMI is
sparsely overlapped, so most windows say nothing about channel ordering; the
count of scored windows is reported alongside the rate precisely so the
sample size is visible rather than implied.

Usage:  uv run python scripts/measure_permutation_full.py [meeting ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from eval.permutation import measure
from pipeline.s5_separate import load_separator, separate_windows

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = Path.home() / "data" / "testvideos" / "raw"
OUT = Path.home() / "data" / "baseline" / "permutation_full.json"

RATE = 16_000
N_SPEAKERS = 4


def load_meeting(meeting: str) -> tuple[np.ndarray, np.ndarray] | None:
    paths = [RAW / f"{meeting}.Headset-{i}.wav" for i in range(N_SPEAKERS)]
    if not all(p.exists() and p.stat().st_size > 100_000 for p in paths):
        return None
    tracks = [sf.read(p, dtype="float32")[0] for p in paths]
    n = min(len(t) for t in tracks)
    refs = np.stack([t[:n] for t in tracks])
    return np.sum(refs, axis=0), refs


def main(argv: list[str]) -> int:
    meetings = argv or sorted({p.name.split(".")[0] for p in RAW.glob("*.Headset-0.wav")})
    if not meetings:
        print(f"no meetings under {RAW}")
        return 2

    model = load_separator(cache=REPO_ROOT / "models" / "sepformer16k")
    results: list[dict[str, Any]] = []

    for m in meetings:
        loaded = load_meeting(m)
        if loaded is None:
            print(f"[{m}] incomplete headsets, skipping")
            continue
        mixture, refs = loaded
        minutes = len(mixture) / RATE / 60
        print(f"[{m}] {minutes:.1f} min, separating ...", flush=True)

        t0 = time.perf_counter()
        sep = separate_windows(mixture, model)
        report = measure(sep.estimates, sep.starts, refs, sep.window_samples)
        elapsed = time.perf_counter() - t0

        # Wilson-ish sanity bound so a small sample cannot masquerade as precision.
        n = max(1, report.n_scored - 1)
        se = float(
            np.sqrt(
                max(report.permutation_error_rate * (1 - report.permutation_error_rate), 1e-9) / n
            )
        )

        print(f"  windows          {report.n_windows} total, {report.n_scored} scored")
        print(f"  skipped (sparse) {report.skipped_silent}")
        print(f"  flips            {report.n_flips}")
        print(
            f"  permutation rate {report.permutation_error_rate:.1%} +/- {1.96 * se:.1%} (95% CI)"
        )
        print(f"  mean best SI-SDR {report.mean_best_si_sdr:.2f} dB")
        print(f"  verdict          {report.verdict()}")
        print(f"  elapsed          {elapsed / 60:.1f} min")

        results.append(
            {
                "meeting": m,
                "minutes": round(minutes, 2),
                "windows_total": report.n_windows,
                "windows_scored": report.n_scored,
                "windows_skipped_sparse": report.skipped_silent,
                "flips": report.n_flips,
                "permutation_error_rate": round(report.permutation_error_rate, 4),
                "ci95": round(1.96 * se, 4),
                "mean_best_si_sdr_db": round(float(report.mean_best_si_sdr), 2),
                "verdict": report.verdict(),
                "elapsed_min": round(elapsed / 60, 1),
            }
        )

    if not results:
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))

    total_scored = sum(int(r["windows_scored"]) for r in results)
    total_flips = sum(int(r["flips"]) for r in results)
    pooled = total_flips / max(1, total_scored - len(results))
    print(f"=== pooled over {len(results)} meetings ===")
    print(f"scored windows {total_scored}, flips {total_flips}")
    print(f"pooled permutation-error rate {pooled:.1%}")
    print(f"report: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
