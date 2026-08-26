"""Speed-augment the WHAM! training noise for the 16 kHz LibriMix config.

Replaces LibriMix's `scripts/augment_train_noise.py`, which depends on
`pysndfx`. That package calls `numpy.fromstring` in binary mode, removed in
NumPy 2, so the upstream script aborts partway through with a ValueError after
the corpus has already been extracted.

The effect being replaced is sox's `speed`: resample the signal and play it
back at the original rate, which shifts tempo and pitch together. `soxr` is
already a dependency and does the resampling at higher quality than sox's
default, so this removes an unmaintained package rather than pinning around it.

Naming matches upstream — `foo.wav` produces `foosp09.wav` and `foosp11.wav` —
because the LibriMix metadata refers to those filenames.

Usage:
    uv run python scripts/augment_wham.py --wham-dir ~/data/corpora/wham_noise
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

SPEEDS = (0.9, 1.1)
EXPECTED_ORIGINALS = 20_000
EXPECTED_TOTAL = 60_000


def _augmented_name(path: Path, speed: float) -> Path:
    suffix = f"sp{str(speed).replace('.', '')}"
    return path.with_name(path.stem + suffix + path.suffix)


def augment_one(path_str: str) -> int:
    """Write both speed variants for one file. Returns how many were created."""
    path = Path(path_str)
    written = 0
    try:
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        return 0
    mono = audio[:, 0]

    for speed in SPEEDS:
        dest = _augmented_name(path, speed)
        if dest.exists():
            continue
        # Resample to rate/speed, then declare the original rate on write: the
        # samples are reinterpreted faster or slower, which is what sox `speed`
        # does. Resampling to a rounded rate and writing that instead would
        # change duration without changing pitch, a different effect entirely.
        resampled = soxr.resample(mono, rate, rate / speed)
        sf.write(dest, np.asarray(resampled, dtype=np.float32), rate)
        written += 1
    return written


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wham-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    train_dir = args.wham_dir / "tr"
    if not train_dir.is_dir():
        print(f"no training noise at {train_dir}")
        return 2

    every = sorted(str(p) for p in train_dir.rglob("*.wav"))
    originals = [p for p in every if "sp" not in Path(p).stem]

    if len(every) >= EXPECTED_TOTAL:
        print(f"already augmented ({len(every)} files); nothing to do")
        return 0
    if not originals:
        print("no original noise files found")
        return 1
    if len(originals) != EXPECTED_ORIGINALS:
        # Not fatal: the count differs between WHAM! releases, and refusing
        # here would block on a cosmetic mismatch.
        print(f"note: {len(originals)} originals, expected {EXPECTED_ORIGINALS}")

    print(f"augmenting {len(originals)} files at speeds {SPEEDS} ...", flush=True)
    created = 0
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(augment_one, p): p for p in originals}
        for fut in as_completed(futures):
            created += fut.result()
            done += 1
            if done % 2000 == 0:
                print(f"  {done}/{len(originals)} files, {created} written", flush=True)

    total = len(list(train_dir.rglob("*.wav")))
    print(f"wrote {created} new files; {total} total in {train_dir}")
    return 0 if total >= EXPECTED_TOTAL or created > 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
