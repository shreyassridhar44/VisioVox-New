"""Measure dataloader throughput and GPU utilisation (Phase 3 exit criterion).

The exit gate is "dataloaders produce correct batches at >= 85% GPU
utilisation". That is a statement about the *loader*, not the model: on this
hardware an A5000 finishes a batch before one Python process can build the
next, and the symptom is a training run that takes twice as long for no visible
reason. Measuring it before any real model exists is the point — a loader
bottleneck found now is a config change, found in Phase 4 it is a week of
confused profiling.

GPU utilisation is sampled from nvidia-smi in a background thread while a
deliberately compute-heavy step runs, so the number reflects whether the loader
can keep the device fed.

Usage:
    uv run python scripts/benchmark_dataloader.py --synthetic
    uv run python scripts/benchmark_dataloader.py --packed ~/data/voxceleb2/packed/test
"""

from __future__ import annotations

import argparse
import contextlib
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

FPS = 25
RATE = 16_000
MOUTH = 96


def make_synthetic(root: Path, speakers: int = 12, clips: int = 8, seconds: float = 6.0) -> None:
    """Packed clips with the same shape and dtype as the real thing.

    Same sizes matter: the benchmark is measuring decompression and memory
    bandwidth, so smaller stand-ins would report a throughput the real corpus
    never reaches.
    """
    frames = int(seconds * FPS)
    for s in range(speakers):
        d = root / f"id{s:05d}"
        d.mkdir(parents=True, exist_ok=True)
        for c in range(clips):
            rng = np.random.default_rng(s * 100 + c)
            t = np.arange(int(seconds * RATE)) / RATE
            audio = (0.3 * np.sin(2 * np.pi * (150 + 20 * s) * t)).astype(np.float32)
            mouth = rng.integers(0, 255, (frames, MOUTH, MOUTH), dtype=np.uint8)
            np.savez(d / f"c{c}.npz", mouth=mouth, audio=audio)


class GpuSampler(threading.Thread):
    """Polls nvidia-smi so utilisation is measured, not inferred from timings."""

    def __init__(self, interval: float = 0.2) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[float] = []
        # not `_stop`: that name shadows Thread._stop(), which join() calls.
        self._halt = threading.Event()
        self._smi = shutil.which("nvidia-smi")

    def run(self) -> None:
        if self._smi is None:
            return
        while not self._halt.wait(self.interval):
            proc = subprocess.run(  # noqa: S603 - resolved path, fixed argv
                [self._smi, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                check=False,
                capture_output=True,
                text=True,
            )
            line = proc.stdout.strip().splitlines()
            if line:
                with contextlib.suppress(ValueError):
                    self.samples.append(float(line[0]))

    def stop(self) -> tuple[float, int]:
        self._halt.set()
        self.join(timeout=3)
        if not self.samples:
            return float("nan"), 0
        return float(np.mean(self.samples)), len(self.samples)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packed", type=Path, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--depth", type=int, default=16, help="stand-in model depth")
    args = ap.parse_args(argv)

    import torch

    from training.dataloader import LoaderConfig, build_loader
    from training.voxceleb_mix import MixConfig

    tmp: tempfile.TemporaryDirectory[str] | None = None
    if args.synthetic or args.packed is None:
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        print("building synthetic packed clips ...", flush=True)
        make_synthetic(root)
    else:
        root = args.packed
        if not root.is_dir():
            print(f"no packed data at {root}; use --synthetic")
            return 2

    loader = build_loader(
        root,
        MixConfig(chunk_seconds=4.0, seed=0),
        LoaderConfig(batch_size=args.batch_size, num_workers=args.workers),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  workers={args.workers}  batch={args.batch_size}")

    # A stand-in for the extractor: heavy enough that a well-fed GPU stays busy,
    # so the number reported is about the loader rather than about an idle step.
    # Sized like a real separation network rather than a toy. A four-layer
    # stand-in finishes a batch in 19 ms, which no loader can keep up with and
    # which no actual model resembles; benchmarking against it reports a
    # dataloader failure that does not exist.
    layers: list[torch.nn.Module] = [torch.nn.Conv1d(1, 256, 33, stride=4, padding=16)]
    for _ in range(args.depth):
        layers += [
            torch.nn.Conv1d(256, 256, 15, padding=7, groups=4),
            torch.nn.GroupNorm(8, 256),
            torch.nn.PReLU(),
        ]
    layers += [torch.nn.Conv1d(256, 1, 33, padding=16)]
    net = torch.nn.Sequential(*layers).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)

    it = iter(loader)
    for _ in range(3):  # warm up workers and CUDA context before timing
        batch = next(it)
        x = batch["mixture"].unsqueeze(1).to(device, non_blocking=True)
        loss = net(x).mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()

    sampler = GpuSampler()
    sampler.start()
    t0 = time.perf_counter()
    seen = 0
    shapes: dict[str, tuple[int, ...]] = {}
    for _ in range(args.batches):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        if not shapes:
            shapes = {k: tuple(v.shape) for k, v in batch.items()}
        x = batch["mixture"].unsqueeze(1).to(device, non_blocking=True)
        _ = batch["mouth"].to(device, non_blocking=True)
        loss = net(x).mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        seen += x.shape[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    util, n_samples = sampler.stop()

    print("\nbatch shapes:")
    for k, v in shapes.items():
        print(f"  {k:<11} {v}")
    print(f"\nitems/s        {seen / elapsed:.1f}")
    print(f"batches/s      {args.batches / elapsed:.2f}")
    print(f"gpu util       {util:.1f}%  ({n_samples} samples)")
    gate = 85.0
    if device.type != "cuda":
        print("\nno CUDA device; utilisation gate not evaluated")
    elif np.isnan(util):
        print("\nnvidia-smi unavailable; utilisation gate not evaluated")
    else:
        print(f"\nexit gate >= {gate:.0f}%: {'PASS' if util >= gate else 'FAIL'}")

    if tmp is not None:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
