"""S5 (Tier 0) — blind separation baseline (docs/25 §4, ADR-0001 decision 2).

This is the baseline the project intends to beat, and the instrument for the
Phase 1 measurement that tests ADR-0001 empirically.

A full recording cannot go through the model in one pass, so it is processed in
windows. PIT-trained separators assign output channels arbitrarily per call, so
window k may emit [Alice, Bob] and window k+1 [Bob, Alice]. Stitching naively
produces a track that swaps speaker mid-sentence -- finding F1.1.

This module therefore keeps the two concerns apart:

- `separate_windows` returns per-window estimates with no stitching at all,
  which is what the permutation measurement needs to see.
- `overlap_add` stitches, given an assignment decided by the caller.

Doing the stitching inside separation would hide exactly the property we are
trying to measure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from .types import ANALYSIS_SAMPLE_RATE, StageResult, StageStatus


class Separator(Protocol):
    """Minimal surface we use from SpeechBrain, so strict typing has something real."""

    def separate_batch(self, mix: torch.Tensor) -> torch.Tensor: ...


STAGE = "S5_separate"
VERSION = "1.0.0"

# 16 kHz checkpoint. sepformer-wsj02mix is 8 kHz and silently resamples,
# which would make these numbers incomparable with everything else.
DEFAULT_MODEL = "speechbrain/sepformer-whamr16k"
# SpeechBrain expects "<kind>:<index>"; a bare "cuda" logs a parse warning
# on every call before falling back to device 0.
DEFAULT_DEVICE = "cuda:0"

WINDOW_SECONDS = 4.0
HOP_SECONDS = 2.0  # 50% overlap: every interior sample is covered twice


@dataclass
class WindowedSeparation:
    """Per-window estimates, deliberately unstitched.

    estimates: (n_windows, n_sources, window_samples)
    starts:    window start offsets in samples
    """

    estimates: np.ndarray
    starts: np.ndarray
    window_samples: int
    hop_samples: int
    sample_rate: int = ANALYSIS_SAMPLE_RATE

    @property
    def n_windows(self) -> int:
        return int(self.estimates.shape[0])

    @property
    def n_sources(self) -> int:
        return int(self.estimates.shape[1])


def load_separator(
    model_name: str = DEFAULT_MODEL, device: str = DEFAULT_DEVICE, cache: Path | None = None
) -> Separator:
    from speechbrain.inference.separation import SepformerSeparation

    savedir = str(cache) if cache is not None else None
    model: Separator = SepformerSeparation.from_hparams(
        source=model_name,
        savedir=savedir,
        run_opts={"device": device},
    )
    return model


def separate_windows(
    audio: np.ndarray,
    model: Separator,
    window_seconds: float = WINDOW_SECONDS,
    hop_seconds: float = HOP_SECONDS,
    device: str = DEFAULT_DEVICE,
) -> WindowedSeparation:
    """Run the separator over sliding windows. No stitching, no reordering."""
    win = int(window_seconds * ANALYSIS_SAMPLE_RATE)
    hop = int(hop_seconds * ANALYSIS_SAMPLE_RATE)
    n = len(audio)

    starts: list[int] = list(range(0, max(1, n - win + 1), hop))
    if starts[-1] + win < n:
        starts.append(n - win)

    out: list[np.ndarray] = []
    for s in starts:
        chunk = audio[s : s + win]
        if len(chunk) < win:  # pad the tail so every window is the same length
            chunk = np.pad(chunk, (0, win - len(chunk)))
        with torch.no_grad():
            tensor = torch.from_numpy(chunk).float().unsqueeze(0).to(device)
            est = model.separate_batch(tensor)  # (1, samples, n_src)
        out.append(est.squeeze(0).transpose(0, 1).cpu().numpy())  # (n_src, samples)

    return WindowedSeparation(
        estimates=np.stack(out),
        starts=np.asarray(starts, dtype=np.int64),
        window_samples=win,
        hop_samples=hop,
    )


def _hann(n: int) -> np.ndarray:
    # periodic Hann; with 50% hop the squared windows sum to a constant
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def overlap_add(
    sep: WindowedSeparation,
    assignment: np.ndarray,
    total_samples: int,
) -> np.ndarray:
    """Stitch windows into continuous tracks under a given channel assignment.

    `assignment[w, k] = c` means window w's output channel c belongs to
    speaker k. The caller decides that; this function only reconstructs.

    Uses a Hann window with normalisation by the accumulated envelope, so the
    result is exact regardless of hop -- verified by the reconstruction test.
    """
    n_src = sep.n_sources
    win = sep.window_samples
    acc = np.zeros((n_src, total_samples), dtype=np.float64)
    env = np.zeros(total_samples, dtype=np.float64)
    w = _hann(win)

    for i, start in enumerate(sep.starts):
        end = min(start + win, total_samples)
        span = end - start
        for k in range(n_src):
            acc[k, start:end] += sep.estimates[i, assignment[i, k], :span] * w[:span]
        env[start:end] += w[:span]

    env[env < 1e-8] = 1.0
    return (acc / env).astype(np.float32)


def identity_assignment(sep: WindowedSeparation) -> np.ndarray:
    """Naive stitching: trust the model's channel order. This is the failure case."""
    return np.tile(np.arange(sep.n_sources), (sep.n_windows, 1))


def separate(
    audio: np.ndarray,
    model: Separator,
    device: str = DEFAULT_DEVICE,
) -> tuple[WindowedSeparation, StageResult]:
    t0 = time.perf_counter()
    result = StageResult(stage=STAGE, status=StageStatus.OK)
    sep = separate_windows(audio, model, device=device)
    result.seconds = time.perf_counter() - t0
    result.detail = f"{sep.n_windows} windows x {sep.n_sources} sources"
    return sep, result
