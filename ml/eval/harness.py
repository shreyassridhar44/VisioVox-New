"""Evaluation harness (docs/08 §2).

Per-item rows are always written. Aggregates without raw rows cannot be
re-sliced or audited later, and the evaluation matrix in docs/08 §3 exists
precisely because aggregate numbers hide everything interesting.

Reporting rules, enforced here rather than left to discipline:

- **mean ± 95% CI by bootstrap**, never a bare mean. A bare mean invites
  reading a 0.3 dB difference as real when the interval is ±1.2 dB.
- **Paired bootstrap** for system comparison. Systems are evaluated on
  identical items, so pairing is both valid and far more sensitive than an
  unpaired test — it removes per-item difficulty, which dominates the variance.
- Fixed seed and deterministic item ordering, so a rerun reproduces exactly.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import bss_decompose, pesq_wb, si_sdr, si_sdri, silence_leakage_db, stoi

BOOTSTRAP_RESAMPLES = 1000
SEED = 20260826


# --------------------------------------------------------------------------
# slicing (docs/08 §3)
# --------------------------------------------------------------------------


def bin_overlap(ratio: float) -> str:
    if ratio < 0.10:
        return "0-10%"
    if ratio < 0.25:
        return "10-25%"
    if ratio < 0.50:
        return "25-50%"
    return ">50%"


def bin_rt60(rt60: float | None) -> str:
    if rt60 is None:
        return "unknown"
    if rt60 < 0.3:
        return "<0.3s"
    if rt60 <= 0.6:
        return "0.3-0.6s"
    return ">0.6s"


def bin_snr(snr_db: float | None) -> str:
    if snr_db is None:
        return "unknown"
    if snr_db > 20:
        return ">20dB"
    if snr_db >= 10:
        return "10-20dB"
    return "<10dB"


def bin_visual(quality: float | None) -> str:
    if quality is None:
        return "absent"
    return "good" if quality >= 0.6 else "degraded"


# --------------------------------------------------------------------------
# items
# --------------------------------------------------------------------------


@dataclass
class EvalSpeaker:
    speaker_id: str
    reference: np.ndarray
    active: np.ndarray | None = None
    gender: str | None = None


@dataclass
class EvalItem:
    """One clip with all of its reference speakers."""

    item_id: str
    mixture: np.ndarray
    speakers: list[EvalSpeaker]
    overlap_ratio: float = 0.0
    rt60: float | None = None
    snr_db: float | None = None
    visual_quality: float | None = None
    same_gender: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n_speakers(self) -> int:
        return len(self.speakers)


# A system takes an item and returns one estimate per speaker, in the same order.
System = Callable[[EvalItem], Sequence[np.ndarray]]


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def evaluate(
    system: System,
    dataset: Iterable[EvalItem],
    out_dir: Path,
    system_name: str = "system",
    heavy_metrics: bool = True,
) -> pd.DataFrame:
    """Run `system` over `dataset` and write per-item rows.

    `heavy_metrics` toggles PESQ and STOI, which dominate runtime on large
    sets. They are on by default because leaving them off silently would make
    two runs incomparable.
    """
    rows: list[dict[str, Any]] = []

    for item in dataset:  # deterministic order is the caller's responsibility
        estimates = system(item)
        if len(estimates) != item.n_speakers:
            raise ValueError(
                f"{item.item_id}: system returned {len(estimates)} estimates "
                f"for {item.n_speakers} speakers"
            )

        for k, spk in enumerate(item.speakers):
            est = np.asarray(estimates[k], dtype=np.float64)
            ref = np.asarray(spk.reference, dtype=np.float64)
            n = min(len(est), len(ref), len(item.mixture))
            est, ref = est[:n], ref[:n]
            mix = np.asarray(item.mixture[:n], dtype=np.float64)
            others = [
                np.asarray(o.reference[:n], dtype=np.float64)
                for j, o in enumerate(item.speakers)
                if j != k
            ]

            bss = bss_decompose(est, ref, others) if others else None
            row: dict[str, Any] = {
                "system": system_name,
                "item": item.item_id,
                "speaker": spk.speaker_id,
                "n_speakers": item.n_speakers,
                "overlap_bin": bin_overlap(item.overlap_ratio),
                "overlap_ratio": item.overlap_ratio,
                "rt60_bin": bin_rt60(item.rt60),
                "snr_bin": bin_snr(item.snr_db),
                "visual_quality_bin": bin_visual(item.visual_quality),
                "same_gender": item.same_gender,
                "si_sdr": si_sdr(est, ref),
                "si_sdri": si_sdri(est, ref, mix),
                "sdr": bss.sdr if bss else float("nan"),
                "sir": bss.sir if bss else float("nan"),
                "sar": bss.sar if bss else float("nan"),
                "silence_leakage_db": (
                    silence_leakage_db(est, spk.active) if spk.active is not None else float("nan")
                ),
            }
            if heavy_metrics:
                row["pesq"] = pesq_wb(est, ref)
                row["stoi"] = stoi(est, ref)
            rows.append(row)

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / f"per_item_{system_name}.parquet", index=False)
    return df


# --------------------------------------------------------------------------
# summarising
# --------------------------------------------------------------------------


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """(mean, low, high) at 95%. NaNs are dropped rather than propagated."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(resamples, arr.size))
    means = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise(
    df: pd.DataFrame,
    metrics: Sequence[str] = ("si_sdri", "sir", "si_sdr", "stoi"),
    by: str | None = None,
) -> pd.DataFrame:
    """mean ± 95% CI per metric, optionally sliced by a column."""
    groups: list[tuple[str, pd.DataFrame]] = (
        [("all", df)] if by is None else [(str(k), g) for k, g in df.groupby(by, dropna=False)]
    )

    rows: list[dict[str, Any]] = []
    for label, g in groups:
        row: dict[str, Any] = {"slice": label, "n": len(g)}
        for m in metrics:
            if m not in g.columns:
                continue
            mean, lo, hi = bootstrap_ci(g[m].to_numpy())
            row[m] = mean
            row[f"{m}_lo"] = lo
            row[f"{m}_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows).sort_values("slice").reset_index(drop=True)


def paired_bootstrap(
    a: pd.DataFrame,
    b: pd.DataFrame,
    metric: str = "si_sdri",
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = SEED,
) -> dict[str, float]:
    """Compare two systems on the same items.

    Pairs on (item, speaker) and bootstraps the per-pair difference. Unpaired
    comparison would be swamped by per-item difficulty, which is identical for
    both systems and therefore pure noise in this context.
    """
    key = ["item", "speaker"]
    merged = a[[*key, metric]].merge(b[[*key, metric]], on=key, suffixes=("_a", "_b"))
    if merged.empty:
        raise ValueError("no shared (item, speaker) pairs — were both run on the same dataset?")

    deltas = (merged[f"{metric}_b"] - merged[f"{metric}_a"]).to_numpy(dtype=np.float64)
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size == 0:
        raise ValueError(f"no finite deltas for {metric}")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, deltas.size, size=(resamples, deltas.size))
    means = deltas[idx].mean(axis=1)

    # Two-sided bootstrap p: how often the resampled mean crosses zero.
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {
        "metric_delta": float(deltas.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "p_value": float(min(1.0, p)),
        "n_pairs": int(deltas.size),
    }


def format_summary(summary: pd.DataFrame, metrics: Sequence[str] = ("si_sdri", "sir")) -> str:
    """Human-readable table with intervals attached to every number."""
    lines = []
    for _, r in summary.iterrows():
        parts = [f"{r['slice']!s:>12}  n={int(r['n']):>4}"]
        for m in metrics:
            if m in r and np.isfinite(r[m]):
                parts.append(f"{m} {r[m]:+6.2f} [{r[f'{m}_lo']:+.2f}, {r[f'{m}_hi']:+.2f}]")
        lines.append("  ".join(parts))
    return "\n".join(lines)
