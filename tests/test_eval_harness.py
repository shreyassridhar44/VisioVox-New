"""Evaluation harness and metric tests.

The load-bearing test is `test_sir_separates_leakage_from_artifact`. The whole
project rests on the claim that SI-SDR cannot express "the other speaker is
inaudible" while SIR can. If that is not demonstrably true of this
implementation, every SIR number downstream is decoration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eval.harness import (
    EvalItem,
    EvalSpeaker,
    bin_overlap,
    bin_rt60,
    bin_snr,
    bootstrap_ci,
    evaluate,
    paired_bootstrap,
    summarise,
)
from eval.metrics import bss_decompose, si_sdr, si_sdri, silence_leakage_db

RATE = 16_000


def _sine(freq: float, seconds: float = 2.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float64)


def _noise(seconds: float = 2.0, amp: float = 0.05, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(int(seconds * RATE))).astype(np.float64)


# --------------------------------------------------------------------------
# si-sdr
# --------------------------------------------------------------------------


def test_perfect_estimate_scores_infinite() -> None:
    x = _sine(220)
    assert si_sdr(x, x) == float("inf")


def test_si_sdr_is_scale_invariant() -> None:
    x = _sine(220)
    assert si_sdr(x * 7.3, x) > 100


def test_si_sdri_measures_improvement_over_the_mixture() -> None:
    target, interferer = _sine(220), _sine(440)
    mixture = target + interferer
    # a perfect estimate must improve on handing back the mixture
    assert si_sdri(target, target, mixture) > 0
    # returning the mixture unchanged is zero improvement, by definition
    assert abs(si_sdri(mixture, target, mixture)) < 1e-6


# --------------------------------------------------------------------------
# the distinction the project depends on
# --------------------------------------------------------------------------


def test_sir_separates_leakage_from_artifact() -> None:
    """Two failures, equally bad by SI-SDR, opposite by SIR.

    `leaky` keeps the target clean but lets the other speaker through.
    `distorted` suppresses the other speaker completely but adds unrelated
    noise of the same energy.

    A listener cares enormously which one they get: one has a second person
    audible, the other is a bit rough. SI-SDR scores them nearly the same. SIR
    must not.
    """
    target = _sine(220)
    interferer = _sine(440)
    artifact = _noise(seed=1)

    # match energies so SI-SDR cannot tell them apart
    scale = np.linalg.norm(interferer) / np.linalg.norm(artifact)
    artifact = artifact * scale

    leaky = target + 0.5 * interferer
    distorted = target + 0.5 * artifact

    sdr_leaky = si_sdr(leaky, target)
    sdr_distorted = si_sdr(distorted, target)
    assert abs(sdr_leaky - sdr_distorted) < 3.0, "SI-SDR should barely distinguish these"

    sir_leaky = bss_decompose(leaky, target, [interferer]).sir
    sir_distorted = bss_decompose(distorted, target, [interferer]).sir
    assert sir_distorted > sir_leaky + 15, (
        f"SIR failed to separate leakage from artifact: "
        f"leaky {sir_leaky:.1f} dB vs distorted {sir_distorted:.1f} dB"
    )


def test_clean_target_has_very_high_sir() -> None:
    target, interferer = _sine(220), _sine(440)
    assert bss_decompose(target, target, [interferer]).sir > 40


def test_more_leakage_lowers_sir_monotonically() -> None:
    target, interferer = _sine(220), _sine(440)
    sirs = [
        bss_decompose(target + g * interferer, target, [interferer]).sir for g in (0.05, 0.2, 0.8)
    ]
    assert sirs[0] > sirs[1] > sirs[2]


# --------------------------------------------------------------------------
# silence leakage
# --------------------------------------------------------------------------


def test_silence_leakage_detects_residual_in_pauses() -> None:
    """Where the target is silent is exactly where a listener notices leakage."""
    frames = 100
    active = np.zeros(frames, dtype=bool)
    active[:50] = True

    samples = frames * 640
    clean = np.zeros(samples)
    clean[: 50 * 640] = _sine(220, seconds=50 * 640 / RATE)

    leaking = clean.copy()
    leaking[50 * 640 :] = _sine(440, seconds=50 * 640 / RATE, amp=0.1)

    assert silence_leakage_db(clean, active) < silence_leakage_db(leaking, active) - 10


# --------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------


def test_bootstrap_interval_brackets_the_mean() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(10.0, 2.0, 200)
    mean, lo, hi = bootstrap_ci(values)
    assert lo < mean < hi
    assert abs(mean - 10.0) < 0.5


def test_interval_narrows_with_more_data() -> None:
    rng = np.random.default_rng(1)
    _, lo_s, hi_s = bootstrap_ci(rng.normal(0, 1, 20))
    _, lo_l, hi_l = bootstrap_ci(rng.normal(0, 1, 2000))
    assert (hi_l - lo_l) < (hi_s - lo_s)


def test_nans_are_dropped_not_propagated() -> None:
    mean, lo, hi = bootstrap_ci([1.0, 2.0, float("nan"), 3.0])
    assert abs(mean - 2.0) < 1e-9
    assert np.isfinite(lo) and np.isfinite(hi)


def test_empty_input_is_nan_not_a_crash() -> None:
    mean, lo, hi = bootstrap_ci([])
    assert np.isnan(mean) and np.isnan(lo) and np.isnan(hi)


# --------------------------------------------------------------------------
# paired comparison
# --------------------------------------------------------------------------


def _frame(system: str, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "system": system,
            "item": [f"i{i}" for i in range(len(values))],
            "speaker": "s0",
            "si_sdri": values,
        }
    )


def test_paired_bootstrap_finds_a_real_improvement() -> None:
    rng = np.random.default_rng(2)
    base = rng.normal(8.0, 3.0, 120)
    better = base + 1.5  # same items, consistently better
    result = paired_bootstrap(_frame("a", list(base)), _frame("b", list(better)))
    assert abs(result["metric_delta"] - 1.5) < 0.2
    assert result["p_value"] < 0.05
    assert result["ci_low"] > 0


def test_paired_bootstrap_does_not_invent_a_difference() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(8.0, 3.0, 120)
    b = a + rng.normal(0.0, 0.05, 120)  # noise only
    result = paired_bootstrap(_frame("a", list(a)), _frame("b", list(b)))
    assert result["p_value"] > 0.05
    assert result["ci_low"] < 0 < result["ci_high"]


def test_pairing_requires_shared_items() -> None:
    a = _frame("a", [1.0, 2.0])
    b = _frame("b", [1.0, 2.0])
    b["item"] = ["x0", "x1"]
    with pytest.raises(ValueError, match="no shared"):
        paired_bootstrap(a, b)


# --------------------------------------------------------------------------
# slicing and end-to-end
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [(0.0, "0-10%"), (0.09, "0-10%"), (0.10, "10-25%"), (0.30, "25-50%"), (0.90, ">50%")],
)
def test_overlap_bins(ratio: float, expected: str) -> None:
    assert bin_overlap(ratio) == expected


def test_rt60_and_snr_bins_handle_missing_values() -> None:
    assert bin_rt60(None) == "unknown"
    assert bin_rt60(0.2) == "<0.3s"
    assert bin_snr(None) == "unknown"
    assert bin_snr(25) == ">20dB"


def test_evaluate_writes_per_item_rows(tmp_path: Path) -> None:
    """Raw rows must always survive — aggregates alone cannot be re-sliced."""
    a, b = _sine(220), _sine(440)
    item = EvalItem(
        item_id="clip1",
        mixture=a + b,
        speakers=[EvalSpeaker("spk0", a), EvalSpeaker("spk1", b)],
        overlap_ratio=0.4,
    )

    def oracle(it: EvalItem) -> list[np.ndarray]:
        return [s.reference for s in it.speakers]

    df = evaluate(oracle, [item], tmp_path, system_name="oracle", heavy_metrics=False)
    assert len(df) == 2
    assert (tmp_path / "per_item_oracle.parquet").exists()
    assert set(df.columns) >= {"si_sdri", "sir", "sar", "overlap_bin", "n_speakers"}
    assert (df["overlap_bin"] == "25-50%").all()
    assert (df["sir"] > 40).all(), "an oracle estimate should have very high SIR"


def test_evaluate_rejects_a_wrong_estimate_count(tmp_path: Path) -> None:
    a, b = _sine(220), _sine(440)
    item = EvalItem("c", a + b, [EvalSpeaker("s0", a), EvalSpeaker("s1", b)])
    with pytest.raises(ValueError, match="returned 1 estimates"):
        evaluate(lambda it: [it.speakers[0].reference], [item], tmp_path, heavy_metrics=False)


def test_summarise_reports_intervals_not_bare_means() -> None:
    df = pd.DataFrame({"si_sdri": [1.0, 2.0, 3.0, 4.0], "n_speakers": [2, 2, 3, 3]})
    overall = summarise(df, metrics=("si_sdri",))
    assert {"si_sdri", "si_sdri_lo", "si_sdri_hi"} <= set(overall.columns)

    sliced = summarise(df, metrics=("si_sdri",), by="n_speakers")
    assert len(sliced) == 2
    assert sliced["n"].sum() == 4
