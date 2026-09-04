"""Libri2Mix TSE dataset tests (Phase 4b).

The property worth defending is that enrolment never comes from the clip being
separated. That mistake does not crash, does not look wrong in a training
curve, and produces a number several dB above the truth — the model learns to
match acoustic detail rather than speaker identity, and the gap only appears on
real input, long after the result has been written down.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from training.librimix_data import (
    CLEAN,
    FRAME_SAMPLES,
    RATE,
    ConcatMixDataset,
    Libri2MixDataset,
    LibriMixConfig,
    to_batch_dict,
)

DIM = 192


def _build_split(root: Path, speakers: int = 4, per_speaker: int = 3) -> tuple[Path, Path]:
    """A miniature Libri2Mix layout with a matching enrolment index."""
    split = root / "train-mini"
    for sub in ("mix_both", "s1", "s2"):
        (split / sub).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    ids, spk_labels, vectors = [], [], []
    seconds = 5.0
    n = int(seconds * RATE)

    utt = 0
    for a in range(speakers):
        for b in range(speakers):
            if a == b:
                continue
            if utt >= speakers * per_speaker:
                break
            name = f"{100 + a}-1-{utt:04d}_{100 + b}-1-{utt:04d}.wav"
            t = np.arange(n) / RATE
            s1 = (0.3 * np.sin(2 * np.pi * (150 + 20 * a) * t)).astype(np.float32)
            s2 = (0.3 * np.sin(2 * np.pi * (150 + 20 * b) * t)).astype(np.float32)
            sf.write(split / "s1" / name, s1, RATE)
            sf.write(split / "s2" / name, s2, RATE)
            sf.write(split / "mix_both" / name, s1 + s2, RATE)

            for slot, spk in ((1, str(100 + a)), (2, str(100 + b))):
                ids.append(f"{name}|{slot}")
                spk_labels.append(spk)
                v = rng.standard_normal(DIM).astype(np.float32)
                vectors.append(v / np.linalg.norm(v))
            utt += 1

    npz = root / "enrol.npz"
    np.savez(
        npz,
        ids=np.array(ids),
        speakers=np.array(spk_labels),
        embeddings=np.stack(vectors),
    )
    return split, npz


@pytest.fixture
def dataset(tmp_path: Path) -> Libri2MixDataset:
    split, npz = _build_split(tmp_path)
    return Libri2MixDataset(split, npz, LibriMixConfig(chunk_seconds=2.0, seed=0))


# --------------------------------------------------------------------------
# the property that matters
# --------------------------------------------------------------------------


def test_enrolment_never_comes_from_the_clip_being_separated(
    tmp_path: Path,
) -> None:
    """Self-enrolment leaks the answer and inflates the result by several dB."""
    split, npz = _build_split(tmp_path)
    ds = Libri2MixDataset(split, npz, LibriMixConfig(chunk_seconds=2.0, seed=1))
    data = np.load(npz)
    index_of = {str(k): i for i, k in enumerate(data["ids"])}
    embeddings = data["embeddings"]

    for i in range(min(len(ds), 30)):
        item = ds.sample(i)
        own_key = None
        for slot in (1, 2):
            key = f"{item.mixture_id}|{slot}"
            if key in index_of and ds.speaker_of[key] == item.target_speaker:
                own_key = key
        assert own_key is not None
        own_vector = embeddings[index_of[own_key]]
        assert not np.allclose(item.enrolment, own_vector), f"item {i} enrolled from its own clip"


def test_enrolment_belongs_to_the_target_speaker(tmp_path: Path) -> None:
    """A cue from the wrong speaker trains the model to extract the wrong voice."""
    split, npz = _build_split(tmp_path)
    ds = Libri2MixDataset(split, npz, LibriMixConfig(seed=2))
    data = np.load(npz)
    lookup = {
        tuple(np.round(v, 6)): str(s)
        for v, s in zip(data["embeddings"], data["speakers"], strict=True)
    }
    for i in range(min(len(ds), 20)):
        item = ds.sample(i)
        owner = lookup.get(tuple(np.round(item.enrolment, 6)))
        assert owner == item.target_speaker


def test_single_clip_speakers_are_excluded(tmp_path: Path) -> None:
    """They cannot supply an enrolment from elsewhere, so they must not be
    targets rather than being quietly self-enrolled."""
    split = tmp_path / "s"
    for sub in ("mix_both", "s1", "s2"):
        (split / sub).mkdir(parents=True)
    n = RATE
    for name in ("1-1-0001_2-1-0001.wav",):
        for sub in ("mix_both", "s1", "s2"):
            sf.write(split / sub / name, np.zeros(n, dtype=np.float32), RATE)
    npz = tmp_path / "e.npz"
    np.savez(
        npz,
        ids=np.array(["1-1-0001_2-1-0001.wav|1", "1-1-0001_2-1-0001.wav|2"]),
        speakers=np.array(["1", "2"]),
        embeddings=np.zeros((2, DIM), dtype=np.float32),
    )
    with pytest.raises(ValueError, match=r"single clip|no usable items"):
        Libri2MixDataset(split, npz)


# --------------------------------------------------------------------------
# shapes and alignment
# --------------------------------------------------------------------------


def test_signals_have_matching_length(dataset: Libri2MixDataset) -> None:
    item = dataset.sample(0)
    want = int(2.0 * RATE)
    assert len(item.mixture) == want
    assert len(item.target) == want
    assert len(item.interferer) == want


def test_activity_mask_is_on_the_frame_grid(dataset: Libri2MixDataset) -> None:
    """Frames must line up with the 25 fps visual grid C2 will add."""
    item = dataset.sample(0)
    assert len(item.active) == len(item.target) // FRAME_SAMPLES


def test_mixture_contains_both_sources(dataset: Libri2MixDataset) -> None:
    item = dataset.sample(0)
    # mix_both adds noise, so this is a correlation check rather than equality
    residual = item.mixture - (item.target + item.interferer)
    assert float(np.abs(residual).max()) < 1.0


def test_enrolment_is_unit_norm(dataset: Libri2MixDataset) -> None:
    item = dataset.sample(3)
    assert float(np.linalg.norm(item.enrolment)) == pytest.approx(1.0, abs=1e-3)


def test_batch_dict_matches_the_trainer_contract(dataset: Libri2MixDataset) -> None:
    """Same keys as the VoxCeleb2 path, so the loop does not branch on corpus."""
    d = to_batch_dict(dataset.sample(0))
    assert set(d) == {"mixture", "target", "interferer", "active", "speaker_embedding"}
    assert all(isinstance(v, np.ndarray) for v in d.values())


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------


def test_sampling_is_reproducible(tmp_path: Path) -> None:
    split, npz = _build_split(tmp_path)
    a = Libri2MixDataset(split, npz, LibriMixConfig(seed=7)).sample(4)
    b = Libri2MixDataset(split, npz, LibriMixConfig(seed=7)).sample(4)
    assert np.array_equal(a.mixture, b.mixture)
    assert np.array_equal(a.enrolment, b.enrolment)
    assert a.target_speaker == b.target_speaker


def test_both_speaker_slots_are_used(dataset: Libri2MixDataset) -> None:
    """Only ever targeting s1 halves the data and biases toward one channel."""
    slots = {dataset.items[i % len(dataset.items)][1] for i in range(len(dataset))}
    assert slots == {1, 2}


def test_short_clips_are_padded(tmp_path: Path) -> None:
    split, npz = _build_split(tmp_path)
    ds = Libri2MixDataset(split, npz, LibriMixConfig(chunk_seconds=10.0, seed=0))
    item = ds.sample(0)
    assert len(item.target) == int(10.0 * RATE)


# --------------------------------------------------------------------------
# the noise-free curriculum condition
# --------------------------------------------------------------------------


def _add_noise_to_mixtures(split: Path, level: float = 0.1) -> None:
    """Make mix_both genuinely noisy.

    `_build_split` writes mix_both as s1 + s2, which is fine for the tests
    above but would make the clean condition indistinguishable from it — and a
    curriculum that trains twice on the same signal is not a curriculum.
    """
    rng = np.random.default_rng(7)
    for path in sorted((split / "mix_both").glob("*.wav")):
        audio, rate = sf.read(path, dtype="float32")
        noise = rng.standard_normal(audio.shape).astype(np.float32) * level
        sf.write(path, audio + noise, rate)


def test_clean_mixture_is_the_sum_of_the_sources(tmp_path: Path) -> None:
    """The noise-free condition costs an add, not a dataset.

    Libri2Mix here was generated with `--types mix_both`, so no mix_clean
    directory exists on disk. For two speakers the clean mixture is exactly
    s1 + s2, and both sources are already there — which is what makes the
    curriculum free rather than a regeneration job.
    """
    split, npz = _build_split(tmp_path)
    _add_noise_to_mixtures(split)
    cfg = LibriMixConfig(chunk_seconds=2.0, seed=0, mixture_type=CLEAN)

    item = Libri2MixDataset(split, npz, cfg).sample(0)
    np.testing.assert_allclose(item.mixture, item.target + item.interferer, rtol=0, atol=1e-6)


def test_clean_is_a_different_signal_from_mix_both(tmp_path: Path) -> None:
    split, npz = _build_split(tmp_path)
    _add_noise_to_mixtures(split)

    clean = Libri2MixDataset(
        split, npz, LibriMixConfig(chunk_seconds=2.0, seed=0, mixture_type=CLEAN)
    ).sample(0)
    noisy = Libri2MixDataset(split, npz, LibriMixConfig(chunk_seconds=2.0, seed=0)).sample(0)

    assert not np.allclose(clean.mixture, noisy.mixture, atol=1e-3)


def test_only_the_mixture_changes_between_conditions(tmp_path: Path) -> None:
    """Target, interferer and enrolment must be identical across the switch.

    If the curriculum moved the target as well, the two stages would be
    optimising different objectives, and the changeover would read as training
    instability rather than as a change of input.
    """
    split, npz = _build_split(tmp_path)
    _add_noise_to_mixtures(split)

    clean = Libri2MixDataset(
        split, npz, LibriMixConfig(chunk_seconds=2.0, seed=0, mixture_type=CLEAN)
    ).sample(3)
    noisy = Libri2MixDataset(split, npz, LibriMixConfig(chunk_seconds=2.0, seed=0)).sample(3)

    np.testing.assert_array_equal(clean.target, noisy.target)
    np.testing.assert_array_equal(clean.interferer, noisy.interferer)
    np.testing.assert_array_equal(clean.enrolment, noisy.enrolment)
    assert clean.target_speaker == noisy.target_speaker


def test_clean_condition_is_easier_than_the_noisy_one(tmp_path: Path) -> None:
    """The premise of the curriculum, stated as a measurement.

    Returning the mixture unchanged scores better against the target when there
    is no noise in it. If that were not true the ordering would be arbitrary.
    """
    split, npz = _build_split(tmp_path)
    _add_noise_to_mixtures(split, level=0.3)

    def passthrough_si_sdr(mixture_type: str) -> float:
        cfg = LibriMixConfig(chunk_seconds=2.0, seed=0, mixture_type=mixture_type)
        ds = Libri2MixDataset(split, npz, cfg)
        scores = []
        for i in range(8):
            item = ds.sample(i)
            err = item.mixture - item.target
            scores.append(
                10 * np.log10(float(np.sum(item.target**2)) / (float(np.sum(err**2)) + 1e-12))
            )
        return float(np.mean(scores))

    assert passthrough_si_sdr(CLEAN) > passthrough_si_sdr("mix_both")


# --------------------------------------------------------------------------
# several splits addressed as one
# --------------------------------------------------------------------------


def test_concat_covers_every_item_of_every_part(tmp_path: Path) -> None:
    a_split, a_npz = _build_split(tmp_path / "a")
    b_split, b_npz = _build_split(tmp_path / "b", speakers=3)
    cfg = LibriMixConfig(chunk_seconds=1.0, seed=0)
    a = Libri2MixDataset(a_split, a_npz, cfg)
    b = Libri2MixDataset(b_split, b_npz, cfg)

    both = ConcatMixDataset([a, b])
    assert len(both) == len(a) + len(b)

    # Every index must resolve, and the halves must map onto the right part —
    # an off-by-one in the bounds would silently train on one split twice.
    assert both.sample(0).mixture_id == a.sample(0).mixture_id
    assert both.sample(len(a)).mixture_id == b.sample(0).mixture_id
    assert both.sample(len(both) - 1).mixture_id == b.sample(len(b) - 1).mixture_id


def test_concat_routes_every_index_without_gaps(tmp_path: Path) -> None:
    a_split, a_npz = _build_split(tmp_path / "a")
    b_split, b_npz = _build_split(tmp_path / "b", speakers=3)
    cfg = LibriMixConfig(chunk_seconds=1.0, seed=0)
    parts = [Libri2MixDataset(a_split, a_npz, cfg), Libri2MixDataset(b_split, b_npz, cfg)]
    both = ConcatMixDataset(parts)

    seen = [both.sample(i).mixture_id for i in range(len(both))]
    expected = [p.sample(i).mixture_id for p in parts for i in range(len(p))]
    assert seen == expected


def test_concat_wraps_out_of_range_indices(tmp_path: Path) -> None:
    """The trainer draws indices with a generator, so wrapping must be safe."""
    split, npz = _build_split(tmp_path)
    both = ConcatMixDataset([Libri2MixDataset(split, npz, LibriMixConfig(chunk_seconds=1.0))])
    assert both.sample(len(both)).mixture_id == both.sample(0).mixture_id


def test_concat_rejects_an_empty_list() -> None:
    with pytest.raises(ValueError, match="at least one split"):
        ConcatMixDataset([])


def test_concat_sums_skipped_speakers(tmp_path: Path) -> None:
    a_split, a_npz = _build_split(tmp_path / "a")
    b_split, b_npz = _build_split(tmp_path / "b", speakers=3)
    cfg = LibriMixConfig(chunk_seconds=1.0)
    parts = [Libri2MixDataset(a_split, a_npz, cfg), Libri2MixDataset(b_split, b_npz, cfg)]
    both = ConcatMixDataset(parts)
    assert both.skipped_single_clip_speakers == sum(p.skipped_single_clip_speakers for p in parts)
