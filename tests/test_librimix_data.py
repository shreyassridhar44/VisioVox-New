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
    FRAME_SAMPLES,
    RATE,
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
