"""Split disjointness tests (R-28).

docs/22 rates speaker leakage between train and eval as High impact and says
results are invalid until it is fixed. It is also silent: nothing crashes, the
numbers simply mean something other than what they claim.

These run in CI so a split change is checked automatically rather than
remembered.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.ami_meta import ensure_annotations, speaker_ids
from eval.splits import (
    AMI_SPLITS,
    SplitReport,
    check_ami_disjoint,
    check_speaker_disjoint,
    series_of,
    split_voxceleb_speakers,
    verify_all,
    voxceleb_speakers,
)

needs_ami_metadata = pytest.mark.skipif(
    not ensure_annotations(), reason="AMI annotations unavailable"
)


# --------------------------------------------------------------------------
# the mechanism
# --------------------------------------------------------------------------


def test_series_identifies_the_room() -> None:
    assert series_of("ES2002a") == "ES"
    assert series_of("TS3003d") == "TS"


def test_clean_split_passes() -> None:
    r = check_speaker_disjoint({"a", "b"}, {"c", "d"}, ("train", "eval"))
    assert r.ok
    verify_all([r])


def test_shared_speaker_is_caught() -> None:
    r = check_speaker_disjoint({"a", "b"}, {"b", "c"}, ("train", "eval"))
    assert not r.ok
    assert r.shared_speakers == {"b"}
    with pytest.raises(AssertionError, match="R-28"):
        verify_all([r])


def test_report_names_the_offenders() -> None:
    r = SplitReport("train", "eval", shared_speakers={"x", "y"})
    text = r.describe()
    assert "2 shared speakers" in text
    assert "x" in text


def test_verify_all_passes_when_every_pair_is_clean() -> None:
    verify_all(
        [
            check_speaker_disjoint({"a"}, {"b"}, ("train", "eval")),
            check_speaker_disjoint({"a"}, {"c"}, ("train", "holdout")),
        ]
    )


# --------------------------------------------------------------------------
# the real AMI splits — this is the gate that matters
# --------------------------------------------------------------------------


@needs_ami_metadata
def test_ami_series_are_room_disjoint() -> None:
    seen: dict[str, str] = {}
    for split, series in AMI_SPLITS.items():
        for s in series:
            assert s not in seen, f"series {s} is in both {seen[s]} and {split}"
            seen[s] = split


@needs_ami_metadata
def test_ami_train_and_eval_share_no_speakers() -> None:
    """The gate. AMI reuses people between groups, so this is not automatic."""
    from eval.ami_meta import load_speakers

    everything = load_speakers()
    by_split: dict[str, list[str]] = {k: [] for k in AMI_SPLITS}
    for meeting in everything:
        for split, series in AMI_SPLITS.items():
            if series_of(meeting) in series:
                by_split[split].append(meeting)

    assert by_split["train"], "no training meetings resolved"
    assert by_split["eval"], "no evaluation meetings resolved"

    verify_all(
        [
            check_ami_disjoint(by_split["train"], by_split["eval"], ("train", "eval")),
            check_ami_disjoint(by_split["train"], by_split["holdout"], ("train", "holdout")),
            check_ami_disjoint(by_split["eval"], by_split["holdout"], ("eval", "holdout")),
        ]
    )


@needs_ami_metadata
def test_same_group_sessions_share_speakers() -> None:
    """Sanity check on the identity source itself.

    If a/b/c/d of one group did *not* share speakers, `global_name` would not
    mean what the disjointness check assumes, and every result above would be
    reassuring for the wrong reason.
    """
    a, b = speaker_ids("ES2002a"), speaker_ids("ES2002b")
    if not a or not b:
        pytest.skip("ES2002a/b not in the metadata")
    assert a & b, "sessions of one group should share participants"


# --------------------------------------------------------------------------
# voxceleb2
# --------------------------------------------------------------------------


def test_voxceleb_split_is_by_speaker_not_utterance(tmp_path: Path) -> None:
    """Splitting utterances would put the same person on both sides."""
    for i in range(20):
        d = tmp_path / f"id{i:05d}"
        d.mkdir()
        (d / "a.npz").write_bytes(b"x")
        (d / "b.npz").write_bytes(b"x")

    speakers = voxceleb_speakers(tmp_path)
    assert len(speakers) == 20

    split = split_voxceleb_speakers(speakers, eval_fraction=0.25)
    assert len(split["eval"]) == 5
    assert len(split["train"]) == 15
    verify_all([check_speaker_disjoint(set(split["train"]), set(split["eval"]), ("train", "eval"))])


def test_voxceleb_split_is_deterministic(tmp_path: Path) -> None:
    for i in range(10):
        d = tmp_path / f"id{i:05d}"
        d.mkdir()
        (d / "a.npz").write_bytes(b"x")
    speakers = voxceleb_speakers(tmp_path)
    assert split_voxceleb_speakers(speakers) == split_voxceleb_speakers(speakers)


def test_empty_directories_are_not_speakers(tmp_path: Path) -> None:
    (tmp_path / "id00001").mkdir()  # no npz inside
    assert voxceleb_speakers(tmp_path) == set()


def test_missing_root_is_empty_not_an_error(tmp_path: Path) -> None:
    assert voxceleb_speakers(tmp_path / "nope") == set()
