"""Dataset splits and disjointness verification (R-28, docs/06 §193).

Speaker leakage between train and eval is the failure that invalidates results
without breaking anything. Everything runs, the numbers look good, and they
mean nothing — the model was tested on people it trained on. It is also easy to
introduce: adding meetings to a split, or trusting that different meeting ids
imply different people.

For AMI neither group nor series disjointness is sufficient on its own. The
same four people appear across the a/b/c/d sessions of a group, and people
recur between groups, so disjointness has to be checked on `global_name`
identities read from the corpus metadata. Room disjointness is separate again,
and is what the series split provides.

The check is a test, so it runs in CI on every split change rather than being
something to remember.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .ami_meta import speaker_ids

# Room-disjoint by construction: each AMI series is a different site.
AMI_SPLITS: dict[str, tuple[str, ...]] = {
    "train": ("ES", "IS"),
    "eval": ("TS",),
    "holdout": ("EN",),
}


@dataclass
class SplitReport:
    """Outcome of checking one pair of splits."""

    left: str
    right: str
    shared_speakers: set[str] = field(default_factory=set)
    shared_rooms: set[str] = field(default_factory=set)

    @property
    def ok(self) -> bool:
        return not self.shared_speakers and not self.shared_rooms

    def describe(self) -> str:
        if self.ok:
            return f"{self.left} / {self.right}: disjoint"
        parts = []
        if self.shared_speakers:
            shown = sorted(self.shared_speakers)[:5]
            parts.append(
                f"{len(self.shared_speakers)} shared speakers "
                f"({', '.join(shown)}{' ...' if len(self.shared_speakers) > 5 else ''})"
            )
        if self.shared_rooms:
            parts.append(f"shared rooms: {', '.join(sorted(self.shared_rooms))}")
        return f"{self.left} / {self.right}: " + "; ".join(parts)


def series_of(meeting: str) -> str:
    """ES2002a -> ES. One series is one physical room."""
    return meeting[:2]


def ami_speakers(meetings: list[str]) -> set[str]:
    """Union of participant identities across meetings, from corpus metadata."""
    out: set[str] = set()
    for m in meetings:
        out |= speaker_ids(m)
    return out


def check_ami_disjoint(left: list[str], right: list[str], names: tuple[str, str]) -> SplitReport:
    """Speaker- and room-disjointness between two lists of AMI meetings."""
    report = SplitReport(left=names[0], right=names[1])
    report.shared_speakers = ami_speakers(left) & ami_speakers(right)
    report.shared_rooms = {series_of(m) for m in left} & {series_of(m) for m in right}
    return report


def check_speaker_disjoint(left: set[str], right: set[str], names: tuple[str, str]) -> SplitReport:
    """Generic speaker-identity check, for corpora without rooms (VoxCeleb2)."""
    return SplitReport(left=names[0], right=names[1], shared_speakers=left & right)


def voxceleb_speakers(packed_split_root: Path) -> set[str]:
    """Speaker ids present in a packed VoxCeleb2 split.

    VoxCeleb2 ids are globally unique, so directory names are identities and no
    metadata lookup is needed.
    """
    root = Path(packed_split_root)
    if not root.is_dir():
        return set()
    return {p.name for p in root.iterdir() if p.is_dir() and any(p.glob("*.npz"))}


def split_voxceleb_speakers(
    speakers: set[str], eval_fraction: float = 0.2, seed: int = 20260826
) -> dict[str, list[str]]:
    """Partition speaker ids into train and eval.

    Split by *speaker*, never by utterance. Splitting utterances would put the
    same person on both sides, which is the exact leak this module exists to
    prevent, and it would inflate results substantially.
    """
    import random

    ordered = sorted(speakers)
    # S311: the split must be reproducible across runs and machines, which
    # is the opposite of what a CSPRNG provides.
    rng = random.Random(seed)  # noqa: S311
    rng.shuffle(ordered)
    n_eval = max(1, int(len(ordered) * eval_fraction))
    return {"eval": sorted(ordered[:n_eval]), "train": sorted(ordered[n_eval:])}


def verify_all(reports: list[SplitReport]) -> None:
    """Raise if any pair leaks. Called by the CI test."""
    bad = [r for r in reports if not r.ok]
    if bad:
        detail = "\n  ".join(r.describe() for r in bad)
        raise AssertionError(
            "split disjointness violated (R-28) — results computed on these "
            f"splits are invalid:\n  {detail}"
        )
