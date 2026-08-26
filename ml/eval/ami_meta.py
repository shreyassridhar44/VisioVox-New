"""AMI corpus metadata: the authoritative channel-to-camera mapping.

This module exists because guessing was wrong. The builder originally assumed
`Headset-N` pairs with `Closeup{N+1}`. AMI publishes the real mapping in
`corpusResources/meetings.xml`, and it is **not** consistent: there are eight
distinct channel-to-camera patterns across the corpus, and only 53 of ~171
meetings match the naive assumption.

Pairing the wrong face with the wrong voice would not have failed loudly. It
would have trained the visual pathway to associate a face with someone else's
speech, and the damage would have surfaced much later as unexplained poor
conditioning.

`global_name` is also read here. It identifies a participant across meetings,
which is what makes genuinely speaker-disjoint splits possible — group-level
disjointness alone is not sufficient, because AMI reuses people between groups.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# defusedxml, not the stdlib parser: this file is downloaded over the
# network, and the stdlib parser is vulnerable to entity-expansion attacks.
from defusedxml.ElementTree import parse as parse_xml

ANNOTATION_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
ANNOTATION_DIR = Path.home() / "data" / "ami" / "annotations"
MEETINGS_XML = ANNOTATION_DIR / "unpacked" / "corpusResources" / "meetings.xml"


@dataclass(frozen=True)
class Speaker:
    """One participant in one meeting."""

    channel: int  # pairs with Headset-{channel}.wav
    camera: str  # e.g. "Closeup2"
    role: str  # PM / ID / UI / ME
    global_name: str  # stable identity across meetings

    @property
    def headset(self) -> str:
        return f"Headset-{self.channel}"


def ensure_annotations() -> bool:
    """Download and unpack the annotation bundle if it is not already present."""
    if MEETINGS_XML.exists():
        return True

    wget, unzip = shutil.which("wget"), shutil.which("unzip")
    if wget is None or unzip is None:
        return False

    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    archive = ANNOTATION_DIR / "ami_public_manual_1.6.2.zip"
    if not archive.exists() or archive.stat().st_size < 1_000_000:
        proc = subprocess.run(  # noqa: S603 - resolved path, fixed argv
            [wget, "-c", "-q", "--tries=3", "--timeout=60", "-O", str(archive), ANNOTATION_URL],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return False

    subprocess.run(  # noqa: S603 - resolved path, fixed argv
        [unzip, "-o", "-q", str(archive), "-d", str(ANNOTATION_DIR / "unpacked")],
        check=False,
        capture_output=True,
        text=True,
    )
    return MEETINGS_XML.exists()


def load_speakers() -> dict[str, list[Speaker]]:
    """meeting id -> its participants, ordered by audio channel."""
    if not ensure_annotations():
        return {}

    root = parse_xml(MEETINGS_XML).getroot()
    out: dict[str, list[Speaker]] = {}
    for meeting in root:
        observation = meeting.get("observation")
        if not observation:
            continue
        speakers: list[Speaker] = []
        for sp in meeting:
            channel, camera = sp.get("channel"), sp.get("camera")
            if channel is None or camera is None:
                continue
            speakers.append(
                Speaker(
                    channel=int(channel),
                    camera=camera,
                    role=sp.get("role") or "",
                    global_name=sp.get("global_name") or "",
                )
            )
        if speakers:
            out[observation] = sorted(speakers, key=lambda s: s.channel)
    return out


def participants_of(meeting: str) -> list[Speaker]:
    return load_speakers().get(meeting, [])


def speaker_ids(meeting: str) -> set[str]:
    """global_name values, for checking split disjointness."""
    return {s.global_name for s in participants_of(meeting) if s.global_name}
