"""Build AMI-Eval: the in-domain evaluation set (ADR-0015).

Replaces the self-recorded VVX corpus, which will not be recorded. Each
participant contributes a (face video, reference audio) pair:

    Closeup{N}.avi   one clean face, continuously visible
    Headset-{N}.wav  that participant's own microphone

The mixture is the sum of the headsets, which is what makes SI-SDR and SIR
meaningful: the references are exactly the components the mixture decomposes
into. A far-field room mic would not be.

Splits are room-disjoint by series and speaker-disjoint by group, because AMI
reuses the same four people across the a/b/c/d sessions of one group:

    ES2002a, ES2002b ...   same room, same four people
    ES2003a ...            same room, different people
    TS3003a ...            different room, different people

So holding out whole groups gives speaker-disjointness, and holding out whole
series gives room-disjointness. docs/06 §193 requires both.

Usage:
    uv run python scripts/build_ami_eval.py --split eval --limit 4
    uv run python scripts/build_ami_eval.py --split train --limit 20
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from eval.ami_meta import Speaker, participants_of
from pipeline.vad import overlap_ratio, speech_masks

BASE = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
ROOT = Path.home() / "data" / "ami"
RAW = ROOT / "raw"
OUT = ROOT / "sets"  # sets/<split>/<meeting>, not eval/eval/...

RATE = 16_000
N_PARTICIPANTS = 4
MIN_BYTES = 100_000
CLIP_SECONDS = 120

# Room-disjoint by construction: each series is a different site.
SPLITS: dict[str, tuple[str, ...]] = {
    # Edinburgh + Idiap for training
    "train": ("ES", "IS"),
    # TNO held out entirely — unseen room and unseen people
    "eval": ("TS",),
    # Edinburgh non-scenario, kept back as a second unseen room
    "holdout": ("EN",),
}


@dataclass
class Participant:
    index: int  # audio channel; pairs with Headset-{index}
    camera: str  # the Closeup that actually shows this person
    role: str
    global_name: str  # stable identity across meetings
    face_video: str
    reference_audio: str
    speaking_ratio: float


@dataclass
class Session:
    meeting: str
    series: str
    group: str
    split: str
    start_s: float
    duration_s: int
    overlap_ratio: float
    mixture: str
    participants: list[Participant]


def remote_size(url: str) -> int | None:
    curl = shutil.which("curl")
    if curl is None:
        return None
    proc = subprocess.run(  # noqa: S603 - resolved path, fixed argv
        [curl, "-sIL", "--max-time", "30", url], check=False, capture_output=True, text=True
    )
    size: int | None = None
    for line in proc.stdout.splitlines():
        if line.lower().startswith("content-length:"):
            with contextlib.suppress(ValueError):
                size = int(line.split(":", 1)[1].strip())
    return size


def fetch(url: str, dest: Path) -> bool:
    """Resumable, and verified against Content-Length.

    Always runs wget -c rather than trusting an existing file's size: a stalled
    transfer leaves a large but truncated file, which once silently cut a
    measurement to a third of its intended length.
    """
    wget = shutil.which("wget")
    if wget is None:
        print("    wget not on PATH")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(  # noqa: S603 - resolved path, fixed argv
        [
            wget,
            "-c",
            "-q",
            "--tries=5",
            "--read-timeout=60",
            "--timeout=60",
            "--waitretry=10",
            "-O",
            str(dest),
            url,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    size = dest.stat().st_size if dest.exists() else 0
    if proc.returncode != 0 or size <= MIN_BYTES:
        if size <= MIN_BYTES:
            dest.unlink(missing_ok=True)
        return False
    expected = remote_size(url)
    if expected is not None and size != expected:
        print(f"    {dest.name}: {size} of {expected} bytes — truncated")
        return False
    return True


def list_meetings(series: tuple[str, ...]) -> list[str]:
    curl = shutil.which("curl")
    if curl is None:
        return []
    proc = subprocess.run(  # noqa: S603 - resolved path, fixed argv
        [curl, "-s", "--max-time", "60", f"{BASE}/"], check=False, capture_output=True, text=True
    )
    found: set[str] = set()
    for s in series:
        # e.g. ES2002a, TS3003d — two-letter series, four digits, one session
        # letter. A length check is fragile here; the pattern is not.
        for name in re.findall(rf"{s}\d{{4}}[a-z]", proc.stdout):
            found.add(name)
    return sorted(found)


def group_of(meeting: str) -> str:
    """ES2002a -> ES2002. The same four people appear across a/b/c/d."""
    return meeting[:6]


def load_mono(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != RATE:
        idx = np.linspace(0, len(audio) - 1, int(len(audio) * RATE / sr))
        audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
    return np.asarray(audio, dtype=np.float32)


def pick_window(masks: np.ndarray, seconds: int) -> tuple[int, float]:
    """Densest-overlap window, skipping the first minute of setup chatter."""
    width = seconds * 100
    n = masks.shape[1]
    overlapping = (masks.sum(axis=0) >= 2).astype(np.float32)
    lead_in = 60 * 100 if n > width + 60 * 100 else 0
    cum = np.concatenate([[0.0], np.cumsum(overlapping)])
    best_start, best = lead_in, -1.0
    for s in range(lead_in, max(lead_in + 1, n - width), 100):
        score = float(cum[s + width] - cum[s])
        if score > best:
            best_start, best = s, score
    return best_start, best / width


def build(meeting: str, split: str) -> Session | None:
    print(f"[{meeting}] {split}")
    people: list[Speaker] = participants_of(meeting)
    by_channel = {p.channel: p for p in people}
    if len(by_channel) != N_PARTICIPANTS:
        # Without the published mapping we would be guessing which face
        # belongs to which voice, which is the one mistake that silently
        # corrupts audio-visual training.
        print("    no camera mapping in AMI metadata; skipping")
        return None
    print("    " + ", ".join(f"ch{c}->{by_channel[c].camera}" for c in sorted(by_channel)))
    heads: list[np.ndarray] = []
    for i in range(N_PARTICIPANTS):
        dest = RAW / f"{meeting}.Headset-{i}.wav"
        if not fetch(f"{BASE}/{meeting}/audio/{meeting}.Headset-{i}.wav", dest):
            print("    missing headset; skipping meeting")
            return None
        heads.append(load_mono(dest))

    n = min(len(h) for h in heads)
    stacked = np.stack([h[:n] for h in heads])
    masks = speech_masks(stacked)
    start_frame, overlap = pick_window(masks, CLIP_SECONDS)
    start_s = start_frame / 100.0
    print(f"    window {start_s:.0f}s +{CLIP_SECONDS}s, overlap {overlap:.1%}")

    a = int(start_s * RATE)
    b = a + CLIP_SECONDS * RATE
    refs = [h[a:b] for h in stacked]
    mixture = np.sum(refs, axis=0)
    peak = float(np.abs(mixture).max())
    if peak > 0.99:
        scale = 0.99 / peak
        mixture = mixture * scale
        refs = [r * scale for r in refs]

    d = OUT / split / meeting
    d.mkdir(parents=True, exist_ok=True)
    sf.write(d / "mixture.wav", mixture, RATE)

    window = masks[:, start_frame : start_frame + CLIP_SECONDS * 100]
    participants: list[Participant] = []
    for i in range(N_PARTICIPANTS):
        sf.write(d / f"ref_spk{i}.wav", refs[i], RATE)

        # Which camera shows this participant is published per meeting; it is
        # NOT Closeup{channel+1}. Eight distinct patterns exist across the
        # corpus and only 53 of ~171 meetings match that guess.
        camera = by_channel[i].camera
        cam = RAW / f"{meeting}.{camera}.avi"
        face_name = ""
        if fetch(f"{BASE}/{meeting}/video/{meeting}.{camera}.avi", cam):
            face_name = f"face_spk{i}.mp4"
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg is not None:
                subprocess.run(  # noqa: S603 - resolved path, fixed argv
                    [
                        ffmpeg,
                        "-loglevel",
                        "error",
                        "-y",
                        "-ss",
                        f"{start_s:.3f}",
                        "-t",
                        str(CLIP_SECONDS),
                        "-i",
                        str(cam),
                        "-an",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "23",
                        "-r",
                        "25",
                        str(d / face_name),
                    ],
                    check=False,
                    capture_output=True,
                )
        else:
            print(f"    no {camera}; speaker {i} will be audio-only")

        participants.append(
            Participant(
                index=i,
                camera=camera,
                role=by_channel[i].role,
                global_name=by_channel[i].global_name,
                face_video=face_name,
                reference_audio=f"ref_spk{i}.wav",
                speaking_ratio=round(float(window[i].mean()), 4),
            )
        )

    return Session(
        meeting=meeting,
        series=meeting[:2],
        group=group_of(meeting),
        split=split,
        start_s=round(start_s, 2),
        duration_s=CLIP_SECONDS,
        overlap_ratio=round(overlap_ratio(window), 4),
        mixture="mixture.wav",
        participants=participants,
    )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=sorted(SPLITS), default="eval")
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--groups-per-series", type=int, default=2)
    args = ap.parse_args(argv)

    meetings = list_meetings(SPLITS[args.split])
    if not meetings:
        print("could not list AMI meetings; is the mirror reachable?")
        return 2

    # One session per group keeps speakers from repeating inside a split.
    seen: dict[str, int] = {}
    chosen: list[str] = []
    for m in meetings:
        g = group_of(m)
        if seen.get(g, 0) >= 1:
            continue
        if sum(1 for c in chosen if c[:2] == m[:2]) >= args.groups_per_series:
            continue
        seen[g] = seen.get(g, 0) + 1
        chosen.append(m)
        if len(chosen) >= args.limit:
            break

    print(f"building {args.split}: {', '.join(chosen)}\n")
    sessions = [s for m in chosen if (s := build(m, args.split)) is not None]
    if not sessions:
        print("no sessions built")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    index = OUT / f"{args.split}.json"
    index.write_text(json.dumps([asdict(s) for s in sessions], indent=2) + "\n")

    groups = {s.group for s in sessions}
    print(f"\n{len(sessions)} sessions, {len(groups)} disjoint groups -> {index}")
    for s in sessions:
        faces = sum(1 for p in s.participants if p.face_video)
        print(f"  {s.meeting}: overlap {s.overlap_ratio:.1%}, {faces}/4 with face video")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
