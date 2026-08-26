"""Build the Phase 1 test set from the AMI Meeting Corpus.

Phase 1 exits on "run on 3 test videos, document every failure precisely"
(docs/21). AMI is the right source: real multi-party conversation with genuine
overlap, synchronised video, and -- the part that matters -- a separate headset
microphone per participant.

Those headsets give us ground-truth references, so the Tier 0 baseline can be
scored (SI-SDR, SIR) on real video instead of only described. Without them the
baseline report would be qualitative, and there would be nothing to compare
Tier 1 against on in-domain data.

The mixture is the sum of the headsets rather than a room mic: it is the same
signal the references decompose into, which is what makes the metric valid.

Licence: AMI is CC BY 4.0 (docs/06 §4, ADR-0013).

Usage:  uv run python scripts/fetch_testvideos.py
"""

from __future__ import annotations

import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

BASE = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
OUT = Path.home() / "data" / "testvideos"
RAW = OUT / "raw"
CLIPS = OUT / "clips"

SAMPLE_RATE = 16_000
CLIP_SECONDS = 90
N_SPEAKERS = 4


@dataclass(frozen=True)
class Meeting:
    """Camera naming differs per AMI site, so the wide view is named per meeting."""

    ident: str
    camera: str
    site: str


MEETINGS = [
    Meeting("ES2002a", "Corner", "Edinburgh"),
    Meeting("IS1000a", "C", "Idiap"),
    Meeting("TS3003a", "Overview1", "TNO"),
]


def fetch(url: str, dest: Path) -> bool:
    """Download unless already present and non-trivial (404 pages are tiny)."""
    if dest.exists() and dest.stat().st_size > 100_000:
        print(f"    have {dest.name}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"    get  {dest.name}", flush=True)
        urllib.request.urlretrieve(url, dest)  # noqa: S310 - fixed https host
    except Exception as exc:
        print(f"    FAIL {dest.name}: {exc}")
        dest.unlink(missing_ok=True)
        return False
    if dest.stat().st_size < 100_000:
        print(f"    FAIL {dest.name}: {dest.stat().st_size} bytes (404 page?)")
        dest.unlink(missing_ok=True)
        return False
    return True


def load_mono_16k(path: Path) -> np.ndarray:
    audio: np.ndarray
    sr: int
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        # AMI headsets are 16 kHz already; resample defensively via ffmpeg-free
        # linear interpolation only if that ever changes.
        idx = np.linspace(0, len(audio) - 1, int(len(audio) * SAMPLE_RATE / sr))
        audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
    return audio


def speech_mask(track: np.ndarray, frame: int = 160) -> np.ndarray:
    """Cheap energy VAD at 10 ms resolution, thresholded per track.

    A fixed dB floor would misjudge headsets recorded at different gains, so
    the threshold is relative to each track's own active level.
    """
    n = len(track) // frame
    energy = (track[: n * frame] ** 2).reshape(n, frame).mean(axis=1)
    db = 10 * np.log10(energy + 1e-10)
    speech_ref = np.percentile(db, 95)
    mask: np.ndarray = db > (speech_ref - 25)
    return mask


def pick_overlap_window(masks: list[np.ndarray], seconds: int) -> tuple[int, float]:
    """Return (start_frame, overlap_ratio) for the densest-overlap window.

    Picking the window by measured overlap, rather than an arbitrary offset,
    is what makes these clips exercise the case the product exists for.
    """
    width = seconds * 100  # 10 ms frames
    n = min(len(m) for m in masks)
    stack = np.stack([m[:n] for m in masks]).astype(np.int16)
    active = stack.sum(axis=0)
    overlapping = (active >= 2).astype(np.float32)
    # Skip the first 60 s: AMI meetings open with setup chatter and silence.
    lead_in = 60 * 100
    if n <= width + lead_in:
        lead_in = 0
    cum = np.concatenate([[0.0], np.cumsum(overlapping)])
    best_start, best = lead_in, -1.0
    for s in range(lead_in, n - width, 100):  # 1 s stride
        score = float(cum[s + width] - cum[s])
        if score > best:
            best_start, best = s, score
    return best_start, best / width


def run(cmd: list[str]) -> None:
    # ruff S603: argv is built from module constants and validated paths,
    # never from user input, and shell=False.
    subprocess.run(cmd, check=True, capture_output=True)  # noqa: S603


def build(meeting: Meeting) -> dict[str, object] | None:
    print(f"\n[{meeting.ident}] {meeting.site}")
    video_src = RAW / f"{meeting.ident}.{meeting.camera}.avi"
    if not fetch(f"{BASE}/{meeting.ident}/video/{meeting.ident}.{meeting.camera}.avi", video_src):
        return None

    headsets: list[np.ndarray] = []
    for i in range(N_SPEAKERS):
        dest = RAW / f"{meeting.ident}.Headset-{i}.wav"
        if not fetch(f"{BASE}/{meeting.ident}/audio/{meeting.ident}.Headset-{i}.wav", dest):
            return None
        headsets.append(load_mono_16k(dest))

    n = min(len(h) for h in headsets)
    headsets = [h[:n] for h in headsets]

    masks = [speech_mask(h) for h in headsets]
    start_frame, overlap = pick_overlap_window(masks, CLIP_SECONDS)
    start_s = start_frame / 100.0
    print(f"    window {start_s:.1f}s +{CLIP_SECONDS}s, overlap ratio {overlap:.3f}")

    a = int(start_s * SAMPLE_RATE)
    b = a + CLIP_SECONDS * SAMPLE_RATE
    refs = [h[a:b] for h in headsets]
    mixture = np.sum(refs, axis=0)
    peak = float(np.abs(mixture).max())
    if peak > 0.99:  # avoid clipping, scale references identically
        scale = 0.99 / peak
        mixture = mixture * scale
        refs = [r * scale for r in refs]

    d = CLIPS / meeting.ident
    d.mkdir(parents=True, exist_ok=True)
    sf.write(d / "mixture.wav", mixture, SAMPLE_RATE)
    for i, r in enumerate(refs):
        sf.write(d / f"ref_spk{i}.wav", r, SAMPLE_RATE)

    # Mux the wide-camera video with the mixture: this is the actual product input.
    run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start_s:.3f}",
            "-t",
            str(CLIP_SECONDS),
            "-i",
            str(video_src),
            "-i",
            str(d / "mixture.wav"),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            str(d / "input.mp4"),
        ]
    )

    speaking = [float(m[start_frame : start_frame + CLIP_SECONDS * 100].mean()) for m in masks]
    return {
        "meeting": meeting.ident,
        "site": meeting.site,
        "camera": meeting.camera,
        "start_s": round(start_s, 2),
        "duration_s": CLIP_SECONDS,
        "overlap_ratio": round(overlap, 4),
        "speaking_ratio": [round(s, 4) for s in speaking],
    }


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    CLIPS.mkdir(parents=True, exist_ok=True)
    results = []
    for m in MEETINGS:
        r = build(m)
        if r is not None:
            results.append(r)

    print(f"\n{len(results)}/{len(MEETINGS)} clips built")
    if not results:
        return 1
    import json

    (CLIPS / "manifest.json").write_text(json.dumps(results, indent=2) + "\n")
    for r in results:
        print(f"  {r['meeting']}: overlap {r['overlap_ratio']}, speakers {r['speaking_ratio']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
