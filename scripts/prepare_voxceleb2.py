"""Prepare VoxCeleb2 for audio-visual training (Tier 2, docs/06 §3, Phase 3).

VoxCeleb2 is the visual-training corpus because each clip contains exactly one
speaker: the face and the voice match by construction, so no active-speaker
detection and no face-to-voice binding is required to build it. That matters
here specifically — the motion-energy ASD was measured to carry no usable
signal (docs/27 §6), so any corpus needing inferred binding is unusable until
Light-ASD lands. VoxCeleb2 needs none.

Output is one packed `.npz` per utterance holding:

    mouth   uint8 (frames, 96, 96)  grayscale mouth ROI at 25 fps
    audio   float32 (samples,)      16 kHz mono, time-aligned with `mouth`

Packed rather than left as video because the dataloader is the bottleneck the
A5000 config is tuned around (docs/25 §5 targets >90% GPU utilisation).
Decoding H.264 per sample per epoch would not reach it; memory-mapped uint8
arrays will.

Mixtures are **not** written. They are simulated on the fly (docs/07 §3), which
is what keeps this at tens of GB rather than terabytes and gives unlimited
mixture variety from the same clips.

Usage:
    uv run python scripts/prepare_voxceleb2.py --split test --limit-speakers 20
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path.home() / "data" / "voxceleb2"
EXTRACT = ROOT / "extracted"
PACKED = ROOT / "packed"

RATE = 16_000
FPS = 25
MOUTH = 96  # output ROI, pixels


@dataclass
class Utterance:
    speaker: str
    session: str
    clip: str
    video: Path
    audio: Path

    @property
    def key(self) -> str:
        return f"{self.speaker}/{self.session}/{self.clip}"


def unzip(archive: Path, dest: Path, marker: str) -> bool:
    """Extract once; `marker` is a path that exists afterwards."""
    if (dest / marker).exists():
        print(f"  already extracted: {archive.name}")
        return True
    if not archive.exists():
        print(f"  missing archive: {archive.name}")
        return False
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {archive.name} ...", flush=True)
    try:
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    except zipfile.BadZipFile:
        print(f"  {archive.name} is not a valid zip (truncated download?)")
        return False
    return True


def find_utterances(limit_speakers: int) -> list[Utterance]:
    """Pair each mp4 with its m4a. Both zips share the id/session/clip layout."""
    video_root = next((p for p in (EXTRACT / "mp4", EXTRACT / "dev" / "mp4") if p.is_dir()), None)
    audio_root = next((p for p in (EXTRACT / "aac", EXTRACT / "dev" / "aac") if p.is_dir()), None)
    if video_root is None or audio_root is None:
        print(f"  expected mp4/ and aac/ under {EXTRACT}")
        return []

    speakers = sorted(p.name for p in video_root.iterdir() if p.is_dir())
    if limit_speakers > 0:
        speakers = speakers[:limit_speakers]

    out: list[Utterance] = []
    for spk in speakers:
        for mp4 in sorted((video_root / spk).rglob("*.mp4")):
            session, clip = mp4.parent.name, mp4.stem
            m4a = audio_root / spk / session / f"{clip}.m4a"
            if m4a.exists():
                out.append(Utterance(spk, session, clip, mp4, m4a))
    return out


def decode_audio(path: Path) -> np.ndarray | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    proc = subprocess.run(  # noqa: S603 - resolved path, fixed argv
        [
            ffmpeg,
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(RATE),
            "-",
        ],
        check=False,
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        return None
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def decode_mouth(path: Path, max_frames: int) -> np.ndarray | None:
    """Grayscale mouth ROI per frame.

    VoxCeleb2 clips are already tight, face-centred crops, so the mouth sits in
    a stable region rather than needing per-frame detection. Taking a fixed
    lower-centre box is both far cheaper than running a detector over every
    frame and more stable — a detector that drops out for a few frames leaves
    holes a fixed crop does not have.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    # crop the lower-middle of the frame, then scale to the ROI size
    vf = f"fps={FPS},crop=iw*0.6:ih*0.36:iw*0.2:ih*0.56,scale={MOUTH}:{MOUTH},format=gray"
    proc = subprocess.run(  # noqa: S603 - resolved path, fixed argv
        [
            ffmpeg,
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-vf",
            vf,
            "-frames:v",
            str(max_frames),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        check=False,
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        return None
    n = len(proc.stdout) // (MOUTH * MOUTH)
    if n == 0:
        return None
    return np.frombuffer(proc.stdout[: n * MOUTH * MOUTH], dtype=np.uint8).reshape(n, MOUTH, MOUTH)


def pack(utt: Utterance, dest_root: Path, max_seconds: float) -> bool:
    dest = dest_root / utt.speaker / f"{utt.session}_{utt.clip}.npz"
    if dest.exists():
        return True

    mouth = decode_mouth(utt.video, int(max_seconds * FPS))
    audio = decode_audio(utt.audio)
    if mouth is None or audio is None:
        return False

    # Trim both to the same duration. Video is authoritative because the ROI
    # cannot be interpolated meaningfully, and a half-frame of trailing audio
    # would put every later sample out of step with its frame.
    n_frames = min(len(mouth), int(len(audio) / RATE * FPS))
    if n_frames < FPS:  # under a second is not useful for conditioning
        return False
    mouth = mouth[:n_frames]
    audio = audio[: int(n_frames / FPS * RATE)]

    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, mouth=mouth, audio=audio.astype(np.float32))
    return True


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("test", "dev"), default="test")
    ap.add_argument("--limit-speakers", type=int, default=0, help="0 = all")
    ap.add_argument("--max-seconds", type=float, default=6.0)
    args = ap.parse_args(argv)

    print(f"extracting voxceleb2 {args.split}")
    ok_v = unzip(ROOT / f"vox2_{args.split}_mp4.zip", EXTRACT, "mp4")
    ok_a = unzip(ROOT / f"vox2_{args.split}_aac.zip", EXTRACT, "aac")
    if not (ok_v and ok_a):
        return 1

    utterances = find_utterances(args.limit_speakers)
    if not utterances:
        print("no utterances found")
        return 1
    speakers = sorted({u.speaker for u in utterances})
    print(f"{len(utterances)} utterances from {len(speakers)} speakers")

    dest_root = PACKED / args.split
    done = failed = 0
    for i, utt in enumerate(utterances, start=1):
        if pack(utt, dest_root, args.max_seconds):
            done += 1
        else:
            failed += 1
        if i % 200 == 0:
            print(f"  {i}/{len(utterances)}  packed={done} failed={failed}", flush=True)

    size = sum(p.stat().st_size for p in dest_root.rglob("*.npz"))
    print(f"\npacked {done}, failed {failed}, {size / 1024**3:.2f} GB -> {dest_root}")
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
