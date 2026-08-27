"""Build a real, playable fixture set for the player (Phase 6).

The sync engine could be typechecked, linted and unit-tested without a single
byte of audio ever reaching it, which is exactly the gap this closes: until the
player has decoded real AAC, crossfaded real speech and shown real captions,
"it works" is a claim about the code rather than about the product.

The material is a genuine 3-speaker Libri3Mix mixture together with its true
isolated sources. That is the honest choice — the sources *are* what a perfect
extractor would return, so the demo shows the product's target behaviour rather
than a mock of it, and the day SEAVE has a checkpoint the same page can be
pointed at real output for comparison.

What this fixture deliberately does not fake:
  - captions are transcribed from the isolated audio, not written by hand
  - speaking ratios and confidences are measured, not invented
  - the speakers are `audio_only`, because a test-pattern video contains no
    faces; inventing thumbnails would exercise a path that is lying

Usage:
    uv run python scripts/make_player_fixtures.py
    uv run python scripts/make_player_fixtures.py --base-url http://localhost:3000
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

from pipeline.s7_transcribe import Transcript, load_transcriber, transcribe

REPO = Path(__file__).resolve().parents[1]
SOURCE = Path.home() / "data" / "Libri3Mix" / "Libri3Mix" / "wav16k" / "min" / "dev"
OUT = REPO / "apps" / "web" / "public" / "fixtures"
SCHEMA = REPO / "packages" / "contracts" / "schemas" / "manifest.schema.json"
WHISPER_CACHE = REPO / "models" / "whisper"

RATE = 16_000
# Crockford base32 minus I, L, O and U — the alphabet the manifest's id patterns
# accept. Ambiguous glyphs are excluded so an id can be read aloud or copied
# from a screenshot without becoming a different id.
CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def ulid(prefix: str, rng: random.Random) -> str:
    return prefix + "".join(rng.choice(CROCKFORD) for _ in range(26))


def run(cmd: list[str]) -> subprocess.CompletedProcess[bytes]:
    """Every ffmpeg invocation goes through here, as in `ml/pipeline/s0_ingest`.

    One wrapper means one place where the argv contract is stated and checked,
    rather than a security annotation repeated at six call sites.
    """
    return subprocess.run(  # noqa: S603 - argv built here, shell=False, no user strings
        cmd, check=True, capture_output=True
    )


@dataclass
class Loudness:
    """Measured pass-1 statistics, fed back into pass 2 (docs/12 §6)."""

    i: str
    tp: str
    lra: str
    thresh: str
    offset: str


def measure_loudness(src: Path) -> Loudness:
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-i", str(src),
        "-af", "loudnorm=I=-16:TP=-1.0:LRA=11:print_format=json",
        "-f", "null", "-",
    ]  # fmt: skip
    text = run(cmd).stderr.decode()
    payload = json.loads(text[text.rindex("{") : text.rindex("}") + 1])
    return Loudness(
        i=payload["input_i"],
        tp=payload["input_tp"],
        lra=payload["input_lra"],
        thresh=payload["input_thresh"],
        offset=payload["target_offset"],
    )


def normalise(src: Path, dest: Path) -> None:
    """Two-pass loudness normalisation to -16 LUFS / -1 dBTP.

    Two passes rather than one because a single-pass loudnorm is a dynamic
    filter: it decides as it goes, so the same input gives a slightly different
    result depending on where it started. An equal-power crossfade into a track
    that is a decibel louder still sounds like a jump, so this is what makes
    switching seamless rather than merely fast (F13).
    """
    m = measure_loudness(src)
    filt = (
        "loudnorm=I=-16:TP=-1.0:LRA=11:"
        f"measured_I={m.i}:measured_TP={m.tp}:measured_LRA={m.lra}:"
        f"measured_thresh={m.thresh}:offset={m.offset}:linear=true:print_format=summary"
    )
    run([
        "ffmpeg", "-hide_banner", "-nostats", "-y", "-i", str(src),
        "-af", filt, "-ar", str(RATE), "-ac", "1", str(dest),
    ])  # fmt: skip


def encode_m4a(src: Path, dest: Path) -> int:
    run([
        "ffmpeg", "-hide_banner", "-nostats", "-y", "-i", str(src),
        "-c:a", "aac", "-b:a", "128k", "-ac", "1", str(dest),
    ])  # fmt: skip
    return dest.stat().st_size


def make_video(dest: Path, seconds: float, width: int, height: int) -> None:
    """A silent test pattern carrying a visible running timestamp.

    The timestamp is the point. It is the only way to check by eye that the
    audio the engine scheduled is still lined up with the picture, and it costs
    nothing next to shipping a real video file in the repository.
    """
    run([
        "ffmpeg", "-hide_banner", "-nostats", "-y",
        "-f", "lavfi", "-i", f"testsrc=size={width}x{height}:rate=25",
        "-t", f"{seconds:.3f}", "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "veryfast", "-movflags", "+faststart",
        "-an", str(dest),
    ])  # fmt: skip


def speaking_ratio(audio: np.ndarray, frame: int = 640) -> float:
    """Fraction of 40 ms frames carrying speech, by energy against the peak.

    Crude next to S2A's VAD, and it does not need to be better: this is a
    display value on a chip, not an input to anything.
    """
    usable = audio[: len(audio) // frame * frame].reshape(-1, frame)
    if usable.size == 0:
        return 0.0
    rms = np.sqrt(np.mean(usable**2, axis=1))
    peak = float(rms.max())
    if peak <= 0:
        return 0.0
    return float(np.mean(rms > peak * 0.08))


def mean_confidence(transcript: Transcript) -> float:
    probs = [w.probability for seg in transcript.segments for w in seg.words]
    return float(np.mean(probs)) if probs else 0.0


def pick_item(name: str | None) -> str:
    if name is not None:
        return name
    mixes = sorted(p.name for p in (SOURCE / "mix_both").glob("*.wav"))
    if not mixes:
        raise SystemExit(f"no Libri3Mix mixtures under {SOURCE}")
    # The longest of a sample: a demo that ends before the listener has
    # switched speakers twice has not demonstrated anything.
    sample = mixes[:400]
    return max(sample, key=lambda n: sf.info(str(SOURCE / "mix_both" / n)).duration)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://localhost:3000")
    ap.add_argument("--item", default=None, help="Libri3Mix mixture filename")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--whisper-model", default="large-v3")
    ap.add_argument("--no-captions", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found")
    if not SOURCE.exists():
        raise SystemExit(f"missing {SOURCE}; generate Libri3Mix first")

    # Seeded, not cryptographic, and deliberately so: regenerating the fixture
    # must produce the same ids, or every rebuild churns the manifest and any
    # bookmark into the demo breaks. These ids identify nothing real.
    rng = random.Random(args.seed)  # noqa: S311
    item = pick_item(args.item)
    print(f"item {item}")

    OUT.mkdir(parents=True, exist_ok=True)
    work = OUT / "_work"
    work.mkdir(exist_ok=True)

    # ---- read the mixture and its true sources -------------------------------
    tracks: dict[str, np.ndarray] = {}
    mixture, rate = sf.read(str(SOURCE / "mix_both" / item), dtype="float32")
    if rate != RATE:
        raise SystemExit(f"expected {RATE} Hz, got {rate}")
    tracks["mixed"] = mixture
    for i in (1, 2, 3):
        audio, _ = sf.read(str(SOURCE / f"s{i}" / item), dtype="float32")
        tracks[f"s{i}"] = audio

    lengths = {len(a) for a in tracks.values()}
    if len(lengths) != 1:
        # Invariant I3. A one-sample mismatch is invisible here and becomes
        # accumulating A/V drift in the player, so it fails the build instead.
        raise SystemExit(f"track length mismatch {sorted(lengths)} — invariant I3")
    n_samples = lengths.pop()
    duration_s = n_samples / RATE
    duration_ms = round(n_samples * 1000 / RATE)
    print(f"  {duration_s:.2f}s, {n_samples} samples per track")

    # ---- normalise, encode ---------------------------------------------------
    encoded: dict[str, int] = {}
    for key, audio in tracks.items():
        raw = work / f"{key}_raw.wav"
        norm = work / f"{key}_norm.wav"
        sf.write(str(raw), audio, RATE)
        normalise(raw, norm)
        encoded[key] = encode_m4a(norm, OUT / f"{key}.m4a")
        print(f"  {key}.m4a  {encoded[key] / 1024:.0f} kB")

    normalised = {k: sf.read(str(work / f"{k}_norm.wav"), dtype="float32")[0] for k in tracks}
    if len({len(a) for a in normalised.values()}) != 1:
        raise SystemExit("normalisation changed track lengths — invariant I3")

    make_video(OUT / "video.mp4", duration_s, args.width, args.height)
    print(f"  video.mp4  {(OUT / 'video.mp4').stat().st_size / 1024:.0f} kB")

    # ---- captions ------------------------------------------------------------
    # The contract requires `captions.vtt` on every speaker, so --no-captions
    # writes an empty transcript rather than omitting the file. That keeps the
    # fast path honest about what it produced: a speaker with nothing to say,
    # not a speaker whose captions are missing.
    transcripts: dict[str, Transcript] = {}
    if args.no_captions:
        print("  --no-captions: writing empty transcripts (audio/video smoke only)")
        for i in (1, 2, 3):
            transcripts[f"s{i}"] = Transcript(segments=[], language="und", language_probability=0.0)
    else:
        print(f"  transcribing with {args.whisper_model} on cpu (the GPU belongs to C1)")
        model = load_transcriber(
            args.whisper_model, device="cpu", compute_type="int8", cache=WHISPER_CACHE
        )
        for i in (1, 2, 3):
            transcript, _ = transcribe(normalised[f"s{i}"], model)
            transcripts[f"s{i}"] = transcript
            words = sum(len(s.words) for s in transcript.segments)
            print(f"  s{i}: {words} words — {transcript.text[:60]}")

    for key, transcript in transcripts.items():
        (OUT / f"{key}.captions.json").write_text(
            json.dumps(transcript.to_json(), indent=1), encoding="utf-8"
        )
        (OUT / f"{key}.vtt").write_text(transcript.to_vtt(), encoding="utf-8")

    # ---- manifest ------------------------------------------------------------
    base = args.base_url.rstrip("/") + "/fixtures"
    speakers: list[dict[str, Any]] = []
    for i in (1, 2, 3):
        key = f"s{i}"
        transcript = transcripts[key]
        speaker: dict[str, Any] = {
            "id": ulid("spk_", rng),
            "ordinal": i,
            "label": f"Speaker {i}",
            "color_token": f"spk-{i}",
            # No faces exist in a test pattern, so claiming `audiovisual` would
            # make the rail render a thumbnail that means nothing.
            "modality": "audio_only",
            "speaking_ratio": round(speaking_ratio(normalised[key]), 4),
            "mean_confidence": round(mean_confidence(transcript), 4),
            "extraction_ok": True,
            "audio": {"faithful": {"url": f"{base}/{key}.m4a", "bytes": encoded[key]}},
            "captions": {
                "vtt": f"{base}/{key}.vtt",
                "json": f"{base}/{key}.captions.json",
            },
        }
        speakers.append(speaker)

    manifest: dict[str, Any] = {
        "project_id": ulid("prj_", rng),
        "manifest_version": "1.0",
        "duration_ms": duration_ms,
        "has_video": True,
        "difficulty": "moderate",
        "overlap_ratio": round(sum(s["speaking_ratio"] for s in speakers) - 1.0, 4),
        "video": {"url": f"{base}/video.mp4", "width": args.width, "height": args.height},
        "speakers": speakers,
        "mixed": {"audio_url": f"{base}/mixed.m4a"},
        "playback_hint": "webaudio",
        "warnings": ["fixture_no_face_tracks"],
        # Fixtures are served from the dev server, not from signed storage, so
        # there is nothing to expire. The field is required, so it is set far
        # out rather than omitted or faked as already-expired.
        "signed_until": "2099-01-01T00:00:00Z",
    }

    from jsonschema import Draft202012Validator, FormatChecker

    validator = Draft202012Validator(json.loads(SCHEMA.read_text()), format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(manifest), key=lambda e: list(e.path))
    if errors:
        for error in errors:
            print(f"  manifest invalid at {list(error.path)}: {error.message}")
        raise SystemExit("fixture manifest does not satisfy the frozen contract")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    shutil.rmtree(work)

    print(f"\nwrote {OUT.relative_to(REPO)}  (manifest validates against contract 1.0)")
    print("  pnpm --filter @visiovox/web dev   then open /demo")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
