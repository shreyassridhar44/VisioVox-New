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
AMI_SETS = Path.home() / "data" / "ami" / "sets"
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


def package_hls(src_m4a: Path, out_dir: Path, name: str) -> str:
    """One audio-only HLS rendition, 4 s fMP4 segments (docs/12 §6 step 4).

    Audio renditions are muxed apart from video so switching audio never
    disturbs the video buffer — the reason a rendition switch costs a few
    hundred milliseconds of sound rather than a visible stall.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    playlist = out_dir / f"{name}.m3u8"
    run([
        "ffmpeg", "-hide_banner", "-nostats", "-y", "-i", str(src_m4a),
        "-c:a", "copy", "-f", "hls", "-hls_segment_type", "fmp4",
        "-hls_time", "4", "-hls_playlist_type", "vod", "-hls_flags", "single_file",
        "-hls_fmp4_init_filename", f"{name}_init.mp4",
        "-hls_segment_filename", str(out_dir / f"{name}.m4s"),
        str(playlist),
    ])  # fmt: skip
    return playlist.name


def write_master_playlist(dest: Path, renditions: list[tuple[str, str]], video: str) -> None:
    """The multivariant playlist that binds the renditions to the video.

    Hand-written rather than produced by ffmpeg: the `EXT-X-MEDIA` group is the
    whole point of this file and ffmpeg's hls muxer has no clean way to emit
    one. The first rendition is DEFAULT=YES, which is what the player falls
    back to before a speaker has been chosen.
    """
    lines = ["#EXTM3U", "#EXT-X-VERSION:7", ""]
    for index, (name, uri) in enumerate(renditions):
        default = "YES" if index == 0 else "NO"
        lines.append(
            '#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="spk",'
            f'NAME="{name}",LANGUAGE="en",DEFAULT={default},'
            f'AUTOSELECT={default},URI="{uri}"'
        )
    lines += [
        "",
        '#EXT-X-STREAM-INF:BANDWIDTH=800000,CODECS="avc1.64001f,mp4a.40.2",AUDIO="spk"',
        video,
        "",
    ]
    dest.write_text("\n".join(lines), encoding="utf-8")


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


def make_grid_video(dest: Path, faces: list[Path], seconds: float, cell: int = 352) -> None:
    """A 2x2 grid of the participants' closeup cameras.

    This is the real product view: four people in a meeting, and you can see
    which of them is talking. It also makes A/V sync checkable against
    something meaningful — lips — rather than against a timecode.
    """
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-y"]
    for face in faces:
        cmd += ["-t", f"{seconds:.3f}", "-i", str(face)]

    # Pad to four cells so a three-person meeting still tiles cleanly.
    n = len(faces)
    scale = "".join(f"[{i}:v]scale={cell}:{cell}[v{i}];" for i in range(n))
    if n == 4:
        layout = f"[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|{cell}_0|0_{cell}|{cell}_{cell}[out]"
    else:
        layout = "".join(f"[v{i}]" for i in range(n)) + f"hstack=inputs={n}[out]"

    cmd += [
        "-filter_complex", scale + layout, "-map", "[out]",
        "-t", f"{seconds:.3f}", "-r", "25", "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "veryfast", "-movflags", "+faststart",
        "-an", str(dest),
    ]  # fmt: skip
    run(cmd)


def extract_thumbnail(source: Path, dest: Path, at_seconds: float = 5.0) -> None:
    """One frame of a participant's face, for the speaker rail.

    Taken a few seconds in rather than at zero: the first frames of an AMI
    closeup are often the room before anyone has settled.
    """
    run([
        "ffmpeg", "-hide_banner", "-nostats", "-y",
        "-ss", f"{at_seconds:.2f}", "-i", str(source),
        "-frames:v", "1", "-vf", "scale=192:-2", str(dest),
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


@dataclass
class Clip:
    """One fixture's worth of source material, whatever produced it."""

    name: str
    tracks: dict[str, np.ndarray]  # "mixed" plus "s1".."sN"
    labels: dict[str, str]
    modality: str
    faces: dict[str, Path]  # empty when the source has no face video
    overlap: float | None
    ratios: dict[str, float]  # empty means measure from the audio


def load_librimix(item: str | None) -> Clip:
    if not SOURCE.exists():
        raise SystemExit(f"missing {SOURCE}; generate Libri3Mix first")
    name = pick_item(item)
    mixture, rate = sf.read(str(SOURCE / "mix_both" / name), dtype="float32")
    if rate != RATE:
        raise SystemExit(f"expected {RATE} Hz, got {rate}")
    tracks = {"mixed": mixture}
    for i in (1, 2, 3):
        tracks[f"s{i}"], _ = sf.read(str(SOURCE / f"s{i}" / name), dtype="float32")
    return Clip(
        name=name,
        tracks=tracks,
        labels={f"s{i}": f"Speaker {i}" for i in (1, 2, 3)},
        # A test pattern has no faces in it, so claiming otherwise would put a
        # meaningless thumbnail on the rail.
        modality="audio_only",
        faces={},
        overlap=None,
        ratios={},
    )


def load_ami(meeting: str | None) -> Clip:
    """An AMI-Eval clip: a real meeting, real faces, real overlap.

    Much the better fixture. It is two minutes rather than twelve seconds, the
    speakers have faces the rail can show, and it is the in-domain material the
    project will actually be judged on (ADR-0015) — so the demo stops being a
    laboratory example and becomes the product's own scenario.
    """
    index_path = AMI_SETS / "eval.json"
    if not index_path.exists():
        raise SystemExit(f"missing {index_path}; run scripts/build_ami_eval.py first")
    entries = json.loads(index_path.read_text())
    entry = next((e for e in entries if e["meeting"] == meeting), None) if meeting else entries[0]
    if entry is None:
        raise SystemExit(f"{meeting} is not in AMI-Eval; have {[e['meeting'] for e in entries]}")

    clip_dir = AMI_SETS / "eval" / entry["meeting"]
    mixture, rate = sf.read(str(clip_dir / entry["mixture"]), dtype="float32")
    if rate != RATE:
        raise SystemExit(f"expected {RATE} Hz, got {rate}")

    tracks: dict[str, np.ndarray] = {"mixed": mixture}
    labels: dict[str, str] = {}
    faces: dict[str, Path] = {}
    ratios: dict[str, float] = {}
    for n, p in enumerate(entry["participants"], start=1):
        key = f"s{n}"
        tracks[key], _ = sf.read(str(clip_dir / p["reference_audio"]), dtype="float32")
        # The AMI role is real information and worth surfacing: in a meeting
        # "the project manager" identifies someone better than "Speaker 2".
        labels[key] = f"Speaker {n} ({p['role']})"
        faces[key] = clip_dir / p["face_video"]
        ratios[key] = round(float(p["speaking_ratio"]), 4)

    return Clip(
        name=entry["meeting"],
        tracks=tracks,
        labels=labels,
        modality="audiovisual",
        faces=faces,
        overlap=round(float(entry["overlap_ratio"]), 4),
        ratios=ratios,
    )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://localhost:3000")
    ap.add_argument("--source", choices=("ami", "librimix"), default="ami")
    ap.add_argument("--meeting", default=None, help="AMI-Eval meeting, e.g. TS3003a")
    ap.add_argument("--item", default=None, help="Libri3Mix mixture filename")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--whisper-model", default="large-v3")
    ap.add_argument("--no-captions", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found")

    # Seeded, not cryptographic, and deliberately so: regenerating the fixture
    # must produce the same ids, or every rebuild churns the manifest and any
    # bookmark into the demo breaks. These ids identify nothing real.
    rng = random.Random(args.seed)  # noqa: S311
    clip = load_ami(args.meeting) if args.source == "ami" else load_librimix(args.item)
    speaker_keys = [k for k in clip.tracks if k != "mixed"]
    print(f"{args.source}: {clip.name}, {len(speaker_keys)} speakers")

    OUT.mkdir(parents=True, exist_ok=True)
    work = OUT / "_work"
    work.mkdir(exist_ok=True)

    tracks = clip.tracks
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

    if clip.faces:
        make_grid_video(OUT / "video.mp4", [clip.faces[k] for k in speaker_keys], duration_s)
        for key in speaker_keys:
            extract_thumbnail(clip.faces[key], OUT / f"{key}.webp")
    else:
        make_video(OUT / "video.mp4", duration_s, args.width, args.height)
    probe = run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0", str(OUT / "video.mp4"),
    ]).stdout.decode()  # fmt: skip
    video_w, video_h = (int(x) for x in probe.strip().rstrip(",").split(",")[:2])
    print(f"  video.mp4  {(OUT / 'video.mp4').stat().st_size / 1024:.0f} kB  {video_w}x{video_h}")

    # ---- HLS renditions ------------------------------------------------------
    # Packaged even for a clip inside the WebAudio envelope: the streaming
    # engine is the path a real 30-minute meeting takes, and an engine with no
    # fixture is an engine nobody has run.
    hls_dir = OUT / "hls"
    hls_names = {"mixed": "Original mix", **clip.labels}
    hls_uris: dict[str, str] = {}
    for key in tracks:
        hls_uris[key] = "audio/" + package_hls(OUT / f"{key}.m4a", hls_dir / "audio", key)
    video_playlist = "video/" + package_hls(OUT / "video.mp4", hls_dir / "video", "video")
    write_master_playlist(
        hls_dir / "master.m3u8",
        [(hls_names[k], hls_uris[k]) for k in ["mixed", *speaker_keys]],
        video_playlist,
    )
    print(f"  hls/master.m3u8  {len(hls_uris)} audio renditions")

    # ---- captions ------------------------------------------------------------
    # The contract requires `captions.vtt` on every speaker, so --no-captions
    # writes an empty transcript rather than omitting the file. That keeps the
    # fast path honest about what it produced: a speaker with nothing to say,
    # not a speaker whose captions are missing.
    transcripts: dict[str, Transcript] = {}
    if args.no_captions:
        print("  --no-captions: writing empty transcripts (audio/video smoke only)")
        for key in speaker_keys:
            transcripts[key] = Transcript(segments=[], language="und", language_probability=0.0)
    else:
        print(f"  transcribing with {args.whisper_model} on cpu (the GPU belongs to C1)")
        model = load_transcriber(
            args.whisper_model, device="cpu", compute_type="int8", cache=WHISPER_CACHE
        )
        for key in speaker_keys:
            transcript, _ = transcribe(normalised[key], model)
            transcripts[key] = transcript
            words = sum(len(s.words) for s in transcript.segments)
            print(f"  {key}: {words} words — {transcript.text[:60]}")

    for key, transcript in transcripts.items():
        (OUT / f"{key}.captions.json").write_text(
            json.dumps(transcript.to_json(), indent=1), encoding="utf-8"
        )
        (OUT / f"{key}.vtt").write_text(transcript.to_vtt(), encoding="utf-8")

    # ---- manifest ------------------------------------------------------------
    base = args.base_url.rstrip("/") + "/fixtures"
    speakers: list[dict[str, Any]] = []
    for i, key in enumerate(speaker_keys, start=1):
        transcript = transcripts[key]
        speaker: dict[str, Any] = {
            "id": ulid("spk_", rng),
            "ordinal": i,
            "label": clip.labels[key],
            "color_token": f"spk-{i}",
            "modality": clip.modality,
            # Measured by the AMI builder where it exists, since it used the
            # real VAD; the energy heuristic here is only a fallback.
            "speaking_ratio": clip.ratios.get(key, round(speaking_ratio(normalised[key]), 4)),
            "mean_confidence": round(mean_confidence(transcript), 4),
            "extraction_ok": True,
            "audio": {
                "faithful": {"url": f"{base}/{key}.m4a", "bytes": encoded[key]},
                "hls": f"{base}/hls/{hls_uris[key]}",
            },
            "captions": {
                "vtt": f"{base}/{key}.vtt",
                "json": f"{base}/{key}.captions.json",
            },
        }
        if key in clip.faces:
            speaker["thumbnail_url"] = f"{base}/{key}.webp"
        speakers.append(speaker)

    manifest: dict[str, Any] = {
        "project_id": ulid("prj_", rng),
        "manifest_version": "1.0",
        "duration_ms": duration_ms,
        "has_video": True,
        "difficulty": "moderate",
        "overlap_ratio": (
            clip.overlap
            if clip.overlap is not None
            else round(sum(float(s["speaking_ratio"]) for s in speakers) - 1.0, 4)
        ),
        "video": {"url": f"{base}/video.mp4", "width": video_w, "height": video_h},
        "speakers": speakers,
        "mixed": {"audio_url": f"{base}/mixed.m4a", "hls": f"{base}/hls/{hls_uris['mixed']}"},
        "master_playlist": f"{base}/hls/master.m3u8",
        # Both fixtures sit inside the WebAudio envelope, so that is the honest
        # hint. The HLS assets are packaged anyway and the demo can force that
        # engine, which is the only way to exercise the path a real long
        # recording will take.
        "playback_hint": "webaudio",
        "warnings": [] if clip.faces else ["fixture_no_face_tracks"],
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
