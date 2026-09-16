"""S10 - Render: mux an isolated speaker back onto the video (docs/28 §W6).

The player streams HLS; this produces the single file people actually keep. Two
rules shape it.

**Never upscale.** A 720p source rendered at 1080p is the same picture with more
bytes and a worse name. The ladder is filtered against the source's real height,
so a 480p upload offers 480p and nothing else, and the UI has an honest list to
show rather than a promise it cannot keep.

**Re-encode the video only when the size changes.** At source resolution the
video stream is copied, which turns a minutes-long transcode into a remux that
runs at I/O speed. Most downloads are at source resolution, so this is the
common path and not the clever one.

Runs under the sandbox like every other ffmpeg touching user media (invariant 5),
though by this point the input is our own output rather than a stranger's file.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

STAGE = "S10_render"
VERSION = "1.0.0"


class RenderError(RuntimeError):
    """The export could not be produced."""


@dataclass(frozen=True)
class Rendition:
    """One rung of the ladder."""

    name: str
    height: int
    video_bitrate: str
    audio_bitrate: str = "192k"


LADDER: tuple[Rendition, ...] = (
    Rendition("1080p", 1080, "5000k"),
    Rendition("720p", 720, "2800k"),
    Rendition("480p", 480, "1400k"),
)


def available_renditions(source_height: int | None) -> list[Rendition]:
    """The rungs this source can honestly offer.

    A source with no video (an audio-only upload) offers none, and the caller
    exports audio instead.

    Below the lowest rung, the answer is a rung named for the source's own
    height rather than the nearest standard one. Offering "480p" for a 360p
    source and then correctly declining to upscale delivers 360p under a label
    that says otherwise — which is the same dishonesty as upscaling, minus the
    wasted bytes. The name has to match what lands on disk.
    """
    if source_height is None or source_height <= 0:
        return []

    usable = [r for r in LADDER if r.height <= source_height]
    if usable:
        return usable

    lowest = min(LADDER, key=lambda r: r.height)
    return [
        Rendition(
            name=f"{source_height}p",
            height=source_height,
            video_bitrate=lowest.video_bitrate,
            audio_bitrate=lowest.audio_bitrate,
        )
    ]


def _ffmpeg() -> str:
    return "ffmpeg"


def render_mp4(
    *,
    video: Path,
    audio: Path,
    out: Path,
    rendition: Rendition | None,
    source_height: int | None,
) -> Path:
    """Mux `audio` onto `video`, scaling only when the rung is below the source.

    `-movflags +faststart` moves the index to the front so the file starts
    playing before it has finished downloading. Without it a 2 GB export appears
    broken in a browser until the last byte lands.
    """
    out.parent.mkdir(parents=True, exist_ok=True)

    argv = [
        _ffmpeg(),
        "-v",
        "error",
        "-y",
        # The input is our own packaged output, but the whitelist stays: it
        # costs nothing and the habit is what keeps an SSRF vector closed.
        "-protocol_whitelist",
        "file",
        "-i",
        str(video),
        "-i",
        str(audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
    ]

    needs_scale = (
        rendition is not None and source_height is not None and rendition.height < source_height
    )
    if needs_scale and rendition is not None:
        argv += [
            "-vf",
            # -2 keeps the width even, which H.264 requires; odd dimensions are
            # a surprisingly common cause of "it encoded but nothing plays it".
            f"scale=-2:{rendition.height}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-b:v",
            rendition.video_bitrate,
        ]
    else:
        # The common path: same picture, new audio. A remux at I/O speed rather
        # than a transcode.
        argv += ["-c:v", "copy"]

    argv += [
        "-c:a",
        "aac",
        "-b:a",
        rendition.audio_bitrate if rendition else "192k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(out),
    ]

    proc = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=3600)  # noqa: S603
    if proc.returncode != 0 or not out.is_file():
        raise RenderError(f"render failed: {proc.stderr.strip()[:300]}")
    return out


def render_audio(*, audio: Path, out: Path, bitrate: str = "192k") -> Path:
    """An audio-only export, for a source that had no video or a caller who
    wants just the isolated voice."""
    out.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        _ffmpeg(),
        "-v",
        "error",
        "-y",
        "-protocol_whitelist",
        "file",
        "-i",
        str(audio),
        "-c:a",
        "aac",
        "-b:a",
        bitrate,
        "-movflags",
        "+faststart",
        str(out),
    ]
    proc = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=1800)  # noqa: S603
    if proc.returncode != 0 or not out.is_file():
        raise RenderError(f"audio render failed: {proc.stderr.strip()[:300]}")
    return out
