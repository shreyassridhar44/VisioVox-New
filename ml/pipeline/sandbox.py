"""Confined execution of media tools on untrusted input (ADR-0009, invariant 5).

The product's core function is **decoding attacker-supplied binary media**.
ffmpeg's demuxers are a large C/C++ surface with a long history of memory-safety
CVEs, so a malicious upload is a plausible remote-code-execution vector rather
than a theoretical one.

Defence in depth, in the order the ADR lists it:

1. **Magic bytes decide the format.** The filename and the declared
   `Content-Type` are ignored entirely - both are attacker-chosen.
2. **`ffprobe` first, under the sandbox, with hard caps.** Metadata is enough to
   reject a decode bomb in seconds without ever allocating a frame buffer.
3. **`ffmpeg` under the sandbox** with an explicit protocol whitelist.
4. **Output validated** before anything downstream trusts it.

**Deviation from ADR-0009, recorded honestly:** the ADR specifies a gVisor or
Kata runtime. Neither is available on the deployment target (WSL2, runc only),
and the zero-cost constraint rules out a separate hardened host. This module
therefore implements the ADR's Option B *with every hardening control the
platform does offer* - no network, read-only rootfs, all capabilities dropped,
non-root, no-new-privileges, seccomp, a pid ceiling, memory and CPU limits, an
empty environment and a wall-clock timeout.

What that costs: a container escape through a kernel vulnerability is no longer
mitigated by a second syscall barrier. What it still buys is the property the
ADR called decisive - **the sandbox holds no credentials and has no network**, so
a fully compromised ffmpeg gets a scratch directory and nothing else. The blast
radius stays "can corrupt one job's temporary files". If `runsc` is ever
installed, `SANDBOX_RUNTIME` switches this to gVisor with no other change.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONTAINER_WORKDIR = "/work"
NOBODY = "65534:65534"


class SandboxError(RuntimeError):
    """The sandbox refused to run, or the tool inside it failed."""


class UnsupportedMediaError(SandboxError):
    """The bytes are not a media container we are willing to open."""


@dataclass(frozen=True)
class Limits:
    """Resource ceilings. Every one of these is a documented attack.

    `memory` bounds a decompression bomb, `cpus` and `timeout` bound a
    complexity bomb, and `pids` bounds a fork bomb - which is otherwise a
    trivially available denial of service from inside any container.
    """

    memory: str = "2g"
    cpus: str = "2"
    pids: int = 128
    tmpfs_size: str = "64m"
    timeout_seconds: int = 900


PROBE_LIMITS = Limits(memory="512m", cpus="1", timeout_seconds=60)


# Magic signatures, as (offset, bytes). The declared extension and MIME type are
# never consulted: both come from the client.
_SIGNATURES: tuple[tuple[int, bytes, str], ...] = (
    (4, b"ftyp", "mp4"),  # mp4/mov/m4a - ISO base media
    (0, b"\x1a\x45\xdf\xa3", "matroska"),  # mkv/webm - EBML
    (0, b"RIFF", "riff"),  # avi/wav
    (0, b"OggS", "ogg"),
    (0, b"fLaC", "flac"),
    (0, b"FLV\x01", "flv"),
    (0, b"ID3", "mp3"),
    (0, b"\xff\xfb", "mp3"),
    (0, b"\xff\xf3", "mp3"),
    (0, b"\xff\xf2", "mp3"),
    (0, b"\x00\x00\x01\xba", "mpeg-ps"),
    (0, b"\x47", "mpeg-ts"),
)


def sniff_container(path: Path) -> str:
    """Identify the container from its bytes, or refuse.

    Refusing an unrecognised file is not merely tidy: it keeps ffmpeg's rarer,
    least-audited demuxers out of reach of anything a stranger uploads.
    """
    try:
        head = path.open("rb").read(16)
    except OSError as exc:
        raise UnsupportedMediaError(f"cannot read {path.name}") from exc

    for offset, signature, name in _SIGNATURES:
        if head[offset : offset + len(signature)] == signature:
            return name
    raise UnsupportedMediaError(
        "This file does not look like a video or audio container we can read."
    )


def docker_available() -> bool:
    return shutil.which("docker") is not None


def image_present(image: str) -> bool:
    if not docker_available():
        return False
    proc = subprocess.run(  # noqa: S603
        ["docker", "image", "inspect", image],  # noqa: S607 - resolved from PATH by design
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def build_argv(
    image: str,
    command: list[str],
    *,
    workdir: Path,
    writable: bool,
    limits: Limits,
    runtime: str | None = None,
) -> list[str]:
    """The docker invocation. Separated so the flags can be asserted in tests.

    A security posture nobody can see is one that quietly regresses; these flags
    are the posture, so they are testable rather than buried in a call.
    """
    argv = [
        "docker",
        "run",
        "--rm",
        # No network at all. ffmpeg's protocol handlers are an SSRF vector
        # reachable through crafted container metadata; this closes it at the
        # runtime layer, and -protocol_whitelist closes it again at the tool.
        "--network",
        "none",
        "--read-only",
        "--user",
        NOBODY,
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        # Empty environment: no credentials reach the sandbox, ever. This is the
        # control the ADR calls decisive.
        "--env-file",
        "/dev/null",
        "--memory",
        limits.memory,
        # Equal to --memory: without it the container can swap indefinitely and
        # the memory ceiling stops bounding anything.
        "--memory-swap",
        limits.memory,
        "--cpus",
        limits.cpus,
        "--pids-limit",
        str(limits.pids),
        "--tmpfs",
        # A tmpfs spec for a path INSIDE the container, not a host temp file.
        f"/tmp:rw,noexec,nosuid,size={limits.tmpfs_size}",  # noqa: S108
        "--workdir",
        CONTAINER_WORKDIR,
        "--mount",
        f"type=bind,source={workdir},target={CONTAINER_WORKDIR}"
        + ("" if writable else ",readonly"),
    ]
    if runtime:
        argv += ["--runtime", runtime]
    argv += [image, *command]
    return argv


def run(
    command: list[str],
    *,
    workdir: Path,
    image: str,
    writable: bool = False,
    limits: Limits | None = None,
    runtime: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a media tool against `workdir`, confined.

    `command` refers to paths under `/work`, which is `workdir` inside the
    container. Callers build those paths explicitly rather than having them
    rewritten here, so what runs is visible at the call site.
    """
    limits = limits or Limits()
    if not docker_available():
        raise SandboxError(
            "docker is required to process user media safely (ADR-0009); refusing "
            "to run ffmpeg unconfined"
        )

    argv = build_argv(
        image, command, workdir=workdir, writable=writable, limits=limits, runtime=runtime
    )
    try:
        return subprocess.run(  # noqa: S603
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=limits.timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        # A timeout here is the complexity-bomb defence firing, not an
        # infrastructure hiccup: say so, so it is not retried forever.
        raise SandboxError(
            f"media tool exceeded {limits.timeout_seconds}s and was stopped"
        ) from exc


def prepare_workdir(workdir: Path, *, writable: bool) -> None:
    """Make the scratch directory reachable by the sandbox user.

    The container runs as nobody (65534) while the scratch files belong to the
    worker's own user, so without this the sandbox cannot read its own input -
    and the failure is a bare "Permission denied" from inside a container, which
    is a genuinely confusing thing to debug.

    World-readable is acceptable because the directory holds one job's temporary
    media and nothing else; it is not a shared location.
    """
    workdir.chmod(0o777 if writable else 0o755)
    for child in workdir.rglob("*"):
        if child.is_file():
            child.chmod(0o666 if writable else 0o644)
        elif child.is_dir():
            child.chmod(0o777 if writable else 0o755)


def probe(
    path: Path,
    *,
    image: str,
    limits: Limits | None = None,
    runtime: str | None = None,
) -> dict[str, Any]:
    """Read stream metadata from untrusted media, confined.

    Magic bytes are checked first, so an unrecognised file never reaches a
    demuxer. Then ffprobe runs under the sandbox with tight caps: metadata alone
    is enough to reject a decode bomb in seconds, without ever allocating a
    frame buffer, which is what makes this the cheap first gate rather than the
    expensive one.
    """
    sniff_container(path)

    workdir = path.parent
    prepare_workdir(workdir, writable=False)

    proc = run(
        [
            "ffprobe",
            "-v",
            "error",
            # ffmpeg's protocol handlers are an SSRF vector reachable through
            # crafted container metadata. --network none closes it at the
            # runtime; this closes it again at the tool, because two
            # independent layers is the point of defence in depth.
            "-protocol_whitelist",
            "file",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            f"{CONTAINER_WORKDIR}/{path.name}",
        ],
        workdir=workdir,
        image=image,
        writable=False,
        limits=limits or PROBE_LIMITS,
        runtime=runtime,
    )

    if proc.returncode != 0:
        # Truncated and never surfaced verbatim to a user: ffprobe's stderr can
        # echo container metadata, which is attacker-controlled.
        raise UnsupportedMediaError(f"could not read this file: {proc.stderr.strip()[:200]}")

    try:
        parsed: dict[str, Any] = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise UnsupportedMediaError("media metadata was unreadable") from exc
    return parsed


@dataclass(frozen=True)
class MediaFacts:
    """The handful of facts admission control actually needs."""

    duration_seconds: float
    has_audio: bool
    has_video: bool
    width: int | None
    height: int | None
    stream_count: int
    format_name: str


def facts_from(parsed: dict[str, Any]) -> MediaFacts:
    streams = parsed.get("streams", []) or []
    fmt = parsed.get("format", {}) or {}
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    try:
        duration = float(fmt.get("duration", 0.0))
    except (TypeError, ValueError):
        duration = 0.0

    return MediaFacts(
        duration_seconds=duration,
        has_audio=audio is not None,
        has_video=video is not None,
        width=int(video["width"]) if video and "width" in video else None,
        height=int(video["height"]) if video and "height" in video else None,
        stream_count=len(streams),
        format_name=str(fmt.get("format_name", "")),
    )


# Ceilings that exist to refuse pathological files rather than to express policy.
# Real limits (duration, speakers) live in configuration; these catch the shapes
# that are never legitimate.
MAX_STREAMS = 16
MAX_PIXELS = 8192 * 8192


def assert_admissible(facts: MediaFacts, *, max_duration_seconds: int) -> None:
    """Reject the malformed-media class before any of it is decoded."""
    if not facts.has_audio:
        raise UnsupportedMediaError("This file has no audio track, so there is nothing to isolate.")

    if facts.duration_seconds <= 0:
        raise UnsupportedMediaError(
            "This file declares no duration, which usually means it is truncated or corrupt."
        )

    if facts.duration_seconds > max_duration_seconds:
        allowed = max_duration_seconds / 60
        actual = facts.duration_seconds / 60
        raise UnsupportedMediaError(
            f"This recording is {actual:.0f} minutes and the limit is {allowed:.0f}. "
            "Processing time scales with length, so please trim it first."
        )

    # A file declaring thousands of streams is a complexity bomb: each one costs
    # a demuxer context before a single frame is decoded.
    if facts.stream_count > MAX_STREAMS:
        raise UnsupportedMediaError("This file declares an implausible number of streams.")

    # The decompression-bomb shape: a tiny file claiming enormous frames.
    if facts.width and facts.height and facts.width * facts.height > MAX_PIXELS:
        raise UnsupportedMediaError("This file declares an implausible frame size.")
