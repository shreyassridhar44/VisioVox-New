"""Fetch VoxCeleb2 from the mmai.io distribution (Tier 2, docs/06 §3).

Two things this handles that a plain wget does not.

**Incomplete TLS chain.** `cn01.mmai.io` serves only its leaf certificate and
omits the Sectigo intermediate, so verification fails with "unable to get local
issuer certificate". The certificate itself is perfectly valid — it names the
intermediate in its Authority Information Access extension. This script fetches
that intermediate, checks it chains to a root already in the system store, and
then verifies normally. It never falls back to --insecure: turning off
verification for a multi-gigabyte download over the public internet is not a
workaround, it is a different bug.

**Resume and truncation.** Parts are tens of gigabytes over an academic link
that drops. Downloads resume, and every completed file is checked against the
server's Content-Length before being treated as done.

The key comes from VOXCELEB2_KEY in .env.local and is never written to disk or
logged.

Usage:
    uv run python scripts/fetch_voxceleb2.py --set test        # ~11 GB, 118 speakers
    uv run python scripts/fetch_voxceleb2.py --set dev --parts 2
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

HOST = "https://cn01.mmai.io/download/voxceleb"
AIA_URL = "http://crt.sectigo.com/SectigoPublicServerAuthenticationCADVR36.crt"
SYSTEM_CA = Path("/etc/ssl/certs/ca-certificates.crt")

CERT_DIR = Path.home() / ".certs"
CHAIN = CERT_DIR / "mmai-chain.pem"
DEST = Path.home() / "data" / "voxceleb2"

# dev video is split across parts aa..ai; test ships as single archives
DEV_PARTS = [f"vox2_dev_mp4_part{c}" for c in "abcdefghi"]
SETS: dict[str, list[str]] = {
    "test": ["vox2_test_txt.zip", "vox2_test_aac.zip", "vox2_test_mp4.zip"],
    "dev": ["vox2_dev_txt.zip", *DEV_PARTS],
}


def run(cmd: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - resolved paths, fixed argv, shell=False
        cmd, check=False, capture_output=True, text=True, timeout=timeout
    )


def ensure_chain() -> Path | None:
    """Build a CA bundle that completes the server's incomplete chain.

    Returns None if the intermediate cannot be verified against a trusted root,
    in which case we refuse to download rather than relax verification.
    """
    if CHAIN.exists() and CHAIN.stat().st_size > SYSTEM_CA.stat().st_size:
        return CHAIN

    curl, openssl = shutil.which("curl"), shutil.which("openssl")
    if curl is None or openssl is None:
        print("curl and openssl are required")
        return None

    CERT_DIR.mkdir(parents=True, exist_ok=True)
    der = CERT_DIR / "sectigo-r36.der"
    pem = CERT_DIR / "sectigo-r36.pem"

    print("fetching the intermediate named in the server certificate's AIA")
    if run([curl, "-sS", "--max-time", "60", "-o", str(der), AIA_URL]).returncode != 0:
        print("  could not fetch the intermediate")
        return None
    if run([openssl, "x509", "-inform", "DER", "-in", str(der), "-out", str(pem)]).returncode != 0:
        print("  intermediate is not a parseable certificate")
        return None

    verify = run([openssl, "verify", "-CAfile", str(SYSTEM_CA), str(pem)])
    if verify.returncode != 0 or "OK" not in verify.stdout:
        # The intermediate does not chain to anything we trust. Stop.
        print(f"  intermediate does not chain to a trusted root: {verify.stdout.strip()}")
        return None
    print("  intermediate verified against the system trust store")

    CHAIN.write_bytes(SYSTEM_CA.read_bytes() + pem.read_bytes())
    return CHAIN


def remote_size(url: str, ca: Path) -> int | None:
    proc = run(
        [shutil.which("curl") or "curl", "-sIL", "--cacert", str(ca), "--max-time", "90", url]
    )
    size: int | None = None
    for line in proc.stdout.splitlines():
        if line.lower().startswith("content-length:"):
            with contextlib.suppress(ValueError):
                size = int(line.split(":", 1)[1].strip())
    return size


def fetch(name: str, key: str, ca: Path) -> bool:
    url = f"{HOST}?key={key}&file={name}"
    dest = DEST / name
    dest.parent.mkdir(parents=True, exist_ok=True)

    expected = remote_size(url, ca)
    have = dest.stat().st_size if dest.exists() else 0
    if expected is not None and have == expected:
        print(f"  have {name} ({have / 1024**3:.2f} GB)")
        return True

    gb = f"{expected / 1024**3:.2f} GB" if expected else "unknown size"
    print(f"  get  {name} ({gb}, resuming from {have / 1024**3:.2f} GB)", flush=True)

    wget = shutil.which("wget")
    if wget is None:
        print("  wget not on PATH")
        return False
    proc = subprocess.run(  # noqa: S603 - resolved path, fixed argv
        [
            wget,
            "-c",
            "-q",
            "--tries=0",
            "--read-timeout=60",
            "--timeout=60",
            "--waitretry=15",
            "--ca-certificate",
            str(ca),
            "-O",
            str(dest),
            url,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    have = dest.stat().st_size if dest.exists() else 0
    if proc.returncode != 0:
        print(f"  FAIL {name}: wget rc={proc.returncode}")
        return False
    if expected is not None and have != expected:
        print(f"  FAIL {name}: {have} of {expected} bytes — truncated")
        return False
    print(f"  ok   {name} ({have / 1024**3:.2f} GB)")
    return True


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", choices=sorted(SETS), default="test")
    ap.add_argument("--parts", type=int, default=0, help="limit dev parts (0 = all)")
    args = ap.parse_args(argv)

    key = os.environ.get("VOXCELEB2_KEY", "").strip()
    if not key:
        print("VOXCELEB2_KEY is not set; it lives in .env.local")
        return 2

    ca = ensure_chain()
    if ca is None:
        print("refusing to download without a verifiable TLS chain")
        return 1

    files = SETS[args.set]
    if args.set == "dev" and args.parts > 0:
        files = [files[0], *DEV_PARTS[: args.parts]]

    print(f"\nvoxceleb2 {args.set}: {len(files)} files -> {DEST}")
    ok = sum(1 for f in files if fetch(f, key, ca))
    total = sum(p.stat().st_size for p in DEST.glob("vox2_*") if p.is_file())
    print(f"\n{ok}/{len(files)} files, {total / 1024**3:.1f} GB on disk")
    return 0 if ok == len(files) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
