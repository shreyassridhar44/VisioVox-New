"""Where a training run has got to, at a glance.

Reading progress off `tail`ed log lines works but buries the only number that
matters — validation SI-SDRi — under a hundred step lines, and gives no sense
of the trend. A C1 run is two days long; the question being asked of it every
few hours is "is this still going to reach the gate", and that is a question
about the shape of the curve rather than the latest value.

So this prints the curve, the recent trend, and a projection to the gate. The
projection is a straight line through the recent validations, which is
deliberately naive: separation runs usually steepen once the model stops
fighting itself, so a projection that falls short is a prompt to look rather
than a verdict.

Usage:
    uv run python scripts/train_status.py
    uv run python scripts/train_status.py --run ~/runs/c1 --log ~/logs/c1.log
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

GATE_DB = 13.0
TOTAL_STEPS = 20_000
BLOCKS = "▁▂▃▄▅▆▇█"

STEP_LINE = re.compile(r"^\s*step\s+(\d+).*?([\d.]+)s/step")


def sparkline(values: list[float]) -> str:
    if not values:
        return ""
    low, high = min(values), max(values)
    span = high - low
    if span <= 0:
        return BLOCKS[0] * len(values)
    return "".join(BLOCKS[min(7, int((v - low) / span * 7.999))] for v in values)


def running(pattern: str = "[t]rain_c1.py") -> bool:
    proc = subprocess.run(  # noqa: S603 - fixed argv, no user strings
        ["/usr/bin/pgrep", "-f", pattern], check=False, capture_output=True
    )
    return proc.returncode == 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=Path.home() / "runs" / "c1")
    ap.add_argument("--log", type=Path, default=Path.home() / "logs" / "c1.log")
    ap.add_argument("--steps", type=int, default=TOTAL_STEPS)
    ap.add_argument("--gate", type=float, default=GATE_DB)
    ap.add_argument("--tail", type=int, default=14, help="validations to plot")
    args = ap.parse_args(argv)

    log_path: Path = args.log
    if not log_path.exists():
        print(f"no log at {log_path}")
        return 2

    step, rate = 0, 0.0
    for line in log_path.read_text(errors="replace").splitlines():
        m = STEP_LINE.match(line)
        if m:
            step, rate = int(m.group(1)), float(m.group(2))

    alive = running()
    written = dt.datetime.fromtimestamp(log_path.stat().st_mtime)
    stale = (dt.datetime.now() - written).total_seconds()

    print(f"  state    {'running' if alive else 'NOT RUNNING'}")
    print(f"  step     {step:,} / {args.steps:,}   ({step / args.steps:.1%})")
    if rate > 0:
        done = (step + 1) * rate
        left = (args.steps - step) * rate
        eta = dt.datetime.now() + dt.timedelta(seconds=left)
        print(f"  pace     {rate:.2f} s/step")
        print(f"  elapsed  {done / 3600:.1f} h        remaining {left / 3600:.1f} h")
        print(f"  finishes {eta:%a %d %b %H:%M}")
    print(f"  log      written {stale / 60:.0f} min ago")

    history_path: Path = args.run / "log.json"
    if not history_path.exists():
        print("\n  no validations yet")
        return 0

    history = json.loads(history_path.read_text())
    scores = [float(e["val_si_sdri"]) for e in history]
    steps = [int(e["step"]) + 1 for e in history]
    best = max(scores)
    best_step = steps[scores.index(best)]

    shown = scores[-args.tail :]
    print(f"\n  val SI-SDRi   {sparkline(shown)}   last {scores[-1]:+.2f} dB")
    print(f"  best          {best:+.2f} dB at step {best_step:,}   gate {args.gate:.0f} dB")

    # Trend over the recent half, which is what a projection can honestly use:
    # the early part of a separation run is dominated by the model getting
    # worse than passthrough before it gets better.
    recent = [(s, v) for s, v in zip(steps, scores, strict=True) if s >= steps[-1] / 2]
    if len(recent) >= 3:
        first_s, first_v = recent[0]
        last_s, last_v = recent[-1]
        if last_s > first_s:
            slope = (last_v - first_v) / (last_s - first_s)
            projected = last_v + slope * (args.steps - last_s)
            print(f"  trend         {slope * 1000:+.2f} dB / 1000 steps since step {first_s:,}")
            verdict = "on track" if projected >= args.gate else "short of the gate"
            print(f"  straight-line {projected:+.1f} dB at step {args.steps:,} — {verdict}")
            print("                (naive: these curves usually steepen, so treat as a prompt)")

    checkpoints = sorted(p.name for p in args.run.glob("*.pt"))
    print(f"\n  checkpoints   {', '.join(checkpoints) if checkpoints else 'none yet'}")
    if "last.pt" in checkpoints:
        age = dt.datetime.now() - dt.datetime.fromtimestamp((args.run / "last.pt").stat().st_mtime)
        print(f"  resume point  last.pt, {age.total_seconds() / 60:.0f} min old")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
