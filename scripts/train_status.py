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
TOTAL_STEPS = 20_000  # only a fallback; the run states its own length
BLOCKS = "▁▂▃▄▅▆▇█"

STEP_LINE = re.compile(r"^\s*step\s+(\d+).*?([\d.]+)s/step")
# The banner `train_c1` prints on start: "device=cuda params=5.0M batch=8x2 steps=60000".
BANNER_STEPS = re.compile(r"\bsteps=(\d+)")
# "resumed last.pt at step 27500, best +10.92 dB". Needed because the printed
# s/step is elapsed divided by steps *since the resume*, not since step zero —
# so without this the elapsed time cannot be reconstructed from the rate.
RESUMED = re.compile(r"resumed .* at step (\d+)")


def sparkline(values: list[float]) -> str:
    if not values:
        return ""
    low, high = min(values), max(values)
    span = high - low
    if span <= 0:
        return BLOCKS[0] * len(values)
    return "".join(BLOCKS[min(7, int((v - low) / span * 7.999))] for v in values)


def running(script: str) -> bool:
    # The bracket keeps the pattern from matching pgrep's own command line.
    pattern = f"[{script[0]}]{script[1:]}"
    proc = subprocess.run(  # noqa: S603 - argv built here, no shell
        ["/usr/bin/pgrep", "-f", pattern], check=False, capture_output=True
    )
    return proc.returncode == 0


def script_for(log_path: Path) -> str:
    """Guess the training script from the log name: c3.log -> train_c3.py.

    Hardcoding `train_c1.py` here made the status of every later stage read
    "NOT RUNNING" while it was training perfectly well — a wrong answer that
    looks like a real one, which is worse than no answer.
    """
    stem = log_path.stem.split("-")[0]
    return f"train_{stem}.py"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=Path.home() / "runs" / "c1")
    ap.add_argument("--log", type=Path, default=Path.home() / "logs" / "c1.log")
    # Read from the run's own banner by default. Passing the wrong total is
    # worse than tedious: every percentage, ETA and projection is computed
    # against it, so a stale value reports a run as further along than it is.
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--gate", type=float, default=GATE_DB)
    ap.add_argument("--tail", type=int, default=14, help="validations to plot")
    ap.add_argument("--script", default=None, help="override the process name to look for")
    args = ap.parse_args(argv)

    log_path: Path = args.log
    if not log_path.exists():
        print(f"no log at {log_path}")
        return 2

    step, rate, declared = 0, 0.0, None
    # (step, cumulative elapsed) for every printed line, so the *marginal* rate
    # can be recovered. The log prints elapsed/steps — a cumulative average —
    # and feeding that to an ETA is wrong whenever the run changes speed. C3
    # drifted from 7.11 to 7.64 s/step as its CPU-side room simulation got
    # heavier, and the cumulative figure still read 7.39, so every estimate
    # came out hours early.
    points: list[tuple[int, float]] = []
    origin = 0  # step the current segment started from; 0 until a resume says otherwise
    for line in log_path.read_text(errors="replace").splitlines():
        resumed = RESUMED.search(line)
        if resumed:
            origin = int(resumed.group(1))
            points.clear()  # the clock restarted; earlier points are a different series
        m = STEP_LINE.match(line)
        if m:
            step, rate = int(m.group(1)), float(m.group(2))
            points.append((step, (step - origin + 1) * rate))
        banner = BANNER_STEPS.search(line)
        if banner:
            declared = int(banner.group(1))

    # A resume restarts the elapsed clock and rewinds the step, so the log is
    # not one monotonic series — it is several. Measuring across a boundary
    # gave a *negative* rate and an ETA in the past. Keep only the current
    # segment: everything since the last point where either counter went
    # backwards.
    segment_start = 0
    for i in range(1, len(points)):
        if points[i][0] <= points[i - 1][0] or points[i][1] < points[i - 1][1]:
            segment_start = i
    segment = points[segment_start:]

    marginal = 0.0
    if len(segment) >= 2:
        window = segment[-21:] if len(segment) > 21 else segment
        (s0, e0), (s1, e1) = window[0], window[-1]
        if s1 > s0 and e1 >= e0:
            marginal = (e1 - e0) / (s1 - s0)

    total: int = args.steps if args.steps is not None else (declared or TOTAL_STEPS)
    alive = running(args.script or script_for(log_path))
    written = dt.datetime.fromtimestamp(log_path.stat().st_mtime)
    stale = (dt.datetime.now() - written).total_seconds()

    print(f"  state    {'running' if alive else 'NOT RUNNING'}")
    print(f"  step     {step:,} / {total:,}   ({step / total:.1%})")
    if rate > 0:
        done = (step + 1) * rate
        # Project on the recent rate, not the run's average.
        pace = marginal or rate
        left = (total - step) * pace
        eta = dt.datetime.now() + dt.timedelta(seconds=left)
        drift = f"  (avg {rate:.2f})" if marginal and abs(marginal - rate) > 0.05 else ""
        print(f"  pace     {pace:.2f} s/step{drift}")
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

    # C2 logs an audio-only score beside the audio-visual one, on the same
    # items. The gap between them is what the visual pathway is actually worth
    # — the headline number alone cannot separate "the video is helping" from
    # "the audio path is still adapting to a new corpus", and during C2 both
    # are moving at once.
    if history and "audio_only" in history[-1]:
        audio = [float(e["audio_only"]) for e in history if "audio_only" in e]
        gains = [
            float(e["val_si_sdri"]) - float(e["audio_only"]) for e in history if "audio_only" in e
        ]
        recent_gain = sum(gains[-5:]) / len(gains[-5:])
        print(f"  audio-only    {sparkline(audio[-args.tail :])}   last {audio[-1]:+.2f} dB")
        # C2 v2 also withholds the voice cue entirely. This column is the
        # honest test of whether the frontend learned anything: AV minus
        # audio-only is a difference between two large numbers and sat inside
        # noise for all of v1, whereas visual-only cannot be faked by the audio
        # path -- there is no speaker embedding to lean on.
        if "visual_only" in history[-1]:
            vis = [float(e["visual_only"]) for e in history if "visual_only" in e]
            print(f"  visual-only   {sparkline(vis[-args.tail :])}   last {vis[-1]:+.2f} dB")
        print(
            f"  visual worth  {gains[-1]:+.2f} dB now, {recent_gain:+.2f} dB over the "
            f"last {min(5, len(gains))} checks   (best {max(gains):+.2f})"
        )

    # Trend over the recent half, which is what a projection can honestly use:
    # the early part of a separation run is dominated by the model getting
    # worse than passthrough before it gets better.
    recent = [(s, v) for s, v in zip(steps, scores, strict=True) if s >= steps[-1] / 2]
    if len(recent) >= 3:
        first_s, first_v = recent[0]
        last_s, last_v = recent[-1]
        if last_s > first_s:
            slope = (last_v - first_v) / (last_s - first_s)
            projected = last_v + slope * (total - last_s)
            print(f"  trend         {slope * 1000:+.2f} dB / 1000 steps since step {first_s:,}")
            verdict = "on track" if projected >= args.gate else "short of the gate"
            print(f"  straight-line {projected:+.1f} dB at step {total:,} — {verdict}")
            print("                (naive: these curves usually steepen, so treat as a prompt)")

    checkpoints = sorted(p.name for p in args.run.glob("*.pt"))
    print(f"\n  checkpoints   {', '.join(checkpoints) if checkpoints else 'none yet'}")
    if "last.pt" in checkpoints:
        age = dt.datetime.now() - dt.datetime.fromtimestamp((args.run / "last.pt").stat().st_mtime)
        print(f"  resume point  last.pt, {age.total_seconds() / 60:.0f} min old")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
