#!/usr/bin/env bash
# Keep C2 v2 alive across the machine going down.
#
# This box has lost power four times mid-run. Each time the fix was the same
# one command and each time the GPU sat idle until somebody noticed; on a
# 59-hour run that is the difference between finishing Tuesday and finishing
# Thursday. Same completion test as chain_c3_c4.sh: train_c2.py writes
# history.json as its final act, so that file existing means "finished" and
# anything else means "resume".
#
# Every flag is repeated on the resume path deliberately. --steps and --out
# are obvious, but the B/C flags configure the dataset and the dropout, and a
# resume that quietly dropped them would carry on training a *different*
# experiment from the one that produced the checkpoint -- while still writing
# to the same log, which is the kind of result that cannot be untangled later.
set -uo pipefail

REPO="$HOME/visiovox/VisioVox-New"
export PATH="$HOME/.local/bin:$PATH"
OUT="$HOME/runs/c2-v2"
LOG="$HOME/logs/c2-v2.log"
POLL=120
MAX_RESUMES=20

log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

running() { pgrep -f "[t]rain_c2.py" >/dev/null; }

resumes=0
log "supervisor started, watching $OUT"

while true; do
  if [ -f "$OUT/history.json" ]; then
    log "c2-v2 finished; supervisor exiting"
    exit 0
  fi
  if running; then
    sleep "$POLL"
    continue
  fi

  resumes=$((resumes + 1))
  if [ "$resumes" -gt "$MAX_RESUMES" ]; then
    log "resumed $MAX_RESUMES times without finishing - giving up"
    exit 1
  fi
  if [ ! -f "$OUT/last.pt" ]; then
    log "no $OUT/last.pt to resume from - giving up"
    exit 1
  fi

  log "c2-v2 stopped early (resume $resumes/$MAX_RESUMES); restarting"
  cd "$REPO" || exit 1
  setsid nohup uv run python scripts/train_c2.py \
    --steps 12000 --val-every 500 \
    --out "$OUT" \
    --drop-audio-cue 0.5 --drop-visual 0.40 \
    --confusable-prob 0.5 --tir -8.0 3.0 \
    --resume \
    >> "$LOG" 2>&1 < /dev/null &
  disown
  sleep 60
done
