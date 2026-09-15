#!/usr/bin/env bash
# Live training dashboard — one screen, refreshed in place.
#
# train_status.py answers "where is this run" in one shot. This is the version
# you leave open on a second monitor: the same summary, plus the recent
# validations and the newest step lines, redrawn every INTERVAL seconds.
#
# The status output is captured *before* the screen is cleared. Clearing first
# and then waiting two seconds for uv to start leaves the terminal blank on
# every refresh, which reads as "the run died" to anyone glancing at it.
#
# Defaults point at the run that is currently going. Override per invocation:
#   RUN=~/runs/c3 LOG=~/logs/c3.log GATE=13 bash scripts/watch_training.sh
set -uo pipefail

export PATH="$HOME/.local/bin:$PATH"
REPO="$HOME/visiovox/VisioVox-New"
RUN="${RUN:-$HOME/runs/c2-v2}"
LOG="${LOG:-$HOME/logs/c2-v2.log}"
GATE="${GATE:-10}"
INTERVAL="${INTERVAL:-30}"

cd "$REPO" || exit 1
rule=$(printf '%0.s-' $(seq 1 72))

while true; do
  status=$(uv run python scripts/train_status.py --run "$RUN" --log "$LOG" --gate "$GATE" 2>&1)
  validations=$(grep -a "AV " "$LOG" 2>/dev/null | tail -5)
  steps=$(grep -aE "^  step " "$LOG" 2>/dev/null | tail -3)
  alive=$(pgrep -fc "[t]rain_c2.py" 2>/dev/null || echo 0)
  guard=$(pgrep -fc "[s]upervise_c2.sh" 2>/dev/null || echo 0)

  clear
  printf '  VisioVox  %s\n' "$(basename "$RUN")"
  printf '  %s  |  trainer %s  |  supervisor %s\n' \
    "$(date '+%a %d %b  %H:%M:%S')" \
    "$([ "$alive" -gt 0 ] && echo up || echo DOWN)" \
    "$([ "$guard" -gt 0 ] && echo up || echo down)"
  printf '  %s\n\n' "$rule"
  printf '%s\n' "$status"
  if [ -n "$validations" ]; then
    printf '\n  recent validations\n%s\n' "$(printf '%s\n' "$validations" | sed 's/^ */    /')"
  fi
  if [ -n "$steps" ]; then
    printf '\n  latest steps\n%s\n' "$(printf '%s\n' "$steps" | sed 's/^ */    /')"
  fi
  printf '\n  %s\n' "$rule"
  printf '  refreshing every %ss — Ctrl-C to stop\n' "$INTERVAL"

  sleep "$INTERVAL"
done
