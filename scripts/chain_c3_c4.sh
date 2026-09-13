#!/usr/bin/env bash
# Wait for C3, then start C4 — and resume anything the machine interrupts.
#
# Two problems this solves. The obvious one is that nobody is watching at
# 22:15 to launch the next stage. The other is that this machine has gone down
# four times mid-run; each time the fix was the same three commands, and each
# time the GPU sat idle until someone noticed. A supervisor turns both into a
# non-event.
#
# The distinction it relies on: a run that *finished* writes history.json,
# because that is the last thing the training script does. A run that was
# killed has checkpoints but no history. So "history.json exists" is a
# trustworthy completion signal, and anything else means resume.
#
# Deliberately not systemd or a cron job: this has to survive a WSL restart
# only as long as the work does, and a script that can be read in one screen
# is easier to trust than a unit file.
set -uo pipefail

REPO="$HOME/visiovox/VisioVox-New"
export PATH="$HOME/.local/bin:$PATH"
POLL=120
MAX_RESUMES=20

log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

running() { pgrep -f "[t]rain_$1.py" >/dev/null; }

# Wait for a stage to leave the process table, resuming it if it died before
# writing history.json. Returns once the stage has genuinely completed.
await_stage() {
  local stage="$1" resumes=0
  while true; do
    if running "$stage"; then
      sleep "$POLL"
      continue
    fi
    if [ -f "$HOME/runs/$stage/history.json" ]; then
      log "$stage completed"
      return 0
    fi
    if [ ! -f "$HOME/runs/$stage/last.pt" ]; then
      log "$stage is not running and has no checkpoint — giving up"
      return 1
    fi
    resumes=$((resumes + 1))
    if [ "$resumes" -gt "$MAX_RESUMES" ]; then
      log "$stage resumed $MAX_RESUMES times without finishing — giving up"
      return 1
    fi
    log "$stage stopped early (resume $resumes/$MAX_RESUMES); restarting"
    cd "$REPO" || return 1
    setsid nohup uv run python "scripts/train_$stage.py" --resume \
      >> "$HOME/logs/$stage.log" 2>&1 < /dev/null &
    disown
    sleep 60
  done
}

log "supervisor started"

if ! await_stage c3; then
  log "aborting: c3 did not complete"
  exit 1
fi

BEST="$HOME/runs/c3/best.pt"
if [ ! -f "$BEST" ]; then
  log "aborting: $BEST missing"
  exit 1
fi

# Snapshot before C4 can touch anything. The C3 result is the evidence for the
# whole realistic-simulation stage and is not cheaply reproducible.
cp -n "$BEST" "$HOME/runs/c3/best-final.pt" 2>/dev/null || true

if running c4 || [ -f "$HOME/runs/c4/history.json" ]; then
  log "c4 already running or complete; nothing to do"
  exit 0
fi

log "starting c4 from $BEST"
cd "$REPO" || exit 1
mkdir -p "$HOME/runs/c4"
setsid nohup uv run python scripts/train_c4.py --init-from "$BEST" \
  > "$HOME/logs/c4.log" 2>&1 < /dev/null &
disown
sleep 60
if running c4; then
  log "c4 running"
else
  log "c4 failed to start — see ~/logs/c4.log"
  exit 1
fi

await_stage c4 && log "c4 completed"
