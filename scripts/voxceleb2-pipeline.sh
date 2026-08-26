#!/usr/bin/env bash
# Chain: fetch VoxCeleb2 -> extract -> pack mouth ROIs. Unattended.
#
# Waits for an in-flight fetch rather than starting a competing one over the
# same files, the same way the LibriMix chain does.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$HOME/logs/voxceleb2-pipeline.log"
mkdir -p "$(dirname "$LOG")"
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

cd "$REPO"
export PATH="$HOME/.local/bin:$PATH"
set -a; [ -f .env.local ] && . ./.env.local; set +a

if tmux has-session -t vox 2>/dev/null; then
  log "fetch already running; waiting"
  while tmux has-session -t vox 2>/dev/null; do sleep 60; done
else
  log "starting fetch"
  uv run python scripts/fetch_voxceleb2.py --set test >> "$LOG" 2>&1 || { log "fetch FAILED"; exit 1; }
fi

# "finished" means both archives are present at a plausible size, not merely
# that the process exited — the fetch is resumable, so an interrupted run also
# exits cleanly.
for f in vox2_test_aac.zip vox2_test_mp4.zip; do
  p="$HOME/data/voxceleb2/$f"
  if [ ! -s "$p" ] || [ "$(stat -c %s "$p")" -lt 1000000000 ]; then
    log "ABORT: $f missing or too small; re-run scripts/fetch_voxceleb2.py"
    exit 1
  fi
done
log "archives present"

log "extracting and packing mouth ROIs"
uv run python scripts/prepare_voxceleb2.py --split test >> "$LOG" 2>&1 || { log "prepare FAILED"; exit 1; }
log "=== voxceleb2 pipeline complete ==="
