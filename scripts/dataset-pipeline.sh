#!/usr/bin/env bash
# Chains fetch -> generate so dataset acquisition is fully unattended.
#
# Waits for an in-flight fetch (tmux session "librimix") if one exists, rather
# than starting a second downloader over the same files.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$HOME/logs/dataset-pipeline.log"
mkdir -p "$(dirname "$LOG")"
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

# Generation extracts the archives and then deletes them, so their absence is
# the normal steady state once generation has run. Judging "needs fetching" by
# archive presence made every re-run re-download 40 GB that was already on disk
# in extracted form. Extracted trees win.
if [ -d "$HOME/data/corpora/LibriSpeech/train-clean-360" ]    && [ -d "$HOME/data/corpora/wham_noise/tr" ]; then
  log "corpora already extracted; skipping fetch"
else
if tmux has-session -t librimix 2>/dev/null; then
  log "fetch already running; waiting for it to finish"
  while tmux has-session -t librimix 2>/dev/null; do sleep 60; done
else
  log "starting fetch"
  bash "$REPO/scripts/fetch-librimix.sh" || { log "fetch FAILED"; exit 1; }
fi

# The fetch is resumable, so "finished" means every archive is present at its
# full size -- not merely that the process exited.
if ! grep -q "all downloads finished" "$HOME/logs/librimix-fetch.log" 2>/dev/null; then
  log "ABORT: fetch did not report completion; re-run scripts/fetch-librimix.sh"
  exit 1
fi
log "fetch complete"

fi

log "starting generation"
bash "$REPO/scripts/generate_librimix.sh" || { log "generation FAILED"; exit 1; }
log "=== dataset pipeline complete ==="
