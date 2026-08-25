#!/usr/bin/env bash
# Download LibriSpeech + WHAM! noise for Libri2Mix generation.
# Resumable (wget -c): safe to re-run after a network drop.
set -u
STORAGE="$HOME/data/corpora"
LOG="$HOME/logs/librimix-fetch.log"
mkdir -p "$STORAGE"

SLR=https://www.openslr.org/resources/12
WHAM=https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip

log(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

fetch(){  # url
  local url="$1" name="${1##*/}"
  log "START $name"
  wget -c --tries=0 --read-timeout=30 --waitretry=10 \
       --progress=dot:giga -a "$LOG" "$url" -P "$STORAGE"
  log "DONE  $name rc=$?"
}

log "=== fetch begin; free: $(df -h "$STORAGE" | tail -1 | awk '{print $4}') ==="
fetch "$SLR/dev-clean.tar.gz"
fetch "$SLR/test-clean.tar.gz"
fetch "$SLR/train-clean-100.tar.gz"
fetch "$SLR/train-clean-360.tar.gz"
fetch "$WHAM"
log "=== all downloads finished; free: $(df -h "$STORAGE" | tail -1 | awk '{print $4}') ==="
ls -lh "$STORAGE" | tee -a "$LOG"
