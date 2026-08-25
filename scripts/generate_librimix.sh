#!/usr/bin/env bash
# Generate Libri2Mix from LibriSpeech + WHAM! noise.
#
# ONE configuration only: 16 kHz, min mode, mix_both. The upstream
# generate_librimix.sh produces 8k+16k x min+max x 3 types, which is ~500 GB+
# of which we would use a fraction (docs/06 §2 "Storage discipline").
#
# Libri3Mix is deliberately skipped: docs/25 §6 simulates 3-speaker mixtures on
# the fly instead of storing them.
#
# Resumable: extraction and generation both skip work already done.
set -euo pipefail

STORAGE="${STORAGE:-$HOME/data/corpora}"
OUTDIR="${OUTDIR:-$HOME/data/Libri2Mix}"
LIBRIMIX_SRC="${LIBRIMIX_SRC:-$HOME/src/LibriMix}"
LOG="${LOG:-$HOME/logs/librimix-generate.log}"
KEEP_ARCHIVES="${KEEP_ARCHIVES:-0}"

mkdir -p "$OUTDIR" "$(dirname "$LOG")"
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

require() {
  local f="$STORAGE/$1"
  [ -s "$f" ] || { log "MISSING $f — run scripts/fetch-librimix.sh first"; exit 1; }
}

log "=== generation begin ==="
for f in dev-clean.tar.gz test-clean.tar.gz train-clean-100.tar.gz \
         train-clean-360.tar.gz wham_noise.zip; do
  require "$f"
done

# ---- extract (idempotent) -------------------------------------------------
if [ ! -d "$STORAGE/LibriSpeech/train-clean-360" ]; then
  for f in dev-clean test-clean train-clean-100 train-clean-360; do
    if [ ! -d "$STORAGE/LibriSpeech/$f" ]; then
      log "extracting $f"
      tar -xzf "$STORAGE/$f.tar.gz" -C "$STORAGE"
    fi
  done
fi

if [ ! -d "$STORAGE/wham_noise" ]; then
  log "extracting wham_noise"
  unzip -qn "$STORAGE/wham_noise.zip" -d "$STORAGE"
fi

log "extracted; free: $(df -h "$STORAGE" | awk 'NR==2{print $4}')"

if [ "$KEEP_ARCHIVES" != "1" ]; then
  log "removing archives (re-downloadable; set KEEP_ARCHIVES=1 to keep)"
  rm -f "$STORAGE"/{dev-clean,test-clean,train-clean-100,train-clean-360}.tar.gz \
        "$STORAGE/wham_noise.zip"
fi

# ---- augment noise --------------------------------------------------------
# Creates the high-frequency-extended noise the 16k config needs. Idempotent
# in effect, but slow, so it is stamped.
STAMP="$STORAGE/.wham_augmented"
if [ ! -f "$STAMP" ]; then
  log "augmenting WHAM! noise for 16 kHz"
  python "$LIBRIMIX_SRC/scripts/augment_train_noise.py" --wham_dir "$STORAGE/wham_noise"
  touch "$STAMP"
else
  log "noise augmentation already done"
fi

# ---- generate -------------------------------------------------------------
log "generating Libri2Mix — 16k / min / mix_both"
python "$LIBRIMIX_SRC/scripts/create_librimix_from_metadata.py" \
  --librispeech_dir "$STORAGE/LibriSpeech" \
  --wham_dir "$STORAGE/wham_noise" \
  --metadata_dir "$LIBRIMIX_SRC/metadata/Libri2Mix" \
  --librimix_outdir "$OUTDIR" \
  --n_src 2 \
  --freqs 16k \
  --modes min \
  --types mix_both

log "=== done; output $(du -sh "$OUTDIR" | cut -f1); free: $(df -h "$OUTDIR" | awk 'NR==2{print $4}') ==="
