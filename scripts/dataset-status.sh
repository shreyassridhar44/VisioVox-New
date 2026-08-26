#!/usr/bin/env bash
# One-glance view of dataset acquisition, since it spans several multi-hour jobs.
set -uo pipefail
printf "%-26s %8s  %s\n" "ARTEFACT" "SIZE" "STATE"
check() {
  local path="$1" want="$2" label="$3"
  if [ -e "$path" ]; then
    local sz; sz=$(du -sh "$path" 2>/dev/null | cut -f1)
    local state="present"
    if [ -n "$want" ] && [ -f "$path" ]; then
      local b; b=$(stat -c %s "$path")
      [ "$b" -lt "$want" ] && state="INCOMPLETE"
    fi
    printf "%-26s %8s  %s\n" "$label" "$sz" "$state"
  else
    printf "%-26s %8s  %s\n" "$label" "-" "missing"
  fi
}
D="$HOME/data"
check "$D/corpora/train-clean-360.tar.gz" 23000000000 "LibriSpeech 360"
check "$D/corpora/wham_noise.zip"         17800000000 "WHAM! noise"
check "$D/librimix"                       ""          "LibriMix (generated)"
check "$D/voxceleb2/vox2_test_aac.zip"     2500000000 "VoxCeleb2 audio"
check "$D/voxceleb2/vox2_test_mp4.zip"     8300000000 "VoxCeleb2 video"
check "$D/voxceleb2/packed"               ""          "VoxCeleb2 mouth ROIs"
check "$D/ami/sets"                       ""          "AMI-Eval"
echo
pgrep -af "dataset-pipeline|voxceleb2-pipeline" >/dev/null \
  && echo "pipelines: running" || echo "pipelines: NOT running  (make datasets)"
