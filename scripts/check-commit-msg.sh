#!/usr/bin/env bash
# Reject AI co-author trailers and generation footers in commit messages.
# Commits in this repository have a single author; see CONTRIBUTING.md.
set -euo pipefail

msg_file="$1"
# Strip comment lines (the commit template) before checking.
body="$(grep -v '^#' "$msg_file" || true)"

patterns=(
  'co-authored-by:[[:space:]]*claude'
  'co-authored-by:[[:space:]]*.*anthropic'
  'generated with[[:space:]]*\[?claude'
  '🤖[[:space:]]*generated'
  '^(assisted|generated)-by:'
)

for p in "${patterns[@]}"; do
  if printf '%s' "$body" | grep -qiE "$p"; then
    echo "commit-msg: rejected — matched forbidden attribution pattern: /$p/" >&2
    echo "This repository's commits carry a single author. Remove the trailer." >&2
    exit 1
  fi
done
exit 0
