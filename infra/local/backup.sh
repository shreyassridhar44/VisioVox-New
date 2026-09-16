#!/usr/bin/env bash
# Backup and restore drill (docs/28 §W9).
#
#   ./infra/local/backup.sh            # take a backup
#   ./infra/local/backup.sh --verify   # take one, then RESTORE it into a
#                                      # throwaway database and check it
#
# An untested backup is not a backup, which is why --verify exists and why it
# actually restores rather than checking the file is non-empty. Run it after any
# schema change.
set -euo pipefail

cd "$(dirname "$0")/../.."
COMPOSE=(docker compose -f infra/docker/compose.prod.yaml --project-directory .)

MEDIA_ROOT=$(grep -E '^MEDIA_ROOT=' .env 2>/dev/null | cut -d= -f2- || echo /srv/media)
MEDIA_ROOT=${MEDIA_ROOT:-/srv/media}
DEST="$MEDIA_ROOT/backups"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
FILE="$DEST/visiovox-$STAMP.sql.gz"
KEEP=14

USER=$(grep -E '^POSTGRES_USER=' .env 2>/dev/null | cut -d= -f2- || echo visiovox)
DB=$(grep -E '^POSTGRES_DB=' .env 2>/dev/null | cut -d= -f2- || echo visiovox)
USER=${USER:-visiovox}
DB=${DB:-visiovox}

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }

mkdir -p "$DEST"

say "Dumping $DB"
"${COMPOSE[@]}" exec -T postgres pg_dump -U "$USER" --clean --if-exists "$DB" | gzip > "$FILE"
say "Wrote $FILE ($(du -h "$FILE" | cut -f1))"

# The media itself is not dumped: it is large, it is already on a separate
# volume, and a source recording can be re-uploaded. The database is what cannot
# be reconstructed - it holds who owns what, which jobs ran, and the audit log.
say "Note: media is not in this backup by design; only the database is."

say "Pruning backups older than the most recent $KEEP"
ls -1t "$DEST"/visiovox-*.sql.gz 2>/dev/null | tail -n +$((KEEP + 1)) | xargs -r rm --

if [[ "${1:-}" == "--verify" ]]; then
  SCRATCH="verify_$STAMP"
  say "Restore drill: loading the dump into $SCRATCH"

  "${COMPOSE[@]}" exec -T postgres psql -U "$USER" -d postgres \
    -c "DROP DATABASE IF EXISTS $SCRATCH;" -c "CREATE DATABASE $SCRATCH;" >/dev/null

  # --clean/--if-exists emit notices for objects that do not exist in a fresh
  # database; those are expected and are not the failure this is looking for.
  if ! gunzip -c "$FILE" | "${COMPOSE[@]}" exec -T postgres psql -U "$USER" -d "$SCRATCH" -q >/dev/null 2>&1; then
    say "RESTORE FAILED"
    "${COMPOSE[@]}" exec -T postgres psql -U "$USER" -d postgres -c "DROP DATABASE IF EXISTS $SCRATCH;" >/dev/null
    exit 1
  fi

  say "Restored. Checking the tables that matter survived:"
  for table in users projects jobs exports share_links audit_events; do
    count=$("${COMPOSE[@]}" exec -T postgres psql -U "$USER" -d "$SCRATCH" -tAc \
      "select count(*) from $table" 2>/dev/null || echo "MISSING")
    printf '  %-14s %s\n' "$table" "$count"
    [[ "$count" == "MISSING" ]] && { say "table $table did not restore"; exit 1; }
  done

  "${COMPOSE[@]}" exec -T postgres psql -U "$USER" -d postgres \
    -c "DROP DATABASE IF EXISTS $SCRATCH;" >/dev/null
  say "Restore drill passed."
fi
