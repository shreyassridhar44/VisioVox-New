#!/usr/bin/env bash
# One-command deploy (docs/28 §W9).
#
#   ./infra/local/deploy.sh            # build, migrate, start
#   ./infra/local/deploy.sh --tunnel   # ...and bring up cloudflared
#   ./infra/local/deploy.sh --rollback # previous images, same data
#
# Refuses rather than guesses: a missing secret, a root-owned media volume or a
# failed migration stops the deploy instead of producing a half-running stack
# that looks fine until someone uploads something.
set -euo pipefail

cd "$(dirname "$0")/../.."
COMPOSE=(docker compose -f infra/docker/compose.prod.yaml --project-directory .)
PROFILES=()
[[ "${1:-}" == "--tunnel" ]] && PROFILES=(--profile tunnel)

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }
die() { printf '\033[31mERROR\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- preflight --
say "Preflight"

[[ -f .env ]] || die ".env is missing. Copy .env.production.example and fill it in."

required=(
  POSTGRES_PASSWORD
  AUTH_SECRET
  S3_ACCESS_KEY_ID
  S3_SECRET_ACCESS_KEY
  PUBLIC_API_URL
  PUBLIC_MEDIA_BASE_URL
  AUDIT_IP_SALT
)
missing=()
for key in "${required[@]}"; do
  grep -qE "^${key}=.+" .env || missing+=("$key")
done
[[ ${#missing[@]} -eq 0 ]] || die "missing in .env: ${missing[*]}"

# The dev defaults are in the repo, so they are the ones most likely to reach
# production by accident.
if grep -qE '^AUTH_SECRET=dev-only' .env; then
  die "AUTH_SECRET is still the development default. Generate one: openssl rand -hex 32"
fi
if grep -qE '^AUDIT_IP_SALT=dev-only' .env; then
  die "AUDIT_IP_SALT is still the development default. Generate one: openssl rand -hex 32"
fi

MEDIA_ROOT=$(grep -E '^MEDIA_ROOT=' .env | cut -d= -f2- || echo /srv/media)
MEDIA_ROOT=${MEDIA_ROOT:-/srv/media}
[[ -d "$MEDIA_ROOT" ]] || die "$MEDIA_ROOT does not exist. Run infra/local/setup-media-volume.sh"

# The reading must come from the media volume, not from /. See docs/track-w/W0.
if [[ "$(stat -c %d "$MEDIA_ROOT")" == "$(stat -c %d /)" ]]; then
  die "$MEDIA_ROOT is on the same filesystem as /. The media volume is not mounted."
fi

avail=$(df -BG --output=avail "$MEDIA_ROOT" | tail -1 | tr -dc '0-9')
say "Media volume: ${avail} GB free"
[[ "$avail" -ge 20 ]] || die "under 20 GB free on $MEDIA_ROOT; refusing to deploy"

mkdir -p "$MEDIA_ROOT"/{work,projects,exports,minio,postgres,redis}

# ------------------------------------------------------------------ rollback --
if [[ "${1:-}" == "--rollback" ]]; then
  say "Rolling back to the previously tagged images"
  docker image inspect visiovox-prod-api:previous >/dev/null 2>&1 \
    || die "no previous image tagged; nothing to roll back to"
  docker tag visiovox-prod-api:previous visiovox-prod-api:latest
  docker tag visiovox-prod-web:previous visiovox-prod-web:latest
  "${COMPOSE[@]}" up -d --no-build api web
  say "Rolled back. Data and migrations are untouched - roll forward if a migration is the problem."
  exit 0
fi

# --------------------------------------------------------------------- build --
say "Tagging the current images as :previous"
for svc in api web; do
  if docker image inspect "visiovox-prod-${svc}:latest" >/dev/null 2>&1; then
    docker tag "visiovox-prod-${svc}:latest" "visiovox-prod-${svc}:previous"
  fi
done

say "Building"
"${COMPOSE[@]}" build

say "Starting data services"
"${COMPOSE[@]}" up -d postgres redis minio

# ----------------------------------------------------------------- migrate --
# Separate from the API so a failed migration stops the deploy rather than
# crash-looping a service behind a healthcheck.
say "Running migrations"
"${COMPOSE[@]}" run --rm migrate || die "migrations failed; the previous release is still running"

# ------------------------------------------------------------------- start --
say "Starting application services"
"${COMPOSE[@]}" "${PROFILES[@]}" up -d

say "Waiting for readiness"
for _ in $(seq 1 60); do
  if "${COMPOSE[@]}" exec -T api curl -fsS http://localhost:8000/readyz >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

"${COMPOSE[@]}" exec -T api curl -fsS http://localhost:8000/readyz >/dev/null 2>&1 \
  || die "the API never became ready. Logs: ${COMPOSE[*]} logs api"

say "GPU worker (runs on the host, not in compose)"
systemctl is-active --quiet visiovox-worker-gpu \
  && echo "  running" \
  || echo "  NOT running - sudo systemctl start visiovox-worker-gpu"

say "Deployed"
"${COMPOSE[@]}" ps --format 'table {{.Name}}\t{{.Status}}'
