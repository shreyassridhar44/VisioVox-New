#!/usr/bin/env bash
# Create and mount the VisioVox media volume (docs/track-w/W0).
#
#   sudo bash infra/local/setup-media-volume.sh
#
# Two modes, chosen automatically:
#
#   vhdx  - a dedicated vhdx already attached by create-media-volume.ps1.
#           Native ext4 on NVMe. Needs a one-time elevated setup.
#   loop  - an ext4 image file on E:, mounted via a loop device. Needs no
#           Administrator at all. Measured at 1.4 GB/s write, 5.2 GB/s read,
#           so the 9p boundary costs far less than expected for the large
#           sequential objects MinIO deals in.
#
# Idempotent: never reformats a volume that already holds a filesystem, never
# duplicates an fstab entry. Safe to re-run.

set -euo pipefail

MOUNT_POINT="${MOUNT_POINT:-/srv/media}"
IMG="${IMG:-/mnt/e/wsl/visiovox-media.img}"
SIZE_G="${SIZE_G:-200}"
MIN_FREE_G="${MIN_FREE_G:-60}"

[ "$(id -u)" -eq 0 ] || { echo "error: run with sudo" >&2; exit 1; }
log() { printf '\033[1m==>\033[0m %s\n' "$*"; }

UNIT="$(systemd-escape -p --suffix=mount "$MOUNT_POINT")"

if findmnt -n "$MOUNT_POINT" >/dev/null 2>&1; then
  log "$MOUNT_POINT is already mounted"
  df -h "$MOUNT_POINT"
  exit 0
fi

# --- pick a mode ------------------------------------------------------------
# A bare disk attached by `wsl --mount --vhd ... --bare` shows up as a disk with
# no filesystem and no mountpoint. If one is there, prefer it.
ROOT_DEV="$(findmnt -no SOURCE / | sed 's/[0-9]*$//')"
DEVICE=""
while read -r name type; do
  [ "$type" = "disk" ] || continue
  dev="/dev/$name"
  [ "$dev" = "$ROOT_DEV" ] && continue
  lsblk -no MOUNTPOINT "$dev" | grep -q . && continue
  DEVICE="$dev"
  break
done < <(lsblk -dno NAME,TYPE)

if [ -n "$DEVICE" ]; then
  log "Found an attached disk: $DEVICE (vhdx mode)"
  SOURCE="$DEVICE"
  FSTAB_OPTS="defaults,nofail,x-systemd.device-timeout=30"
else
  log "No attached disk; using an image file on E: (loop mode)"
  AVAIL_G=$(df -BG --output=avail "$(dirname "$IMG")" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
  mkdir -p "$(dirname "$IMG")"
  if [ ! -f "$IMG" ]; then
    [ "$AVAIL_G" -ge $((SIZE_G + MIN_FREE_G)) ] || {
      echo "error: only ${AVAIL_G}G free; refusing to leave under ${MIN_FREE_G}G" >&2
      exit 1
    }
    log "Creating ${SIZE_G}G image at $IMG"
    # Not sparse: drvfs/NTFS commits the space immediately. That is a feature
    # here - the capacity is reserved rather than discovered to be absent later.
    truncate -s "${SIZE_G}G" "$IMG"
  fi
  SOURCE="$IMG"
  FSTAB_OPTS="loop,nofail,x-systemd.requires=/mnt/e"
fi

# --- format only if unformatted --------------------------------------------
FSTYPE="$(blkid -o value -s TYPE "$SOURCE" 2>/dev/null || true)"
if [ -z "$FSTYPE" ]; then
  log "Formatting ext4"
  # -m 0: the default 5% root reserve would be 10 GB of unusable space here.
  # Headroom is managed by the API's disk_reserved_bytes, where it is visible.
  mkfs.ext4 -q -m 0 -L visiovox-media -E lazy_itable_init=1,lazy_journal_init=1 "$SOURCE"
elif [ "$FSTYPE" = "ext4" ]; then
  log "Already ext4 - not reformatting"
else
  echo "error: $SOURCE holds a '$FSTYPE' filesystem; refusing to overwrite" >&2
  exit 1
fi

# --- fstab ------------------------------------------------------------------
# UUID for a real device, because attach order decides whether it is sdc or sdd.
# The image file is referenced by path, since it has no stable device.
if [ -n "$DEVICE" ]; then
  SRC_SPEC="UUID=$(blkid -o value -s UUID "$SOURCE")"
else
  SRC_SPEC="$SOURCE"
fi

mkdir -p "$MOUNT_POINT"
if grep -qF "$MOUNT_POINT" /etc/fstab; then
  log "fstab entry already present"
else
  log "Adding fstab entry"
  printf '%s  %s  ext4  %s  0  0\n' "$SRC_SPEC" "$MOUNT_POINT" "$FSTAB_OPTS" >> /etc/fstab
fi

# --- mount, via systemd -----------------------------------------------------
# NOT a bare `mount`. Every `wsl.exe ... bash -c` invocation gets its OWN mount
# namespace, so a mount made that way is invisible to Docker, to the API and to
# the very next command - it looks like it worked and did not. systemd runs in
# PID 1's namespace, so a unit start puts the mount where everything can see it.
log "Mounting via systemd ($UNIT)"
systemctl daemon-reload
systemctl start "$UNIT"

# The volume is the application's own storage, so it belongs to the user the
# API and workers run as. Left owned by root, every job dies with EACCES
# creating its scratch directory, before stage S0 runs.
OWNER="${SUDO_USER:-$(logname 2>/dev/null || echo root)}"
chown "$OWNER:$OWNER" "$MOUNT_POINT"
chmod 0755 "$MOUNT_POINT"

mkdir -p "$MOUNT_POINT/work" "$MOUNT_POINT/projects" "$MOUNT_POINT/minio"
chown "$OWNER:$OWNER" "$MOUNT_POINT/work" "$MOUNT_POINT/projects"
chmod 0775 "$MOUNT_POINT/work" "$MOUNT_POINT/projects"
# MinIO runs as its own uid inside its container.
chmod 0777 "$MOUNT_POINT/minio"

# --- verify -----------------------------------------------------------------
# The whole point: this must not be the same filesystem as /. If it is, readings
# taken here are the vhdx's virtual maximum and mean nothing.
ROOT_ID=$(nsenter -t 1 -m -- stat -c %d /)
MEDIA_ID=$(nsenter -t 1 -m -- stat -c %d "$MOUNT_POINT")
if [ "$ROOT_ID" = "$MEDIA_ID" ]; then
  echo "error: $MOUNT_POINT is still on the same filesystem as / - mount did not take" >&2
  exit 1
fi

log "Mounted in PID 1's namespace, distinct from /"
nsenter -t 1 -m -- df -h "$MOUNT_POINT"
systemctl is-enabled "$UNIT" >/dev/null 2>&1 && log "Will remount at boot (WantedBy=local-fs.target)"

cat <<EOF

Next:
  1. Set MEDIA_ROOT=$MOUNT_POINT and MEDIA_MINIO_PATH=$MOUNT_POINT/minio in .env
     (repo root - Compose reads it via --project-directory .)
  2. make dev   # recreates MinIO on the media volume
EOF
