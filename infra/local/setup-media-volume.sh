#!/usr/bin/env bash
# Format and mount the VisioVox media volume (docs/track-w/W0).
#
# Run AFTER create-media-volume.ps1 has attached the vhdx to the WSL2 VM:
#
#   sudo bash infra/local/setup-media-volume.sh
#
# Idempotent: an already-formatted volume is never reformatted, and an existing
# fstab entry is left alone. Re-run it after a reboot if the mount is missing.
#
# Why fstab is keyed on UUID: the attach order decides whether the volume shows
# up as /dev/sdc or /dev/sdd, so a device path would mount the wrong disk - or
# nothing - on some boots. `nofail` matters just as much: without it, a boot
# where the vhdx was not attached drops the distro into an emergency shell.

set -euo pipefail

MOUNT_POINT="${MOUNT_POINT:-/srv/media}"
EXPECTED_MIN_GB="${EXPECTED_MIN_GB:-100}"

if [ "$(id -u)" -ne 0 ]; then
  echo "error: run with sudo" >&2
  exit 1
fi

log() { printf '\033[1m==>\033[0m %s\n' "$*"; }

# --- already mounted? -------------------------------------------------------
if mountpoint -q "$MOUNT_POINT"; then
  log "$MOUNT_POINT is already mounted:"
  df -h "$MOUNT_POINT"
  exit 0
fi

# --- find the attached, unformatted or ext4 disk ----------------------------
# The media volume is the disk with no partition table that is not the rootfs.
log "Looking for the attached media disk..."
ROOT_DEV="$(findmnt -no SOURCE / | sed 's/[0-9]*$//')"
CANDIDATES=()
while read -r name type size; do
  [ "$type" = "disk" ] || continue
  dev="/dev/$name"
  [ "$dev" = "$ROOT_DEV" ] && continue
  # Skip anything holding a mounted filesystem already.
  if lsblk -no MOUNTPOINT "$dev" | grep -q .; then continue; fi
  CANDIDATES+=("$dev:$size")
done < <(lsblk -dno NAME,TYPE,SIZE)

if [ "${#CANDIDATES[@]}" -eq 0 ]; then
  cat >&2 <<'EOF'
error: no attached media disk found.

The vhdx is probably not attached to the WSL2 VM. From an ELEVATED PowerShell:

    wsl --mount --vhd E:\wsl\media.vhdx --bare

Remember this does not survive a reboot unless the logon task is registered.
EOF
  exit 1
fi

if [ "${#CANDIDATES[@]}" -gt 1 ]; then
  echo "error: more than one candidate disk; refusing to guess:" >&2
  printf '  %s\n' "${CANDIDATES[@]}" >&2
  echo "Set DEVICE=/dev/sdX explicitly and re-run." >&2
  exit 1
fi

DEVICE="${DEVICE:-${CANDIDATES[0]%%:*}}"
SIZE="${CANDIDATES[0]##*:}"
log "Using $DEVICE ($SIZE)"

# --- format only if it is not already ext4 ----------------------------------
FSTYPE="$(blkid -o value -s TYPE "$DEVICE" 2>/dev/null || true)"
if [ -z "$FSTYPE" ]; then
  log "Formatting $DEVICE as ext4 (this is a fresh volume)..."
  # -m 0: no root reserve. This volume holds media, not system files, and 5% of
  # 250 GB is 12 GB of nothing. Headroom is managed explicitly by the API's
  # disk_reserved_bytes instead, where it is visible and tunable.
  mkfs.ext4 -m 0 -L visiovox-media "$DEVICE"
elif [ "$FSTYPE" = "ext4" ]; then
  log "$DEVICE already holds an ext4 filesystem - not reformatting."
else
  echo "error: $DEVICE holds a '$FSTYPE' filesystem. Refusing to overwrite it." >&2
  exit 1
fi

# --- mount ------------------------------------------------------------------
UUID="$(blkid -o value -s UUID "$DEVICE")"
mkdir -p "$MOUNT_POINT"

if ! grep -q "$UUID" /etc/fstab; then
  log "Adding an fstab entry (by UUID, nofail)"
  printf 'UUID=%s  %s  ext4  defaults,nofail,x-systemd.device-timeout=30  0  2\n' \
    "$UUID" "$MOUNT_POINT" >> /etc/fstab
else
  log "fstab entry already present"
fi

log "Mounting $MOUNT_POINT"
mount "$MOUNT_POINT"

# MinIO runs as its own uid in the container; the API writes here too. 0775 with
# a shared group is the least-surprising arrangement for a single-host deploy.
chmod 0775 "$MOUNT_POINT"

# --- verify -----------------------------------------------------------------
AVAIL_GB=$(df -BG --output=avail "$MOUNT_POINT" | tail -1 | tr -dc '0-9')
log "Mounted. ${AVAIL_GB} GB available."

if [ "$AVAIL_GB" -lt "$EXPECTED_MIN_GB" ]; then
  echo "warning: only ${AVAIL_GB} GB available, expected at least ${EXPECTED_MIN_GB}" >&2
fi

# The whole point of the exercise: this must NOT be the same filesystem as /.
if [ "$(stat -c %d "$MOUNT_POINT")" = "$(stat -c %d /)" ]; then
  echo "error: $MOUNT_POINT is on the same filesystem as / - the mount did not take." >&2
  echo "Disk readings from here would be the vhdx's virtual maximum, which is the" >&2
  echo "exact bug this volume exists to prevent." >&2
  exit 1
fi

log "Verified: $MOUNT_POINT is a distinct filesystem from /."
df -h "$MOUNT_POINT"
