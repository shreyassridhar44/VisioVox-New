<#
.SYNOPSIS
  Create and attach the VisioVox media volume (docs/track-w/W0).

.DESCRIPTION
  Media lives on a dedicated ext4 volume, never on the distro's own vhdx. Two
  reasons, and the second is the one that bites:

    1. D: hosts the distro vhdx and has ~5 GB free. There is no room there.
    2. `df` inside the distro reports the vhdx's VIRTUAL maximum, not real host
       free space - it will cheerfully report hundreds of GB on a full drive. A
       separate volume is what makes the disk check mean something.

  Creates an expandable vhdx on E: (it consumes only what is used), attaches it
  to the WSL2 VM, and optionally registers a logon task so the attach survives a
  reboot - `wsl --mount` does not persist on its own.

  Safe to re-run: it never overwrites an existing vhdx, and skips steps already
  done.

.NOTES
  MUST RUN ELEVATED. `wsl --mount` and `diskpart` both require Administrator.
  Hyper-V is not required; this uses diskpart, not New-VHD.

.EXAMPLE
  .\create-media-volume.ps1
  .\create-media-volume.ps1 -SizeGB 200 -RegisterLogonTask
#>
[CmdletBinding()]
param(
    [string]$VhdPath  = 'E:\wsl\media.vhdx',
    [int]$SizeGB      = 250,
    [string]$Distro   = 'VisioVox',
    [switch]$RegisterLogonTask
)

$ErrorActionPreference = 'Stop'

function Assert-Elevated {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($id)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'This script must run in an elevated PowerShell (Run as Administrator).'
    }
}

Assert-Elevated

$vhdDir = Split-Path -Parent $VhdPath
if (-not (Test-Path $vhdDir)) {
    Write-Host "Creating $vhdDir"
    New-Item -ItemType Directory -Path $vhdDir -Force | Out-Null
}

# --- guard against filling the drive we are trying to move onto -------------
$driveLetter = (Split-Path -Qualifier $VhdPath).TrimEnd(':')
$free = (Get-PSDrive $driveLetter).Free / 1GB
Write-Host ("Free on {0}: {1:N1} GB" -f $driveLetter, $free)
if ($free -lt 40) {
    throw "Only $([math]::Round($free,1)) GB free on ${driveLetter}: - refusing to add a volume there."
}

# --- create -----------------------------------------------------------------
if (Test-Path $VhdPath) {
    Write-Host "vhdx already exists at $VhdPath - not recreating."
} else {
    $sizeMB = $SizeGB * 1024
    $script = Join-Path $env:TEMP 'visiovox-create-vdisk.txt'
    # Expandable: the file grows with use rather than reserving $SizeGB up front.
    "create vdisk file=`"$VhdPath`" maximum=$sizeMB type=expandable" | Set-Content -Path $script -Encoding ascii
    Write-Host "Creating $VhdPath ($SizeGB GB, expandable)..."
    $out = & diskpart /s $script 2>&1
    Remove-Item $script -Force -ErrorAction SilentlyContinue
    if ($LASTEXITCODE -ne 0) { $out | Write-Host; throw "diskpart failed (exit $LASTEXITCODE)" }
    Write-Host 'Created.'
}

# --- attach -----------------------------------------------------------------
# --bare: hand WSL the raw block device. The filesystem does not exist yet on a
# fresh disk, so letting WSL try to mount one would fail.
Write-Host "Attaching to the WSL2 VM..."
& wsl.exe --mount --vhd $VhdPath --bare
if ($LASTEXITCODE -ne 0) {
    Write-Warning 'Attach failed. If it reports the disk is already attached, that is fine.'
}

# --- persistence ------------------------------------------------------------
if ($RegisterLogonTask) {
    $taskName = 'VisioVox-AttachMediaVolume'
    Write-Host "Registering logon task '$taskName'..."
    $action  = New-ScheduledTaskAction -Execute 'wsl.exe' -Argument "--mount --vhd `"$VhdPath`" --bare"
    $trigger = New-ScheduledTaskTrigger -AtLogOn
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -RunLevel Highest
    $settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
        -Principal $principal -Settings $settings -Force | Out-Null
    Write-Host 'Registered. The volume will re-attach at logon.'
} else {
    Write-Host ''
    Write-Warning 'No logon task registered. `wsl --mount` does NOT survive a reboot:'
    Write-Warning 'after restarting, /srv/media would be an empty directory and MinIO would'
    Write-Warning 'silently write into the distro vhdx instead. Re-run with -RegisterLogonTask.'
}

Write-Host ''
Write-Host 'Next, inside the distro:'
Write-Host "  wsl -d $Distro -- sudo bash `$HOME/visiovox/VisioVox-New/infra/local/setup-media-volume.sh"
