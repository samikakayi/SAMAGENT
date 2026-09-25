<#
.SYNOPSIS
  Remove SAM 2's shortcuts (Desktop, Start menu, Startup). Nothing else.

.DESCRIPTION
  SAM itself, its .venv, its data (data\sam2.sqlite3), the key store, the
  icon and the OmniRoute gateway are left exactly as they are. By default
  only shortcuts that start SAM 2 (their arguments name SAM.pyw) are removed,
  so a shortcut of the same name that belongs to something else stays; -All
  removes the three shortcut names whatever they point to.
  A running SAM keeps running until you quit it from its tray icon.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\uninstall.ps1
#>
[CmdletBinding()]
param(
    [switch]$All,
    [switch]$DryRun,
    # Test hooks (default: the real Windows folders).
    [string]$DesktopDir = "",
    [string]$ProgramsDir = "",
    [string]$StartupDir = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not $DesktopDir) { $DesktopDir = [Environment]::GetFolderPath("Desktop") }
if (-not $ProgramsDir) { $ProgramsDir = [Environment]::GetFolderPath("Programs") }
if (-not $StartupDir) { $StartupDir = [Environment]::GetFolderPath("Startup") }

$Shortcuts = @(
    (Join-Path $DesktopDir "SAM.lnk"),
    (Join-Path $ProgramsDir "SAM.lnk"),
    (Join-Path $StartupDir "SAM (background).lnk")
)
$Shell = New-Object -ComObject WScript.Shell
$removed = 0
foreach ($Path in $Shortcuts) {
    if (-not (Test-Path -LiteralPath $Path)) { continue }
    $isSam2 = $false
    try { $isSam2 = $Shell.CreateShortcut($Path).Arguments -match "SAM\.pyw" } catch { }
    if (-not ($isSam2 -or $All)) {
        Write-Host "Kept $Path (it does not start SAM 2; use -All to remove it anyway)" -ForegroundColor DarkGray
        continue
    }
    if ($DryRun) {
        Write-Host "[dry run] would remove $Path" -ForegroundColor Yellow
    } else {
        Remove-Item -LiteralPath $Path -Force
        Write-Host "Removed $Path" -ForegroundColor DarkGray
    }
    $removed++
}
if ($DryRun) { Write-Host "Dry run finished: nothing was changed." -ForegroundColor Green }
elseif ($removed) { Write-Host "SAM's shortcuts are removed. SAM keeps running until you quit it from its tray icon." -ForegroundColor Green }
else { Write-Host "No SAM 2 shortcuts were found." -ForegroundColor Green }
