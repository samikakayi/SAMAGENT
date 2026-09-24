[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Removes only what install-desktop.ps1 added. SAM itself, its data and the
# OmniRoute gateway are left as they are.
$Shortcuts = @(
    (Join-Path ([Environment]::GetFolderPath("Desktop")) "SAM.lnk"),
    (Join-Path ([Environment]::GetFolderPath("Programs")) "SAM.lnk"),
    (Join-Path ([Environment]::GetFolderPath("Startup")) "SAM (background).lnk")
)
foreach ($Path in $Shortcuts) {
    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Force
        Write-Host "Removed $Path" -ForegroundColor DarkGray
    }
}
Write-Host "SAM's shortcuts are removed. SAM keeps running until you quit it from the tray or restart Windows." -ForegroundColor Green
