[CmdletBinding()]
param(
    # Start SAM in the background when Windows starts, so "Hey SAM" is always
    # listening. Pass -NoAutostart to only add the desktop and Start menu icons.
    [switch]$NoAutostart
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$VenvPythonW = Join-Path $ProjectRoot ".venv\Scripts\pythonw.exe"
$Launcher = Join-Path $ProjectRoot "desktop\sam_desktop.pyw"
if (-not (Test-Path -LiteralPath $VenvPythonW -PathType Leaf)) {
    throw "SAM is not set up yet. Run .\setup.ps1 first."
}

$AppDir = Join-Path $env:LOCALAPPDATA "SAM"
$Icon = Join-Path $AppDir "sam.ico"
New-Item -ItemType Directory -Path $AppDir -Force | Out-Null

# The tray icon (open / restart / quit) needs pystray. Without it SAM still
# starts and opens from the shortcut; only the tray menu is missing.
Write-Host "Installing the tray icon support..." -ForegroundColor Cyan
& $VenvPython -m pip install --disable-pip-version-check --quiet "pystray>=0.19"
if ($LASTEXITCODE -ne 0) {
    Write-Host "  pystray could not be installed; SAM will run without a tray icon." -ForegroundColor Yellow
}

& $VenvPython $Launcher --write-icon $Icon
if ($LASTEXITCODE -ne 0) { throw "Could not create SAM's icon." }

$Shell = New-Object -ComObject WScript.Shell
function New-SamShortcut([string]$Path, [string]$Arguments, [string]$Description) {
    $Link = $Shell.CreateShortcut($Path)
    $Link.TargetPath = $VenvPythonW
    $Link.Arguments = "`"$Launcher`" $Arguments".Trim()
    $Link.WorkingDirectory = $ProjectRoot
    $Link.IconLocation = "$Icon,0"
    $Link.Description = $Description
    $Link.Save()
    Write-Host "  $Path" -ForegroundColor DarkGray
}

Write-Host "Adding SAM's shortcuts:" -ForegroundColor Cyan
$Desktop = [Environment]::GetFolderPath("Desktop")
$StartMenu = Join-Path ([Environment]::GetFolderPath("Programs")) "SAM.lnk"
New-SamShortcut (Join-Path $Desktop "SAM.lnk") "" "Open SAM"
New-SamShortcut $StartMenu "" "Open SAM"

$Startup = Join-Path ([Environment]::GetFolderPath("Startup")) "SAM (background).lnk"
if ($NoAutostart) {
    if (Test-Path -LiteralPath $Startup) { Remove-Item -LiteralPath $Startup -Force }
    Write-Host "SAM will not start with Windows (-NoAutostart)." -ForegroundColor DarkGray
} else {
    New-SamShortcut $Startup "--background" "Start SAM in the background so Hey SAM is listening"
    Write-Host "SAM will start in the background when you sign in to Windows." -ForegroundColor DarkGray
}

Write-Host "Done. Open SAM from the desktop or the Start menu." -ForegroundColor Green
Write-Host "To remove the shortcuts: .\uninstall-desktop.ps1" -ForegroundColor DarkGray
