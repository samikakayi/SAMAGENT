<#
.SYNOPSIS
  Install SAM 2 for this Windows user: Python 3.13 venv, requirements, icon
  and the three shortcuts (Desktop, Start menu, Startup/background).

.DESCRIPTION
  Safe to run again at any time (idempotent):
  - .venv is created with Python 3.13 if missing; a .venv made with another
    Python (SAM v1 used 3.12) or a broken one is moved aside to
    .venv.old-<timestamp> and rebuilt. Nothing is deleted.
  - pip install -r requirements.txt (pinned versions; a no-op when satisfied).
  - The icon is written to %LOCALAPPDATA%\SAM\sam.ico.
  - Shortcuts "SAM.lnk" (Desktop), "SAM.lnk" (Start menu) and
    "SAM (background).lnk" (Startup) point to
    .venv\Scripts\pythonw.exe SAM.pyw --home <SamHome> [--background].
    They replace SAM v1's shortcuts of the same names (v1's
    install-desktop.ps1 made them), so v1 no longer starts at sign-in.
  Keys are never read, copied or printed here: SAM 2 reads them at runtime
  from <SamHome>\.env and <SamHome>\data\secrets.json.

.PARAMETER SamHome
  Folder holding .env and data\ (the keys). Default: the repository root; if
  that folder has no keys yet but the old SAM folder
  (%USERPROFILE%\Desktop\SAM-Agent) has them, that folder is used.

.PARAMETER NoAutostart
  Do not start SAM at sign-in (removes the Startup shortcut if present).

.PARAMETER DryRun
  Print every step without changing anything.

.PARAMETER StopV1
  If SAM v1 (sam_desktop.pyw, sam_backend, its LiteLLM proxy) still runs from
  this folder, stop those processes first. Without it the installer stops and
  asks you to quit v1 from its tray icon: v1 must not keep running on a moved
  virtual environment.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\install.ps1
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -DryRun
#>
[CmdletBinding()]
param(
    [string]$SamHome = "",
    [switch]$NoAutostart,
    [switch]$DryRun,
    [string]$Python = "",
    [switch]$SkipVenv,
    [switch]$SkipPip,
    [switch]$SkipCheck,
    # Stop SAM v1's processes that run from this folder instead of asking to quit them.
    [switch]$StopV1,
    # Test hooks (default: the real Windows folders).
    [string]$DesktopDir = "",
    [string]$ProgramsDir = "",
    [string]$StartupDir = "",
    [string]$IconPath = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $Venv "Scripts\python.exe"
$VenvPythonW = Join-Path $Venv "Scripts\pythonw.exe"
$Launcher = Join-Path $RepoRoot "SAM.pyw"
$Requirements = Join-Path $RepoRoot "requirements.txt"
$WantedPython = "3.13"

function Say([string]$Text, [string]$Color = "Gray") { Write-Host $Text -ForegroundColor $Color }
function Step([string]$Text) { Say "" ; Say "== $Text" "Cyan" }
function Plan([string]$Text) { Say "  [dry run] would $Text" "Yellow" }

function Get-PythonVersion([string]$Exe) {
    # Windows PowerShell 5.1 turns redirected stderr of a native command into a
    # terminating error under "Stop": use "Continue" around native probes.
    $ErrorActionPreference = "Continue"
    if (-not $Exe) { return $null }
    try {
        $out = & $Exe -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) { return ([string]$out).Trim() }
    } catch { }
    return $null
}

function Find-Python313 {
    $ErrorActionPreference = "Continue"
    $candidates = New-Object System.Collections.Generic.List[string]
    if ($Python) { $candidates.Add($Python) }
    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($py) {
        try {
            $exe = & $py.Source "-$WantedPython" -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $exe) { $candidates.Add(([string]$exe).Trim()) }
        } catch { }
    }
    $candidates.Add((Join-Path $env:LOCALAPPDATA "Programs\Python\Python313\python.exe"))
    $candidates.Add((Join-Path $env:ProgramFiles "Python313\python.exe"))
    $onPath = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($onPath) { $candidates.Add($onPath.Source) }
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf) -and
            ((Get-PythonVersion $candidate) -eq $WantedPython)) {
            return $candidate
        }
    }
    return $null
}

function Resolve-SamHome {
    if ($SamHome) { return (Resolve-Path -LiteralPath $SamHome).Path }
    $hasKeys = { param($dir) (Test-Path -LiteralPath (Join-Path $dir ".env")) -or
                             (Test-Path -LiteralPath (Join-Path $dir "data\secrets.json")) }
    if (& $hasKeys $RepoRoot) { return $RepoRoot }
    $old = Join-Path $env:USERPROFILE "Desktop\SAM-Agent"
    if ((Test-Path -LiteralPath $old) -and (& $hasKeys $old)) {
        Say "  No keys in $RepoRoot; using the folder that has them: $old" "DarkGray"
        return $old
    }
    return $RepoRoot
}

$Home2 = Resolve-SamHome
$AppDir = Join-Path $env:LOCALAPPDATA "SAM"
if (-not $IconPath) { $IconPath = Join-Path $AppDir "sam.ico" }
if (-not $DesktopDir) { $DesktopDir = [Environment]::GetFolderPath("Desktop") }
if (-not $ProgramsDir) { $ProgramsDir = [Environment]::GetFolderPath("Programs") }
if (-not $StartupDir) { $StartupDir = [Environment]::GetFolderPath("Startup") }

Say "SAM 2 installer$(if ($DryRun) { ' (DRY RUN: nothing will be changed)' })" "Green"
Say "  repository : $RepoRoot"
Say "  SAM_HOME   : $Home2"

# --- 0. SAM v1 must not be running ------------------------------------------------------------
# On the real switch-over the repository is v1's folder: its .venv (Python 3.12)
# runs sam_desktop.pyw, sam_backend (port 8877) and a LiteLLM proxy (port 4000).
# Windows lets step 1 rename that folder while they run, which would leave v1
# alive with its microphone open and sharing the free model quota until the
# next reboot (repair review 2026-09-24). So stop here and ask first.
function Get-V1Processes {
    $ErrorActionPreference = "Continue"
    $venvFull = [IO.Path]::GetFullPath($Venv)
    $venvVersion = if (Test-Path -LiteralPath $VenvPython -PathType Leaf) { Get-PythonVersion $VenvPython } else { $null }
    $venvMoves = ($venvVersion -ne $WantedPython) -and -not $SkipVenv
    $found = @()
    try { $procs = @(Get-CimInstance Win32_Process -ErrorAction Stop) } catch { return @() }
    foreach ($p in $procs) {
        $exe = [string]$p.ExecutablePath
        $cmd = [string]$p.CommandLine
        $inVenv = $exe -and $exe.StartsWith($venvFull, [StringComparison]::OrdinalIgnoreCase)
        $v1 = $cmd -match 'sam_backend|sam_desktop\.pyw|litellm'
        $here = $cmd.IndexOf($RepoRoot, [StringComparison]::OrdinalIgnoreCase) -ge 0
        if (($inVenv -and ($v1 -or $venvMoves)) -or ($v1 -and $here)) { $found += $p }
    }
    return $found
}
Step "SAM v1 (the old SAM) must be closed first"
$running = @(Get-V1Processes)
if ($running.Count -gt 0) {
    Say "  Still running from this folder:" "Yellow"
    foreach ($p in $running) {
        $line = [string]$p.CommandLine
        Say ("    pid {0}: {1}" -f $p.ProcessId, $line.Substring(0, [Math]::Min(120, $line.Length))) "Yellow"
    }
    if ($DryRun) {
        Plan "stop here: quit SAM v1 from its tray icon (or run with -StopV1) before installing"
    } elseif ($StopV1) {
        foreach ($p in $running) {
            try { Stop-Process -Id $p.ProcessId -ErrorAction Stop; Say "  stopped pid $($p.ProcessId)" }
            catch { Say "  could not stop pid $($p.ProcessId): $_" "Red" }
        }
        Start-Sleep -Seconds 2
    } else {
        Say "  Quit SAM v1 first: right-click its tray icon and choose Quit (or run this again with -StopV1)." "Yellow"
        throw "SAM v1 is still running from $RepoRoot; nothing was changed."
    }
} else {
    Say "  not running"
}

# --- 1. Python 3.13 venv ----------------------------------------------------------------------
Step "Python $WantedPython virtual environment"
if ($SkipVenv) {
    Say "  skipped (-SkipVenv)"
} else {
    $venvVersion = if (Test-Path -LiteralPath $VenvPython -PathType Leaf) { Get-PythonVersion $VenvPython } else { $null }
    if ($venvVersion -eq $WantedPython) {
        Say "  .venv is healthy (Python $venvVersion)"
    } else {
        $base = Find-Python313
        if (-not $base) {
            throw "Python $WantedPython was not found. Install it from https://www.python.org/downloads/ (or 'py install $WantedPython'), then run this again."
        }
        Say "  using $base"
        if (Test-Path -LiteralPath $Venv) {
            $aside = "$Venv.old-$(Get-Date -Format yyyyMMdd-HHmmss)"
            $why = if ($venvVersion) { "made with Python $venvVersion" } else { "broken" }
            if ($DryRun) { Plan "move the existing .venv ($why) to $aside" }
            else {
                Say "  existing .venv is ${why}: moving it to $aside" "Yellow"
                try { Rename-Item -LiteralPath $Venv -NewName (Split-Path -Leaf $aside) }
                catch { throw "Could not move the old .venv aside (is SAM or another program using it? close it and retry): $_" }
            }
        }
        if ($DryRun) { Plan "create .venv with $base" }
        else {
            & $base -m venv $Venv
            if ($LASTEXITCODE -ne 0) { throw "python -m venv failed ($LASTEXITCODE)" }
            Say "  created .venv" "Green"
        }
    }
}

# --- 2. requirements --------------------------------------------------------------------------
Step "Python packages (requirements.txt)"
if ($SkipPip) { Say "  skipped (-SkipPip)" }
elseif ($DryRun) { Plan "run: .venv\Scripts\python.exe -m pip install -r requirements.txt" }
else {
    & $VenvPython -m pip install --disable-pip-version-check -r $Requirements
    if ($LASTEXITCODE -ne 0) { throw "pip install failed ($LASTEXITCODE). Check the internet connection and run this again." }
    Say "  packages are installed" "Green"
}

# --- 3. icon ----------------------------------------------------------------------------------
Step "Icon"
if ($DryRun) { Plan "write $IconPath" }
else {
    if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) { throw ".venv is missing: run without -SkipVenv first." }
    & $VenvPython $Launcher --write-icon $IconPath
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $IconPath)) { throw "Could not write SAM's icon." }
    Say "  $IconPath"
}

# --- 4. shortcuts -----------------------------------------------------------------------------
Step "Shortcuts"
$Shell = $null
function New-SamShortcut([string]$Path, [string]$Arguments, [string]$Description) {
    if ($DryRun) { Plan "write $Path -> pythonw.exe $Arguments" ; return }
    if (-not $script:Shell) { $script:Shell = New-Object -ComObject WScript.Shell }
    $parent = Split-Path -Parent $Path
    if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    $link = $script:Shell.CreateShortcut($Path)
    $link.TargetPath = $VenvPythonW
    $link.Arguments = $Arguments
    $link.WorkingDirectory = $RepoRoot
    $link.IconLocation = "$IconPath,0"
    $link.Description = $Description
    $link.Save()
    Say "  $Path"
}
$LinkArgs = "`"$Launcher`" --home `"$Home2`""
New-SamShortcut (Join-Path $DesktopDir "SAM.lnk") $LinkArgs "SAM 2 - voice assistant and trading analyst"
New-SamShortcut (Join-Path $ProgramsDir "SAM.lnk") $LinkArgs "SAM 2 - voice assistant and trading analyst"
$startupLink = Join-Path $StartupDir "SAM (background).lnk"
if ($NoAutostart) {
    if (Test-Path -LiteralPath $startupLink) {
        if ($DryRun) { Plan "remove $startupLink" } else { Remove-Item -LiteralPath $startupLink -Force; Say "  removed $startupLink" }
    }
    Say "  SAM will not start at sign-in (-NoAutostart)."
} else {
    New-SamShortcut $startupLink "$LinkArgs --background" "Start SAM 2 at sign-in (island only)"
}

# --- 5. check ---------------------------------------------------------------------------------
Step "Check"
function Test-SamLoads {
    $ErrorActionPreference = "Continue"
    # --check prints booleans only (never a key value) and starts nothing.
    $json = & $VenvPython -m sam --check --home $Home2 2>$null
    if ($LASTEXITCODE -eq 0) { Say "  SAM 2 loads all its parts." "Green"; return }
    $failed = ""
    try { $failed = ((($json -join "`n") | ConvertFrom-Json).packages.failed) -join ", " } catch { }
    Say "  Some parts of SAM 2 did not load: $failed (details: %LOCALAPPDATA%\SAM2\logs\sam2.log)" "Yellow"
}
if ($SkipCheck -or $DryRun) { Say "  skipped" } else { Test-SamLoads }

Say ""
if ($DryRun) { Say "Dry run finished: nothing was changed." "Green" }
else {
    Say "Done. Start SAM from the desktop or the Start menu (shortcut 'SAM')." "Green"
    Say "SAM v1 no longer starts at sign-in." "DarkGray"
    Say "To remove the shortcuts: scripts\uninstall.ps1" "DarkGray"
}
