<#
.SYNOPSIS
  Run SAM 2 from source (a console you can read), its tests or its live
  acceptance checks.

.DESCRIPTION
  SAM_HOME is the folder holding .env and data\ (the keys and the DPAPI key
  store). Default: $env:SAM_HOME if set, else %USERPROFILE%\Desktop\SAM-Agent
  when it exists (the real keys, MetaTrader 5 and TradingView of this PC),
  else the repository root. Keys are never printed: SAM reads them itself.

.EXAMPLE
  scripts\dev.ps1                 # run SAM 2 with its UI, log to this console
.EXAMPLE
  scripts\dev.ps1 -NoUi           # core only (voice/tools/monitor); Ctrl+C quits
.EXAMPLE
  scripts\dev.ps1 -Check          # load every package, print a status (no key values)
.EXAMPLE
  scripts\dev.ps1 -Test           # pytest (no network, no speakers, no live apps)
.EXAMPLE
  scripts\dev.ps1 -Acceptance -Only launcher   # live checks on this PC
#>
[CmdletBinding()]
param(
    [string]$SamHome = "",
    [switch]$NoUi,
    [switch]$Check,
    [switch]$Test,
    [switch]$Acceptance,
    [string]$Only = "",
    [string]$TestArgs = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "No .venv yet: run scripts\install.ps1 -NoAutostart (or: py -3.13 -m venv .venv; .venv\Scripts\pip install -r requirements.txt)."
}
if (-not $SamHome) {
    $old = Join-Path $env:USERPROFILE "Desktop\SAM-Agent"
    if ($env:SAM_HOME) { $SamHome = $env:SAM_HOME }
    elseif (Test-Path -LiteralPath (Join-Path $old "data")) { $SamHome = $old }
    else { $SamHome = $RepoRoot }
}
$env:SAM_HOME = $SamHome
$env:PYTHONIOENCODING = "utf-8"
# Same default as SAM.pyw: numpy's OpenBLAS otherwise commits a buffer per CPU
# thread (measured: 852 MB -> 148 MB private bytes with 2 threads).
if (-not $env:OPENBLAS_NUM_THREADS) { $env:OPENBLAS_NUM_THREADS = "2" }
Write-Host "SAM_HOME = $SamHome" -ForegroundColor DarkGray

Push-Location $RepoRoot
try {
    if ($Test) {
        $basetemp = Join-Path $RepoRoot "work\pytest-dev"
        $extra = if ($TestArgs) { $TestArgs -split " " } else { @() }
        & $Python -m pytest --basetemp $basetemp -p no:cacheprovider @extra
    } elseif ($Acceptance) {
        $extra = if ($Only) { @("--only", $Only) } else { @() }
        & $Python (Join-Path $RepoRoot "acceptance\run_all.py") --home $SamHome @extra
    } elseif ($Check) {
        & $Python -m sam --check --home $SamHome
    } else {
        $extra = if ($NoUi) { @("--no-ui") } else { @() }
        & $Python -m sam --console --home $SamHome @extra
    }
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
