$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    throw "SAM is not set up yet. Run .\setup.ps1 first."
}

$TestRunRoot = Join-Path $ProjectRoot ("work\test-runs\" + [guid]::NewGuid().ToString("N"))
$TestTemp = Join-Path $TestRunRoot "tmp"
$PytestBaseTemp = Join-Path $TestRunRoot "pytest"
New-Item -ItemType Directory -Path $TestTemp -Force | Out-Null
$env:TEMP = $TestTemp
$env:TMP = $TestTemp

Push-Location $ProjectRoot
try {
    & $VenvPython -m pytest -q (Join-Path $ProjectRoot "tests") --basetemp $PytestBaseTemp -p no:cacheprovider
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
