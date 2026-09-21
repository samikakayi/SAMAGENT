[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)][int]$Port = 8765
)

$ErrorActionPreference = "Continue"
$ProjectRoot = $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

Write-Host "SAM diagnostics (read-only)" -ForegroundColor Cyan
Write-Host "Project: $ProjectRoot"

if (Test-Path -LiteralPath $VenvPython -PathType Leaf) {
    & $VenvPython -c "import sys; print('Python:', sys.version.split()[0]); import fastapi; print('FastAPI:', fastapi.__version__); import MetaTrader5 as mt5; print('MetaTrader5:', mt5.__version__)"
} else {
    Write-Warning "The virtual environment is missing. Run .\setup.ps1"
}

$SystemOllama = Get-Command ollama -ErrorAction SilentlyContinue
$OllamaExecutable = if ($SystemOllama) { $SystemOllama.Source } else {
    Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "tools") -Directory -Filter "ollama-v*" -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        ForEach-Object { Join-Path $_.FullName "ollama.exe" } |
        Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
        Select-Object -First 1
}
if ($OllamaExecutable) {
    Write-Host "Ollama executable: $OllamaExecutable"
    & $OllamaExecutable list
} else {
    Write-Warning "Ollama is not installed or is not on PATH."
}

$TradingView = Get-Process -Name TradingView -ErrorAction SilentlyContinue
Write-Host "TradingView processes: $(@($TradingView).Count)"
$MetaTrader = Get-Process -Name terminal64 -ErrorAction SilentlyContinue
Write-Host "MetaTrader processes: $(@($MetaTrader).Count)"

try {
    $Health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 3
    Write-Host "SAM service: ONLINE · version $($Health.version) · uptime $($Health.uptime_seconds)s" -ForegroundColor Green
} catch {
    Write-Warning "SAM service is not responding on port $Port."
}
