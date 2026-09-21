[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)][int]$Port = 4000
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$LiteLLM = Join-Path $ProjectRoot ".venv\Scripts\litellm.exe"
$ConfigPath = Join-Path $ProjectRoot "litellm-config.yaml"
if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    throw "SAM is not set up. Run .\setup.ps1 first."
}

if (-not (Test-Path -LiteralPath $LiteLLM -PathType Leaf)) {
    throw "LiteLLM is optional and not installed. Run .\setup.ps1 -InstallLiteLLM"
}

Write-Host "LiteLLM is starting on http://127.0.0.1:$Port" -ForegroundColor Cyan
& $LiteLLM --config $ConfigPath --host 127.0.0.1 --port $Port
