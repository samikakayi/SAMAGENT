[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)][int]$Port = 8765,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    throw "SAM is not set up yet. Run .\setup.ps1 first."
}

$HostAddress = "127.0.0.1"
$env:SAM_HOST = $HostAddress
$env:SAM_PORT = $Port.ToString()
$env:SAM_ALLOW_LAN = "false"
$LocalUrl = "http://127.0.0.1:$Port"
$BrowserJob = $null
$ManagedOllamaProcess = $null

function Get-RunningSam {
    # Any web server can answer /api/health with something. Reusing the port
    # only makes sense when the responder actually identifies itself as SAM,
    # otherwise an unrelated local app would be mistaken for a running instance.
    foreach ($Attempt in 1..3) {
        try {
            $Response = Invoke-RestMethod -Uri "$LocalUrl/api/health" -TimeoutSec 2
            if ($Response.name -eq "SAM") { return $Response }
            return $null
        } catch {
            if ($Attempt -lt 3) { Start-Sleep -Milliseconds 250 }
        }
    }
    return $null
}

function Test-SamPortInUse {
    $Client = [System.Net.Sockets.TcpClient]::new()
    try {
        $Connection = $Client.ConnectAsync($HostAddress, $Port)
        return ($Connection.Wait(750) -and $Client.Connected)
    } catch {
        return $false
    } finally {
        $Client.Dispose()
    }
}

function Resolve-SamOllama {
    $SystemOllama = Get-Command ollama -ErrorAction SilentlyContinue
    if ($SystemOllama) { return $SystemOllama.Source }

    $Portable = Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "tools") -Directory -Filter "ollama-v*" -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        ForEach-Object { Join-Path $_.FullName "ollama.exe" } |
        Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
        Select-Object -First 1
    return $Portable
}

function Test-OllamaApi {
    try {
        $null = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 2
        return $true
    } catch {
        return $false
    }
}

# Starting SAM twice should be harmless: reuse the healthy instance instead of
# letting Uvicorn fail with Windows socket error 10048.
$RunningSam = Get-RunningSam
if ($null -ne $RunningSam) {
    # Starting twice must never produce a second backend, a second monitor
    # worker, or a second set of websocket consumers.
    Write-Host "SAM is already running at $LocalUrl - reusing it." -ForegroundColor Green
    Write-Host ("  database: {0}   audit chain: {1}   uptime: {2}s" -f `
        $RunningSam.database, $RunningSam.audit_chain_valid, [math]::Round($RunningSam.uptime_seconds, 0)) -ForegroundColor DarkGray
    if (-not $NoBrowser) {
        Start-Process $LocalUrl
    }
    exit 0
}

if (Test-SamPortInUse) {
    # A clean, actionable message beats a PowerShell stack trace here.
    Write-Host "Port $Port is already in use by another application (it does not identify itself as SAM)." -ForegroundColor Red
    Write-Host "  Close that application, or start SAM on another port:  .\start.ps1 -Port 8877" -ForegroundColor Yellow
    exit 1
}

$OllamaExecutable = Resolve-SamOllama
if ($OllamaExecutable -and -not (Test-OllamaApi)) {
    Write-Host "Starting local Ollama..." -ForegroundColor Cyan
    $OllamaUserProfile = Join-Path $ProjectRoot "data\ollama-user"
    $OllamaModels = Join-Path $ProjectRoot "data\ollama-models"
    New-Item -ItemType Directory -Path $OllamaUserProfile -Force | Out-Null
    New-Item -ItemType Directory -Path $OllamaModels -Force | Out-Null
    $PreviousUserProfile = $env:USERPROFILE
    $PreviousOllamaModels = $env:OLLAMA_MODELS
    try {
        # Keep portable Ollama's identity and model blobs inside SAM's data directory.
        $env:USERPROFILE = $OllamaUserProfile
        $env:OLLAMA_MODELS = $OllamaModels
        $ManagedOllamaProcess = Start-Process -FilePath $OllamaExecutable -ArgumentList "serve" -WindowStyle Hidden -PassThru
    } finally {
        $env:USERPROFILE = $PreviousUserProfile
        if ($null -eq $PreviousOllamaModels) { Remove-Item Env:OLLAMA_MODELS -ErrorAction SilentlyContinue }
        else { $env:OLLAMA_MODELS = $PreviousOllamaModels }
    }
    foreach ($Attempt in 1..20) {
        if (Test-OllamaApi) { break }
        Start-Sleep -Milliseconds 500
    }
    if (-not (Test-OllamaApi)) {
        if ($ManagedOllamaProcess -and -not $ManagedOllamaProcess.HasExited) {
            Stop-Process -Id $ManagedOllamaProcess.Id -Force -ErrorAction SilentlyContinue
        }
        throw "Ollama was found but its local API did not start."
    }
}

# Optional LiteLLM gateway. SAM works without it by talking to Ollama and
# OpenRouter directly, so a missing or failing gateway must never block startup.
$LiteLLM = Join-Path $ProjectRoot ".venv\Scripts\litellm.exe"
$LiteLLMConfig = Join-Path $ProjectRoot "litellm-config.yaml"
function Test-LiteLLMApi {
    try {
        $null = Invoke-RestMethod -Uri "http://127.0.0.1:4000/health/liveliness" -TimeoutSec 2
        return $true
    } catch {
        return $false
    }
}
if ((Test-Path -LiteralPath $LiteLLM -PathType Leaf) -and (Test-Path -LiteralPath $LiteLLMConfig -PathType Leaf)) {
    if (Test-LiteLLMApi) {
        Write-Host "LiteLLM gateway already running on 127.0.0.1:4000" -ForegroundColor Green
    } else {
        Write-Host "Starting the LiteLLM gateway..." -ForegroundColor Cyan
        # The gateway inherits this process's environment, so a key exported here
        # reaches it without ever being written into the config file.
        $ManagedLiteLLM = Start-Process -FilePath $LiteLLM `
            -ArgumentList "--config", $LiteLLMConfig, "--host", "127.0.0.1", "--port", "4000" `
            -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru
        foreach ($Attempt in 1..25) {
            if (Test-LiteLLMApi) { break }
            Start-Sleep -Milliseconds 600
        }
        if (Test-LiteLLMApi) {
            Write-Host "LiteLLM gateway is ready." -ForegroundColor Green
        } else {
            Write-Host "LiteLLM did not become ready; SAM will route directly to Ollama and OpenRouter." -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "LiteLLM is not installed; SAM will route directly to its providers." -ForegroundColor DarkGray
}

if (-not $NoBrowser) {
    $BrowserJob = Start-Job -ScriptBlock {
        param($Url)
        Start-Sleep -Seconds 2
        Start-Process $Url
    } -ArgumentList $LocalUrl
}

Write-Host "SAM is starting at $LocalUrl" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop it."

Push-Location $ProjectRoot
try {
    & $VenvPython -m sam_backend
}
finally {
    Pop-Location
    if ($null -ne $BrowserJob) {
        Remove-Job -Job $BrowserJob -Force -ErrorAction SilentlyContinue
    }
    if ($ManagedOllamaProcess -and -not $ManagedOllamaProcess.HasExited) {
        Stop-Process -Id $ManagedOllamaProcess.Id -Force -ErrorAction SilentlyContinue
    }
}
