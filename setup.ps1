[CmdletBinding()]
param(
    [string]$Python,
    [switch]$InstallOllama,
    [switch]$PullRecommendedModel,
    [switch]$InstallLiteLLM,
    [switch]$InstallDesktopAutomation,
    [switch]$InstallLocalVoice,
    [string]$RecommendedModel = "qwen3.5:4b"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = $PSScriptRoot
$VirtualEnvironment = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VirtualEnvironment "Scripts\python.exe"
$SetupOllamaProcess = $null

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

function Test-SamOllamaApi {
    try {
        $null = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 2
        return $true
    } catch {
        return $false
    }
}

function Resolve-SamPython {
    if ($Python) {
        if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
            throw "Python was not found at: $Python"
        }
        return @{ Executable = (Resolve-Path -LiteralPath $Python).Path; Prefix = @() }
    }

    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($PyLauncher) {
        return @{ Executable = $PyLauncher.Source; Prefix = @("-3") }
    }

    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($PythonCommand) {
        return @{ Executable = $PythonCommand.Source; Prefix = @() }
    }

    throw "Python 3.11 or newer is required. Install it, then run setup.ps1 again."
}

function Invoke-SamPython {
    param(
        [Parameter(Mandatory = $true)][hashtable]$Runtime,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )
    & $Runtime.Executable @($Runtime.Prefix) @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python exited with code $LASTEXITCODE."
    }
}

Write-Host "Preparing SAM..." -ForegroundColor Cyan
$Runtime = Resolve-SamPython
Invoke-SamPython -Runtime $Runtime -Arguments @(
    "-c",
    "import sys; assert sys.version_info >= (3, 11), 'SAM requires Python 3.11+'; print('Using Python', sys.version.split()[0])"
)

if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    Invoke-SamPython -Runtime $Runtime -Arguments @("-m", "venv", $VirtualEnvironment)
}

& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not update pip." }

$LockedRequirements = Join-Path $ProjectRoot "requirements-lock.txt"
& $VenvPython -m pip install -r $LockedRequirements
if ($LASTEXITCODE -ne 0) { throw "Could not install SAM's Python packages." }

if ($InstallLiteLLM) {
    & $VenvPython -m pip install "litellm[proxy]"
    if ($LASTEXITCODE -ne 0) { throw "LiteLLM installation did not complete successfully." }
}

if ($InstallDesktopAutomation) {
    & $VenvPython -m pip install pywinauto
    if ($LASTEXITCODE -ne 0) { throw "Desktop Automation installation did not complete successfully." }
}

if ($InstallLocalVoice) {
    & $VenvPython -m pip install faster-whisper silero-vad onnxruntime piper-tts sounddevice
    if ($LASTEXITCODE -ne 0) { throw "Local Voice installation did not complete successfully." }
}

$EnvironmentFile = Join-Path $ProjectRoot ".env"
$EnvironmentExample = Join-Path $ProjectRoot ".env.example"
if ((Test-Path -LiteralPath $EnvironmentExample) -and -not (Test-Path -LiteralPath $EnvironmentFile)) {
    Copy-Item -LiteralPath $EnvironmentExample -Destination $EnvironmentFile
    Write-Host "Created .env from the safe example."
}

foreach ($DirectoryName in @("data", "workspace", "work")) {
    $DirectoryPath = Join-Path $ProjectRoot $DirectoryName
    if (-not (Test-Path -LiteralPath $DirectoryPath)) {
        New-Item -ItemType Directory -Path $DirectoryPath | Out-Null
    }
}

$TestRunRoot = Join-Path $ProjectRoot ("work\test-runs\" + [guid]::NewGuid().ToString("N"))
$TestTemp = Join-Path $TestRunRoot "tmp"
$PytestBaseTemp = Join-Path $TestRunRoot "pytest"
New-Item -ItemType Directory -Path $TestTemp -Force | Out-Null
$env:TEMP = $TestTemp
$env:TMP = $TestTemp

if ($InstallOllama) {
    $OllamaExecutable = Resolve-SamOllama
    if ($OllamaExecutable) {
        Write-Host "Ollama is already available at: $OllamaExecutable"
    } else {
        $Winget = Get-Command winget -ErrorAction SilentlyContinue
        if (-not $Winget) {
            throw "winget is unavailable. Install Ollama manually or place the official portable build in tools\ollama-v<version>."
        }
        & $Winget.Source install --id Ollama.Ollama --exact --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) { throw "Ollama installation did not complete successfully." }
    }
}

if ($PullRecommendedModel) {
    $OllamaExecutable = Resolve-SamOllama
    if (-not $OllamaExecutable) {
        throw "Ollama is not available yet. Install/start Ollama, then run: ollama pull $RecommendedModel"
    }
    $OllamaUserProfile = Join-Path $ProjectRoot "data\ollama-user"
    $OllamaModels = Join-Path $ProjectRoot "data\ollama-models"
    New-Item -ItemType Directory -Path $OllamaUserProfile -Force | Out-Null
    New-Item -ItemType Directory -Path $OllamaModels -Force | Out-Null
    $PreviousUserProfile = $env:USERPROFILE
    $PreviousOllamaModels = $env:OLLAMA_MODELS
    try {
        $env:USERPROFILE = $OllamaUserProfile
        $env:OLLAMA_MODELS = $OllamaModels
        if (-not (Test-SamOllamaApi)) {
            $SetupOllamaProcess = Start-Process -FilePath $OllamaExecutable -ArgumentList "serve" -WindowStyle Hidden -PassThru
            foreach ($Attempt in 1..20) {
                if (Test-SamOllamaApi) { break }
                Start-Sleep -Milliseconds 500
            }
            if (-not (Test-SamOllamaApi)) { throw "Ollama's local API did not start." }
        }
        & $OllamaExecutable pull $RecommendedModel
    } finally {
        if ($SetupOllamaProcess -and -not $SetupOllamaProcess.HasExited) {
            Stop-Process -Id $SetupOllamaProcess.Id -Force -ErrorAction SilentlyContinue
        }
        $env:USERPROFILE = $PreviousUserProfile
        if ($null -eq $PreviousOllamaModels) { Remove-Item Env:OLLAMA_MODELS -ErrorAction SilentlyContinue }
        else { $env:OLLAMA_MODELS = $PreviousOllamaModels }
    }
    if ($LASTEXITCODE -ne 0) { throw "The model download did not complete successfully." }
}

Push-Location $ProjectRoot
try {
    & $VenvPython -m pytest -q (Join-Path $ProjectRoot "tests") --basetemp $PytestBaseTemp -p no:cacheprovider
    if ($LASTEXITCODE -ne 0) { throw "SAM was installed, but its self-tests failed." }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "SAM is ready." -ForegroundColor Green
Write-Host "Start it with: .\start.ps1"
