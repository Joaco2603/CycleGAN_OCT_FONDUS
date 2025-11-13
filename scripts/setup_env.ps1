param(
    [string]$EnvName = ".venv"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

if (-not (Test-Path $EnvName)) {
    Write-Host "Creating virtual environment $EnvName"
    python -m venv $EnvName
}

$python = Join-Path $EnvName "Scripts/python.exe"
$pip = Join-Path $EnvName "Scripts/pip.exe"

& $python -m pip install --upgrade pip
& $pip install -r requirements.txt

Write-Host "Environment ready. Activate with:`n`n    $EnvName\Scripts\Activate.ps1`n"
