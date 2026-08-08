[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Bundle = Join-Path $Root 'bundle'
$FactorMiner = Join-Path $Bundle 'agent-framework\FactorMiner'
$Python = Join-Path $FactorMiner '.venv\Scripts\python.exe'

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is not in PATH. Install uv first, then rerun setup.ps1."
}
if (-not (Test-Path (Join-Path $Root 'runtime\.env'))) {
    throw "Missing runtime\.env. Create it locally from runtime\.env.template; do not put the key in source control."
}
Push-Location $FactorMiner
try {
    uv sync --locked --extra llm --no-dev --index "https://pypi.tuna.tsinghua.edu.cn/simple"
} finally {
    Pop-Location
}
if (-not (Test-Path $Python)) {
    throw "uv completed but the expected interpreter was not created: $Python"
}
& $Python (Join-Path $Root 'scripts\verify_bundle.py')
if ($LASTEXITCODE -ne 0) { throw "Bundle verification failed; no LLM request was sent." }
Write-Host "Setup and offline verification succeeded. Start with .\start.ps1" -ForegroundColor Green
