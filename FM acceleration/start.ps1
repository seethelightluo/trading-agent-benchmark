[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runtime = Join-Path $Root 'runtime'
$Script = Join-Path $Root 'scripts\fm_worker.ps1'
$Python = Join-Path $Root 'bundle\agent-framework\FactorMiner\.venv\Scripts\python.exe'

if (-not (Test-Path $Python)) { throw "Run .\setup.ps1 first." }
if (-not (Test-Path (Join-Path $Runtime '.env'))) { throw "Missing runtime\.env." }
New-Item -ItemType Directory -Force -Path (Join-Path $Runtime 'pids'), (Join-Path $Runtime 'logs'), (Join-Path $Runtime 'state') | Out-Null

$Assignments = @(
    @{ Id = 1; Wls = '1' },
    @{ Id = 2; Wls = '2' },
    @{ Id = 3; Wls = '3' }
)
foreach ($Assignment in $Assignments) {
    $PidFile = Join-Path $Runtime ("pids\worker{0}.pid" -f $Assignment.Id)
    if (Test-Path $PidFile) {
        $OldPid = [int](Get-Content $PidFile -Raw)
        if (Get-Process -Id $OldPid -ErrorAction SilentlyContinue) {
            Write-Host "worker$($Assignment.Id) already running (PID $OldPid); skip."
            continue
        }
    }
    $Args = "-NoProfile -ExecutionPolicy Bypass -File `"$Script`" -WorkerId $($Assignment.Id) -WorldlineCsv `"$($Assignment.Wls)`""
    $Process = Start-Process -FilePath 'powershell.exe' -ArgumentList $Args -WindowStyle Hidden -PassThru
    Set-Content -Path $PidFile -Value $Process.Id -NoNewline
    Write-Host "Started worker$($Assignment.Id): WL$($Assignment.Wls) PID $($Process.Id)"
    if ($Assignment.Id -lt 3) { Start-Sleep -Seconds 20 }
}
Write-Host "Use .\status.ps1 for live progress." -ForegroundColor Green
