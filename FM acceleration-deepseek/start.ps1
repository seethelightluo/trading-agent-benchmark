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

# deepseek 分支：9 条 WL 各一个 worker，共 9 并发
$Assignments = @(
    @{ Id = 1;  Wls = '1' },
    @{ Id = 2;  Wls = '2' },
    @{ Id = 3;  Wls = '3' },
    @{ Id = 4;  Wls = '4' },
    @{ Id = 5;  Wls = '5' },
    @{ Id = 6;  Wls = '6' },
    @{ Id = 7;  Wls = '7' },
    @{ Id = 8;  Wls = '8' },
    @{ Id = 9;  Wls = '9' }
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
}
Write-Host "Use .\status.ps1 for live progress." -ForegroundColor Green
