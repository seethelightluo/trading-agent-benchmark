# run_warmup_ds.ps1 — deepseek 版共享 FM warmup（200 轮，后台持久化）
# 用法：powershell -ExecutionPolicy Bypass -File run_warmup_ds.ps1
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runtime = Join-Path $Root 'runtime'
$Agent = Join-Path $Root 'bundle\agent-framework'
$Python = Join-Path $Agent 'FactorMiner\.venv\Scripts\python.exe'
$Runner = Join-Path $Root 'scripts\portable_runner.py'
$State = Join-Path $Runtime 'state\ds_warmup.json'
$Log = Join-Path $Runtime 'logs\ds_warmup.log'
$PidFile = Join-Path $Runtime 'pids\ds_warmup.pid'
$EnvFile = Join-Path $Runtime '.env'

function Import-DotEnv([string]$Path) {
    Get-Content $Path | ForEach-Object {
        $Line = $_.Trim()
        if ($Line -and -not $Line.StartsWith('#')) {
            $Pair = $Line.Split('=', 2)
            if ($Pair.Count -eq 2) {
                $Name = $Pair[0].Trim(); $Value = $Pair[1].Trim().Trim('"').Trim("'")
                [Environment]::SetEnvironmentVariable($Name, $Value, 'Process')
            }
        }
    }
}
Import-DotEnv $EnvFile
if (-not $env:OPENAI_API_URL -or -not $env:OPENAI_API_KEY) { throw 'Missing OPENAI_API_URL/OPENAI_API_KEY in runtime/.env' }
$env:OPENBLAS_NUM_THREADS = '1'; $env:OMP_NUM_THREADS = '1'; $env:MKL_NUM_THREADS = '1'; $env:NUMEXPR_NUM_THREADS = '1'
$env:PYTHONUTF8 = '1'; $env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONPATH = "$Agent;$Agent\FactorMiner"

if (Test-Path $PidFile) {
    $OldPid = [int](Get-Content $PidFile -Raw)
    if (Get-Process -Id $OldPid -ErrorAction SilentlyContinue) {
        Write-Host "deepseek warmup already running (PID $OldPid)."
        exit 0
    }
}
New-Item -ItemType Directory -Force -Path (Join-Path $Runtime 'logs'), (Join-Path $Runtime 'pids') | Out-Null
$QuotedRunner = '"' + $Runner + '"'
$QuotedState = '"' + $State + '"'
$ArgList = @($QuotedRunner, "--mode", "fm", "--warmup-only", "--fm-cadence", "10", "--fm-iterations", "200", "--fm-target", "110", "--fm-batch-size", "40", "--fm-evaluation-workers", "4", "--fm-max-windows", "0", "--max-attempts", "0", "--state", $QuotedState)
$Process = Start-Process -FilePath $Python -ArgumentList $ArgList -WindowStyle Hidden -RedirectStandardOutput $Log -RedirectStandardError "$Log.err" -PassThru
Set-Content -Path $PidFile -Value $Process.Id -NoNewline
Write-Host "deepseek warmup started PID $($Process.Id); log: $Log; state: $State"
Write-Host "监控: Get-Content $Log -Wait ; 状态: $State"
