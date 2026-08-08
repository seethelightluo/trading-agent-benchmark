[CmdletBinding()]
param()
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runtime = Join-Path $Root 'runtime'
$Agent = Join-Path $Root 'bundle\agent-framework'
$Profile = 'live_cap1000000_atomicxfer1_ic0p007_ir0p084_rho0p5_i200_t110_b40_oi1'
Write-Host ("FM portable status {0}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'))
foreach ($Worker in 1..3) {
    $PidFile = Join-Path $Runtime ("pids\worker{0}.pid" -f $Worker)
    $Pid = if (Test-Path $PidFile) { [int](Get-Content $PidFile -Raw) } else { $null }
    $Proc = if ($Pid) { Get-Process -Id $Pid -ErrorAction SilentlyContinue } else { $null }
    if ($Proc) { Write-Host ("worker{0}: RUNNING pid={1} cpu={2:N1}s" -f $Worker,$Pid,$Proc.CPU) } else { Write-Host "worker$Worker: STOPPED" }
}
foreach ($Wl in 4..9) {
    $Online = Join-Path $Agent ("results\fm\WL{0}\{1}\online_mining" -f $Wl,$Profile)
    $Windows = Join-Path $Online 'windows'
    $Latest = Get-ChildItem $Windows -Filter 'window_state.json' -Recurse -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
    $State = Join-Path $Runtime ("state\wl{0}.json" -f $Wl)
    if ($Latest) {
        $Window = Get-Content $Latest.FullName -Raw | ConvertFrom-Json
        Write-Host ("WL{0}: {1} mine={2} combine={3} iteration={4}" -f $Wl,$Latest.Directory.Name,$Window.mining_complete,$Window.combination_complete,$Window.completed_iteration)
    } elseif (Test-Path $State) {
        Write-Host "WL$Wl: scheduler state exists; waiting for first online window artifact"
    } else {
        Write-Host "WL$Wl: QUEUED"
    }
}
Write-Host "Recent errors (if any):"
Get-ChildItem (Join-Path $Runtime 'logs') -Filter '*.log' -ErrorAction SilentlyContinue | ForEach-Object {
    $Hit = Select-String -Path $_.FullName -Pattern '429 Too Many|503 Service|Traceback|contract mismatch|refresh failed|rc=[1-9]' | Select-Object -Last 1
    if ($Hit) { Write-Host ("  {0}: {1}" -f $_.Name,$Hit.Line.Trim()) }
}
