[CmdletBinding()]
param()
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runtime = Join-Path $Root 'runtime'
$Agent = Join-Path $Root 'bundle\agent-framework'
$Profile = 'live_cap1000000_atomicxfer1_ic0p007_ir0p084_rho0p5_i200_t110_b5_oi1'
$Assignments = @{ 1 = @(1); 2 = @(2); 3 = @(3); 4 = @(4); 5 = @(5); 6 = @(6); 7 = @(7); 8 = @(8); 9 = @(9) }
Write-Host ("FM deepseek status {0}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'))
foreach ($Worker in 1..9) {
    $PidFile = Join-Path $Runtime ("pids\worker{0}.pid" -f $Worker)
    $Wpid = if (Test-Path $PidFile) { [int](Get-Content $PidFile -Raw) } else { $null }
    $Proc = if ($Wpid) { Get-Process -Id $Wpid -ErrorAction SilentlyContinue } else { $null }
    $WlText = ($Assignments[$Worker] | ForEach-Object { "WL$_" }) -join '/'
    if ($Proc) { Write-Host ("worker{0}: RUNNING pid={1} cpu={2:N1}s  ->  {3}" -f $Worker,$Wpid,$Proc.CPU,$WlText) } else { Write-Host ("worker{0}: STOPPED  ->  {1}" -f $Worker,$WlText) }
}
Write-Host ""
foreach ($Wl in 1..9) {
    $Online = Join-Path $Agent ("results\fm\WL{0}\{1}\online_mining" -f $Wl,$Profile)
    $Windows = Join-Path $Online 'windows'
    $Latest = Get-ChildItem $Windows -Filter 'window_state.json' -Recurse -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
    $StatePath = Join-Path $Runtime ("state\wl{0}.json" -f $Wl)
    $State = if (Test-Path $StatePath) { Get-Content $StatePath -Raw | ConvertFrom-Json } else { $null }
    $WlKey = "wl$Wl"
    $Progress = if ($State -and $State.$WlKey.fm_progress) { $State.$WlKey.fm_progress } else { $null }
    $BatchPath = Join-Path $Online 'mining_batches.jsonl'
    $IterTotal = if (Test-Path $BatchPath) { (Get-Content $BatchPath).Count } else { 0 }
    $LibPath = Join-Path $Online 'factor_library.json'
    $LibSize = 0
    if (Test-Path $LibPath) {
        $Lib = Get-Content $LibPath -Raw | ConvertFrom-Json
        $LibSize = if ($Lib.PSObject.Properties.Name -contains 'factors') { @($Lib.factors).Count } elseif ($Lib -is [System.Array]) { @($Lib).Count } else { 0 }
    }
    $EquityPath = Join-Path $Agent ("results\fm\WL{0}\{1}\forward_adaptive_v4\equity.csv" -f $Wl,$Profile)
    $Nav = $null
    if (Test-Path $EquityPath) {
        $LastRow = Get-Content $EquityPath | Select-Object -Last 1
        if ($LastRow -and $LastRow -notmatch '^date,') { $Nav = $LastRow.Split(',')[2] }
    }
    if ($Latest -and $Progress) {
        $Window = Get-Content $Latest.FullName -Raw | ConvertFrom-Json
        $NavText = if ($Nav) { [math]::Round([double]$Nav,0) } else { 'n/a' }
        Write-Host ("WL{0}: phase={1} win={2} iter={3}/{4} lib={5} NAV={6}" -f $Wl,$Progress.phase,$Latest.Directory.Name,$Window.completed_iteration,$IterTotal,$LibSize,$NavText)
    } elseif ($Progress) {
        $NavText = if ($Nav) { [math]::Round([double]$Nav,0) } else { 'n/a' }
        Write-Host ("WL{0}: phase={1} iter={2} lib={3} NAV={4} (等待首个在线窗口产物)" -f $Wl,$Progress.phase,$IterTotal,$LibSize,$NavText)
    } elseif ($Latest) {
        $Window = Get-Content $Latest.FullName -Raw | ConvertFrom-Json
        $NavText = if ($Nav) { [math]::Round([double]$Nav,0) } else { 'n/a' }
        Write-Host ("WL{0}: win={1} iter={2}/{3} lib={4} NAV={5}" -f $Wl,$Latest.Directory.Name,$Window.completed_iteration,$IterTotal,$LibSize,$NavText)
    } else {
        Write-Host ("WL{0}: QUEUED" -f $Wl)
    }
}
Write-Host ""
Write-Host "Recent errors (if any):"
$Found = $false
Get-ChildItem (Join-Path $Runtime 'logs') -Filter '*.log' -ErrorAction SilentlyContinue | ForEach-Object {
    $Hit = Select-String -Path $_.FullName -Pattern '429 Too Many|503 Service|Traceback|contract mismatch|refresh failed|rc=[1-9]' | Select-Object -Last 1
    if ($Hit) { $Found = $true; Write-Host ("  {0}: {1}" -f $_.Name,$Hit.Line.Trim()) }
}
if (-not $Found) { Write-Host "  (none)" }
