[CmdletBinding()]
param()
$ErrorActionPreference = 'SilentlyContinue'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runtime = Join-Path $Root 'runtime'
$Agent = Join-Path $Root 'bundle\agent-framework'
$Profile = 'live_cap1000000_atomicxfer1_ic0p007_ir0p084_rho0p5_i200_t110_b40_oi1'
$TotalWindows = 247   # 2467 trading days / cadence-10, matches Linux host

Write-Host ("time={0} CST" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
Write-Host "--- FM online (WL4-WL9, Windows U9 275HX) ---"

# Worker -> assigned WL sequence (worker1: WL4->WL7, etc.)
$WorkerWls = @{ 1 = @(4,7); 2 = @(5,8); 3 = @(6,9) }

foreach ($Worker in 1..3) {
    # Determine which WL this worker is currently on
    $PidFile = Join-Path $Runtime ("pids\worker{0}.pid" -f $Worker)
    $Wpid = if (Test-Path $PidFile) { [int](Get-Content $PidFile -Raw) } else { 0 }
    $Proc = if ($Wpid) { Get-Process -Id $Wpid -ErrorAction SilentlyContinue } else { $null }
    $Status = if ($Proc) { "RUNNING" } else { "STOPPED" }

    # Find the active WL for this worker (the one with a forward_state or most recent window)
    $ActiveWl = $null
    foreach ($wl in $WorkerWls[$Worker]) {
        $wd = Join-Path $Agent ("results\fm\WL{0}\{1}\online_mining\windows" -f $wl,$Profile)
        if (Test-Path $wd) { $ActiveWl = $wl; break }   # first non-empty WL in sequence
    }

    # Status line for each WL assigned to this worker
    foreach ($wl in $WorkerWls[$Worker]) {
        $OnlineDir = Join-Path $Agent ("results\fm\WL{0}\{1}\online_mining" -f $wl,$Profile)
        $WinDir = Join-Path $OnlineDir 'windows'
        $WinCount = if (Test-Path $WinDir) { (Get-ChildItem $WinDir -Directory).Count } else { 0 }

        # Library size
        $LibFile = Join-Path $OnlineDir 'factor_library.json'
        $Lib = if (Test-Path $LibFile) { (Get-Content $LibFile -Raw | ConvertFrom-Json).factors.Count } else { 0 }

        # NAV from forward state
        $FwdState = Join-Path $Agent ("results\fm\WL{0}\{1}\forward_adaptive_v4\forward_state.json" -f $wl,$Profile)
        $Nav = 0; $Days = 0; $LastDate = ""
        if (Test-Path $FwdState) {
            $fs = Get-Content $FwdState -Raw | ConvertFrom-Json
            $Nav = [math]::Round($fs.nav)
            $LastDate = $fs.last_processed_date
            # trading days elapsed since baseline
            $Baseline = [datetime]"2026-07-16"
            $Last = [datetime]$LastDate
            $Days = [int]($Last - $Baseline).TotalDays
        }

        # Error counts from this worker's log
        $LogFile = Join-Path $Runtime ("logs\worker{0}.log" -f $Worker)
        $c429 = 0; $c503 = 0
        if (Test-Path $LogFile) {
            $hits = Select-String -Path $LogFile -Pattern 'HTTP/1\.1 (\d{3})' -AllMatches
            foreach ($h in $hits) {
                $code = $h.Matches[0].Groups[1].Value
                if ($code -eq '429') { $c429++ }
                if ($code -eq '503') { $c503++ }
            }
        }

        $Tag = if ($WinCount -gt 0) { ("win={0}/{1}" -f $WinCount,$TotalWindows) } else { "QUEUED" }
        $NavStr = if ($Nav -gt 0) { ("NAV={0:N0} ({1}d)" -f $Nav,$Days) } else { "" }
        $ErrStr = ("429={0} 503={1}" -f $c429,$c503)

        if ($WinCount -gt 0) {
            Write-Host ("  FM WL{0}: pid={1} {2} | online {3} lib={4} {5} | {6}" -f $wl,$Wpid,$Status,$Tag,$Lib,$NavStr,$ErrStr)
        } else {
            Write-Host ("  FM WL{0}: pid={1} {2} | {3}" -f $wl,$Wpid,$Status,$Tag)
        }
    }
}
Write-Host ""
