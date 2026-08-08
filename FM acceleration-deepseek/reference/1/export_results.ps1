[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runtime = Join-Path $Root 'runtime'
$Agent = Join-Path $Root 'bundle\agent-framework'
$Profile = 'live_cap1000000_atomicxfer1_ic0p007_ir0p084_rho0p5_i200_t110_b40_oi1'
$Running = Get-ChildItem (Join-Path $Runtime 'pids') -Filter 'worker*.pid' -ErrorAction SilentlyContinue | Where-Object {
    $Id = [int](Get-Content $_.FullName -Raw); Get-Process -Id $Id -ErrorAction SilentlyContinue
}
if ($Running) { throw 'Workers are still running. Finish the six WLs and run .\stop.ps1 before export.' }
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$Stage = Join-Path $Runtime ("exports\stage_$Stamp")
$Out = Join-Path $Runtime ("exports\FM_WL4_9_results_$Stamp.zip")
New-Item -ItemType Directory -Force -Path $Stage, (Join-Path $Stage 'fm'), (Join-Path $Stage 'states'), (Join-Path $Stage 'logs') | Out-Null
$Fingerprint = '8410ae8bbd86fd8735de5ea4823e4924cebf977e51e2946854378fba46018c28'
foreach ($Wl in 4..9) {
    $Source = Join-Path $Agent ("results\fm\WL{0}" -f $Wl)
    if (-not (Test-Path $Source)) { throw "Missing output for WL$Wl: $Source" }
    Copy-Item $Source (Join-Path $Stage 'fm') -Recurse
    $State = Join-Path $Runtime ("state\wl{0}.json" -f $Wl)
    if (Test-Path $State) { Copy-Item $State (Join-Path $Stage ("states\wl{0}.json" -f $Wl)) }
}
Copy-Item (Join-Path $Runtime 'logs\*.log') (Join-Path $Stage 'logs') -ErrorAction SilentlyContinue
@{
    schema_version = 1
    created_at = (Get-Date).ToString('o')
    worldlines = @(4,5,6,7,8,9)
    warmup_fingerprint = $Fingerprint
    source_agent_framework = (Resolve-Path $Agent).Path
    run_profile = $Profile
} | ConvertTo-Json | Set-Content (Join-Path $Stage 'export_manifest.json')
if (Test-Path $Out) { Remove-Item $Out -Force }
Compress-Archive -Path (Join-Path $Stage '*') -DestinationPath $Out -CompressionLevel Optimal
Remove-Item $Stage -Recurse -Force
Write-Host "Export created: $Out" -ForegroundColor Green
