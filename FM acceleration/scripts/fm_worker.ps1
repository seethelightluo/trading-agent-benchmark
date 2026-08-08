[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][int]$WorkerId,
    [Parameter(Mandatory=$true)][string]$WorldlineCsv
)
$ErrorActionPreference = 'Continue'
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Runtime = Join-Path $Root 'runtime'
$Agent = Join-Path $Root 'bundle\agent-framework'
$Python = Join-Path $Agent 'FactorMiner\.venv\Scripts\python.exe'
$Runner = Join-Path $Root 'scripts\portable_runner.py'
$WindowMigrator = Join-Path $Root 'scripts\migrate_window_states.py'
$Log = Join-Path $Runtime ("logs\worker{0}.log" -f $WorkerId)
$EnvFile = Join-Path $Runtime '.env'

function Import-DotEnv([string]$Path) {
    Get-Content $Path | ForEach-Object {
        $Line = $_.Trim()
        if ($Line -and -not $Line.StartsWith('#')) {
            $Pair = $Line.Split('=', 2)
            if ($Pair.Count -eq 2) {
                $Name = $Pair[0].Trim()
                $Value = $Pair[1].Trim().Trim('"').Trim("'")
                [Environment]::SetEnvironmentVariable($Name, $Value, 'Process')
            }
        }
    }
}
function Log([string]$Message) {
    Add-Content -Path $Log -Value ("{0} {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'),$Message) -Encoding UTF8
}
New-Item -ItemType Directory -Force -Path (Join-Path $Runtime 'logs'), (Join-Path $Runtime 'state') | Out-Null
if (-not (Test-Path $Python)) { Log 'ERROR: missing uv environment; run setup.ps1'; exit 2 }
if (-not (Test-Path $EnvFile)) { Log 'ERROR: missing runtime/.env'; exit 2 }
Import-DotEnv $EnvFile
if (-not $env:OPENAI_API_URL -or -not $env:OPENAI_API_KEY) { Log 'ERROR: required OpenAI environment values are absent'; exit 2 }
$env:OPENBLAS_NUM_THREADS = '1'; $env:OMP_NUM_THREADS = '1'; $env:MKL_NUM_THREADS = '1'; $env:NUMEXPR_NUM_THREADS = '1'
$env:PYTHONUTF8 = '1'; $env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONPATH = "$Agent;$Agent\FactorMiner"
foreach ($WlText in $WorldlineCsv.Split(',')) {
    $Wl = [int]$WlText.Trim()
    $State = Join-Path $Runtime ("state\wl{0}.json" -f $Wl)
    while ($true) {
        # Offline, idempotent performance-equivalence preflight: validate the
        # shipped certificate, back up historic window contracts, and retag
        # their warm-up fingerprint b93cbb67 -> 8410ae8b.  It never starts FM
        # while the migration cannot proceed, so no scheduler can run with a
        # mismatched seed.
        & $Python $WindowMigrator --worldline $Wl 2>&1 | Out-File -FilePath $Log -Append -Encoding UTF8
        if ($LASTEXITCODE -ne 0) {
            Log "WINDOW_MIGRATION_REFUSED WL$Wl rc=$LASTEXITCODE; retrying in 10 seconds without starting FM"
            Start-Sleep -Seconds 10
            continue
        }
        Log "START WL$Wl (resume state=$State; performance bridge b93cbb67->8410ae8b)"
        & $Python $Runner --mode fm --only $Wl --fm-cadence 10 --fm-iterations 200 --fm-target 110 --fm-batch-size 40 --fm-online-iterations 1 --fm-evaluation-workers 4 --fm-max-windows 0 --max-attempts 0 --fm-performance-equivalent-from b93cbb67ae2e48c9be026297cee2fe40fdbfb2cf5cbfa03c5d6bf89376964b3c --state $State 2>&1 | Out-File -FilePath $Log -Append -Encoding UTF8
        $Rc = $LASTEXITCODE
        if ($Rc -eq 0) { Log "DONE WL$Wl"; break }
        Log "PIPELINE_EXIT WL$Wl rc=$Rc; retrying in 10 seconds from checkpoint"
        Start-Sleep -Seconds 10
    }
}
Log 'WORKER_DONE'
