[CmdletBinding()]
param()
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$PidDir = Join-Path $Root 'runtime\pids'
if (-not (Test-Path $PidDir)) { exit 0 }
Get-ChildItem $PidDir -Filter 'worker*.pid' | ForEach-Object {
    $Pid = [int](Get-Content $_.FullName -Raw)
    $Process = Get-Process -Id $Pid -ErrorAction SilentlyContinue
    if ($Process) {
        & taskkill.exe /PID $Pid /T /F | Out-Null
        Write-Host "Stopped $($_.BaseName) PID $Pid (child processes included)."
    }
    Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue
}
