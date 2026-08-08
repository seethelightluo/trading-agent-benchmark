# One-command Windows entrypoint: create/sync the uv environment, validate the
# offline payload, then detach the three persistent FM workers.
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $Root 'setup.ps1')
& (Join-Path $Root 'start.ps1')
