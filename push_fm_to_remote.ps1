# push_fm_to_remote.ps1 — 每 5 小时：本地 git 同步（含 deepseek 版）推送 FM 分支 + 远端 .90 拉取
# 用法：powershell -ExecutionPolicy Bypass -File push_fm_to_remote.ps1
$ErrorActionPreference = 'Continue'
$Repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$Stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Write-Host "[$Stamp] push_fm_to_remote start"

# 1) 本地：镜像 4 个 FM 目录 -> git 仓库并 commit+push origin FM
& (Join-Path $Repo 'sync_fm_to_git.ps1')
$PushRc = $LASTEXITCODE
Write-Host "  local git sync+pushed (rc=$PushRc)"

# 2) 远端：.90 拉取 FM 分支（增量，含全部实验数据）
$RemoteDir = '/home/lxx/trade-agent-benchmark/report-and-output/FM-live'
$RemoteCmd = "if [ -d `"$RemoteDir/.git`" ]; then cd `"$RemoteDir`" && git fetch origin FM && git reset --hard origin/FM; else git clone -b FM git@github.com:seethelightluo/trading-agent-benchmark.git `"$RemoteDir`"; fi && echo REMOTE_UPDATE_OK"
$Out = ssh -o BatchMode=yes -o ConnectTimeout=30 192.168.71.90 $RemoteCmd 2>&1
Write-Host "  remote: $($Out -join ' | ')"
Write-Host "[$Stamp] push_fm_to_remote done"
