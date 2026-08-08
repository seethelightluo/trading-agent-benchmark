# sync_fm_to_git.ps1 — 把 D 盘三个 FM 目录增量同步进本地 git 仓库并推送 FM 分支
# 用法：powershell -ExecutionPolicy Bypass -File sync_fm_to_git.ps1
$ErrorActionPreference = 'Continue'
$Repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$Stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Write-Host "[$Stamp] sync start"

# 1) 镜像三个 FM 目录（排除 .venv/缓存/临时 tar）
$Jobs = @(
    @{ Src = 'D:\FM acceleration';          Dst = Join-Path $Repo 'FM acceleration' },
    @{ Src = 'D:\FM acceleration-deepseek'; Dst = Join-Path $Repo 'FM acceleration-deepseek' },
    @{ Src = 'D:\FM-WL4-9data';             Dst = Join-Path $Repo 'FM-WL4-9data' },
    @{ Src = 'D:\WL-data-final';            Dst = Join-Path $Repo 'WL-data-final' }
)
foreach ($Job in $Jobs) {
    if (Test-Path -LiteralPath $Job.Src) {
        robocopy $Job.Src $Job.Dst /MIR /XD .venv __pycache__ node_modules /XF wl8_missing.tar *.tar *.zip /NFL /NDL /NJH /NJS /R:2 /W:2 | Out-Null
        Write-Host ("  mirrored {0} (rc={1})" -f $Job.Src, $LASTEXITCODE)
    }
}

# 2) 镜像中禁用的 .gitignore 若被重新复制则改名（避免 bundle/ 等规则生效）
$Disabled = Join-Path $Repo 'FM acceleration\.gitignore.disabled'
if (-not (Test-Path -LiteralPath $Disabled)) {
    Rename-Item -LiteralPath (Join-Path $Repo 'FM acceleration\.gitignore') -NewName '.gitignore.disabled' -Force -ErrorAction SilentlyContinue
}

# 3) git add + commit + push
git -C $Repo add -A 2>$null
$Changes = git -C $Repo status --short
if ($Changes) {
    $Count = @($Changes).Count
    git -C $Repo commit -m "FM sync: $($Changes.Count) changed paths @ $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" 2>$null | Out-Null
    git -C $Repo push origin FM 2>$null
    Write-Host "  committed and pushed $Count paths"
} else {
    Write-Host "  no changes"
}
Write-Host ("[{0}] sync done" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
