# Git 自动保存

`git_autopush.sh` 由本机 systemd user timer 每 5 小时运行一次。它只提交
`.gitignore` 允许进入仓库的文件，自动保留远端提交并推送 `main`；遇到冲突、未完成 rebase
或认证错误会失败并留在本机，不会强制覆盖远端历史。

手动运行：

```bash
/home/lxx/trade-agent-benchmark/ops/git_autopush.sh
```

查看定时器：

```bash
systemctl --user status trade-agent-benchmark-autopush.timer
journalctl --user -u trade-agent-benchmark-autopush.service -n 100 --no-pager
```

## FM-live 数据盘备份

`report-and-output/FM-live` 已移到 `/data/trade-agent-benchmark/FM-live`，不再进入本仓库。
每日快照由 `trade-agent-benchmark-fm-live-backup.timer` 创建，保存在
`/data/trade-agent-benchmark/FM-live-snapshots`，只保留最近 5 次；快照之间对未变化文件使用
硬链接以节省数据盘空间。

查看备份状态：

```bash
systemctl --user status trade-agent-benchmark-fm-live-backup.timer
journalctl --user -u trade-agent-benchmark-fm-live-backup.service -n 100 --no-pager
```
