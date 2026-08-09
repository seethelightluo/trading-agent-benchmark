# Git 自动保存

`git_autopush.sh` 由本机 systemd user timer 每 5 小时运行一次。它只提交
`.gitignore` 允许进入仓库的文件，自动保留远端提交并推送 `main`；遇到冲突、未完成 rebase
或认证错误会失败并留在本机，不会强制覆盖远端历史。

## 大阶段里程碑

`git_milestone_push.sh <label>` 用于 warmup 或 WL 结束等大阶段。它共享 Git
锁，先把所有非忽略文件（包括运行结果、日志和因子）提交到本机，再 fetch/push
到 `origin/main`；网络暂时失败时本机 checkpoint 不会丢失，并会重试。环境、密钥和
虚拟环境仍由 `.gitignore` 排除。

当前 DeepSeek AC runner 在启动、shared warmup 完成、每个 WL 完成以及全部配置 WL
完成时调用该命令。`trade-agent-benchmark-milestone-watch.service` 同时监视当前
DeepSeek 的 `run_state.json`，用于覆盖已经启动的旧进程；首次启动只建立基线，不会
重复推送旧结果。

手动运行：

```bash
/home/lxx/trade-agent-benchmark/ops/git_autopush.sh
```

手动创建阶段 checkpoint：

```bash
/home/lxx/trade-agent-benchmark/ops/git_milestone_push.sh deepseek-warmup-complete
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
