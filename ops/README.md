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
