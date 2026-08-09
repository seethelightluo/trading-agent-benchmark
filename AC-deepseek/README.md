# AC DeepSeek 9-WL 实验

这是从 `agent-framework/AlphaCrafter` 复制出的独立 AC 实验，不修改当前
`ac relay + gpt-5.6-terra` 运行。模型配置为 `deepseek-v4-flash-free`，API
入口为本机 `/home/lxx/opencode-api` 的 OpenAI 兼容网关。由于该 DeepSeek
上游的原生 Responses 工具多轮会丢失 `reasoning_content`，副本在同一网关
上使用 Chat Completions 兼容路径，并保持 AC 原有的工具循环和 warmup 配置。

启动脚本会：

1. 完成独立的 40-cycle shared warmup；
2. 从 warmup 状态播种 WL1–WL3；
3. 同时启动 3 个 AC worldline 进程，每个进程仍保留 3 个 Miner 并发、Screener/Trader 顺序；
4. 将状态写入 `results/ac9wl_deepseek/run_state.json`，失败的 WL 每 60 秒按 `--resume` 续跑。

```bash
cd /home/lxx/trade-agent-benchmark/AC-deepseek
setsid nohup ./run_deepseek_ac9.sh \
  > results/ac9wl_deepseek/supervisor.log 2>&1 < /dev/null &
echo $! > results/ac9wl_deepseek/supervisor.pid
```

网关当前全局 `CONCURRENCY=12`，因此 3 WL × 3 Miner 的上游请求最多占用 9 个
并发槽，给 Screener/Trader 和重试留有余量，而不是无限制地同时打到上游。
