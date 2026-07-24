# Trade Agent Benchmark：Agent 使用与执行进度

最后更新：2026-07-24（Asia/Shanghai）

## 1. 当前结论

- 实际仓库目录是 `/home/lxx/trade-agent-benchmark`（不是目标文本里的 `trade-agent-benchmarke`）。
- Python 统一使用仓库根目录的 uv 环境：`/home/lxx/trade-agent-benchmark/.venv/bin/python`。
- AlphaCrafter（AC）使用 OpenAI **Responses API** 驱动工具循环；FactorMiner（FM）使用 OpenAI **Chat Completions API** 生成因子候选。
- 2026-07-24 已用一次性凭证验证指定服务：正确 base URL 必须带 `/v1`；`gpt-5.6-terra` 存在，并能返回原生 `function_call`、JSON arguments 和 `call_id`。
- AC 的正确 CLI 是位置参数：`python main.py wl1 --config run_config.yaml --resume`。上游 README 中的 `--session_id` 与当前代码不一致，不能照抄。
- 首轮真实冒烟已连续发现并修复/定位多个运行器问题，完整 2-cycle 尚未通过，不能启动全量。

## 2. 数据如何进入两个 Agent 框架

统一入口是 `adapters/build_inputs.py`：

1. 读取某条 `WL*_full.parquet` 和 `ASSETS.yaml`。
2. 从 `AlphaCrafter/alphacrafter/sandbox/template_a/` 复制并重建 `sandbox/wlN/`。
3. 写入 AC 的：
   - `persistent/stock_data/`：15 个可交易资产；
   - `persistent/index_data/`：5 个只读宏观/状态信号；
   - `persistent/date.json`：当前交易日和完整交易日日历；
   - `persistent/account.json`：现金、持仓、订单、watch list；
   - `persistent/stock_news/`：与世界线阶段事件对齐、且滞后公开的新闻。
4. 同时写入 FM 的 `FactorMiner/data/panel.parquet` 和 walk-forward 配置。

重建 WL1：

```bash
cd /home/lxx/trade-agent-benchmark/agent-framework
VENV=/home/lxx/trade-agent-benchmark/.venv/bin/python

$VENV -m adapters.build_inputs \
  --panel ../data-prepare/online-worldline/WL1_full.parquet \
  --assets ASSETS.yaml \
  --ac-session wl1 \
  --fm-dir FactorMiner/data \
  --stage-news ../data-prepare/online-worldline/WL1_stage_news.json
```

`build_inputs` 会删除并重建目标 session，因此只能在希望从干净状态开始时调用；断点续跑时不要重建 session。

## 3. AlphaCrafter Agent 的真实工作方式

### 3.1 启动和 session

AC 必须从 `AlphaCrafter/alphacrafter/` 作为 cwd 启动。仓库代码同时使用：

- `from agent ...`：依赖 cwd 是 `alphacrafter/`；
- `from alphacrafter ...`：依赖 `AlphaCrafter/` 在 `PYTHONPATH`。

因此调度器会同时设置 cwd 和 `PYTHONPATH`。直接运行示例：

```bash
cd /home/lxx/trade-agent-benchmark/agent-framework/AlphaCrafter/alphacrafter
PYTHONPATH=/home/lxx/trade-agent-benchmark/agent-framework/AlphaCrafter \
  /home/lxx/trade-agent-benchmark/.venv/bin/python \
  main.py wl1 --config run_config.yaml --resume
```

### 3.2 一个 cycle 的 Agent 顺序

每个 cycle 固定为：

1. 3 个 Miner 并发执行；
2. Screener 汇总本 cycle 的 Miner 输出，并参考历史 Screener 信息；
3. Trader 接收本 cycle 的 Screener 输出，改写/验证策略并推进模拟交易。

各角色的实际工具集：

- Miner：`read_file`、`write_file`、`shell`；负责在 `workspace/factors/` 和 `workspace/scripts/` 中研究、生成因子。
- Screener：`shell`、股票/指数数据、因子搜索、财报、新闻；负责筛选和组合因子、判断市场状态。
- Trader：`read_file`、`write_file`、`shell`、`backtest`、`step`；策略不是直接调用下单工具，而是写入 `workspace/strategy.py`。策略 hook 在每个交易日调用 `alphacrafter.sim.utils.add_order/cancel_order`。

底层 `Agent.run()` 使用 Responses API：模型返回 `function_call`，本地执行对应函数，再把 `function_call_output` 放回下一轮输入；模型不再请求工具、达到 finish condition、被中断或达到 iteration 上限时结束。

### 3.3 cadence 和交易推进

调度器设置 `AC_CADENCE_DAYS=10`。Trader 即使给 `step(days=5)`，`StepTool` 也会强制每个 cycle 推进 10 个交易日，保证与 FM 的 10B 再平衡频率一致。Trader 每 cycle 只能调用一次 `step`。

关键状态：

- 时间游标：`sandbox/wlN/persistent/date.json`
- 账户：`sandbox/wlN/persistent/account.json`
- 策略：`sandbox/wlN/workspace/strategy.py`
- Agent 细日志：`sandbox/wlN/logs/*_agent*.json`
- cycle 汇总：`sandbox/wlN/logs/workflow.json`

### 3.4 `--resume` 的语义

AC 从 `workflow.json` 找到最后一个 Miner、Screener、Trader 都成功的完整 cycle，并从下一 cycle 继续；各 Agent 还会从自己的日志恢复最后输入。调度器外层再用 `results/run_state.json` 记录每条世界线 AC/FM 是否完成。

注意：重建 session 或删除其 logs 会清掉 AC 的 cycle 恢复点；只删除 `results/run_state.json` 不会删除 AC 内部日志。

## 4. FactorMiner Agent 的真实工作方式

FM 的 live 入口最终调用 `factorminer ... mine`，内部运行 RalphLoop：

1. 读取长表 panel 并构造 data tensor/returns；
2. 用 LLM 生成/改进候选表达式；
3. 对候选做因果性、数值、IC、相关性、换手等评估；
4. 将通过门槛的因子放入 library，并保存到输出目录。

FM 的 OpenAI provider 使用 Chat Completions，不依赖 AC 的 Responses tool-calling；base URL 和 key 仍从 `OPENAI_API_URL`、`OPENAI_API_KEY` 读取。调度器先把世界线日频 panel 重采样为 `10B`，再调用 live 配置 `FactorMiner/factorminer/configs/fm_live.yaml`。调试可用 `--fm-mock` 避免 API 调用。

## 5. 推荐运行方式

### 5.1 无 API 的本地回归

```bash
cd /home/lxx/trade-agent-benchmark
.venv/bin/python -m unittest -v \
  agent-framework/scheduler/test_run_pipeline.py \
  agent-framework/adapters/test_build_inputs.py
(cd agent-framework/AlphaCrafter/alphacrafter && \
  /home/lxx/trade-agent-benchmark/.venv/bin/python -m unittest -v \
  test_workflow_runtime.py)
```

### 5.2 1 WL × 2 cycle 真冒烟

一次性凭证只通过进程环境注入，不应提交：

```bash
cd /home/lxx/trade-agent-benchmark/agent-framework
export OPENAI_API_URL='http://HOST:PORT/v1'
export OPENAI_API_KEY='一次性密钥'
rm -f results/run_state.json results/run_state.tmp
/home/lxx/trade-agent-benchmark/.venv/bin/python \
  -m scheduler.run_pipeline --only 1 --mode ac \
  --max-cycles 2 --max-attempts 1
```

判定通过必须同时满足：

- 进程退出码为 0，输出 `WL1 AC 完成`；
- `workflow.json` 有 cycle 1、2 的 Miner/Screener/Trader 成功记录；
- `date.json.current_date` 前进约 20 个交易日；
- Trader/工具日志存在有效 function call，且不是空响应；
- `strategy.py` 已从模板空 hook 变为有效策略；
- `account.json` 有合理订单/持仓/净值变化。若策略明确跳过交易，必须能从 Screener/Trader 输出解释，不能只凭 rc=0 判通过。

### 5.3 全量守护运行

只有真冒烟通过后才可启动：

```bash
cd /home/lxx/trade-agent-benchmark
setsid nohup bash agent-framework/scheduler/run_all.sh \
  --mode both --cadence 10 \
  > agent-framework/results/run_all.log 2>&1 &

tail -f agent-framework/results/run_pipeline.log
```

`run_all.sh` 会在 `run_pipeline` 非零退出后等待 10 秒重拉；`run_pipeline` 内部对 API/子进程失败采用 0、60、600、3600 秒递增退避，成功后重置。

## 6. 2026-07-24 修复与验证记录

已完成并推送：

- `f555190`：AC 子进程补 `AlphaCrafter/` 到 `PYTHONPATH`，修复 `ModuleNotFoundError: alphacrafter`；增加回归测试；AC/FM 切换到 `gpt-5.6-terra` 模型注册和配置。
- `8e85860`：修复 AC session 位置参数、失败退出码、Miner cycle 日志、Trader shell 工具和 `StepTool` 策略热加载，并补充回归测试与本文档。
- API 探测：不带 `/v1` 的 `/models` 不是标准 JSON；带 `/v1/models` 正常列出 `gpt-5.6-terra`。
- Responses tool probe：成功收到 `function_call(name=probe, arguments={"value":7})` 和有效 `call_id`。

当前真实冒烟状态：

- 数据重建成功：83,347 行、20 资产，WL1 起点 `2026-07-16`。
- AC 已成功启动全部 Miner/Screener/Trader，真实 Miner function calling、`shell` 和 `write_file` 均可工作，证明 `gpt-5.6-terra` 与 AC 工具协议兼容。
- 冒烟暴露两个运行时问题：Agent shell 找不到 `python`（调度器 PATH 未包含 `.venv/bin`），以及并发 Miner 首次写入时 `workspace/factors`、`workspace/scripts` 尚未创建。
- 冒烟还暴露数据语义错配：旧提示硬编码 CSI300/数百只股票，Miner 因 15 资产小截面而主动判无效，导致 cycle 1 的 Screener/Trader 正确跳过且时间未前进。

本轮已修复并通过本地回归、待提交/真实复测：

- AC 子进程 PATH 将 uv 环境 `.venv/bin` 放在首位，同时保留继承 PATH；Miner 的 `python` 脚本可在同一环境执行。
- session 构建时确定性预建 `workspace/factors` 和 `workspace/scripts`，消除并发首次写入竞态。
- AC 公共、Miner、Screener、Trader 和 factor-mining skill 提示已改成真实的 15 个可交易跨资产 + 5 个只读信号语义，明确不得要求 50/80/300 个成分股。
- Trader 明确遵循当前模拟器的 long-only、T+1、100 单位整手和单边 3 bps 摩擦；不再建议不可执行的裸空头。
- 每个 Miner 现在收到当前日期和自己的上一周期反馈，不再统一传入空上下文。
- 首次复测中 Miner 已真实执行脚本并持久化因子，Screener 也成功给出 factor ensemble；随后发现 `BacktestTool` 与旧版 `StepTool` 一样缓存 Agent 初始化时的空策略 hook，导致 Trader 改写策略后回测仍为零持仓。
- `BacktestTool` 现于每次调用前重载 `strategy.py`，并确保结果日志目录存在。针对性单测覆盖初始化 hook 与运行 hook 的切换。
- 修复后用冒烟中真实生成的策略做 20 日本地回测，得到非零仓位和收益（平均 gross position 约 28.89%），确认不再执行空模板。
- 本地回归共 10 项通过，另通过 `git diff --check`。

## 7. 密钥安全

- `.env` 已被 Git 忽略，任何提交前仍需用 `git status` 和 `git diff --cached` 复核。
- 本轮指定 key 是一次性凭证：只用于 API 协议探测和 WL1 冒烟；验证结束后必须从 `AlphaCrafter/.env` 清除，不用于后台全量任务。
- `progress.md`、日志和提交信息中不得出现明文 key。
