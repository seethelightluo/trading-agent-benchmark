# runFM.md：FactorMiner 运行手册（恢复版）

> 本文件根据 2026-07-26、2026-08-03 和 2026-08-06 Codex 会话中的最终修改、当前 `scheduler/run_pipeline.py`、`ASSETS.yaml` 与 `agent-framework/progress.md` 重建。运行结果不在本文件中恢复。

## 1. 运行结构

FM 由三段组成：`mine`（唯一消耗 LLM）→ `combine`（确定性）→ `run_forward`（确定性）。

```text
共享 warmup（一次，9 条 WL 复用）
  截止 2026-07-15 的历史切片
  Ralph loop -> factor_library + memory + checkpoint
  fingerprint 命中 -> 0 LLM 复用；不命中 -> 新 stage
  每次 mining 后：size <= 30 全部保留；size > 30 按 |IC|*|ICIR| 保留 best30
  顶层库与 checkpoint/library 同步写入，resume 不复活已淘汰因子
       |
逐 WL 前向
  每 10 个交易日：刷新库信号 -> online mine -> trim -> combine -> 原子调仓
  日间只做本地 mark-to-market，不调用 LLM
```

论文/上游 FactorMiner 原生负责因子挖掘、准入、经验记忆和组合；本 benchmark 额外负责 expanding-window、信号刷新、库容淘汰、top-10 组合、成本死区、资产迁移调仓和九条世界线编排。

## 2. 固定实验契约

| 项目 | 当前口径 |
|---|---|
| 初始资本 | `1_000_000` USD-equivalent |
| 可交易资产 | 15 个；现金不是第 16 个资产 |
| 因子准入 | `abs(IC) >= 0.007`、`abs(ICIR) >= 0.084`、库内最大 `abs(Spearman rho) < 0.5` |
| 因子库 | `<=30` 全部保留；`>30` 按 `abs(IC)*abs(ICIR)` 淘汰到 best30 |
| 活跃组合 | 最多 10 个因子 |
| warmup | `200` iterations、目标池 `110`、每批 `40` candidates |
| online | 每个 10 交易日窗口默认追加 `1` 次 Ralph iteration |
| 投资 | long-only、15 项权重非负且和为 1、允许小数份额、cash=0 |
| 成本 | 首建仓免费；后续只有 gross edge 严格超过单边迁移额×3bp才调仓，实际只对单边迁移额收 `3 bps` |

唯一合同源是 `agent-framework/ASSETS.yaml`；过时的 100M、6 bps、整手和现金持有口径不适用于当前 benchmark。

### 2.1 统一 proposal / gate 合同

每个 10 个交易日决策点先生成研究 proposal，再由确定性执行层决定是否成交：

```text
current_weights, proposed_target_weights, executed_target_weights
forecast_returns, factor_ids, horizon_days=10
one_way_turnover = 0.5 * sum(abs(target-current))
gross_edge_bps = 10000 * sum((target-current) * forecast_returns)
decision_edge_threshold_bps = one_way_turnover * 3
actual_cost = NAV * one_way_turnover * 3 / 10000
```

首次 `2026-07-16` 建仓豁免门控；没有有效 ensemble 时使用 15 资产等权 `1/15`。之后严格要求 `gross_edge_bps > one_way_turnover * 3`，否则保存研究/proposed target 但保持 executed target、真实持仓和 cash 不变。

## 3. 完整共享 warmup

在 `agent-framework` 目录执行：

```bash
cd /home/lxx/trade-agent-benchmark/agent-framework
mkdir -p results
FM_PIPELINE_LOG_PATH="$PWD/results/full_warmup_fm.log" \
setsid --wait nohup /home/lxx/trade-agent-benchmark/.venv/bin/python -m scheduler.run_pipeline \
  --mode fm --warmup-only \
  --fm-iterations 200 --fm-batch-size 40 --max-attempts 0 \
  --state results/full_warmup_fm_state.json \
  </dev/null > results/full_warmup_fm.log 2>&1 &
echo $! > results/full_warmup_fm.pid
```

结构化监控：

```bash
/home/lxx/trade-agent-benchmark/.venv/bin/python -m scheduler.monitor_fm_warmup \
  --state results/full_warmup_fm_state.json \
  --log results/full_warmup_fm.log \
  --pid-file results/full_warmup_fm.pid \
  --watch --interval 30
```

原始日志另看：`tail -F results/full_warmup_fm.log`。监控会显示轮次、候选数、当前库容、ETA、最近重试和最近错误。

`--max-attempts 0` 表示正式任务无限重试，退避为 `0s -> 60s -> 600s -> 3600s`，之后持续每小时重试，成功后退避复位。冒烟测试才使用有限次数。

## 4. 断点、指纹和成功判定

失败或机器重启后重新执行同一启动命令即可。state、warmup stage 和 checkpoint 会被复用；fingerprint 不一致时不会静默混用旧库，而是创建新的 immutable stage。不要手工删除 `results/full_warmup_fm_state.json` 或对应 checkpoint。

成功必须同时满足：

1. 日志末尾出现 `共享 warm-up 结束：AC=False FM=True`；
2. `full_warmup_fm_state.json` 中 `shared_warmup.fm_done=true`；
3. 导出库与 `checkpoint/library.json` 的 factor IDs、signals 一致，库大小不超过 30。

## 5. 已知问题与修复

| 问题 | 当前处理 |
|---|---|
| expanding window 导致 checkpoint signal shape 不一致 | 每个 online window 挖矿前刷新 checkpoint/library 的信号；导出库也同步刷新 |
| trim 后 resume 复活旧大库 | `_trim_factor_library(..., checkpoint_library_json_path=...)` 对两套库和 signals 同步写入，并校验 IDs |
| API 临时失败 | `max-attempts=0` 使用持续退避；正式运行不要用 `3` 次上限 |
| 计算慢 | 2026-08-03 P0/P1：daily singleton preprocess 快路径、候选评估确定性 ProcessPool；父进程单独写 library/checkpoint |
| 运行监控困难 | `scheduler/monitor_fm_warmup.py` 输出结构化阶段、ETA 和错误 |

## 6. 已有证据边界

- 2026-07-26 的 live 10 轮验证：10/10 轮、400 candidates、3 因子录取，准入门槛和顶层库/checkpoint signals 一致。
- 同期端到端 forward 验证覆盖信号刷新、online mining、combine、非等权 tilt、cash=0、小数份额、首建仓免费和 3 bps 迁移成本；试跑没有自然触发 F5 成本死区或 `>30` 淘汰路径，二者由单测覆盖。
- 2026-08-03 的性能记录：真实 35,184 行 daily panel 的 preprocessing 从约 318.0s 降到 14.4s；40-candidate 隔离 smoke 的 Ralph 约 279.5s。实际时长仍取决于模型响应、API 限流和录取率。
- 全量 warmup 不是“保证无条件成功”：代码具备 checkpoint、指纹、重试和离线测试证据，但真实 API 配额、上游服务和未覆盖的最终录取分布仍是外部条件。保守预留约 18–24 小时；不要把历史短 smoke 当成保证。

## 7. 相关入口

- `agent-framework/scheduler/run_pipeline.py`
- `agent-framework/scheduler/monitor_fm_warmup.py`
- `agent-framework/FactorMiner/factorminer/core/ralph_loop.py`
- `agent-framework/FactorMiner/factorminer/agent/llm_interface.py`
- `agent-framework/ASSETS.yaml`
- `agent-framework/progress.md`
- `RUN.md`
