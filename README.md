# Trading Agent Benchmark

这是 AC（AlphaCrafter）与 FM（FactorMiner）在九条合成世界线上的实验工作区。当前实验合同与运行入口以以下文件为准：

- [RUN.md](/home/lxx/trade-agent-benchmark/RUN.md)：总运行手册、数据和投资合同。
- [runFM.md](/home/lxx/trade-agent-benchmark/runFM.md)：FactorMiner warmup、checkpoint、重试和验收。
- [agent-framework/progress.md](/home/lxx/trade-agent-benchmark/agent-framework/progress.md)：AC/FM 实际入口、恢复语义和历史验证。
- [agent-framework/ASSETS.yaml](/home/lxx/trade-agent-benchmark/agent-framework/ASSETS.yaml)：15 资产、准入、成本和容量的单一合同源。
- [agent-framework/scheduler/run_pipeline.py](/home/lxx/trade-agent-benchmark/agent-framework/scheduler/run_pipeline.py)：共享 warmup 与逐 WL 前向编排。

当前 portfolio contract：初始资金 `1_000_000`，15 个可交易资产，首建仓免费，后续资产迁移按 `3 bps` 收费；允许小数份额，现金不是资产且建仓后为零。研究库 `<=30` 全留，超过 30 才淘汰到 best30，活跃组合最多 10 个因子。

本文件是误删后的导航恢复版；详细研究说明见 `research/`，参数记录见 `llm-params/`，运行产出归档位于 `report-and-output/` 和 `winbackup/`。

