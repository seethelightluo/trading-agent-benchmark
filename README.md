# Trading Agent Benchmark

这是 AC（AlphaCrafter）与 FM（FactorMiner）在九条合成世界线上的实验工作区。当前实验合同与运行入口以以下文件为准：

- [RUN.md](/home/lxx/trade-agent-benchmark/RUN.md)：总运行手册、数据和投资合同。
- [runFM.md](/home/lxx/trade-agent-benchmark/runFM.md)：FactorMiner warmup、checkpoint、重试和验收。
- [agent-framework/progress.md](/home/lxx/trade-agent-benchmark/agent-framework/progress.md)：AC/FM 实际入口、恢复语义和历史验证。
- [agent-framework/ASSETS.yaml](/home/lxx/trade-agent-benchmark/agent-framework/ASSETS.yaml)：15 资产、准入、成本和容量的单一合同源。
- [agent-framework/scheduler/run_pipeline.py](/home/lxx/trade-agent-benchmark/agent-framework/scheduler/run_pipeline.py)：共享 warmup 与逐 WL 前向编排。

当前 portfolio contract：初始资金 `1_000_000`，15 个可交易资产，首建仓免费，后续资产迁移按 `3 bps` 收费；允许小数份额，现金不是资产且建仓后为零。研究库 `<=30` 全留，超过 30 才淘汰到 best30，活跃组合最多 10 个因子。

本文件是误删后的导航恢复版；详细研究说明见 `research/`，参数记录见 `llm-params/`，运行产出归档位于 `report-and-output/` 和 `winbackup/`。

录取过的因子中最低 |IC| ≈ 0.024、最低 |ICIR| ≈ 0.084，落在全部候选的前 ~20%（IC）/ 前 ~15%（ICIR）（口径：FM 两实验 13,155 个去重 warmup 提案公式的全候选分布；名义门 0.007 本身有 ~61% 候选能过，所以它不 binding，真正的筛选在 ICIR + 相关性门 + 容量）。

§4.1 已据此改写并推送（56e3db04，重试后 ls-remote 验证一致）：

删掉了原来那段“κ 缩放 = 等检验力”的错误推导；
换成诚实表述：0.007/0.084 是 500 资产参考门按 κ≈0.17 的刻意宽松换算（"a deliberately permissive translation, not a significance calibration"），并明确写出为什么不能用 N 单变量缩放（单期噪声 1/√(N−1) 在 N=15 大 5.9 倍；mean-IC 显著性依赖观测数而非 N）；
选择压力的实际位置：名义 IC 门过 ~60% 候选 → 录取边界 ≈ 全候选前 20%/前 15% → 相关性门 ρ<0.5 在线移除 99%+ 的过筛者 → 容量 30 定库规模。
编译 47 页不变、4 处既有 overfull、无新问题；第十五轮已记入改进计划。这段现在把“门槛宽松”从一个隐患变成了论文的一个可审计的机制陈述——审稿人问“0.007 是不是太松”，答案就在正文里：松是设计使然，选择性由下游 gate 实现，且有数字佐证。


后续需要将agent框架代码中写的门槛进行修正。