# Codex 会话恢复清单

恢复日期：2026-08-08。目标是恢复脚本和 Markdown，不重建大体量运行结果。

## 已从会话原文恢复

- `runFM.md`：按 2026-07-26 最终修改、2026-08-03 性能记录和当前代码重建。
- `research/research1.md`：按 2026-08-04 会话输出规范化恢复。
- `research/answer-research1.md`：从 2026-08-04 的完整 heredoc 原文恢复。
- `llm-params/key-params.md`：从 2026-08-06 的最终 heredoc 原文恢复。
- `agent-framework/scheduler/monitor_fm_warmup.py`：从 2026-07-26 的完整 heredoc 原文恢复。
- `agent-framework/scheduler/test_monitor_fm_warmup.py`：从同一会话的完整 heredoc 原文恢复。
- `agent-framework/AlphaCrafter/alphacrafter/utils/atomic_io.py`：从同一会话的完整 heredoc 原文恢复。
- `agent-framework/AlphaCrafter/alphacrafter/factor_contract.py`：从同一会话的完整 heredoc 原文恢复，并补回 ensemble 校验接口。
- `agent-framework/AlphaCrafter/alphacrafter/sim/utils/rebalance_to_weights.py`：依据同一会话中的完整源码输出重建；保留调仓合同，但不宣称字节级相同。
- `agent-framework/AlphaCrafter/alphacrafter/test_news_visibility.py`：从同一会话的完整 heredoc 原文恢复。

## 已存在的部分恢复内容

- `agent-framework/AlphaCrafter/alphacrafter/sandbox/recovered_factors/`：2,411 个从 AC Agent 日志/会话内容恢复的 `.py` 因子脚本；其中 2,325 个可解析，86 个日志片段语法损坏，详见 `recovery/FACTOR_RECOVERY_AUDIT.md`。
- `agent-framework/AlphaCrafter/alphacrafter/sandbox/recovered_logs/`：5 个 Agent JSON 日志，约 167MB。
- `report-and-output/fm_warmup_mock/`：FM warmup mock 的 checkpoint、library、batch/session 日志。
- `winbackup/FM-WL4-9data/`：FM WL4–WL9 结果、运行状态、日志和文档归档。

## 证据来源

- Codex session `019f9d5f-5df5-73b1-b6c6-02d3dc68dc1b`：FM runbook、monitor、warmup/checkpoint 修复和性能记录。
- Codex session `019fcace-4282-7283-bc8c-8b70440a7e64`：`research1.md` 原文和 `answer-research1.md`。
- Codex session `019fd60b-8a97-7c01-b73f-a5c8726591dd`：AC 备份、relay、参数记录和误删后的会话检索。

## 限制

原始 `research1.md` 的终端输出存在换行/公式渲染损失，因此恢复版做了清理；`runFM.md` 是基于会话补丁和当前代码重建，不声称与删除前字节级相同。没有把 API key 写入恢复文档。

2026-07-26 会话曾显示 `/home/lxx/trade-agent-benchmark/FM acceleration` Windows 部署包及提交 `524f247`，但当前 Git 对象和工作区均找不到该提交/目录，不能从现有会话文本完整重建 bundle；因此没有伪造该目录。`report-and-output/AC_backup_20260806_155647` 也未在当前工作区找到，清单不再把它列为现存备份。
