# 恢复因子脚本审计

审计日期：2026-08-08

来源目录：`agent-framework/AlphaCrafter/alphacrafter/sandbox/recovered_factors/`

- 恢复文件：2,411 个 `.py`
- `ast.parse` 可解析：2,325 个
- 语法损坏或截断：86 个
- `recovered_logs/` 中 5 个 JSON Agent 日志：全部可解析

这些文件来自 Agent 输出/工具日志恢复，不等同于一次干净的源码导出。对于 86 个损坏片段，本次保留原始恢复内容，未根据文件名或上下文猜测修复；运行前应只从可解析文件开始，并重新走因子准入、IC/ICIR 和相关性合同。

快速复核命令：

```bash
cd /home/lxx/trade-agent-benchmark
PYTHONPATH=agent-framework/AlphaCrafter \
  .venv/bin/python -m py_compile \
  agent-framework/AlphaCrafter/alphacrafter/sim/utils/rebalance_to_weights.py
```

如需清理 86 个片段，建议另建隔离目录并保留原文件，不要覆盖本目录的恢复证据。
