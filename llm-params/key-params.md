# 关键参数记录（2026-08-06）

## 核心参数
- 模型名：项目侧统一 `gpt-5.6-terra`（经本地 relay `/home/lxx/ac-llm-relay` 映射到实际上游 `gpt-5.6-luna`，opencode.ai zen/go）
- reasoning：`mode=standard, effort=medium`（官方默认；AC 请求未显式发送，由端点默认生效）
- temperature / top_p / seed：均为官方默认 config（`temperature=1, top_p=1`），请求方发送的值不生效

## 备注
- AC（Responses API）：仅发送 `model/tools/input/instructions/parallel_tool_calls`，不发送任何采样参数
- FM（Chat Completions）：显式发送 `temperature=0.8, max_tokens=4096`，实测被网关忽略（语法接受、语义不生效；reasoning 模型下 randomness 由 effort 控制，`max_tokens` 已弃用，需用 `max_completion_tokens`）
