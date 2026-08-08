# WL-data-final — 最终扩展版实验数据集

> 创建：2026-08-08 ｜ 本目录是 FM WL1-9 实验的**最终扩展版**数据快照，可直接作为后续实验/合并的权威输入。

## 内容

| 目录 | 内容 |
|------|------|
| `panels/` | 9 条 WL 完整面板 `WL1-9_full.parquet`（warmup 2020-01-01~2026-07-16 真实 + 在线 2026-07-17~2035-12-31 合成，20 资产） |
| `news/` | 9 条 WL 阶段新闻 `WL1-9_stage_news.json`（扩展版，含 news_date 与阶段资产终点） |
| `wordline-md/` | 9 条世界线叙事源 `wordline1-9.md` |
| `research/` | 最原始产物：`blackswan_9worldlines_final_conclusion.md` + `final_synthesis_report.md`（9 条黑天鹅世界线最终综合结论，两个措辞版本）。来源：sumtime 数据；生成平台：superagents agent 平台 [https://superagents.go-goal.cn/](https://superagents.go-goal.cn/) |
| `scripts/` | 数据生成脚本（`gen_worldline_online.py` / `make_panel.py` / `asset_spec.py` / `fetch_daily_data.py`） |
| `docs/` | `process.md`（生成方法）/ `plan.md`（数据生成逻辑）/ `WAYPOINTS.md` / `deep-information-因子挖掘-2026-07-20.md`（详细推演记录） |
| `manifest.json` | 每条 WL 的 warmup 指纹校验结果 |

## 扩展版口径说明

- **WL1-3**：使用 data-prepare 当前完整版 wordline（阶段全部有结束日期），在线段与 data-prepare CSV 逐值一致；warmup 段取自 FM 实际跑的 bundle（确保与 WL4-9 共享同一 `8410ae8b` warmup 指纹）。
- **WL4-8**：bundle 中 FM 实际运行使用的面板（与 data-prepare CSV 一致，仅浮点精度差异）。
- **WL9**：bundle 中 FM 实际运行的**扩展版**面板——阶段三以 2030-12-31 为锚点（对应 wordline9.md 阶段三的 2030 终点，re-anchor 后精确命中）。注意当前 `wordline9.md` 阶段三标题为 `（2030）` 无区间，用当前脚本重新生成会丢失该 2030 锚点；**实验请直接使用本目录 `panels/WL9_full.parquet` 与 `news/WL9_stage_news.json`（扩展版，含 2030-12-31 阶段）**。

## 指纹契约

- 共享 warmup 指纹：`8410ae8bbd86fd8735de5ea4823e4924cebf977e51e2946854378fba46018c28`
- `manifest.json` 中 9 条 WL 的 warmup 指纹均匹配该值（`match=true`）。
- warmup 段（≤2026-07-15 的 tradable 资产）是 `fm_history_digest` 的唯一输入，全部 WL 一致 → 可共享同一 warm-up stage，无需重新进行 200 轮 warm-up。

## 如何运行 WL1-9

将 `panels/WL1-9_full.parquet` 放入 FM bundle 的 `data-prepare/online-worldline/` 后，用 `scripts/gen_worldline_online.py` 或直接复用现有 FM 运行器即可；每条 WL 从 `8410ae8b` warmup stage 续跑，不会重新挖 warm-up。
