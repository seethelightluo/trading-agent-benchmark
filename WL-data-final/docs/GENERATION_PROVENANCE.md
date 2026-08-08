# GENERATION_PROVENANCE.md — 生成逻辑、参数快照与来源链

> 本文档固化 WL-data-final 的完整生成链路：初始研究 → 世界线规划 → 合成数据，并如实标注每一环的**已有证据**与**缺失源头**。
> 更新：2026-08-08（blackswan/final_synthesis 上游文件已补齐）

## 1. 数据生成链路总览

```
[初始研究 + 数据预测]  (来源：sumtime 数据；生成平台：superagents agent 平台 https://superagents.go-goal.cn/)
        │  产出：核心因果链
        ▼
[世界线规划]  blackswan_9worldlines_final_conclusion.md + final_synthesis_report.md —— 见 §4 缺失说明
        │  产出：9 条 WL 阶段终点表（wordline1-9.md 来源标注的"最终完整版总表"）
        ▼
[解析锚点]  parse_worldline() 抽取阶段结束日与资产终点
        │  产物：WL<n>_stage_news.json（含 news_date 与 asset_endpoints）
        ▼
[合成在线数据]  gen_worldline_online.py（参数见 §2）
        │  产物：WL<n>_online.csv + WL<n>_full.parquet（本目录 panels/）
        ▼
[warmup 拼接]  真实历史 ≤2026-07-16 + 在线合成 → full 面板（本目录 warmup/panel.csv）
```

## 2. 合成参数快照（与 scripts/gen_worldline_online.py 逐项核对）

| 参数 | 值 | 记录位置 |
|------|----|---------|
| re-anchor | ON（默认，--no-reanchor 关闭） | process.md §7；WAYPOINTS.md |
| derive-fx | ON（默认） | process.md §6；WAYPOINTS.md |
| price-leads-news | ON（默认，--no-lead 关闭） | process.md §5；WAYPOINTS.md |
| LEAD_TIME_FRAC | 0.35（news 在段内 35% 处破裂） | gen_worldline_online.py；plan.md §7.3-11 |
| LEAD_MOVE_FRAC | 0.25（news 前先走 25% leak） | gen_worldline_online.py；plan.md §7.3-11 |
| GBB 噪声 | ON（默认，--no-noise 关闭）；σ=warmup 已实现波动率，端点归零命中锚点 | process.md §2-3；WAYPOINTS.md |
| 噪声种子 | 每 (WL, 资产)：zlib.crc32(f"{wl}|{aid}")；每段子种子 crc32(f"{seed}|{k}") | process.md §4 |
| FX β | USDCNY +0.303 / USDJPY +0.889 / EURUSD −1.063（warmup 实测，DXY 派生） | process.md §6 |
| 线性资产地板 | VIX≥9.0 / US10Y≥0.05 / CN10Y≥0.05 | process.md §8 |
| warmup 截止 | 2026-07-16（Agent 可见 ≤2026-07-15） | process.md；ASSETS.yaml |

复现命令（默认全开，即 §2 参数）：
```
python scripts/gen_worldline_online.py            # 9 条 WL
python scripts/gen_worldline_online.py --only 1   # 单条
```

## 3. 已有证据（本目录内）

- `panels/WL1-9_full.parquet`：9 条 WL 完整面板（manifest.json 校验 9/9 指纹 = 8410ae8b）。
- `news/WL1-9_stage_news.json`：9 条 WL 阶段新闻（WL9 为扩展版，含 2030-12-31 锚点）。
- `wordline-md/wordline1-9.md`：世界线阶段终点表（来源标注：策略研究员最终完整版总表 2026/07/17 11:52）。
- `docs/deep-information-因子挖掘-2026-07-20.md`：2026-07-17 群聊推演记录（9 条 WL 年度推演、基线统一、半导体挑战、校验修正、最终总表）。
- `docs/process.md`：生成方法单一事实源（σ 取法、GBB 采样、种子、leak、FX 派生、re-anchor、地板）。
- `docs/plan.md`：数据生成逻辑与 7.3 节生成规则。
- `warmup/panel.csv` + `warmup/COVERAGE.md`：真实历史面板（σ/β 计算输入）。
- `scripts/`：gen_worldline_online.py / make_panel.py / asset_spec.py / fetch_daily_data.py。

## 4. 来源标注与缺口说明

| 项 | 说明 |
|--------|------|
| sumtime 初始研究 + 数据预测 | 无需追溯：`research/` 内两个文件即最原始产物。数据来源标注为 sumtime 数据，生成平台为 superagents agent 平台（https://superagents.go-goal.cn/）；不要求还原 sumtime 内部生成过程。 |
| blackswan_9worldlines_final_conclusion.md | **最原始产物**，已归档至 `research/`（SHA256 73195d7f5494c094…）。来源：sumtime 数据；由 superagents agent 平台生成（https://superagents.go-goal.cn/）。 |
| final_synthesis_report.md | **最原始产物**，已归档至 `research/`（SHA256 993ff856a08a1b97…）。来源：sumtime 数据；由 superagents agent 平台生成（https://superagents.go-goal.cn/）。两者为同一份结论的两个措辞版本（725 行 diff）。 |
| 最终完整版总表（wordline md 来源） | wordline1-9.md 头部引用该总表（策略研究员 2026/07/17 11:52）；总表内容以 deep-information 推演记录形式存在，但独立文件缺失。 |
| generation_meta.json | plan.md §7.3-10 要求每资产每月 σ/News 摘要/种子/起止锚审计文件；生成时未保留。 |
| WL9 扩展版 wordline 源 | bundle WL9 面板以 2030-12-31 为锚点（扩展版）；当前 wordline9.md 阶段三 `（2030）` 无区间，无法用脚本复现该锚点；扩展版只能从 panels/WL9_full.parquet 反推。 |

**自洽性结论**：生成链路自洽闭环——`research/`（最原始产物：sumtime 数据 + superagents agent 平台生成）→ deep-information 推演记录 → wordline1-9.md 阶段终点表 → 合成参数（§2）→ 完整面板（panels/），每一环均有据可查。`sumtime` 的内部生成逻辑按约定不追溯，仅作来源标注。"最终完整版总表"独立文件无留存，但内容已以 deep-information 推演记录形式保留。
