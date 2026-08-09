# AC / FM 实验规划演进全景 Checklist

> 重建日期：2026-08-09（Asia/Shanghai）  
> 目的：从工作区 Markdown、Codex 记录和 Claude Code 会话中，恢复 2026-07-26 至 2026-08-06 期间 AC/FM 实验的规划演进、问题、决策、实现变化，以及它们和上游原生框架的差异。  
> 本文先记录“为什么这样设计、做过什么修正、证据在哪里”；按照当前用户要求，**暂不把本文件当作当前代码全量一致性审计，也不在本轮统一比较所有现状**。待用户确认后，再按第 11 节执行统一 current-state comparison。

## 0. 阅读规则与证据等级

- `[已证实]`：有仓库文件、结构化状态、测试、日志或 Codex/Claude 会话直接支持。
- `[历史记录]`：来自指定日期的 Codex rollout summary、Codex session 或 Claude Code JSONL；它描述当时状态，不自动代表 2026-08-09 的 live 状态。
- `[设计决策]`：用户明确提出或文档明确规定的实验合同；不等于已被完整运行验证。
- `[实现修正]`：代码、配置或运行器已经改过，但仍需用第 11 节的统一审计确认端到端效果。
- `[待决策]`：存在口径冲突、证据不足或需要用户选择，不能擅自合并。

本文件刻意不记录 API key、管理员密码、真实 token、完整 Authorization、外部账户标识或网关私密配置。相关文件只作为“路径/系统存在”的证据引用。

---

## 1. 总体结论先行

### 1.1 实验已经从“上游示例直接跑”演进为“可恢复的 15 资产、9 世界线基准”

当前规划链条是：

```text
真实历史 ≤ 2026-07-15
    -> 20 资产输入（15 可交易 + 5 只读信号）
    -> 共享 warmup，只研究不推进交易
    -> 因子准入 / 滚动库 / active top-10 / quality tilt
    -> 2026-07-16 起 9 条独立 WL 前向
    -> 每 10 个交易日一次 Agent 决策，日内本地估值
    -> 15 资产、long-only、fractional、cash=0、后续迁移 3 bps
```

### 1.2 最大的规划修正不是“把 warmup 改成最多 10 个因子”

最终口径已经纠正为三层生命周期：

1. 候选因子经过 IC、ICIR、相关性门槛；
2. 当前滚动因子库最多 30 个，超出时从质量末尾淘汰；
3. 每个决策时点从当前库选最多 10 个活跃因子，按质量/IC 做 tilt，可以少于 10 个。

因此，“最多 10”是 active ensemble 的上限，不是 warmup 研究数量上限，也不是因子库容量上限。

### 1.3 当前仍然需要区分三种“通过”

- API `/v1/models` 或一次 function-call 成功：只证明协议探测通过；
- warmup workflow 完整、因子产物/ensemble/checkpoint/日期边界正确：证明研究阶段通过；
- 9 条 WL 的每个 online block 都满足 15 持仓、cash=0、权重和为 1、3 bps 成本和最终日期：才是完整 AC/FM 前向实验通过。

---

## 2. 基线：进入 7月26日之前已经确定的实验宇宙

### 2.1 数据宇宙

- `[已证实]` 基准为 15 个可交易资产 + 5 个只读指数/宏观/汇率信号，共 20 个输入资产。
- `[已证实]` 可交易集合：8 权益、3 商品、2 加密、2 债券；只读信号为 `DXY`、`USDCNY`、`USDJPY`、`EURUSD`、`VIX`。
- `[设计决策]` 信号资产可用于状态识别、美元折算和研究，但不进入 15 项持仓权重。
- `[已证实]` 真实历史冻结在 2026-07-15；2026-07-16 以后是合成世界线，不得把合成未来当成真实市场结果。
- `[已证实]` 九条 WL 应在当时可见数据内前向揭示，不能按完整 WL 结果人工调参或挑选世界线重跑。

主要证据：`plan.md`、`process.md`、`agent-framework/progress.md`、`agent-framework/ASSETS.yaml`、`WL-data-final/README.md`。

### 2.2 数据生成与来源演进

- 初始计划由 `data-prepare/asset-daily-data`、`wordline-simple` 和 `gen_worldline_online.py` 生成世界线。
- 数据层后来补齐 USD 计价、EURUSD、FX beta 派生、price-leads-news、re-anchor、GBB 噪声与确定性种子。
- `WL-data-final` 是 8月8日整理出的最终实验快照，包含 9 个 panel、9 个 stage-news、worldline markdown、warmup panel、scripts、provenance 和 manifest。
- `[已证实]` WL1–8 的最终 panel 可由 warmup + wordline 源在浮点舍入范围内重建，新闻 JSON 一致；WL9 因阶段三只有“2030”锚点而不能由当前解析器完整重建，必须直接使用最终快照。
- `[已证实]` 当前生成包没有逐面板完整 SHA manifest，也没有完整 `generation_meta.json`；manifest 中的 `8410ae8b...` 是共享 warmup 指纹，不是整个 9-WL 面板指纹。
- `[已证实]` 若干信号资产合成 high/low 为空，原因是零振幅 fallback 缺失；部分 stage end 是周末锚点而非可成交 bar。

这意味着：`WL-data-final` 是运行时权威输入快照，但不是“默认脚本无修改即可从零生成且 OHLC 完整”的自包含生成器。

---

## 3. 规划演进时间线（2026-07-26 至 2026-08-06）

### 3.1 2026-07-26：FM 长跑契约、恢复能力和归档标准成形

证据：Codex rollout `2026-07-26T07-41-40-fQcX-fm_ac_runtime_and_complete_nine_worldline_archive.md`，以及当前 `runFM.md` 对该会话结果的恢复。

#### 做出的决定

- `[设计决策]` 因子库容量 `<=30` 时全部保留；只有 `>30` 才按 `abs(IC) * abs(ICIR)` 保留 best 30。
- `[设计决策]` trim 必须同时更新导出库、checkpoint/library 和 signals；resume 不得复活已淘汰因子。
- `[设计决策]` 正式 API 失败采用 `0s -> 60s -> 600s -> 3600s` 退避，成功后复位；`--max-attempts 0` 表示无限重试。
- `[实现修正]` 日频 singleton preprocessing 走 P0 快路径，候选评估走确定性 ProcessPool；父进程独占持久化写入，避免并发写坏 library/checkpoint。
- `[实现修正]` Windows 加速 bundle 与 Linux runtime 分开，不能用 Windows bundle 覆盖正在运行的 runtime。
- `[实现修正]` 旧 fingerprint 到新 fingerprint 用性能等效证书桥接，先备份 window state，只允许经证书批准的字段迁移，避免不必要的重新 warmup。
- `[实现修正]` AC WL 并发上限固定为 2；不能只看 PID 退出，必须以 `simulation_complete=true` 作为 durable completion gate，再推进下一个 WL。

#### 当时遇到的问题

- 初始测试把 `FactorMiner.factorminer` 当成导入路径，实际包路径是 `factorminer`，导致测试失败；修正为把 `agent-framework/FactorMiner` 放入路径。
- 旧 bundle 的 fingerprint/代码与 Linux 当前实现不一致；通过复制 P0/P1、更新 verifier/merge/export fingerprint 和证书解决。
- AC 并发如果只按进程数量推进，会出现 WL3 未真正完成就启动后续 WL 的风险；改为完成标记闸门。

#### 结果和证据边界

- `[历史记录]` FM WL1–WL9 当时已有完整结构化归档：每条 246 个 online windows、2468 行 equity、247 次决策、最终到 2035-12-31，cash=0、无 lookahead、库容量不超过 30。
- `[历史记录]` 归档逐文件 SHA 为 13533/13533，来源到归档等价审计通过。
- `[历史记录]` 这证明当时 FM 归档的完整性，不自动证明后来切换 API、修改 AC online 实现后的新运行结果。

### 3.2 2026-07-25/26 的 FM 门槛修订：留下了一个必须保留的口径分叉

证据：`agent-framework/progress.md` §0、`runFM.md` §2、`runAC.md` §2、`ASSETS.yaml`。

- 早期 15 资产缩放曾写成 `abs(IC)>=0.04`、`abs(ICIR)>=0.10`；ICIR 由 `0.5*sqrt((15-1)/(500-1))=0.08375` 向上取 `0.10`。
- 后续正式 AC/FM runbook 改成 `abs(IC)>=0.007`、`abs(ICIR)>=0.084`，即将论文门槛按横截面规模直接缩放：

  ```text
  0.04 * sqrt(14/499) ~= 0.00669 -> 0.007
  0.50 * sqrt(14/499) ~= 0.08375 -> 0.084
  ```

- `[待决策]` 这是规划演进，不应把两个值假装成同一时期的合同。统一比较时必须以当时具体 run 的 fingerprint、ASSETS.yaml、config 和日志为准。
- `[历史记录]` progress.md 记载的第一轮真实新门槛 smoke 是 5 轮、batch 8、40 candidates、1 admission；随后暴露 NumPy `ndarray` 直接 JSON 序列化 bug。
- `[实现修正]` combine 现在递归转换 NumPy ndarray/scalar；真实 1 因子组合能保存 `ls_cumulative` JSON array。
- `[实现修正]` 修改代码指纹后没有伪造旧 checkpoint，而是按新指纹重新跑合法 smoke；第二次随机候选无录取也被视为合法的全现金结果。

### 3.3 2026-08-03：把“能跑”推进为“可持续跑”

证据：Codex 7/26 rollout 中的 P0/P1 记录、`runFM.md` §5/§6、`FM acceleration/P0_P1_PERFORMANCE_EQUIVALENT_UPDATE_20260803.md`。

- `[实现修正]` expanding-window 下，online mining 前刷新 checkpoint/library 的 signal shape；导出库也同步刷新。
- `[实现修正]` trim 后 resume 不得恢复旧大库；对顶层库和 checkpoint/library 同步写入并校验 factor IDs。
- `[实现修正]` 正式运行不得使用有限 3 次上限；API 临时失败必须持续重试。
- `[已证实]` 真实 35,184 行 daily panel 的 preprocessing 从约 318 秒降至约 14.4 秒；40 candidate 隔离 Ralph smoke 约 279.5 秒。
- `[证据边界]` 全量 warmup 预估约 18–24 小时，只是估计，不是成功保证；真实耗时受模型响应、限流和录取率影响。

### 3.4 2026-08-04：从“因子挖掘”扩展到可审计的条件/洞察驱动挖掘

证据：Codex rollout `2026-08-04T03-25-42-fMhK-mechanism_c_dynamic_factor_mining_research.md`、`research/answer-research1.md`。

- `[已证实]` FactorMiner 原生已有 Retrieve -> Generate -> Evaluate -> LibraryUpdate -> Distill，带 typed DSL、memory policy、regime-aware retrieval、specialist/debate、checkpoint 和 provenance。
- `[已证实]` AlphaCrafter 已有 `factors/*.json`、IC/ICIR/相关性门、revalidation/deprecated、`memory.txt` 和 workflow checkpoint，但没有 FM 式统一结构化经验蒸馏闭环。
- `[设计决策]` 仅在 prompt 中加入“换方向”属于 steering，不是 conditional mining。
- `[设计建议]` 若继续做机制 C，需要一等公民的 `ResearchMandate`/`ObjectiveSpec`，固定数据截止、universe、targets、gates、prompt hash，在 round 边界切换方向；memory 要分 global/objective/context scope；候选轨迹和 provenance 必须可审计。
- `[契约限制]` 当前 15 资产 fully-invested long-only 只能把负向研究映射为低配；不能直接执行半导体净空头，除非另行改变做空/保证金/风险合同。

这一阶段的核心变化是：研究方向可以受人工洞察引导，但不能绕过数据 cutoff、因子准入、持久化和反证机制。

### 3.5 2026-08-06：AC 运行暂停、备份、provider 切换和协议分层

证据：Codex rollout `2026-08-06T07-48-28-T0XX-ac_runtime_backup_relay_params_and_date_diagnosis.md`、Claude 会话 `1302022e-...jsonl`、`066bea6e-...jsonl`、`runAC.md` §4–§9。

#### AC 运行方式被固定

- AC 子进程必须在 `agent-framework/AlphaCrafter/alphacrafter` cwd 运行，同时把 `AlphaCrafter/` 加入 `PYTHONPATH`。
- 当前 CLI 使用位置参数 `main.py wlN --config run_config.yaml --resume`，不能照抄上游的 `--session_id`。
- scheduler 启动时读取 `.env`；仅修改 `.env` 不会改变已经运行的 supervisor/子进程，切换 provider 必须停旧树并用新环境重启。
- SIGSTOP 子进程后，旧 supervisor 不会自动重启它们，因此可以先冻结、备份，再切换端点。

#### 备份和恢复决定

- `[用户要求]` 备份必须包含 AC sandbox、warmup、所有 WL 进度、日志和状态，而不是只保留最终报告。
- `[历史记录]` `AC_backup_20260806_155647` 复制了 WL1–WL9 sandbox 和 AC results，并做了关键状态 SHA 对照。
- `[历史记录]` 当时 WL1–4 已完成，WL5/WL6 未完成，WL7–9 尚未开始；这是 8月6日快照，不是当前 live 判断。

#### WL 日期“回退”被重新解释

- `[已证实]` BacktestTool 会先把 live `date.json/account.json` 存入 `backtest_transaction.json`，临时把 live 日期回卷到历史回测窗口，结束后在 `finally` 恢复。
- 因此 supervisor 瞬时读取 `date.json` 可能看到较早日期；判断是否丢进度必须联合检查 transaction marker、`snapshot.json` 的单调序列、live date 和日志。
- 这不是把未来数据泄漏进 online 的许可；它只是 AC 内置研究工具对 live 状态的临时复用，统一审计仍需确认恢复边界。

#### Terra/Luna 外部 relay 决策

- `[用户决策]` 不改项目内 provider 逻辑；在项目外建立 `/home/lxx/ac-llm-relay/`，项目侧只把 base URL 指向本地 relay，配置中的模型名仍保留 `gpt-5.6-terra`。
- relay 将 Terra 请求转到上游 Luna；AC Responses 和 FM Chat Completions 均走 OpenAI-compatible `/v1`。
- `/v1/models`、`/v1/responses`、`/v1/chat/completions` 返回 200 只证明协议探测通过，不能证明完整 warmup/WL 完成。
- relay 必须以 `setsid nohup` 脱离命令会话；普通 nohup 曾出现会话结束后 relay 消失。
- `[参数证据]` AC Responses 未显式传 temperature/top_p/seed/reasoning；记录为 `mode=standard, effort=medium`，temperature/top_p 走上游默认。FM 代码则发送 `temperature=0.8,max_tokens=4096`，但具体网关可能忽略其语义。

#### sub2api 和 DeepSeek 路线

- `[历史记录]` Claude Code 于 8月6日创建 `/home/lxx/sub2api-codex`，以 Docker Compose 启动 sub2api，端口 8080，配置 OpenAI API 反代。
- `[历史记录]` 同日另建 `/home/lxx/opencode-api` 的 DeepSeek 免费网关，目标是多 key、WARP 出口、OpenAI/Anthropic 兼容和重试；这是外部网关工程，不等同于 AC 原生协议。
- `[用户决策]` DeepSeek AC 后续应通过 sub2api 输出 Responses 兼容接口，尽量复用 Luna 的 prompt、tool schema 和 Responses history；不再依赖简单的本地 Python Chat->Responses 转换。
- `[证据边界]` 8月6日完成的是网关部署/路由方向和 AC 暂停切换上下文；免费 DeepSeek 的 429、IP 池滚动、兼容性失败等主要问题发生在 8月8日以后，不应倒填到 7/26–8/6 的规划事实中。

---

## 4. AC 规划演进：原生流程到 benchmark 合同

### 4.1 保留的原生 AC 骨架

根据 `agent-framework/AlphaCrafter/README.md`、`main.py`、配置和运行文档，原生骨架仍是：

1. Miner 发现/写入因子；
2. Screener 读取研究结果、筛选和组合；
3. Trader 写策略并执行交易；
4. 工具循环通过 OpenAI Responses 的 function call/function output 推进；
5. sandbox、workflow log、account、date 和策略文件构成可恢复状态。

### 4.2 benchmark 对 AC 的显式变化

| 维度 | 原生/上游倾向 | benchmark 当前规划 |
|---|---|---|
| 研究频率 | README 描述 daily rotation | Agent cycle 每 10 个交易日；本地行情/估值仍每日推进 |
| universe | 上游示例偏股票横截面/CSI300 | 15 可交易跨资产 + 5 只读信号 |
| 研究边界 | session 自由使用数据 | 研究 cutoff 固定为 2026-07-15 |
| warmup | 原生没有本 benchmark 的共享 40-cycle 契约 | 40 个 cycle，3 Miner 并发 + Screener + Trader |
| 因子门槛 | 上游/论文值不直接适配 15 资产 | `0.007/0.084/rho<0.5`，但早期曾有 `0.04/0.10` |
| 因子库 | 原生有因子保存/复验，但无本合同滚动三层定义 | 持久库 <=30；超出按 q=abs(IC)*abs(ICIR) 淘汰 |
| active ensemble | 原生模板有固定 `FW` 结构 | 每次从当前库选 <=10，质量/IC tilt，可少于 10 |
| 交易资产 | 上游模拟可含现金/整数交易语义 | 只有 15 个 tradable，long-only，fractional，cash=0 |
| 首次建仓 | 不属于原生论文骨架的 benchmark 约束 | 2026-07-16 将 1M 全部配置到 15 个资产 |
| 摩擦 | 上游示例成本模型不同 | 首次免费，后续资产迁移额一次收 3 bps |
| 数据 | 原生 sandbox/template | scheduler 固定读取 `WL-data-final/panels` 和 `news` |
| provider | 原生配置直接调用 provider | 项目模型名保留 Terra，由外部 relay 映射 Luna/DeepSeek |
| 并发 | 原生 README 不定义本实验 durable gate | AC scheduler 最大并发 2，完成标记才推进 |

### 4.3 AC 中最容易重复出现的错误

- 把 `active <=10` 写成 `warmup <=10`；
- 使用固定 `FW` 代替动态 active ensemble；
- 因子通过只写在 Miner 文本/日志，没有落盘 JSON、公式和同周期 IC/ICIR；
- 复用旧 workflow/checkpoint 时没有验证 fingerprint、date boundary 和 frozen account；
- 只修改 `.env` 后 SIGCONT，实际旧进程仍使用旧端点；
- 把 supervisor 瞬时读取的回测日期当成 live 日期回退；
- 只用 API 200 或 function-call 探测作为完整 AC 成功证据；
- 仍使用旧数据目录或旧的 `run_state.json` 作为跳过依据；
- 旧式整手/整数数量导致首次 15 资产分配不全，产生现金残留；
- Trader 的 cash/no-trade 语义与 fully-invested 合同冲突。

---

## 5. FM 规划演进：原生 Ralph 到 9-WL 前向合同

### 5.1 保留的原生 FM 骨架

根据 `agent-framework/FactorMiner/README.md`、`docs/architecture.md`、`docs/metrics.md` 和 `factorminer` 代码，原生/仓库当前骨架包括：

- typed factor DSL 和 expression tree；
- RalphLoop（paper-style mining）与 HelixLoop（扩展研究）；
- Retrieve -> Generate -> Evaluate -> LibraryUpdate -> Distill 阶段；
- IC/ICIR、相关性/冗余和 library admission；
- structured memory、memory policy、prompt context、lifecycle、checkpoint/provenance；
- runtime recomputation：分析/benchmark 应根据公式和输入数据重新计算，而不是盲信保存的摘要。

### 5.2 benchmark 对 FM 的显式变化

| 维度 | 原生/仓库框架 | benchmark 当前规划 |
|---|---|---|
| 数据 | 论文/仓库含股票或 Binance paper-style 配置，另有小样本 smoke | 20 资产日频 panel，15 可交易 + 5 signal |
| warmup | Ralph/Helix 按配置迭代 | 共享 warmup 200 iterations、目标池 110、每批 40 candidates（AC 另有 40 cycles） |
| library | 有 admission/replacement | <=30 全保留；>30 按 `abs(IC)*abs(ICIR)` trim；export/checkpoint/signals 同步 |
| forward | runtime benchmark 可冻结 Top-K | 每 10 个交易日刷新、online Ralph 一次、trim、combine、原子调仓 |
| active | paper benchmark 有 Top-K freeze | 当前活跃因子最多 10，并保留 sign(IC) 和 tilt |
| 组合 | 原生 benchmark 组合口径依配置 | 15 项权重和 1、long-only、fractional、cash=0 |
| 成本 | 非统一上游成本 | 首次免费，后续迁移名义额 3 bps |
| 世界线 | 原生没有九条本项目 WL 运行器 | 共享 warmup 后 9 条独立 forward，不能用未来 WL 结果调参 |
| 重试 | 框架自身可配置 | 正式任务无限重试，0/60/600/3600 退避 |
| 性能 | 原生结构功能优先 | P0 singleton + P1 ProcessPool + 父进程持久化 |

### 5.3 FM 中必须继续防的错误

- trim 只更新导出文件，不更新 checkpoint/library，导致 resume 复活淘汰因子；
- expanding window 改变 signal shape 后继续使用旧 checkpoint；
- 以 `mean(abs(IC_t))` 代替 paper-mode `abs(mean(IC_t))`；
- 把 mock 0 因子当成 pipeline 失败，或把真实短 smoke 当成全量成功保证；
- 以 finite retry 结束任务，造成暂时 API 错误被误报为实验终止；
- 并行 worker 直接写持久化库，造成重复、丢失或非确定性状态；
- 把 110 目标池误写为最终库容，或把 10 active top-k 误写成研究数量限制；
- 使用历史完整世界线结果人工挑选参数，破坏 forward benchmark 的因果口径。

---

## 6. 问题—决策—修正矩阵

| ID | 阶段 | 问题 | 决策/修正 | 当前证据状态 |
|---|---|---|---|---|
| P01 | 7/25–7/26 | 15 资产门槛从论文股票口径直接照抄不合适 | 曾先试 `.04/.10`，后形成 `.007/.084` 缩放口径 | `[待决策]` 必须按具体 run fingerprint 统一 |
| P02 | 7/25 | combine 不能 JSON 序列化 ndarray | 递归转换 ndarray/scalar，真实 1 因子复验 | `[已证实]` smoke 记录支持 |
| P03 | 7/25 | `miner_miner_1` phase 名被验收器漏认 | 验收器同时接受实际双前缀和旧前缀 | `[实现修正]` 有 workflow 验收证据 |
| P04 | 7/25 | resume 到 max_cycles 仍启动新 cycle | 在创建 Agent 前早退 | `[实现修正]` 有测试 `test_resume_at_max_cycles...` |
| P05 | 7/26 | trim 后 checkpoint 恢复旧大库 | export/checkpoint/signals 同步 trim + IDs 校验 | `[历史记录]` FM 归档和 runbook 支持 |
| P06 | 7/26 | API 暂时错误结束长任务 | max-attempts=0 + 0/60/600/3600 无限退避 | `[实现修正]` runbook 已记录 |
| P07 | 7/26–8/3 | preprocessing/candidate evaluation 太慢 | P0 singleton、P1 ProcessPool、父进程独占写入 | `[已证实]` 性能数据和测试支持 |
| P08 | 7/26 | AC 并发按 PID 推进不可靠 | 最大并发 2，以 `simulation_complete` durable gate 推进 | `[历史记录]` controller 证据 |
| P09 | 8/6 | 仅改 `.env` 无法切 provider | 停旧树、外部 relay、重启 supervisor | `[历史记录]` relay/进程证据 |
| P10 | 8/6 | WL 日期看似回退 | 检查 backtest transaction/snapshot/log；识别临时回卷 | `[已证实]` BacktestTool 路径和快照证据 |
| P11 | 8/6 | 备份只保留最终报告不够恢复 | 复制 sandbox、results、logs、state、进度和 SHA | `[历史记录]` AC backup 证书 |
| P12 | 8/6 | API 200 被误当完整 AC 成功 | 将协议探测、warmup 通过、WL 完成分级 | `[设计决策]` runAC §6 |
| P13 | 8/6 | 直接把 DeepSeek Chat 结果脚本转换为 Responses 不稳 | 改走 sub2api Responses bridge，保持 AC prompt/tool/history | `[历史记录/实现修正]` 8/6 部署方向 |
| P14 | 7/26–8/6 | 原生固定 FW 与动态 top-k/tilt 不一致 | active ensemble 必须显式持久化，固定 FW 不能冒充同步合同 | `[设计决策]` 仍需端到端验收 |
| P15 | 7/26–8/6 | 交易整数/订单部分成交导致 cash 残留 | online fractional + 原子 rebalance + fully-invested 兜底 | `[实现修正]` 代码/测试有证据，需 WL 审计 |
| P16 | 7/26–8/6 | 生成脚本与最终数据包路径漂移 | scheduler 直接固定 `WL-data-final/panels/news`；生成器缺口单列 | `[实现修正]` 数据审计已记录 |
| P17 | 8/4 | prompt 加方向不是条件挖掘 | mandate/objective/evaluator/memory/provenance 必须一起变化 | `[研究结论]` answer-research1 已核验 |

---

## 7. 运行与恢复的统一操作原则

### 7.1 任何长任务启动前

- [ ] 确认没有重复的 scheduler、supervisor、`main.py wlN` 子进程。
- [ ] 确认要跑的 provider、model、base URL 是同一份启动环境，不只看 `.env` 文件。
- [ ] 确认 `WL-data-final` 的 panel/news 存在，不能静默回退旧目录。
- [ ] 确认 warmup fingerprint、代码 fingerprint、资产合同和 config 互相匹配。
- [ ] 确认 warmup session 是研究冻结状态：日期为 `2026-07-16`、visible through `2026-07-15`、账户仍全现金无持仓无订单。
- [ ] 确认 active ensemble 中每个 factor ID 都在当前滚动库内，库容量不超过 30，active 不超过 10。
- [ ] 确认备份在任何会写 state 的进程暂停后制作，并带关键文件 SHA。

### 7.2 每个 online block 后

- [ ] 因子库是否滚动 trim，且 export/checkpoint/signals ID 一致。
- [ ] 是否显式保存当前 active factor IDs、方向、质量分数和 tilt 权重。
- [ ] 是否只含 15 个 tradable positions，5 个 signal 没有被买入。
- [ ] 权重是否非负且和为 1；数量是否允许小数；cash 是否为 0。
- [ ] 首个 2026-07-16 block 是否完整配置 1M；后续 rebalance 是否只收迁移额 3 bps。
- [ ] 若日期看似回退，是否先查 `backtest_transaction.json` 和 `snapshot.json`，而不是直接重置 cursor。
- [ ] 是否把失败保留在原 block 的 retry/resume 边界，而不是无证据地推进到下一个 block。

### 7.3 每个大阶段结束后

- [ ] 本机 git 记录代码、配置、文档、运行结果和状态（环境/凭据除外）。
- [ ] 远端 GitHub 推送阶段性 commit。
- [ ] data 盘生成完整快照，保存最近 5 次，并记录 snapshot 时间、commit、运行状态和 SHA。
- [ ] README/progress/checklist 同步写清“已完成、未完成、排除项和恢复命令”。

---

## 8. 7/26–8/6 期间已经形成的参数清单

### AC

- [x] 3 Miner 并发，然后 Screener，再 Trader。
- [x] 共享 warmup 40 AC cycles；每 cycle 5 个 Agent 步骤。
- [x] AC Responses API，function_call -> 本地工具 -> function_call_output。
- [x] 15 tradable + 5 signal；研究 cutoff 2026-07-15。
- [x] `abs(IC)>=0.007`、`abs(ICIR)>=0.084`、最大 abs Spearman rho `<0.5`（注意早期 `.04/.10` 分叉）。
- [x] persistent library <=30，超出按 `abs(IC)*abs(ICIR)` 从末尾淘汰。
- [x] active <=10，质量/IC tilt，可少于 10。
- [x] online 10 trading-day cadence，fractional，long-only，15 assets，cash=0。
- [x] 2026-07-16 首次 1M 全投；后续迁移 3 bps。
- [x] AC direct CLI 位置参数和 cwd/PYTHONPATH 约束。
- [x] 运行切 provider 前停止并重启 supervisor/子进程。

### FM

- [x] Ralph mining -> combine -> forward。
- [x] 200 iterations、target pool 110、batch 40（这是 FM warmup 口径，不是 active top-k）。
- [x] <=30 全保留，>30 按质量 trim best30；checkpoint/export/signals 同步。
- [x] 每 10 trading-day online refresh/mining/combine/rebalance。
- [x] 15 assets、fractional、long-only、cash=0、首建仓免费、后续 3 bps。
- [x] P0/P1 加速、无限重试、结构化 monitor、可恢复 state。
- [x] 9-WL 结果归档有完整性审计，但该历史归档必须与后来 provider/代码变更区分。

### Provider/运维

- [x] Terra 作为项目内模型名，外部 relay 映射实际 Luna。
- [x] AC/FM 端点要求 OpenAI-compatible `/v1`。
- [x] sub2api-codex 项目外部署，8月6日服务健康并监听 8080。
- [x] DeepSeek 路线不把 keys 或真实凭据写入仓库；凭据应在外部配置。
- [x] 暂停时使用 SIGSTOP/停止 supervisor，备份后再切换环境。
- [x] 当前 2026-08-09 Luna 3WL runner 已暂停；本 checklist 不恢复它。

---

## 9. 已确认正式合同（2026-08-09）

以下口径覆盖 AC 两个版本和 FM benchmark live 路径；历史运行若使用过其他口径，必须保留原始配置并标记为非正式 benchmark 结果。

- 因子准入只承认 `abs(IC) >= 0.007` 且 `abs(ICIR) >= 0.084`；`.04/.10` 仅可作为历史/native paper 配置，不能作为本实验正式结果。
- 因子质量为 `quality = abs(IC) * abs(ICIR)`。同一可见数据、共同有效样本上的 `abs(Spearman rho) < 0.5` 时两个因子都保留；`>= 0.5` 时只保留质量较高者，淘汰较低者；同质量按稳定 factor ID 决定。
- 因子库容量为 30，先做门槛和 pairwise 冲突淘汰，再按质量保留 best30；活跃 ensemble 最多 10 个，可少于 10 个，按质量 tilt 分配权重。滚动更新时淘汰质量较低或冲突的因子，不允许被 resume 复活。
- 所有历史因子若缺少可重算 formula/signal/provenance，不得把缺失信号静默当作 `rho=0`；必须恢复信号或进入 quarantine，并写入迁移审计。
- 交易 universe 固定为 15 个 tradable 资产，long-only、允许小数持仓、零权重资产允许存在，现金始终为 0。
- 首次 `2026-07-16` 全部 `1M` 建仓：有合格 ensemble 时使用其目标权重；没有时使用 15 资产等权 `1/15`。首次建仓豁免成本门控。
- 后续每 10 个交易日才产生一次研究/交易决策。共同 forecast predictor 产生 proposed target；`one_way_turnover = 0.5 * sum(abs(target-current))`，`gross_edge_bps = 10000 * sum((target-current) * forecast_returns)`。
- 固定决策门槛为 3bp：只有 `gross_edge_bps > 3.0` 才执行，`<= 3.0` 一律 no-trade。实际成本为 `NAV * one_way_turnover * 3 / 10000`；成本门控不改变因子准入。
- no-trade 必须持久化 proposed target、forecast、研究因子、turnover、edge、skip reason；executed target 和真实持仓保持不变。账户修复不得把合法 no-trade 变成交易，原始 `add_order` 不得绕过统一 gate。
- AC 和 FM 使用同一 proposal/gate 字段与 deterministic fixture；Luna 3WL 在代码、迁移、单测和 smoke 完成前继续暂停。

## 10. 当前代码审计—不一致项与解决清单

以下每项都记录当前证据、违反合同、解决方式、验收条件和是否需要旧状态迁移。运行结果、因子 JSON、sandbox、DeepSeek/Luna 状态不在本轮文档提交中修改。

### 10.1 配置与准入

1. **决策门槛仍为 6bp**。证据：`agent-framework/ASSETS.yaml:53` 和 AC-deepseek 对应配置为 `min_round_trip_edge_bps: 6`。违反：固定 3bp 且严格 `edge > 3bp` 才交易。解决：改为 benchmark `decision_edge_threshold_bps: 3`，同时保留单边 `friction_bps: 3`；不再用双边 6bp 推导门槛。验收：配置、forward state、audit 均为 3.0；2.99/3.00 跳过、3.01 执行。迁移：需要为新 contract/code fingerprint 重新标记旧 forward state，不覆盖旧结果。
2. **FM benchmark live 仍读取 `.04/.10`**。证据：`FactorMiner/factorminer/configs/fm_live.yaml` 为 `0.04/0.10`，native `default.yaml`/`walkforward.yaml` 亦有旧 paper 默认。违反：正式 benchmark 路径只承认 `.007/.084`。解决：修改 `fm_live.yaml` 为正式值，并在 native 配置中明确 `non_benchmark_paper_config`，让 scheduler 注入 benchmark contract。验收：live manifest 和运行时配置不再出现 `.04/.10`。迁移：旧 FM library/checkpoint 需按新门槛离线重审，不能当作新合同结果。
3. **AC 只检查候选自报的最大相关性**。证据：`factor_contract.py:evaluate_factor()` 读取 `max_abs_library_correlation`，在 `>=0.5` 时拒绝，但不重算候选与已存因子的 pairwise rho，也不比较双方质量。违反：相关冲突必须比较双方 `abs(IC)*abs(ICIR)`。解决：引入共享 deterministic library policy，要求共同样本 signal，冲突时保留高质量者。验收：相关/质量 fixture 结果确定，低质量因子进入 evicted/quarantine，不能仅靠自报 0.0 通过。迁移：已有 AC 因子必须恢复 signal/provenance；无法恢复者 quarantine。
4. **AC `enforce_library()` 只有容量超限排序**。证据：当前只在超过 30 个文件时按质量排序，没有 pairwise 冲突淘汰。违反：pairwise 冲突处理必须先于 best30。解决：先应用门槛，再共同样本 pairwise 淘汰，再按 quality 和稳定 ID 排序截断 30，并同步 audit/ensemble。验收：`rho<.5` 双保留、`rho>=.5` 只保留高质量、容量/resume 确定。迁移：需重建 AC library/checkpoint/signals/audit 的一致快照。
5. **历史 AC artifact 缺少可重算 signal**。证据：大量 factor JSON 为 `formula: null`，已有 `max_abs_library_correlation=0` 不能证明真实 rho 为零。违反：缺失 signal 不得静默等于零相关。解决：恢复真实 signal 或 provenance；不能恢复则写 quarantine 原因，禁止进入正式 library。验收：每个正式因子有 signal hash/共同样本来源，或明确 quarantine。迁移：是，且不能重新消耗已完成的 LLM warmup。

### 10.2 AC 执行层

6. **两个 AC 版本执行层未完全一致**。证据：两份 `factor_contract.py`、rebalance helper、`add_order.py` 一致，但 `step.py` SHA 不一致。违反：Luna/DeepSeek 必须使用同一代码合同。解决：统一 `step.py` 及其 proposal/gate 调用路径。验收：两份执行文件 SHA 一致，共同 fixture 输出一致。迁移：旧运行保留；新 fingerprint 后才能恢复。
7. **策略可直接 rebalance，原始 order 仍可用**。证据：策略直接调用 `rebalance_to_weights()`，`add_order` 仍可调用。违反：Trader/Screener 只能产生 proposal，不能绕过 gate 改账户。解决：统一 deterministic execution firewall；`add_order` 只能被拒绝或转成 proposal，不能直接改变 benchmark 账户。验收：绕过 gate 的测试失败并留下审计记录。迁移：需要迁移账户执行审计，不修改历史成交。
8. **AC 只有实际迁移成本，没有预测 edge gate**。证据：`rebalance_to_weights()` 能按单边迁移额扣 3bp，但无固定 3bp 预测收益门控。违反：`edge<=3bp` 必须跳过。解决：接入共同 forecast/proposal gate，区分 proposed/executed target。验收：2.99、3.00、3.01 fixture 分别得到 skip、skip、execute，成本为单边迁移额×3bp。迁移：需要新的 decision audit schema。
9. **账户自愈可能覆盖合法 no-trade**。证据：`ensure_fully_invested()` 在策略 hook 后可自动 rebalance；当前不能区分账户损坏修复和合法不调仓。违反：no-trade 时真实持仓、executed target 不变。解决：只在明确 cash/position 损坏时以 repair reason 执行；正常 portfolio 不得触发 fallback rebalance。验收：合法 no-trade 不被自愈改写，损坏账户仍可修复。迁移：旧 `last_target_weights` 需核验后再作为 repair fallback。
10. **AC 当前策略含固定资产 FW/直接权重**。证据：workspace strategy 存在固定 FW 和直接 rebalance。违反：正式策略须由动态 active ensemble 和确定性 forecast 驱动。解决：统一 proposal 生成接口，保留因子质量 tilt，禁止固定 FW 绕过动态库。验收：同一 factor fixture 下两份 AC 输出同一 target/gate。迁移：已有策略文件只作为历史 artifact，不直接当新合同执行代码。

### 10.3 FM 前向与合同持久化

11. **FM turnover 是双边 L1**。证据：`fm_walk_forward.py` 使用 `sum(abs(target-current))`，而 AC 使用单边迁移额。违反：正式合同统一 `0.5*sum(abs(delta))`。解决：统一 helper、预测 edge、实际成本和 decision 字段。验收：同一 target/current 在 AC/FM turnover、cost、gate 完全相同。迁移：旧 forward state 的成本和 edge 不能直接与新结果拼接。
12. **FM 首次无因子可能保持全现金**。证据：`_target_weights()` 无有效因子时返回空 target，旧测试 `test_first_trade_uses_baseline_open_and_does_not_buy_all_assets` 体现旧口径。违反：首次 `2026-07-16` 无 ensemble 也必须 15 资产等权全仓。解决：首个决策空 target 时生成 `1/15`；后续空 target 只保持真实持仓。验收：首笔 15 资产、cash=0、fractional quantity；后续空 target 不变仓。迁移：旧首次全现金结果不能标为正式合同结果。
13. **FM 有 forecast，AC 没有同套 predictor**。证据：FM `_target_weights()` 已计算 factor direction/rank/vol-adjusted forecast；AC workspace strategy 仍有固定 FW/手写权重。违反：AC/FM 必须共用 deterministic predictor 和 proposal schema。解决：抽出共享 predictor/gate 纯函数，AC 和 FM 使用相同输入/fixture。验收：共同 fixture 产生相同 forecast、turnover、edge、决策。迁移：需要为 AC 旧 ensemble 补充 forecast provenance。
14. **文档未完整写清决策成本语义**。证据：`runAC.md`/`runFM.md` 只记录 3bp 执行成本，未完整记录固定 3bp gate、`edge<=3bp`、proposal/executed 分离、no-trade 持久化。违反：运行、恢复、验收必须可由文档重现。解决：同步 runbook 和 checklist；运行状态写入 schema/version/fingerprint。验收：新运行可仅依 runbook 重建 gate 与 cost。迁移：旧 state 需注明缺少字段，不能静默补写为真实历史。
15. **旧运行结果与新合同混杂风险**。证据：当前 worktree 有大量日志、因子、sandbox、DeepSeek/Luna 状态修改。违反：本次代码/文档提交不得覆盖或混入运行结果。解决：只做 scoped staging；新合同生成新 fingerprint，运行前复制 warmup/library/WL state/account/workflow/logs。验收：git diff --cached 只含预期文档或对应代码；Luna 继续暂停。迁移：是，且必须保留 WL-data-final、旧因子 artifact 和运行状态原样备份。

### 10.4 已解决的原待决策项

- [x] `.04/.10` 与 `.007/.084`：正式合同只认 `.007/.084`，前者标为历史/native paper。
- [x] rho 方向：`abs(Spearman rho) < .5` 双保留；`>=.5` 按 `abs(IC)*abs(ICIR)` 淘汰低质量者。
- [x] 成本门槛：固定 3bp，严格 `edge > 3bp`；不是双边 6bp，也不是 `max(6bp, cost)`。
- [x] no-trade 语义：研究/proposed 持久化，executed target/真实持仓不变；账户自愈只修复损坏。
- [x] AC 动态 ensemble：active 最多 10、可少于 10，质量 tilt；固定 FW 不能替代正式动态策略。

### 10.5 迁移风险清单

- [ ] 以 `WL-data-final` 为唯一最终输入，并保留其 README、provenance、manifest 缺口记录；不修改历史数据快照。
- [ ] 复制 AC 两版本 warmup、library、signals、checkpoint、WL state、account、workflow 和 logs 后再生成新 contract/code fingerprint。
- [ ] 对 `formula=null` 或无 signal hash 的旧因子执行恢复/验证；无法恢复的进入 quarantine，不伪造 rho=0。
- [ ] 旧 `.04/.10`、6bp、双边 L1、首次全现金 forward state 只保留为历史结果，不能与新合同结果合并。
- [ ] Luna 3WL 保持暂停；先完成无 API 单测、AC/FM deterministic smoke、1WL×2 block，才创建/恢复新 online run。

## 11. 历史记录与仍需验证的事项

### 11.1 因子门槛的历史冲突

`agent-framework/progress.md` 的早期记录保留 `.04/.10`。本项已决策：正式 benchmark 只承认 `.007/.084`，旧门槛必须在历史 manifest 中保留，不能被改写为新合同结果。

### 11.2 AC 原生 fixed FW 与动态 factor ensemble

`template_a` 的 `FW=(.17,.15,...,.05)` 是固定的 10 项策略模板。此项已决策：正式 benchmark 采用动态 active ensemble、质量 tilt 和滚动淘汰；fixed FW 只能作为历史 artifact，不能绕过正式 proposal/gate。

### 11.3 DeepSeek 与 Luna 的“原生 Responses”定义

sub2api 可以向 AC 暴露 `/v1/responses`，但如果上游实际仍是 Chat Completions，仍需验证桥接是否完整保留：

- tool call 的 `call_id`、arguments、output item；
- 多轮 Responses history；
- reasoning/effort 字段；
- SSE/non-streaming 一致性；
- 错误、429、pool exhausted 的重试语义。

“端点返回 200”不等于“与 Luna 原生行为等价”。

### 11.4 最终数据包的可复现性

WL1–8 可在浮点误差内重建，WL9 当前不能完整由 markdown 解析器重建；high/low 还有 fallback 缺口。是否先修生成器并重新生成最终数据，还是把 `WL-data-final` 固化为不可变输入快照，需要用户决定。

---

## 12. 当前用户关心的“错误为什么又冒出来”对应关系

以下不是本轮对当前代码的最终判定，而是把历史上已经出现过的错误按机制归类：

1. **状态闸门错误**：验收器漏 phase、resume 超过 max cycle 仍创建新 cycle；结果看起来像重复运行。
2. **持久化一致性错误**：因子文本通过但 JSON 没保存/没读回；或 trim 只改 export 未改 checkpoint；结果看起来像“挖到了但库为空”或“旧因子复活”。
3. **provider 生命周期错误**：修改 `.env` 但旧进程没有重启；结果看起来像“已经切换 API，实际仍打旧端点”。
4. **状态观测错误**：BacktestTool 临时回卷 live date；结果看起来像日期倒退/进度丢失。
5. **交易执行错误**：旧式整数/整手订单和上一收盘价订单部分成交；结果出现非 15 资产、现金残留或首次建仓不满仓。
6. **协议兼容性错误**：Chat 转 Responses 只转换外壳，没有完整保留工具历史/错误语义；结果表现为 API 200 但 Agent cycle 失败。
7. **数据路径错误**：旧 generator 默认路径与 `WL-data-final` 不一致；结果可能悄悄读取旧数据或无法复现 WL9。
8. **证据等级错误**：把一次 probe、短 smoke、历史归档或日志文本当成完整 9-WL 成功；结果产生“文档说完成、实际一运行又失败”的错觉。

---

## 13. 执行中的统一 current-state comparison

本节是后续工作入口，本轮不执行、不替用户裁决。

### 13.1 代码与合同对照

- [ ] 从 `ASSETS.yaml`、`runAC.md`、`runFM.md` 读取单一合同，确认门槛、library cap、active top-k、15 assets、cash=0、3 bps。
- [ ] 对 AC 两个版本逐项审计 `factor_contract.py`、Screener ensemble、Trader strategy、StepTool、rebalance helper。
- [ ] 对 FM audit `RalphLoop`、`combine`、`run_forward`、trim、checkpoint/library/signals 和 cost dead-zone。
- [ ] 确认 fixed FW 是否被真正替换/融合为动态 factor ensemble。

### 13.2 数据与 fingerprint 对照

- [ ] 确认所有 scheduler 使用 `WL-data-final`，不读旧 `online-worldline`。
- [ ] 计算 9 panel/news 的逐文件 SHA，并补充或记录 WL9 2030 anchor。
- [ ] 审计 high/low 空值、周末 stage anchor 和 generator fallback。
- [ ] 对 warmup fingerprint、代码 fingerprint、配置 fingerprint 做逐层差异报告。

### 13.3 运行状态与恢复对照

- [ ] 在不恢复 Luna 的前提下，读取暂停时三条 WL 的 date/account/snapshot/transaction/workflow。
- [ ] 审计 DeepSeek warmup 是否停在原 block、失败是否在原地重试、是否有因子 JSON/公式/指标可加载。
- [ ] 检查当前运行树、relay、sub2api 的实际启动环境和日志，不以历史 PID 或旧报告代替。
- [ ] 选择“继续已有 warmup/checkpoint”还是“按新 fingerprint 建新 stage”，不得混用。

### 13.4 结果完整性对照

- [ ] 每个 WL 每个 online block：15 positions、cash=0、weight sum=1、fractional quantity、rebalance history、3 bps。
- [ ] 每个 active ensemble：ID 在库内、最多 10、保存 sign/quality/tilt。
- [ ] 每次 trim：export/checkpoint/signals ID 一致，resume 不复活。
- [ ] 大阶段结束后：git commit、远端 push、data 盘快照、最近 5 次保留策略和 SHA。

---

## 14. 证据索引

### 工作区文档

- `runAC.md`：最终 AC 数据输入审计、15 资产合同、warmup/online 生命周期、fractional rebalance、relay、DeepSeek 路由和恢复判定。
- `runFM.md`：FM mine/combine/forward、200/110/40 参数、trim、无限重试、P0/P1、归档证据。
- `agent-framework/progress.md`：7月25日门槛 smoke、phase/resume bug、AC warmup 验收、早期 `.04/.10` 口径。
- `plan.md` / `process.md`：20 资产、USD 计价、世界线生成、price-leads-news、前向禁止调参。
- `WL-data-final/README.md`：最终快照内容、共享 warmup 指纹、WL9 扩展版说明。
- `WL-data-final/docs/GENERATION_PROVENANCE.md`：生成参数、来源、缺失的 generation_meta、WL9 与逐文件 manifest 缺口。
- `agent-framework/AlphaCrafter/README.md`、`alphacrafter/run_config.yaml`：原生 AC 三角色骨架、CLI、模型和 benchmark config。
- `agent-framework/FactorMiner/README.md`、`docs/architecture.md`、`docs/metrics.md`、`docs/repo-audit.md`：原生 FM loop、memory、DSL、runtime recomputation 和 metric semantics。
- `research/answer-research1.md`：机制 C 的代码/文献核验和 conditional mining 边界。

### Codex 记录

- `/home/lxx/.codex/memories/rollout_summaries/2026-07-26T07-41-40-fQcX-fm_ac_runtime_and_complete_nine_worldline_archive.md`：FM 长跑、retry、trim、P0/P1、AC 并发、FM 9-WL 归档。
- `/home/lxx/.codex/memories/rollout_summaries/2026-08-04T03-25-42-fMhK-mechanism_c_dynamic_factor_mining_research.md`：机制 C、FM/AC 架构差异、long-only 限制。
- `/home/lxx/.codex/memories/rollout_summaries/2026-08-06T07-48-28-T0XX-ac_runtime_backup_relay_params_and_date_diagnosis.md`：AC 暂停、备份、relay、参数和 BacktestTool 日期诊断。
- 对应原始 session：`/home/lxx/.codex/sessions/2026/07/26/`、`/home/lxx/.codex/sessions/2026/08/04/`、`/home/lxx/.codex/sessions/2026/08/06/`。

### Claude Code 会话证据

- `/home/lxx/.claude/projects/-home-lxx/1302022e-3413-474f-8595-b8d09b5f447e.jsonl`：8月6日暂停 AC 的用户指令。
- `/home/lxx/.claude/projects/-home-lxx/066bea6e-ce17-4076-b2c2-a53a70d3a3fc.jsonl`：8月6日 sub2api-codex 部署、Codex/Luna relay 上下文。
- `/home/lxx/.claude/projects/-home-lxx/f091b4e4-b17d-4c6d-b8aa-3235adaaa44b.jsonl`：8月6日 DeepSeek 免费网关设计；8月8日以后的 WARP 修正不纳入本期规划事实。
- `/home/lxx/.claude/projects/-home-lxx/3ccbb546-818f-431c-a3b4-9b034d85038c.jsonl`：7月25日打开 runFM 的上下文；未把无关 SSH 工作误计入 AC/FM 决策。

---

## 15. 本轮交付确认

- [x] 已建立本文件并覆盖 AC、FM、数据、provider、运维、恢复和研究方向。
- [x] 已按日期重建 7/26–8/6 的关键规划演进。
- [x] 已区分原生框架能力、benchmark 改动、历史结果和当前待验证项。
- [x] 已记录已知错误、原因、修正和容易重复运行/重复挖掘的入口。
- [x] 已记录门槛冲突、WL9 复现缺口、fixed FW/动态 tilt、Responses bridge 等未决事项。
- [x] 已保留后续统一 current-state comparison 的检查入口。
- [ ] 用户确认最终门槛、AC dynamic ensemble 合成方式、DeepSeek Responses 等价标准和 WL-data-final 固化策略。
- [ ] 用户确认后执行第 11 节的统一代码/数据/运行/结果审计。
