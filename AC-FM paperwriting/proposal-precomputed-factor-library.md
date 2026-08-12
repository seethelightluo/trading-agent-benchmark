# 开题报告：面向虚构世界线基准的交易 Agent 优化——基于预计算大规模因子库的在线轻量选择

> 撰写日期：2026-08-12
> 依托工作区：`/home/lxx/trade-agent-benchmark`（AlphaCrafter + FactorMiner 九条世界线前向走步基准）与 `/home/lxx/trade-agent-crisis`（量化交易学习仓库，含因子/世界线新颖性调研）
> 依据文档：`plan.md`、`RUN.md`、`runAC.md`、`runFM.md`、`checklist.md`、`agent-framework/progress.md`、`agent-framework/plan.md`、`research/research1.md`、`research/answer-research1.md`、`WL-data-final/research/*`、`llm-params/key-params.md`、`AC-FM paperwriting/WL-para.md`、crisis 仓库 `README.md` 与 `research_notes/*`

---

## 摘要

LLM 交易 Agent（AlphaCrafter、FactorMiner 等）在"每 10 个交易日在线挖因子"的范式下存在三重瓶颈：**LLM 调用量巨大**（AC 全量约 6–10 万次调用、4.5–7 亿 token）、**因子在新型 regime 中快速失效**（虚构世界线正是为此设计）、**在线研究难以审计与复现**。本课题提出以"**离线预计算大规模因子库 + 在线轻量选择与条件激活**"为核心优化方向：在共享 warmup 阶段用确定性 DSL 与并行计算一次性预计算数千个候选因子，冻结入库；在线阶段每个决策点仅做信号刷新、尾窗准入评估、top-10 选择、IC 加权组合与 3bp 成本门控，把每个决策点的 LLM 调用从 O(10^1) 降到 O(0–1)。并以九条虚构黑天鹅世界线（2026-07-16→2035-12-31）为评测环境，与 AC/FM 在线挖矿基线做成本—收益对照。

---

## 一、研究背景与意义

### 1.1 LLM 因子挖掘 Agent 的兴起与瓶颈

以 FactorMiner（arXiv:2602.14670）、AlphaAgent（arXiv:2502.16789）、Alpha-GPT（arXiv:2308.00016）、R&D-Agent-Quant（arXiv:2505.15155）为代表的 LLM 因子挖掘框架，已证明"LLM 生成公式 → 统计准入 → 经验记忆迭代"可以产出有效 alpha。但现有范式默认**在线持续挖矿**：每个决策点由 LLM 生成候选、回测、筛选。其代价随决策点数量线性放大。

本工作区的实测数据（`plan.md §8.8`）给出了直接证据：

| 框架 | 每决策点 LLM 调用 | token | 全量（9 WL × ~246 决策点） |
|---|---|---|---|
| AC（Responses，5 相位全 LLM） | ~27（warmup 实测）~45（前向） | ~180k–300k | **~60k–100k 调用 / ~4.5–7 亿 token** |
| FM（Chat Completions，仅 mine 调用 LLM） | ~1 | ~5k | ~2.3k 调用 / ~11M token |

结论：**在线挖矿是成本与延迟的主要来源，也是配额等待（5h 刷新退避）与运行时长的主因。** 若把"挖"与"选"解耦——因子离线预计算、在线只选择——成本可下降 1–2 个数量级。

### 1.2 回测不可信与虚构世界线评测

Profit Mirage（arXiv:2510.07920）证明 LLM 金融 Agent 存在信息泄露/历史记忆问题，历史回测收益虚高；KTD-Fin（arXiv:2605.28359）通过记忆受控重新评测 LLM 交易 Agent。本工作区则采用更激进的解法：**用人工构造、逻辑自洽的虚构未来世界线作为评测环境**（2026-07-16 后为合成行情，价格-leads-news、DXY-β 派生汇率、GBB 噪声），使 Agent 不可能见过"未来"。九条世界线全部为并列前向评估环境，不分训练/验证/测试线；2026-07-15 前冻结人工配置。该评测设计的动机与工具均有先例，但"人工叙事世界线作为交易 Agent 评测环境"这一组合在公开文献中未见先例（见 crisis 仓库 `worldline_validation_novelty.md`），是本工作区自身的方法论贡献。

### 1.3 本课题的意义

1. **工程意义**：把交易 Agent 的研究成本从"每决策点 LLM 挖矿"压缩到"离线一次性 + 在线轻量选择"，使全量 9 WL 长跑从"数周挂机 + 配额等待"变为"可预算、可并行、可中断续跑"。
2. **方法论意义**：预计算库天然形成"影子因子池 + 条件激活"结构——这是工业界（WorldQuant 影子 alpha、对冲基金 playbook）已有、但公开文献缺乏的形态（见 crisis 仓库 `novelty_analysis.md` 空白点分析）。
3. **评测意义**：在九条黑天鹅世界线上，可以量化回答"预计算因子在 novel regime 中的衰减曲线""条件激活相对连续择时的增益"等此前难以严格测评的问题。

---

## 二、国内外研究现状

### 2.1 LLM 因子挖掘

- **FactorMiner**（arXiv:2602.14670）：自进化 Agent，`Retrieve→Generate→Evaluate→LibraryUpdate→Distill` 分层，P_success/P_fail 经验记忆、skills、multi-target research score、regime-aware memory。本工作区已按其论文重实现（Ralph/Helix 双 lane）。
- **AlphaAgent**（arXiv:2502.16789）：针对 LLM 因子同质化与 alpha decay，引入正则化探索。
- **Alpha-GPT / Alpha-GPT 2.0**（arXiv:2308.00016）：human-in-the-loop 因子挖掘，Man Group 合作。
- **R&D-Agent-Quant**（arXiv:2505.15155）：数据驱动因子 + 模型联合优化多 Agent 框架。
- 共同点：**产出物是公式 + 统计元数据，在线生成、逐个评估**；缺少"大规模离线预计算 + 条件激活"的形态。

### 2.2 因子库 / 影子因子 / 条件激活

- WorldQuant BRAIN 等平台有完整因子生命周期（提交→审核→上线→退役），但**激活是一次性决策**，没有"条件未满足时休眠、条件满足时按档位升格"的状态机。
- 对冲基金的事件 playbook（Millennium/Citadel 对选举、央行会议、地缘冲突预置情景组合）在工业界证明"为未发生事件预置交易方案"真实存在，但停留在**人工文档 + 人工执行**。
- crisis 仓库 `novelty_analysis.md` 的空白点结论：**"离散前置条件状态机驱动的因子激活"与"LLM 一次性产出因子+条件+档位+退出规则的完整 bundle"在公开文献中未见先例**——这是本课题可占据的空白。

### 2.3 交易 Agent 评测方法

- 回测盲化：Profit Mirage（arXiv:2510.07920）、KTD-Fin（arXiv:2605.28359）。
- 模拟器评测：Wah & Wellman（arXiv:1906.12010）历史回放 vs 交互式模拟；EvoMarket、KineticSim 等 2025–2026 模拟器生态。
- 情景压力测试：CCAR/DFAST 等监管压力测试、BarraOne/Axioma 的 what-if 分析（静态冲击，无自适应 Agent）。
- 本工作区的九条世界线属于"人工叙事驱动的虚构未来 + 自适应 Agent 前向走步"的组合，工具层有先例、组合本身为空白。

### 2.4 现状小结

**空白一**：无"离线大规模预计算因子库 + 在线轻量选择"的系统形态与公开评测。
**空白二**：无"离散条件因子激活状态机"（连续 regime 择时 ≠ 离散条件激活）。
**空白三**：无在虚构世界线上对因子衰减与条件激活的严格定量评测。

本课题以空白一为主线，空白二、三为支撑。

---

## 三、现有 Benchmark 基础与问题分析

### 3.1 Benchmark 合同（权威口径，`runAC.md`/`checklist.md`）

- 数据：`WL-data-final/`，9 条世界线面板各 83,347 行、20 列资产；真实历史严格截至 **2026-07-15**，**2026-07-16** 为 1,000,000 USD-equivalent 全现金账户的首个前向执行日，在线终点 2035-12-31。
- 宇宙：15 可交易（000300.SH、SPX、HSI、N225、SX5E、000688.SH、SOX、NDX、XAU、COPPER、WTI、BTC、ETH、US10Y、CN10Y）+ 5 只读信号（DXY、USDCNY、USDJPY、EURUSD、VIX）。
- 决策：每 10 个交易日一个 cycle；约 246 决策点/WL。
- 因子准入：`abs(IC)≥0.007`、`abs(ICIR)≥0.084`、库内最大 `abs(Spearman rho)<0.5`（FM live 文档另记 0.04/0.10；单一事实源为 `ASSETS.yaml`）；研究库容量 30，活跃组合 ≤10。
- 组合：long-only、权重非负和为 1、允许小数份额、online cash=0；首次建仓免费，后续单边迁移按 3bps 收；`gross_edge_bps > one_way_turnover×3` 才执行，否则 no-trade 并持久化 proposal。
- 防穿越：Agent 只能看到当前游标前的数据；9 条 WL 在线状态彼此独立；禁止按 WL 成绩人工改参重跑。

### 3.2 现有 Agent 运行结构

- **AC**：每 cycle 3 个 Miner（并行）+ Screener + Trader，共 5 个 LLM 相位；Miner 在线写因子脚本、回测、入库；Screener 选 top-10 并出 ensemble；Trader 定权重并通过 `step` 推进 10 个交易日。全部走 Responses API（`gpt-5.6-terra`，reasoning `standard/medium`）。
- **FM**：RalphLoop 在线迭代（warmup 200 iter、target 110、batch 40；online 每窗追加 1 iter），`combine` 确定性选 top-10 + IC 加权。
- 两者共享 warmup 数据，但因子库/记忆/账户按 WL 独立克隆。

### 3.3 问题提炼

1. **成本失控**：AC 全量 ~60k–100k 调用、~5 亿 token；每个决策点 27–45 次调用中大部分是"重复研究"，跨决策点信息复用差。
2. **在线研究质量不稳定**：LLM 挖矿依赖提示词、记忆与随机性，同一数据下候选质量方差大、难审计（candidate parse rate、thesis alignment 等缺乏契约）。
3. **因子在 novel regime 中的衰减无法被"挖得更勤"解决**：世界线的价值在于出现历史未见的 regime；此时在线挖矿仍受限于"用过去数据验证未来 regime"的归纳盲区，反而是**预置的、覆盖多种条件的因子库 + 可观察状态激活**更有优势。
4. **无因子生命周期契约**：现有库只有"准入/淘汰"，没有"休眠/激活/退出"状态，条件因子无法表达（crisis 仓库空白点二）。

---

## 四、研究目标与研究内容

### 4.1 总体目标

构建并验证"**离线预计算大规模因子库 + 在线轻量选择与条件激活**"的交易 Agent 优化框架（下称 **P-Factor Agent**，Precomputed-Factor Agent），在九条虚构世界线上与 AC/FM 在线挖矿基线对比，在**不劣化风险调整收益的前提下，把每决策点 LLM 调用降到 O(0–1)**，并给出因子衰减与条件激活的定量证据。

### 4.2 研究内容一：离线大规模因子预计算（因子库构建）

- 复用 FactorMiner 的类型化 DSL（`$open/$high/$low/$close/$volume/$vwap/$returns` 算子集 + SignedPower/Med/Rsqua re/Slope/Resi/TsDecay/Scale 等论文算子），扩展条件标签字段。
- 用**确定性生成器（组合枚举 + 随机种子）**在共享 warmup 数据（2020-01-01~2026-07-15）上预计算数千–上万候选因子；全量并行（ProcessPool，已有 14.4s/35000 行预处理与确定性候选评估基础设施）。
- 冻结候选库为不可变 artifact：公式、AST、IC/ICIR/换手/覆盖率/相关性、condition tag、provenance、指纹。**每个 WL 复用同一候选库，但只在各自可见数据上计算 signal**。

### 4.3 研究内容二：因子库治理与条件标签

- 在现有准入（0.007/0.084/rho<0.5）与 capacity 30 之上，增加：候选库→影子库→活跃库三级结构；影子因子带 condition（可观察状态表达式：VIX 阈值、DXY 方向、利率水平、相关性 regime、新闻事件标签）。
- 确定性淘汰与 quarantine（对齐 `factor_contract.py` 现有语义），保证 resume 不复活淘汰因子。

### 4.4 研究内容三：在线轻量选择与条件激活

- 每个决策点（10 交易日）：
  1. 用当前可见数据刷新全部候选 signal（确定性、免费）；
  2. 尾窗（如 120 日）重算准入指标，筛选活跃候选；
  3. 条件状态机：满足 condition 的休眠因子按档位激活（0%→2%→5% 示例档位），违反退出条件则降档/退役；
  4. 按 `abs(IC)×abs(ICIR)` 质量 + 方向 sign 选 top-10，IC 加权合成 ensemble；
  5. 组合层用现有 proposal/gate（3bp 迁移成本、gross_edge 门槛）出 target weights。
- LLM 角色收窄为"**可选**的研究总监 Agent"：仅当统计选择器产生冲突或零活跃因子时，用 1 次调用生成新候选（O(0–1) 调用）；不改变 evaluator 与组合合同（对齐 `research1.md` 机制 C 的 mandate 冻结原则）。

### 4.5 研究内容四：评测与成本—收益分析

- 与 AC、FM 基线在 9 WL 上跑完整前向，对比：NAV 曲线、Sharpe、maxDD、Calmar、换手/摩擦成本、LLM 调用数与 token、运行时长、因子存活率。
- 消融：库规模（1k/5k/10k）、选择策略（纯 IC 加权 vs 条件激活）、LLM 总监开/关、成本 gate 有/无。
- 特别分析：**warmup 预计算因子在每条 WL 的 IC 衰减曲线**，验证"预计算库 + 条件激活"相对"在线挖矿"在 regime 切换时的相对优势。

---

## 五、关键科学问题与创新点

**Q1（成本—收益）**：离线预计算因子库相对在线挖矿，在九条世界线上的风险调整收益差异与 LLM 成本下降幅度各是多少？——假设：收益差异在统计上不显著劣于基线，而成本下降 1–2 个数量级。

**Q2（条件激活）**：离散条件因子激活状态机相对连续 regime 择时/无条件库，能否在 regime 切换（世界线阶段边界、news 事件）时改善组合表现？——假设：在 9 条世界线中至少 5 条产生正的边际 IC/组合增益。

**Q3（衰减）**：预计算因子在虚构 novel regime 中的 IC 衰减速率与可观察状态（VIX/DXY/利率/相关性）的预测关系？——假设：状态条件化能显著解释并部分对冲衰减。

创新点：

1. **离线/在线解耦的系统形态**：大规模确定性预计算 + 在线零/低 LLM 选择，公开文献未见先例（空白一）。
2. **条件因子激活状态机**：因子携带"条件+档位+退出"元信息，生命周期含休眠态（空白二）。
3. **虚构世界线上的因子衰减与条件激活定量评测**（空白三）。
4. **以 LLM 调用成本为显式设计目标**的 Agent 架构评估（成本进入 loss/评测维度）。

---

## 六、技术路线

```
WL-data-final 面板（9 WL，2020-2035）
        │ 共享 warmup 段（≤2026-07-15）
        ▼
┌─────────────────────────────── 离线阶段（一次性，可并行） ───────────────────────────────┐
│  ① DSL 算子库 + 确定性候选生成器 ──► ② 全量并行预计算（数千–上万候选）                       │
│  ③ 统一评估（IC/ICIR/换手/覆盖/rho/condition tag）──► ④ 不可变候选库 artifact（含指纹）      │
└────────────────────────────────────────────────────────────────────────────────────────────┘
        │ 每 WL 独立克隆 + 只读可见数据
        ▼
┌─────────────────────────────── 在线阶段（每 10 交易日，确定性为主） ───────────────────────┐
│  ⑤ 信号刷新（当前可见窗口）─► ⑥ 尾窗准入重评 ─► ⑦ 条件状态机（休眠/激活/退出）              │
│  ⑧ 质量排序选 top-10 + IC 加权 ─► ⑨ proposal/gate（3bp 门槛）─► ⑩ step 推进 10 日           │
│  （可选）⑪ 研究总监 LLM：仅零活跃/冲突时生成新候选（O(0–1) 调用/决策点）                     │
└────────────────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
   9 WL NAV/风险指标 + LLM 成本审计 + 因子存活/衰减曲线（对照 AC / FM）
```

阶段划分：

1. **W1 因子库构建**：扩展 DSL 与 condition schema；生成器 + 并行评估；库容量与多样性实验。
2. **W2 在线选择器**：信号刷新、尾窗准入、top-10 + IC 加权、组合 gate（全部复用现有 `factor_contract.py`/`portfolio_contract.py`/`rebalance_to_weights.py`）。
3. **W3 条件激活**：条件表达式求值器、档位状态机、退出规则；在 1–2 条 WL 冒烟。
4. **W4 全量对照**：P-Factor vs AC vs FM 在 9 WL 全量前向（无人工调参；结果持久化到 `report-and-output/`）。
5. **W5 分析与论文**：衰减曲线、消融、成本—收益、与基线统计检验。

---

## 七、实验设计

### 7.1 评测环境与基线

- 环境：九条世界线完整前向（2026-07-16→2035-12-31），共享 warmup 一次，合同参数与现有 benchmark 完全一致（1M、15+5、3bps、10 日 cadence、活跃≤10、库≤30）。
- 基线 A：AC（在线挖矿，Responses API，standard/medium）。
- 基线 B：FM（在线 Ralph 迭代，每窗 1 iter）。
- 处理：P-Factor Agent（离线库 + 在线选择，LLM 总监关闭/开启两档）。

### 7.2 指标

- 组合：NAV 曲线、总收益、年化、Sharpe、maxDD、Calmar、平均仓位、换手与摩擦成本。
- 因子：在线 IC/ICIR、因子存活率（半衰期）、条件激活前后 IC 变化、库内 rho 分布。
- 成本：LLM 调用数、token、API 失败/重试、墙钟时长、决策点延迟分布。
- 统计：per-WL 配对比较（NAV/Sharpe），Bootstrap/符号检验，避免小样本结论过强。

### 7.3 消融矩阵

| 维度 | 档位 |
|---|---|
| 库规模 | 1k / 5k / 10k 候选 |
| 在线选择 | 纯 IC 加权 / 条件激活状态机 |
| LLM 总监 | 关 / 开（仅零活跃或冲突时 1 次调用） |
| 成本 gate | 开（3bp）/ 关 |
| 尾窗 | 60 / 120 / 250 日 |

### 7.4 防穿越与合同红线（必须满足）

- 预计算只用 ≤2026-07-15 数据；任何 WL 在线数据不得进入候选生成或参数选择。
- 候选库、条件表达式、选择策略在 2026-07-16 前冻结；9 条 WL 之间不共享在线因子经验。
- 不依据 WL 结果调参重跑；每次对照使用独立随机种子审计。
- 沿用现有指纹与 checkpoint 机制，断点续跑不复活已淘汰因子。

### 7.5 风险与应对

| 风险 | 应对 |
|---|---|
| warmup 预计算因子在合成世界线中整体失效 | 统计失效前已按合同做库内多样性上限；条件激活保留在线补充通道 |
| 条件表达式过拟合 warmup | 条件表达式只允许可观察状态量（VIX/DXY/利率/相关性/news 标签），且数量/复杂度设限 |
| 小样本（15 资产、9 WL）统计力不足 | 以 per-WL 配对 + 非参检验为主，明确报告不确定性下界 |
| 多假设检验 | 库规模/选择策略为消融而非"挑最优"，冻结协议并记录全部尝试 |
| LLM 总监引入点时泄漏 | 总监只接受截止当前游标的可见摘要；输出仅限新候选公式，不改变 evaluator |

---

## 八、预期成果

1. **系统**：P-Factor Agent 实现（复用现有 `agent-framework` 基础设施），离线候选库生成器 + 在线选择器 + 条件激活状态机，全部带指纹与 checkpoint。
2. **数据**：9 WL 完整前向结果（NAV/因子/成本审计），AC/FM/P-Factor 三基线对照；因子衰减与条件激活实证曲线。
3. **论文/报告**：开题→中期→结题三阶段文档；目标投稿方向为 LLM×量化交叉（因子挖掘、交易 Agent 评测），核心卖点：离线/在线解耦、条件激活状态机、虚构世界线评测。
4. **开源**：候选库 artifact schema、评测脚本与复现参数（对齐 `WL-para.md` 的可复现原则）。

---

## 九、进度安排（建议 12 周）

| 周次 | 里程碑 |
|---|---|
| W1–W2 | 完成候选库生成器与 condition schema；预计算 5k 候选并评估 |
| W3–W4 | 在线选择器与组合 gate 联调；1 条 WL 冒烟通过 |
| W5–W6 | 条件激活状态机；2 条 WL 对照预跑 |
| W7–W8 | 9 WL 全量对照第一轮（AC/FM/P-Factor） |
| W9–W10 | 消融矩阵 + 因子衰减分析 |
| W11–W12 | 统计检验、文档与论文初稿；复现包收尾 |

---

## 十、参考文献

> 以下文献出处来自本工作区研究笔记；`worldline_validation_novelty.md` 已核验的引用以 ✅ 标注，未核验/疑似虚构的引用未列入。投稿前需逐条重新核验。

[1] Wang et al., FactorMiner: A Self-Evolving Agent with Skills and Experience Memory for Financial Alpha Discovery. arXiv:2602.14670. ✅
[2] Tang et al., AlphaAgent: LLM-Driven Alpha Mining with Regularized Exploration to Counteract Alpha Decay. arXiv:2502.16789.
[3] Wang et al., Alpha-GPT: Human-AI Interactive Alpha Mining for Quantitative Investment. arXiv:2308.00016 (v2 2025).
[4] Li et al., R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization. arXiv:2505.15155.
[5] Profit Mirage: Revisiting Information Leakage in LLM-based Financial Agents. arXiv:2510.07920. ✅
[6] From Knowing to Doing: A Memory-Controlled Benchmark for LLM Trading Agents on Stock Markets (KTD-Fin). arXiv:2605.28359. ✅
[7] FinMem: A Performance-Enhanced LLM Trading Agent. arXiv:2311.13743. ✅
[8] FinCon: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement. arXiv:2407.06567. ✅
[9] TradingAgents: Multi-Agents LLM Financial Trading Framework. arXiv:2412.20138. ✅
[10] Wah, Wellman. How to Evaluate Trading Strategies: Single Agent Market Replay or Multiple Agent Interactive Simulation? arXiv:1906.12010. ✅
[11] 工作区内部文档：`plan.md`、`RUN.md`、`runAC.md`、`runFM.md`、`checklist.md`、`progress.md`、`research1.md`、`answer-research1.md`、`worldline_validation_novelty.md`、`novelty_analysis.md`、`WL-para.md`。

---

*（完）本开题报告基于现有工作区文档撰写；引用文献与 arXiv 编号以仓库笔记为来源，正式提交前需按 `worldline_validation_novelty.md` 的方法逐条核验。*
