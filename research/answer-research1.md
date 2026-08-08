# 对 `research1.md` 的核验：机制 C（动态改变 Miner 挖掘目标）与“人工洞察主导的可迭代因子 Agent”可行性

> 核验日期：2026-08-04  
> 核验对象：当前工作区 `/home/lxx/trade-agent-benchmark`，重点为 `agent-framework/FactorMiner`，并交叉查看当前 AlphaCrafter 的持久化契约。  
> 文献范围：以 arXiv 原文和当前仓库代码为准；“喘吁吁”按上下文理解为“查一查/调研一下别人是否做过”。

## 0. 结论先行

**机制 C 可实现，而且当前 FactorMiner 已有约 60% 的结构基础；但 `research1.md` 把“Prompt 改方向”和“真正改变可验证的挖掘目标”混在了一起，并且对 FactorMiner 的记忆存储、检索方式、AlphaCrafter 的持久化现状有多处错误或过度概括。**

最重要的判断如下：

1. **只在每轮 Prompt 中加入人工洞察，工程上很容易，但不等于改变挖掘目标。** 当前 `RalphLoop` 已有 `Retrieve -> Generate -> Evaluate -> LibraryUpdate -> Distill` 的分层，`PromptContextBuilder` 也允许附加额外上下文。因此加入“本轮研究方向”很直接；但当前评价门仍主要围绕 IC/ICIR、跨目标分数和与已有库的相关性，若评价函数不改，LLM 最终仍会被同一个统计门槛选择。
2. **真正的 Conditional Mining 必须同时改变四件事：生成条件、评价目标、记忆作用域、因子库/溯源。** 只改 Prompt 会产生“口头上挖半导体空头，实际上仍按全市场绝对 IC 入库”的目标错位。
3. **“人工洞察主导、不同轮次挖不同方向”的 Agent 并不空白。** Alpha-GPT 已明确支持人类在 review 阶段修改后续轮次搜索；AlphaAgent 用用户指定研究方向/市场洞察初始化假设，并由回测反馈迭代；R&D-Agent-Quant 动态生成目标对齐 Prompt，并用多臂老虎机选择下一优化方向；2026 年的 QuantaAlpha 和 *From Hypotheses to Factors* 又进一步做了多方向规划、轨迹演化和固定协议下的顺序假设搜索。因此，若做新 Agent，创新点不能只写成“人给洞察，Agent 下一轮换方向”。
4. **当前仓库最有价值的创新空间** 是：把人工洞察编译成一个可审计、可证伪、带有效期和数据边界的 `ResearchMandate`，再让 FactorMiner 在轮次边界动态分配研究方向，同时保持每个轮次内部的评价协议冻结；并将记忆改为 `global + objective-scoped + regime-scoped`，避免把“与当前观点不一致”错误地写成全局 `P_fail`。
5. **半导体泡沫的例子在当前 benchmark 数据契约下不能直接成立。** 当前 FM DSL 主要是 OHLCV/amount/vwap/returns，当前交易落地又是 15 资产、全额投资、long-only。要挖“半导体板块内空头因子”至少需要足够多的半导体股票、点时行业标签、点时基本面/新闻快照和允许表达空头或低配的执行契约。当前 15 资产世界线不满足这个评估条件。
6. **建议先做“人工 mandate + 固定数据”的最小版本，不要第一版就让新闻 Agent 自由改 fitness。** 先证明不同 mandate 能产生统计上和语义上不同的候选分布、且不会污染全局记忆；再接入带时间戳和来源的新闻/基本面 Agent。

综合评估：

| 层级 | 含义 | 当前可实现性 | 预计改造量 | 结论 |
|---|---|---:|---:|---|
| C0 | Prompt 注入人工方向 | 很高 | 1–3 天 | 可做，但只是 steering |
| C1 | 不同方向动态分配候选预算/专家 | 高 | 3–7 天 | 当前 debate/specialist 架构可复用 |
| C2 | 方向对应不同 universe、target、约束和 fitness | 中高 | 2–4 周 | 这是“真正机制 C” |
| C3 | 新闻/基本面 Agent 自动生成、更新、撤销 mandate | 中 | 4–8+ 周 | 难点是点时数据、泄漏、错误洞察和审计，不是 LLM 调用 |

---

## 1. `research1.md` 的逐项真伪核验

### 1.1 `P_succ / P_fail` 是 AST/Expression Tree 存储吗？

**结论：对当前 FactorMiner 是错误的；把“因子表达形式”和“经验记忆形式”混淆了。**

当前实现中：

- 因子公式以 DSL 字符串保存，例如 `Neg(CsRank(Delta($close, 5)))`；运行评价时才由 parser 解析成表达式树并执行。
- `Factor` 持久化字段是 `name/formula/category/IC/ICIR/win_rate/max_correlation/research_metrics/provenance` 等，`signals` 明确不写入 JSON：`agent-framework/FactorMiner/factorminer/core/factor_library.py:28-101`。
- `P_succ` 是 `SuccessPattern`，字段是自然语言 `name/description/template/success_rate/example_factors/...`；`P_fail` 是 `ForbiddenDirection`，字段是 `name/description/correlated_factors/typical_correlation/reason/...`：`factorminer/memory/memory_store.py:43-102`。
- 完整经验记忆 `ExperienceMemory` 将 `state/success_patterns/forbidden_directions/insights/version` 序列化为 JSON：`factorminer/memory/memory_store.py:129-166`。

FactorMiner 论文原文也说，实践中经验被保存为**紧凑的自然语言模板和规范示例**，而不是把全部候选 AST 作为 `P_succ/P_fail` 的主存储。[1]

因此，更准确的表述是：

> FactorMiner 的**可执行因子**是可解析为 AST 的 DSL 公式；FactorMiner 的**经验记忆**主要是由公式结果蒸馏出的自然语言结构模板、禁区和策略洞察。二者不是同一种存储。

### 1.2 每次生成/选择因子要把所有记忆都过一遍吗？

**结论：不需要，这一方向判断是对的；但 `research1.md` 对检索机制的描述不准确。**

当前 paper memory policy 每轮最多取：

- 8 个成功方向；
- 10 个失败方向；
- 10 个最近 insight。

见 `factorminer/architecture/memory_policy.py:102-117` 和 `factorminer/memory/retrieval.py:208-288`。Prompt 还只放最近 10 条 rejection：`factorminer/agent/prompt_builder.py:303-314`。

但是，当前默认检索不是“对当前候选做树编辑距离并从整个库里找 AST 最近邻”。默认逻辑主要按：

- pattern confidence / occurrence；
- domain saturation；
- 最近 rejection 与 forbidden direction 的名称/关键词重叠；
- 最近 insight；
- 可选 family/KG/regime 扩展。

仓库中没有发现 tree-edit-distance 被用于默认 experience-memory retrieval。`agent/critic.py` 有公式字符串编辑距离，AlphaAgent 论文有 AST 相似度，但这不能反推 FactorMiner 默认检索就是树编辑距离。[2]

### 1.3 “FactorMiner 使用树编辑距离 + 秩相关性检索记忆”是真的吗？

**结论：当前仓库证据不支持，论文原文也没有这样描述默认 retrieval。**

- 秩相关性确实用于因子预测评价和库内冗余约束。
- AST 是公式执行和某些结构分析的基础。
- 但 memory retrieval 论文描述为依据当前 library diagnostics 和近期拒绝原因，检索自然语言 recommended/forbidden templates。[1]
- 当前代码也主要是规则化排序、关键词/类别、family/KG/regime 扩展。

这很可能是把 FactorMiner、AlphaAgent、一般 AST 相似度方法混写到了一起。

### 1.4 因子持久化元数据包含 IC、ICIR、相关性、换手、最大回撤和每日截面分数吗？

**结论：部分正确、部分错误。**

当前 `Factor` 核心持久化包括 IC、paper IC、ICIR、win rate、入库时最大相关性、研究指标和 provenance；但：

- `signals` 不序列化；分析时按公式和指定数据重新计算。
- `Factor` 核心对象没有独立的 `max_drawdown` 字段。
- turnover 可存在于 research score 的派生指标中，但不是 `Factor` 顶层固定字段。
- “每天 5000 只股票的截面向量”是因子运行后的可能产物，不是当前库 JSON 的默认持久化内容。

因此基本面/行业 Agent 若要审查当前暴露，需要**在指定时点重新计算 signals，并与点时行业标签、可交易 universe 和组合构造规则结合**，不能只读 `factor_library.json`。

### 1.5 “记忆让高 IC 因子接受率从 20% 提升到 60%”是真的吗？

**结论：数字有论文出处，但 `research1.md` 的措辞过强。**

FactorMiner 的 memory ablation 报告：Have-Memory 产生 96 个 high-quality candidates，yield 为 60%；No-Memory 产生 32 个，yield 为 20%。但该消融为了获得足够样本，使用了较宽松的 `|IC| > 0.02` 和冗余阈值 `0.85`。[1]

所以能说：

> 在论文特定、放宽阈值的 memory ablation 中，高质量候选 yield 从 20% 提升到 60%。

不能直接扩展为：

> 在任意市场、当前 15 资产配置、正式 0.04/0.5 门槛下，最终入库成功率必然从 20% 提升到 60%。

### 1.6 “FactorMiner 是纯数据挖掘记忆，缺少新闻/基本面/人工行业洞察”是真的吗？

**结论：对当前 canonical FM 基本成立，但“纯数据”并不完全准确。**

当前 FactorMiner 并非纯随机数据搜索：

- System prompt 含经济直觉和因子设计规则；
- 有自然语言成功/失败模板；
- 有静态领域 specialist；
- 有基于收益序列检测的 `RegimeAwareMemoryPolicy`，会把 BULL/BEAR/SIDEWAYS 上下文注入 Prompt：`factorminer/architecture/memory_policy.py:210-301`。

但它的 canonical raw features 仍是 `$open/$high/$low/$close/$volume/$amt/$vwap/$returns`，没有当前新闻、公司基本面、行业标签或人类观点的正式数据契约；当前 regime 也是从 returns 检测，不是新闻/宏观语义状态。因此，“缺少外部点时基本面/新闻/人工洞察闭环”成立。

### 1.7 “AlphaCrafter 缺乏持久化持续学习，因子质量不会越来越好”是真的吗？

**结论：对当前 checkout 是过时且过度绝对。**

当前 AlphaCrafter 已有：

- 持久化 `factors/*.json` 因子库；
- IC/ICIR/相关性硬门；
- 容量 30、active top-10 的确定性保留/选择契约：`alphacrafter/utils/factor_contract.py:16-54`；
- Miner 定期复验、失效因子 deprecated 的指令；
- `memory.txt` 持久交易反馈，Miner/Screener 可读取；
- cycle/workflow/checkpoint/resume 持久化。

因此不能说“没有持久化”或“质量不会提高”。更准确的是：

> AlphaCrafter 有持久因子库、交易反馈和复验机制，但缺少 FactorMiner 那种显式、结构化、可版本化的 `P_succ/P_fail/insight` 形成—检索—演化闭环；它的学习质量更多依赖 Agent 是否正确读取和利用文件，而不是统一的确定性 memory policy。

### 1.8 机制 A/B/C 的判断

#### 机制 A：按当前因子暴露审查

**可行，但需要额外数据和确定性计算。** 正确方法不是让 LLM 只读公式猜行业，而是：

1. 在 cutoff 时点重算因子 score；
2. 用点时行业分类、benchmark weight 和风险暴露矩阵做 attribution；
3. 计算 top-quantile concentration、行业 beta、组合边际暴露；
4. Agent 只解释结果或提出约束，暴露数值由代码计算。

但“因基本面冲突就把因子写入全局 `P_fail`”是错误设计。该因子可能只是**当前 mandate 不适用**，不是全局统计失败。应记录为 `mandate_mismatch` 或 `temporarily_downweighted`，不能污染全局 forbidden memory。

#### 机制 B：组合层硬约束

**概念上可行，但 `research1.md` 的做空示例不兼容当前 benchmark 的真实交易契约。** 当前 FM forward portfolio 明确是 15 资产、fully-invested、long-only，目标权重非负且和为 1：`agent-framework/scheduler/fm_walk_forward.py:1-8,157-190`。AlphaCrafter Trader 也被要求 long-only、无现金。

因此在当前系统里，`Sector_Weight(Semiconductor) <= -5%` 不能直接执行。可实现的动作是：

- 相对基准低配但权重仍非负；
- 将受负面观点影响的资产权重降至下限；
- 把释放的权重转向防御资产；
- 或修改整个交易/保证金/风险契约后才允许真正 short。

#### 机制 C：动态改变 Miner 目标

**方向正确，但“改 Prompt 或适应度函数”不是二选一，而是至少二者都要改。** 另外，“行业中性”不等于“各行业分数均匀分布”。更标准的实现是：

- 对 signal 做行业哑变量、beta、size 等风险暴露残差化；或
- 在组合层施加行业相对基准暴露约束；或
- 在 fitness 中惩罚行业暴露的平方范数。

---

## 2. 当前 FactorMiner 为什么适合改造成机制 C

当前代码已有以下可复用边界：

### 2.1 Stage 已解耦

`RalphLoop` 已明确分成 retrieve、generate、evaluate、library update、distill，见：

- `factorminer/architecture/stages.py:11-103`
- `factorminer/core/ralph_loop.py:982-1078`

这意味着可以在不推翻主循环的前提下，给每个 iteration 绑定一个不可变的 objective/mandate。

### 2.2 Prompt 上下文允许扩展

`PromptContextBuilder.build(..., extras=...)` 会把额外结构写入 prompt context：`factorminer/architecture/prompt_context.py:19-40`。当前 `_stage_generate` 只额外注入 `dataset_contract`：`factorminer/core/ralph_loop.py:1037-1052`。

因此加入：

```python
extras={
    "dataset_contract": ...,
    "research_mandate": mandate.to_dict(),
    "round_plan": direction_allocator.current_plan(),
}
```

在架构上很自然。随后 `PromptBuilder.build_user_prompt` 增加 `CURRENT RESEARCH MANDATE`、`FALSIFICATION TESTS`、`ALLOWED/REQUIRED EXPOSURES` 等 section 即可。

### 2.3 已有多 target 和 research score

`ValidationPipeline` 已支持 `target_panels`、`target_horizons`；research 模式支持：

- 多 horizon 权重；
- bootstrap/shrinkage/LCB；
- turnover penalty；
- redundancy penalty；
- residual IC / effective-rank gain。

见：

- `factorminer/core/ralph_loop.py:251-310,406-475`
- `factorminer/architecture/evaluation_kernel.py:62-118`
- `factorminer/utils/config.py:449-597`
- `factorminer/configs/helix_research.yaml:6-65`

这已经不是只能做单一全市场 IC 的框架。真正缺的是：**target 仍在 run 初始化时静态配置，尚无每轮可版本化、带 universe mask/方向/暴露约束的 ObjectiveSpec。**

### 2.4 已有 regime-aware retrieval

`RegimeAwareMemoryPolicy` 会根据当前收益检测的 regime 重排推荐方向，并写入 Prompt。它可作为 `InsightAwareMemoryPolicy` 的样板：把“数据检测的 regime”扩展为“有来源、有有效期、可证伪的人工/Agent mandate”。

### 2.5 已有 specialist/debate 和每个 specialist 的成功率反馈

`SpecialistConfig` 可定义 domain、hypothesis、preferred operators/features 和 avoid patterns；Debate generator 会追踪 specialist admissions/rejections。因此“本轮给价值修复 40% 预算、行业中性 30%、反拥挤 20%、自由探索 10%”可以通过动态创建 specialist 或动态分配 batch quota 实现，而不必重写 DSL/evaluator。

### 2.6 已有 provenance、manifest、checkpoint

Factor 和 run 已有 provenance/manifest；这对机制 C 非常重要，因为动态目标若没有 objective version，就无法知道某因子是在什么观点、数据 cutoff、universe 和 fitness 下入库的。

---

## 3. 当前代码离“真正机制 C”还差什么

### 3.1 缺少一等公民 `ResearchMandate / ObjectiveSpec`

当前 config 中的 research objective 是静态的枚举和 horizon weights：`single_horizon/weighted_multi_horizon/pareto_multi_horizon/net_ir`。它表达不了：

- 本轮研究主题；
- 人类原始洞察和来源；
- 适用 universe/sector/assets；
- long/underweight/neutral 等方向语义；
- 有效起止时间；
- 必须使用/禁止使用的数据；
- 暴露约束；
- 可证伪条件；
- 对多个研究方向的预算。

建议新增：

```yaml
mandate_id: semis_bubble_2026q3_v1
created_at: 2026-08-04T00:00:00+08:00
source_type: human
thesis: "美股半导体估值和资本开支预期透支"
confidence: 0.65
valid_from: 2026-08-04
valid_until: 2026-10-31
scope:
  universe: us_equity
  sector: semiconductors
trade_semantics:
  desired_exposure: underweight
  allow_short: false
horizons: [5, 20, 60]
generation:
  preferred_families: [valuation_revision, crowding, earnings_revision, liquidity]
  forbidden_shortcuts: [future_fundamental, post_close_news]
evaluation:
  primary: conditional_signed_ic
  robustness: [walk_forward, regime_stability, sector_residual_ic]
  exposure_penalty:
    semiconductor_beta: 0.20
falsification:
  - "若盈利预期上修覆盖估值压缩，降低或撤销该 mandate"
  - "若 sector-neutral OOS IC 不显著，拒绝该方向"
```

### 3.2 当前评价对“方向语义”不够严格

当前 paper gate 主要看 `abs(mean(IC_t))`。这对发现“有预测力但方向可翻转”的公式合理，却不能证明它符合“做空半导体”这一语义。

机制 C 需要明确：

- `signal > 0` 是 long 还是 short；
- 入库时是否允许自动翻转符号；
- thesis alignment 是由规则、LLM 还是实证暴露验证；
- “全市场预测好”与“半导体子域预测好”如何加权；
- 子域样本不足时必须返回 `insufficient_evidence`，而不是硬算高 IC。

建议把评价拆成向量：

\[
Score(f,m)=w_q Q_{OOS}+w_c C_{conditional}+w_a A_{thesis}
-w_r R_{redundancy}-w_e E_{exposure}-w_t T_{turnover}
\]

其中：

- `Q_OOS`：固定 protocol 的 OOS 预测质量；
- `C_conditional`：mandate universe 内的 signed IC/IR；
- `A_thesis`：假设—公式—实际暴露一致性；
- `R_redundancy`：与全局库和同 mandate 库的冗余；
- `E_exposure`：不希望的行业/风格暴露；
- `T_turnover`：换手和成本。

`A_thesis` 不能只交给 LLM。至少应包含确定性的实际 score/portfolio exposure 验证。

### 3.3 当前记忆没有 scope，会发生“观点污染”

`ExperienceMemory` 目前没有 `mandate_id/universe/regime/horizon/data_version`。如果同一个全局 memory 连续经历：

- 第 1 轮：挖半导体空头；
- 第 2 轮：挖 AI 资本开支受益多头；
- 第 3 轮：挖全市场行业中性；

同一个公式在不同目标下的 success/failure 含义可能相反。若全部写进同一个 `P_succ/P_fail`，会出现灾难性冲突。

建议三层记忆：

1. `M_global`：语法失败、数值不稳定、普遍冗余、数据泄漏等跨目标知识；
2. `M_objective[mandate_family]`：某类假设下成功/失败的结构模式；
3. `M_context[regime, universe, horizon]`：仅在特定市场状态/股票池/预测期成立的经验。

失败原因至少要分类为：

```text
parse_error
runtime_error
predictive_failure
redundancy_failure
mandate_mismatch
exposure_violation
insufficient_sample
stale_mandate
leakage_violation
execution_failure
```

只有前两类、明确的全局冗余和泄漏违规适合进入全局 forbidden；`mandate_mismatch` 不应成为全局 `P_fail`。

### 3.4 当前 lifecycle/distill 信息不足

当前 lifecycle 主要记录 formula、stage、IC、ICIR、admitted/reason，重建 trajectory 时没有 mandate/objective version、target stats、exposure、数据 cutoff 和完整 early-rejection 类型：`factorminer/architecture/lifecycle.py:57-167`。

动态目标 Agent 必须让每条轨迹至少携带：

```text
mandate_id / objective_version / round_id
human_or_agent_source / evidence_snapshot_hash
universe_mask_hash / target_spec_hash / data_cutoff
prompt_hash / generator_family / specialist_id
target_stats / exposure_stats / alignment_score
rejection_class / falsification_result
```

否则无法审计“为什么第 12 轮说这个因子成功，第 20 轮又说失败”。

### 3.5 当前行业/基本面/新闻数据契约缺失

“人工洞察半导体泡沫”不是一句 Prompt 就能变成可验证目标。至少需要：

- 点时行业分类，避免今天的 GICS 标签回填过去；
- 足够大的行业内截面；
- 点时估值、盈利预期修正、资本开支、库存、订单等基本面；
- 新闻发布时间、抓取时间、可见时间、来源；
- 资产映射和 delisting/survivorship 处理；
- 观点产生时间与回测 cutoff 的严格隔离。

当前 canonical FactorMiner DSL 不能引用这些字段；因此要么扩展 DSL leaf/features，要么让 mandate 只改变 universe/target/constraint，而不要求公式直接读取新闻文本。

### 3.6 当前交易层是 long-only

因此“做空方向”在研究层和交易层必须分开：

- 研究层可以验证某 signal 对未来收益的负向预测；
- 当前执行层只能把它转成少配/零配，不能形成负权重；
- 若要真 short，必须单独迁移账户、保证金、借券、gross/net exposure、费用和风控契约。

---

## 4. 推荐的新 Agent：Mandate-Driven Factor Research Agent

### 4.1 角色分工

```text
Human Insight / Point-in-time Evidence
                |
                v
        Insight Compiler Agent
                |
      ResearchMandate (versioned)
                |
                v
     Mandate Validator / Risk Gate
                |
                v
  Direction Portfolio & Budget Allocator
       |          |          |
       v          v          v
 Specialist A  Specialist B  Free Exploration
       \          |          /
                Miner
       Retrieve -> Generate -> Evaluate
              -> Admit -> Distill
                |
                v
 Global Library + Mandate Libraries + Scoped Memory
                |
                v
       Human Review / Next-round Update
```

#### Insight Compiler Agent

负责把人类自然语言、新闻摘要、行业观察编译为 `ResearchMandate`。它**不能直接修改评价代码**，只能从白名单 objective primitives 中选择。

#### Mandate Validator

确定性检查：

- 数据是否存在且在 cutoff 前可见；
- universe 是否有足够资产和历史；
- 是否与 long-only/成本/风险契约冲突；
- 观点是否可证伪；
- 是否只是不可测试的叙事；
- 是否与已有 mandate 重复或冲突。

不通过时应拒绝或降级为“仅观察”，而不是强行挖因子。

#### Direction Portfolio & Budget Allocator

不要让一个强观点吃掉 100% 搜索预算。建议每轮维持：

- 50–70%：当前人工/Agent mandate；
- 20–30%：与 mandate 相邻但不相同的反证/替代机制；
- 10–20%：不受当前观点约束的自由探索。

可借鉴 R&D-Agent-Quant 的 multi-armed-bandit 思路，但 reward 应使用延迟、去偏、OOS 指标，不能直接追逐最近一轮 in-sample IC。[4]

#### Miner

FactorMiner 保持其擅长的 DSL 生成和确定性执行。动态变化的是：

- prompt mandate；
- specialist 配额；
- universe/target；
- objective weights；
- memory retrieval scope。

#### Reviewer / Human Gate

人类每轮看到：

- 本轮探索了哪些方向；
- 每个方向候选数、通过率、OOS 质量；
- 失败是统计失败、语义不一致、暴露违规还是样本不足；
- Agent 建议下一轮扩大、缩小、反证或终止哪个 mandate。

人可以修改 mandate，但修改会生成新版本，不能覆盖历史版本。

### 4.2 每轮闭环

1. **Freeze**：冻结本轮 `mandate + data cutoff + universe + targets + gates`。
2. **Retrieve**：从 global/objective/context memory 分层检索。
3. **Allocate**：为多个方向分配候选 budget。
4. **Generate**：specialists 根据 mandate 生成公式及自然语言 rationale。
5. **Compile/Execute**：parser 和 deterministic engine 计算 signal。
6. **Validate semantics**：检查公式含义、实际 exposure 和 mandate 一致性。
7. **Evaluate**：固定 IC/ICIR/LCB/OOS/turnover/redundancy/exposure gate。
8. **Admit**：进入 global evergreen library、mandate library，或仅进入 observation pool。
9. **Distill**：按 scope 写入成功/失败经验。
10. **Review**：输出结构化 round report；人/allocator 决定下一轮方向。
11. **Version**：若目标变化，创建 `objective_version + 1`，下一轮生效。

### 4.3 因子库不应只有一个

建议区分：

- **Evergreen Library**：跨 mandate、跨 regime 仍稳健的通用因子；
- **Conditional Library**：只在特定 mandate/regime/universe 下有效；
- **Observation Pool**：有逻辑但证据不足，等待更多数据；
- **Rejected Archive**：完整保存失败和原因，但不全部进入 Prompt。

选因子时，先由 active mandate 和当前数据状态决定哪些 conditional factors 可激活，再与 evergreen factors 组合。

---

## 5. “半导体泡沫”例子的正确实现方式

### 5.1 当前不能直接做的原因

1. 当前 benchmark 是 15 个跨资产序列，不是足够大的美股半导体截面。
2. 当前 DSL 没有行业、估值、盈利预期、库存、新闻字段。
3. 当前实盘模拟是 long-only，不能做净空头。
4. 当前 paper IC 用绝对值门槛，不能自动保证方向和观点一致。

### 5.2 可行的研究版本

假设换成包含美股点时股票池的数据：

#### 方向 1：半导体行业内相对弱势/拥挤反转

- Universe：点时 semiconductor constituents；
- 目标：未来 5/20 日行业内相对收益；
- 因子：量价、流动性、拥挤、盈利预期修正等；
- 评价：sector-internal signed IC + OOS LCB + turnover；
- 执行：long-only 时只低配最差分位，不做负权重。

#### 方向 2：全市场行业中性因子

- 先对 signal 回归行业/size/beta，取 residual signal；
- 评价 residual IC 和边际 rank gain；
- 组合层限制行业相对基准偏离；
- 不是要求“各行业分数均匀”。

#### 方向 3：反证方向

同时挖：

- AI capex/盈利上修是否能解释高估值；
- 半导体内部哪些子行业仍有正向预期修正；
- 泡沫观点是否只是在做 short momentum。

若只让 Agent 寻找支持“泡沫”的因子，会产生 confirmation bias。每个 human mandate 都应自动生成至少一个 falsification/counter-mandate。

---

## 6. arXiv：别人是否做过

### 6.1 直接先行工作

| 工作 | 与本方案重合点 | 未完全覆盖的部分 |
|---|---|---|
| **Alpha-GPT**（arXiv:2308.00016）[3] | 人类提供自然语言交易想法；review 后给反馈；修改自动作用于后续 alpha-mining rounds；明确 interactive mode | 没有当前 FactorMiner 这种明确的 scoped `P_succ/P_fail` 记忆和 objective-version 隔离 |
| **AlphaAgent**（arXiv:2502.16789）[2] | idea agent 融合 human knowledge、research reports、market insights；初始假设来自 user-assigned direction/market insight；评价反馈驱动下一轮假设和因子 | 更偏假设—因子一致性、AST originality 和 complexity；未把人类观点生命周期/有效期/多 mandate 预算作为核心契约 |
| **R&D-Agent-Quant**（arXiv:2505.15155）[4] | Research stage 动态设置 goal-aligned prompts；历史 hypothesis/feedback 生成下一假设；multi-armed bandit 自适应选择下一优化方向 | 是全栈因子—模型联合研发框架，不是专门为 FactorMiner 的 conditional memory 设计 |
| **QuantaAlpha**（arXiv:2602.07085）[5] | 多个互补 research directions 初始化；对完整研究轨迹做 mutation/crossover；保留验证过的 hypothesis/repair segments | 主要是自治轨迹演化，不以 human mandate 的审计、撤销和 scoped memory 为中心 |
| **From Hypotheses to Factors**（arXiv:2604.26747）[6] | Agent 读取 append-only trace，逐轮提出可证伪 hypothesis；round summary 决定下一 search direction；Agent 控制研究方向但不能改评价协议 | 论文是加密资产场景；没有与 FactorMiner 的 Psucc/Pfail、动态 specialist 预算和外部人工洞察生命周期结合 |
| **Beyond Prompting**（arXiv:2603.14288）[7] | 自治 Agent 做多轮假设、因子、固定协议评价和 memory/policy update，强调 OOS 纪律 | 更强调 autonomous factor investing，人工逐轮主导不是核心 |
| **FactorMiner**（arXiv:2602.14670）[1] | Retrieve–Generate–Evaluate–Distill；持久成功/失败模式；正适合作为底层 Miner | 原论文目标主要是高 IC、低冗余的正交因子库，不是人工观点驱动的动态 objective portfolio |

### 6.2 文献结论

因此：

- “LLM/Agent 根据人工洞察在后续轮次改变挖掘方向”——**已有人做过，Alpha-GPT 是非常直接的先例**。
- “Agent 根据反馈自动改变下一研究方向”——**AlphaAgent、R&D-Agent-Quant、QuantaAlpha、From Hypotheses to Factors 都已覆盖不同版本**。
- “将 FactorMiner 的经验记忆、动态目标、多方向预算、人工 mandate 生命周期、目标作用域隔离和固定协议审计结合”——**仍有明确工程与研究空间**。

可以把新 Agent 的论文/项目贡献定义为：

> **Mandate-conditioned, protocol-frozen, memory-scoped factor discovery**：人类或证据 Agent 提供可证伪 mandate；系统只在轮次边界改变方向；每轮评价规则冻结；成功/失败经验按目标与市场上下文隔离；同时保留反证预算和全局自由探索。

这个定义比“给 FactorMiner 加新闻 Agent”更清楚，也更容易做消融实验。

---

## 7. 最小可行版本（MVP）

### Phase 0：不改 FactorMiner，只验证 Prompt steering（1–3 天）

- 人工写 3 个静态 mandate YAML；
- 自定义 `PromptBuilder` 注入 mandate；
- 固定 evaluator 不变；
- 比较不同 mandate 生成的 operator/family/feature 分布。

**通过标准**：方向间候选结构显著不同，语义符合率提高；但明确标注这还不是 conditional fitness。

### Phase 1：加入 Direction Allocator 和动态 specialist（3–7 天）

- 每轮选择 2–4 个 mandate；
- 动态建立 `SpecialistConfig`；
- 分配 batch quota；
- 保留 10–20% free exploration；
- round manifest 保存预算和 prompt hash。

**通过标准**：预算被严格执行；失败方向会降预算但不会立即归零；结果可复现。

### Phase 2：ObjectiveSpec + conditional evaluator（2–4 周）

- 扩展 `DatasetContract` 支持 universe mask、sector metadata 和 target spec；
- `ValidationPipeline` 接收 per-round immutable `ObjectiveSpec`；
- 加 signed conditional IC、sector residual IC、exposure penalty；
- lifecycle/provenance 写入 objective version；
- global/conditional libraries 分离。

**通过标准**：同一公式在不同 objective 下得到可解释的不同判定；resume 后 objective hash 一致；不允许中途偷改门槛。

### Phase 3：Scoped Memory（1–2 周，可与 Phase 2 部分并行）

- memory entry 增加 scope；
- retrieval 先按 objective/context 过滤，再排序；
- `mandate_mismatch` 不写全局 forbidden；
- 测试跨目标污染和旧 schema migration。

### Phase 4：新闻/基本面 Insight Agent（4–8+ 周）

- 只读取 point-in-time evidence snapshots；
- 输出引用、时间、confidence、有效期、反证条件；
- Validator 决定是否转成 active mandate；
- 人类可 approve/edit/reject；
- 禁止 Agent 直接看 holdout 表现后改新闻解释。

---

## 8. 必须做的实验与消融

至少比较：

1. `FM-base`：当前 FactorMiner；
2. `Prompt-only`：只注入人工方向；
3. `Fitness-only`：只改 conditional evaluator；
4. `Prompt + Fitness`；
5. `Prompt + Fitness + Scoped Memory`；
6. `Full`：再加 direction allocator 和 human/insight update。

指标不应只看最终收益：

- candidate parse rate；
- thesis semantic alignment；
- realized exposure alignment；
- per-direction high-quality yield；
- OOS signed IC/ICIR/LCB；
- global/conditional library redundancy；
- turnover/cost/capacity；
- direction diversity 和 entropy；
- false-discovery/FDR；
- memory contamination rate；
- mandate 终止/反证成功率；
- 人类修改前后方向稳定性；
- 多随机种子和多个市场窗口稳健性。

特别要验证：

- 人工观点错误时，系统能否停止而不是不断寻找支持证据；
- 新闻 Agent 是否引入未来信息；
- 目标变化后旧 `P_fail` 是否错误压制新方向；
- 动态方向是否只是提高 in-sample IC、却损害 frozen holdout；
- long-only 落地是否仍与研究层的“空头因子”含义一致。

---

## 9. 风险与设计红线

### 9.1 Confirmation bias

人类说“半导体泡沫”，Agent 很容易把任务理解为“找出证明泡沫的公式”。解决办法是强制 counter-mandate、falsification tests 和退出条件。

### 9.2 动态目标导致 p-hacking

若每轮根据 validation/holdout 结果修改目标、阈值、universe 或 horizon，系统会自动化 HARKing。正确做法：

- 每轮 protocol 冻结；
- 只允许在下一轮创建新 objective version；
- test/holdout 不参与搜索策略更新；
- 所有尝试计入 multiple-testing budget。

### 9.3 非点时新闻/基本面泄漏

新闻正文发布时间、数据库入库时间、财报发布日期、修订历史必须可追踪。没有 point-in-time 数据时，不应宣称完成了历史新闻驱动回测。

### 9.4 小样本条件 IC

行业内资产太少时，截面 IC 极不稳定。15 资产跨资产 universe 尤其不适合验证细行业观点。必须设最小资产数、最小有效日期数和不确定性下界。

### 9.5 记忆污染和概念漂移

动态目标会让同一模式在不同上下文下标签相反。没有 scoped memory 时，长期运行可能比无记忆更差。

### 9.6 研究层和交易层契约不一致

能验证负向预测，不代表当前 long-only simulator 能做空。文档、factor direction、portfolio mapping 和风险约束必须一致。

---

## 10. 最终建议

### 是否值得做？

**值得。** 当前 FactorMiner 的 stage、multi-target research score、regime memory、specialist debate、provenance 已经提供了很好的底座，没必要重新造一个 Miner。

### 应该从哪里开始？

先实现：

1. `ResearchMandate` schema；
2. 每轮不可变 `ObjectiveSpec`；
3. Prompt 注入和动态 specialist 配额；
4. objective-scoped lifecycle/provenance；
5. scoped memory；
6. conditional evaluator。

最后再接新闻/基本面 Agent。

### 新 Agent 的最合理定位

不是“替 FactorMiner 选几个热门行业”，而是：

> **研究总监 Agent（Research Director）**：将人工洞察和点时证据转成可证伪 mandate，在多个方向间分配探索预算，冻结每轮实验协议，监督 FactorMiner 生成与评价，并根据结构化证据而非叙事更新下一轮方向。

### 对 `research1.md` 的总体评级

- **核心直觉：正确。** 顶层经济/行业洞察与底层符号因子搜索确实互补。
- **机制 C 的方向：正确且可做。**
- **FactorMiner 细节：多处混淆。** 特别是 `P_succ/P_fail` 的 AST 存储、tree-edit retrieval、持久化 signals/metadata。
- **AlphaCrafter 现状：描述过时。** 当前已有持久因子库和交易反馈，只是缺少 FM 风格的统一经验蒸馏策略。
- **创新性判断：需要收窄。** “人工指导后续轮次”已有直接先例；真正可形成差异的是 mandate contract、protocol freeze、scoped memory、反证机制和多方向预算。

---

## 参考文献（arXiv 原文）

[1] Wang et al., **FactorMiner: A Self-Evolving Agent with Skills and Experience Memory for Financial Alpha Discovery**, arXiv:2602.14670, 2026. <https://arxiv.org/abs/2602.14670>

[2] Tang et al., **AlphaAgent: LLM-Driven Alpha Mining with Regularized Exploration to Counteract Alpha Decay**, arXiv:2502.16789, 2025. <https://arxiv.org/abs/2502.16789>

[3] Wang et al., **Alpha-GPT: Human-AI Interactive Alpha Mining for Quantitative Investment**, arXiv:2308.00016, original 2023, v2 2025. <https://arxiv.org/abs/2308.00016>

[4] Li et al., **R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization**, arXiv:2505.15155, 2025. <https://arxiv.org/abs/2505.15155>

[5] Han et al., **QuantaAlpha: An Evolutionary Framework for LLM-Driven Alpha Mining**, arXiv:2602.07085, 2026. <https://arxiv.org/abs/2602.07085>

[6] Huang et al., **From Hypotheses to Factors: Constrained LLM Agents in Cryptocurrency Markets**, arXiv:2604.26747, 2026. <https://arxiv.org/abs/2604.26747>

[7] Huang and Fan, **Beyond Prompting: Autonomous Factor Investing via Agentic AI**, arXiv:2603.14288, 2026. <https://arxiv.org/abs/2603.14288>
