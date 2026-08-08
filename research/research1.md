# 研究原稿：人工洞察驱动的动态因子挖掘

> 本文件根据 2026-08-04 Codex 会话中读取到的 165 行原稿恢复，并对终端换行、公式下标和乱码做了规范化；原始运行结果不包含在内。

FactorMiner 的 `P_success/P_fail` 经验记忆和 AlphaCrafter 的多 Agent loop 具有互补性：前者擅长量价因子挖掘和失败经验沉淀，后者擅长新闻、基本面和策略执行。一个自然的融合方向，是让人工或 Agent 的行业洞察影响因子权重、组合约束以及后续挖掘目标。

## 研究问题

当前 FactorMiner 主要依据价量数据和统计指标挖掘因子，缺少“当前基本面/新闻是否支持该因子”的条件判断。AlphaCrafter 则缺少 FactorMiner 式的持久经验库和持续质量筛选。需要区分：

- 因子公式及其 AST/统计元数据的持久化；
- `P_success/P_fail` 的经验记忆和检索；
- 当前宏观洞察对候选降权、拒绝或下一轮搜索方向的影响。

## 三种融合机制

### 机制 A：因子暴露度审查

新因子产生后，计算其 top quantile 的行业、资产和风险暴露。如果人工洞察认为半导体泡沫较大，而候选的多头结果高度集中于半导体，则把它标记为宏观逻辑冲突，拒绝入库或降权。该方法不要求修改 Miner 的生成器，但需要行业标签、点时数据和暴露度审计。

### 机制 B：组合层硬约束

FactorMiner 仍按原统计协议挖掘，基本面 Agent 作为风险总监把洞察翻译为组合约束，例如行业权重上限、行业中性、资产低配或风险预算。这是最容易与现有执行器结合的方式，但不会改变因子的统计目标。

### 机制 C：动态改变 Miner 挖掘目标

基本面 Agent 将洞察转为下一轮可验证的 mandate，例如：

1. 在有足够行业截面和点时标签的前提下，挖掘半导体行业内相对弱势/反转方向；
2. 同时挖掘行业中性或残差化因子；
3. 规定 horizon、universe、目标变量、暴露约束和反证条件；
4. 在下一轮边界切换 Prompt、specialist 配额和 fitness，而不是在一轮中途修改协议。

这才是 Conditional Mining。只改 Prompt 而不改 evaluator、memory scope 和 provenance，实际上仍然是按全市场 IC 选择因子，容易形成“叙事上换方向、统计上不换目标”的错位。

## 初步判断

- 只注入方向 Prompt：可行且改动小，但只是 steering。
- 动态方向预算和 specialist：可行，可复用当前 debate/specialist 架构。
- 方向对应 universe、target、fitness 和记忆作用域：可行，但需要新的不可变 `ObjectiveSpec`/`ResearchMandate` 合同。
- 新闻 Agent 自动生成和撤销 mandate：可行性中等，主要风险是点时泄漏、错误洞察、confirmation bias 和审计困难。

第一版建议保留固定 evaluator，先做人工 mandate 的静态对照；之后再做每轮方向 allocator、scoped memory、签名 IC 和 point-in-time 新闻。每轮冻结 `mandate + cutoff + universe + target + gates`，下一轮才允许创建新版本，并保留 counter-mandate/falsification 预算。

## 当前 benchmark 的限制

当前 universe 是 15 个跨资产序列，不是足够大的美股半导体截面；DSL 主要是 OHLCV/returns，缺少行业和点时基本面字段；执行契约是 long-only、全投资。因此“半导体泡沫”只能作为架构示例，不能直接宣称已经在当前 benchmark 中完成行业内空头因子验证。

