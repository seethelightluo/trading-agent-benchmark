# AC 实验 Checklist

> 本地仓库只运行 AlphaCrafter（AC）。本文只记录 AC 的原生规则、当前资产世界线合同、两套独立 shared warmup、代码修复和运行迁移；不记录其他框架的运行计划。

## 1. 两套实验边界

- Terra AC：一份 Terra shared warmup，供 Terra 实验内的世界线复用。
- DeepSeek AC：另一份 DeepSeek shared warmup，供 DeepSeek 实验内的世界线复用。
- Terra 与 DeepSeek 不共享 warmup、不复制因子、不混用 checkpoint、workspace 或运行状态。
- 两份 warmup 均必须原样保留；无论已经完成还是正在运行，都不能 reset、清理、覆盖或因 online 合同变化而重新挖掘。
- 旧 online 阶段使用了已废弃的调仓语义，全部作为历史结果保留，但不再 resume、续跑或作为新合同的起点。

## 2. 当前资产世界线正式合同

### 2.1 数据与时间

- 权威数据目录：`WL-data-final/`。
- 15 个 tradable asset 进入持仓；DXY、USDCNY、USDJPY、EURUSD、VIX 仅为观察信号，不得持仓。
- 研究可见截止：`2026-07-15`。
- 首个 online 执行日：`2026-07-16`。
- 初始资金：`1_000_000` USD-equivalent。
- AC cycle：每 10 个交易日一个研究/决策 block。
- shared warmup 目标：40 个 AC cycle；每 cycle 为 3 Miner + Screener + Trader。

### 2.2 因子库

- 正式准入只承认 `abs(IC) >= 0.007` 且 `abs(ICIR) >= 0.084`。
- 质量定义：`quality = abs(IC) * abs(ICIR)`。
- 使用同一可见数据和共同有效样本计算真实 `abs(Spearman rho)`。
- `rho < 0.5`：两个因子都保留。
- `rho >= 0.5`：保留质量较高者，淘汰质量较低者；同质量按稳定 factor ID 决定。
- 冲突处理后，滚动因子库最多 30 个，再按质量保留 best30。
- active ensemble 最多 10 个，可以少于 10 个；按质量/IC tilt 分配权重并保留 `sign(IC)` 方向。
- “最多 10”只限制 active ensemble，不限制 warmup 的研究轮数或候选数量。
- 缺少可恢复 formula、signal 或 provenance 的历史因子进入 quarantine，不能把自报 `rho=0` 当作真实相关性证据。
- 被淘汰的 factor ID 必须同步从 library、audit、checkpoint、signals 和 ensemble 中移除，resume 不得复活。

### 2.3 组合与交易

- long-only，仅允许 15 个 tradable asset；目标权重非负且和为 1；允许 fractional quantity；online cash 始终为 0。
- 首次 `2026-07-16` 无条件全仓 1M：有合格 ensemble 时使用其 target；没有时使用 15 资产等权 `1/15`。
- 后续 Trader 只产生 proposal，不直接改变账户。
- `one_way_turnover = 0.5 * sum(abs(target-current))`。
- `gross_edge_bps = 10000 * sum((target-current) * forecast_returns)`。
- 后续只有 `gross_edge_bps > one_way_turnover * 3` 才执行；`edge <= migration * 3bp` 一律 no-trade。
- 该门槛不是资产总额固定 3bp，也不是双边固定 6bp。
- 实际成本：`NAV * one_way_turnover * 3 / 10000`。
- no-trade 必须保存 proposed target、executed target、forecast、factor IDs、turnover、edge、threshold、actual cost、executed 和 skip reason；真实持仓不变。
- 15 资产 benchmark 的 `add_order` 路径禁止绕过 gate；`ensure_fully_invested()` 只能修复明确账户损坏。

## 3. AC 原版规则审计

### 3.1 原版因子冲突

原版 `factor_screening.md` 只是方法提示：对选中因子计算 pairwise correlation，以
`correlation > 0.7` 识别 cluster，可选择保留最高 ICIR、正交化或限制相关组权重。

原版没有确定性的共同样本 signal 入库校验、冲突双方质量比较、滚动 library=30 或
active<=10 硬限制。当前 `.007/.084/.5`、`quality=abs(IC)*abs(ICIR)` 和低质量淘汰规则属于
本资产世界线适配层。

### 3.2 原版交易与费用

原版 strategy baseline 是通用股票示例：`TOP_N=50`、gross exposure `0.6`、直接调用
`add_order`，订单可 pending。原版 Exchange 使用 1bp commission + 2bp 买卖滑点，支持
T+0、做空、20% short margin、80% maintenance margin；订单价格不在当日 low/high 时保持
pending。

这些原版规则不作为本世界线 online 合同。15 资产 benchmark 必须使用原子 fractional
rebalance 和 proposal gate，避免部分成交、现金残留、做空以及订单绕过成本决策。

## 4. 两套 shared warmup 保留与 online 迁移

### 4.1 warmup 保留原则

- Terra warmup 与 DeepSeek warmup 是两份独立研究资产，各自保留自己的因子、公式、脚本、ensemble、audit、workflow、checkpoint 和日志。
- 修改在线调仓规则不触发 warmup 重跑，也不修改 warmup 的 `account.json`、`date.json`、workflow 或因子 JSON。
- 对正在运行的 warmup，只允许暂停/恢复进程，不允许删除、迁移、批量重写或重新播种 workspace。
- 新合同代码只能在新 online run 中使用；若 warmup 继续运行，必须保持其独立 workspace 和 provider 配置。

### 4.2 旧 online 放弃

- Luna/Terra 旧 3WL online 的账户、日志、状态和结果全部保留为历史证据。
- DeepSeek 旧 online 状态同样保留为历史证据。
- 旧 online 不再使用 `--resume`，不进行补跑，不与新合同结果拼接。
- 两套 AC 的 `portfolio_contract_version` 现在为 `ac-worldline-v2-migration-gate`。
- 已初始化但缺少该版本标记的旧 online account 会被执行层 fail-closed；代码不会清除订单、重写账户或自动迁移旧持仓。
- 新 online 必须从对应实验自己的 warmup 复制出新 workspace，并生成新的 code/contract fingerprint、account 和 workflow；不得从旧 online account 续跑。

## 5. 已落地代码修复

- Terra/DeepSeek 的 `portfolio_contract.py` 完全一致，统一 proposal、turnover、edge、3bp migration gate 和 decision audit。
- Terra/DeepSeek 的 `rebalance_to_weights.py` 完全一致，支持 fractional quantity、15 资产、cash=0、首次全仓和 no-trade 持久化。
- Terra/DeepSeek 的 `factor_contract.py` 完全一致，执行 `.007/.084`、真实 signal pairwise rho、质量冲突淘汰、capacity 30、quarantine 和 audit。
- AC Miner 自报 correlation 只作为 provenance/audit 字段，不再在候选阶段直接拒绝；真实 signal 才是冲突判定依据。
- `add_order` 对 15 资产账户硬失败，不能绕过 proposal/gate。
- `ensure_fully_invested()` 在清理 pending order 前先检查新 portfolio contract，旧 online 状态不会被改写。
- warmup 中若仍保存原生直连 `add_order` 的历史 strategy，只在新 online workspace 播种时安装 proposal/gate adapter；warmup 原件不改写。两份 AC 的 A/US Exchange 保存器保留 rebalance 扩展审计字段，post-tick 不得抹掉 contract/version、proposal、执行成本或 no-trade 记录。
- DeepSeek resume retry 状态在测试构造的 Launcher 场景下也有默认值，不会产生伪失败。

## 6. 启动前检查

- [ ] 确认只选择 Terra 或 DeepSeek 其中一个实验，不跨实验复制 warmup 因子。
- [ ] 确认对应 shared warmup 的 workflow、factor library、ensemble、checkpoint、日期边界和 fingerprint 完整。
- [ ] 确认新 online 使用新 account、new workspace、new workflow 和新 contract fingerprint。
- [ ] 确认旧 online PID 已停止/暂停且没有使用 `--resume` 的启动器。
- [ ] 先通过无 API 单测、factor conflict fixture、portfolio gate fixture 和 1 block smoke，再启动长跑。
- [ ] 新 online 首次日期为 `2026-07-16`，首次账户最终为 15 个 fractional positions、cash=0。

## 7. 验收标准

- [ ] 两套 AC 源代码中对应 contract、factor policy、rebalance helper、StepTool 的行为一致。
- [ ] `rho < .5` 双保留；`rho >= .5` 只保留高质量因子；容量超过 30 时稳定淘汰末尾。
- [ ] 首次建仓无条件执行；迁移比例为 `x` 时，edge 不超过 `3x bp` 跳过，严格超过才执行。
- [ ] no-trade 不改变真实持仓、executed target 和 cash。
- [ ] 旧 online account 无新版本标记时 fail-closed 且不被改写。
- [ ] Terra 与 DeepSeek warmup 原文件、状态、日志和因子产物均未被代码修复修改。

## 8. 当前进度（最后 live audit）

- Terra shared warmup：40/40 cycle，workflow 完整；warmup 保留。
- DeepSeek shared warmup：6 个完整 cycle，第 7 cycle 已开始但只完成 1 个 Miner 步骤；目标 40 cycle；warmup 保留并可从原位置恢复。
- Terra 旧 3WL online：已暂停并放弃，不删除结果。
- Terra 新 Luna online：`ac_luna_3wl_v4` 已启动，`terra_v4_wl1/2/3` 三 WL 并行；首个 10 日 seed block 完成，三账户均通过合同 smoke。
- Terra v2/v3 启动尝试：保留为失败/修复证据，不与 v4 拼接或 resume。
- DeepSeek 旧 online：已暂停/不再恢复，不删除结果。

## 9. 证据入口

- `runAC.md`
- `agent-framework/ASSETS.yaml`
- `agent-framework/AlphaCrafter/portfolio_contract.py`
- `agent-framework/AlphaCrafter/alphacrafter/factor_contract.py`
- `agent-framework/AlphaCrafter/alphacrafter/sim/utils/rebalance_to_weights.py`
- `agent-framework/AlphaCrafter/alphacrafter/agent/toolkit/step.py`
- `WL-data-final/`
- Terra warmup：`agent-framework/AlphaCrafter/alphacrafter/sandbox/ws1/`
- DeepSeek warmup：`AC-deepseek/AlphaCrafter/alphacrafter/sandbox/ws1/`
