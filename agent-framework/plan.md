# 详细落地计划：日频 + 统一 3bps 摩擦的 AlphaCrafter & FactorMiner 前向走步（Walk-Forward）测试

> 本文件基于 [refer.md](refer.md) 的核心讨论整理而成，是该测试的**权威工程蓝图**。
> 关键结论：**恢复两个框架的原生日频（Daily）**，叠加**全资产统一单边 3bps（1bp 佣金 + 2bp 滑点）交易摩擦**，仅对两个框架各做**一处最小改造**，即可让大模型自主学会"静若处子、动若脱兔"的日频调仓艺术。
> 编写日期：2026-07-20

---

## 0. 一句话目标

在 2026.07.16 虚构基准上，把 **AlphaCrafter**（宏观常识研判 + 日频组合优化）与 **FactorMiner**（纯量价日频因子自进化）放进一个**物理沙箱**里，让它们从 2026.07.16 滚动跑到 2035.12.31，**每日**根据"截至当日可见的全部历史"做调仓决策；用项目生成的**九条虚构世界线**作为评估 ground truth，计算扣除模拟摩擦后的策略净值。

---

## 1. refer.md 核心讨论摘要（决策演化路径）

讨论经历了三轮用户拍板，最终收敛为"日频 + 3bps 摩擦 + 两处最小改造"：

| 轮次 | 用户指令 | 工程结论 |
|------|---------|---------|
| 第 1 轮 | 时间线改为 2026.07.16 基线；2020-2026 为历史 warm-up；之后为 agent 学习/交易期；确认两个框架原生频段；FactorMiner 不带新闻 | AlphaCrafter 原生**日频**（Miner/Screener/Trader 每日旋转）；FactorMiner 原生**日频因子评估引擎**，本身不做持仓 |
| 第 2 轮 | 提出交易摩擦成本问题；**不要**强行按月，用**原生日频**，否则太假；若没有摩擦就按规则设费率，但"如何反馈给 agent 决策"是核心问题 | 区分"硬数理约束"与"软语义反射"两条路径让 LLM 对摩擦产生"痛感" |
| 第 3 轮（**最终**） | 只要改 AlphaCrafter 的"维度 A：数理约束"就够；FactorMiner 只要激活 `turnover_aware`；**全资产统一 1bp 佣金 + 2bp 滑点**，鼓励有效交易 | ✅ **确认为最终方案**：两处最小改造 + 统一 3bps |

**最终方案的三大优势**（refer.md 第四节）：
1. **绝对公平的跨资产横截面赛跑** —— 15 项可交易资产站在完全相同的摩擦起跑线，不会被某类资产原生低费率带偏。
2. **极力鼓励"有效交易（Effective Trading）"** —— 往返 6bps 成本，只有确信超额收益远超 6bps 时 agent 才会动手。
3. **无作弊、高保真的净值曲线** —— 挤掉高频刷单的"虚假繁荣"水分。

---

## 2. 测试设定（基准与时间轴）

### 2.1 时间轴

- **历史 warm-up（2020.01.01 — 2026.07.15）**：真实历史，用于两套系统的冷启动（AlphaCrafter 初始状态、FactorMiner 初始因子库与经验记忆）。
- **在线滚动（2026.07.16 — 2035.12.31）**：策略真正学习、交易、自适应进化的时间线。第 $t$ 天，agent 只能看到 $[2020.01.01,\ t]$ 的历史价格与 $t$ 之前的新闻，逐日更新持仓与因子公式。

> **边界红线：2026.07.16 起全部为项目生成的虚构世界线数据，不得标记或解释为真实市场数据。**

### 2.2 20 项基准数据宇宙（15 可交易 + 5 只读参考）

| 资产类别 | 标的 | 基线 | 来源 |
|---------|------|------|------|
| 可交易·权益 | 沪深300 (000300.SH) | 4,608 | 世界线统一基线（虚构段起点） |
| 可交易·权益 | 标普500 (SPX) | 7,534 | 世界线统一基线（虚构段起点） |
| 可交易·权益 | 恒生指数 (HSI) | 24,586 | 世界线统一基线（虚构段起点） |
| 可交易·权益 | 日经225 (N225) | 68,000 | 世界线统一基线（虚构段起点） |
| 可交易·权益 | 斯托克50 (SX5E) | 5,100 | 世界线统一基线（虚构段起点） |
| 可交易·权益 | 科创50 (000688.SH) | 1,920 | 世界线统一基线（虚构段起点） |
| 可交易·权益 | 费城半导体 (SOX) | 5,800 | 世界线统一基线（虚构段起点） |
| 可交易·权益 | 纳斯达克100 (NDX) | 20,500 | 世界线统一基线（虚构段起点） |
| 可交易·商品 | 黄金 (XAU) | $4,050/oz | 世界线统一基线（虚构段起点） |
| 可交易·商品 | 铜 (LME Copper) | $13,600/t | 世界线统一基线（虚构段起点） |
| 可交易·商品 | 原油 (WTI) | $79/bbl | 世界线统一基线（虚构段起点） |
| 可交易·加密 | BTC | $64,800 | 世界线统一基线（虚构段起点） |
| 可交易·加密 | ETH | $1,920 | 世界线统一基线（虚构段起点） |
| 可交易·债券 | 美债10Y (US10Y) | 4.30% | 世界线统一基线（虚构段起点） |
| 可交易·债券 | 中债10Y (CN10Y) | 2.20% | 世界线统一基线（虚构段起点） |
| 只读参考 | 美元指数 (DXY) | 100.5 | 世界线统一基线（不持仓） |
| 只读参考 | USD/CNY | 6.78 | 世界线统一基线（不持仓） |
| 只读参考 | USD/JPY | 162 | 世界线统一基线（不持仓） |
| 只读参考 | EUR/USD | 1.16 | 世界线统一基线（不持仓） |
| 只读参考 | VIX | 16 | 世界线统一基线（不持仓） |

### 2.3 统一交易费率（全资产）

- **单边总摩擦 = 3bps = 0.0003**（1bp 佣金 + 2bp 滑点），对 15 项**可交易资产**完全一致；5 项只读参考不下单、不产生交易摩擦。
- 日频往返成本 = 6bps。年化高换手策略将被严重惩罚。

---

## 3. 框架获取与目录结构

### 3.1 两个框架

| 框架 | 论文 | 角色 | 仓库（已核实） | 本地目录 |
|------|------|------|------|---------|
| **AlphaCrafter** | arXiv:2605.05580（Miner + Screener + Trader 全栈多智能体） | 宏观常识研判 + **日频组合优化主决策中枢** | `github.com/NJU-LINK/AlphaCrafter`（官方，南京大学 NJU-LINK org，MIT，2026-07-17 仍在维护） | `agent-framework/AlphaCrafter/` |
| **FactorMiner** | arXiv:2602.14670（Ralph Loop：retrieve-generate-evaluate-distill，含 Skills + Experience Memory） | **纯量价日频因子**自进化开采（不读新闻） | `github.com/minihellboy/factorminer`（**社区重实现**，93★，MIT；论文官方作者尚未公开代码） | `agent-framework/FactorMiner/` |

> **出处说明（PROVENANCE）**：
> - AlphaCrafter 为论文官方仓库，结构已核实：根包 `alphacrafter/`，入口 `python main.py --session_id <id>`，Docker Compose 跑沙箱，session 目录含 `config/`、`logs/`、`persistent/{index_data,stock_data,stock_financial_statements,stock_news,account.json,date.json}`、`workspace/`。
> - FactorMiner `minihellboy/factorminer` 是第三方"based on the paper"的独立重实现（README 自述，并打包了论文 PDF `2602.14670v1.pdf`），**非作者官方代码**。其模块路径（`core/cli.py`、`core/miner/registry.py`、`user_workspace/custom_fitness/turnover_aware.py`、`user_workspace/config.json`）需在下载后核对——若与 refer.md / 本计划引用的路径不一致，以实际仓库为准并回写本文件。底层依赖 `gplearn`（遗传规划）。

### 3.2 目标目录结构

```
trade-agent-research/
├── data-prepare/                     # 已有：九条世界线 + 基准
│   ├── wordline-simple/wordline1..9.md   # ground-truth 未来行情
│   ├── data2020-2026/                # 待下载：warm-up 历史 OHLCV
│   └── data_processor/               # 待建：数据格式化脚本
├── agent-framework/                  # ← 本文件夹
│   ├── refer.md                      # 原始讨论
│   ├── plan.md                       # ← 本文件
│   ├── AlphaCrafter/                 # 待下载
│   ├── FactorMiner/                  # 待下载
│   ├── sandbox/                      # 待建：物理沙箱（§6）
│   ├── scheduler/                    # 待建：Master 日频调度器（§7）
│   └── results/                      # 待建：每日权重、因子、净值（§8）
└── report-and-output/                # 已有：评估报告输出
```

---

## 4. AlphaCrafter 改造方案（"维度 A：数理约束"在真实代码里的落点）

> ⚠️ **代码核查结论（下载仓库后确认）**：AlphaCrafter 的 Trader 是一个 **LLM 智能体**，通过 `agent/toolkit/add_order.py`、`cancel_order.py` 等工具对一个**模拟交易所**下单，并在 `sandbox/<session>/workspace/strategy.py` 写策略代码 —— 它**并不存在** refer.md 伪代码里那个独立的"MPT 凸优化器文件"。
>
> 因此 refer.md 说的"维度 A 数理约束"，在真实代码里**真正的落点是模拟交易所的成交结算**：`alphacrafter/sim/exchange_a.py`（A 股）与 `alphacrafter/sim/exchange_us.py`（美股）。当前 A 股佣金硬编码为 `self.commission_rate = 0.0002`（2bps），且**无滑点模型**（按订单价成交）。
>
> 每一笔成交扣 3bps，本身就是"硬数理约束"：LLM Trader 每天看到账户净值被磨损，自然会被倒逼减少无意义调仓 —— 这正是 refer.md 想要的"不敏感带/死区"效果。无需单独的 L1 惩罚文件。

### 4.1 目标函数（语义不变，落地形态为交易所结算）

每日求解最优权重向量 $w_t$（由 LLM Trader 通过下单近似实现）：

$$
\max_{w_t}\ \left(\ w_t^{T}\alpha_t \ -\ \lambda\, w_t^{T}\Sigma w_t \ -\ \gamma\,\lVert w_t - w_{t-1}\rVert_1 \times \text{cost}\ \right),\quad \text{cost}=0.0003
$$

### 4.2 具体改动（两处）

**(a) 统一成交摩擦到 3bps（核心）** —— 改 `sim/exchange_a.py` 与 `sim/exchange_us.py`：

```python
# 原: self.commission_rate = 0.0002   (仅佣金, 无滑点)
self.commission_rate = 0.0001          # 佣金 1bp
self.slippage_rate   = 0.0002          # 滑点 2bp (新增)
# 单边总摩擦 = 1bp + 2bp = 3bps = 0.0003，全资产统一
```

并在订单撮合处对成交价加滑点（买单价 ×(1+slippage)，卖单价 ×(1−slippage)），佣金按成交额 × commission_rate 扣除。**所有资产（权益/商品/加密/国债/汇率）走同一套费率，不分类。**

**(b)（可选）显式换手死区** —— 若想让 LLM Trader 提前"感知"而不只是事后被扣钱，可在 `agent/instructions/trader.py`（Trader 的 prompt）中补一段提示：*"每次调仓单边成本 3bps、往返 6bps；只在预期超额收益显著超过 6bps 时才调整持仓，避免对微小信号频繁换仓。"* 这对应 refer.md 的"软语义反射"，但**非必需**——单纯靠 (a) 的硬扣费即可实现死区。

### 4.3 不敏感带（Dead-band）机理

每笔成交扣 3bps，相当于在净值曲线上对"调仓"这一动作征收固定税。若某资产的预期 Alpha 微弱（如仅 0.01% 超额），调仓带来的收益不够抵扣 3bps 损耗，理性 Trader 会选择不换仓。优化/决策层面无需显式写惩罚项，**交易所结算层的固定费率本身就是死区**。

### 4.4 注意事项

- AlphaCrafter 原生是**单市场横截面**（CSI300 或 S&P500，A 股 T+1 不准卖）。本测试要做**15 项跨资产**配置，需要：① 扩展交易所支持多类资产标的；② 放开 A 股 T+1 / 做空限制以适配组合权重 $w_t$；③ 将债券收益率转换为基准约定的可交易收益序列。5 项指数/宏观参考只作观察输入，不转换为持仓。这是比"加摩擦"更大的工作量，列入 §10 风险。

---

## 5. FactorMiner 改造方案（仅激活 `turnover_aware`）

**只需改一处**：在 `user_workspace/custom_fitness/turnover_aware.py` 注册并激活换手率适应度钩子，费率统一 0.03。

### 5.1 纯价格驱动（去新闻）

将输入特征严格限定为 `["open","high","low","close","volume"]`，不注册任何情感分析列。`user_workspace/config.json`：

```json
{
  "data_fields": ["open", "high", "low", "close", "volume"],
  "use_news_sentiment": false
}
```

### 5.2 turnover_aware 钩子（V4）

```python
from core.miner.registry import EvaluatorRegistry

@EvaluatorRegistry.register_fitness_hook("turnover_aware")
def turnover_aware(factor_values, returns, base_metrics: dict) -> dict:
    ic = float(base_metrics.get("IC", 0.0))
    turnover = float(base_metrics.get("Turnover", 0.0))

    # 统一单边费率惩罚：3bps；turnover 为日频双边换手率(0~1)
    penalty = 0.03 * turnover

    return {
        "fitness_score": abs(ic) * 100 - penalty,
        "turnover_penalty": penalty,
    }
```

### 5.3 经验记忆红线阻断（Experience Memory）

当一个高频量价因子（如 `close - delay(close, 1)`）因换手率高达 80% 导致 fitness 为负被淘汰时，其 AST 子树会被蒸馏进失败禁忌区 $P_{\text{fail}}$。后续 LLM 在生成新因子时被限制读取 $P_{\text{fail}}$，被迫组合长周期平滑算子（`rolling_mean(close,20)`、`decay_linear()`、`Rsquare()` 等），从代码生成源头逼迫 agent 变稳健。

### 5.4 FactorMiner 不直接决定仓位

FactorMiner 是因子发现引擎，无原生仓位逻辑。其产出（top 公式）交由 §7 调度器做**截面分位数评分**，与 AlphaCrafter 的权重相互印证（详见 §7.3）。

---

## 6. 沙箱隔离与防穿越（Anti-Lookahead）

refer.md 强调"通过物理沙箱确保大模型绝对无法利用联网搜索（Search）作弊获取未来走势"。落地措施：

1. **网络隔离**：调度器以无网环境（容器/namespace 断网）运行 agent，LLM API 走白名单代理，禁止任意 web search。
2. **时间游标（Time Cursor）**：维护一个全局 `current_date`，调度器在第 $t$ 天只把 $[2020.01.01,\ t]$ 的价格切片喂给 agent；九条世界线的 2026.07.16 之后的"未来价"在评估阶段才解封。
3. **新闻切片**：仅 AlphaCrafter 使用，每月（或每日）只注入 $t$ 时刻之前的新闻 JSON；FactorMiner 完全不注入新闻。
4. **审计日志**：记录每次 agent 可见的数据窗口，便于事后核查有无穿越。

---

## 7. Master 日频调度器（scheduler/）

### 7.1 日频主循环（2026.07.16 → 2035.12.31）

```
for t in trading_days(2026.07.16, 2035.12.31):
    price_slice = load_prices("2020.01.01", t)          # 仅历史，防穿越
    news_slice  = load_news(None, t)                      # 仅 t 之前

    # --- AlphaCrafter（含新闻） ---
    AC.run_daily(price_slice, news_slice)                 # Miner/Screener 日频
    w_t = AC.Trader.optimize(alpha_t, Sigma, w_prev,      # §4 换手惩罚
                             gamma=1.0, cost=0.0003)

    # --- FactorMiner（纯价格，可降频触发以省算力） ---
    if t 是月末 or 触发条件:
        FM.run(price_slice, config="pure_price.json")     # §5
        top_alphas = FM.top_formulas(k=10)
        rank_score = cross_section_rank(top_alphas, price_slice)

    # --- 结算 ---
    pnl, cost_paid = settle(w_prev -> w_t, price_t, cost=0.0003)
    w_prev = w_t
    log(t, w_t, pnl, cost_paid)
```

### 7.2 摩擦结算

每日按 $\text{cost\_paid} = \lVert w_t - w_{t-1}\rVert_1 \times 0.0003$ 从净值中扣除，并写入持久化记忆 H（供后续可能的语义反射扩展使用，当前方案不强依赖）。

### 7.3 两套系统的协同

- **AlphaCrafter 输出**：可解释的每日资产权重 $w_t$（主决策）。
- **FactorMiner 输出**：15 可交易资产截面分位数得分（数理佐证；5 信号作输入特征不持仓）。
- 默认以 **AlphaCrafter 的 $w_t$ 为执行权重**；FactorMiner 的得分作为**独立基线**单独跑一条净值曲线用于对比（§8.2）。是否融合为复合信号，作为后续实验变量。

---

## 8. 评估（对照九条世界线）

### 8.1 净值与成本分解

逐日输出：总净值、毛收益、累计摩擦成本、换手率、持仓向量 $w_t$。

### 8.2 九线对比

把每条世界线（WL1 台海 / WL7 AI 算法挤兑 / …）作为未来行情分别回放，输出每条世界线下的：
- 年化收益、Sharpe、最大回撤
- 总换手率、累计摩擦占比
- 关键事件日的调仓行为（如 WL1 2028.4 闪电战前后是否减仓科技/加仓黄金）

### 8.3 两条净值曲线

1. **AlphaCrafter 主曲线**（含新闻 + 换手惩罚）
2. **FactorMiner 截面排名曲线**（纯量价 + turnover_aware）

对比二者在不同世界线下的表现差异，验证"宏观常识 vs 纯数理"两条路径各自的价值。

---

## 9. 执行步骤清单

1. **下载框架**（当前阶段）：
   - 下载 AlphaCrafter、FactorMiner 到 `agent-framework/`（用 ghfast.top 加速 + setsid 断点续传 wget）。
   - 记录来源与版本到 `PROVENANCE.md`。
2. **数据下载**：拉取 20 项基准数据（15 可交易 + 5 只读参考）2020.01.01-2026.07.15 真实日频 OHLCV；2026.07.16 起只使用虚构世界线生成数据。
3. **数据格式化**：写 `data_processor/` 把各源对齐成两框架所需的统一宽表。
4. **AlphaCrafter 改造**：按 §4 改 Trader 优化器（加 L1 换手惩罚，cost=0.0003）。
5. **FactorMiner 改造**：按 §5 配 pure_price + 激活 turnover_aware。
6. **搭沙箱**：按 §6 实现网络隔离与时间游标。
7. **写调度器**：按 §7 实现日频主循环与摩擦结算。
8. **冷启动**：用 2020-2026 数据初始化两框架状态/因子库/经验记忆。
9. **滚动跑批**：2026.07.16 → 2035.12.31，逐九条世界线回放。
10. **出报告**：§8 指标与净值曲线写入 `report-and-output/`。

---

## 10. 开放问题与风险

- **gamma 调参**：换手惩罚权重过强会让 agent 几乎不交易（欠拟合），过弱则多动症复发。建议先用 WL7（震荡市）做网格搜索。
- **官方代码可用性**：两篇论文较新（2026.02 / 2026.05），若官方尚未开源，需评估社区实现与论文一致性，可能需要按论文复现关键模块（Trader 优化器、turnover_aware 钩子）。
- **算力**：日频 × 约 9.5 年 × 20 项输入 × 9 世界线 × 两个 LLM 框架，LLM 调用量巨大；FactorMiner 可降频到月末触发以控成本。
- **债券/汇率/波动率的"价格"处理**：收益率类资产（US10Y/CN10Y/VIX）需转换为可交易的"价格序列"或收益序列，再进入因子算子。
- **加密货币 7×24 交易日历**与权益日历不对齐，需统一到"权益交易日"对齐切片。

---

## 11. 当前进度（2026-07-20 实时跟踪）

> 本节为执行过程中的实时状态记录，与上文前瞻计划对照。项目目录已由 `trade-agent-research` 改名为 `trade-agent-benchmark`。

### 11.1 已完成

| # | 项目 | 落点（文件） | 状态 |
|---|------|------------|------|
| 1 | 框架获取 | `AlphaCrafter/`、`FactorMiner/` | ✅ 下载完成（来源见 §3.1） |
| 2 | 20 项基准宇宙单一事实源（15 可交易 + 5 只读参考） | `ASSETS.yaml`（含 baseline_date/warmup_start/online_end/friction_bps） | ✅ |
| 3 | AC 摩擦小修 | `AlphaCrafter/alphacrafter/sim/exchange_a.py:47-49`、`exchange_us.py:47-49`（commission_rate=0.0001 + slippage_rate=0.0002，买卖对称，覆盖开/平/做空/部分成交） | ✅ |
| 4 | FM 摩擦小修 | `FactorMiner/factorminer/configs/default.yaml:261-265`（execution.cost_bps=3.0；portfolio.py 按 cost_bps/10000×换手 扣净收益；admission.turnover_penalty=0.05 保留） | ✅ |
| 5 | 数据适配器 | `adapters/build_inputs.py`（230 行：规范长表 → AC session[stock_data+date.json+account.json+每月1日新闻] + FM panel.parquet + walkforward.yaml） | ✅ 代码就绪 |
| 6 | 前向调度器 | `scheduler/walk_forward.py`（205 行：dryrun/ac/fm/both 四模式、防穿越切片 panel_t、月首新闻日志、FM 频率 daily/monthly 可调） | ✅ 代码就绪 |

### 11.2 与原计划的偏差（按 §10 末尾规则显式记录）

- **§4.2「Trader 优化器加 L1 换手惩罚」→ 实际改为交易所层摩擦**：未改 Trader 目标函数，而是在 `exchange_a/us.py` 的成交环节直接收 1bp 佣金 + 2bp 滑点（合计单边 3bps）。两者等价地制造"不敏感带"——当调仓潜在收益 < 3bps 时净额为负，agent 自然不动；且交易所层实现更贴近实盘结算。若后续需要更激进的换手抑制，再在 Trader 目标函数叠加 γ‖w_t−w_{t-1}‖₁。
- **§5.2「激活 turnover_aware.py 钩子」→ 实际用 FM 原生 `execution.cost_bps` 旋钮**：FactorMiner 真实代码（非 refer.md 转述）的摩擦机制是 `evaluation/portfolio.py` 的 `transaction_cost_bps`，按 `cost_bps/10000 × 换手` 从因子回测净收益扣除；配合 `admission.turnover_penalty=0.05` 作评分层选择压力。无需自写 `turnover_aware.py`。
- **§9.3「data_processor 统一宽表」→ 实际为 `adapters/build_inputs.py` + `ASSETS.yaml`**：用一张长表（date,asset_id,OHLCV[,amount]）作单一输入契约，由适配器分别生成两框架格式，比"宽表"更贴合 FM 的长表 loader。

### 11.3 接下来要做（按优先级）

1. **真实数据落盘**（最高优先）：
   - 2020.01.01–2026.07.15 warm-up 真实日频 OHLCV（20 项基准数据）→ `data-prepare/asset-daily-data/`。
   - 2026.07.16–2035.12.31 前向未来价：由 `data-prepare/wordline-simple/wordline1-8.md` 的阶段终点表插值出日频路径。
   - 拼成规范长表 `panel.parquet`（csv 亦可）。
2. **运行环境**：`.venv` 装 `pandas/numpy/pyarrow/pyyaml` + 两框架各自依赖（AC: openai/dotenv/cvxpy/yaml；FM: click/xgboost 等）+ 配 LLM API Key。系统 Python 3.14 当前无 pip/pandas，需先解决环境（`ensurepip` 不可用，可能需 `apt install python3-pandas python3-pip` 或用 `uv`）。
3. **dryrun 全链路验证**：`python -m adapters.build_inputs --panel … --assets ASSETS.yaml …` 生成 session；`python -m scheduler.walk_forward --session wf19 --mode dryrun` 跑通游标推进 + 切片 + 月首新闻日志（无 LLM、无数据依赖即可验证逻辑）。
4. **live 接线**：`--mode ac/fm/both` 实跑；补全 `scheduler/walk_forward.py` 的 FM TODO——解析 `top_formulaic_alphas.json`、在 `panel_t` 上算截面分 → top 1/3 等权做多。
5. **评估层（§8）**：净值/成本分解、九世界线对比、两 leg 净值曲线 → `report-and-output/`。
6. **清理冗余**：`integration/`（早期版本）已被 `adapters/`+`scheduler/` 取代，验证新链路无误后删除。

---

*附：本计划锁定 refer.md 第 641-655 行"日频完整计划"的最终方案。任何偏离"原生日频 + 统一 3bps + 两处最小改造"的改动，需在此文件显式记录并说明理由。*
