# plan.md — 世界线基准数据宇宙与 Agent 执行计划

> 创建：2026-07-20　|　目标来源：`data-prepare/wordline-simple/wordline1-9.md` + 统一基线价格表

---

## 一、总目标

为 9 条世界线准备基准数据宇宙：**15 项可交易资产 + 5 项只读指数/宏观参考**。真实历史严格截至
**2026-07-15**；从 **2026-07-16** 起的数据均为虚构世界线数据。该数据用于因子挖掘、回测与 Agent
决策，资产定义以 `data-prepare/asset_spec.py` 和 `agent-framework/ASSETS.yaml` 为单一事实源。

## 二、基准数据宇宙（15 可交易 + 5 只读参考）

| 角色 | 类别 | 数量 | 资产 ID |
|---|---|---:|---|
| 可交易 | 权益 | 8 | `000300.SH`, `SPX`, `HSI`, `N225`, `SX5E`, `000688.SH`, `SOX`, `NDX` |
| 可交易 | 商品 | 3 | `XAU`, `COPPER`, `WTI` |
| 可交易 | 加密 | 2 | `BTC`, `ETH` |
| 可交易 | 债券 | 2 | `US10Y`, `CN10Y` |
| 只读参考 | 指数/宏观/汇率 | 5 | `DXY`, `USDCNY`, `USDJPY`, `EURUSD`, `VIX` |

只读参考仅供 Agent 识别宏观状态或完成 USD 折算，不进入持仓权重向量。仓库仍保留旧世界线抓取产生的
`KOSPI`、`USDKRW`、`JP_SEMI_EQUIP` 文件用于历史兼容和审计，但它们**不属于基准 Agent 输入宇宙**。

## 三、数据源与口径

- 每个资产的当前来源优先级以 `data-prepare/asset_spec.py` 为准，抓取器按候选顺序先成先用，避免文档与代码映射漂移。
- **新浪 / 东方财富 / akshare / 中国银行**：覆盖权益、商品、债券、DXY 与汇率，是当前国内可达环境下的主要来源。
- **Binance**：提供 BTC/ETH 日线 O/H/L/C/Volume。
- **Yahoo Finance / CBOE**：作为国际源或长尾兜底；请求失败时由抓取器执行退避和来源切换。

## 四、输出结构

```
data-prepare/asset-daily-data/
├── <ASSET>.csv          # 每个资产一文件：date,open,high,low,close,volume(,adjclose)
├── all_close_wide.csv   # 日期(并集) × 资产收盘 宽表（非交易日为空）
├── COPPER_USD_PER_TON.csv  # 铜 USD/吨（HG=F × 2204.62262）
└── COVERAGE.md          # 每资产：起止日、行数、缺失天数、单位、来源
```

- 日期范围：`2020-01-01 ~ 2026-07-15`；各资产按自身交易日历（加密 365 天，股市工作日，各国节假日不同 → 宽表必有空值，正常）。
- 每资产 CSV 含各自交易日；宽表取所有日期并集。

## 五、执行步骤与进度

| # | 步骤 | 状态 |
|---|------|------|
| 0 | 安装 git / uv（uv 用官方安装器 → `~/.local/bin`） | ✅ 完成（git 2.53.0 / uv 0.11.29） |
| 1 | `git init` + `.gitignore` + user 配置 + 初始提交 | ✅ 完成（commit ea78c80，314 文件） |
| 2 | `uv venv .venv` + 装 pandas/requests/akshare/pyarrow/pyyaml | ✅ 完成 |
| 3 | 写 `data-prepare/fetch_daily_data.py`（多源 + 重试 + 宽表 + COVERAGE） | ✅ 完成（含 asset_spec/make_panel） |
| 4 | 运行抓取，产出 `asset-daily-data/`（20 基准 + 3 历史兼容产物） | ✅ 完成（Agent 真实历史截断至 2026-07-15） |
| 5 | 校验：每资产日期范围/行数/缺失，与基线 2026-07 价比对 | ✅ 完成（见 `COVERAGE.md`） |
| 6 | 提交并推送到 GitHub（SSH） | ✅ 完成（commit b086837+） |
| 7 | 合成在线世界线日频（2026-07-16 ~ 2035-12-31，逐 WL 末阶段） | ✅ 完成（`gen_worldline_online.py` + 9 条 WL，re-anchor） |
| 8 | build_inputs + walk_forward dryrun 全链路 | ✅ 完成（详见 [`RUN.md`](RUN.md)） |
| 9 | live LLM 前向跑批（ac/fm/both） | 🟡 AC 2-cycle 真烟测完成；全量待受控启动 |

## 六、注意事项

- 本机走 Clash TUN 代理：**国际站（Binance/Yahoo/FRED/astral.sh）正常**；**国内 apt 源 InRelease 过代理 SSL 会断流**（aliyun/tuna 的 .deb 仍可下，故 git 能装）。数据抓取走国际站，不受影响。
- akshare 依赖较多，安装偏慢；若失败则中债10Y 改直连 chinabond/eastmoney API。
- 宽表非交易日空值是预期行为，非缺失。

---

## 七、Agent 配置宇宙与计价基准（关键架构修订 · 2026-07-22）

> 本节细化第二节的同一套 **15 可交易 + 5 只读参考**口径，适用于 2026-07-16 之后的虚构前向持仓阶段。
> 已同步：`agent-framework/ASSETS.yaml`、`integration/asset_universe.py`、`gen_worldline_online.py`。

### 7.1 可交易持仓宇宙 = 15（汇率 / 波动率剔除为信号）

**汇率（DXY / USDCNY / USDJPY）与 VIX 不进入持仓权重向量 w_t**，降级为"宏观 / 状态信号特征"：
- VIX 是期权隐含波动率指数，**不可现货持有**（仅衍生品），塞进 MPT 协方差矩阵会量纲错配。
- 汇率是宏观传导媒介，非独立配置资产（本测试不做杠杆 FX）。

**可交易持仓（15 项，参与 ∑wᵢ = 1 与组合优化）：**

| 类别 | 数量 | 资产 |
|---|---:|---|
| 权益 | 8 | 沪深300、标普500、恒生、日经225、斯托克50、科创50、费城半导体、纳斯达克100 |
| 商品 | 3 | 黄金、铜、原油 |
| 加密 | 2 | BTC、ETH |
| 债券 | 2 | 美债10Y、中债10Y |

**宏观 / 状态信号（5 项，仅作 Agent 输入特征，不持仓）：** DXY、USDCNY、USDJPY、EURUSD、VIX
→ 供 AlphaCrafter Screener 判 Risk-On/Off；供 FactorMiner 生成跨资产因子（如 `Ts_Rank(VIX, 20)`）。其中 USDCNY/USDJPY/EURUSD 同时承担对应外币资产的 USD 折算（见 §7.2）。

> 合计 **20 = 15 可交易持仓 + 5 信号指标**。

### 7.2 统一以 USD 计价与结算（单一计价货币）

所有持仓资产统一收敛到**美元计价**，避免组合 NAV 与协方差矩阵 Σ 的货币量纲错配；日收益 R_t 已含"价格变动 + 汇率变动"的综合回报。

- **原生 USD（不变）**：SPX、NDX、SOX、XAU、Copper、WTI、BTC、ETH、US10Y
- **CNY 资产**：`P_USD(t) = P_CNY(t) / USDCNY(t)` —— 沪深300、科创50、中债10Y
- **JPY 资产**：`P_USD(t) = P_JPY(t) / USDJPY(t)` —— 日经225
- **EUR 资产**：`P_USD(t) = P_EUR(t) × EURUSD(t)` —— 斯托克50（注意 EURUSD 报价为「USD per EUR」，故相乘，与 CNY/JPY 相除相反）
- **HKD 资产**：恒生—— HKD 与美元**联系汇率（peg ≈ 7.80，区间 7.75–7.85）**，用常数折算 `P_USD = P_HKD / 7.80`，**无需抓取汇率**（误差 <1.3%，可忽略）。

### 7.3 前向结束时间 = 世界线末阶段真实结束日（不硬编码 2030）

融合 / 前向终止日 **不固定 2030-12-31**，而取**当前世界线最后一个阶段的真实结束日期**。
- 例：WL1 末阶段「新均衡形成（2030–2031）」→ 终点落在 2031，硬编码 2030-12-31 会**截断**该 WL。
- 不同世界线终点不同 → **逐 WL 读取 `wordlineN.md` 末阶段日期**，作为该 WL 前向窗口终点。
- 现有 `ASSETS.yaml`(`online_end`)、`integration/asset_universe.py`(`FORWARD_END`)、`gen_worldline_online.py` 中的 `2030-12-31` 需改为按 WL 动态读取。

### 7.4 待同步项（2026-07-22 已完成）

- [x] **补抓 EURUSD**（斯托克50 的 EUR→USD 折算，BOC 欧元/美元中间价比，国内源免 Yahoo）；恒生用 HKD-USD 联系汇率常数 ≈7.80，无需抓取。
- [x] `ASSETS.yaml` / `asset_universe.py` / `asset_spec.py` / `build_inputs.py`：宇宙拆为 **15 可交易 + 5 信号 = 20**，信号项（DXY/USDCNY/USDJPY/EURUSD/VIX）不进权重向量（AC watch_list 仅 15，信号入 index_data）。
- [x] 前向终点改为**逐 WL 动态**读取 `max(阶段 end_date)`（9 条 WL 均至 2035-12-31，原 2030-12-31 会截断 5 年）；`gen_worldline_online.py` 已重生成。
