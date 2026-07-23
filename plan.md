# plan.md — 世界线可交易资产每日数据抓取与仓库托管

> 创建：2026-07-20　|　目标来源：`data-prepare/wordline-simple/wordline1-9.md` + 统一基线价格表

---

## 一、总目标

为 9 条世界线中出现的全部**可交易资产**，抓取 **2020-01-01 至 2026-07-15 的每日价格/收益率数据**，
作为后续因子挖掘、回测、Agent 决策的基础数据集。以 `wordline` 基线表为资产口径，单位与基线对齐。

## 二、资产清单（21 项 + 1 可选代理）

| # | 资产 | 中文 | 数据源 | Ticker / 代码 | 单位 | 备注 |
|---|------|------|--------|--------------|------|------|
| 1 | CSI300 | 沪深300 | Yahoo | `000300.SS` | 指数点 | |
| 2 | SP500 | 标普500 | Yahoo | `^GSPC` | 指数点 | |
| 3 | HSI | 恒生指数 | Yahoo | `^HSI` | 指数点 | |
| 4 | N225 | 日经225 | Yahoo | `^N225` | 指数点 | |
| 5 | SX5E | 欧洲斯托克50 | Yahoo | `^STOXX50E` | 指数点 | |
| 6 | STAR50 | 科创50 | Yahoo | `000688.SS` | 指数点 | |
| 7 | SOX | 费城半导体 | Yahoo | `^SOX` | 指数点 | |
| 8 | NDX | 纳斯达克100 | Yahoo | `^NDX` | 指数点 | |
| 9 | KOSPI | 韩国KOSPI | Yahoo | `^KS11` | 指数点 | WL3 特有 |
| 10 | GOLD | 黄金 | Yahoo | `GC=F` | USD/oz | COMEX 期货，≈XAU |
| 11 | COPPER | 铜 | Yahoo | `HG=F` | USD/lb | COMEX；附 USD/吨(×2204.62)列 |
| 12 | WTI | WTI原油 | Yahoo | `CL=F` | USD/桶 | |
| 13 | BTC | 比特币 | Binance | `BTCUSDT` | USD(USDT) | klines 日线 |
| 14 | ETH | 以太坊 | Binance | `ETHUSDT` | USD(USDT) | klines 日线 |
| 15 | US10Y | 美债10Y收益率 | Yahoo | `^TNX` | % | |
| 16 | CN10Y | 中债10Y收益率 | akshare | `bond_china_yield`(10年) | % | chinabond.com.cn |
| 17 | DXY | 美元指数 | Yahoo | `DX-Y.NYB` | 指数 | |
| 18 | USDCNY | 美元/人民币 | Yahoo | `CNY=X` | 汇率 | |
| 19 | USDJPY | 美元/日元 | Yahoo | `JPY=X` | 汇率 | WL5 特有 |
| 20 | USDKRW | 美元/韩元 | Yahoo | `KRW=X` | 汇率 | WL3 特有 |
| 21 | VIX | 波动率指数 | Yahoo | `^VIX` | 指数 | |
| 可选 | JP_SEMI_EQUIP | 日本半导体设备指数 | (代理) | `6857.T`+`8035.T` | — | WL5；无公开指数，用 Advantest/Tokyo Electron 等权代理，标注 |

## 三、数据源与口径

- **Binance**（加密）：`GET /api/v3/klines?symbol=&interval=1d&startTime=&limit=1000`，按 1000 根分页，O/H/L/C/Volume。实测可用，3 段即可覆盖。
- **Yahoo Finance**（股/汇/商/波）：`query1.finance.yahoo.com/v8/finance/chart/{sym}?period1=&period2=&interval=1d`，含 adjclose。已验证全部 ticker 可用。带 UA + 退避重试（429）。
- **akshare**（中债10Y）：`ak.bond_china_yield(start_date, end_date)` 取「10 年」列。Stooq 已加 JS 反爬，弃用。

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
| 4 | 运行抓取，产出 `asset-daily-data/`（19 基准 + 3 世界线特有） | ✅ 完成（2020-01-02 ~ 2026-07-21，0 NaN） |
| 5 | 校验：每资产日期范围/行数/缺失，与基线 2026-07 价比对 | ✅ 完成（见 `COVERAGE.md`） |
| 6 | 提交并推送到 GitHub（SSH） | ✅ 完成（commit b086837+） |
| 7 | 合成在线世界线日频（2026-07-17 ~ 2035-12-31，逐 WL 末阶段） | ✅ 完成（`gen_worldline_online.py` + 9 条 WL，re-anchor） |
| 8 | build_inputs + walk_forward dryrun 全链路 | ✅ 完成（详见 [`RUN.md`](RUN.md)） |
| 9 | live LLM 前向跑批（ac/fm/both） | ⛔ 待 API Key（dryrun 已验证逻辑） |

## 六、注意事项

- 本机走 Clash TUN 代理：**国际站（Binance/Yahoo/FRED/astral.sh）正常**；**国内 apt 源 InRelease 过代理 SSL 会断流**（aliyun/tuna 的 .deb 仍可下，故 git 能装）。数据抓取走国际站，不受影响。
- akshare 依赖较多，安装偏慢；若失败则中债10Y 改直连 chinabond/eastmoney API。
- 宽表非交易日空值是预期行为，非缺失。

---

## 七、Agent 配置宇宙与计价基准（关键架构修订 · 2026-07-22）

> 本节决定 **Agent 如何使用数据**，适用于 2026-07-16 之后的前向持仓阶段。
> 第二节"资产清单"是**数据抓取口径**（更宽，含信号与 WL 特有项）；本节是 **Agent 持仓配置口径**（更窄）。
> 需同步：`agent-framework/ASSETS.yaml`、`integration/asset_universe.py`、`gen_worldline_online.py`。

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
