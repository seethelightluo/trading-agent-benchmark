# plan.md — 世界线可交易资产每日数据：抓取 + 虚构生成

> 子模块：`data-prepare`　|　创建：2026-07-20　|　更新：2026-07-22
> 资产口径来源：`data-prepare/wordline-simple/wordline1-9.md` + 统一基线价格表
> 范围：① 2020-01-01 ~ 2026-07-15 **真实历史**抓取（第二~六节）；② 2026-07-16 ~ 2030-12-31 **虚构（前向世界线）生成**（第七节）

---

## 一、总目标

为 9 条世界线中出现的全部**可交易资产**，准备两段数据：
1. **真实历史**（2020-01-01 ~ 2026-07-15）：从 Binance/Yahoo/akshare 抓取，作为 warm-up 冷启动与因子/回测基础。
2. **虚构前向**（2026-07-16 ~ 2030-12-31）：以世界线月末目标估值为锚，用「AI 解析 News → 动态波动率 → 几何布朗桥」生成日频路径，作为 Agent 前向走步的"未来"。

以 `wordline` 基线表为资产口径，单位与基线对齐。

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
| 2 | `uv venv .venv` + 装 pandas/requests/akshare | ✅ 完成（pandas 3.0.3 / requests 2.34.2 / akshare 1.18.64） |
| 3 | 写 `data-prepare/fetch_daily_data.py`（三源 + 重试 + 宽表 + COVERAGE） | ⏳ 待写 |
| 4 | 运行抓取，产出 `asset-daily-data/` | ⏳ 待跑 |
| 5 | 校验：每资产日期范围/行数/缺失，与基线 2026-07 价比对 | ⏳ 待做 |
| 6 | 提交并推送到 GitHub（SSH：`/home/lxx/id_ed25519`，用户 `seethelightluo`） | ⏳ 待做 |

## 六、注意事项

- 本机走 Clash TUN 代理：**国际站（Binance/Yahoo/FRED/astral.sh）正常**；**国内 apt 源 InRelease 过代理 SSL 会断流**（aliyun/tuna 的 .deb 仍可下，故 git 能装）。数据抓取走国际站，不受影响。
- akshare 依赖较多；若失败则中债10Y 改直连 chinabond/eastmoney API。
- 宽表非交易日空值是预期行为，非缺失。

---

## 七、2026-07-16 ~ 2030-12-31 虚构（前向世界线）每日数据生成

> 本节针对**虚构数据**（synthetic / 前向世界线未来）。生成 agent 必须遵循本节规则。
> 真实历史数据（第二~六节）只抓取、不生成；虚构数据只生成、且不得回灌进真实段。

### 7.1 方法选择：强烈推荐「AI 解析 News → 几何布朗桥（GBB）」，**不**用 statsmodels / arch

在本场景下，直接用 `statsmodels` / `arch`（如 GARCH）拟合-外推会遇到根本性阻力，甚至无法完成任务。
**采用「AI + 几何布朗桥（Geometric Brownian Bridge, GBB）」**，原因有二：

**优势 1：完美的"靶向控制"能力（月末精准到达估值）**
- 布朗桥的数学本质是**两端固定**的插值模型。无论中间的波动率被 AI 调到多高，公式中的方差收敛机制都会强制价格在**月末那一天精准收敛到世界线给定的目标估值**。
- 这保证了大趋势（宏观估值）绝对不会因为微观的剧烈波动而脱轨。GARCH 等外推模型无法保证命中指定终点，会与 `wordline1-9.md` 的月末估值表冲突。

**优势 2：AI 天然擅长情感与事件冲击量化（非线性）**
- News 对波动率的影响（事件驱动型波动）高度非线性：
  - 利空且不确定（如"公司遭监管突击调查，面临退市风险"）→ AI 识别恐慌，输出高波动率（如 80%）。
  - 平稳或符合预期（如"本月例行董事会，无重大决议"）→ AI 识别为平静期，输出低波动率（如 15%）。
- `arch`/GARCH 只能从**历史已实现波动率**外推，无法消费**外部非结构化文本**；要让它读 News，仍需先做情感量化，不如直接让 LLM 输出波动率。

> 结论：波动率由「AI 读 News 生成」，价格路径由「几何布朗桥保证命中月末锚点」。两者分工，缺一不可。

### 7.2 生成流水线（Pipeline）

```
月末目标估值（wordline 表）          每月 News 文本
        │                                │
        │  起点锚 / 终点锚                ▼
        │                    [LLM 情感打分器] prompt → 输出年化波动率 σ（如 0.1~1.0）
        │                                │ volatility=σ
        ▼                                ▼
   generate_geometric_brownian_bridge(start=S, end=月末目标, vol=σ, horizon=当月交易日数, seed=…)
                                │
                                ▼
                   该资产当月日频 OHLCV（两端严格命中锚点）
```

- 输入端：把当月 News 文本喂给 LLM。
- AI 映射：Prompt 要求 LLM 充当"金融情绪打分器"，输出一个具体的**年化波动率数值**（建议区间 0.1~1.0，并给出判断依据）。
- 生成端：把动态波动率传入 `generate_geometric_brownian_bridge(...)` 作为该月 `volatility` 参数。

### 7.3 虚构数据生成规则（生成 agent 必须遵循）

1. **只用 GBB，不用统计外推**：禁止用 `statsmodels`/`arch`/GARCH 直接外推生成前向价格；波动率可参考历史，但路径必须由几何布朗桥生成。
2. **两端严格锚定**：每月布朗桥起点 = 上一月末实际收盘（首月 = 2026-07-16 基线）；终点 = `wordline1-9.md` 该世界线该资产当月末目标估值。方差收敛机制必须保证**末日收盘 = 终点锚**（容差 ≤ 1e-6，或显式置终）。
3. **波动率由 AI 读 News 生成**：`σ` 必须来自当月 News 文本的 LLM 打分，不得用常数、不得静默套用历史均值。News 缺失时，回落到该资产近 60 日已实现波动率并记日志标注"无新闻回落"。
4. **逐世界线独立**：9 条世界线分别生成，月末锚点取各自 `wordlineN.md`；不同世界线之间不得串数据。
5. **可复现（定种子）**：每条路径用确定性种子，键为 `(world_line, asset, month)`；相同输入重生成必须逐 K 线一致。
6. **单位与基线对齐**：指数用点数、商品用 USD（铜另附 USD/吨）、加密用 USD、债券/汇率/VIX 用其数值序列；起价以 2026-07-16 统一基线表为准。
7. **严格防穿越**：虚构段（≥ 2026-07-16）不得写入真实历史段（≤ 2026-07-15）；下游 Agent 在第 t 天只见 `[2020-01-01, t]`，生成产物按时段分目录存放。
8. **OHLCV 合规**：O/H/L/C 须满足 `low ≤ open,close ≤ high`；非权益资产（债券/汇率/VIX）无原生 volume 时，按规则合成或置 0 并保持全段一致。
9. **合理波动幅度**：σ 的 AI 输出须落入合理区间（建议年化 0.1~1.0）；极端值（如战争级 VIX 飙升）需与该世界线叙事强度匹配，单日跳变不得超出布朗桥方差允许范围。
10. **产物与校验**：输出结构与第四节一致（`<ASSET>.csv` + 宽表 + `COVERAGE.md`），并额外生成 `generation_meta.json`（每资产每月：σ、News 摘要、种子、起止锚），便于审计与回溯。
