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
| 2 | `uv venv .venv` + 装 pandas/requests/akshare | ⏳ 进行中（后台） |
| 3 | 写 `data-prepare/fetch_daily_data.py`（三源 + 重试 + 宽表 + COVERAGE） | ⏳ 待写 |
| 4 | 运行抓取，产出 `asset-daily-data/` | ⏳ 待跑 |
| 5 | 校验：每资产日期范围/行数/缺失，与基线 2026-07 价比对 | ⏳ 待做 |
| 6 | 提交并推送到 GitHub（SSH：`/home/lxx/id_ed25519`） | ⏳ 待做 |

## 六、注意事项

- 本机走 Clash TUN 代理：**国际站（Binance/Yahoo/FRED/astral.sh）正常**；**国内 apt 源 InRelease 过代理 SSL 会断流**（aliyun/tuna 的 .deb 仍可下，故 git 能装）。数据抓取走国际站，不受影响。
- akshare 依赖较多，安装偏慢；若失败则中债10Y 改直连 chinabond/eastmoney API。
- 宽表非交易日空值是预期行为，非缺失。
