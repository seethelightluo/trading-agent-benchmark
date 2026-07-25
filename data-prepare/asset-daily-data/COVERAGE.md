# COVERAGE — 资产日频数据覆盖与校验

> 生成：2026-07-22　|　范围：2020-01-01 ~ 抓取末端　|　原始真实数据锚点含 2026-07-16；Agent 研究截止 2026-07-15

## 1. 每资产覆盖

| asset_id | 名称 | 类别 | 单位 | 所用源 | 起始 | 末端 | 行数 | 基线(2026-07-16) | 基线比对 | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|
| `EURUSD` | 欧元/美元 | fx | 汇率 | BOC 欧元/美元 cross | 2020-01-02 | 2026-07-22 | 1582 | 1.08 | ⚠️ 1.144 vs 1.08 (+5.9%) | BOC 欧元/美元交叉；用于斯托克50 USD 折算（P_usd=P×EURUSD） |

## 2. 单位与口径说明

- 指数/商品/汇率/波动率：`close` 为源端原生报价（指数点 / USD-oz / USD-桶 / 汇率 / 指数）。
- **US10Y/CN10Y**：`close` 为收益率百分数（4.30 = 4.30%）。ASSETS.yaml 基线为小数（0.043），比对时 ×100。
- **COPPER**：源 HG=F 为 USD/lb，基线表为 USD/吨；另出 `COPPER_USD_PER_TON.csv`（×2204.62262）。
- **SOX/债券/汇率无原生 OHLCV**：open/high/low 用 close 填充、volume=0；不影响日频收益计算。
- **KOSPI/USDKRW/JP_SEMI_EQUIP**：WL3/WL5 特有，非 19 基准资产；CSV 落盘，默认不进 panel。
- **基线比对说明**：基线来自 refer.md 的「实际市场估计」，与 2026-07 真实价可能有偏差（尤其 SOX/NDX/CN10Y 估计显著偏离真实）；以真实抓取价为准。
- 宽表 `all_close_wide.csv` 取日期并集，各市场交易日历/节假日不同 → 空值为预期，非缺失。

## 3. 复跑

```bash
.venv/bin/python data-prepare/fetch_daily_data.py            # 复用已落盘 CSV
.venv/bin/python data-prepare/fetch_daily_data.py --force    # 全量重抓
.venv/bin/python data-prepare/fetch_daily_data.py --only VIX,USDJPY  # 补抓缺口
```
