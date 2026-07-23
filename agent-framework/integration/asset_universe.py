"""资产注册表（2026.07.16 统一基线）。

2026-07-22 关键修订（见 plan.md §7）：**15 可交易持仓 + 5 宏观/状态信号 = 20**。
  - 可交易（UNIVERSE，15）：进持仓权重向量 w_t，参与 ∑wᵢ=1 与组合优化。
  - 信号（SIGNALS，5：DXY/USDCNY/USDJPY/EURUSD/VIX）：仅作 Agent 输入特征，不持仓。
    其中 USDCNY/USDJPY/EURUSD 兼做对应外币资产的 USD 折算（见 TO_USD）。

来源：refer.md「统一基线价格表」+ plan.md §7。
所有可交易资产共用统一单边 3bps（1bp 佣金 + 2bp 滑点）摩擦，见 ``friction.py``。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Asset:
    asset_id: str          # FM panel 里的 asset_id；也是 AC 文件名 stem
    name_zh: str
    asset_class: str       # equity / commodity / crypto / bond / fx / vol
    baseline: float        # 2026.07.16 基准价（指数点位 / 美元 / % / 汇率）
    ccy: str = "usd"       # 原生计价货币；非 usd 者见 TO_USD 折算


# === 可交易持仓（15）：权益8 + 商品3 + 加密2 + 债券2 ===
UNIVERSE: List[Asset] = [
    # 权益
    Asset("000300.SH", "沪深300",     "equity",    4608.0,  "cny"),
    Asset("SPX",       "标普500",     "equity",    7534.0,  "usd"),
    Asset("HSI",       "恒生指数",    "equity",    24586.0, "hkd"),
    Asset("N225",      "日经225",     "equity",    68000.0, "jpy"),
    Asset("SX5E",      "斯托克50",    "equity",    5100.0,  "eur"),
    Asset("000688.SH", "科创50",      "equity",    1920.0,  "cny"),
    Asset("SOX",       "费城半导体",  "equity",    5800.0,  "usd"),
    Asset("NDX",       "纳斯达克100", "equity",    20500.0, "usd"),
    # 商品
    Asset("XAU",       "黄金",        "commodity", 4050.0,  "usd"),
    Asset("COPPER",    "LME铜",       "commodity", 13600.0, "usd"),
    Asset("WTI",       "原油WTI",     "commodity", 79.0,    "usd"),
    # 加密货币
    Asset("BTC",       "比特币",      "crypto",    64800.0, "usd"),
    Asset("ETH",       "以太坊",      "crypto",    1920.0,  "usd"),
    # 债券（收益率数值序列当 close）
    Asset("US10Y",     "美债10Y",     "bond",      4.30,    "usd"),
    Asset("CN10Y",     "中债10Y",     "bond",      2.20,    "cny"),
]

# === 宏观/状态信号（5）：仅特征，不持仓 ===
SIGNALS: List[Asset] = [
    Asset("DXY",    "美元指数",  "fx",  100.5, "usd"),
    Asset("USDCNY", "USD/CNY",   "fx",  6.78,  "usd"),
    Asset("USDJPY", "USD/JPY",   "fx",  162.0, "usd"),
    Asset("EURUSD", "EUR/USD",   "fx",  1.08,  "usd"),
    Asset("VIX",    "VIX",       "vol", 16.0,  "usd"),
]

# 快速查找表（可交易）
BY_ID: Dict[str, Asset] = {a.asset_id: a for a in UNIVERSE}
IDS: List[str] = [a.asset_id for a in UNIVERSE]              # 权重向量维度的 asset_id（15）
AC_SYMBOLS: List[str] = IDS                                  # AC 文件名 = asset_id

# 信号（不进权重向量）
SIGNAL_IDS: List[str] = [a.asset_id for a in SIGNALS]
ALL_IDS: List[str] = IDS + SIGNAL_IDS                        # 数据层全量（20）

# === USD 折算（§7.2）：op div=P/rate, mul=P×rate；rate 为信号 asset_id 或常数 ===
# 未列出者 ccy=usd 原生 USD。HSI 用 HKD-USD 联系汇率常数 7.80（peg，无需抓取）。
TO_USD: Dict[str, dict] = {
    "000300.SH": dict(rate="USDCNY", op="div"),
    "000688.SH": dict(rate="USDCNY", op="div"),
    "CN10Y":     dict(rate="USDCNY", op="div"),
    "HSI":       dict(rate=7.80,     op="div"),
    "N225":      dict(rate="USDJPY", op="div"),
    "SX5E":      dict(rate="EURUSD", op="mul"),
}

# 时间线（与 refer.md / plan.md 一致）
HISTORY_START = "2020-01-01"   # warm-up 历史数据起点
BASELINE_DATE = "2026-07-16"   # 在线滚动迭代起点（基线日）
# 前向终点：不硬编码 2030；取世界线末阶段真实结束日。9 条 WL 末阶段均结束于 2035-12-31。
FORWARD_END   = "2035-12-31"   # 全局上限；实际逐 WL 由 gen_worldline_online 取 max(阶段 end_date)


def baseline_table() -> str:
    """返回 markdown 基线表（15 可交易 + 5 信号），便于写入报告。"""
    rows = ["| asset_id | 名称 | 类别 | 原生货币 | 2026.07.16 基准 | 角色 |",
            "|---|---|---|---|---:|---|"]
    for a in UNIVERSE:
        rows.append(f"| {a.asset_id} | {a.name_zh} | {a.asset_class} | {a.ccy} | {a.baseline} | 可交易 |")
    for a in SIGNALS:
        rows.append(f"| {a.asset_id} | {a.name_zh} | {a.asset_class} | {a.ccy} | {a.baseline} | 信号 |")
    return "\n".join(rows)


if __name__ == "__main__":
    print(f"{len(UNIVERSE)} tradable + {len(SIGNALS)} signals = {len(ALL_IDS)} assets.")
    print(baseline_table())
