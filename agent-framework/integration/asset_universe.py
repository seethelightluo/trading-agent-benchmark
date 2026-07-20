"""19 类可交易资产注册表（2026.07.16 统一基线）。

来源：refer.md「统一基线价格表」。把跨大类资产（权益指数/商品/加密/债券/汇率/波动率）
统一映射为两套框架都能消费的符号：
  - AlphaCrafter：每资产一个 ``index_data/<symbol>.csv``（跨类资产无 PE/PS/PB/DYR 基本面，
    统一作为「指数」喂入；债券/汇率/波动率用其数值序列当 close）。
  - FactorMiner ：长表 ``panel`` 里的 ``asset_id``。

所有资产共用统一单边 3bps（1bp 佣金 + 2bp 滑点）摩擦，见 ``friction.py``。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Asset:
    asset_id: str          # FM panel 里的 asset_id；也是 AC 的 index_data 文件名 stem
    name_zh: str
    asset_class: str       # equity / commodity / crypto / bond / fx / vol
    baseline: float        # 2026.07.16 基准价（指数点位 / 美元 / % / 汇率）
    ac_symbol: str         # AC 文件名（与 asset_id 一致，显式列出便于核对）


# 顺序即截面排序的默认 universe 顺序
UNIVERSE: List[Asset] = [
    # 权益
    Asset("000300.SH", "沪深300",     "equity",    4608.0,  "000300.SH"),
    Asset("SPX",       "标普500",     "equity",    7534.0,  "SPX"),
    Asset("HSI",       "恒生指数",    "equity",    24586.0, "HSI"),
    Asset("N225",      "日经225",     "equity",    68000.0, "N225"),
    Asset("SX5E",      "斯托克50",    "equity",    5100.0,  "SX5E"),
    Asset("000688.SH", "科创50",      "equity",    1920.0,  "000688.SH"),
    Asset("SOX",       "费城半导体",  "equity",    5800.0,  "SOX"),
    Asset("NDX",       "纳斯达克100", "equity",    20500.0, "NDX"),
    # 商品
    Asset("XAU",       "黄金",        "commodity", 4050.0,  "XAU"),
    Asset("COPPER",    "LME铜",       "commodity", 13600.0, "COPPER"),
    Asset("WTI",       "原油WTI",     "commodity", 79.0,    "WTI"),
    # 加密货币
    Asset("BTC",       "比特币",      "crypto",    64800.0, "BTC"),
    Asset("ETH",       "以太坊",      "crypto",    1920.0,  "ETH"),
    # 债券（收益率数值序列当 close；做多=收益率头寸的代理）
    Asset("US10Y",     "美债10Y",     "bond",      4.30,    "US10Y"),
    Asset("CN10Y",     "中债10Y",     "bond",      2.20,    "CN10Y"),
    # 汇率
    Asset("DXY",       "美元指数",    "fx",        100.5,   "DXY"),
    Asset("USDCNY",    "USD/CNY",     "fx",        6.78,    "USDCNY"),
    Asset("USDJPY",    "USD/JPY",     "fx",        162.0,   "USDJPY"),
    # 波动率
    Asset("VIX",       "VIX",         "vol",       16.0,    "VIX"),
]

# 快速查找表
BY_ID: Dict[str, Asset] = {a.asset_id: a for a in UNIVERSE}
IDS: List[str] = [a.asset_id for a in UNIVERSE]
AC_SYMBOLS: List[str] = [a.ac_symbol for a in UNIVERSE]

# 时间线（与 refer.md 一致）
HISTORY_START = "2020-01-01"   # warm-up 历史数据起点
BASELINE_DATE = "2026-07-16"   # 在线滚动迭代起点（基线日）
FORWARD_END   = "2030-12-31"   # 前向走步终点


def baseline_table() -> str:
    """返回 markdown 基线表，便于写入报告。"""
    rows = ["| asset_id | 名称 | 类别 | 2026.07.16 基准 |", "|---|---|---|---:|"]
    for a in UNIVERSE:
        rows.append(f"| {a.asset_id} | {a.name_zh} | {a.asset_class} | {a.baseline} |")
    return "\n".join(rows)


if __name__ == "__main__":
    print(f"{len(UNIVERSE)} assets registered.")
    print(baseline_table())
