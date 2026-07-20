"""统一交易摩擦成本（全资产单边 3bps = 1bp 佣金 + 2bp 滑点）。

refer.md 最终方案：取消按月阻断、恢复原生日频，用「底层数理约束」倒逼 agent 学会
「静若处子、动若脱兔」。本模块集中维护摩擦常数与两个落地点的配置写入：

  1. AlphaCrafter 交易所（已在 sim/exchange_a.py、sim/exchange_us.py 内嵌）：
       - commission_rate = 0.0001  (1bp)
       - slippage_rate   = 0.0002  (2bp，买高卖低)
     成交价 = close × (1 ± slippage)，另按成交额收 commission。

  2. FactorMiner 评分/回测：
       - execution.cost_bps = 3.0  (已在 configs/default.yaml 写入)
       - portfolio.py 按 cost_bps/10000 × 换手 从因子净收益扣除。
       - admission.turnover_penalty = 0.05（评分层选择压力，抑制高换手因子）。

本文件仅作单一事实来源 + 校验工具，不重复硬编码常数。
"""
from __future__ import annotations

COMMISSION_BPS = 1.0   # 单边佣金
SLIPPAGE_BPS = 2.0     # 单边滑点
TOTAL_SINGLE_SIDE_BPS = COMMISSION_BPS + SLIPPAGE_BPS   # = 3.0 bps 单边
ROUND_TRIP_BPS = 2 * TOTAL_SINGLE_SIDE_BPS             # = 6.0 bps 往返

# 小数形式（直接用于乘法）
COMMISSION_RATE = COMMISSION_BPS / 10000.0   # 0.0001
SLIPPAGE_RATE = SLIPPAGE_BPS / 10000.0       # 0.0002
COST_BPS = TOTAL_SINGLE_SIDE_BPS             # FactorMiner execution.cost_bps


def apply_slippage(close: float, is_buy: bool) -> float:
    """对收盘价施加滑点：买单成交价上浮，卖单成交价下挫。"""
    factor = 1 + SLIPPAGE_RATE if is_buy else 1 - SLIPPAGE_RATE
    return round(close * factor, 4)


def turnover_cost(weight_prev: float, weight_new: float) -> float:
    """给定单资产相邻两期权重，返回调仓产生的单边摩擦成本（占净资产比例）。

    用于前向调度器在统一账本里核算两 leg 的净 NAV，与交易所内部扣费口径一致。
    """
    turnover = abs(weight_new - weight_prev)          # 单边换手
    return turnover * (COST_BPS / 10000.0)


if __name__ == "__main__":
    print(f"commission = {COMMISSION_BPS}bps, slippage = {SLIPPAGE_BPS}bps")
    print(f"single-side = {TOTAL_SINGLE_SIDE_BPS}bps, round-trip = {ROUND_TRIP_BPS}bps")
    print(f"commission_rate = {COMMISSION_RATE}, slippage_rate = {SLIPPAGE_RATE}")
    print(f"FactorMiner execution.cost_bps = {COST_BPS}")
