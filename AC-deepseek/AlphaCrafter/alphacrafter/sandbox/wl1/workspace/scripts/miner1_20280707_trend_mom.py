"""miner_1 2028-07-07: Trend-conditioned momentum.

Motivation (from memory): momentum anchor mom_120d_skip5 keeps whipsawing
(top picks WTI/NDX/SOX/BTC/ETH crashed repeatedly in 2027-2028 risk-off).
Hypothesis: momentum predictive power is concentrated in assets whose price is
above their medium-term trend (MA60). Gating momentum by trend should reduce
whipsaw exposure while keeping uptrend momentum.

Construction (no lookahead):
  mom = close.shift(5)/close.shift(125) - 1           (120d momentum, 5d skip)
  trend = (close / close.rolling(60).mean()) - 1       (deviation from MA60)
  v1 (hard gate):  factor = mom * (trend > 0)
  v2 (soft scale): factor = mom * tanh(4*trend)         (smooth interaction)
  v3 (soft, gentle): factor = mom * (0.5 + 0.5*tanh(2*trend))
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
from miner1_lib_20280707 import (ASSETS, load_prices, factor_panel, ic_analysis,
                                 print_metrics, recent_metrics)


def make(mode):
    def fn(df, obs, a):
        close = df["close"]
        mom = close.shift(5) / close.shift(125) - 1.0
        ma60 = close.rolling(60).mean()
        trend = close / ma60 - 1.0
        if mode == "v1_hard":
            return mom.where(trend > 0, 0.0)
        if mode == "v2_soft":
            return mom * np.tanh(4.0 * trend)
        if mode == "v3_gentle":
            return mom * (0.5 + 0.5 * np.tanh(2.0 * trend))
        raise ValueError(mode)
    return fn


def main():
    frames = load_prices()
    print(f"assets={len(ASSETS)} days={len(frames['SPX'])} "
          f"last={frames['SPX'].index[-1]}")
    for mode in ["v1_hard", "v2_soft", "v3_gentle"]:
        panel, good = factor_panel(make(mode), frames)
        m = ic_analysis(panel, good, frames)
        print_metrics(f"trend_mom_120d [{mode}]", m)
        r = recent_metrics(panel, good, frames)
        print(f"  RECENT(2026-07-16..): adm IC={r['adm_ic']:.4f} "
              f"ICIR={r['adm_icir']:+.4f} hit={r['adm_hit']:.3f} n={r['adm_n_dates']}")


if __name__ == "__main__":
    main()
