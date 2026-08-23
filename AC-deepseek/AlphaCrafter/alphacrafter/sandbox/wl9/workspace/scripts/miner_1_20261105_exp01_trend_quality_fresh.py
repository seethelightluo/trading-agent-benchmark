"""miner_1 (2026-11-05): fresh trend-quality / risk-conditioned factor family.

Existing library already covers: momentum (10/120 skip5), volatility (vol_z, vol_of_vol,
kurt, skew, bb_width, rng_pos), macro-beta (VIX, DXY, USDJPY, CNY), streak, days_since_high,
ac1, kaufman_eff, vixreg conditional.

Idea: explore factor constructs that are NOT simple momentum/vol and avoid duplicating the
library, to keep library correlation low and add orthogonal alpha:
  A1 trend_consist_30 : fraction of co-directional daily moves over 30d (trend persistence,
                        independent of magnitude -> distinct from momentum)
  A2 clv_trend_20     : close-location-value (close in 20d range) * sign(20d trend)
  A3 sortino_20       : rolling 20d mean return / downside (neg-only) std  (risk-adjusted return)
  A4 high_retrace_20  : - (drawdown from 20d high) i.e. how far below recent high (mean reversion)
  A5 vix_conditioned_mom: momentum interacting with VIX regime gap
Gate on 15-asset universe: abs(IC)>=0.0070 & abs(ICIR)>=0.0840 @ h=10.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner3_20260730_harness import load_closes, evaluate, load_macro

def main():
    closes = load_closes()
    macro = load_macro()
    vix = macro.get("VIX")
    print(f"assets loaded: {len(closes)}  macro loaded: {len(macro)}  dates: "
          f"{min(len(s) for s in closes.values())}..{max(len(s) for s in closes.values())}")

    # A1: trend consistency (fraction of days with same sign as net window move)
    vals = {}
    for a, s in closes.items():
        r = s.pct_change()
        m30 = s.shift(1) / s.shift(31) - 1.0
        cons = (r.shift(1).fillna(0).rolling(30).apply(
            lambda x: float((np.sign(x) == np.sign(m30.loc[x.index[-1]])).mean()) if len(x) else np.nan,
            raw=False) if False else None)
        # simpler: fraction of up days where trend is up, down days where trend is down
        up = (r > 0).astype(float)
        trend_up = (m30 > 0).astype(float)
        agree = (up - trend_up).abs()  # 0 when agree, 1 when disagree
        vals[a] = (1 - agree.rolling(30).mean()).shift(0)
    evaluate(closes, vals, "A1 trend_consist_30 (fraction co-directional)")

    # A2: close-location-value * sign(20d momentum)
    vals = {}
    for a, s in closes.items():
        hi = s.rolling(20).max()
        lo = s.rolling(20).min()
        rng = (hi - lo).replace(0, np.nan)
        clv = (s - lo) / rng
        m20 = s / s.shift(20) - 1.0
        vals[a] = (clv * np.sign(m20)).shift(1)
    evaluate(closes, vals, "A2 clv_trend_20 (clv20*sign(mom20))")

    # A3: sortino-like risk-adjusted return 20d
    vals = {}
    for a, s in closes.items():
        r = s.pct_change()
        mu = r.rolling(20).mean()
        neg = r.clip(upper=0)
        dsd = neg.rolling(20).std().replace(0, np.nan)
        vals[a] = (mu / dsd).shift(1)
    evaluate(closes, vals, "A3 sortino_20 (mean/downside-std)")

    # A4: retracement from 20d high (mean-reversion distance)
    vals = {}
    for a, s in closes.items():
        hi = s.rolling(20).max()
        vals[a] = (-(hi / s - 1.0)).shift(1)
    evaluate(closes, vals, "A4 high_retrace_20 (-dist from 20d high)")

    # A5: VIX-conditioned momentum (mom10 times risk-regime)
    if vix is not None:
        vals = {}
        vixg = vix.pct_change().rolling(5).mean()
        regime = (vixg > 0).astype(float).replace(0, -1.0)  # +1 rising vol, -1 falling vol
        for a, s in closes.items():
            m10 = s / s.shift(10) - 1.0
            # align regime to this asset's calendar
            reg = regime.reindex(s.index).shift(1).fillna(-1.0)
            vals[a] = (m10.shift(1) * reg)
        evaluate(closes, vals, "A5 vix_cond_mom10 (mom10 * vol-regime sign)")

if __name__ == "__main__":
    main()
