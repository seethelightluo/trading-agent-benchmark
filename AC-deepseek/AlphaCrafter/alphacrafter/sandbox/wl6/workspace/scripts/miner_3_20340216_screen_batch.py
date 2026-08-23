"""Batch screen of new factor candidates.

Candidates (all long-only-orientable, aim additive to library):
 A. ma_dist_rz_20: (close - sma20)/rolling_std(ret,20)*sqrt(20) -- risk-adjusted price distance
 B. vol_scaled_mom_20: mom20 / rv20  (risk-scaled momentum)
 C. breakout_40: close/rolling_max(close,40) - 1   (proximity to 40d high)
 D. down_frac_20: fraction of up days > 20d avg (breadth of uptrend)
 E. amihud_illiq_20: mean(|ret|/volume) * 1e6 over 20d (liquidity)
 F. ret_above_sma_cond: (close/sma20 -1) yes/ no signed by trend quality
"""
from factor_validation_lib import load_panel, ic_analysis, print_report
import pandas as pd
import numpy as np

panel = load_panel()
ret = panel.pct_change()

cands = {}

# B: risk-scaled momentum
mom20 = panel / panel.shift(20) - 1.0
rv20 = ret.rolling(20).std()
cands["vol_scaled_mom_20"] = (mom20 / rv20.replace(0, np.nan)).rank(axis=1, pct=True)

# A: price distance from sma20 risk-normalized
sma20 = panel.rolling(20).mean()
cands["ma_dist_rz_20"] = ((panel - sma20) / (rv20 * np.sqrt(20))).replace([np.inf, -np.inf], np.nan)

# C: breakout proximity
cands["breakout_40"] = (panel / panel.rolling(40).max() - 1.0)

# D: breadth up days 20d minus 0.5
up = (ret > 0).rolling(20).mean()
cands["breadth_20"] = up - 0.5

# E: amihud illiquidity (higher = illiquid)
illiq = (ret.abs() / panel.rolling(10).max()).rolling(20).mean()
cands["amihud_illiq_20"] = -illiq

for name, f in cands.items():
    res = ic_analysis(f, panel, horizon=10, label=name)
    print_report(res)
    print()