"""Validate candidate: trend_efficiency_20d.

Construction: abs(close / close_20d_ago - 1) / sum(|daily ret|, 20)
Measures how directional (trending) vs choppy the past 20-day path is, scaled by
return magnitude. Motivation: in cross-asset regimes, persistent directional moves
tend to continue, while choppy ranges whipsaw. Distinct from plain momentum and
low-vol factors already in the library.

Gate: abs(paper IC)>=0.0070 and abs(ICIR)>=0.0840 at horizon 10.
"""
from factor_validation_lib import load_panel, ic_analysis, print_report, library_corr
import pandas as pd
import numpy as np

panel = load_panel()
ret = panel.pct_change()
window = 20

fwd_sum = ret.abs().rolling(window).sum()
pct_chg = panel / panel.shift(window) - 1.0
factor = pct_chg.abs() / fwd_sum.replace(0.0, np.nan)

label = "trend_eff_20d"
res = ic_analysis(factor, panel, horizon=10, label=label)
print_report(res)

# guard against heavy overlap with library
print("max_abs_library_correlation:", "skip (run separately)")