"""Sanity check: recompute persisted factor metrics with shared validation framework."""
import sys
sys.path.insert(0, "scripts")
import pandas as pd
from factor_validation_lib import load_panel, ic_analysis, print_report

panel = load_panel()  # through visible data
print("panel shape:", panel.shape, "| dates:", panel.index.min().date(), "->", panel.index.max().date())
print("n assets:", panel.shape[1])
print(panel.columns.tolist())

# Persisted factor replications
def mom(close, lookback, skip):
    return close.shift(skip) / close.shift(skip + lookback) - 1.0

for lb, sk, fid in [(10, 5, "mom_10d_skip5"), (120, 5, "mom_120d_skip5")]:
    f = mom(panel, lb, sk)
    res = ic_analysis(f, panel, horizon=10, label=fid)
    print_report(res)

# vol of vol 20x60
import numpy as np
ret = panel.pct_change()
f_vov = ret.rolling(20).std().rolling(60).std()
res = ic_analysis(f_vov, panel, horizon=10, label="vol_of_vol20x60")
print_report(res)
