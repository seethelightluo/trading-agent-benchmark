"""Sanity check: re-validate existing library factors with miner3 framework."""
import sys
sys.path.insert(0, 'scripts')
import pandas as pd
from miner3_lib import build_panel, run_validation

prices = build_panel()
panel = pd.DataFrame(prices)
ret = panel.pct_change()

# 1) mom_10d_skip5: close.shift(5)/close.shift(15)-1
def mom10(p):
    return (p.shift(5) / p.shift(15) - 1.0)

# 2) vol_of_vol20x60: std(pct_change,20).rolling(60).std()
def vov(p):
    r = p.pct_change()
    return r.rolling(20).std().rolling(60).std()

run_validation(mom10, 'mom_10d_skip5_RE', 'sanity')
run_validation(vov, 'vol_of_vol20x60_RE', 'sanity')
