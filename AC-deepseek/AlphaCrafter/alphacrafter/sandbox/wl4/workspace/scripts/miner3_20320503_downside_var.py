"""miner3_20320503_downside_var.py
Candidate: downside_var_60d = -percentile(daily log ret, 5) over 60d / realized vol(60d).
Normalized tail-loss intensity: how fat the left tail is relative to overall vol.
High value => severe crash risk (should be negative for forward returns);
low value => resilient asset.
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'scripts')
from miner3_harness import load_panel, factor_series, forward_returns, evaluate_factor

panel = load_panel()

def fn(s, df):
    lr = np.log(df['close']).diff()
    var5 = -lr.rolling(60).quantile(0.05)
    vol = lr.rolling(60).std()
    return var5 / vol

F = factor_series(panel, fn)
R10 = forward_returns(panel, h=10)
print('PANEL: %d instruments, F dates=%d' % (len(panel), F.shape[0]))
evaluate_factor(F, R10, name='downside_var_60d')
