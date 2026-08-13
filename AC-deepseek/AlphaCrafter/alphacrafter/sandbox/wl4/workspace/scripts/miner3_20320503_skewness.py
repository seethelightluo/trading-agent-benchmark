"""miner3_20320503_skewness.py
Candidate: realized_skew_60d = skewness of daily log returns over 60d.
Idea: positively skewed assets (occasional large up-moves, small steady downs)
tend to be rewarded; negative skew (crash-prone) assets tend to underperform.
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'scripts')
from miner3_harness import load_panel, factor_series, forward_returns, evaluate_factor

panel = load_panel()

def fn_skew(s, df):
    lr = np.log(df['close']).diff()
    return lr.rolling(60).skew()

F = factor_series(panel, fn_skew)
R10 = forward_returns(panel, h=10)
print('PANEL: %d instruments, F dates=%d' % (len(panel), F.shape[0]))
evaluate_factor(F, R10, name='realized_skew_60d')
