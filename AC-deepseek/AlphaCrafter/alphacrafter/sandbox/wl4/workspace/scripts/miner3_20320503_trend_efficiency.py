"""miner3_20320503_trend_efficiency.py
Candidate: trend_efficiency_N = |log(close_t/close_{t-N})| / sum_{i=1..N} |log(close_{t-i+1}/close_{t-i})|
Idea: cross-asset trend cleanliness. Assets whose price path is a clean directional
trend (high efficiency, little retracement) tend to keep trending; choppy paths tend
to revert. Validated at 20d and 60d windows.
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'scripts')
from miner3_harness import load_panel, factor_series, forward_returns, daily_ic, evaluate_factor

panel = load_panel()

def make_eff(N):
    def fn(s, df):
        close = df['close']
        lr = np.log(close).diff()
        gross = lr.abs().rolling(N).sum()
        net = np.log(close / close.shift(N)).abs()
        return net / gross
    return fn

F20 = factor_series(panel, make_eff(20), extra=None)
F60 = factor_series(panel, make_eff(60), extra=None)
R10 = forward_returns(panel, h=10)

print('PANEL: %d instruments, F20 dates=%d, F60 dates=%d' % (len(panel), F20.shape[0], F60.shape[0]))
print(F20.tail(3).T)

evaluate_factor(F20, R10, name='trend_efficiency_20d')
evaluate_factor(F60, R10, name='trend_efficiency_60d')
