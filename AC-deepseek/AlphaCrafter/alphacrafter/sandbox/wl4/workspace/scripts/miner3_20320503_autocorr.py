"""miner3_20320503_autocorr.py
Candidate: ret_autocorr_20d = 1-day log-return autocorrelation over 20d window.
Idea: positive autocorrelation => trending micro-structure (continuation);
negative autocorrelation => mean reversion. Cross-asset: assets whose daily
returns keep the same sign on average may continue.
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'scripts')
from miner3_harness import load_panel, factor_series, forward_returns, evaluate_factor

panel = load_panel()

def fn_ac(s, df):
    lr = np.log(df['close']).diff()
    return lr.rolling(20).apply(lambda x: pd.Series(x).autocorr(1) if len(x) == 20 else np.nan, raw=False)

F = factor_series(panel, fn_ac)
R10 = forward_returns(panel, h=10)
print('PANEL: %d instruments, F dates=%d' % (len(panel), F.shape[0]))
evaluate_factor(F, R10, name='ret_autocorr_20d')
