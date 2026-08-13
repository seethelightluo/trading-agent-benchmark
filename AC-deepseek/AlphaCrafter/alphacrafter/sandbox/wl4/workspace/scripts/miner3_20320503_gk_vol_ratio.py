"""miner3_20320503_gk_vol_ratio.py
Candidate: gk_vol_ratio_20x60 = Garman-Klass intraday-range vol(20d) / vol(60d).
Garman-Klass uses OHLC so it captures intraday information beyond close-based
vol (which existing vol_ratio_20_60 already uses). Low ratio => vol compressing
(calm regime); high ratio => vol expanding (stress). Cross-asset defensive tilt.
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'scripts')
from miner3_harness import load_panel, factor_series, forward_returns, evaluate_factor

panel = load_panel()

def gk(df, N):
    o, h, l, c = df['open'], df['high'], df['low'], df['close']
    log_h_l = np.log(h / l)
    log_c_o = np.log(c / o)
    # Garman-Klass vol estimator (per day), then realized vol over window
    v = 0.5 * log_h_l**2 - (2*np.log(2) - 1) * log_c_o**2
    v = v.clip(lower=1e-12)
    return np.sqrt(v.rolling(N).mean())

def fn(s, df):
    gk20 = gk(df, 20)
    gk60 = gk(df, 60)
    return gk20 / gk60

F = factor_series(panel, fn)
R10 = forward_returns(panel, h=10)
print('PANEL: %d instruments, F dates=%d' % (len(panel), F.shape[0]))
evaluate_factor(F, R10, name='gk_vol_ratio_20x60')
