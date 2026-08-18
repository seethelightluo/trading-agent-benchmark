"""miner_1 2027-09-28: trend_pos_20d — cross-sectional demeaned (close/SMA20 - 1).
Motivation: screener flagged rel_mom_20d_skip5 whipsaw on crypto/commodity legs
(WTI -26% cum drag). Trend-position relative to MA20 is a smoother trend-strength
signal that may complement raw momentum without the skip-window noise.
"""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, 'scripts')
from factor_validator import (load_close_panel, full_validation, rank_demean,
                              apply_factor_per_asset)

panel = load_close_panel()

def _trend_pos(s):
    return s / s.rolling(20).mean() - 1.0

raw = apply_factor_per_asset(panel, _trend_pos)
factor_df = raw.apply(rank_demean, axis=1)
factor_df = factor_df.replace([np.inf, -np.inf], np.nan)

res = full_validation('trend_pos_20d', factor_df, panel,
                      artifact_path='factors/trend_pos_20d.signal.npy')
print('script finished')
