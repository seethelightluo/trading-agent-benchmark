"""
Exploration: trend_conviction_20 (trend conviction factor)
Combines:
  1. Recent return magnitude (20d)
  2. Trend consistency (ratio of up-days to total in last 20 days)
  3. Normalized by volatility (20d std dev)
  
This is different from pure momentum (mom_20 tested before) because it weights
the direction by how CONSISTENT the price action has been. A steady uptrend with
many small up days gets a higher score than a volatile uptrend with a few big up days.

Formula: trend_conviction = (close/close_20ago - 1) * (up_days_ratio) / (vol_20d + 0.01)
where up_days_ratio = count(close > close_1ago) / 20

Rationale: In high-vol cross-asset regimes, momentum can be noisy. Consistency 
screens out noise-driven moves and captures genuine trends.
"""

import numpy as np
import pandas as pd
import sys, os, json

sys.path.insert(0, 'scripts')
from factor_validation_lib import (
    ASSETS, load_closes, load_index, validate_factor, print_result,
    IC_GATE, ICIR_GATE, CURRENT_DATE, artifact_b64, max_library_corr
)

# ------------------------------------------------------------
# Factor function
# ------------------------------------------------------------
def trend_conviction(close, vol, open_, high, low, macro, window=20):
    """Trend conviction = (ret_20d) * (up_ratio_20d) / (vol_20d + 0.01)"""
    ret = close / close.shift(window) - 1.0
    # Up days ratio
    up = (close.diff() > 0).astype(float)
    up_ratio = up.rolling(window).mean()
    # Vol