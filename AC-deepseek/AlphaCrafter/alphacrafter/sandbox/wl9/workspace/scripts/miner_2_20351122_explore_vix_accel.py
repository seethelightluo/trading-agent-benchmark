"""
Factor Exploration: VIX Regime Acceleration Factor (vix_roc_accel)
Idea: Measure how rapidly VIX is accelerating - when short-term VIX changes vastly
outpace medium-term changes, it signals panic. In such regimes, safe havens outperform
risk assets. The factor ranks assets by their defensive qualities during VIX accelerations.

Current regime: VIX 37.82 (90.4th percentile), 60d ROC +100.5%, 20d ROC +31.8%
This is an ideal test environment.

Construction: 
  1. Compute VIX_ROC(5) / VIX_ROC(20) - acceleration ratio
  2. Map assets to expected defensive score based on historical VIX-regime beta
  3. Rank cross-sectionally

Admission gates: |IC| >= 0.0070, |ICIR| >= 0.0840
"""

import json
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

# Configuration
WATCHLIST = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX', 
             'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
FACTOR_ID = "vix_roc_accel"
FACTOR_NAME = "VIX ROC Acceleration"
HORIZON = 10  # forward return horizon for IC computation

def compute_vix_roc_accel_factor(vix_series, asset_returns_dict, lookback=60):
    """
    Compute factor that captures the VIX acceleration regime.
    
    For each date in the history:
    1. Compute VIX short-term ROC (5d) and medium-term ROC (20d)
    2. VIX acceleration =