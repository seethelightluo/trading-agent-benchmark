#!/usr/bin/env python
"""
Rolling Sharpe Ratio Factor Exploration
Date: 2027-07-28

Motivation:
Existing momentum factors (mom_10d_skip5, mom_120d_skip5) capture raw returns
but ignore volatility. A rolling Sharpe ratio adjusts returns by their risk,
potentially providing a cleaner risk-adjusted momentum signal. This is
particularly relevant in a cross-asset universe where volatility regimes
vary dramatically across equity indices, commodities, and crypto.

Construction:
- Compute daily returns over a rolling window (60d)
- Sharpe = mean(returns) / std(returns) * sqrt(252) annualized
- Skip 5 days to avoid microstructure noise (same convention as existing mom factors)

Hypothesis:
- Stocks/assets with higher rolling Sharpe ratios outperform
- Should be complementary to raw momentum (orthogonal signal)
"""

import json, os, sys, math
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

np.seterr(all='ignore')

CURRENT_DATE = "2027-07-28"
WATCH_LIST = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
              'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

# Admission gates: abs IC >= 0.0070, abs ICIR >= 0.0840
IC_GATE = 0.0070
ICIR_GATE = 0.0840

def load_data(days=700):
    """Load price data for all watchlist instruments."""
    data = {}
    for sym in WATCH_LIST:
        df = get_stock_daily_data(symbol=sym, days=days)
        if df is None or len(df) < 60:
            df = get_index_daily_data(symbol=sym, days=days)
        if df is not None and len(df) > 60:
            data[sym] = df
    return data

def safe_corr(x, y):
    mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
    if mask.sum() < 8:
        return 0.0
    r, p = pearsonr(x[mask], y[mask])
    return r if not np.isnan(r) else 0.0

def compute_forward_returns(close, horizon=10):
    fwd = np.full(len(close), np.nan)
    for i in range(len(close) - horizon):
        fwd[i] = close[i+horizon] / close[i] - 1.0
    return fwd

# =====================================================================
# FACTOR: Rolling Sharpe Ratio (60d, skip 5)
# =====================================================================
def calc_rolling_sharpe_60d(df, skip=5, window=60):
    """Compute rolling Sharpe ratio: mean(ret)/std(ret) * sqrt(252), skip 5 days."""
    c = df['close'].values
    n = len(c)
    if n < window + skip + 1:
        return np.full(n, np.nan)
    
    # Daily returns
    ret = np.full(n, np.nan)
    for i in range(1, n):
        ret[i] = c[i] / c[i-1] - 1.0
    
    factor = np.full(n, np.nan)
    for i in range(window + skip, n):
        seg = ret[i-window+1-5:i+1-5]  # skip 5 then use window
        if np.std(seg) > 1e-10 and np.sum(~np.isnan(seg)) > window//2:
            sharpe = np.mean(seg) / np.std(seg) * math.sqrt(252)
            factor[i] = sharpe
    return factor

# =====================================================================
# FACTOR: Rolling Sharpe Ratio (20d, skip 5) - shorter horizon
# =====================================================================
def calc_rolling_sharpe_20d(df, skip=5, window=20):
    """Compute rolling Sharpe ratio 20d window."""
    c = df['close'].values
    n = len(c)
    if n < window + skip + 1:
        return np.full(n, np.nan)
    
    ret = np.full(n, np.nan)
    for i in range(1, n):
        ret[i] = c[i] / c[i-1] - 1.0
    
    factor = np.full(n, np.nan)
    for i in range(window + skip, n):
        seg = ret[i-w