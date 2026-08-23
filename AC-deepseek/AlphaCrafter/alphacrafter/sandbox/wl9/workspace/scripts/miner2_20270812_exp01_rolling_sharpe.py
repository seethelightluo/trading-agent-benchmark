#!/usr/bin/env python
"""
Rolling Sharpe Ratio Factor Validation
Date: 2027-08-12

Motivation:
Existing momentum factors (mom_10d_skip5, mom_120d_skip5) capture raw returns
but ignore volatility. A rolling Sharpe ratio adjusts returns by their risk,
potentially providing a cleaner risk-adjusted momentum signal. This is
particularly relevant in a cross-asset universe where volatility regimes
vary dramatically across equity indices, commodities, and crypto.

Construction:
- Compute daily returns over rolling windows (20d, 60d, 120d)
- Sharpe = mean(returns) / std(returns) * sqrt(252) annualized
- Skip 5 days gap to avoid microstructure noise

Hypothesis:
- Assets with higher rolling Sharpe ratios tend to outperform
- Should be complementary to raw momentum

Admission gates: abs IC >= 0.0070, abs ICIR >= 0.0840
"""

import json, os, sys, math, base64, zlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

np.seterr(all='ignore')

CURRENT_DATE = "2027-08-12"
WATCH_LIST = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
              'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

IC_GATE = 0.0070
ICIR_GATE = 0.0840

def load_data(days=800):
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

def safe_rank_ic(factor_vals, forward_rets):
    """Rank IC = Spearman correlation between factor values and forward returns."""
    mask = ~(np.isnan(factor_vals) | np.isnan(forward_rets) | np.isinf(factor_vals) | np.isinf(forward_rets))
    if mask.sum() < 8:
        return np.nan, mask.sum()
    from scipy.stats import spearmanr
    r, p = spearmanr(factor_vals[mask], forward_rets[mask])
    return r if not np.isnan(r) else np.nan, mask.sum()

def compute_forward_returns(close, horizon=10):
    fwd = np.full(len(close), np.nan)
    for i in range(len(close) - horizon):
        fwd[i] = close[i + horizon] / close[i] - 1.0
    return fwd

def rolling_sharpe(returns, window):
    """Rolling Sharpe = mean(ret)/std(ret) * sqrt(252)."""
    sharpe = np.full(len(returns), np.nan)
    for i in range(window, len(returns)):
        seg = returns[i-window:i]
        if np.std(seg) > 1e-10:
            sharpe[i] = np.mean(seg) / np.std(seg) * math.sqrt(252)
    return sharpe

def build_signal_panel(data, window, skip=5):
    """
    Build a cross-sectional factor signal panel.
    For each date, compute rolling Sharpe for each asset.
    With skip: evaluate using data up to t-skip, predict fwd returns starting at t.
    """
    # Step 1: Build aligned DataFrame of daily close prices
    all_closes = {}
    for sym, df in data.items():
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        all_closes[sym] = df['close']
    
    close_df = pd.DataFrame(all_closes).sort_index()
    rets = close_df.pct_change()
    
    # Step 2: Compute rolling Sharpe for each asset
    sharpe_vals = {}
    for sym in