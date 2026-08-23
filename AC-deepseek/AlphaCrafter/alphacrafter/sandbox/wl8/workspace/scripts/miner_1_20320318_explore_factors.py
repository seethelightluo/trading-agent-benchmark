"""
Factor Mining Exploration - 2032-03-18
Current date: 2032-03-18
All 44 library factors evicted (correlation conflicts).
Goal: Find novel factors with uncorrelated signal structure.

Ideas to explore:
1. consistency_20: Ratio of positive days / total days over 20d (directional consistency)
2. cross_range_20: (high-close)/close over 20d, cross-sectionally normalized (range expansion)
3. vol_adj_mom_20: Momentum gated by volume confirmation 
4. ret_skew_20: Daily return skewness over 20d (asymmetry signal)
5. hi_lo_contraction_20: (high-low)/close vs its own 60d mean (volatility regime change)
"""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, skew
import json
import warnings
warnings.filterwarnings('ignore')

watchlist = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX', 
             'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

# Load data - use a large window for validation
lookback = 750
data = {}
for sym in watchlist:
    df = get_stock_daily_data(symbol=sym, days=lookback)
    if df is not None and len(df) >= 120:
        # Ensure sorted
        df = df.sort_values('date').reset_index(drop=True)
        data[sym] = df
    else:
        print(f"WARNING: {sym} has insufficient data: {len(df) if df is not None else 0}")

instruments_available = list(data.keys())
print(f"Instruments available: {len(instruments_available)}")
print(f"Date range: {data[instruments_available[0]]['date'].iloc[0].strftime('%Y-%m-%d')} "
      f"to {data[instruments_available[0]]['date'].iloc[-1].strftime('%Y-%m-%d')}")
print()

# Build aligned price matrix
dates = data[instruments_available[0]]['date'].values
T = len(dates)
close_matrix = np.full((T, len(instruments_available)), np.nan)
high_matrix = np.full((T, len(instruments_available)), np.nan)
low_matrix = np.full((T, len(instruments_available)), np.nan)
volume_matrix = np.full((T, len(instruments_available)), np.nan)

for j, sym in enumerate(instruments_available):
    df = data[sym]
    close_matrix[:len(df), j] = df['close'].values
    high_matrix[:len(df), j] = df['high'].values
    low_matrix[:len(df), j] = df['low'].values
    volume_matrix[:len(df), j] = df['volume'].values

# Compute forward returns (1-day, 5-day, 10-day, 20-day)
ret_fwd_1 = np.full_like(close_matrix, np.nan)
ret_fwd_5 = np.full_like(close_matrix, np.nan)
ret_fwd_10 = np.full_like(close_matrix, np.nan)
ret_fwd_20 = np.full_like(close_matrix, np.nan)

for j in range(len(instruments_available)):
    for t in range(T-1):
        ret_fwd_1[t, j] = close_matrix[t+1, j] / close_matrix[t, j] - 1
    for t in range(T-5):
        ret_fwd_5[t, j] = close_matrix[t+5, j] / close_matrix[t, j] - 1
    for t in range(T-10):
        ret_fwd_10[t, j] = close_matrix[t+10, j] / close_matrix[t, j] - 1
    for t in range(T-20):
        ret_fwd_20[t, j] = close_matrix[t+20, j] / close_matrix[t, j] - 1

# Compute daily returns for factor calc
daily_ret = np.full_like(close_matrix, np.nan)
for j in range(len(instruments_available)):
    daily_ret[1:, j] = close_matrix[1:, j] / close_matrix[:-1, j] - 1

def compute_ic(factor_vals, fwd_ret, min_valid=8):
    """Compute cross-sectional IC for each date."""
    n_dates = min(len(factor_vals), len(fwd_ret))
    ics = []
    valid_dates = 0
    for t in range(n_dates):
        ff = factor_vals[t, :]
        rr = fwd_ret[t, :]
        mask = ~(np.isnan(ff) | np.isnan(rr))
        if np.sum(mask) >= min_valid:
            r, _ = pearsonr(ff[mask], rr[mask])
            if not np.isnan(r):