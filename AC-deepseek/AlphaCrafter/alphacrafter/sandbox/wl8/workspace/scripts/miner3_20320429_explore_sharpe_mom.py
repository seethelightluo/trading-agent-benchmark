#!/usr/bin/env python3
"""
Explore: Volatility-Adjusted Sharpe Momentum (sharpe_mom_20)

Idea: Pure momentum (mom_10d_skip5) was evicted due to library correlation, but
also suffered whipsaw in high-vol regimes. Instead of using raw returns, use
risk-adjusted momentum: 20-day return divided by 20-day daily return std.
This normalizes across very different asset classes (crypto vol >> rates vol)
and should be more robust during the current high-VIX regime (VIX=41.26).

Construction: sharpe_mom_20 = close_ret(close, 20) / rolling_std(daily_ret, 20)
where close_ret = (close_t / close_t-20 - 1)
and rolling_std uses daily pct_change over 20 days.

Validation: Compute IC against forward 10-day returns across all watchlist assets.
Check ICIR, coverage, turnover, decay.
"""
import numpy as np
import pandas as pd
import json
import sys
from datetime import datetime

from alphacrafter.sim.utils import get_stock_daily_data

WATCHLIST = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
             'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
MIN_LOOKBACK = 60  # need at least 60 days for 20d vol adj + forward
FORWARD_HORIZONS = [1, 3, 5, 10, 20]

def load_data(symbol, lookback=400):
    """Load stock data for a symbol."""
    df = get_stock_daily_data(symbol, lookback)
    if df is None or len(df) < MIN_LOOKBACK:
        return None
    df['symbol'] = symbol
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df

def compute_sharpe_mom_20(df):
    """Compute sharpe_mom_20 = 20-day return / 20-day vol."""
    prices = df['close'].values
    if len(prices) < 21:
        return np.full(len(prices), np.nan)
    
    # 20-day returns
    ret_20 = prices[20:] / prices[:-20] - 1.0
    # 20-day daily std (using daily returns)
    daily_ret = prices[1:] / prices[:-1] - 1.0
    vol_20 = np.array([np.std(daily_ret[max(0, i-19):i+1]) for i in range(len(daily_ret))])
    
    # Align: ret_20[i] corresponds to position i+20, vol_20[i+19] corresponds to position i+20
    # We need vol_20 over the same window as ret_20
    result = np.full(len(prices), np.nan)
    for i in range(20, len(prices)):
        r = prices[i] / prices[i-20] - 1.0
        v = np.std(daily_ret[i-20:i])  # daily returns over last 20 days
        if v > 0:
            result[i] = r / v
    return result

def compute_forward_returns(df, horizon):
    """Compute forward N-day returns for each row."""
    prices = df['close'].values
    forward_ret = np.full(len(prices), np.nan)
    for i in range(len(prices) - horizon):
        forward_ret[i] = prices[i + horizon] / prices[i] - 1.0
    return forward_ret

def rank_ic(factor_vals, forward_rets):
    """Compute Spearman rank IC between factor and forward returns."""
    valid = (~np.isnan(factor_vals)) & (~np.isnan(forward_rets))
    if valid.sum() < 8:
        return np.nan, 0
    from scipy.stats import spearmanr
    rho, pval = spearmanr(factor_vals[valid], forward_rets[valid])
    return rho, valid.sum()

print("=" * 60)
print("SHARPE MOMENTUM FACTOR EXPLORATION (sharpe_mom_20)")
print(f"Current date: 2032-04-29, VIX=41.26 (high-vol regime)")
print(f"