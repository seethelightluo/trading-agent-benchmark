"""
miner2_20320513_explore_voladj_mom_dispersion.py
Explore two novel factor families:
1) Volatility-adjusted momentum (Sharpe-ratio style): return / volatility
2) Cross-asset dispersion: rank-based momentum dispersion

Current date: 2032-05-13
"""

from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import numpy as np
import pandas as pd
import json, os, sys

WATCHLIST = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX', 
             'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
MIN_DAYS = 120  # need at least 120 days lookback
HORIZON = 10   # forward return horizon

def get_data(symbol, days=300):
    """Fetch daily data, return df or None"""
    try:
        df = get_stock_daily_data(symbol, days)
        return df
    except:
        return None

def compute_forward_returns(df, horizon=10):
    """Compute forward returns for each date"""
    closes = df['close'].values
    fwd_ret = np.full(len(closes), np.nan)
    for i in range(len(closes) - horizon):
        fwd_ret[i] = closes[i + horizon] / closes[i] - 1
    return fwd_ret

def compute_vol_adj_mom(df, mom_window=20, vol_window=20):
    """
    Factor 1: Volatility-adjusted momentum (Sharpe-like)
    = (return over mom_window) / (volatility over vol_window)
    Returns array of factor values aligned with df dates (last valid = len(df)-horizon)
    """
    closes = df['close'].values
    pct_chg = df['pct_change'].values
    
    factor = np.full(len(closes), np.nan)
    for i in range(mom_window + vol_window, len(closes)):
        ret = closes[i] / closes[i - mom_window] - 1
        vol = np.std(pct_chg[i - vol_window:i]) * np.sqrt(252)
        if vol > 1e-10:
            factor[i] = ret / vol
        else:
            factor[i] = 0.0  # frozen asset -> zero signal
    return factor

def compute_ma_reversion(df, ma_window=60):
    """
    Factor 2: Distance from moving average (mean reversion)
    = (close / SMA(close, 60) - 1)
    Negative values mean below MA = potential mean reversion up
    """
    closes = df['close'].values
    factor = np.full(len(closes), np.nan)
    for i in range(ma_window, len(closes)):
        sma = np.mean(closes[i - ma_window:i])
        if sma > 1e-10:
            factor[i] = closes[i] / sma - 1
        else:
            factor[i] = 0.0
    return factor

def compute_vol_term_structure(df, short_window=10, long_window=60):
    """
    Factor 3: Volatility term structure
    = short_term_vol / long_term_vol - 1
    Rising ratio means short-term vol > long-term vol = regime shift / panic