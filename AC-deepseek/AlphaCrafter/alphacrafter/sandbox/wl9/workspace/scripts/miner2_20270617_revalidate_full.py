#!/usr/bin/env python
"""
Full re-validation of all factors in the library as of 2027-06-17.
Tests IC and ICIR over recent period to detect factor decay.
"""
import json, os, sys, math
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict

np.seterr(all='ignore')

CURRENT_DATE = "2027-06-17"
MIN_DAYS = 500  # ~2 years of trading data
WATCH_LIST = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
              'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load_data(days=500):
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
    """Pearson correlation with NaN protection."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 8:
        return 0.0
    r, _ = pearsonr(x[mask], y[mask])
    return r if not np.isnan(r) else 0.0

def compute_forward_returns(close, horizon=10):
    """Compute forward returns for a given horizon in trading days."""
    fwd = np.full_like(close, np.nan)
    for i in range(len(close) - horizon):
        fwd[i] = close[i+horizon] / close[i] - 1.0
    return fwd

def evaluate_factor(data, calc_fn, factor_id, horizon=10):
    """Cross-sectional IC evaluation across dates."""
    ic_list = []
    n_dates_list = []
    
    # Find common dates
    common_dates = None
    for sym, df in data.items():
        dates = set(df['date'].astype(str).values)
        if common_dates is None:
            common_dates = dates
        else:
            common_dates &= dates
    
    if common_dates is None:
        return {'ic': 0, 'icir': 0, 'n_ic_dates': 0, 'hit_ratio': 0.5,
                'n_dates_ge8': 0, 'total_candidates': 0}
    
    common_dates = sorted(common_dates)
    
    # Need at least MIN_DAYS price history before the start
    min_idx = MIN_DAYS - 1
    usable_dates = [d for d in common_dates if d >= common_dates[max(0, min(common_dates.index(d) if d in common_dates else 0, len(common_dates)-1))]]
    # Simpler approach: just iterate through dates with enough lookback
    # Actually, let's do a date-index based approach
    
    all_factors = {}
    all_returns = {}
    
    for sym, df in data.items():
        df = df.copy()
        df['date_str'] = df['date'].astype(str)
        df = df.sort_values('date')
        
        # Compute factor values
        factor_vals = calc_fn(df)
        if factor_vals is None:
            continue
        
        # Compute forward returns
        fwd_ret = compute_forward_returns(df['close'].values, horizon)
        
        # Map to dates
        for i in range(len(df)):
            d = df['date_str'].values[i]
            if pd.isna(factor_vals[i]) or pd.isna(fwd_ret[i]):
                continue
            if d not in all_factors:
                all_factors[d] = {}
                all_returns[d] = {}
            all_factors[d][sym] = float(factor_vals[i])
            all_returns[d][sym] = float(fwd_ret[i])
    
    # Compute daily cross-sectional IC
    for d in sorted(all_factors.keys()):
        vals = []
        rets = []
        for sym in all_factors[d]:
            if sym in all_returns[d]:
                vals.append(all_factors[d][sym])
                rets.append(all_returns[d][sym])
        vals = np.array(vals)
        rets = np.array(rets)
        
        # Filter invalids
        mask = ~(np.isnan(vals) | np.isnan(rets) | np.isinf(vals) | np.isinf(rets))
        if mask.sum() >= 8:
            ic_list.append(safe_corr(vals[mask], rets[mask]))
            n_dates_list.append(int(mask.sum()))
    
    ic_arr = np.array(ic_list)
    n_ic = len(ic_arr)
    
    if n_ic < 10:
        return {'ic': 0, 'icir': 0, 'n_ic_dates': n_ic, 'hit_ratio': 0.5,
                'n_dates_ge8': len(ic_list), 'total_candidates': n_ic}
    
    mean_ic = float(np.mean(ic_arr))
    std_ic = float(np.std(ic_arr)) if np.std(ic_arr) > 1e-10 else 0.0001
    icir = float(mean_ic / std_ic) if std_ic > 0 else 0.0
    hit = float(np.mean(np.abs(ic_arr) > 0.02))  # proxy for directional signal
    
    return {
        'ic': mean_ic,
        'icir': icir,
        'n_ic_dates': n_ic,
        'hit_ratio': float(np.mean(ic_arr > 0)),
        'n_dates_ge8': len(ic_list),
        'total_candidates': n_ic
    }


# =========================================================================
# FACTOR DEFINITIONS
# =========================================================================

def calc_mom_120d_skip5(df):
    """Momentum: skip 5 then 120d return"""
    c = df['close'].values
    if len(c) < 130:
        return np.full(len(c), np.nan)
    ret = np.full(len(c), np.nan)
    for i in range(125, len(c)):
        ret[i] = c[i] / c[i-125] - 1.0
    return ret

def calc_mom_10d_skip5(df):
    """Momentum: skip 5 then 10d return"""
    c = df['close'].values
    if len(c) < 20:
        return np.full(len(c), np.nan)
    ret = np.full(len(c), np.nan)
    for i in range(15, len(c)):
        ret[i] = c[i] / c[i-15] - 1.0
    return ret

def calc_vix_beta_cond_60x20(df):
    """Rolling beta to VIX returns (requires VIX data)"""
    c