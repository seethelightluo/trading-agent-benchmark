"""
Re-validate the two active factors: flip_mom_20x10 and mom_diff_20_60
Last validated: 2031-02-05 (over 13 months ago)
"""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import json

current_date = "2032-03-18"
watchlist = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX', 'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

# Fetch data
data = {}
for sym in watchlist:
    df = get_stock_daily_data(symbol=sym, days=400)
    if df is not None and len(df) > 130:
        data[sym] = df
    else:
        print(f"WARNING: {sym} insufficient data ({len(df) if df is not None else 0} days)")

print(f"Loaded {len(data)} instruments with sufficient data")

# =====================================================
# Factor 1: flip_mom_20x10
# =====================================================
def calc_flip_mom_20x10(df, lookback=20):
    """1 - close/shift(rolling_min(close, lookback), 1) 
    A mean-reversion/reversal signal - distance from recent low."""
    closes = df['close'].values
    if len(closes) < lookback + 2:
        return None
    roll_min = pd.Series(closes).rolling(window=lookback).min().values
    prev_min = np.roll(roll_min, 1)
    prev_min[0] = np.nan
    factor = 1 - closes / prev_min
    factor = np.where(np.isfinite(factor), factor, np.nan)
    return factor

# =====================================================
# Factor 2: mom_diff_20_60
# =====================================================
def calc_mom_diff_20_60(df):
    """Momentum acceleration: (close/shift(close,20)-1) - (close/shift(close,60)-1)"""
    closes = df['close'].values
    if len(closes) < 61:
        return None
    mom20 = np.full_like(closes, np.nan)
    mom60 = np.full_like(closes, np.nan)
    if len(closes) > 20:
        mom20[20:] = closes[20:] / closes[:-20] - 1
    if len(closes) > 60:
        mom60[60:] = closes[60:] / closes[:-60] - 1
    factor = mom20 - mom60
    return factor

# =====================================================
# Cross-sectional IC computation
# =====================================================
def compute_ic_series(factor_func, forward_days=10):
    first_sym = list(data.keys())[0]
    all_dates = data[first_sym]['date'].values
    
    ic_values = []
    usable_dates = []
    coverage_days = []
    
    for i in range(120, len(all_dates) - forward_days):
        dt = all_dates[i]
        
        fvals = {}
        for sym in data.keys():
            df = data[sym]
            if i < len(df):
                factor_arr = factor_func(df.iloc[:i+1])
                if factor_arr is not None and len(factor_arr) > 1:
                    val = factor_arr[-1]
                    if np.isfinite(val):
                        fvals[sym] = val
        
        rets = {}
        for sym in data.keys():
            df = data[sym]
            if i + forward_days < len(df):
                fwd_ret = df.iloc[i + forward_days]['close'] / df.iloc[i]['close'] - 1
                if np.isfinite(fwd_ret):
                    rets[sym] = fwd_ret
        
        common = set(fvals.keys()) & set(rets.keys())
        if len(common) >= 8:
            f_vals = np.array([fvals[s] for s in common])
            r_vals = np.array([rets[s] for s in common])
            mask = np.isfinite(f_vals) & np.isfinite(r_vals)
            if mask.sum() >= 8:
                ic, pval = pearsonr(f_vals[mask], r_vals[mask])
                ic_values.append(ic)
                usable_dates.append(dt)
                coverage_days.append(mask.sum())
    
    ic_arr = np.array(ic_values)
    if len(ic_arr) == 0:
        return {'ic_mean': 0, 'icir': 0, 'n_dates': 0, 'n_obs': 0}
    
    ic_mean = np.mean(ic_arr)
    ic_std = np.std(ic_arr) if len(ic_arr) > 1 else 1e-10
    icir = ic_mean / ic_std if ic_std > 0 else 0
    ic_hit = np.mean(np.sign(ic_arr) == np.sign(ic_mean)) if ic_mean != 0 else 0.5
    
    return {
        'ic_mean': ic_mean,
        'icir': icir,
        'ic_hit_ratio': ic_hit,
        'ic_std': ic_std,
        'n_dates': len(ic_arr),
        'avg_coverage': np.mean(coverage_days) if coverage_days else 0,
        'first_date': str(usable_dates[0]) if usable_dates else None,
        'last_date': str(usable_dates[-1]) if usable_dates else None
    }

# All-period validation
print("\n=== Re-validating flip_mom_20x10 (all data) ===")
for fwd in [5, 10, 21]:
    res = compute_ic_series(calc_flip_mom_20x10, forward_days=fwd)
    print(f"  Forward {fwd}d: IC={res['ic_mean']:.4f}, ICIR={res['icir']:.4f}, "
          f"Hit={res['ic_hit_ratio']:.2f}, n_dates={res['n_dates']}, "
          f"avg_cov={res['avg_coverage']:.1f}")

# Recent 252d validation (last ~1yr of trading)
print("\n=== Re-validating flip_mom_20x10 (last 252d) ===")
def compute_ic_series_windowed(factor_func, forward_days=10, window=252):
    first_sym = list(data.keys())[0]
    all_dates = data[first_sym]['date'].values
    start_idx = max(120, len(all_dates) - window)
    
    ic_values = []
    useful_dates = []
    for i in range(start_idx, len(all_dates) - forward_days):
        dt = all_dates[i]
        fvals, rets = {}, {}
        for sym in data.keys():
            df = data[sym]
            if i < len(df):
                factor_arr = factor_func(df.iloc[:i+1])
                if factor_arr is not None and len(factor_arr) > 1:
                    val = factor_arr[-1]
                    if np.isfinite(val):
                        fvals[sym] = val
        for sym in data.keys():
            df = data[sym]
            if i + forward_days < len(df):
                fwd_ret = df.iloc[i + forward_days]['close'] / df.iloc[i]['close'] - 1
                if np.isfinite(fwd_ret):
                    rets[sym] = fwd_ret
        common = set(fvals.keys()) & set(rets.keys())
        if len(common) >= 8:
            f_vals = np.array([fvals[s] for s in common])
            r_vals = np.array([rets[s] for s in common])
            mask = np.isfinite(f_vals) & np.isfinite(r_vals)
            if mask.sum() >= 8:
                ic, _ = pearsonr(f_vals[mask], r_vals[mask])
                ic_values.append(ic)
                useful_dates.append(dt)
    ic_arr = np.array(ic_values)