"""
Comprehensive Factor Re-validation & Exploration Script
Current date: 2032-05-13 (visible through 2032-05-12)

Tasks:
1. Re-validate flip_mom_20x10 (all data + last 252d)
2. Re-validate mom_diff_20_60 (all data + last 252d)
3. Explore new factor: cross-asset volatility regime adaptation
   - The idea: when cross-asset volatility is high (VIX elevated, equity
     vol high), momentum works differently from low-vol regimes.
     A factor that adapts to vol regimes using macro signals.
"""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import json, os, sys, time

current_date = "2032-05-13"
np.random.seed(42)

watchlist = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX', 'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

# Fetch data - use 700 days to have enough history
print("Fetching data...")
data = {}
for sym in watchlist:
    df = get_stock_daily_data(symbol=sym, days=700)
    if df is not None and len(df) > 200:
        data[sym] = df
    else:
        print(f"  WARNING: {sym} insufficient data ({len(df) if df is not None else 0} days)")

print(f"Loaded {len(data)} instruments with sufficient data")

# Also fetch macro signals for regime analysis
dxy = get_index_daily_data(symbol='DXY', days=700)
vix = get_index_daily_data(symbol='VIX', days=700)
usdcny = get_index_daily_data(symbol='USDCNY', days=700)
print(f"Macro: DXY={dxy is not None}, VIX={vix is not None}, USDCNY={usdcny is not None}")

# =====================================================
# FACTOR 1: flip_mom_20x10
# =====================================================
def calc_flip_mom_20x10(df, lookback=20):
    """1 - close / shift(rolling_min(close, lookback), 1)"""
    closes = df['close'].values
    if len(closes) < lookback + 2:
        return None
    roll_min = pd.Series(closes).rolling(window=lookback).min().values
    prev_min = np.roll(roll_min, 1)
    prev_min[0] = np.nan
    factor = 1 - closes / prev_min
    return np.where(np.isfinite(factor), factor, np.nan)

# =====================================================
# FACTOR 2: mom_diff_20_60
# =====================================================
def calc_mom_diff_20_60(df):
    """Momentum acceleration: mom20 - mom60"""
    closes = df['close'].values
    if len(closes) < 61:
        return None
    n = len(closes)
    mom20 = np.full(n, np.nan)
    mom60 = np.full(n, np.nan)
    if n > 20:
        mom20[20:] = closes[20:] / closes[:-20] - 1
    if n > 60:
        mom60[60:] = closes[60:] / closes[:-60] - 1
    return mom20 - mom60

# =====================================================
# Compute IC series
# =====================================================
def compute_ic_series(factor_func, forward_days=10, max_dates=None):
    """Compute cross-sectional IC across all dates"""
    first_sym = list(data.keys())[0]
    all_dates = data[first_sym]['date'].values
    n = len(all_dates)
    
    ic_values = []
    usable_dates = []
    coverage_info = []
    
    start_i = 120
    end_i = n - forward_days
    if max_dates is not None and end_i - start_i > max_dates:
        start_i = end_i - max_dates
    
    for i in range(start_i, end_i):
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
                ic, pval = pearsonr(f_vals[mask], r_vals[mask])
                ic_values.append(ic)
                usable_dates.append(dt)
                coverage_info.append(mask.sum())
    
    ic_arr = np.array(ic_values)
    if len(ic_arr) == 0:
        return {'ic_mean': 0, 'icir': 0, 'ic_std': 0, 'n_dates': 0, 'avg_coverage': 0}
    
    ic_mean = np.mean(ic_arr)
    ic_std = np.std(ic_arr) if len(ic_arr) > 1 else 1e-10
    icir = ic_mean / ic_std if ic_std > 0 else 0
    ic_hit = np.mean(np.sign(ic_arr) == np.sign(ic_mean)) if ic_mean != 0 else 0.5
    
    return {
        'ic_mean': float(f'{ic_mean:.6f}'),
        'icir': float(f'{icir:.6f}'),
        'ic_std': float(f'{ic_std:.6f}'),
        'ic_hit_ratio': float(f'{ic_hit:.4f}'),
        'n_dates': len(ic_arr),
        'avg_coverage': float(f'{np.mean(coverage_info):.1f}'),
        'first_date': str(usable_dates[0]) if usable_dates else None,
        'last_date': str(usable_dates[-1]) if usable_dates else None
    }

# =====================================================
# RE-VALIDATION: flip_mom_20x10
# =====================================================
print("\n" + "="*70)
print("RE-VALIDATING flip_mom_20x10")
print("="*70)

for fwd in [5, 10, 21]:
    res = compute_ic_series(calc_flip_mom_20x10, forward_days=fwd)
    print(f"  Fwd {fwd}d: IC={res['ic_mean']:.4f} ICIR={res['icir']:.4f} "
          f"Hit={res['ic_hit_ratio']:.2f} n={res['n_dates']} cov={res['avg_coverage']:.1f} "
          f"period={res['first_date'][:10] if res['first_date'] else 'N/A'} -> {res['last_date'][:10] if res['last_date'] else 'N/A'}")

# Recent 252d
print("\n  -- Last 252 trading days --")
for fwd in [5, 10, 21]:
    res = compute_ic_series(calc_flip_mom_20x10, forward_days=fwd, max_dates=300)