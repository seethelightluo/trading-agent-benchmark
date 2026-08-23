"""
miner2_20320708_explore_novel_v2.py
Explore novel factor candidates for 15-instrument cross-asset universe.
Data through 2032-07-07.

Factor candidates:
A) Cross-asset breadth: fraction of assets with positive 10d return
B) Risk-adjusted reversal: -(1d/5d return) / 20d vol
C) Cross-sectional dispersion of momentum: std of 20d returns across assets
D) Macro beta to DXY (observation-only)
E) Vol of vol ratio: vol_10d / vol_60d
F) Price acceleration: (short_mom - long_mom) / vol
G) Skewness factor: skew of 30d returns
H) Drawup from 60d max: (close - 60d_max) / 60d_max
"""
import numpy as np
from scipy.stats import spearmanr

from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCHLIST = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
             'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
MACRO_SIGNALS = ['DXY', 'USDCNY']
HORIZON = 10
MIN_VALID = 8

print("="*70)
print("FACTOR MINER 2 -- Novel Factor Exploration (2032-07-08)")
print("="*70)

# ---- data loading ----
data = {}
for sym in WATCHLIST:
    df = get_stock_daily_data(sym, 500)
    if df is not None and len(df) >= 200:
        data[sym] = df
    else:
        print(f"WARN: {sym} has {len(df) if df is not None else 0} days, skip")

macro_data = {}
for sym in MACRO_SIGNALS:
    df = get_index_daily_data(sym, 500)
    if df is not None and len(df) >= 200:
        macro_data[sym] = df
        print(f"MACRO {sym}: {len(df)} days loaded")
    else:
        print(f"MACRO WARN: {sym} insufficient: {len(df) if df is not None else 0}")

print(f"\nValid instruments: {len(data)}/15")

# Build common date index across assets
all_dates = None
for sym, df in data.items():
    dts = set(str(d)[:10] for d in df['date'].values)
    if all_dates is None:
        all_dates = dts
    else:
        all_dates &= dts

for sym, df in macro_data.items():
    dts = set(str(d)[:10] for d in df['date'].values)
    all_dates &= dts

common_dates = sorted(all_dates)
print(f"Common dates (incl macro): {len(common_dates)}, range {common_dates[0]} to {common_dates[-1]}")

date_to_idx = {d:i for i,d in enumerate(common_dates)}
n_dates = len(common_dates)
n_assets = len(data)
assets_list = sorted(data.keys())

close_mat = np.full((n_dates, n_assets), np.nan)
pct_mat = np.full((n_dates, n_assets), np.nan)

for j, sym in enumerate(assets_list):
    df = data[sym]
    for _, row in df.iterrows():
        d = str(row['date'])[:10]
        if d in date_to_idx:
            i = date_to_idx[d]
            close_mat[i,j] = row['close']
            p = row['pct_change']
            pct_mat[i,j] = p if not np.isnan(p) else 0.0

macro_mat = {}
for m_sym, m_df in macro_data.items():
    m_arr = np.full(n_dates, np.nan)
    for _, row in m_df.iterrows():
        d = str(row['date'])[:10]
        if d in date_to_idx:
            i = date_to_idx[d]
            m_arr[i] = row['close']
    macro_mat[m_sym] = m_arr

print(f"Matrix shape: {close_mat.shape}, NaN in close: {np.isnan(close_mat).mean():.4f}")

# Forward returns (10d horizon)
fwd_ret = np.full((n_dates, n_assets), np.nan)
for j in range(n_assets):
    for i in range(n_dates):
        if i + 10 < n_dates:
            fwd_ret[i,j] = close_mat[i+10,j]/close_mat[i,j] - 1

# Cross-sectional IC
def cs_ic(factor, fwd, min_valid=MIN_VALID):
    ics = []
    for t in range(n_dates):
        fv = factor[t]
        rv = fwd[t]
        valid = ~(np.isnan(fv) | np.isnan(rv))
        n_valid = int(np.sum(valid))
        if n_valid >= min_valid:
            fv_v = fv[valid]
            rv_v = rv[valid]
            if np.std(fv_v) > 1e-10 and np.std(rv_v) > 1e-10:
                ic, _ = spearmanr(fv_v, rv_v)
                if not np.isnan(ic):
                    ics.append(ic)
    return np.array(ics)

def report_factor(name, factor_vals):
    ics = cs_ic(factor_vals, fwd_ret)
    if len(ics) < 5:
        print(f"\n  {name}: too few obs ({len(ics)})")
        return None
    mean_ic = float(np.mean(ics))
    mean_abs_ic = float(np.mean(np.abs(ics)))
    ic_std = float(np.std(ics))
    icir = mean_ic / max(ic_std, 1e-10)
    hit = float(np.mean(ics > 0))
    pass_007 = float(np.mean(np.abs(ics) >= 0.007))
    coverage = float(np.mean(~np.isnan(factor_vals)))
    # Also compute daily ICIR (not scaled by sqrt) for admission gate
    daily_icir = icir  # This is already daily (mean/std, no sqrt)
    
    print(f"\n  --- {name} ---")
    print(f"  Dates with valid IC: {len(ics)}")
    print(f"  Mean IC: {mean_ic:.6f}")
    print(f"  Mean |IC|: {mean_abs_ic:.6f}")
    print(f"  IC SD: {ic_std:.6f}")
    print(f"  ICIR (daily): {daily_icir:.4f}")
    print(f"  IC > 0: {hit:.3f}")
    print(f"  |IC| >= 0.007: {pass_007:.3f}")
    print(f"  Coverage: {coverage:.4f}")
    
    # GATE: abs daily IC >= 0.007 AND abs daily ICIR >= 0.084
    passes_gate = (mean_abs_ic >= 0.007) and (abs(daily_icir) >= 0.084)
    print(f"  PASSES GATE (abs_ic>=0.007 & abs_icir>=0.084): {passes_gate}")
    
    return {
        'name': name, 'dates': len(ics), 'mean_ic': mean_ic,
        'mean_abs_ic': mean_abs_ic, 'icir': daily_icir,
        'hit_rate': hit, 'pass_rate_007': pass_007, 'coverage': coverage,
        'passes_gate': passes_gate
    }

results = {}

# =========== FACTOR A: Cross-asset breadth (positive 10d return) ===========
print("\n" + "="*70)
print("FACTOR A: Cross-Asset Breadth (fraction of assets with positive 10d ret)")
factor_a = np.full((n_dates, n_assets), np.nan)
for t in range(20, n_dates):
    rets_10d = np.array([close_mat[t,j]/close_mat[max(0,t-10),j]-1 
                         if not np.isnan(close_mat[t,j]) and not np.isnan(close_mat[max(0,t-10),j])
                         else np.nan for j in range(n_assets)])
    breadth = float(np.nanmean(rets_10d > 0)) if np.sum(~