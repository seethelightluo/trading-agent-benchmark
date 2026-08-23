"""
miner2_20320707_explore_novel_factors.py
Explore novel factor candidates for 15-instrument cross-asset universe.
Current data through 2032-07-07.

Factor candidates:
A) Vol-adjusted momentum: ret_20d / (vol_20d * sqrt(252))
B) Downside vol ratio: downside_vol_20 / total_vol_20
C) Cross-section rank momentum (centered rank of 10d return)
D) Vol term structure: vol_10d / vol_60d - 1
E) Price acceleration: (fast_ret - slow_ret) / vol
"""

from alphacrafter.sim.utils import get_stock_daily_data
import numpy as np
from scipy.stats import spearmanr

WATCHLIST = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
             'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
HORIZON = 10
MIN_VALID = 8

# ---- data loading ----
print("="*70)
print("FACTOR MINER 2 — Novel Factor Exploration (2032-07-07)")
print("="*70)

data = {}
for sym in WATCHLIST:
    df = get_stock_daily_data(sym, 500)
    if df is not None and len(df) >= 200:
        data[sym] = df
    else:
        print(f"WARN: {sym} has {len(df) if df is not None else 0} days, skip")

print(f"\nValid instruments: {len(data)}/15")

# Build common date index
all_dates = None
for sym, df in data.items():
    dts = set(str(d)[:10] for d in df['date'].values)
    if all_dates is None:
        all_dates = dts
    else:
        all_dates &= dts
common_dates = sorted(all_dates)
print(f"Common dates: {len(common_dates)}, range {common_dates[0]} to {common_dates[-1]}")

# Build aligned matrices
n_dates = len(common_dates)
n_assets = len(data)
date_to_idx = {d:i for i,d in enumerate(common_dates)}
assets_list = list(data.keys())

close_mat = np.full((n_dates, n_assets), np.nan)
pct_mat = np.full((n_dates, n_assets), np.nan)

for j, sym in enumerate(assets_list):
    df = data[sym]
    for _, row in df.iterrows():
        d = str(row['date'])[:10]
        if d in date_to_idx:
            i = date_to_idx[d]
            close_mat[i,j] = row['close']
            pct = row['pct_change']
            pct_mat[i,j] = pct if not np.isnan(pct) else 0.0

print(f"Matrix shape: {close_mat.shape}, NaN in close: {np.isnan(close_mat).mean():.4f}")

# Forward returns
fwd_ret = np.full((n_dates, n_assets), np.nan)
for j in range(n_assets):
    fwd_ret[:,j] = np.array([close_mat[i+10,j]/close_mat[i,j]-1 if i+10<n_dates else np.nan for i in range(n_dates)])

# Helper for cross-sectional IC
def cs_ic(factor, fwd, min_valid=MIN_VALID):
    ics, n_dt = [], 0
    for t in range(n_dates):
        fv = factor[t]; rv = fwd[t]
        valid = ~(np.isnan(fv) | np.isnan(rv))
        if np.sum(valid) >= min_valid and np.std(fv[valid]) > 1e-10 and np.std(rv[valid]) > 1e-10:
            ic, _ = spearmanr(fv[valid], rv[valid])
            if not np.isnan(ic):
                ics.append(ic); n_dt += 1
    return np.array(ics), n_dt

def report_factor(name, factor_vals):
    ics, n_dt = cs_ic(factor_vals, fwd_ret)
    if len(ics) < 5:
        print(f"\n  {name}: too few obs ({len(ics)})")
        return None
    abs_ic = np.abs(ics)
    icir_denom = np.std(ics) if np.std(ics) > 0 else 1e-10
    mean_ic = np.mean(ics)
    mean_abs_ic = np.mean(abs_ic)
    icir = mean_ic / icir_denom  # raw ICIR
    icir_abs = mean_abs_ic / np.std(abs_ic) if np.std(abs_ic) > 0 else 0
    hit = np.mean(ics > 0)
    pass_pct = np.mean(abs_ic >= 0.007)
    cov = np.mean(~np.isnan(factor_vals))
    
    print(f"\n  --- {name} ---")
    print(f"  Dates with valid IC: {n_dt}")
    print(f"  Mean IC: {mean_ic:.6f}")
    print(f"  Mean |IC|: {mean_abs_ic:.6f}")
    print(f"  IC SD: {np.std(ics):.6f}")
    print(f"  ICIR: {icir:.4f}")
    print(f"  ICIR (abs): {icir_abs:.4f}")
    print(f"  IC > 0: {hit:.3f}")
    print(f"  |IC| >= 0.007: {pass_pct:.3f}")
    print(f"  Coverage: {cov:.4f}")
    
    # Compute ICIR scaled by sqrt(dates) for proper comparison
    icir_scaled = mean_ic / np.std(ics) * np.sqrt(n_dt) if np.std(ics) > 0 else 0
    abs_icir_scaled = mean_abs_ic / np.std(abs_ic) * np.sqrt(n_dt) if np.std(abs_ics) > 0 else 0
    
    return {
        'name': name,
        'dates': n_dt,
        'mean_ic': mean_ic,
        'mean_abs_ic': mean_abs_ic,
        'ic_std': np.std(ics),
        'icir': icir_scaled,
        'abs_icir': abs_icir_scaled,
        'hit_rate': hit,
        'pass_rate_007': pass_pct,
        'coverage': cov
    }

# =========================================
# FACTOR A: Vol-adjusted Momentum (Sharpe style)
# =========================================
print("\n" + "="*70)
print("FACTOR A: Vol-adjusted Momentum (20d ret / 20d vol * sqrt(252))")
factor_a = np.full((n_dates, n_assets), np.nan)
for j in range(n_assets):
    vals = np.full(n_dates, np.nan)
    for i in range(40, n_dates):
        ret = close_mat[i,j] / close_mat[i-20,j] - 1
        v = np.std(pct_mat[i-20:i,j]) * np.sqrt(252)
        vals[i] = ret / v if v > 1e-10 else 0.0
    factor_a[:,j] = vals
res_a = report_factor("Vol-adj Mom (A)", factor_a)

# =========================================
# FACTOR B: Downside Vol Ratio
# =========================================
print("\n" + "="*70)
print("FACTOR B: Downside Vol Ratio (downside_vol_20 / total_vol_20)")
factor_b = np.full((n_dates, n_assets), np.nan)
for j in range(n_assets):
    vals = np.full(n_dates, np.nan)
    for i in range(20, n_dates):
        rets = pct_mat[i-20:i,j]
        tv = np.std(rets)
        down = rets[rets < 0]
        dv = np.std(down) if len(down) > 2 else tv
        vals[i] = dv / tv if tv > 1e-10 else 0.0
    factor_b[:,j] = vals
res_b = report_factor("Downside Vol Ratio (B)", factor_b)

# =========================================
# FACTOR C: Cross-sectional rank momentum (centered)
# =========================================
print("\n" + "="*70)
print("FACTOR C: