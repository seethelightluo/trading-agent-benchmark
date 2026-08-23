"""
miner2_20320708_novel_factors_v3.py
Explore novel factor candidates for 15-instrument cross-asset universe.
Data through 2032-07-07.

Factor candidates:
A) Risk-adjusted reversal: -(1d ret)/20d_vol
B) Vol term structure: vol_10d / vol_60d - 1
C) Cross-sectional dispersion of momentum: std of 20d returns
D) Price acceleration: (ret_10d - ret_30d) / vol_20d
E) Drawup ratio: (close - 60d_min) / (60d_max - 60d_min)
F) Cross-asset relative strength: rank(20d ret) minus cross-sectional median rank
G) Volatility beta to VIX (conditional)
H) Skewness of 30d daily returns
"""
import numpy as np
from scipy.stats import spearmanr, skew as sp_skew
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCHLIST = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
             'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
MACRO = ['DXY', 'USDCNY', 'VIX']
HORIZON = 10
MIN_VALID = 8

print("=" * 70)
print("FACTOR MINER 2 -- Novel Factor Exploration V3")
print("Data through 2032-07-07 (latest available)")
print("=" * 70)

# Load data
data = {}
for sym in WATCHLIST:
    df = get_stock_daily_data(sym, 800)
    if df is not None and len(df) >= 300:
        data[sym] = df
    else:
        print(f"WARN: {sym} insufficient: {len(df) if df is not None else 0}")

macro = {}
for sym in MACRO:
    df = get_index_daily_data(sym, 800)
    if df is not None and len(df) >= 300:
        macro[sym] = df
        print(f"  MACRO {sym}: {len(df)} days")

print(f"Valid instruments: {len(data)}/15")

# Build common dates
common = None
for sym, df in data.items():
    dts = set(str(d)[:10] for d in df['date'].values)
    common = dts if common is None else common & dts
for sym, df in macro.items():
    dts = set(str(d)[:10] for d in df['date'].values)
    common &= dts

common_dates = sorted(common)
print(f"Common dates: {len(common_dates)}, range {common_dates[0]} to {common_dates[-1]}")

d2i = {d: i for i, d in enumerate(common_dates)}
N = len(common_dates)
M = len(data)
assets = sorted(data.keys())

close = np.full((N, M), np.nan)
pct = np.full((N, M), np.nan)
for j, sym in enumerate(assets):
    df = data[sym]
    for _, row in df.iterrows():
        d = str(row['date'])[:10]
        if d in d2i:
            i = d2i[d]
            close[i, j] = row['close']
            p = row['pct_change']
            pct[i, j] = p if not np.isnan(p) else 0.0

macro_mat = {}
for sym, df in macro.items():
    arr = np.full(N, np.nan)
    for _, row in df.iterrows():
        d = str(row['date'])[:10]
        if d in d2i:
            i = d2i[d]
            arr[i] = row['close']
    macro_mat[sym] = arr

# Forward returns (10d)
fwd_ret = np.full((N, M), np.nan)
for j in range(M):
    for i in range(HORIZON, N - HORIZON):
        if not np.isnan(close[i, j]) and not np.isnan(close[i+HORIZON, j]) and close[i, j] > 0:
            fwd_ret[i, j] = close[i+HORIZON, j] / close[i, j] - 1

# Also compute 1d forward return for shorter-horizon check
fwd_1d = np.full((N, M), np.nan)
for j in range(M):
    for i in range(N - 1):
        if not np.isnan(close[i, j]) and not np.isnan(close[i+1, j]) and close[i, j] > 0:
            fwd_1d[i, j] = close[i+1, j] / close[i, j] - 1

print(f"Matrix: {N} dates x {M} assets, close NaN {np.isnan(close).mean():.4f}")

# IC function
def calc_ics(factor, fwd, min_valid=MIN_VALID):
    ics = []
    for t in range(N):
        fv = factor[t]
        rv = fwd[t]
        valid = ~(np.isnan(fv) | np.isnan(rv))
        nv = int(np.sum(valid))
        if nv >= min_valid:
            fvv = fv[valid]
            rvv = rv[valid]
            if np.std(fvv) > 1e-10 and np.std(rvv) > 1e-10:
                ic, _ = spearmanr(fvv, rvv)
                if not np.isnan(ic):
                    ics.append(ic)
    return np.array(ics)

def report_factor(name, factor_vals, fwd=fwd_ret):
    ics = calc_ics(factor_vals, fwd)
    if len(ics) < 5:
        print(f"\n  {name}: too few obs ({len(ics)}), skipping")
        return None
    mean_ic = float(np.mean(ics))
    mean_abs_ic = float(np.mean(np.abs(ics)))
    ic_std = float(np.std(ics)) if np.std(ics) > 0 else 1e-10
    daily_icir = mean_ic / ic_std
    hit = float(np.mean(ics > 0))
    pass_007 = float(np.mean(np.abs(ics) >= 0.007))
    coverage = float(np.mean(~np.isnan(factor_vals)))
    
    # Check gate: abs daily IC >= 0.007 AND abs daily ICIR >= 0.084
    passes = (mean_abs_ic >= 0.007) and (abs(daily_icir) >= 0.084)
    
    print(f"\n  --- {name} ---")
    print(f"  Obs (valid dates): {len(ics)}")
    print(f"  Mean IC: {mean_ic:.6f}")
    print(f"  Mean |IC|: {mean_abs_ic:.6f}")
    print(f"  IC SD: {ic_std:.6f}")
    print(f"  Daily ICIR: {daily_icir:.4f}")
    print(f"  IC > 0: {hit:.3f}")
    print(f"  |IC| >= 0.007: {pass_007:.3f}")
    print(f"  Coverage: {coverage:.4f}")
    print(f"  PASSES GATE (abs_ic>=0.007 & abs_icir>=0.084): {passes}")
    if len(ics) >= 20:
        print(f"  [10d] ICIR scaled: {daily_icir * np.sqrt(len(ics)):.4f}")
    return {'name': name, 'n': len(ics), 'ic': mean_ic, 'abs_ic': mean_abs_ic,
            'icir': daily_icir, 'hit': hit, 'pass_007': pass_007, 'cov': coverage,
            'pass': passes}

print("\n" + "=" * 70)
results = []

# ====================
# FACTOR A: Risk-adjusted reversal
# ====================
print("FACTOR A: Risk-Adjusted Reversal (-1d_ret / 20d_vol)")
factor_a = np.full((N, M), np.nan)
for j in range(M):
    vals = np.full(N, np.nan)
    for i in range(21, N):
        ret_1d = pct[i, j]
        vol_20d = np.std(pct[i-20:i, j])
        if not np.isnan(ret_1d) and vol_20d > 1e-10:
            vals[i