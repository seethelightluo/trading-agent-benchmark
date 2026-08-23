"""
miner2_20320513_explore_novel_p1.py
Explore novel factor ideas:
- Factor A: Volatility-adjusted momentum (Sharpe-ratio style)
- Factor B: Distance from 60-day MA (mean reversion signal)
- Factor C: Volatility term structure (short/long vol ratio)
- Factor D: Cross-asset rank dispersion

Current date: 2032-05-13
"""

from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

WATCHLIST = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX', 
             'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
HORIZON = 10

def get_data(symbol, days=300):
    try:
        return get_stock_daily_data(symbol, days)
    except:
        return None

def compute_fwd_ret(closes, horizon=HORIZON):
    fwd = np.full(len(closes), np.nan)
    for i in range(len(closes) - horizon):
        fwd[i] = closes[i + horizon] / closes[i] - 1
    return fwd

def compute_factor_a_voladj_mom(closes, pct_chg, mom=20, vol=20):
    """Factor A: Volatility-adjusted momentum = ret/vol * sqrt(252)"""
    vals = np.full(len(closes), np.nan)
    for i in range(mom + vol, len(closes)):
        ret = closes[i] / closes[i - mom] - 1
        v = np.std(pct_chg[i - vol:i])
        if v > 1e-10:
            vals[i] = ret / v
        else:
            vals[i] = 0.0
    return vals

def compute_factor_b_ma_reversion(closes, ma=60):
    """Factor B: (close/SMA - 1), distance from moving average"""
    vals = np.full(len(closes), np.nan)
    for i in range(ma, len(closes)):
        sma = np.mean(closes[i - ma:i])
        vals[i] = closes[i] / sma - 1 if sma > 1e-10 else 0.0
    return vals

def compute_factor_c_vol_ts(pct_chg, short=10, long=60):
    """Factor C: vol_ts = short_vol / long_vol - 1"""
    vals = np.full(len(pct_chg), np.nan)
    for i in range(long, len(pct_chg)):
        sv = np.std(pct_chg[i - short:i])
        lv = np.std(pct_chg[i - long:i])
        if lv > 1e-10:
            vals[i] = sv / lv - 1
        else:
            vals[i] = 0.0
    return vals

def compute_factor_d_mom50(closes):
    """Factor D: Plain 50-day momentum for comparison"""
    vals = np.full(len(closes), np.nan)
    for i in range(50, len(closes)):
        vals[i] = closes[i] / closes[i - 50] - 1
    return vals

def compute_ic(factor_vals, fwd_ret_vals, min_valid=8):
    """Cross-sectional IC at each date"""
    n = min(len(factor_vals), len(fwd_ret_vals))
    ic_list = []
    for t in range(n):
        f = factor_vals[t]
        r = fwd_ret_vals[t]
        if np.isnan(f) or np.isnan(r):
            continue
        # need complete cross-section
        valid = ~(np.isnan(f) | np.isnan(r))
        if np.sum(valid) >= min_valid:
            ic, _ = spearmanr(f[valid], r[valid])
            if not np.isnan(ic):
                ic_list.append(ic)
    return np.array(ic_list)

print("=" * 70)
print("FACTOR MINING: Novel Factor Exploration")
print(f"Current date: 2032-05-13 | Forward horizon: {HORIZON}d")
print("=" * 70)

# Load all data
data = {}
valid = []
for sym in WATCHLIST:
    df = get_data(sym, 400)
    if df is not None and len(df) >= 200:
        data[sym] = df
        valid.append(sym)
    else:
        print(f"WARNING: {sym} has insufficient data ({len(df) if df is not None else 0} days)")

print(f"\nValid instruments: {len(valid)}/{len(WATCHLIST)}")
print(f"Trading days available: {len(data[valid[0]]) if valid else 0}")
print(f"Date range: {data[valid[0]]['date'].iloc[0]} to {data[valid[0]]['date'].iloc[-1]}")
print()

# Build aligned date-indexed factor matrices
# Use a common set of dates across all assets
common_dates = None
for sym in valid:
    dts = set(data[sym]['date'].astype(str).values)
    if common_dates is None:
        common_dates = dts
    else:
        common_dates = common_dates & dts

common_dates = sorted(common_dates)
print(f"Common dates across all assets: {len(common_dates)}")

# Create aligned arrays
n_dates = len(common_dates)
n_assets = len(valid)
date_to_idx = {d: i for i, d in enumerate(common_dates)}

# Pre-compute components
close_mat = np.full((n_dates, n_assets), np.nan)
pct_mat = np.full((n_dates, n_assets), np.nan)

for j, sym in enumerate(valid):
    df = data[sym]
    for _, row in df.iterrows():
        d = str(row['date'])[:10]
        if d in date_to_idx:
            i = date_to_idx[d]
            close_mat[i, j] = row['close']
            pct_mat[i, j] = row['pct_change'] if not np.isnan(row['pct_change']) else 0.0

# Compute forward returns
fwd_ret_mat = np.full((n_dates, n_assets), np.nan)
for j in range(n_assets):
    fwd_ret_mat[:, j] = compute_fwd_ret(close_mat[:, j], HORIZON)

print(f"\nAligned matrix shape: {close_mat.shape}")
print(f"Last date: {common_dates[-1]}")
print(f"NaN ratio in close: {np.isnan(close_mat).mean():.3f}")
print()

# ============================================================
# Factor A: Volatility-adjusted momentum (20d ret / 20d vol)
# ============================================================
print("--- FACTOR A: Vol-adjusted Momentum (20d ret / 20d vol) ---")
factor_a = np.full((n_dates, n_assets), np.nan)
for j in range(n_assets):
    factor_a[:, j] = compute_factor_a_voladj_mom(close_mat[:, j], pct_mat[:, j], 20, 20)

ics_a = compute_ic(factor_a, fwd_ret_mat, min_valid=8)
print(f"Valid IC observations: {len(ics_a)} (dates with >=8 valid assets)")
print(f"Mean IC: {np.mean(ics_a):.6f}")
print(f"Std IC:  {np.std(ics_a):.6f}")
print(f"ICIR:    {np.mean(ics_a) / np.std(ics_a) * np.sqrt(len(ics_a)):.4f}")
print(f"IC > 0:  {np.mean(ics_a > 0):.3f}")
print(f"IC > 0.007: {np.mean(ics_a > 0.007):.3f}")
abs_ic = np.abs(ics_a)
print(f"Mean |IC|: {np.mean(abs_ic):.6f}")
print(f"Pass rate (|IC|>=0.007): {np.mean(abs_ic >= 0.0070):.3f}")
# Coverage check
valid_a = ~np.isnan(factor_a)
cov_a = valid_a.mean()
print(f"Coverage: {cov_a:.4f}")
print()

# ============================================================
# Factor B: Distance from 60-day MA
# ============================================================
print("--- FACTOR B: Distance from 60-day MA (mean reversion) ---")
factor_b = np.full((n_dates, n_assets), np.nan