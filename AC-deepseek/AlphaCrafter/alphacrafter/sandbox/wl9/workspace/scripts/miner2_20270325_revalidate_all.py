#!/usr/bin/env python
"""
Re-validation of all 17 existing factors as of 2027-03-25.
Many factors last validated 2026-07/08; refresh overdue.
Uses 15-instrument cross-asset watchlist.

Benchmark admission gates: |daily_IC| >= 0.0070, |daily_ICIR| >= 0.0840
"""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import json, os, sys

CURRENT_DATE = "2027-03-25"

watch_list = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX",
              "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
macro_list = ["DXY","USDCNY","USDJPY","EURUSD","VIX"]

print(f"=== REVALIDATION: {CURRENT_DATE} ===")

# Fetch data - up to 750 days for coverage
all_syms = watch_list + macro_list
data = {}
for sym in all_syms:
    df = get_stock_daily_data(sym, 750)
    if df is None or len(df) < 200:
        df = get_index_daily_data(sym, 750)
    if df is not None and len(df) >= 200:
        data[sym] = df

print(f"Loaded {len(data)} symbols")

# Align dates
min_dates = max(d.index[0] for d in data.values())
aligned = {s: d.loc[d.index >= min_dates] for s, d in data.items()}
common = aligned[watch_list[0]].index
for s in watch_list[1:]:
    common = common.intersection(aligned[s].index)

print(f"Common dates: {len(common)} from {common[0]} to {common[-1]}")

# Build matrices
close_mat = np.column_stack([aligned[s]['close'].reindex(common).values for s in watch_list])
ret_mat = np.column_stack([aligned[s]['pct_change'].reindex(common).fillna(0).values for s in watch_list])
low_mat = np.column_stack([aligned[s]['low'].reindex(common).values for s in watch_list])
high_mat = np.column_stack([aligned[s]['high'].reindex(common).values for s in watch_list])

# Macro signals
dxy = aligned['DXY']['close'].values if 'DXY' in aligned else None
vix = aligned['VIX']['close'].values if 'VIX' in aligned else None
usdcny = aligned['USDCNY']['close'].values if 'USDCNY' in aligned else None

T, N = ret_mat.shape
print(f"Panel shape: {T} days x {N} assets")

# Forward returns h=1
def forward_returns(ret, h=1):
    fwd = np.full_like(ret, np.nan)
    fwd[:-h] = np.array([np.sum(ret[i+1:i+1+h], axis=0) for i in range(len(ret)-h)])
    return fwd

fwd1 = forward_returns(ret_mat, 1)

# Rolling helpers
def roll_mean(x, w):
    return pd.DataFrame(x).rolling(w, min_periods=w).mean().values

def roll_std(x, w):
    return pd.DataFrame(x).rolling(w, min_periods=w).std(ddof=0).values

###############################################################################
# Compute all 17 factors
###############################################################################

# F1: mom_120d_skip5
mom120 = np.full_like(close_mat, np.nan)
for i in range(125, T):
    mom120[i] = close_mat[i-5] / np.maximum(close_mat[i-5-120], 1e-10) - 1

# F2: mom_10d_skip5
mom10 = np.full_like(close_mat, np.nan)
for i in range(15, T):
    mom10[i] = close_mat[i-5] / np.maximum(close_mat[i-5-10], 1e-10) - 1

# F3: vol_z_20d
mu20 = roll_mean(ret_mat, 20)
sd20 = roll_std(ret_mat, 20)
vol_z20 = (ret_mat - mu20) / np.maximum(sd20, 1e-10)

# F4: kaufman_eff_20d
kaufman = np.full_like(close_mat, np.nan)
for i in range(20, T):
    direction = np.abs(close_mat[i] - close_mat[i-20])
    vol_tot = np.sum(np.abs(np.diff(close_mat[i-20:i+1], axis=0)), axis=0)
    kaufman[i] = direction / np.maximum(vol_tot, 1e-10)

# F5: bb_width_20d (upper-lower)/ma = 4*std/ma
ma20 = roll_mean(close_mat, 20)
sd20c = roll_std(close_mat, 20)
bbw = (4*sd20c) / np.maximum(ma20, 1e-10)

# F6: beta_VIX_60
beta_vix = np.full_like(ret_mat, np.nan)
if vix is not None:
    vix_ret = np.diff(vix, prepend=0)
    for i in range(60, T):
        rx = ret_mat[i-60:i]
        vx = vix_ret[i-60:i]
        vx_var = np.var(vx)
        if vx_var > 1e-12:
            cov = np.mean((rx - np.mean(rx, axis=0)) * (vx - np.mean(vx))[:, None], axis=0)
            beta_vix[i] = cov / vx_var

# F7: cny_beta_60
cny_beta = np.full_like(ret_mat, np.nan)
if usdcny is not None:
    cny_ret = np.diff(usdcny, prepend=0)
    for i in range(60, T):
        rx = ret_mat[i-60:i]
        cx = cny_ret[i-60:i]
        cx_var = np.var(cx)
        if cx_var > 1e-12:
            cov = np.mean((rx - np.mean(rx, axis=0)) * (cx - np.mean(cx))[:, None], axis=0)
            cny_beta[i] = cov / cx_var

# F8: ac1_120d
ac1 = np.full_like(ret_mat, np.nan)
for i in range(121, T):
    s = ret_mat[i-120:i]
    for j in range(N):
        a = s[:-1, j]; b = s[1:, j]
        if np.std(a) > 1e-10 and np.std(b) > 1e-10:
            ac1[i,j] = np.corrcoef(a, b)[0,1]
        else:
            ac1[i,j] = 0

# F9: skew_20d
skew20 = np.full_like(ret_mat, np.nan)
for i in range(20, T):
    for j in range(N):
        s = ret_mat[i-20:i, j]
        if np.std(s) > 1e-10:
            skew20[i,j] = pd.Series(s).skew()
        else:
            skew20[i,j] = 0

# F10: rng_pos_20d
rng_pos = np.full_like(close_mat, np.nan)
for i in range(20, T):
    lo = np.min(low_mat[i-20:i], axis=0)
    hi = np.max(high_mat[i-20:i], axis=0)
    denom = np.maximum(hi - lo, 1e-10)
    rng_pos[i] = (close_mat[i] - lo) / denom

# F11: kurt_20d
kurt20 = np.full_l
