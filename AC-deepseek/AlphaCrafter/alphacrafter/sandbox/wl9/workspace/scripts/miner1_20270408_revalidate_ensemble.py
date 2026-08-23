#!/usr/bin/env python
"""Re-validate all 17 effective factor candidates as of 2027-04-08 across 15-instrument cross-asset universe.
Gates: |IC|>=0.0070, |ICIR|>=0.0840"""
import numpy as np, pandas as pd, json, warnings, os, sys
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
warnings.filterwarnings('ignore')

DT = "2027-04-08"
WL = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
ML = ["DXY","USDCNY","USDJPY","EURUSD","VIX"]
N = len(WL)
HORIZON = 10  # forward return horizon for IC

print(f"=== REVALIDATION: {DT} ===")
print(f"Watchlist: {N} tradable assets + {len(ML)} macro signals")

# Load data
data = {}
for s in WL + ML:
    df = get_stock_daily_data(s, 750)
    if df is None or len(df) < 200:
        df = get_index_daily_data(s, 750)
    if df is not None and len(df) >= 200:
        data[s] = df

print(f"Loaded {len(data)} symbols")

# Align dates
min_dt = max(d.index[0] for d in data.values() if len(d) > 0)
aligned = {s: d.loc[d.index >= min_dt] for s, d in data.items()}
common = aligned[WL[0]].index
for s in WL[1:]:
    if s in aligned:
        common = common.intersection(aligned[s].index)

print(f"Common dates: {len(common)}")

# Build panels
C = np.column_stack([aligned[s]['close'].reindex(common).values for s in WL])
R = np.column_stack([aligned[s]['pct_change'].reindex(common).fillna(0).values for s in WL])
Lo = np.column_stack([aligned[s]['low'].reindex(common).values for s in WL])
Hi = np.column_stack([aligned[s]['high'].reindex(common).values for s in WL])
V = np.column_stack([aligned[s]['volume'].reindex(common).values if 'volume' in aligned[s].columns else np.zeros(len(common)) for s in WL])

# Macro signals
dxy = aligned['DXY']['close'].values if 'DXY' in aligned else None
vix = aligned['VIX']['close'].values if 'VIX' in aligned else None
cny = aligned['USDCNY']['close'].values if 'USDCNY' in aligned else None
jpy = aligned['USDJPY']['close'].values if 'USDJPY' in aligned else None
eur = aligned['EURUSD']['close'].values if 'EURUSD' in aligned else None

T, N = R.shape
print(f"Panel: {T}d x {N}a")

# Forward returns at HORIZON
fwd = np.full_like(R, np.nan)
for i in range(T - HORIZON):
    fwd[i] = R[i + HORIZON]

def rank_ic(factor, fwd_r, min_valid=8):
    """Compute cross-sectional rank IC between factor and forward returns."""
    dates_valid = 0
    ics = []
    for t in range(len(factor)):
        f = factor[t]
        r = fwd_r[t]
        valid = ~(np.isnan(f) | np.isnan(r))
        n_valid = np.sum(valid)
        if n_valid >= min_valid:
            f_valid = f[valid]
            r_valid = r[valid]
            from scipy.stats import spearmanr
            rho, _ = spearmanr(f_valid, r_valid)
            if not np.isnan(rho):
                ics.append(rho)
                dates_valid += 1
    if len(ics) < 10:
        return 0.0, 0.0, 0, 0.0
    ic_arr = np.array(ics)
    ic_mean = np.mean(ic_arr)
    ic_std = np.std(ic_arr, ddof=1)
    icir = ic_mean / ic_std if ic_std > 1e-10 else 0.0
    return ic_mean, icir, dates_valid, np.mean(np.abs(ic_arr))

from scipy.stats import spearmanr

def evaluate_factor(factor_values, name, direction=1):
    ic, icir, n_dates, abs_ic = rank_ic(factor_values, fwd)
    print(f"  {name:30s}: IC={ic:+.6f}  ICIR={icir:+.6f}  n_dates={n_dates}  hit={(np.sign(ic)*direction>0).mean() if len(ic)>0 else 0:.3f}")
    passes = (abs(ic) >= 0.007) and (abs(icir) >= 0.084)
    status = "PASS" if passes else "FAIL"
    print(f"    -> {status} (gate: |IC|>=0.007, |ICIR|>=0.084)")
    return ic, icir, n_dates, passes

results = {}

# === COMPUTE FACTORS ===
print("\n--- Computing factors ---")

# 1. mom_120d_skip5
mom120 = np.full_like(C, np.nan)
for i in range(125, T):
    mom120[i] = C[i-5] / np.maximum(C[i-125], 1e-10) - 1
print("  mom_120d_skip5 computed")

# 2. mom_10d_skip5
mom10 = np.full_like(C, np.nan)
for i in range(15, T):
    mom10[i] = C[i-5] / np.maximum(C[i-15], 1e-10) - 1
print("  mom_10d_skip5 computed")

# 3. kaufman_eff_20d
kauf = np.full_like(C, np.nan)
for i in range(20, T):
    d = np.abs(C[i] - C[i-20])
    vt = np.sum(np.abs(np.diff(C[i-20:i+1], axis=0)), axis=0)
    kauf[i] = d / np.maximum(vt, 1e-10)
print("  kaufman_eff_20d computed")

# 4. vol_z_20d
mu20 = pd.DataFrame(R).rolling(20, min_periods=20).mean().values
sd20 = pd.DataF