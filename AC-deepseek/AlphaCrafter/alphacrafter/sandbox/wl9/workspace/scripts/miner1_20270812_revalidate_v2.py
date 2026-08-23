#!/usr/bin/env python3
"""Efficient revalidation of ALL factor library factors.
Current date: 2027-08-12
Last validated: 2026-07 to 2026-08 (12-13 months ago)
"""
import json, os, sys, math, glob, time
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict

np.seterr(all='ignore')

acct = get_account_dict()
watch_list = acct.get('watch_list', [])
print(f"Watchlist ({len(watch_list)}): {watch_list}")

# Load index data
idx_ids = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']
idx_data = {}
for iid in idx_ids:
    df = get_index_daily_data(symbol=iid, days=2000)
    if df is not None and len(df) > 60:
        idx_data[iid] = df

# Load instrument data
inst = {}
for sym in watch_list:
    df = get_stock_daily_data(symbol=sym, days=2000)
    if df is not None and len(df) > 60:
        inst[sym] = df
print(f"Instruments: {list(inst.keys())}")

# Build aligned close DataFrame
closes = {}
for sym, df in inst.items():
    closes[sym] = df.set_index('date')['close']
close_df = pd.DataFrame(closes).sort_index()
close_df = close_df.astype(float)
print(f"Close: {close_df.shape}, dates {close_df.index[0].date()} to {close_df.index[-1].date()}")
returns = close_df.pct_change()

# Macro data aligned
vix_df = idx_data['VIX'].set_index('date')['close'] if 'VIX' in idx_data else None
dxy_df = idx_data['DXY'].set_index('date')['close'] if 'DXY' in idx_data else None
cny_df = idx_data['USDCNY'].set_index('date')['close'] if 'USDCNY' in idx_data else None

# Forward returns (10-day)
fwd_10 = close_df.shift(-10) / close_df - 1

def compute_ic_series(factor_df, forward_df, min_valid=8):
    """Cross-sectional IC per date."""
    common_idx = factor_df.index.intersection(forward_df.index)
    ic_vals, n_vals = [], []
    for date in common_idx:
        fv = factor_df.loc[date].dropna()
        fr = forward_df.loc[date].dropna()
        valid = fv.index.intersection(fr.index)
        if len(valid) < min_valid:
            continue
        fvv = fv[valid].values
        frv = fr[valid].values
        if np.std(fvv) > 1e-12 and np.std(frv) > 1e-12:
            r, _ = pearsonr(fvv, frv)
            ic_vals.append(r)
            n_vals.append(len(valid))
    return np.array(ic_vals), np.array(n_vals)

def report(name, factor_df, forward_df, label="fwd10d"):
    ic_arr, n_arr = compute_ic_series(factor_df, forward_df)
    if len(ic_arr) < 4:
        print(f"  {name:30s} SKIP (only {len(ic_arr)} IC dates)")
        return None
    mean_ic = float(np.mean(ic_arr))
    icir = float(mean_ic / np.std(ic_arr)) if np.std(ic_arr) > 1e-12 else 0
    hit = float(np.mean(ic_arr > 0))
    print(f"  {name:30s} dates={len(ic_arr):4d} IC={mean_ic:+.6f} ICIR={icir:+.6f} hit={hit:.3f} avg_n={np.mean(n_arr):.0f} [{label}]")
    return {'mean_ic': mean_ic, 'icir': icir, 'hit_ratio': hit, 'n_dates': len(ic_arr)}

print("\n" + "="*70)
print("REVALIDATING ALL FACTORS (10d forward returns)")
print("="*70)

results = {}

# ========== 1. beta_VIX_60 ==========
if vix_df is not None:
    t0 = time.time()
    vix_ret = vix_df.pct_change()
    common_idx = close_df.index.intersection(vix_df.index)
    beta_vix = pd.DataFrame(index=common_idx, columns=close_df.columns, dtype=float)
    for sym in close_df.columns:
        for i in range(60, len(common_idx)):
            date = common_idx[i]
            ir = close_df[sym].pct_change().loc[common_idx[i-60]:date].dropna()
            vr = vix_ret.loc[common_idx[i-60]:date].dropna()
            valid = ir.index.intersection(vr.index)
            if len(valid) > 30:
                beta_vix.loc[date, sym] = np.cov(ir.loc[valid], vr.loc[valid])[0,1] / np.var(vr.loc[valid])
    results['beta_VIX_60'] = report('beta_VIX_60', beta_vix, fwd_10)
    print(f"  (elapsed: {time.time()-t0:.1f}s)")

# ========== 2. kaufman_eff_20d ==========
t0 = time.time()
kaufman = pd.DataFrame(index=close_df.index, columns=close_df.columns, dtype=float)
for sym in close_df.columns:
    for i in range(20, len(close_df)):
        px = close_df[sym].iloc[i-20:i+1].values
        direction = abs(px[-1] - px[0])
        volatility = np.sum(np.abs(np.diff(px)))
        kaufman.loc[close_df.index[i], sym] = direction / volatility if volatility > 1e-12 else 0
results['kaufman_eff_20d'] = report('kaufman_eff_20d', kaufman, fwd_10)
print(f"  (elapsed: {time.time()-t0:.1f}s)")

# ========== 3. mom_120d_skip5 ==========
t0 = time.time()
mom_120 = close_df.pct_change(120)
results['mom_120d_skip5'] = report('mom_120d_skip5', mom_120.shift(5), fwd_10)
print(f"  (elapsed: {time.time()-t0:.1f}s)")

# ========== 4. bb_width_20d ==========
t0 = time.time()
sma20 = close_df.rolling(20).mean()
std20 = close_df.rolling(20).std()
bb_width = (2 * std20) / sma20
results['bb_width_20d'] = report('bb_width_20d', bb_width.shift(1), fwd_10)
print(f"  (elapsed: {time.time()-t0:.1f}s)")

# ========== 5. cny_beta_60 ==========
if cny_df is not None:
    t0 = time.time()
    cny_ret = cny_df.pct_change()
    common_idx = close_df.index.intersection(cny_df.index)
    beta_cny = pd.DataFrame(index=common_idx, columns=close_df.columns, dtype=float)
    for sym in close_df.columns:
        for i in range(60, len(common_idx)):
            date = common_idx[i]
            ir = close_df[sym].pct_change().loc[common_idx[i-60]:date].dropna()
            cr = cny_ret.loc[common_idx[i-60]:date].dropna()
            valid = ir.index.intersection(cr.index)
            if len(valid) > 30:
                beta_cny.loc[date, sym] = np.cov(ir.loc[valid], cr.loc[valid])[0,1] / np.var(cr.loc[valid])
    results['cny_beta_60'] = report('cny_beta_60', beta_cny, fwd_10)
    print(f"  (elapsed: {time.time()-t0:.1f}s)")

# ========== 6. vol_z_20d ==========
t0 = time.time()
vol_20 = returns.rolling(20).std()
vol_mean = vol_20.rolling(120