"""
miner2_20351206_vix_accel_factor.py
Validate VIX Acceleration factors vs the 15-asset cross-section.

Factor ideas:
1. VIX_accel_v1 = -VIX_roc_5d (negated: high = VIX falling fast = risk-on)
2. VIX_accel_v2 = asset beta to VIX_accel (second derivative of VIX)
3. VIX_accel_v3 = asset beta to VIX_roc_5d (first derivative sensitivity)
"""

from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

watchlist = ['000300.SH', '000688.SH', 'SPX', 'HSI', 'N225', 'SX5E', 'SOX', 'NDX', 
             'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

data = {}
for sym in watchlist:
    df = get_stock_daily_data(sym, days=2500)
    if df is not None and len(df) >= 200:
        data[sym] = df

vix = get_index_daily_data('VIX', days=2500)

dates_all = set()
for sym, df in data.items():
    dates_all.update(df['date'].values)
dates_all = sorted(dates_all)

close_df = pd.DataFrame({'date': pd.to_datetime(dates_all)})
close_df.set_index('date', inplace=True)
for sym in watchlist:
    if sym in data:
        s = pd.Series(data[sym]['close'].values, index=pd.to_datetime(data[sym]['date'].values))
        close_df[sym] = s
close_df = close_df.dropna(how='all')

ret_df = close_df.pct_change()
vix_s = pd.Series(vix['close'].values, index=pd.to_datetime(vix['date'].values))

common_dates = close_df.index.intersection(vix_s.index)
ret_df = ret_df.loc[common_dates]
vix_s = vix_s.loc[common_dates]

print(f"Working dates: {len(common_dates)}, from {common_dates[0]} to {common_dates[-1]}")

# VIX ROC series
vix_roc_5d = vix_s.pct_change(5)
vix_roc_20d = vix_s.pct_change(20)
vix_accel = vix_roc_5d - 0.25 * vix_roc_20d

print(f"\n--- VIX Statistics (latest) ---")
print(f"VIX close: {vix_s.iloc[-1]:.2f}")
print(f"VIX 5d ROC: {vix_roc_5d.iloc[-1]*100:.2f}%")
print(f"VIX 20d ROC: {vix_roc_20d.iloc[-1]*100:.2f}%")
print(f"VIX accel: {vix_accel.iloc[-1]*100:.2f}%")

# Factor v2: beta of asset return to VIX_accel (60d)
window = 60
min_valid = 10

factor_v2 = pd.DataFrame(index=ret_df.index, columns=ret_df.columns, dtype=float)
x_accel = vix_accel.values
for asset in ret_df.columns:
    y = ret_df[asset].values
    beta = np.full(len(y), np.nan)
    for i in range(window, len(y)):
        valid = ~(np.isnan(y[i-window:i]) | np.isnan(x_accel[i-window:i]))
        if valid.sum() >= min_valid:
            xv = x_accel[i-window:i][valid]
            yv = y[i-window:i][valid]
            if np.std(xv) > 1e-10:
                beta[i] = np.cov(xv, yv)[0,1] / np.var(xv)
            else:
                beta[i] = 0.0
    factor_v2[asset] = beta

# Factor v3: beta of asset return to VIX_roc_5d (60d) 
factor_v3 = pd.DataFrame(index=ret_df.index, columns=ret_df.columns, dtype=float)
x_roc5 = vix_roc_5d.values
for asset in ret_df.columns:
    y = ret_df[asset].values
    beta = np.full(len(y), np.nan)
    for i in range(window, len(y)):
        valid = ~(np.isnan(y[i-window:i]) | np.isnan(x_roc5[i-window:i]))
        if valid.sum() >= min_valid:
            xv = x_roc5[i-window:i][valid]
            yv = y[i-window:i][valid]
            if np.std(xv) > 1e-10:
                beta[i] = np.cov(xv, yv)[0,1] / np.var(xv)
            else:
                beta[i] = 0.0
    factor_v3[asset] = beta

# Forward returns
fwd_ret_10d = ret_df.rolling(10).apply(lambda x: (1+x).prod()-1).shift(-10)
fwd_ret_5d = ret_df.rolling(5).apply(lambda x: (1+x).prod()-1).shift(-5)

def compute_ic_metrics(factor_df, fwd_ret_df, factor_name):
    """Compute cross-sectional IC for each date."""
    ics = []
    for dt in factor_df.index:
        fv = factor_df.loc[dt].values
        rv = fwd_ret_df.loc[dt].values if dt in fwd_ret_df.index else np.full(len(fv), np.nan)
        valid = ~(np.isnan(fv) | np.isnan(rv))
        if valid.sum() >= 8:
            fv_v = fv[valid]
            rv_v = rv[valid]
            # Check for variance
            if np.std(fv_v) > 1e-10 and np.std(rv_v) > 1e-10:
                ic, _ = pearsonr(fv_v, rv_v)
                ics.append(ic)
    
    ics = np.array(ics)
    if len(ics) > 0:
        ic_mean = np.mean(ics)
        ic_std = np.std(ics, ddof=1)
        icir = ic_mean / ic_std if ic_std > 0 else 0
        hit = np.mean(np.sign(ics) == np.sign(ic_mean)) if ic_mean != 0 else 0.5
        return {
            'n_dates': len(ics),
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'icir': icir,
            'hit_ratio': hit
        }
    return None

# Compute IC for 10d horizon
print(f"\n\n========== IC Analysis: 10-day horizon ==========")
# Need to align factor and forward return dates
# Factor date t -> fwd_ret[t] is 10d return starting at t
valid_mask_v2 = factor_v2.dropna(how='all', axis=0).index
valid_mask_v3 = factor_v3.dropna(how='all', axis=0).index
common_idx = valid_mask_v2.intersection(valid_mask_v3).intersection(fwd_ret_10d.index)

f2_10d = compute_ic_metrics(factor_v2.loc[common_idx], fwd_ret_10d.loc[common_idx], "VIX_accel_beta_60d")
f3_10d = compute_ic_metrics(factor_v3.loc[common_idx], fwd_ret_10d.loc[common_idx], "VIX_roc5_beta_60d")

print(f"\nFactor v2 (VIX Acceleration Beta 60d) - IC(10d):")
if f2_10d: print(f"  n_dates={f2_10d['n_dates']}, IC={f2_10d['ic_mean']:.6f}, ICIR={f2_10d['icir']:.6f}, hit={f2_10d['hit_ratio']:.4f}")
print(f"\nFactor v3 (VIX ROC5 Beta 60d) - IC(10d):")
if f3_10d: print(f"  n_dates={f3_10d['n_dates']}, IC={f3_10d['ic_mean']:.6f}, ICIR={f3_10d['icir']:.6f}, hit={f3_10d['hit_ratio']:.4f}")

#