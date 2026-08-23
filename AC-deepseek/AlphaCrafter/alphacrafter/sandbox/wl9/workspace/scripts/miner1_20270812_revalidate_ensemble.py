"""
Comprehensive revalidation of ensemble factors and all library factors.
Current date: 2027-08-12
Most factors last validated: 2026-07 to 2026-08 (12-13 months ago)
"""
import sys, json, os, math, glob
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict

# Get account to know the watchlist
acct = get_account_dict()
watch_list = acct.get('watch_list', [])
print(f"Watchlist ({len(watch_list)} instruments): {watch_list}")

# Fetch macro index data
index_ids = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']
index_data = {}
for idx_id in index_ids:
    df = get_index_daily_data(symbol=idx_id, days=2000)
    if df is not None and len(df) > 60:
        index_data[idx_id] = df
        print(f"  {idx_id}: {len(df)} days, date range {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}")
    else:
        print(f"  {idx_id}: no data or insufficient")

# Fetch instrument data
inst_data = {}
for sym in watch_list:
    df = get_stock_daily_data(symbol=sym, days=2000)
    if df is not None and len(df) > 60:
        inst_data[sym] = df
    else:
        print(f"  {sym}: insufficient data ({len(df) if df is not None else 0})")

print(f"\nInstruments with data: {list(inst_data.keys())}")

# Current date from data
current_date = inst_data[watch_list[0]]['date'].iloc[-1]
print(f"Current date: {current_date.date()}")

def get_sorted_close(inst_data):
    """Get aligned close prices"""
    closes = {}
    for sym, df in inst_data.items():
        closes[sym] = df.set_index('date')['close']
    return pd.DataFrame(closes)

close_df = get_sorted_close(inst_data)
print(f"Close df shape: {close_df.shape}, date range: {close_df.index.min().date()} to {close_df.index.max().date()}")

returns_1d = close_df.pct_change()
returns_5d = close_df.pct_change(5)
returns_10d = close_df.pct_change(10)
returns_20d = close_df.pct_change(20)

# Helper: compute rolling IC between factor values and forward returns
def compute_ic_series(factor_values, forward_returns, min_valid=8):
    """Compute cross-sectional IC for each date"""
    dates = factor_values.index
    ic_list = []
    n_obs_list = []
    
    for i, date in enumerate(dates):
        if date not in forward_returns.index:
            continue
        fv = factor_values.loc[date].dropna()
        fr = forward_returns.loc[date]
        valid = fv.index.intersection(fr.dropna().index)
        if len(valid) < min_valid:
            continue
        fv_v = fv[valid].values
        fr_v = fr[valid].values
        if np.std(fv_v) > 0 and np.std(fr_v) > 0:
            r, _ = pearsonr(fv_v, fr_v)
            ic_list.append(r)
            n_obs_list.append(len(valid))
    
    return np.array(ic_list), np.array(n_obs_list)

def factor_report(name, factor_df, ic_array, n_obs, forward_label):
    """Print IC stats"""
    if len(ic_array) < 5:
        print(f"  {name:30s} SKIP (only {len(ic_array)} dates with >=8 valid)")
        return None
    
    mean_ic = np.mean(ic_array)
    std_ic = np.std(ic_array)
    icir = mean_ic / std_ic if std_ic > 0 else 0
    hit_ratio = np.mean(ic_array > 0)
    
    print(f"  {name:30s} dates={len(ic_array):4d} mean_IC={mean_ic:+.6f} ICIR={icir:+.6f} hit={hit_ratio:.3f} avg_n={np.mean(n_obs):.0f} [{forward_label}]")
    
    return {'mean_ic': mean_ic, 'icir': icir, 'hit_ratio': hit_ratio, 'dates': len(ic_array)}


# =====================================================================
# Compute all factor values
# =====================================================================
print("\n" + "="*80)
print("COMPUTING FACTOR VALUES")
print("="*80)

factor_results = {}

# ---- 1. beta_VIX_60 ----
if 'VIX' in index_data:
    vix = index_data['VIX'].set_index('date')['close']
    vix_ret = vix.pct_change()
    aligned_idx = close_df.index.intersection(vix.index)
    beta_vix = pd.DataFrame(index=aligned_idx, columns=close_df.columns, dtype=float)
    for sym in close_df.columns:
        for i in range(60, len(aligned_idx)):
            date = aligned_idx[i]
            inst_rets = close_df[sym].pct_change().loc[aligned_idx[i-60]:date]
            vix_rets = vix_ret.loc[aligned_idx[i-60]:date]
            valid = inst_rets.dropna().index.intersection(vix_rets.dropna().index)
            if len(valid) > 30:
                beta_vix.loc[date, sym] = np.cov(inst_rets.loc[valid], vix_rets.loc[valid])[0,1] / np.var(vix_rets.loc[valid])
    factor_results['beta_VIX_60'] = beta_vix

# ---- 2. kaufman_eff_20d ----
kaufman = pd.DataFrame(index=close_df.index, columns=close_df.columns, dtype=float)
for sym in close_df.columns:
    for i in range(20, len(close_df)):
        px = close_df[sym].iloc[i-20:i+1].values
        direction = abs(px[-1] - px[0])
        volatility = np.sum(np.abs(np.diff(px)))
        kaufman.loc[close_df.index[i], sym] = direction / volatility if volatility > 0 else 0