"""
Compute correlation of vix_roc_20d with existing factors in the library.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import json, os
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict

data = get_account_dict()
watchlist = data['watch_list']

safe_havens = ['XAU', 'US10Y', 'CN10Y']

vix_df = get_index_daily_data("VIX", 800)
vix_close = vix_df.set_index('date')['close']
vix_roc = vix_close.pct_change(20)

asset_dfs = {}
for sym in watchlist:
    df = get_stock_daily_data(sym, 500)
    asset_dfs[sym] = df.set_index('date')

dates = vix_roc.dropna().index

# Compare new factor vs beta_VIX_60 signals
vix_full = get_index_daily_data("VIX", 800).set_index('date')['close']

common_dates = []
new_vecs = []
beta_vecs = []

for idx in range(60, len(dates)):
    date = dates[idx]
    if date not in vix_roc.index:
        continue
    vix_trend = vix_roc.loc[date]
    new_sig = []
    beta_sig = []
    valid = True
    for sym in watchlist:
        new_val = vix_trend if sym in safe_havens else -vix_trend
        new_sig.append(new_val)
        
        df = asset_dfs.get(sym)
        if df is None or date not in df.index:
            valid = False
            break
        idx_loc = df.index.get_loc(date)
        if idx_loc < 60:
            valid = False
            break
        asset_ret = df['close'].iloc[idx_loc-60:idx_loc+1].pct_change().iloc[1:]
        vix_s = vix_full.loc[df.index[idx_loc-60]:date].pct_change().iloc[1:]
        if len(asset_ret) < 30 or len(vix_s) < 30:
            valid = False
            break
        ar = np.array(asset_ret)
        vr = np.array(vix_s)
        if len(ar) != len(vr):
            ml = min(len(ar), len(vr))
            ar = ar[-ml:]
            vr = vr[-ml:]
        if np.std(vr) > 0 and np.std(ar) > 0:
            beta_val = np.cov(ar, vr)[0,1] / np.var(vr)
            beta_sig.append(beta_val)
        else:
            valid = False
            break
    
    if valid and len(new_sig) == 15 and len(beta_sig) == 15:
        common_dates.append(date)
        new_vecs.append(new_sig)
        beta_vecs.append(beta_sig)

rho_over_time = []
for i in range(len(common_dates)):
    rho = np.corrcoef(new_vecs[i], beta_vecs[i])[0, 1]
    rho_over_time.append(rho)

rho_array = np.array(rho_over_time)
max_abs_corr = np.max(np.abs(rho_array))
print(f"Dates compared: {len(common_dates)}")
print(f"Mean abs rho vs beta_VIX_60: {np.mean(np.abs(rho_array)):.4f}")
print(f"Max abs rho vs beta_VIX_60: {max_abs_corr:.4f}")
print(f"max_abs_library_correlation = {max_abs_corr:.4f}")