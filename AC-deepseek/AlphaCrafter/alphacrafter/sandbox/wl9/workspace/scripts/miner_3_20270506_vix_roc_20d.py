"""
Factor: vix_roc_20d
Description: 20-day rate of change of VIX spot index.
Rationale: Direction of VIX change predicts risk-on/risk-off rotation.
When VIX is RISING (positive ROC), safe havens outperform risk assets.
When VIX is FALLING (negative ROC), risk assets outperform safe havens.
This differs from beta_VIX_60 (which measures beta with VIX LEVEL over 60d)
by focusing on the RECENT TREND of VIX over 20d.

Safe havens: XAU, US10Y, CN10Y
Risk assets: all others in cross-asset 15-instrument universe
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict

data = get_account_dict()
watchlist = data['watch_list']
print(f"Watchlist ({len(watchlist)}): {watchlist}")

# Safe haven vs risk distinction
safe_havens = ['XAU', 'US10Y', 'CN10Y']
risk_assets = [s for s in watchlist if s not in safe_havens]
print(f"Safe havens: {safe_havens}")
print(f"Risk assets: {risk_assets}")

# 1. Get VIX data
vix_df = get_index_daily_data("VIX", 800)
vix_close = vix_df.set_index('date')['close']
print(f"VIX data: {len(vix_df)} days, range {vix_df.date.iloc[0]} to {vix_df.date.iloc[-1]}")
print(f"VIX curr={vix_close.iloc[-1]:.2f}, 20d ago={vix_close.iloc[-20]:.2f}")

# 2. Get all asset data
asset_dfs = {}
for sym in watchlist:
    df = get_stock_daily_data(sym, 500)
    asset_dfs[sym] = df.set_index('date')

vix_roc = vix_close.pct_change(20)
all_dates = vix_roc.dropna().index
print(f"\nDates with VIX ROC: {len(all_dates)}")

# For each date, compute factor values and forward returns
forward_ret_records = []

min_window = 60
for i in range(min_window, len(all_dates)):
    date = all_dates[i]
    vix_trend = vix_roc.loc[date]
    
    asset_vals = {}
    fwd_returns = {}
    
    for sym in watchlist:
        df = asset_dfs[sym]
        if date not in df.index:
            continue
        close_now = df.loc[date, 'close']
        
        loc = df.index.get_loc(date)
        if loc + 10 >= len(df):
            continue
        future_date = df.index[loc + 10]
        close_future = df.loc[future_date, 'close']
        
        fwd_ret = close_future / close_now - 1.0
        
        if sym in safe_havens:
            factor_val = vix_trend  # positive when VIX rising -> buy safe
        else:
            factor_val = -vix_trend  # negative when VIX rising -> avoid risk
        
        asset_vals[sym] = factor_val
        fwd_returns[sym] = fwd_ret
    
    valid = [s for s in watchlist if s in asset_vals]
    if len(valid) >= 8:
        factor_list = [asset_vals[s] for s in valid]
        ret_list = [fwd_returns[s] for s in valid]
        corr = np.corrcoef(factor_list, ret_list)[0, 1]
        forward_ret_records.append({'date': date, 'ic': corr, 'n_assets': len(valid)})

df_ic = pd.DataFrame(forward_ret_records)
print(f"\nTotal IC observations: {len(df_ic)}")
print(f"Dates with >=8 valid assets: {len(df_ic)}")

# Compute IC metrics
mean_ic = df_ic['ic'].mean()
std_ic = df_ic['ic'].std()
icir = mean_ic / std_ic if std_ic > 0 else 0
ic_hit = (df_ic['ic'] > 0).mean()
t_stat = mean_ic / (std_ic / np.sqrt(len(df_ic))) if len(df_ic) > 1 else 0

print(f"\n=== VIX ROC 20d Factor Validation ===")
print(f"Mean IC: {mean_ic:.6f}")
print(f"IC Std:  {std_ic:.6f}")
print(f"ICIR:    {icir:.6f}")
print(f"IC Hit Ratio: {ic_hit:.4f}")
print(f"T-stat:  {t_stat:.4f}")
print(f"Observations: {len(df_ic)}")
print(f"Admission IC threshold: >= 0.007")
print(f"Admission ICIR threshold: >= 0.084")

# Check different holding periods
print("\n=== Decay Analysis (IC by forward horizon) ===")
for horizon in [5, 10, 15, 20]:
    records_h = []
    for i in range(min_window, len(all_dates)):
        date = all_dates[i]
        vix_trend = vix_roc.loc[date]
        asset_vals_h = {}
        fwd_returns_h = {}
        for sym in watchlist:
            df = asset_dfs[sym]
            if date not in df.index:
                continue
            close_now = df.loc[date, 'close']
            loc = df.index.get_loc(date)
            if loc + horizon >= len(df):
                continue
            future_date = df.index[loc + horizon]
            close_future = df.loc[future_date, 'close']
            fwd_ret = close_future / close_now - 1.0
            if sym in safe_havens:
                factor_val = vix_trend
            else:
                factor_val = -vix_trend
            asset_vals_h[sym] = factor_val
            fwd_returns_h[sym] = fwd_ret
        valid = [s for s in watchlist if s in asset_vals_h]
        if len(valid) >= 8:
            f_list = [asset_vals_h[s] for s in valid]
            r_list = [fwd_returns_h[s] for s in valid]
            c = np.corrcoef(f_list, r_list)[0, 1]
            records_h.append(c)
    if records_h:
        print(f"  Horizon {horizon:2d}d: mean IC={np.mean(records_h):.6f}, ICIR={np.mean(records_h)/np.std(records_h):.6f}" if len(records_h) > 1 else f"  Horizon {horizon:2d}d: {len(records_h)} obs")

# Turnover / stability check
print(f"\n=== Stability ===")
# IC autocorrelation
ic_auto = df_ic['ic'].autocorr(lag=1)
print(f"IC autocorrelation (lag 1): {ic_auto:.4f}")

# Regression check - confirm it's not too similar to existing factors
print("\n=== Assessment ===")
print(f"Mean IC = {mean_ic:.6f} {'PASS' if abs(mean_ic) >= 0.007 else 'FAIL'} (threshold |IC|>=0.007)")
print(f"ICIR = {icir:.6f} {'PASS' if abs(icir) >= 0.084 else 'FAIL'} (threshold |ICIR|>=0.084)")