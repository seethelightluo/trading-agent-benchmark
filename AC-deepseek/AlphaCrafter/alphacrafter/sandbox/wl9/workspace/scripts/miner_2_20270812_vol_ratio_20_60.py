"""
Factor: vol_ratio_20_60
Description: Ratio of 20-day rolling volatility to 60-day rolling volatility.
When short-term (20d) vol is high relative to medium-term (60d) vol (ratio > 1),
it signals recent turbulence/regime change, which may predict mean reversion
or defensive rotation. When ratio < 1, it indicates calm trending conditions.

This is distinct from vol_of_vol20x60 (volatility-of-volatility) which measures
changes in vol, and from kurtosis/skew which measure return distribution shape.
"""
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict

# Get watchlist
account = get_account_dict()
assets = account["watch_list"]
print(f"Watchlist ({len(assets)} assets): {assets}")

# Fetch data for all assets - lookback needs at least 60+ days for vol calc, plus 10 for forward return
LOOKBACK = 150  # generous to get enough data
MIN_LOOKBACK = 75

dfs = {}
for a in assets:
    try:
        df = get_stock_daily_data(a, days=LOOKBACK)
        if df is not None and len(df) >= MIN_LOOKBACK:
            dfs[a] = df
        else:
            print(f"  {a}: insufficient data ({len(df) if df is not None else 0} rows)")
    except Exception as e:
        print(f"  {a}: error - {e}")

print(f"\nAssets with sufficient data: {len(dfs)}")

# Also fetch VIX for regime context
vix_df = get_index_daily_data("VIX", days=LOOKBACK)
if vix_df is not None:
    vix_close = vix_df["close"].astype(float)
    print(f"VIX data available: {len(vix_df)} rows")

# Build aligned panel of daily returns
asset_returns = {}
for a, df in dfs.items():
    ret = df["close"].astype(float).pct_change()
    ret.name = a
    asset_returns[a] = ret

ret_panel = pd.concat(asset_returns.values(), axis=1, join="inner").dropna()
print(f"\nReturn panel shape: {ret_panel.shape} (dates x assets)")
print(f"Date range: {ret_panel.index[0].date()} to {ret_panel.index[-1].date()}")

# Compute the factor: vol_ratio_20_60 = 20d_vol / 60d_vol
# Using exponentially-weighted or simple rolling standard deviation
vol_20 = ret_panel.rolling(20).std()
vol_60 = ret_panel.rolling(60).std()

# Avoid division by zero
vol_ratio = vol_20 / vol_60.replace(0, np.nan)

# Forward returns for IC: 5-day, 10-day, 20-day horizons
fwd_returns = {}
for horizon, label in [(5, "5d"), (10, "10d"), (20, "20d")]:
    fwd_ret = ret_panel.shift(-horizon).rolling(horizon).mean() * np.sqrt(252)  # annualized avg return
    fwd_returns[label] = fwd_ret

# Now compute ICs
results = {}
for horizon_label, fwd in fwd_returns.items():
    # Align factor and forward returns
    aligned_factor = vol_ratio.dropna()
    aligned_fwd = fwd.reindex(aligned_factor.index).dropna()
    common_idx = aligned_factor.index.intersection(aligned_fwd.index)
    
    ic_values = []
    n_obs_dates = 0
    n_instruments_list = []
    
    for date in common_idx:
        f_vals = aligned_factor.loc[date]
        r_vals = aligned_fwd.loc[date]
        
        # Filter to valid pairs
        valid = ~(f_vals.isna() | r_vals.isna())
        f_vals = f_vals[valid]
        r_vals = r_vals[valid]
        
        if len(f_vals) >= 8:  # minimum 8 of 15 for valid cross-section
            corr = f_vals.corr(r_vals)
            if not np.isnan(corr):
                ic_values.append(corr)
                n_obs_dates += 1
                n_instruments_list.append(len(f_vals))
    
    if len(ic_values) > 0:
        ic_series = pd.Series(ic_values)
        mean_ic = float(ic_series.mean())
        std_ic = float(ic_series.std())
        icir = mean_ic / std_ic if std_ic > 0 else 0
        hit_ratio = float((ic_series > 0).mean()) if mean_ic > 0 else flo