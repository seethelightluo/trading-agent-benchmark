"""
miner3_20351108_exp01_price_acceleration.py
Explore Price Acceleration factor: momentum-of-momentum.
Factor = (pct_change_10d) - (pct_change_30d)
Positive = asset accelerating (short-term > medium-term)
Negative = asset decelerating (short-term < medium-term)
Expectation: accelerating assets continue to outperform (positive IC)
"""
import sys, json, math, datetime
import numpy as np
from collections import defaultdict
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

# Get watchlist
acct = get_account_dict()
watch_list = list(acct.get('watch_list', []))
print(f"Watch list ({len(watch_list)}): {watch_list}")

# Fetch enough data
MIN_DAYS = 60
data = {}
for sym in watch_list:
    df = get_stock_daily_data(sym, 200)
    if df is None or len(df) < MIN_DAYS:
        print(f"  WARNING: {sym} insufficient data ({len(df) if df is not None else 0} days)")
        data[sym] = None
    else:
        data[sym] = df
        print(f"  {sym}: {df['date'].min().date()} to {df['date'].max().date()}, {len(df)} days")

# Compute factor values at each date
# Factor = close_10d_ago / close - close_30d_ago / close (i.e., short-term ret minus medium ret)
# Actually: pct_change over 10d minus pct_change over 30d

factor_name = "price_accel_10_30"
dates_list = []

for sym in watch_list:
    df = data[sym]
    if df is None or len(df) < 40:
        continue
    closes = df['close'].values
    dates_seq = df['date'].values
    for i in range(30, len(closes)):
        ret_10 = closes[i] / closes[i-10] - 1.0
        ret_30 = closes[i] / closes[i-30] - 1.0
        factor_val = ret_10 - ret_30  # acceleration
        dt = pd.Timestamp(dates_seq[i])
        dates_list.append((dt, sym, factor_val, closes[i]))

import pandas as pd

# Build panel
df_factor = pd.DataFrame(dates_list, columns=['date', 'asset', 'factor', 'close'])
df_factor['date'] = pd.to_datetime(df_factor['date'])
print(f"\nFactor shape: {df_factor.shape}")
print(f"Date range: {df_factor['date'].min()} to {df_factor['date'].max()}")
print(f"Factor stats:\n{df_factor['factor'].describe()}")

# Compute forward returns (10-day)
df_factor_sorted = df_factor.sort_values(['asset', 'date'])
df_factor_sorted['fwd_ret_10'] = df_factor_sorted.groupby('asset')['close'].transform(
    lambda x: x.shift(-10) / x - 1.0
)

# Remove rows without forward return
valid = df_factor_sorted.dropna(subset=['fwd_ret_10'])
print(f"\nValid rows with forward return: {len(valid)}")

# Cross-sectional IC per date
ic_values = []
n_valid_dates = 0
n_dates_ge8 = 0

for dt, grp in valid.groupby('date'):
    if len(grp) < 8:
        continue
    n_dates_ge8 += 1
    f_vals = grp['factor'].values
    r_vals = grp['fwd_ret_10'].values
    if np.std(f_vals) > 1e-10 and np.std(r_vals) > 1e-10:
        rho = np.corrcoef(f_vals, r_vals)[0, 1]
        ic_values.append(rho)
        n_valid_dates += 1

ic_arr = np.array(ic_values)
if len(ic_arr) > 0:
    mean_ic = np.mean(ic_arr)
    std_ic = np.std(ic_arr)
    icir = mean_ic / std_ic